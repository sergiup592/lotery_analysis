"""
Ingestion of official FDJ EuroMillions archives (winners per prize rank).

FDJ publishes per-draw ZIP/CSV archives that include, for every draw, the
Europe-wide number of winners at each of the 13 prize ranks. Those counts are
the only public signal of *what other players picked*: low-tier winner counts
swing draw-to-draw depending on how "human-popular" the drawn numbers are.
`popularity_fit.py` turns that signal into empirically calibrated popularity
weights.

Drop the FDJ archives (ZIP or extracted CSV) anywhere under ``lottery_data/``
or the project root. This module finds them, parses them defensively
(delimiter/encoding/header variants differ across vintages), and caches a tidy
table to ``lottery_data/winner_counts.csv``.
"""
from __future__ import annotations

import logging
import re
import unicodedata
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import BONUS_ERA_START, DATA_DIR, PROJECT_ROOT

logger = logging.getLogger(__name__)

WINNER_COUNTS_CACHE = DATA_DIR / "winner_counts.csv"

# EuroMillions prize ranks as ordered in FDJ archives (post Sep-2016 rules).
# rank -> (main_hits, star_hits)
#
# NOTE the order of ranks 6 and 7: FDJ ranks by prize value, so 3+2 (rank 6)
# comes BEFORE 4+0 (rank 7), unlike hit-count ordering. Verified empirically
# on the archives themselves: anchoring implied ticket sales on the 3+0 tier
# reproduces every tier's winner count only under this mapping, including the
# star-suppression fingerprint of the first star-12 draw (27/09/2016), where
# tiers involving stars collapsed while 4+0 and 3+0 did not. The probabilities
# of 3+2 and 4+0 differ by just 2%, so a correlation check cannot tell them
# apart -- do not "fix" this mapping without re-running that verification.
RANK_TIERS: Dict[int, Tuple[int, int]] = {
    1: (5, 2),
    2: (5, 1),
    3: (5, 0),
    4: (4, 2),
    5: (4, 1),
    6: (3, 2),
    7: (4, 0),
    8: (2, 2),
    9: (3, 1),
    10: (3, 0),
    11: (1, 2),
    12: (2, 1),
    13: (2, 0),
}

_ENCODINGS = ("utf-8-sig", "utf-8", "cp1252", "latin-1")
_BALL_COLUMNS = ("boule_1", "boule_2", "boule_3", "boule_4", "boule_5")
_STAR_COLUMNS = ("etoile_1", "etoile_2")
_GAGNANT_RE = re.compile(r"gagnant.*?rang\s*_?(\d+)")
_DATE_FORMATS = ("%d/%m/%Y", "%Y%m%d", "%d/%m/%y", "%Y-%m-%d")


def _normalize_header(header: str) -> str:
    text = unicodedata.normalize("NFKD", str(header))
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"\s+", "_", text.strip().lower())


def _decode(payload: bytes) -> str:
    for encoding in _ENCODINGS:
        try:
            return payload.decode(encoding)
        except UnicodeDecodeError:
            continue
    return payload.decode("latin-1", errors="replace")


def _sniff_delimiter(text: str) -> str:
    sample = text[:4000]
    return ";" if sample.count(";") >= sample.count(",") else ","


def _parse_draw_date(raw_value: object) -> Optional[pd.Timestamp]:
    token = str(raw_value).strip()
    if not token or token.lower() == "nan":
        return None
    for fmt in _DATE_FORMATS:
        try:
            return pd.Timestamp(pd.to_datetime(token, format=fmt))
        except (ValueError, TypeError):
            continue
    return None


def _winner_columns(columns: Sequence[str]) -> Dict[str, Dict[int, str]]:
    """
    Map scope -> {prize rank -> column name} for winner-count columns.

    FDJ vintages carry up to three variants per rank: Europe-wide
    (``..._en_europe``), France-only (``..._en_france``), and unlabelled
    (older exports). Some vintages (e.g. Mar 2019 - Feb 2020) include the
    Europe columns but leave them zero-filled, so scope must be chosen
    per draw at parse time, never globally.
    """
    scopes: Dict[str, Dict[int, str]] = {"europe": {}, "france": {}, "generic": {}}
    for column in columns:
        match = _GAGNANT_RE.search(column)
        if not match:
            continue
        rank = int(match.group(1))
        if rank not in RANK_TIERS:
            continue
        if "etoile" in column:  # Etoile+ side game, not the main draw
            continue
        if "france" in column:
            scopes["france"].setdefault(rank, column)
        elif "europe" in column:
            scopes["europe"].setdefault(rank, column)
        else:
            scopes["generic"].setdefault(rank, column)
    return {scope: mapping for scope, mapping in scopes.items() if mapping}


# Low tiers used to decide whether a scope's counts are actually populated:
# these are won tens of thousands of times per draw, so zero means "empty".
_SCOPE_PROBE_RANKS = (10, 11, 12, 13)
# Pari-mutuel sharing happens across the whole pool, so Europe-wide counts are
# preferred; unlabelled columns are Europe-wide in older vintages; France-only
# is a valid (national) sample of the same selection behaviour, used last.
_SCOPE_PREFERENCE = ("europe", "generic", "france")


def _read_count(record: Dict[str, object], column: Optional[str]) -> Optional[int]:
    if not column:
        return None
    raw_value = record.get(column)
    try:
        count = int(float(str(raw_value).replace(" ", "").replace(",", ".")))
    except (TypeError, ValueError):
        return None
    return count if count >= 0 else None


def _extract_winner_row(
    record: Dict[str, object],
    scope_columns: Dict[str, Dict[int, str]],
) -> Optional[Tuple[str, Dict[int, int]]]:
    """Pick the first scope whose probe tiers are populated; return its counts."""
    for scope in _SCOPE_PREFERENCE:
        mapping = scope_columns.get(scope)
        if not mapping or any(rank not in mapping for rank in RANK_TIERS):
            continue
        counts = {rank: _read_count(record, mapping[rank]) for rank in RANK_TIERS}
        if any(count is None for count in counts.values()):
            continue
        if all(counts[rank] > 0 for rank in _SCOPE_PROBE_RANKS):
            return scope, counts  # type: ignore[return-value]
    return None


def _parse_csv_text(text: str, source_name: str) -> Optional[pd.DataFrame]:
    """
    Parse one FDJ CSV. Manual field splitting on purpose: several vintages
    write a trailing delimiter on data rows (one more field than the header),
    which makes ``pd.read_csv`` silently treat the first column as an index
    and shift every value one column left. Fields never contain quoted
    delimiters in these exports, so splitting is safe.
    """
    delimiter = _sniff_delimiter(text)
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        return None
    header = [_normalize_header(column) for column in lines[0].split(delimiter)]
    width = len(header)
    records = []
    for line in lines[1:]:
        fields = line.split(delimiter)
        if len(fields) < width:
            fields = fields + [""] * (width - len(fields))
        records.append(fields[:width])
    frame = pd.DataFrame(records, columns=header, dtype=str)

    date_column = next(
        (column for column in frame.columns if "date_de_tirage" in column or column == "date"),
        None,
    )
    scope_columns = _winner_columns(frame.columns)
    has_balls = all(column in frame.columns for column in _BALL_COLUMNS)
    has_stars = all(column in frame.columns for column in _STAR_COLUMNS)
    if date_column is None or not scope_columns or not has_balls or not has_stars:
        logger.info("Skipping %s: missing draw/winner columns.", source_name)
        return None

    rows: List[Dict[str, object]] = []
    dropped = 0
    for record in frame.to_dict(orient="records"):
        draw_date = _parse_draw_date(record.get(date_column))
        if draw_date is None:
            continue
        try:
            mains = sorted(int(float(record[column])) for column in _BALL_COLUMNS)
            stars = sorted(int(float(record[column])) for column in _STAR_COLUMNS)
        except (TypeError, ValueError, KeyError):
            continue
        extracted = _extract_winner_row(record, scope_columns)
        if extracted is None:
            dropped += 1
            continue
        scope, counts = extracted
        row: Dict[str, object] = {
            "date": draw_date,
            "main_numbers": mains,
            "bonus_numbers": stars,
            "scope": scope,
        }
        for rank, count in counts.items():
            row[f"winners_rank_{rank}"] = count
        rows.append(row)

    if not rows:
        return None
    if dropped:
        logger.info("Dropped %d rows without populated winner counts in %s", dropped, source_name)
    logger.info("Parsed %d draws from %s", len(rows), source_name)
    return pd.DataFrame(rows)


def _iter_archive_payloads(paths: Iterable[Path]) -> Iterable[Tuple[str, str]]:
    """Yield (source_name, csv_text) for every CSV found in zips/plain files."""
    for path in paths:
        if path.suffix.lower() == ".zip":
            try:
                with zipfile.ZipFile(path) as archive:
                    for member in archive.namelist():
                        if member.lower().endswith(".csv"):
                            yield f"{path.name}:{member}", _decode(archive.read(member))
            except zipfile.BadZipFile:
                logger.warning("Bad zip file: %s", path)
        elif path.suffix.lower() == ".csv":
            yield path.name, _decode(path.read_bytes())


def discover_archive_files() -> List[Path]:
    """FDJ archives dropped in lottery_data/ or the project root (zip or csv)."""
    seen: List[Path] = []
    for directory in (DATA_DIR, PROJECT_ROOT):
        if not directory.is_dir():
            continue
        for path in sorted(directory.iterdir()):
            if path == WINNER_COUNTS_CACHE:
                continue
            if path.suffix.lower() in {".zip", ".csv"}:
                seen.append(path)
    return seen


def load_winner_counts(
    archive_paths: Optional[Sequence[Path]] = None,
    use_cache: bool = True,
    era_start: str = BONUS_ERA_START,
) -> pd.DataFrame:
    """
    Tidy per-draw winner counts for the current star era, sorted by date.

    Columns: date, main_numbers (list[int]), bonus_numbers (list[int]),
    winners_rank_1 .. winners_rank_13.
    """
    if archive_paths is None:
        archive_paths = discover_archive_files()

    frames = [
        parsed
        for source_name, text in _iter_archive_payloads(archive_paths)
        if (parsed := _parse_csv_text(text, source_name)) is not None
    ]

    if not frames:
        if use_cache and WINNER_COUNTS_CACHE.exists():
            logger.info("No archives found; using cache %s", WINNER_COUNTS_CACHE)
            return read_winner_counts_cache()
        raise FileNotFoundError(
            "No FDJ winner-count archives found. Download the EuroMillions "
            "history ZIPs from https://www.fdj.fr/jeux-de-tirage/euromillions-my-million/historique "
            f"and place them in {DATA_DIR}."
        )

    merged = pd.concat(frames, ignore_index=True)
    merged.drop_duplicates(subset=["date"], keep="last", inplace=True)
    merged = merged[merged["date"] >= pd.Timestamp(era_start)]
    merged.sort_values("date", inplace=True)
    merged.reset_index(drop=True, inplace=True)
    if merged.empty:
        raise ValueError(f"Archives parsed, but no draws on/after {era_start}.")

    _validate_winner_counts(merged)
    if use_cache:
        write_winner_counts_cache(merged)
    return merged


def _validate_winner_counts(frame: pd.DataFrame) -> None:
    for mains, stars in zip(frame["main_numbers"], frame["bonus_numbers"]):
        if len(set(mains)) != 5 or not all(1 <= number <= 50 for number in mains):
            raise ValueError(f"Invalid main numbers in archive: {mains}")
        if len(set(stars)) != 2 or not all(1 <= number <= 12 for number in stars):
            raise ValueError(f"Invalid star numbers in archive: {stars}")
    low_tier = frame["winners_rank_13"].astype(float)
    if (low_tier <= 0).mean() > 0.02:
        raise ValueError(
            "Rank-13 winner counts are frequently zero; the rank->column mapping "
            "looks wrong for this archive vintage."
        )


def write_winner_counts_cache(frame: pd.DataFrame, path: Path = WINNER_COUNTS_CACHE) -> None:
    serializable = frame.copy()
    serializable["main_numbers"] = serializable["main_numbers"].apply(
        lambda values: " ".join(str(value) for value in values)
    )
    serializable["bonus_numbers"] = serializable["bonus_numbers"].apply(
        lambda values: " ".join(str(value) for value in values)
    )
    serializable.to_csv(path, index=False)
    logger.info("Cached %d draws to %s", len(frame), path)


def read_winner_counts_cache(path: Path = WINNER_COUNTS_CACHE) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["date"] = pd.to_datetime(frame["date"])
    for column in ("main_numbers", "bonus_numbers"):
        frame[column] = frame[column].apply(
            lambda raw: [int(token) for token in str(raw).split()]
        )
    return frame


def winners_matrix(frame: pd.DataFrame) -> np.ndarray:
    """(n_draws, 13) winner counts, column r-1 = rank r."""
    columns = [f"winners_rank_{rank}" for rank in RANK_TIERS]
    return frame[columns].to_numpy(dtype=float)
