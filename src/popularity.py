"""
Model of how popular a ticket is with human players.

Why this exists: most EuroMillions prize tiers are pari-mutuel, so a win is
shared among everyone holding the same combination. Picking *unpopular*
combinations does not change the probability of winning, but it raises the
expected payout conditional on a win. This is the only edge supported by the
lottery literature (Haigh 1997; Farrell, Hartley, Lanot & Walker 2000):
players over-pick calendar numbers (birthdays), "lucky" numbers, and visually
patterned tickets.

Two weight sources, in order of preference:

1. **Fitted weights** (``lottery_data/popularity_fitted.json``), estimated by
   ``popularity_fit.py`` from official FDJ winner-counts-per-tier archives.
   Within-draw tier ratios cancel ticket sales, leaving a direct measurement
   of how over-picked the drawn numbers were. Run
   ``python3 main.py --calibrate-popularity`` to (re)fit.
2. **Heuristic priors** (the functions below) as a documented fallback when no
   fitted file exists.

Combination-level pattern bumps (consecutive runs, all-calendar tickets, ...)
are not identifiable from low-tier winner counts and remain documented priors
in both modes.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

from .config import BONUS_NUMBER_RANGE, DATA_DIR, MAIN_NUMBER_RANGE

logger = logging.getLogger(__name__)

FITTED_WEIGHTS_FILE = DATA_DIR / "popularity_fitted.json"

_LUCKY_MAIN = {3, 7, 11, 13, 17}
_STRONG_LUCKY_MAIN = 7


def prior_main_weights() -> np.ndarray:
    """Heuristic relative pick-popularity of each main number (mean 1.0)."""
    weights = np.ones(MAIN_NUMBER_RANGE, dtype=float)
    for number in range(1, MAIN_NUMBER_RANGE + 1):
        idx = number - 1
        if number <= 31:  # birthdays (day-of-month)
            weights[idx] *= 1.30
        if number <= 12:  # doubles as a month number
            weights[idx] *= 1.12
        if number in _LUCKY_MAIN:
            weights[idx] *= 1.10
        if number == _STRONG_LUCKY_MAIN:
            weights[idx] *= 1.15
        if number >= 41:  # documented under-picking of high numbers
            weights[idx] *= 0.88
    return weights / weights.mean()


def prior_bonus_weights() -> np.ndarray:
    """Heuristic relative pick-popularity of each lucky star (mean 1.0)."""
    weights = np.ones(BONUS_NUMBER_RANGE, dtype=float)
    for number in range(1, BONUS_NUMBER_RANGE + 1):
        idx = number - 1
        if number == 7:
            weights[idx] *= 1.20
        if number <= 9:  # legacy 1-9 era habits persist on playslips
            weights[idx] *= 1.05
    return weights / weights.mean()


# Backwards-compatible aliases.
main_number_weights = prior_main_weights
bonus_number_weights = prior_bonus_weights


def _load_fitted(path: Path) -> Optional[Dict[str, np.ndarray]]:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        main = np.asarray(payload["main_weights"], dtype=float)
        bonus = np.asarray(payload["bonus_weights"], dtype=float)
        if main.shape != (MAIN_NUMBER_RANGE,) or bonus.shape != (BONUS_NUMBER_RANGE,):
            raise ValueError("fitted weight vectors have wrong length")
        if not (np.isfinite(main).all() and np.isfinite(bonus).all()):
            raise ValueError("fitted weights contain non-finite values")
        if (main <= 0).any() or (bonus <= 0).any():
            raise ValueError("fitted weights must be positive")
        return {
            "main": main / main.mean(),
            "bonus": bonus / bonus.mean(),
            "fitted_at": payload.get("fitted_at_utc", "unknown"),
        }
    except (ValueError, KeyError, json.JSONDecodeError, OSError) as exc:
        logger.warning("Ignoring invalid fitted weights file %s: %s", path, exc)
        return None


MAIN_POPULARITY = prior_main_weights()
BONUS_POPULARITY = prior_bonus_weights()
WEIGHTS_SOURCE = "heuristic priors"


def reload_weights(path: Path = FITTED_WEIGHTS_FILE) -> str:
    """Refresh module weights from the fitted file; returns the active source."""
    global MAIN_POPULARITY, BONUS_POPULARITY, WEIGHTS_SOURCE
    fitted = _load_fitted(path)
    if fitted is not None:
        MAIN_POPULARITY = fitted["main"]
        BONUS_POPULARITY = fitted["bonus"]
        WEIGHTS_SOURCE = f"fitted from winner counts ({fitted['fitted_at'][:10]})"
    else:
        MAIN_POPULARITY = prior_main_weights()
        BONUS_POPULARITY = prior_bonus_weights()
        WEIGHTS_SOURCE = "heuristic priors"
    return WEIGHTS_SOURCE


def weights_provenance() -> str:
    return WEIGHTS_SOURCE


reload_weights()


def ticket_popularity(main_numbers: Sequence[int], bonus_numbers: Sequence[int]) -> float:
    """
    Popularity score for a full ticket. 1.0 ~ an average random ticket;
    higher = more co-players expected, lower = fewer.
    """
    mains = sorted(int(value) for value in main_numbers)
    bonuses = sorted(int(value) for value in bonus_numbers)

    score = float(np.mean([MAIN_POPULARITY[number - 1] for number in mains]))
    score *= float(np.mean([BONUS_POPULARITY[number - 1] for number in bonuses])) ** 0.5

    score += _pattern_bump(mains)
    return float(score)


def _pattern_bump(mains: Sequence[int]) -> float:
    """Additive bumps for visually patterned tickets that humans over-play."""
    bump = 0.0
    values = list(mains)

    if all(number <= 31 for number in values):  # pure birthday ticket
        bump += 0.25

    longest_run = _longest_consecutive_run(values)
    if longest_run >= 3:
        bump += 0.15
    if longest_run >= 5:  # e.g. 1,2,3,4,5 — massively over-played
        bump += 0.45

    diffs = {b - a for a, b in zip(values, values[1:])}
    if len(diffs) == 1:  # arithmetic progression (10,20,30,40,50 etc.)
        bump += 0.20

    if all(number % 5 == 0 for number in values):
        bump += 0.10

    return bump


def _longest_consecutive_run(sorted_values: Sequence[int]) -> int:
    longest = 1
    current = 1
    for previous, value in zip(sorted_values, sorted_values[1:]):
        current = current + 1 if value == previous + 1 else 1
        longest = max(longest, current)
    return longest
