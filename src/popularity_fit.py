"""
Empirical calibration of number popularity from official winner counts.

Why this works: every EuroMillions prize tier is won by tickets sharing k main
numbers and s stars with the draw. For a player pool of (unknown) size N with
number-level pick weights w (mains) and v (stars), the expected winner count at
tier (k, s) is approximately

    E[W] = N * q(k, s) * mean_w_drawn^k * mean_w_undrawn^(5-k)
                       * mean_v_drawn^s * mean_v_undrawn^(2-s)

where q(k, s) is the exact hypergeometric tier probability for a uniformly
random ticket. Taking *ratios of adjacent tiers within the same draw* cancels
the unknown N (and every draw-level effect: jackpot size, weekday, sales):

    log(W_a / W_b) - log(q_a / q_b) = log(mean_w_drawn / mean_w_undrawn)

for tiers differing by one main hit, and the analogous star expression for
tiers differing by one star hit. Each draw therefore yields a direct, sales-free
*measurement* of how over-picked its drawn numbers were. Winner counts at the
low tiers are 10^4-10^6, so Poisson noise on these measurements is tiny.

Inverting the measurement gives the mean pick-weight of the 5 drawn numbers,
which is linear in per-number weights -> ordinary least squares on
interpretable features (calendar range, month range, lucky numbers, trend)
recovers the weight of every number. No SciPy required.

This estimates *number-level* popularity. Combination-level pattern effects
(e.g. 1-2-3-4-5) barely move low-tier counts and stay as documented priors in
``popularity.py``.

References: Farrell, Hartley, Lanot & Walker (2000); Haigh (1997); Riedwyl &
Henze (1998) pioneered fitting conscious selection from winner counts.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from math import comb
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import BONUS_NUMBER_RANGE, DATA_DIR, MAIN_NUMBER_RANGE, N_BONUS, N_MAIN, PREDICTIONS_DIR
from .winner_data import RANK_TIERS, load_winner_counts, winners_matrix

logger = logging.getLogger(__name__)

FITTED_WEIGHTS_FILE = DATA_DIR / "popularity_fitted.json"
FIT_REPORT_FILE = PREDICTIONS_DIR / "popularity_fit_report.json"

_LUCKY_MAIN = (3, 7, 11, 13, 17)

# Within-draw tier-ratio contrasts. Ranks must share the other dimension.
# (rank_numerator, rank_denominator), using the FDJ rank order of RANK_TIERS
# (rank 6 = 3+2, rank 7 = 4+0).
MAIN_CONTRASTS: Tuple[Tuple[int, int], ...] = (
    (5, 9),    # (4,1) / (3,1)
    (7, 10),   # (4,0) / (3,0)
    (9, 12),   # (3,1) / (2,1)
    (10, 13),  # (3,0) / (2,0)
    (8, 11),   # (2,2) / (1,2)
)
STAR_CONTRASTS: Tuple[Tuple[int, int], ...] = (
    (5, 7),    # (4,1) / (4,0)
    (6, 9),    # (3,2) / (3,1)
    (9, 10),   # (3,1) / (3,0)
    (8, 12),   # (2,2) / (2,1)
    (12, 13),  # (2,1) / (2,0)
)
MIN_TIER_COUNT = 20  # skip a contrast when either tier has fewer winners


def tier_probability(main_hits: int, star_hits: int) -> float:
    """Exact P(tier) for one uniformly random ticket."""
    main_part = comb(N_MAIN, main_hits) * comb(MAIN_NUMBER_RANGE - N_MAIN, N_MAIN - main_hits)
    star_part = comb(N_BONUS, star_hits) * comb(BONUS_NUMBER_RANGE - N_BONUS, N_BONUS - star_hits)
    return (
        main_part / comb(MAIN_NUMBER_RANGE, N_MAIN)
        * star_part / comb(BONUS_NUMBER_RANGE, N_BONUS)
    )


RANK_PROBABILITIES: Dict[int, float] = {
    rank: tier_probability(*tier) for rank, tier in RANK_TIERS.items()
}


def main_feature_matrix() -> Tuple[np.ndarray, List[str]]:
    numbers = np.arange(1, MAIN_NUMBER_RANGE + 1)
    features = np.column_stack(
        [
            np.ones_like(numbers, dtype=float),
            (numbers <= 31).astype(float),
            (numbers <= 12).astype(float),
            np.isin(numbers, _LUCKY_MAIN).astype(float),
            (numbers == 7).astype(float),
            (numbers >= 41).astype(float),
            (numbers - numbers.mean()) / MAIN_NUMBER_RANGE,
        ]
    )
    names = ["intercept", "calendar_day", "month_range", "lucky_set", "seven", "high_range", "linear_trend"]
    return features, names


def star_feature_matrix() -> Tuple[np.ndarray, List[str]]:
    numbers = np.arange(1, BONUS_NUMBER_RANGE + 1)
    features = np.column_stack(
        [
            np.ones_like(numbers, dtype=float),
            (numbers == 7).astype(float),
            (numbers <= 9).astype(float),
            (numbers - numbers.mean()) / BONUS_NUMBER_RANGE,
        ]
    )
    names = ["intercept", "seven", "legacy_range", "linear_trend"]
    return features, names


def _contrast_measurements(
    winners: np.ndarray,
    contrasts: Sequence[Tuple[int, int]],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Per-draw inverse-variance-weighted measurement of
    log(mean_weight_drawn / mean_weight_undrawn), plus its standard error.
    Returns (y, sigma); entries are NaN when no contrast was usable.
    """
    n_draws = winners.shape[0]
    y = np.full(n_draws, np.nan)
    sigma = np.full(n_draws, np.nan)

    log_q = {rank: np.log(RANK_PROBABILITIES[rank]) for rank in RANK_TIERS}
    for draw_idx in range(n_draws):
        estimates: List[float] = []
        precisions: List[float] = []
        for rank_a, rank_b in contrasts:
            count_a = winners[draw_idx, rank_a - 1]
            count_b = winners[draw_idx, rank_b - 1]
            if count_a < MIN_TIER_COUNT or count_b < MIN_TIER_COUNT:
                continue
            measurement = np.log(count_a / count_b) - (log_q[rank_a] - log_q[rank_b])
            variance = (1.0 / count_a) + (1.0 / count_b)  # Poisson, delta method
            estimates.append(measurement)
            precisions.append(1.0 / variance)
        if not estimates:
            continue
        weights = np.asarray(precisions)
        values = np.asarray(estimates)
        y[draw_idx] = float(np.sum(weights * values) / np.sum(weights))
        sigma[draw_idx] = float(np.sqrt(1.0 / np.sum(weights)))
    return y, sigma


def _invert_main(y: np.ndarray) -> np.ndarray:
    """Solve m / ((50 - 5m)/45) = e^y for m (mean weight of drawn mains)."""
    expy = np.exp(y)
    return (MAIN_NUMBER_RANGE * expy) / ((MAIN_NUMBER_RANGE - N_MAIN) + N_MAIN * expy)


def _invert_star(y: np.ndarray) -> np.ndarray:
    """Solve v / ((12 - 2v)/10) = e^y for v (mean weight of drawn stars)."""
    expy = np.exp(y)
    return (BONUS_NUMBER_RANGE * expy) / ((BONUS_NUMBER_RANGE - N_BONUS) + N_BONUS * expy)


def _forward_main(mean_drawn: np.ndarray) -> np.ndarray:
    undrawn = (MAIN_NUMBER_RANGE - N_MAIN * mean_drawn) / (MAIN_NUMBER_RANGE - N_MAIN)
    return np.log(mean_drawn / undrawn)


def _forward_star(mean_drawn: np.ndarray) -> np.ndarray:
    undrawn = (BONUS_NUMBER_RANGE - N_BONUS * mean_drawn) / (BONUS_NUMBER_RANGE - N_BONUS)
    return np.log(mean_drawn / undrawn)


def _drawn_feature_means(
    number_lists: Sequence[Sequence[int]],
    features: np.ndarray,
) -> np.ndarray:
    rows = np.zeros((len(number_lists), features.shape[1]))
    for idx, numbers in enumerate(number_lists):
        rows[idx] = features[np.asarray(numbers, dtype=int) - 1].mean(axis=0)
    return rows


def _weighted_least_squares(
    design: np.ndarray,
    response: np.ndarray,
    sigma: np.ndarray,
) -> np.ndarray:
    scale = 1.0 / np.clip(sigma, 1e-9, None)
    coefficients, *_ = np.linalg.lstsq(design * scale[:, None], response * scale, rcond=None)
    return coefficients


def _normalized_weights(features: np.ndarray, coefficients: np.ndarray, floor: float = 0.25) -> np.ndarray:
    weights = np.clip(features @ coefficients, floor, None)
    return weights / weights.mean()


@dataclass
class FitResult:
    main_weights: np.ndarray
    bonus_weights: np.ndarray
    main_coefficients: Dict[str, float]
    bonus_coefficients: Dict[str, float]
    diagnostics: Dict[str, object] = field(default_factory=dict)

    def to_payload(self) -> Dict[str, object]:
        return {
            "fitted_at_utc": datetime.now(timezone.utc).isoformat(),
            "main_weights": [round(float(value), 6) for value in self.main_weights],
            "bonus_weights": [round(float(value), 6) for value in self.bonus_weights],
            "main_coefficients": self.main_coefficients,
            "bonus_coefficients": self.bonus_coefficients,
            "diagnostics": self.diagnostics,
        }


def _holdout_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    actual = y_true[mask]
    predicted = y_pred[mask]
    if actual.size < 3:
        return {"n": int(actual.size)}
    residual = actual - predicted
    baseline = actual - actual.mean()
    r_squared = 1.0 - float(np.sum(residual**2) / np.sum(baseline**2))
    correlation = float(np.corrcoef(actual, predicted)[0, 1])
    return {
        "n": int(actual.size),
        "r_squared": round(r_squared, 4),
        "correlation": round(correlation, 4),
        "rmse": round(float(np.sqrt(np.mean(residual**2))), 5),
        "actual_sd": round(float(actual.std()), 5),
    }


def _rank_mapping_check(winners: np.ndarray) -> Dict[str, float]:
    """Mean winner counts must track exact tier probabilities (rank mapping sanity)."""
    mean_counts = winners.mean(axis=0)
    usable = mean_counts > 0
    log_counts = np.log(mean_counts[usable])
    log_probs = np.log(np.asarray([RANK_PROBABILITIES[rank] for rank in RANK_TIERS])[usable])
    correlation = float(np.corrcoef(log_counts, log_probs)[0, 1])
    implied_sales = float(np.median(winners.sum(axis=1) / sum(RANK_PROBABILITIES.values())))
    return {
        "log_count_vs_log_prob_correlation": round(correlation, 4),
        "implied_median_tickets_per_draw": round(implied_sales, 0),
    }


def fit_popularity(
    data: Optional[pd.DataFrame] = None,
    holdout_fraction: float = 0.2,
    prior_main_weights: Optional[np.ndarray] = None,
    prior_bonus_weights: Optional[np.ndarray] = None,
) -> FitResult:
    if data is None:
        data = load_winner_counts()
    if len(data) < 60:
        raise ValueError(
            f"Only {len(data)} draws with winner counts; need at least 60 for a stable fit."
        )

    winners = winners_matrix(data)
    rank_check = _rank_mapping_check(winners)
    if rank_check["log_count_vs_log_prob_correlation"] < 0.95:
        raise ValueError(
            "Winner counts do not track tier probabilities "
            f"(corr={rank_check['log_count_vs_log_prob_correlation']}); "
            "rank mapping for this archive vintage looks wrong."
        )

    y_main, sigma_main = _contrast_measurements(winners, MAIN_CONTRASTS)
    y_star, sigma_star = _contrast_measurements(winners, STAR_CONTRASTS)

    main_features, main_names = main_feature_matrix()
    star_features, star_names = star_feature_matrix()
    main_design = _drawn_feature_means(list(data["main_numbers"]), main_features)
    star_design = _drawn_feature_means(list(data["bonus_numbers"]), star_features)

    n_draws = len(data)
    split = max(int(round(n_draws * (1.0 - holdout_fraction))), 30)

    def fit_segment(
        design: np.ndarray,
        response: np.ndarray,
        sigma: np.ndarray,
        invert,
        rows: slice,
    ) -> np.ndarray:
        mask = np.isfinite(response[rows])
        segment_design = design[rows][mask]
        segment_target = invert(response[rows][mask])
        segment_sigma = sigma[rows][mask]
        return _weighted_least_squares(segment_design, segment_target, segment_sigma)

    # --- holdout evaluation (fit on the past, score on the future) ---
    train = slice(0, split)
    test = slice(split, n_draws)
    main_coeff_train = fit_segment(main_design, y_main, sigma_main, _invert_main, train)
    star_coeff_train = fit_segment(star_design, y_star, sigma_star, _invert_star, train)
    fitted_main_train = _normalized_weights(main_features, main_coeff_train)
    fitted_star_train = _normalized_weights(star_features, star_coeff_train)

    def predicted_y(weights: np.ndarray, number_lists: Sequence[Sequence[int]], forward) -> np.ndarray:
        means = np.asarray(
            [weights[np.asarray(numbers, dtype=int) - 1].mean() for numbers in number_lists]
        )
        return forward(means)

    validation: Dict[str, object] = {
        "holdout_draws": int(n_draws - split),
        "main_fitted": _holdout_metrics(
            y_main[test],
            predicted_y(fitted_main_train, list(data["main_numbers"])[split:], _forward_main),
        ),
        "star_fitted": _holdout_metrics(
            y_star[test],
            predicted_y(fitted_star_train, list(data["bonus_numbers"])[split:], _forward_star),
        ),
    }
    if prior_main_weights is not None:
        validation["main_prior_heuristic"] = _holdout_metrics(
            y_main[test],
            predicted_y(
                prior_main_weights / prior_main_weights.mean(),
                list(data["main_numbers"])[split:],
                _forward_main,
            ),
        )
    if prior_bonus_weights is not None:
        validation["star_prior_heuristic"] = _holdout_metrics(
            y_star[test],
            predicted_y(
                prior_bonus_weights / prior_bonus_weights.mean(),
                list(data["bonus_numbers"])[split:],
                _forward_star,
            ),
        )

    # --- final fit on all draws ---
    everything = slice(0, n_draws)
    main_coefficients = fit_segment(main_design, y_main, sigma_main, _invert_main, everything)
    star_coefficients = fit_segment(star_design, y_star, sigma_star, _invert_star, everything)
    main_weights = _normalized_weights(main_features, main_coefficients)
    bonus_weights = _normalized_weights(star_features, star_coefficients)

    diagnostics: Dict[str, object] = {
        "n_draws": int(n_draws),
        "date_range": [str(data["date"].iloc[0].date()), str(data["date"].iloc[-1].date())],
        "scope_counts": (
            {str(key): int(value) for key, value in data["scope"].value_counts().items()}
            if "scope" in data.columns
            else {}
        ),
        "rank_mapping_check": rank_check,
        "usable_main_measurements": int(np.isfinite(y_main).sum()),
        "usable_star_measurements": int(np.isfinite(y_star).sum()),
        "median_measurement_se_main": round(float(np.nanmedian(sigma_main)), 5),
        "main_signal_sd": round(float(np.nanstd(y_main)), 5),
        "star_signal_sd": round(float(np.nanstd(y_star)), 5),
        "validation": validation,
    }

    return FitResult(
        main_weights=main_weights,
        bonus_weights=bonus_weights,
        main_coefficients={
            name: round(float(value), 5) for name, value in zip(main_names, main_coefficients)
        },
        bonus_coefficients={
            name: round(float(value), 5) for name, value in zip(star_names, star_coefficients)
        },
        diagnostics=diagnostics,
    )


def run_calibration(write_files: bool = True) -> FitResult:
    """Fit from archives and persist fitted weights + a diagnostics report."""
    from . import popularity  # local import to avoid a cycle

    result = fit_popularity(
        prior_main_weights=popularity.prior_main_weights(),
        prior_bonus_weights=popularity.prior_bonus_weights(),
    )
    if write_files:
        FITTED_WEIGHTS_FILE.write_text(json.dumps(result.to_payload(), indent=2), encoding="utf-8")
        FIT_REPORT_FILE.write_text(json.dumps(result.to_payload(), indent=2), encoding="utf-8")
        logger.info("Wrote fitted weights to %s", FITTED_WEIGHTS_FILE)
        popularity.reload_weights()
    return result


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    result = run_calibration()
    print("=== Popularity calibration ===")
    print(json.dumps(result.to_payload(), indent=2))
    print(f"Weights file: {FITTED_WEIGHTS_FILE}")
    print(f"Report file:  {FIT_REPORT_FILE}")


if __name__ == "__main__":
    main()
