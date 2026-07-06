from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from .config import (
    BONUS_NUMBER_RANGE,
    DEFAULT_CANDIDATE_POOL_SIZE,
    DEFAULT_POPULARITY_WEIGHT,
    MAIN_NUMBER_RANGE,
    N_BONUS,
    N_MAIN,
)
from .popularity import ticket_popularity

EPSILON = 1e-12


@dataclass(frozen=True)
class CoverageConfig:
    overlap_penalty: float = 0.90
    coverage_bonus: float = 0.85
    candidate_pool_size: int = DEFAULT_CANDIDATE_POOL_SIZE
    popularity_weight: float = DEFAULT_POPULARITY_WEIGHT
    disjoint_main: bool = False

    @classmethod
    def from_dict(cls, payload: Dict[str, Any] | None) -> "CoverageConfig":
        payload = payload or {}
        return cls(
            overlap_penalty=float(payload.get("overlap_penalty", 0.90)),
            coverage_bonus=float(payload.get("coverage_bonus", 0.85)),
            candidate_pool_size=max(50, int(payload.get("candidate_pool_size", DEFAULT_CANDIDATE_POOL_SIZE))),
            popularity_weight=max(0.0, float(payload.get("popularity_weight", DEFAULT_POPULARITY_WEIGHT))),
            disjoint_main=bool(payload.get("disjoint_main", False)),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overlap_penalty": self.overlap_penalty,
            "coverage_bonus": self.coverage_bonus,
            "candidate_pool_size": self.candidate_pool_size,
            "popularity_weight": self.popularity_weight,
            "disjoint_main": self.disjoint_main,
        }


def normalize_probabilities(probabilities: Sequence[float], expected_size: int) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=float)
    if probs.shape != (expected_size,):
        raise ValueError(f"Expected probability vector of length {expected_size}, got {probs.shape}")
    probs = np.clip(probs, EPSILON, None)
    return probs / probs.sum()


def apply_temperature(probabilities: Sequence[float], temperature: float, expected_size: int) -> np.ndarray:
    probs = normalize_probabilities(probabilities, expected_size)
    if temperature <= 0 or abs(float(temperature) - 1.0) < 1e-9:
        return probs
    adjusted = probs ** (1.0 / float(temperature))
    return adjusted / adjusted.sum()


def build_candidate_pool(
    main_probabilities: Sequence[float],
    bonus_probabilities: Sequence[float],
    candidate_pool_size: int,
    seed: int,
    source: str,
) -> List[Dict[str, Any]]:
    main_probs = normalize_probabilities(main_probabilities, MAIN_NUMBER_RANGE)
    bonus_probs = normalize_probabilities(bonus_probabilities, BONUS_NUMBER_RANGE)
    rng = np.random.default_rng(seed)
    candidates: List[Dict[str, Any]] = []
    seen = set()

    def add_candidate(main_numbers: Sequence[int], bonus_numbers: Sequence[int]) -> None:
        main_tuple = tuple(sorted(int(value) for value in main_numbers))
        bonus_tuple = tuple(sorted(int(value) for value in bonus_numbers))
        combo_key = main_tuple + bonus_tuple
        if combo_key in seen:
            return
        seen.add(combo_key)
        candidates.append(build_candidate_record(main_tuple, bonus_tuple, main_probs, bonus_probs, source))

    top_main = tuple(np.argsort(main_probs)[-N_MAIN:] + 1)
    top_bonus = tuple(np.argsort(bonus_probs)[-N_BONUS:] + 1)
    add_candidate(top_main, top_bonus)

    attempts = max(candidate_pool_size * 6, 300)
    while len(candidates) < candidate_pool_size and attempts > 0:
        attempts -= 1
        main_numbers = rng.choice(
            np.arange(1, MAIN_NUMBER_RANGE + 1),
            size=N_MAIN,
            replace=False,
            p=main_probs,
        )
        bonus_numbers = rng.choice(
            np.arange(1, BONUS_NUMBER_RANGE + 1),
            size=N_BONUS,
            replace=False,
            p=bonus_probs,
        )
        add_candidate(main_numbers, bonus_numbers)

    candidates.sort(key=lambda candidate: candidate["base_score"], reverse=True)
    return candidates


def deduplicate_candidates(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduplicated: List[Dict[str, Any]] = []
    seen = set()
    for candidate in candidates:
        key = tuple(candidate["main_numbers"]) + tuple(candidate["bonus_numbers"])
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(candidate)
    return deduplicated


def select_diverse_tickets(
    candidates: List[Dict[str, Any]],
    num_tickets: int,
    main_probabilities: Sequence[float],
    bonus_probabilities: Sequence[float],
    overlap_penalty: float,
    coverage_bonus: float,
    source: str,
    popularity_weight: float = 0.0,
    disjoint_main: bool = False,
    ev_model: Any | None = None,
) -> List[Dict[str, Any]]:
    if num_tickets <= 0:
        return []

    main_probs = normalize_probabilities(main_probabilities, MAIN_NUMBER_RANGE)
    bonus_probs = normalize_probabilities(bonus_probabilities, BONUS_NUMBER_RANGE)
    remaining = [candidate.copy() for candidate in candidates]

    # Economic mode: rank in euros. The candidate's economic value replaces the
    # log-probability heuristic, and the diversification terms are rescaled by
    # the pool's EV spread so their *relative* strength matches legacy mode.
    diversity_scale = 1.0
    if ev_model is not None:
        ev_model.attach(remaining)
        ev_values = np.asarray([candidate["ev_eur"] for candidate in remaining], dtype=float)
        spread = float(ev_values.std())
        diversity_scale = spread if spread > 1e-9 else max(float(ev_values.mean()) * 0.01, 1e-6)

    selected: List[Dict[str, Any]] = []
    covered_main = set()
    covered_bonus = set()
    # Hard disjointness is only feasible while enough numbers remain uncovered.
    enforce_disjoint = bool(disjoint_main) and num_tickets * N_MAIN <= MAIN_NUMBER_RANGE

    while remaining and len(selected) < num_tickets:
        best_idx = -1
        best_score = -float("inf")

        for idx, candidate in enumerate(remaining):
            if enforce_disjoint and covered_main and any(
                number in covered_main for number in candidate["main_numbers"]
            ):
                continue
            score = float(candidate["ev_eur"]) if ev_model is not None else float(candidate["base_score"])
            if selected:
                worst_overlap = max(_overlap_cost(candidate, existing) for existing in selected)
                score -= float(overlap_penalty) * diversity_scale * worst_overlap

            coverage_gain = sum(main_probs[number - 1] for number in candidate["main_numbers"] if number not in covered_main)
            coverage_gain += 0.75 * sum(
                bonus_probs[number - 1] for number in candidate["bonus_numbers"] if number not in covered_bonus
            )
            score += float(coverage_bonus) * diversity_scale * coverage_gain

            # The EV objective already prices popularity in euros; the additive
            # popularity penalty only applies in legacy mode.
            if ev_model is None and popularity_weight > 0.0:
                score -= float(popularity_weight) * (float(candidate.get("popularity", 1.0)) - 1.0)

            if score > best_score:
                best_idx = idx
                best_score = score

        if best_idx < 0:
            # Disjoint pool exhausted: relax the hard constraint for remaining slots.
            enforce_disjoint = False
            continue

        chosen = remaining.pop(best_idx)
        chosen["source"] = source
        chosen["final_score"] = float(best_score)
        selected.append(chosen)
        covered_main.update(chosen["main_numbers"])
        covered_bonus.update(chosen["bonus_numbers"])

    return selected


class CoverageOptimizer:
    """Generate diversified coverage tickets from a uniform draw model."""

    def __init__(self, seed: int = 42, config: CoverageConfig | None = None):
        self.seed = int(seed)
        self.config = config or CoverageConfig()

    def generate(
        self,
        num_tickets: int,
        candidate_pool_size: int | None = None,
        ev_model: Any | None = None,
    ) -> List[Dict[str, Any]]:
        main_probs = np.full(MAIN_NUMBER_RANGE, 1.0 / MAIN_NUMBER_RANGE)
        bonus_probs = np.full(BONUS_NUMBER_RANGE, 1.0 / BONUS_NUMBER_RANGE)
        pool_size = max(50, int(candidate_pool_size or self.config.candidate_pool_size))
        candidates = build_candidate_pool(
            main_probs,
            bonus_probs,
            candidate_pool_size=pool_size,
            seed=self.seed,
            source="Coverage",
        )
        if ev_model is not None:
            candidates.extend(
                ev_model.anti_popular_candidates(
                    max(pool_size // 3, 50), self.seed, main_probs, bonus_probs, source="Coverage"
                )
            )
            candidates = deduplicate_candidates(candidates)
        return select_diverse_tickets(
            candidates,
            num_tickets=num_tickets,
            main_probabilities=main_probs,
            bonus_probabilities=bonus_probs,
            overlap_penalty=self.config.overlap_penalty,
            coverage_bonus=self.config.coverage_bonus,
            source="Coverage",
            popularity_weight=self.config.popularity_weight,
            disjoint_main=self.config.disjoint_main,
            ev_model=ev_model,
        )


def build_candidate_record(
    main_numbers: Tuple[int, ...],
    bonus_numbers: Tuple[int, ...],
    main_probabilities: np.ndarray,
    bonus_probabilities: np.ndarray,
    source: str,
) -> Dict[str, Any]:
    main_values = [int(value) for value in main_numbers]
    bonus_values = [int(value) for value in bonus_numbers]
    main_index = np.asarray(main_values, dtype=int) - 1
    bonus_index = np.asarray(bonus_values, dtype=int) - 1
    expected_main_hits = float(main_probabilities[main_index].sum())
    expected_bonus_hits = float(bonus_probabilities[bonus_index].sum())
    confidence = float((expected_main_hits / N_MAIN) * 0.75 + (expected_bonus_hits / N_BONUS) * 0.25)
    base_score = float(np.log(main_probabilities[main_index]).sum() + 0.55 * np.log(bonus_probabilities[bonus_index]).sum())
    return {
        "main_numbers": main_values,
        "bonus_numbers": bonus_values,
        "confidence": confidence,
        "expected_main_hits": expected_main_hits,
        "expected_bonus_hits": expected_bonus_hits,
        "base_score": base_score,
        "popularity": ticket_popularity(main_values, bonus_values),
        "source": source,
    }


def _overlap_cost(left: Dict[str, Any], right: Dict[str, Any]) -> float:
    main_overlap = len(set(left["main_numbers"]) & set(right["main_numbers"])) / float(N_MAIN)
    bonus_overlap = len(set(left["bonus_numbers"]) & set(right["bonus_numbers"])) / float(N_BONUS)
    return main_overlap + (0.7 * bonus_overlap)
