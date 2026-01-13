import itertools
import logging
from typing import Dict, List, Tuple

import numpy as np

from .config import BONUS_NUMBER_RANGE, MAIN_NUMBER_RANGE, N_BONUS, N_MAIN

logger = logging.getLogger(__name__)


class CoverageOptimizer:
    """Generate tickets that maximize pair coverage across a ticket budget."""

    def __init__(
        self,
        main_range: int = MAIN_NUMBER_RANGE,
        bonus_range: int = BONUS_NUMBER_RANGE,
        n_main: int = N_MAIN,
        n_bonus: int = N_BONUS,
        seed: int = 42,
    ):
        self.main_range = main_range
        self.bonus_range = bonus_range
        self.n_main = n_main
        self.n_bonus = n_bonus
        self.rng = np.random.default_rng(seed)
        self._reset_coverage()

    def _reset_coverage(self) -> None:
        self.covered_main_pairs = set()
        self.covered_bonus_pairs = set()
        self.covered_main_nums = set()
        self.covered_bonus_nums = set()
        self.covered_cross_pairs = set()

    def _sample_ticket(self) -> Tuple[np.ndarray, np.ndarray]:
        main_nums = self.rng.choice(
            np.arange(1, self.main_range + 1),
            size=self.n_main,
            replace=False,
        )
        bonus_nums = self.rng.choice(
            np.arange(1, self.bonus_range + 1),
            size=self.n_bonus,
            replace=False,
        )
        return np.sort(main_nums), np.sort(bonus_nums)

    def _score_ticket(self, main_nums: np.ndarray, bonus_nums: np.ndarray) -> float:
        main_pairs = list(itertools.combinations(main_nums, 2))
        bonus_pairs = list(itertools.combinations(bonus_nums, 2))
        cross_pairs = [(m, b) for m in main_nums for b in bonus_nums]

        new_main_pairs = sum(1 for p in main_pairs if p not in self.covered_main_pairs)
        new_bonus_pairs = sum(1 for p in bonus_pairs if p not in self.covered_bonus_pairs)
        new_main_nums = sum(1 for n in main_nums if n not in self.covered_main_nums)
        new_bonus_nums = sum(1 for n in bonus_nums if n not in self.covered_bonus_nums)
        new_cross_pairs = sum(1 for p in cross_pairs if p not in self.covered_cross_pairs)

        return (
            1.0 * new_main_pairs
            + 1.0 * new_bonus_pairs
            + 0.2 * new_main_nums
            + 0.3 * new_bonus_nums
            + 0.05 * new_cross_pairs
        )

    def _update_coverage(self, main_nums: np.ndarray, bonus_nums: np.ndarray) -> None:
        for pair in itertools.combinations(main_nums, 2):
            self.covered_main_pairs.add(pair)
        for pair in itertools.combinations(bonus_nums, 2):
            self.covered_bonus_pairs.add(pair)
        for n in main_nums:
            self.covered_main_nums.add(int(n))
        for n in bonus_nums:
            self.covered_bonus_nums.add(int(n))
        for m in main_nums:
            for b in bonus_nums:
                self.covered_cross_pairs.add((int(m), int(b)))

    def generate(
        self,
        num_tickets: int,
        candidates_per_ticket: int = 2000,
    ) -> List[Dict]:
        self._reset_coverage()
        tickets = []
        seen = set()

        if num_tickets <= 0:
            return tickets

        logger.info(
            "Generating %d coverage-optimized tickets (%d candidates each)...",
            num_tickets,
            candidates_per_ticket,
        )

        for _ in range(num_tickets):
            best = None
            best_score = -1.0

            for _ in range(candidates_per_ticket):
                main_nums, bonus_nums = self._sample_ticket()
                combo_key = tuple(main_nums) + tuple(bonus_nums)
                if combo_key in seen:
                    continue

                score = self._score_ticket(main_nums, bonus_nums)
                if score > best_score:
                    best = (main_nums, bonus_nums)
                    best_score = score

            if best is None:
                break

            main_nums, bonus_nums = best
            self._update_coverage(main_nums, bonus_nums)
            seen.add(tuple(main_nums) + tuple(bonus_nums))
            tickets.append(
                {
                    "main_numbers": main_nums.tolist(),
                    "bonus_numbers": bonus_nums.tolist(),
                    "coverage_score": float(best_score),
                    "source": "Coverage",
                }
            )

        return tickets
