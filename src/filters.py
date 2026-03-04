import logging
from typing import List, Tuple

import numpy as np

from .config import N_MAIN, MAIN_NUMBER_RANGE

logger = logging.getLogger(__name__)

class StatisticalFilter:
    """
    Advanced statistical filters to reject improbable lottery combinations.
    Based on historical probability distributions.
    """
    
    def __init__(self):
        # Calculate expected sum range dynamically
        # Min possible sum: sum(1..N)
        # Max possible sum: sum(Range-N+1..Range)
        
        self.min_possible = sum(range(1, N_MAIN + 1))
        self.max_possible = sum(range(MAIN_NUMBER_RANGE - N_MAIN + 1, MAIN_NUMBER_RANGE + 1))
        
        # We want the "likely" range (approx middle 60-80%)
        # A simple heuristic is 20% buffer from min/max
        total_span = self.max_possible - self.min_possible
        buffer = int(total_span * 0.2)
        
        self.min_sum = self.min_possible + buffer
        self.max_sum = self.max_possible - buffer
        self.max_consecutive_pairs = 2
        self.max_decade_repeat = 3
        self.max_last_digit_repeat = 3
        
        # Fallback/Sanity check (ensure valid range)
        if self.min_sum >= self.max_sum:
            # Default to broad range if calculation fails (unlikely)
            self.min_sum = int(self.min_possible * 1.2)
            self.max_sum = int(self.max_possible * 0.8)

    def fit(self, historical_data) -> None:
        """
        Fit filter thresholds to historical draws.
        Keeps robust defaults when data is too small.
        """
        if historical_data is None or historical_data.empty or 'main_numbers' not in historical_data:
            return

        if len(historical_data) < 25:
            return

        main_series = historical_data['main_numbers']
        sums = main_series.apply(sum).values
        self.min_sum = int(max(self.min_possible, np.floor(np.quantile(sums, 0.05)) - 2))
        self.max_sum = int(min(self.max_possible, np.ceil(np.quantile(sums, 0.95)) + 2))
        if self.min_sum >= self.max_sum:
            self.min_sum = int(self.min_possible * 1.2)
            self.max_sum = int(self.max_possible * 0.8)

        consecutive = main_series.apply(self._count_consecutive_pairs).values
        decade_max = main_series.apply(self._max_decade_bucket_count).values
        digit_max = main_series.apply(self._max_last_digit_count).values

        # Slightly relaxed upper bounds to avoid over-filtering.
        self.max_consecutive_pairs = int(max(1, min(N_MAIN - 1, np.ceil(np.quantile(consecutive, 0.98)))))
        self.max_decade_repeat = int(max(2, min(N_MAIN, np.ceil(np.quantile(decade_max, 0.99)))))
        self.max_last_digit_repeat = int(max(2, min(N_MAIN, np.ceil(np.quantile(digit_max, 0.99)))))

        logger.info(
            "Fitted statistical filter thresholds: sum[%d,%d], max_consec=%d, max_decade=%d, max_last_digit=%d",
            self.min_sum,
            self.max_sum,
            self.max_consecutive_pairs,
            self.max_decade_repeat,
            self.max_last_digit_repeat,
        )
        
    def validate(self, main_nums: List[int]) -> Tuple[bool, str]:
        """
        Validate a set of main numbers against statistical rules.
        Returns (is_valid, reason).
        """
        if not self._check_sum(main_nums):
            return False, "Sum out of range"
            
        if not self._check_odd_even(main_nums):
            return False, "Improbable Odd/Even ratio"
            
        if not self._check_high_low(main_nums):
            return False, "Improbable High/Low ratio"
            
        if not self._check_consecutive(main_nums):
            return False, "Too many consecutive numbers"
            
        if not self._check_decade_distribution(main_nums):
            return False, "Unbalanced decade distribution"
            
        if not self._check_last_digit(main_nums):
            return False, "Too many same last digits"
            
        return True, "OK"

    def _check_sum(self, nums: List[int]) -> bool:
        total = sum(nums)
        return self.min_sum <= total <= self.max_sum

    def _check_odd_even(self, nums: List[int]) -> bool:
        # Acceptable ratios (Odd:Even): 2:3, 3:2, 1:4, 4:1
        # Reject extreme: 0:5, 5:0 (unless very confident, but here we filter)
        odds = sum(1 for n in nums if n % 2 != 0)
        return 1 <= odds <= 4

    def _check_high_low(self, nums: List[int]) -> bool:
        # High/Low split. Low = 1-25, High = 26-50
        # Acceptable: 2:3, 3:2, 1:4, 4:1
        mid = MAIN_NUMBER_RANGE // 2
        lows = sum(1 for n in nums if n <= mid)
        return 1 <= lows <= 4

    def _check_consecutive(self, nums: List[int]) -> bool:
        consecutive_count = self._count_consecutive_pairs(nums)
        return consecutive_count <= self.max_consecutive_pairs

    def _check_decade_distribution(self, nums: List[int]) -> bool:
        # 1-10, 11-20, ..., 41-50 bucket mapping.
        return self._max_decade_bucket_count(nums) <= self.max_decade_repeat

    def _check_last_digit(self, nums: List[int]) -> bool:
        return self._max_last_digit_count(nums) <= self.max_last_digit_repeat

    def _count_consecutive_pairs(self, nums: List[int]) -> int:
        sorted_nums = sorted(nums)
        consecutive_count = 0
        for i in range(len(sorted_nums) - 1):
            if sorted_nums[i + 1] - sorted_nums[i] == 1:
                consecutive_count += 1
        return consecutive_count

    def _max_decade_bucket_count(self, nums: List[int]) -> int:
        decades = {}
        for n in nums:
            decade_bucket = (n - 1) // 10
            decades[decade_bucket] = decades.get(decade_bucket, 0) + 1
        return max(decades.values()) if decades else 0

    def _max_last_digit_count(self, nums: List[int]) -> int:
        counts = {}
        for n in nums:
            d = n % 10
            counts[d] = counts.get(d, 0) + 1
        return max(counts.values()) if counts else 0
