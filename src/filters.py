from typing import List, Tuple
from .config import N_MAIN, MAIN_NUMBER_RANGE

class StatisticalFilter:
    """
    Advanced statistical filters to reject improbable lottery combinations.
    Based on historical probability distributions.
    """
    
    def __init__(self):
        # Calculate expected sum range dynamically
        # Min possible sum: sum(1..N)
        # Max possible sum: sum(Range-N+1..Range)
        
        min_possible = sum(range(1, N_MAIN + 1))
        max_possible = sum(range(MAIN_NUMBER_RANGE - N_MAIN + 1, MAIN_NUMBER_RANGE + 1))
        
        # We want the "likely" range (approx middle 60-80%)
        # A simple heuristic is 20% buffer from min/max
        total_span = max_possible - min_possible
        buffer = int(total_span * 0.2)
        
        self.min_sum = min_possible + buffer
        self.max_sum = max_possible - buffer
        
        # Fallback/Sanity check (ensure valid range)
        if self.min_sum >= self.max_sum:
            # Default to broad range if calculation fails (unlikely)
            self.min_sum = int(min_possible * 1.2)
            self.max_sum = int(max_possible * 0.8)
        
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
        sorted_nums = sorted(nums)
        consecutive_count = 0
        for i in range(len(sorted_nums) - 1):
            if sorted_nums[i+1] - sorted_nums[i] == 1:
                consecutive_count += 1
        # Reject if 3 or more consecutive numbers (which means 2 or more steps of 1)
        # Also rejects 2 separate pairs (1,2, 5,6) which is also rare-ish but maybe acceptable.
        # For now, strict: max 1 consecutive pair.
        return consecutive_count < 2

    def _check_decade_distribution(self, nums: List[int]) -> bool:
        # Decades: 0-9, 10-19, etc.
        # Assuming 1-based indexing: 1-9 (0s), 10-19 (10s), 20-29 (20s)...
        decades = {}
        for n in nums:
            d = n // 10
            decades[d] = decades.get(d, 0) + 1
        # Reject if any decade has > 3 numbers
        return max(decades.values()) <= 3

    def _check_last_digit(self, nums: List[int]) -> bool:
        last_digits = [n % 10 for n in nums]
        counts = {}
        for d in last_digits:
            counts[d] = counts.get(d, 0) + 1
        # Reject if any digit appears > 3 times
        return max(counts.values()) <= 3
