import pandas as pd
import numpy as np
import re
from datetime import datetime
import logging
from typing import Tuple, List, Dict
from .config import DATA_FILE, N_MAIN

logger = logging.getLogger(__name__)

class LotteryDataManager:
    """Unified data manager for lottery analysis."""
    
    def __init__(self, file_path: str = str(DATA_FILE)):
        self.file_path = file_path
        self.data = None
        
    def load_data(self) -> pd.DataFrame:
        """Load and parse lottery data from file."""
        if self.data is not None:
            return self.data
            
        try:
            with open(self.file_path, 'r') as f:
                content = f.read()
                
            # Regex to parse the specific format:
            # "Tuesday\n4th March 2025 10 15 20 25 30 05 08 €100,000,000 Won"
            # Handles newlines and variable spacing
            draw_pattern = r"((?:\w+)\s+\d+(?:st|nd|rd|th)?\s+(?:\w+)\s+\d{4})\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(€[\d,]+)\s+(Roll|Won)"
            
            draws = re.findall(draw_pattern, content)
            
            parsed_data = []
            for draw in draws:
                date_str = draw[0].replace('\n', ' ')
                # Remove ordinal suffixes (st, nd, rd, th) for easier parsing
                date_str = re.sub(r'(\d+)(st|nd|rd|th)', r'\1', date_str)
                
                try:
                    date = datetime.strptime(date_str, "%A %d %B %Y")
                except ValueError:
                    # Fallback for manual parsing if needed
                    continue
                    
                numbers = [int(x) for x in draw[1:8]]
                main_numbers = numbers[:N_MAIN]
                bonus_numbers = numbers[N_MAIN:]
                
                parsed_data.append({
                    'date': date,
                    'main_numbers': main_numbers,
                    'bonus_numbers': bonus_numbers,
                    'jackpot': draw[8],
                    'result': draw[9]
                })
                
            self.data = pd.DataFrame(parsed_data)
            self.data.sort_values('date', inplace=True)
            self.data.reset_index(drop=True, inplace=True)
            
            logger.info(f"Successfully loaded {len(self.data)} draws.")
            return self.data
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def calculate_gap_states(self, data: pd.DataFrame, main_range: int, bonus_range: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the 'Gap State' for each draw.
        A Gap State is a vector where index i represents how many draws ago number i+1 appeared.
        
        Returns:
            main_gaps: Array of shape (n_draws, main_range)
            bonus_gaps: Array of shape (n_draws, bonus_range)
        """
        n_draws = len(data)
        main_gaps = np.zeros((n_draws, main_range))
        bonus_gaps = np.zeros((n_draws, bonus_range))
        
        # Initialize counters (start with a reasonable default, e.g., range size, assuming they haven't appeared recently)
        current_main_gaps = np.full(main_range, main_range // 2)
        current_bonus_gaps = np.full(bonus_range, bonus_range // 2)
        
        for i, row in data.iterrows():
            # Record current state BEFORE updating with this draw's numbers
            # The model needs to predict THIS draw based on the gaps leading UP TO it.
            # So for row i, the input features should be the gaps *before* row i's numbers are revealed.
            main_gaps[i] = current_main_gaps.copy()
            bonus_gaps[i] = current_bonus_gaps.copy()
            
            # Update Main Gaps
            # Increment all gaps by 1
            current_main_gaps += 1
            # Reset gaps for numbers that appeared in this draw
            for num in row['main_numbers']:
                if 1 <= num <= main_range:
                    current_main_gaps[num-1] = 0
                    
            # Update Bonus Gaps
            current_bonus_gaps += 1
            for num in row['bonus_numbers']:
                if 1 <= num <= bonus_range:
                    current_bonus_gaps[num-1] = 0
                    
        return main_gaps, bonus_gaps

    def calculate_frequency_features(self, data: pd.DataFrame, windows: List[int] = [10, 50, 100]) -> Dict[str, np.ndarray]:
        """
        Calculate rolling frequency of numbers.
        Returns a dictionary where keys are 'main_win_X' or 'bonus_win_X' and values are arrays of shape (n_draws, number_range).
        """
        n_draws = len(data)
        features = {}
        
        # Pre-calculate presence matrices
        from .config import MAIN_NUMBER_RANGE, BONUS_NUMBER_RANGE
        
        main_presence = np.zeros((n_draws, MAIN_NUMBER_RANGE))
        bonus_presence = np.zeros((n_draws, BONUS_NUMBER_RANGE))
        
        for i, row in data.iterrows():
            for num in row['main_numbers']:
                if 1 <= num <= MAIN_NUMBER_RANGE:
                    main_presence[i, num-1] = 1
            for num in row['bonus_numbers']:
                if 1 <= num <= BONUS_NUMBER_RANGE:
                    bonus_presence[i, num-1] = 1
                    
        # Calculate rolling sums
        for w in windows:
            # We want the frequency in the *previous* w draws relative to the current row
            # So we shift by 1 and then roll
            # Rolling sum of the last w rows
            
            # Main
            # Shift 1 to not include current draw in its own prediction features
            # (Though for training at row i, we want features from i-1, i-2... so we just use the rolling sum up to i-1)
            # Pandas rolling is inclusive of the current row by default.
            # So we calculate rolling on the whole series, then shift 1.
            
            main_roll = pd.DataFrame(main_presence).rolling(window=w, min_periods=1).sum().shift(1).fillna(0).values
            bonus_roll = pd.DataFrame(bonus_presence).rolling(window=w, min_periods=1).sum().shift(1).fillna(0).values
            
            # Normalize by window size to get frequency/probability
            features[f'main_freq_{w}'] = main_roll / w
            features[f'bonus_freq_{w}'] = bonus_roll / w
            
        return features

    def calculate_cooccurrence_features(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate Co-occurrence (Affinity) features.
        For each draw t, calculates a vector for t+1 where index i is the 
        sum of co-occurrence counts between number i+1 and all numbers in draw t.
        
        Returns:
            main_affinity: (n_draws, MAIN_NUMBER_RANGE)
            bonus_affinity: (n_draws, BONUS_NUMBER_RANGE)
        """
        from .config import MAIN_NUMBER_RANGE, BONUS_NUMBER_RANGE
        
        n_draws = len(data)
        
        # 1. Build Global Co-occurrence Matrix (Main)
        # Count how often num i and num j appear together
        main_cooc = np.zeros((MAIN_NUMBER_RANGE, MAIN_NUMBER_RANGE))
        
        for _, row in data.iterrows():
            nums = [n for n in row['main_numbers'] if 1 <= n <= MAIN_NUMBER_RANGE]
            for n1 in nums:
                for n2 in nums:
                    if n1 != n2:
                        main_cooc[n1-1, n2-1] += 1
                        
        # Normalize rows to get probabilities P(n2 | n1)
        # Add 1 to avoid division by zero
        main_cooc_norm = main_cooc / (np.sum(main_cooc, axis=1, keepdims=True) + 1e-9)
        
        # 2. Build Global Co-occurrence Matrix (Bonus)
        bonus_cooc = np.zeros((BONUS_NUMBER_RANGE, BONUS_NUMBER_RANGE))
        for _, row in data.iterrows():
            nums = [n for n in row['bonus_numbers'] if 1 <= n <= BONUS_NUMBER_RANGE]
            for n1 in nums:
                for n2 in nums:
                    if n1 != n2:
                        bonus_cooc[n1-1, n2-1] += 1
                        
        bonus_cooc_norm = bonus_cooc / (np.sum(bonus_cooc, axis=1, keepdims=True) + 1e-9)
        
        # 3. Calculate Affinity Features for each draw
        main_affinity = np.zeros((n_draws, MAIN_NUMBER_RANGE))
        bonus_affinity = np.zeros((n_draws, BONUS_NUMBER_RANGE))
        
        # For the first draw, we have no history, so 0 affinity
        # For draw i, we look at numbers in draw i-1
        for i in range(1, n_draws):
            prev_row = data.iloc[i-1]
            
            # Main Affinity
            prev_main = [n for n in prev_row['main_numbers'] if 1 <= n <= MAIN_NUMBER_RANGE]
            if prev_main:
                # Sum of probabilities: For each candidate 'c', sum P(c | p) for all p in prev_main
                # This tells us: "How likely is 'c' given the numbers we just saw?"
                for c in range(MAIN_NUMBER_RANGE):
                    score = 0
                    for p in prev_main:
                        score += main_cooc_norm[p-1, c]
                    main_affinity[i, c] = score
                    
            # Bonus Affinity
            prev_bonus = [n for n in prev_row['bonus_numbers'] if 1 <= n <= BONUS_NUMBER_RANGE]
            if prev_bonus:
                for c in range(BONUS_NUMBER_RANGE):
                    score = 0
                    for p in prev_bonus:
                        score += bonus_cooc_norm[p-1, c]
                    bonus_affinity[i, c] = score
                    
        return main_affinity, bonus_affinity

    def calculate_date_features(self, data: pd.DataFrame) -> np.ndarray:
        """
        Calculate Date-based features (Day of Week, Month).
        Returns array of shape (n_draws, 7 + 12).
        """
        n_draws = len(data)
        # Day of Week (0-6) -> One-Hot (7)
        dow = data['date'].dt.dayofweek.values
        dow_onehot = np.zeros((n_draws, 7))
        dow_onehot[np.arange(n_draws), dow] = 1
        
        # Month (1-12) -> One-Hot (12) - adjust to 0-11 index
        month = data['date'].dt.month.values - 1
        month_onehot = np.zeros((n_draws, 12))
        month_onehot[np.arange(n_draws), month] = 1
        
        return np.hstack([dow_onehot, month_onehot])

    def calculate_hot_cold_features(self, data: pd.DataFrame, window: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate Hot/Cold status features.
        Returns continuous normalized values:
        Hot: Frequency in last 'window' draws / window size.
        Cold: Gap size / max_gap (normalized to 0-1 range approx).
        
        Returns:
            main_hot_cold: (n_draws, MAIN_NUMBER_RANGE * 2) [Hot scores, Cold scores]
            bonus_hot_cold: (n_draws, BONUS_NUMBER_RANGE * 2)
        """
        from .config import MAIN_NUMBER_RANGE, BONUS_NUMBER_RANGE
        
        n_draws = len(data)
        
        # We can reuse gap states for Cold calculation
        main_gaps, bonus_gaps = self.calculate_gap_states(data, MAIN_NUMBER_RANGE, BONUS_NUMBER_RANGE)
        
        # Main Hot (Frequency)
        main_presence = np.zeros((n_draws, MAIN_NUMBER_RANGE))
        for i, row in data.iterrows():
            for num in row['main_numbers']:
                if 1 <= num <= MAIN_NUMBER_RANGE:
                    main_presence[i, num-1] = 1
                    
        main_roll = pd.DataFrame(main_presence).rolling(window=window, min_periods=1).sum().shift(1).fillna(0).values
        main_hot = main_roll / window # Normalized frequency (0.0 to 1.0)
        
        # Main Cold (Normalized Gap)
        # Normalize by a reasonable max gap (e.g., 50) to keep values in 0-1 range mostly
        # Sigmoid or simple division? Simple division is fine for trees/NN.
        main_cold = np.clip(main_gaps / 50.0, 0.0, 1.0)
        
        # Bonus Hot
        bonus_presence = np.zeros((n_draws, BONUS_NUMBER_RANGE))
        for i, row in data.iterrows():
            for num in row['bonus_numbers']:
                if 1 <= num <= BONUS_NUMBER_RANGE:
                    bonus_presence[i, num-1] = 1
                    
        bonus_roll = pd.DataFrame(bonus_presence).rolling(window=window, min_periods=1).sum().shift(1).fillna(0).values
        bonus_hot = bonus_roll / window
        
        # Bonus Cold
        bonus_cold = np.clip(bonus_gaps / 20.0, 0.0, 1.0)
        
        main_features = np.hstack([main_hot, main_cold])
        bonus_features = np.hstack([bonus_hot, bonus_cold])
        
        return main_features, bonus_features

    def calculate_global_features(self, data: pd.DataFrame, window: int = 10) -> np.ndarray:
        """
        Calculate Global Pattern features (Sum, Odd/Even, High/Low, Decades).
        Returns array of shape (n_draws, 9). 
        Features:
        1. Rolling Avg Sum (normalized)
        2. Rolling Avg Odd Count (normalized)
        3. Rolling Avg High Count (normalized)
        4. Rolling Avg Low Count (normalized)
        5-9. Rolling Avg Decade Counts (1-10, 11-20, 21-30, 31-40, 41-50)
        """
        # 1. Sum
        sums = data['main_numbers'].apply(sum)
        # Normalize sum: Max sum is approx 50+49+48+47+46 = 240. Min is 1+2+3+4+5 = 15.
        sums_norm = sums / 250.0 
        
        # 2. Odd/Even
        odds = data['main_numbers'].apply(lambda x: sum(1 for n in x if n % 2 != 0))
        odds_norm = odds / 5.0
        
        # 3. High/Low (High > 25)
        highs = data['main_numbers'].apply(lambda x: sum(1 for n in x if n > 25))
        highs_norm = highs / 5.0
        lows = data['main_numbers'].apply(lambda x: sum(1 for n in x if n <= 25))
        lows_norm = lows / 5.0
        
        # 4. Decades
        def get_decades(nums):
            counts = [0] * 5
            for n in nums:
                if 1 <= n <= 10: counts[0] += 1
                elif 11 <= n <= 20: counts[1] += 1
                elif 21 <= n <= 30: counts[2] += 1
                elif 31 <= n <= 40: counts[3] += 1
                elif 41 <= n <= 50: counts[4] += 1
            return np.array(counts) / 5.0
            
        decades = np.vstack(data['main_numbers'].apply(get_decades).values)
        
        # Combine raw features
        raw_features = np.column_stack([sums_norm, odds_norm, highs_norm, lows_norm, decades])
        
        # Calculate Rolling Averages
        # We want the trend leading UP TO the current draw.
        # Shift 1 so row i contains average of i-window to i-1
        rolling_features = pd.DataFrame(raw_features).rolling(window=window, min_periods=1).mean().shift(1).fillna(0).values
        
        return rolling_features

    def calculate_gap_delta_features(self, data: pd.DataFrame, windows: List[int] = [10, 50]) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
        """
        Calculate deviations of current gap state from rolling averages over specified windows.
        Returns dictionaries mapping window -> delta array of shape (n_draws, main_range / bonus_range).
        """
        from .config import MAIN_NUMBER_RANGE, BONUS_NUMBER_RANGE
        main_gaps, bonus_gaps = self.calculate_gap_states(data, MAIN_NUMBER_RANGE, BONUS_NUMBER_RANGE)

        main_deltas = {}
        bonus_deltas = {}
        main_df = pd.DataFrame(main_gaps)
        bonus_df = pd.DataFrame(bonus_gaps)

        for w in windows:
            main_roll = main_df.rolling(window=w, min_periods=1).mean().shift(1).fillna(0).values
            bonus_roll = bonus_df.rolling(window=w, min_periods=1).mean().shift(1).fillna(0).values
            main_deltas[w] = main_gaps - main_roll
            bonus_deltas[w] = bonus_gaps - bonus_roll

        return main_deltas, bonus_deltas

    def build_feature_vector_for_next_draw(self, data: pd.DataFrame) -> np.ndarray:
        """
        Assemble the feature vector used by tree/XGBoost models for the next draw.
        Centralizes the feature construction to ensure consistent ordering and to avoid duplication.
        """
        from .config import MAIN_NUMBER_RANGE, BONUS_NUMBER_RANGE

        main_gaps, bonus_gaps = self.calculate_gap_states(data, MAIN_NUMBER_RANGE, BONUS_NUMBER_RANGE)
        freq_features = self.calculate_frequency_features(data)
        main_hot_cold, bonus_hot_cold = self.calculate_hot_cold_features(data)
        main_gap_delta, bonus_gap_delta = self.calculate_gap_delta_features(data)

        last_main_gap = main_gaps[-1]
        last_bonus_gap = bonus_gaps[-1]

        last_freqs = [v[-1] for v in freq_features.values()]
        last_main_hc = main_hot_cold[-1]
        last_bonus_hc = bonus_hot_cold[-1]

        # Affinity for a hypothetical next draw: append a dummy row to extend affinity arrays by one.
        last_date_val = data.iloc[-1]['date']
        # We need to increment date for the dummy row to get correct date features for next draw?
        # For now, we just use the last known date features or we could project.
        # Actually, for date features of the *next* draw, we should use the date of the next draw.
        # But we don't know it. Assuming weekly?
        # Let's just use the last draw's date features as a proxy or 0 if unknown.
        # Better: The model predicts based on *previous* sequence.
        # If we are predicting for a future draw, we might know the date.
        # But for simplicity, let's assume the "next" draw is roughly same time or just use last available.
        # Actually, `calculate_date_features` uses the date of the row.
        # If we want features for the *prediction* target, we usually don't have them unless we input them.
        # But here we are building features *from history* to predict next.
        # Wait, `build_feature_vector_for_next_draw` is used for XGBoost input.
        # XGBoost predicts `y` (numbers) given `X` (features from previous draws).
        # So `X` should contain info *available* before the draw.
        # Date of the draw *is* available (we know when we are playing).
        # So we should ideally project the date.
        # Let's add 7 days to the last date.
        next_date = last_date_val + pd.Timedelta(days=7) # Assumption: Weekly
        
        dummy_row = pd.DataFrame([{
            'date': next_date,
            'main_numbers': [],
            'bonus_numbers': [],
            'jackpot': '',
            'result': ''
        }])
        data_augmented = pd.concat([data, dummy_row], ignore_index=True)
        
        # Recalculate date features for the augmented data to get the next date's features
        date_features_aug = self.calculate_date_features(data_augmented)
        target_date_feat = date_features_aug[-1]
        
        main_aff_aug, bonus_aff_aug = self.calculate_cooccurrence_features(data_augmented)
        global_feats_aug = self.calculate_global_features(data_augmented)

        last_main_affinity = main_aff_aug[-1]
        last_bonus_affinity = bonus_aff_aug[-1]
        last_global = global_feats_aug[-1]

        return np.hstack([
            last_main_gap, last_bonus_gap,
            *[main_gap_delta[w][-1] for w in sorted(main_gap_delta.keys())],
            *[bonus_gap_delta[w][-1] for w in sorted(bonus_gap_delta.keys())],
            *last_freqs, 
            target_date_feat,
            last_main_hc, last_bonus_hc,
            last_main_affinity, last_bonus_affinity,
            last_global
        ])
