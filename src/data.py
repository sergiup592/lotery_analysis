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
        Calculate Co-occurrence (Affinity) features WITHOUT data leakage.
        For each draw t, calculates a vector based ONLY on historical data (draws 0 to t-1).

        This prevents the model from "seeing" future co-occurrence patterns during training.

        Returns:
            main_affinity: (n_draws, MAIN_NUMBER_RANGE)
            bonus_affinity: (n_draws, BONUS_NUMBER_RANGE)
        """
        from .config import MAIN_NUMBER_RANGE, BONUS_NUMBER_RANGE

        n_draws = len(data)
        main_affinity = np.zeros((n_draws, MAIN_NUMBER_RANGE))
        bonus_affinity = np.zeros((n_draws, BONUS_NUMBER_RANGE))

        # Build co-occurrence incrementally to avoid data leakage
        main_cooc = np.zeros((MAIN_NUMBER_RANGE, MAIN_NUMBER_RANGE))
        bonus_cooc = np.zeros((BONUS_NUMBER_RANGE, BONUS_NUMBER_RANGE))

        for i in range(n_draws):
            if i > 0:
                # Calculate affinity BEFORE updating with current draw
                # Normalize current co-occurrence matrix
                main_row_sums = np.sum(main_cooc, axis=1, keepdims=True) + 1e-9
                main_cooc_norm = main_cooc / main_row_sums

                bonus_row_sums = np.sum(bonus_cooc, axis=1, keepdims=True) + 1e-9
                bonus_cooc_norm = bonus_cooc / bonus_row_sums

                # Use previous draw's numbers to calculate affinity
                prev_row = data.iloc[i-1]

                prev_main = [n for n in prev_row['main_numbers'] if 1 <= n <= MAIN_NUMBER_RANGE]
                if prev_main:
                    for c in range(MAIN_NUMBER_RANGE):
                        score = sum(main_cooc_norm[p-1, c] for p in prev_main)
                        main_affinity[i, c] = score

                prev_bonus = [n for n in prev_row['bonus_numbers'] if 1 <= n <= BONUS_NUMBER_RANGE]
                if prev_bonus:
                    for c in range(BONUS_NUMBER_RANGE):
                        score = sum(bonus_cooc_norm[p-1, c] for p in prev_bonus)
                        bonus_affinity[i, c] = score

            # NOW update co-occurrence with current draw (for future rows)
            row = data.iloc[i]
            main_nums = [n for n in row['main_numbers'] if 1 <= n <= MAIN_NUMBER_RANGE]
            for n1 in main_nums:
                for n2 in main_nums:
                    if n1 != n2:
                        main_cooc[n1-1, n2-1] += 1

            bonus_nums = [n for n in row['bonus_numbers'] if 1 <= n <= BONUS_NUMBER_RANGE]
            for n1 in bonus_nums:
                for n2 in bonus_nums:
                    if n1 != n2:
                        bonus_cooc[n1-1, n2-1] += 1

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

    def calculate_gap_delta_features(self, data: pd.DataFrame, windows: List[int] = [10, 30, 50]) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
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

    def calculate_variance_entropy_features(self, data: pd.DataFrame, window: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate variance and entropy of number distributions in recent draws.
        Higher entropy/variance = more random/diverse patterns.
        Lower = more concentrated/predictable patterns.

        Returns:
            main_features: (n_draws, 2) [variance, entropy]
            bonus_features: (n_draws, 2) [variance, entropy]
        """
        from .config import MAIN_NUMBER_RANGE, BONUS_NUMBER_RANGE
        from scipy.stats import entropy

        n_draws = len(data)
        main_features = np.zeros((n_draws, 2))
        bonus_features = np.zeros((n_draws, 2))

        for i in range(window, n_draws):
            # Get recent window
            recent_data = data.iloc[max(0, i-window):i]

            # Main numbers analysis
            main_counts = np.zeros(MAIN_NUMBER_RANGE)
            for _, row in recent_data.iterrows():
                for num in row['main_numbers']:
                    if 1 <= num <= MAIN_NUMBER_RANGE:
                        main_counts[num-1] += 1

            # Normalize to probability distribution
            main_probs = main_counts / (main_counts.sum() + 1e-9)
            main_features[i, 0] = np.var(main_counts)  # Variance
            main_features[i, 1] = entropy(main_probs + 1e-9)  # Shannon entropy

            # Bonus numbers analysis
            bonus_counts = np.zeros(BONUS_NUMBER_RANGE)
            for _, row in recent_data.iterrows():
                for num in row['bonus_numbers']:
                    if 1 <= num <= BONUS_NUMBER_RANGE:
                        bonus_counts[num-1] += 1

            bonus_probs = bonus_counts / (bonus_counts.sum() + 1e-9)
            bonus_features[i, 0] = np.var(bonus_counts)
            bonus_features[i, 1] = entropy(bonus_probs + 1e-9)

        return main_features, bonus_features

    def calculate_momentum_features(self, data: pd.DataFrame, windows: List[int] = [10, 30]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate rate of change (momentum) in number frequencies.
        Positive momentum = increasing frequency, Negative = decreasing frequency.

        Returns:
            main_momentum: (n_draws, MAIN_NUMBER_RANGE)
            bonus_momentum: (n_draws, BONUS_NUMBER_RANGE)
        """
        from .config import MAIN_NUMBER_RANGE, BONUS_NUMBER_RANGE

        n_draws = len(data)
        main_momentum = np.zeros((n_draws, MAIN_NUMBER_RANGE))
        bonus_momentum = np.zeros((n_draws, BONUS_NUMBER_RANGE))

        # Calculate frequency at two time points and find the difference (momentum)
        short_window = windows[0]
        long_window = windows[1]

        for i in range(long_window, n_draws):
            # Short-term frequency (recent)
            recent_main = np.zeros(MAIN_NUMBER_RANGE)
            recent_bonus = np.zeros(BONUS_NUMBER_RANGE)
            for j in range(i - short_window, i):
                for num in data.iloc[j]['main_numbers']:
                    if 1 <= num <= MAIN_NUMBER_RANGE:
                        recent_main[num-1] += 1
                for num in data.iloc[j]['bonus_numbers']:
                    if 1 <= num <= BONUS_NUMBER_RANGE:
                        recent_bonus[num-1] += 1

            # Long-term frequency (historical)
            historical_main = np.zeros(MAIN_NUMBER_RANGE)
            historical_bonus = np.zeros(BONUS_NUMBER_RANGE)
            for j in range(i - long_window, i - short_window):
                for num in data.iloc[j]['main_numbers']:
                    if 1 <= num <= MAIN_NUMBER_RANGE:
                        historical_main[num-1] += 1
                for num in data.iloc[j]['bonus_numbers']:
                    if 1 <= num <= BONUS_NUMBER_RANGE:
                        historical_bonus[num-1] += 1

            # Normalize and calculate momentum (difference)
            recent_main_norm = recent_main / (short_window + 1e-9)
            historical_main_norm = historical_main / ((long_window - short_window) + 1e-9)
            main_momentum[i] = recent_main_norm - historical_main_norm

            recent_bonus_norm = recent_bonus / (short_window + 1e-9)
            historical_bonus_norm = historical_bonus / ((long_window - short_window) + 1e-9)
            bonus_momentum[i] = recent_bonus_norm - historical_bonus_norm

        return main_momentum, bonus_momentum

    def calculate_spread_features(self, data: pd.DataFrame, window: int = 10) -> np.ndarray:
        """
        Calculate distribution spread metrics for drawn numbers.
        Features: range, std dev, coefficient of variation.

        Returns:
            spread_features: (n_draws, 3) [range, std_dev, cv]
        """
        n_draws = len(data)
        spread_features = np.zeros((n_draws, 3))

        for i in range(n_draws):
            nums = data.iloc[i]['main_numbers']
            if len(nums) > 0:
                spread_features[i, 0] = max(nums) - min(nums)  # Range
                spread_features[i, 1] = np.std(nums)  # Standard deviation
                mean_val = np.mean(nums)
                spread_features[i, 2] = spread_features[i, 1] / (mean_val + 1e-9)  # CV

        # Calculate rolling averages
        spread_df = pd.DataFrame(spread_features)
        rolling_spread = spread_df.rolling(window=window, min_periods=1).mean().shift(1).fillna(0).values

        return rolling_spread

    def infer_next_draw_date(self, data: pd.DataFrame) -> pd.Timestamp:
        """
        Infer the next draw date based on the most common draw weekdays in the data.
        Falls back to +7 days if inference is not possible.
        """
        if data.empty:
            raise ValueError("Cannot infer next draw date from empty data.")

        last_date = data.iloc[-1]['date']
        weekday_counts = data['date'].dt.dayofweek.value_counts()
        if weekday_counts.empty:
            return last_date + pd.Timedelta(days=7)

        # Use top 2 draw weekdays if present (e.g., Tue/Fri). Otherwise use the single most common weekday.
        allowed_days = list(weekday_counts.index[:2])
        next_date = last_date + pd.Timedelta(days=1)
        guard = 0
        while next_date.dayofweek not in allowed_days:
            next_date += pd.Timedelta(days=1)
            guard += 1
            if guard > 14:
                return last_date + pd.Timedelta(days=7)
        return next_date

    def build_feature_vector_for_next_draw(self, data: pd.DataFrame) -> np.ndarray:
        """
        Assemble the feature vector used by tree/XGBoost models for the next draw.
        Centralizes the feature construction to ensure consistent ordering and to avoid duplication.
        Includes advanced features: variance/entropy, momentum, and spread metrics.
        """
        from .config import MAIN_NUMBER_RANGE, BONUS_NUMBER_RANGE

        # 1. Augment data with dummy row for next draw
        # We must do this FIRST so that rolling window calculations (which shift by 1)
        # correctly produce values for this new row based on the actual history.
        next_date = self.infer_next_draw_date(data)

        dummy_row = pd.DataFrame([{
            'date': next_date,
            'main_numbers': [],  # Empty lists for numbers
            'bonus_numbers': [],
            'jackpot': '',
            'result': ''
        }])
        data_augmented = pd.concat([data, dummy_row], ignore_index=True)

        # 2. Calculate features on augmented data
        # Note: All feature identifiers (gaps, freq, etc.) will calculate values
        # for all rows. We only care about the values for the LAST row (the dummy one).
        
        # Gaps
        main_gaps, bonus_gaps = self.calculate_gap_states(data_augmented, MAIN_NUMBER_RANGE, BONUS_NUMBER_RANGE)
        
        # Frequency
        freq_features = self.calculate_frequency_features(data_augmented)
        
        # Hot/Cold
        main_hot_cold, bonus_hot_cold = self.calculate_hot_cold_features(data_augmented)
        
        # Gap Deltas
        main_gap_delta, bonus_gap_delta = self.calculate_gap_delta_features(data_augmented)

        # Advanced Features
        main_var_ent, bonus_var_ent = self.calculate_variance_entropy_features(data_augmented)
        main_momentum, bonus_momentum = self.calculate_momentum_features(data_augmented)
        spread_features = self.calculate_spread_features(data_augmented)
        
        # Co-occurrence & Global
        main_affinity, bonus_affinity = self.calculate_cooccurrence_features(data_augmented)
        global_features = self.calculate_global_features(data_augmented)
        
        # Date Features
        date_features = self.calculate_date_features(data_augmented)

        # 3. Extract features for the last row (index -1)
        last_main_gap = main_gaps[-1]
        last_bonus_gap = bonus_gaps[-1]
        
        # Unpack dicts in sorted order of keys
        last_gap_deltas_main = [main_gap_delta[w][-1] for w in sorted(main_gap_delta.keys())]
        last_gap_deltas_bonus = [bonus_gap_delta[w][-1] for w in sorted(bonus_gap_delta.keys())]
        
        # CRITICAL: Use sorted keys to ensure deterministic ordering between training and inference.
        # This prevents feature misalignment bugs caused by dictionary iteration order.
        sorted_freq_keys = sorted(freq_features.keys())
        last_freqs = [freq_features[k][-1] for k in sorted_freq_keys]
        
        last_main_hc = main_hot_cold[-1]
        last_bonus_hc = bonus_hot_cold[-1]
        
        last_main_aff = main_affinity[-1]
        last_bonus_aff = bonus_affinity[-1]
        
        last_global = global_features[-1]
        
        last_main_ve = main_var_ent[-1]
        last_bonus_ve = bonus_var_ent[-1]
        
        last_main_mom = main_momentum[-1]
        last_bonus_mom = bonus_momentum[-1]
        
        last_spread = spread_features[-1]
        
        last_date = date_features[-1]

        # 4. Stack into single vector
        # ORDER MUST MATCH `src/tree.py` _prepare_data
        # [gaps, gap_deltas(sorted keys), freqs(sorted keys), date, hot_cold, affinity, global, var_ent, momentum, spread]
        
        feature_vector = np.hstack([
            last_main_gap, last_bonus_gap,
            *last_gap_deltas_main,
            *last_gap_deltas_bonus,
            *last_freqs,
            last_date,
            last_main_hc, last_bonus_hc,
            last_main_aff, last_bonus_aff,
            last_global,
            last_main_ve, last_bonus_ve,
            last_main_mom, last_bonus_mom,
            last_spread
        ])
        
        # Runtime validation: log feature vector size for debugging
        # Expected size breakdown (approximate, depends on config):
        # main_gap: 50, bonus_gap: 12, gap_deltas: 3*50 + 3*12 = 186, freqs: 6 windows * (50+12) = 372
        # date: 19, hot_cold: 2*(50+12) = 124, affinity: 50+12 = 62, global: 9
        # var_ent: 2+2 = 4, momentum: 50+12 = 62, spread: 3
        logger.debug(f"Feature vector size for next draw: {len(feature_vector)}")
        
        return feature_vector
