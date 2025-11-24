import numpy as np
import pandas as pd
import logging
from typing import List, Dict
from .config import STATISTICAL_CONFIG, N_MAIN, N_BONUS, MAIN_NUMBER_RANGE, BONUS_NUMBER_RANGE

logger = logging.getLogger(__name__)

class StatisticalModel:
    """Robust Statistical Analysis for Lottery Prediction."""
    
    def __init__(self):
        self.main_freq = None
        self.bonus_freq = None
        self.main_recency = None
        self.bonus_recency = None
        self.params = STATISTICAL_CONFIG
        
    def train(self, data: pd.DataFrame):
        """Analyze historical data to build statistical models."""
        logger.info("Analyzing statistical patterns...")
        if data is None or data.empty:
            raise ValueError("No data provided for statistical analysis.")
        
        # 1. Frequency Analysis (Hot Numbers)
        # Use defined window to capture recent trends
        window = self.params.get('hot_cold_window', 50)
        
        recent_main = data['main_numbers'].tail(window)
        recent_bonus = data['bonus_numbers'].tail(window)
        
        self.main_freq = self._calculate_frequency(recent_main, MAIN_NUMBER_RANGE)
        self.bonus_freq = self._calculate_frequency(recent_bonus, BONUS_NUMBER_RANGE)
        
        # 2. Recency Analysis
        self.main_recency = self._calculate_recency(data['main_numbers'], MAIN_NUMBER_RANGE)
        self.bonus_recency = self._calculate_recency(data['bonus_numbers'], BONUS_NUMBER_RANGE)
        
        logger.info("Statistical analysis complete.")

    def _calculate_frequency(self, number_series: pd.Series, number_range: int) -> np.ndarray:
        """Calculate normalized frequency for each number."""
        counts = np.zeros(number_range)
        total_picks = 0
        
        for numbers in number_series:
            for num in numbers:
                if 1 <= num <= number_range:
                    counts[num-1] += 1
                total_picks += 1
                    
        if total_picks == 0:
            return counts
            
        return counts / total_picks

    def _calculate_recency(self, number_series: pd.Series, number_range: int) -> np.ndarray:
        """Calculate days since last drawn for each number."""
        last_seen = np.full(number_range, -1)
        total_draws = len(number_series)
        
        # Iterate backwards
        for i in range(total_draws - 1, -1, -1):
            numbers = number_series.iloc[i]
            days_ago = total_draws - 1 - i
            
            for num in numbers:
                if 1 <= num <= number_range:
                    if last_seen[num-1] == -1:
                        last_seen[num-1] = days_ago
                        
        # Handle numbers never seen (set to max)
        last_seen[last_seen == -1] = total_draws
        return last_seen

    def predict(self, recent_data: pd.DataFrame, num_predictions: int = 5) -> List[Dict]:
        """Generate predictions based on statistical weights."""
        if self.main_freq is None:
            raise ValueError("Model not trained. Call train() first.")
        if recent_data is None or recent_data.empty:
            raise ValueError("No recent data provided for prediction.")
            
        predictions = []
        for _ in range(num_predictions):
            # Calculate weights combining Frequency (Hot) and Recency (Due)
            # We want a mix of Hot numbers and Due numbers
            
            # Normalize recency (higher is more due)
            main_recency_score = self.main_recency / (np.max(self.main_recency) + 1)
            bonus_recency_score = self.bonus_recency / (np.max(self.bonus_recency) + 1)
            
            # Combine scores (0.6 Freq, 0.4 Recency)
            main_weights = (self.main_freq * 0.6) + (main_recency_score * 0.4)
            bonus_weights = (self.bonus_freq * 0.6) + (bonus_recency_score * 0.4)
            
            # Add some noise for variety
            main_weights += np.random.normal(0, 0.05, size=len(main_weights))
            bonus_weights += np.random.normal(0, 0.05, size=len(bonus_weights))
            
            # Ensure non-negative
            main_weights = np.maximum(main_weights, 0.001)
            bonus_weights = np.maximum(bonus_weights, 0.001)
            
            # Normalize to sum to 1
            main_weights /= np.sum(main_weights)
            bonus_weights /= np.sum(bonus_weights)
            # Keep a copy so we can expose the probability mass to downstream ranking
            main_prob_vector = main_weights.copy()
            bonus_prob_vector = bonus_weights.copy()
            
            # Sample
            main_nums = np.random.choice(
                np.arange(1, MAIN_NUMBER_RANGE + 1), 
                size=N_MAIN, 
                replace=False, 
                p=main_weights
            )
            
            bonus_nums = np.random.choice(
                np.arange(1, BONUS_NUMBER_RANGE + 1), 
                size=N_BONUS, 
                replace=False, 
                p=bonus_weights
            )
            
            # Calculate confidence score based on historical frequency of selected numbers
            main_conf = np.mean(main_prob_vector[main_nums-1])
            bonus_conf = np.mean(bonus_prob_vector[bonus_nums-1])
            
            predictions.append({
                'main_numbers': sorted(main_nums.tolist()),
                'bonus_numbers': sorted(bonus_nums.tolist()),
                'confidence': float((main_conf + bonus_conf) / 2),
                'main_prob_vector': main_prob_vector.tolist(),
                'bonus_prob_vector': bonus_prob_vector.tolist(),
                'expected_main_prob': float(np.sum(main_prob_vector[main_nums-1])),
                'expected_bonus_prob': float(np.sum(bonus_prob_vector[bonus_nums-1])),
                'source': 'Statistical'
            })
            
        return predictions
