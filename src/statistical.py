import numpy as np
import pandas as pd
import logging
from typing import List, Dict
from .config import (
    STATISTICAL_CONFIG,
    N_MAIN,
    N_BONUS,
    MAIN_NUMBER_RANGE,
    BONUS_NUMBER_RANGE,
    SAMPLING_CONFIG,
)

logger = logging.getLogger(__name__)

class StatisticalModel:
    """Robust Statistical Analysis for Lottery Prediction."""
    
    def __init__(self):
        self.main_freq = None
        self.bonus_freq = None
        self.main_recency = None
        self.bonus_recency = None
        self.params = STATISTICAL_CONFIG
        self.sampling = SAMPLING_CONFIG
        
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
                    total_picks += 1  # Only count valid numbers
                    
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

        # Normalize recency scores (higher means more due).
        main_recency_score = self.main_recency / (np.max(self.main_recency) + 1)
        bonus_recency_score = self.bonus_recency / (np.max(self.bonus_recency) + 1)

        # Mix hot and due signals. Normalize if custom weights don't sum to 1.
        freq_weight = float(self.params.get("frequency_weight", 0.6))
        recency_weight = float(self.params.get("recency_weight", 0.4))
        mix_total = max(1e-9, freq_weight + recency_weight)
        freq_weight /= mix_total
        recency_weight /= mix_total

        main_weights = (self.main_freq * freq_weight) + (main_recency_score * recency_weight)
        bonus_weights = (self.bonus_freq * freq_weight) + (bonus_recency_score * recency_weight)

        # Sharpen distributions so candidate ranking has clearer signal.
        main_prob_vector = self._apply_sampling_bias(
            main_weights,
            top_k=self.sampling.get("top_k_main"),
            temperature=self.sampling.get("temperature_main"),
            top_p=self.sampling.get("top_p_main"),
            required_nonzero=N_MAIN,
        )
        bonus_prob_vector = self._apply_sampling_bias(
            bonus_weights,
            top_k=self.sampling.get("top_k_bonus"),
            temperature=self.sampling.get("temperature_bonus"),
            top_p=self.sampling.get("top_p_bonus"),
            required_nonzero=N_BONUS,
        )

        # Build deterministic elite candidates first, then stochastic tail for diversity.
        main_order = np.argsort(main_prob_vector)[::-1]
        bonus_order = np.argsort(bonus_prob_vector)[::-1]
        elite_limit = max(2, min(num_predictions, int(self.params.get("elite_candidates", 12))))
        main_step = max(1, int(self.params.get("elite_main_step", 3)))
        bonus_step = max(1, int(self.params.get("elite_bonus_step", 1)))

        predictions = []
        seen = set()
        candidate_budget = max(num_predictions * 4, elite_limit)

        for _ in range(candidate_budget):
            if len(predictions) >= num_predictions:
                break

            idx = len(predictions)
            if idx < elite_limit:
                main_shift = (idx * main_step) % MAIN_NUMBER_RANGE
                bonus_shift = (idx * bonus_step) % BONUS_NUMBER_RANGE
                main_idx = np.take(main_order, np.arange(main_shift, main_shift + N_MAIN), mode="wrap")
                bonus_idx = np.take(bonus_order, np.arange(bonus_shift, bonus_shift + N_BONUS), mode="wrap")
            else:
                main_idx = np.random.choice(
                    np.arange(MAIN_NUMBER_RANGE),
                    size=N_MAIN,
                    replace=False,
                    p=main_prob_vector,
                )
                bonus_idx = np.random.choice(
                    np.arange(BONUS_NUMBER_RANGE),
                    size=N_BONUS,
                    replace=False,
                    p=bonus_prob_vector,
                )

            main_nums = sorted((main_idx + 1).tolist())
            bonus_nums = sorted((bonus_idx + 1).tolist())
            combo_key = tuple(main_nums + bonus_nums)
            if combo_key in seen:
                continue
            seen.add(combo_key)

            main_conf = float(np.mean(main_prob_vector[main_idx]))
            bonus_conf = float(np.mean(bonus_prob_vector[bonus_idx]))
            confidence = (main_conf + bonus_conf) / 2.0

            # Keep elite candidates slightly prioritized in downstream ranking.
            if idx < elite_limit:
                confidence *= 1.05

            predictions.append({
                "main_numbers": main_nums,
                "bonus_numbers": bonus_nums,
                "confidence": float(confidence),
                "main_prob_vector": main_prob_vector.tolist(),
                "bonus_prob_vector": bonus_prob_vector.tolist(),
                "expected_main_prob": float(np.sum(main_prob_vector[main_idx])),
                "expected_bonus_prob": float(np.sum(bonus_prob_vector[bonus_idx])),
                "source": "Statistical",
            })

        return predictions

    def _apply_sampling_bias(
        self,
        probs: np.ndarray,
        top_k: int,
        temperature: float,
        top_p: float = None,
        required_nonzero: int = None,
    ) -> np.ndarray:
        """Apply temperature and top-k/top-p filtering to a probability vector."""
        probs = np.array(probs, dtype=np.float64)
        probs = np.maximum(probs, 1e-10)
        probs /= np.sum(probs)
        probs = np.clip(probs, 1e-10, 1.0 - 1e-10)

        logits = np.log(probs / (1 - probs))
        if temperature and temperature > 0:
            logits = logits / temperature

        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)

        effective_top_k = top_k if top_k else probs.size
        if required_nonzero is not None:
            effective_top_k = max(effective_top_k, required_nonzero)

        if top_p is not None and 0 < top_p < 1:
            sorted_indices = np.argsort(probs)[::-1]
            cumsum = np.cumsum(probs[sorted_indices])
            cutoff_idx = np.searchsorted(cumsum, top_p) + 1
            cutoff_idx = max(cutoff_idx, required_nonzero or 1)
            nucleus_indices = sorted_indices[:cutoff_idx]
            mask = np.zeros_like(probs)
            mask[nucleus_indices] = 1.0
            probs = probs * mask
        elif effective_top_k and effective_top_k < probs.size:
            top_indices = np.argpartition(probs, -effective_top_k)[-effective_top_k:]
            mask = np.zeros_like(probs)
            mask[top_indices] = 1.0
            probs = probs * mask

        total = probs.sum()
        if total <= 0:
            probs = np.ones_like(probs)
            total = probs.sum()
        if required_nonzero is not None and np.count_nonzero(probs) < required_nonzero:
            probs = np.ones_like(probs)
            total = probs.sum()

        return probs / total
