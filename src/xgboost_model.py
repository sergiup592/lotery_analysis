import xgboost as xgb
import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Tuple
from .config import N_MAIN, N_BONUS, MAIN_NUMBER_RANGE, BONUS_NUMBER_RANGE, SAMPLING_CONFIG
from .data import LotteryDataManager
from .acceleration import get_xgboost_device_params

logger = logging.getLogger(__name__)

class XGBoostModel:
    """XGBoost Model for Lottery Prediction."""
    
    def __init__(self):
        self.main_models = [] # One model per number (binary classification) or one multi-output?
        # XGBoost doesn't natively support multi-label output in the same way as RF.
        # Standard approach: One binary classifier per number (One-vs-Rest).
        # Or use 'multi:softprob' if we treat it as one multiclass problem (but we pick 5 numbers).
        # Better: Binary classification for each number 1..50.
        self.bonus_models = []
        self.is_trained = False
        self.sampling = SAMPLING_CONFIG
        self.device_params = get_xgboost_device_params()
        
    def train(self, data: pd.DataFrame):
        """Train the XGBoost models."""
        logger.info("Training XGBoost Model...")
        
        X, y_main, y_bonus = self._prepare_data(data)
        if len(X) == 0:
            raise ValueError("Not enough data to train XGBoostModel.")

        # Simple time-based split for early stopping (last 10% for validation)
        split_idx = max(1, int(len(X) * 0.9))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_main_train, y_main_val = y_main[:split_idx], y_main[split_idx:]
        y_bonus_train, y_bonus_val = y_bonus[:split_idx], y_bonus[split_idx:]
        
        # Train Main Number Models
        self.main_models = []
        for i in range(MAIN_NUMBER_RANGE):
            # Target: Did number i+1 appear?
            y = y_main_train[:, i]
            pos = float(np.sum(y))
            neg = float(len(y) - pos)
            scale_pos_weight = (neg / pos) if pos > 0 else 1.0
            scale_pos_weight = float(np.clip(scale_pos_weight, 1.0, 50.0))
            
            model = xgb.XGBClassifier(
                n_estimators=800,
                max_depth=8,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='binary:logistic',
                scale_pos_weight=scale_pos_weight,
                n_jobs=-1,
                eval_metric='logloss',
                random_state=42,
                **self.device_params
            )
            model.fit(
                X_train,
                y,
                eval_set=[(X_val, y_main_val[:, i])],
                verbose=False
            )
            self.main_models.append(model)
            
        # Train Bonus Number Models
        self.bonus_models = []
        for i in range(BONUS_NUMBER_RANGE):
            y = y_bonus_train[:, i]
            pos = float(np.sum(y))
            neg = float(len(y) - pos)
            scale_pos_weight = (neg / pos) if pos > 0 else 1.0
            scale_pos_weight = float(np.clip(scale_pos_weight, 1.0, 50.0))
            
            model = xgb.XGBClassifier(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.03,
                subsample=0.85,
                colsample_bytree=0.9,
                objective='binary:logistic',
                scale_pos_weight=scale_pos_weight,
                n_jobs=-1,
                eval_metric='logloss',
                random_state=42,
                **self.device_params
            )
            model.fit(
                X_train,
                y,
                eval_set=[(X_val, y_bonus_val[:, i])],
                verbose=False
            )
            self.bonus_models.append(model)
            
        self.is_trained = True
        logger.info("XGBoost training complete.")
        
    def _apply_sampling_bias(self, probs: np.ndarray, top_k: int, temperature: float,
                               top_p: float = None, required_nonzero: int = None) -> np.ndarray:
        """
        Reweight probabilities using proper temperature scaling and nucleus sampling.
        Temperature is applied to logits for proper Boltzmann distribution.
        """
        probs = np.array(probs, dtype=np.float64)
        probs = np.clip(probs, 1e-10, 1.0 - 1e-10)

        # Convert to logits for proper temperature scaling
        logits = np.log(probs / (1 - probs))

        if temperature and temperature > 0:
            logits = logits / temperature

        # Convert back via softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)

        effective_top_k = top_k if top_k else probs.size
        if required_nonzero is not None:
            effective_top_k = max(effective_top_k, required_nonzero)

        # Nucleus (top-p) sampling
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

    def predict(self, data: pd.DataFrame, num_predictions: int = 5) -> List[Dict]:
        """Generate predictions using the trained model."""
        if not self.is_trained:
            logger.warning("XGBoost model not trained. Training now...")
            self.train(data)
            
        # Prepare input for the *next* draw
        dm = LotteryDataManager()
        X_next = dm.build_feature_vector_for_next_draw(data).reshape(1, -1)
        
        # Predict Probabilities
        main_probs = np.zeros(MAIN_NUMBER_RANGE)
        for i, model in enumerate(self.main_models):
            main_probs[i] = model.predict_proba(X_next)[0][1]
            
        bonus_probs = np.zeros(BONUS_NUMBER_RANGE)
        for i, model in enumerate(self.bonus_models):
            bonus_probs[i] = model.predict_proba(X_next)[0][1]
            
        # Reweight distribution to emphasize top candidates
        main_probs = self._apply_sampling_bias(
            main_probs,
            top_k=self.sampling.get("top_k_main"),
            temperature=self.sampling.get("temperature_main"),
            top_p=self.sampling.get("top_p_main"),
            required_nonzero=N_MAIN
        )
        bonus_probs = self._apply_sampling_bias(
            bonus_probs,
            top_k=self.sampling.get("top_k_bonus"),
            temperature=self.sampling.get("temperature_bonus"),
            top_p=self.sampling.get("top_p_bonus"),
            required_nonzero=N_BONUS
        )
        main_prob_vector = main_probs.copy()
        bonus_prob_vector = bonus_probs.copy()
        
        predictions = []
        for _ in range(num_predictions):
            # Sample Main
            main_nums = np.random.choice(
                np.arange(1, MAIN_NUMBER_RANGE + 1),
                size=N_MAIN,
                replace=False,
                p=main_probs
            )
            
            # Sample Bonus
            bonus_nums = np.random.choice(
                np.arange(1, BONUS_NUMBER_RANGE + 1),
                size=N_BONUS,
                replace=False,
                p=bonus_probs
            )
            
            # Confidence
            main_conf = np.mean([main_probs[n-1] for n in main_nums])
            bonus_conf = np.mean([bonus_probs[n-1] for n in bonus_nums])
            
            predictions.append({
                'main_numbers': sorted(main_nums.tolist()),
                'bonus_numbers': sorted(bonus_nums.tolist()),
                'confidence': float((main_conf + bonus_conf) / 2),
                'main_prob_vector': main_prob_vector.tolist(),
                'bonus_prob_vector': bonus_prob_vector.tolist(),
                'expected_main_prob': float(np.sum(main_prob_vector[main_nums-1])),
                'expected_bonus_prob': float(np.sum(bonus_prob_vector[bonus_nums-1])),
                'source': 'XGBoost'
            })
            
        return predictions

    def _prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare features (X) and targets (y) for training with enhanced features."""
        dm = LotteryDataManager()
        main_gaps, bonus_gaps = dm.calculate_gap_states(data, MAIN_NUMBER_RANGE, BONUS_NUMBER_RANGE)
        main_gap_delta, bonus_gap_delta = dm.calculate_gap_delta_features(data)
        freq_features = dm.calculate_frequency_features(data)
        date_features = dm.calculate_date_features(data)
        main_hot_cold, bonus_hot_cold = dm.calculate_hot_cold_features(data)
        main_affinity, bonus_affinity = dm.calculate_cooccurrence_features(data)
        global_features = dm.calculate_global_features(data)

        # New advanced features
        main_var_ent, bonus_var_ent = dm.calculate_variance_entropy_features(data)
        main_momentum, bonus_momentum = dm.calculate_momentum_features(data)
        spread_features = dm.calculate_spread_features(data)

        X = []
        y_main = []
        y_bonus = []

        # Start from index 100 to ensure we have history for frequency windows (max window 100)
        start_idx = 100
        if len(data) <= start_idx:
            start_idx = 1

        for i in range(start_idx, len(data)):
            # Features - Order must match build_feature_vector_for_next_draw:
            # Gaps, Gap Deltas, Freqs, Date, Hot/Cold, Affinity, Global, Var/Ent, Momentum, Spread

            row_features = [main_gaps[i], bonus_gaps[i]]
            for w in sorted(main_gap_delta.keys()):
                row_features.append(main_gap_delta[w][i])
            for w in sorted(bonus_gap_delta.keys()):
                row_features.append(bonus_gap_delta[w][i])

            # CRITICAL: Use sorted keys to ensure deterministic ordering (matches build_feature_vector_for_next_draw)
            for k in sorted(freq_features.keys()):
                row_features.append(freq_features[k][i])

            row_features.append(date_features[i])
            row_features.append(main_hot_cold[i])
            row_features.append(bonus_hot_cold[i])

            # Add Affinity Features
            row_features.append(main_affinity[i])
            row_features.append(bonus_affinity[i])

            # Add Global Features
            row_features.append(global_features[i])

            # Add New Advanced Features
            row_features.append(main_var_ent[i])
            row_features.append(bonus_var_ent[i])
            row_features.append(main_momentum[i])
            row_features.append(bonus_momentum[i])
            row_features.append(spread_features[i])

            X.append(np.hstack(row_features))

            # Targets
            row = data.iloc[i]

            main_target = np.zeros(MAIN_NUMBER_RANGE)
            for num in row['main_numbers']:
                if 1 <= num <= MAIN_NUMBER_RANGE:
                    main_target[num-1] = 1
            y_main.append(main_target)

            bonus_target = np.zeros(BONUS_NUMBER_RANGE)
            for num in row['bonus_numbers']:
                if 1 <= num <= BONUS_NUMBER_RANGE:
                    bonus_target[num-1] = 1
            y_bonus.append(bonus_target)

        return np.array(X), np.array(y_main), np.array(y_bonus)
