import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D, Embedding, Add
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import numpy as np
import logging
import pandas as pd
from typing import Dict, Tuple, List
from .acceleration import configure_tensorflow
from .config import (
    NEURAL_MODEL_PARAMS,
    NEURAL_ARCH_CONFIG,
    MODELS_DIR,
    N_MAIN,
    N_BONUS,
    MAIN_NUMBER_RANGE,
    BONUS_NUMBER_RANGE,
    SAMPLING_CONFIG,
)

logger = logging.getLogger(__name__)

class NeuralModel:
    """Transformer-based Neural Network with PPO Fine-tuning for Lottery Prediction."""
    
    def __init__(self):
        self.main_model = None
        self.bonus_model = None
        # Critic models for PPO (Value function)
        self.main_critic = None
        self.bonus_critic = None
        
        self.models_loaded = False
        self.params = NEURAL_MODEL_PARAMS
        self.arch = NEURAL_ARCH_CONFIG
        self.sampling = SAMPLING_CONFIG
        
    def build_models(self):
        """Build or load the neural network models."""
        configure_tensorflow()
        main_model_path = MODELS_DIR / "main_transformer.keras"
        bonus_model_path = MODELS_DIR / "bonus_transformer.keras"

        # Calculate input dimensions with enhanced features
        # Components: gaps + gap_deltas(3 windows) + freqs(3) + date(19) + hot/cold(2*range) + affinity(range) + global(9) + var/ent(2) + momentum(range) + spread(3)
        global_feature_dim = 9
        var_ent_dim = 2
        spread_dim = 3
        gap_delta_windows = 3  # windows [10, 30, 50]
        self.main_input_dim = (
            MAIN_NUMBER_RANGE +                          # gaps
            (gap_delta_windows * MAIN_NUMBER_RANGE) +    # gap deltas
            (3 * MAIN_NUMBER_RANGE) +                    # freqs
            19 +                                         # date
            (2 * MAIN_NUMBER_RANGE) +                    # hot/cold
            MAIN_NUMBER_RANGE +                          # affinity
            global_feature_dim +                         # global
            var_ent_dim +                                # var/entropy
            MAIN_NUMBER_RANGE +                          # momentum
            spread_dim                                   # spread
        )
        self.bonus_input_dim = (
            BONUS_NUMBER_RANGE +
            (gap_delta_windows * BONUS_NUMBER_RANGE) +
            (3 * BONUS_NUMBER_RANGE) +
            19 +
            (2 * BONUS_NUMBER_RANGE) +
            BONUS_NUMBER_RANGE +
            global_feature_dim +
            var_ent_dim +
            BONUS_NUMBER_RANGE +
            spread_dim
        )

        if main_model_path.exists() and bonus_model_path.exists():
            try:
                logger.info("Loading existing models...")
                self.main_model = load_model(main_model_path)
                self.bonus_model = load_model(bonus_model_path)
                
                # Check input shape compatibility
                expected_seq_len = self.params['sequence_length']
                if self.main_model.input_shape[1] != expected_seq_len:
                    logger.warning(
                        "Main model sequence length mismatch: %s vs %s. Rebuilding...",
                        self.main_model.input_shape[1],
                        expected_seq_len,
                    )
                    raise ValueError("Sequence length mismatch")
                if self.main_model.input_shape[-1] != self.main_input_dim:
                    logger.warning(f"Main model input shape mismatch: {self.main_model.input_shape[-1]} vs {self.main_input_dim}. Rebuilding...")
                    raise ValueError("Shape mismatch")
                    
                if self.bonus_model.input_shape[1] != expected_seq_len:
                    logger.warning(
                        "Bonus model sequence length mismatch: %s vs %s. Rebuilding...",
                        self.bonus_model.input_shape[1],
                        expected_seq_len,
                    )
                    raise ValueError("Sequence length mismatch")
                if self.bonus_model.input_shape[-1] != self.bonus_input_dim:
                    logger.warning(f"Bonus model input shape mismatch: {self.bonus_model.input_shape[-1]} vs {self.bonus_input_dim}. Rebuilding...")
                    raise ValueError("Shape mismatch")

                # We also need critics for PPO, but we can rebuild them or load if saved
                # For simplicity, we rebuild critics as they are for training only
                self.main_critic = self._create_critic_model(self.main_input_dim, "main_critic")
                self.bonus_critic = self._create_critic_model(self.bonus_input_dim, "bonus_critic")
                
                self.models_loaded = True
                return
            except Exception as e:
                logger.warning(f"Failed to load models: {e}. Rebuilding...")
        
        logger.info("Building new Transformer models...")
        self.main_model = self._create_transformer_model(
            input_dim=self.main_input_dim,
            output_size=MAIN_NUMBER_RANGE,
            name="main"
        )
        self.bonus_model = self._create_transformer_model(
            input_dim=self.bonus_input_dim,
            output_size=BONUS_NUMBER_RANGE,
            name="bonus"
        )
        
        self.main_critic = self._create_critic_model(self.main_input_dim, "main_critic")
        self.bonus_critic = self._create_critic_model(self.bonus_input_dim, "bonus_critic")
        
        self.models_loaded = False

    def _create_transformer_model(self, input_dim: int, output_size: int, name: str) -> Model:
        """Create a Transformer Encoder model."""
        sequence_length = self.params['sequence_length']
        d_model = self.arch.get('d_model', 192)
        num_layers = self.arch.get('num_layers', 6)
        num_heads = self.arch.get('num_heads', 4)
        ff_mult = self.arch.get('ff_multiplier', 2.0)
        dropout_rate = self.arch.get('dropout', 0.0)
        label_smoothing = self.arch.get('label_smoothing', 0.0)

        # Input
        inp = Input(shape=(sequence_length, input_dim))
        
        # Projection to d_model (if input_dim != d_model, or just to embed)
        x = Dense(d_model)(inp)
        
        # Positional Encoding
        # Simple learnable positional encoding
        positions = tf.range(sequence_length)
        positions = tf.expand_dims(positions, 0) # Shape (1, sequence_length)
        pos_emb = Embedding(input_dim=sequence_length, output_dim=d_model)(positions) # Shape (1, sequence_length, d_model)
        x = Add()([x, pos_emb])
        
        # Transformer Blocks
        for _ in range(num_layers): # deeper stack configurable for more capacity
            # Multi-Head Attention
            attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, x)
            attn_output = Dropout(dropout_rate)(attn_output)
            x = Add()([x, attn_output])
            x = LayerNormalization()(x)
            
            # Feed Forward
            ff_output = Dense(int(d_model * ff_mult), activation='relu')(x)
            ff_output = Dropout(dropout_rate)(ff_output)
            ff_output = Dense(d_model)(ff_output)
            ff_output = Dropout(dropout_rate)(ff_output)
            x = Add()([x, ff_output])
            x = LayerNormalization()(x)
            
        # Global Pooling
        x = GlobalAveragePooling1D()(x)
        
        # Output Head
        x = Dense(128, activation='relu')(x)
        x = Dropout(max(dropout_rate, 0.1))(x)
        
        # Output probability for each number
        out = Dense(output_size, activation='sigmoid', name=f'{name}_output')(x)
        
        model = Model(inputs=inp, outputs=out, name=f'{name}_transformer')
        loss_fn = tf.keras.losses.BinaryCrossentropy(label_smoothing=label_smoothing)
        model.compile(optimizer=Adam(learning_rate=self.params['learning_rate']),
                     loss=loss_fn,
                     metrics=['accuracy'])
        return model

    def _create_critic_model(self, input_dim: int, name: str) -> Model:
        """Create a Critic model (Value Function) for PPO."""
        # Similar architecture to Actor but outputs a single scalar (Value)
        sequence_length = self.params['sequence_length']
        
        inp = Input(shape=(sequence_length, input_dim))
        d_model = 64
        x = Dense(d_model)(inp)
        x = GlobalAveragePooling1D()(x)
        x = Dense(64, activation='relu')(x)
        out = Dense(1, name=f'{name}_output')(x) # Linear output for Value
        
        model = Model(inputs=inp, outputs=out, name=f'{name}_model')
        model.compile(optimizer=Adam(learning_rate=self.params['learning_rate']), loss='mse')
        return model

    def train(self, data: pd.DataFrame):
        """Train the models using Supervised Learning."""
        logger.info("Preparing training data...")
        sequences, main_targets, bonus_targets = self._prepare_sequences(data)
        
        # Train Main Model
        logger.info("Training Main Transformer...")
        self.main_model.fit(
            sequences[0], main_targets,
            epochs=self.params['epochs'],
            batch_size=self.params['batch_size'],
            validation_split=0.2,
            callbacks=[
                EarlyStopping(patience=10, restore_best_weights=True),
                ReduceLROnPlateau(factor=0.5, patience=5)
            ],
            verbose=1
        )
        self.main_model.save(MODELS_DIR / "main_transformer.keras")
        
        # Train Bonus Model
        logger.info("Training Bonus Transformer...")
        self.bonus_model.fit(
            sequences[1], bonus_targets,
            epochs=self.params['epochs'],
            batch_size=self.params['batch_size'],
            validation_split=0.2,
            callbacks=[
                EarlyStopping(patience=10, restore_best_weights=True),
                ReduceLROnPlateau(factor=0.5, patience=5)
            ],
            verbose=1
        )
        self.bonus_model.save(MODELS_DIR / "bonus_transformer.keras")
        self.models_loaded = True

    def _prepare_sequences(self, data: pd.DataFrame) -> Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray, np.ndarray]:
        """Create sequences of Features for training with enhanced features."""
        from .data import LotteryDataManager
        dm = LotteryDataManager()

        # 1. Calculate all features
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

        # 2. Stack Features (MUST match order in build_models input_dim calculation)
        # Main: gaps, gap_deltas, freqs, date, hot_cold, affinity, global, var_ent, momentum, spread
        main_feats_list = [main_gaps]
        for w in sorted(main_gap_delta.keys()):
            main_feats_list.append(main_gap_delta[w])
        for w in [10, 50, 100]:
            main_feats_list.append(freq_features[f'main_freq_{w}'])
        main_feats_list.append(date_features)
        main_feats_list.append(main_hot_cold)
        main_feats_list.append(main_affinity)
        main_feats_list.append(global_features)
        main_feats_list.append(main_var_ent)
        main_feats_list.append(main_momentum)
        main_feats_list.append(spread_features)
        main_features = np.hstack(main_feats_list)

        # Bonus: gaps, gap_deltas, freqs, date, hot_cold, affinity, global, var_ent, momentum, spread
        bonus_feats_list = [bonus_gaps]
        for w in sorted(bonus_gap_delta.keys()):
            bonus_feats_list.append(bonus_gap_delta[w])
        for w in [10, 50, 100]:
            bonus_feats_list.append(freq_features[f'bonus_freq_{w}'])
        bonus_feats_list.append(date_features)
        bonus_feats_list.append(bonus_hot_cold)
        bonus_feats_list.append(bonus_affinity)
        bonus_feats_list.append(global_features)
        bonus_feats_list.append(bonus_var_ent)
        bonus_feats_list.append(bonus_momentum)
        bonus_feats_list.append(spread_features)
        bonus_features = np.hstack(bonus_feats_list)

        sequences_main = []
        sequences_bonus = []
        main_targets = []
        bonus_targets = []

        seq_len = self.params['sequence_length']

        for i in range(seq_len, len(data)):
            seq_main = main_features[i-seq_len+1 : i+1]
            seq_bonus = bonus_features[i-seq_len+1 : i+1]

            target_draw = data.iloc[i]

            main_target = np.zeros(MAIN_NUMBER_RANGE)
            for num in target_draw['main_numbers']:
                if 1 <= num <= MAIN_NUMBER_RANGE:
                    main_target[num-1] = 1

            bonus_target = np.zeros(BONUS_NUMBER_RANGE)
            for num in target_draw['bonus_numbers']:
                if 1 <= num <= BONUS_NUMBER_RANGE:
                    bonus_target[num-1] = 1

            sequences_main.append(seq_main)
            sequences_bonus.append(seq_bonus)
            main_targets.append(main_target)
            bonus_targets.append(bonus_target)

        return (np.array(sequences_main), np.array(sequences_bonus)), np.array(main_targets), np.array(bonus_targets)

    def train_ppo(self, data: pd.DataFrame, epochs: int = None):
        """
        Fine-tune the model using Proximal Policy Optimization (PPO).
        """
        logger.info("Starting PPO Fine-Tuning...")
        
        sequences, main_targets, bonus_targets = self._prepare_sequences(data)
        main_seqs, bonus_seqs = sequences
        
        # Hyperparameters
        from .config import PPO_PARAMS
        if epochs is None:
            epochs = int(PPO_PARAMS.get("epochs", 50))
        clip_ratio = PPO_PARAMS['clip_ratio']
        gamma = PPO_PARAMS['gamma']
        lam = PPO_PARAMS['lam']
        entropy_coef = PPO_PARAMS['entropy_coef']
        
        main_optimizer = Adam(learning_rate=PPO_PARAMS['learning_rate'])
        bonus_optimizer = Adam(learning_rate=PPO_PARAMS['learning_rate'])
        
        batch_size = PPO_PARAMS['batch_size']
        n_batches = len(main_seqs) // batch_size
        if n_batches == 0:
            logger.warning("Not enough data for PPO batching; skipping PPO fine-tuning.")
            return

        def _normalize_tensor(x: tf.Tensor) -> tf.Tensor:
            mean = tf.reduce_mean(x)
            std = tf.math.reduce_std(x) + 1e-8
            return (x - mean) / std
        
        def _compute_gae(deltas: tf.Tensor) -> tf.Tensor:
            """
            Generalized Advantage Estimation placeholder.
            Here trajectories are length-1, so GAE reduces to the TD error (delta).
            """
            return deltas
        
        for epoch in range(epochs):
            total_reward = 0
            
            indices = np.arange(len(main_seqs))
            np.random.shuffle(indices)
            
            for b in range(n_batches):
                batch_idx = indices[b*batch_size : (b+1)*batch_size]
                
                # Get batch data
                states_main = main_seqs[batch_idx]
                states_bonus = bonus_seqs[batch_idx]
                
                # 1. Rollout (Collect Trajectories)
                # We need "Old" probabilities for PPO ratio
                old_main_probs = self.main_model(states_main, training=False)
                old_bonus_probs = self.bonus_model(states_bonus, training=False)
                
                # Sample Actions
                # Normalize for sampling
                old_main_probs_norm = old_main_probs / tf.reduce_sum(old_main_probs, axis=1, keepdims=True)
                old_bonus_probs_norm = old_bonus_probs / tf.reduce_sum(old_bonus_probs, axis=1, keepdims=True)
                
                main_actions = tf.random.categorical(tf.math.log(old_main_probs_norm + 1e-9), num_samples=N_MAIN)
                bonus_actions = tf.random.categorical(tf.math.log(old_bonus_probs_norm + 1e-9), num_samples=N_BONUS)
                
                # Calculate Enhanced Rewards with near-miss detection
                rewards = []
                for i in range(batch_size):
                    # Main Reward
                    selected_main = set(main_actions[i].numpy() + 1)
                    true_main = set(np.where(main_targets[batch_idx[i]] == 1)[0] + 1)
                    hits = len(selected_main & true_main)

                    # Bonus Reward
                    selected_bonus = set(bonus_actions[i].numpy() + 1)
                    true_bonus = set(np.where(bonus_targets[batch_idx[i]] == 1)[0] + 1)
                    bonus_hits = len(selected_bonus & true_bonus)

                    # Base reward (exponential scaling for hits)
                    r = 0.0
                    if hits > 0:
                        r += (2.0 ** hits) - 1.0  # Exponential: 1, 3, 7, 15, 31 for 1-5 hits
                    if bonus_hits > 0:
                        r += (1.5 ** bonus_hits) * 2.0  # Bonus hits worth more

                    # Near-miss reward: reward numbers that were off by 1 or 2
                    # This helps the model learn to concentrate probability mass near correct numbers
                    near_miss_main = 0
                    for sel in selected_main:
                        for true in true_main:
                            diff = abs(sel - true)
                            if diff == 1:
                                near_miss_main += 0.3  # Adjacent number
                            elif diff == 2:
                                near_miss_main += 0.1  # Two away
                    r += near_miss_main

                    near_miss_bonus = 0
                    for sel in selected_bonus:
                        for true in true_bonus:
                            diff = abs(sel - true)
                            if diff == 1:
                                near_miss_bonus += 0.4
                            elif diff == 2:
                                near_miss_bonus += 0.15
                    r += near_miss_bonus

                    # Probability calibration bonus - reward confident correct predictions
                    selected_main_indices = np.array(list(selected_main)) - 1
                    selected_bonus_indices = np.array(list(selected_bonus)) - 1
                    selected_main_probs = old_main_probs_norm[i].numpy()[selected_main_indices]
                    selected_bonus_probs = old_bonus_probs_norm[i].numpy()[selected_bonus_indices]
                    prob_confidence = float(np.mean(selected_main_probs)) + float(np.mean(selected_bonus_probs))

                    # If prediction was correct AND confident, big bonus
                    if hits >= 2:
                        r += prob_confidence * 5.0
                    elif hits >= 1:
                        r += prob_confidence * 2.0

                    # Penalty for overconfident wrong predictions
                    if hits == 0 and prob_confidence > 0.6:
                        r -= 2.0

                    # Baseline subtraction: subtract expected random performance
                    # Random chance of hitting k main numbers follows hypergeometric distribution
                    # Expected hits for random = N_MAIN * (N_MAIN / MAIN_NUMBER_RANGE) = 5 * (5/50) = 0.5
                    baseline_reward = 0.5
                    r -= baseline_reward

                    rewards.append(r)
                
                rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
                
                # Calculate Advantages (TD error / GAE for length-1 trajectories)
                values_main = self.main_critic(states_main)
                values_bonus = self.bonus_critic(states_bonus)
                # For single-step episodes, GAE collapses to TD error; we keep gamma/lam hooks for clarity
                deltas_main = rewards - tf.squeeze(values_main)
                deltas_bonus = rewards - tf.squeeze(values_bonus)
                adv_main = _normalize_tensor(_compute_gae(deltas_main) * gamma)
                adv_bonus = _normalize_tensor(_compute_gae(deltas_bonus) * gamma)
                
                # 2. Update Policy (Actor) and Value (Critic)
                with tf.GradientTape(persistent=True) as tape:
                    # New Probabilities
                    new_main_probs = self.main_model(states_main, training=True)
                    new_bonus_probs = self.bonus_model(states_bonus, training=True)
                    
                    # Calculate Log Probs of Selected Actions
                    def get_log_probs(probs, actions):
                        # Normalize
                        probs_norm = probs / tf.reduce_sum(probs, axis=1, keepdims=True)
                        batch_indices = tf.expand_dims(tf.range(batch_size), 1)
                        batch_indices = tf.tile(batch_indices, [1, actions.shape[1]])
                        gather_indices = tf.stack([batch_indices, tf.cast(actions, tf.int32)], axis=2)
                        selected_probs = tf.gather_nd(probs_norm, gather_indices)
                        return tf.reduce_sum(tf.math.log(selected_probs + 1e-9), axis=1)
                    
                    new_log_probs_main = get_log_probs(new_main_probs, main_actions)
                    old_log_probs_main = get_log_probs(old_main_probs, main_actions)
                    
                    new_log_probs_bonus = get_log_probs(new_bonus_probs, bonus_actions)
                    old_log_probs_bonus = get_log_probs(old_bonus_probs, bonus_actions)
                    
                    # Ratio
                    ratio_main = tf.exp(new_log_probs_main - old_log_probs_main)
                    ratio_bonus = tf.exp(new_log_probs_bonus - old_log_probs_bonus)
                    
                    # Entropy (for exploration)
                    # Entropy = -sum(p * log(p))
                    # We approximate it using the sampled log probs or calculate exactly if possible.
                    # Since we have probabilities, we can calculate it exactly for the batch.
                    # new_main_probs: (B, 50)
                    entropy_main = -tf.reduce_sum(new_main_probs * tf.math.log(new_main_probs + 1e-9), axis=1)
                    entropy_bonus = -tf.reduce_sum(new_bonus_probs * tf.math.log(new_bonus_probs + 1e-9), axis=1)
                    
                    
                    # PPO Loss
                    surr1_main = ratio_main * adv_main
                    surr2_main = tf.clip_by_value(ratio_main, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv_main
                    actor_loss_main = -tf.reduce_mean(tf.minimum(surr1_main, surr2_main) + entropy_coef * entropy_main)
                    
                    surr1_bonus = ratio_bonus * adv_bonus
                    surr2_bonus = tf.clip_by_value(ratio_bonus, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv_bonus
                    actor_loss_bonus = -tf.reduce_mean(tf.minimum(surr1_bonus, surr2_bonus) + entropy_coef * entropy_bonus)
                    
                    # Critic Loss
                    new_values_main = self.main_critic(states_main, training=True)
                    critic_loss_main = tf.reduce_mean(tf.square(rewards - tf.squeeze(new_values_main)))
                    
                    new_values_bonus = self.bonus_critic(states_bonus, training=True)
                    critic_loss_bonus = tf.reduce_mean(tf.square(rewards - tf.squeeze(new_values_bonus)))
                    
                    total_loss_main = actor_loss_main + 0.5 * critic_loss_main
                    total_loss_bonus = actor_loss_bonus + 0.5 * critic_loss_bonus
                    
                # Apply Gradients
                grads_main = tape.gradient(total_loss_main, self.main_model.trainable_variables + self.main_critic.trainable_variables)
                grads_bonus = tape.gradient(total_loss_bonus, self.bonus_model.trainable_variables + self.bonus_critic.trainable_variables)
                
                main_optimizer.apply_gradients(zip(grads_main, self.main_model.trainable_variables + self.main_critic.trainable_variables))
                bonus_optimizer.apply_gradients(zip(grads_bonus, self.bonus_model.trainable_variables + self.bonus_critic.trainable_variables))
                
                total_reward += tf.reduce_sum(rewards)
                
            avg_reward = total_reward / (n_batches * batch_size)
            logger.info(f"PPO Epoch {epoch+1}/{epochs} - Avg Reward: {avg_reward:.4f}")
            
        self.main_model.save(MODELS_DIR / "main_transformer.keras")
        self.bonus_model.save(MODELS_DIR / "bonus_transformer.keras")
        logger.info("PPO Fine-Tuning Complete.")

    def _apply_sampling_bias(self, probs: np.ndarray, top_k: int, temperature: float,
                               top_p: float = None, required_nonzero: int = None) -> np.ndarray:
        """
        Reweight probabilities using proper temperature scaling and nucleus sampling.

        Temperature is correctly applied to log-odds (logits), not raw probabilities.
        This gives proper Boltzmann distribution behavior.

        Args:
            probs: Raw probability predictions from model
            top_k: Keep only top-k highest probability numbers
            temperature: Controls sampling randomness (lower = more deterministic)
            top_p: Nucleus sampling threshold - keep smallest set summing to top_p
            required_nonzero: Minimum non-zero entries needed for sampling
        """
        probs = np.array(probs, dtype=np.float64)
        probs = np.clip(probs, 1e-10, 1.0 - 1e-10)  # Avoid log(0)

        # Convert to logits (log-odds) for proper temperature scaling
        logits = np.log(probs / (1 - probs))

        # Apply temperature to logits (proper Boltzmann scaling)
        if temperature and temperature > 0:
            logits = logits / temperature

        # Convert back to probabilities via softmax
        exp_logits = np.exp(logits - np.max(logits))  # Subtract max for numerical stability
        probs = exp_logits / np.sum(exp_logits)

        effective_top_k = top_k if top_k else probs.size
        if required_nonzero is not None:
            effective_top_k = max(effective_top_k, required_nonzero)

        # Apply nucleus (top-p) sampling if specified
        if top_p is not None and 0 < top_p < 1:
            sorted_indices = np.argsort(probs)[::-1]
            cumsum = np.cumsum(probs[sorted_indices])
            cutoff_idx = np.searchsorted(cumsum, top_p) + 1
            cutoff_idx = max(cutoff_idx, required_nonzero or 1)
            nucleus_indices = sorted_indices[:cutoff_idx]
            mask = np.zeros_like(probs)
            mask[nucleus_indices] = 1.0
            probs = probs * mask
        # Apply top-k masking
        elif effective_top_k and effective_top_k < probs.size:
            top_indices = np.argpartition(probs, -effective_top_k)[-effective_top_k:]
            mask = np.zeros_like(probs)
            mask[top_indices] = 1.0
            probs = probs * mask

        total = probs.sum()
        if total <= 0:
            probs = np.ones_like(probs)
            total = probs.sum()
        # Ensure we have enough non-zero entries to sample without replacement
        if required_nonzero is not None and np.count_nonzero(probs) < required_nonzero:
            probs = np.ones_like(probs)
            total = probs.sum()
        return probs / total

    def predict(self, recent_data: pd.DataFrame, num_predictions: int = 5) -> List[Dict]:
        """Generate predictions using probabilistic sampling with enhanced features."""
        if not self.models_loaded:
            raise ValueError("Models not loaded. Call build_models() and optionally train().")

        from .data import LotteryDataManager
        dm = LotteryDataManager()
        # 1. Add a dummy row for the next draw so the final timestep matches training alignment.
        next_date = dm.infer_next_draw_date(recent_data)
        dummy_row = pd.DataFrame([{
            'date': next_date,
            'main_numbers': [],
            'bonus_numbers': [],
            'jackpot': '',
            'result': ''
        }])
        data_augmented = pd.concat([recent_data, dummy_row], ignore_index=True)

        # 2. Calculate all features on augmented data
        main_gaps, bonus_gaps = dm.calculate_gap_states(data_augmented, MAIN_NUMBER_RANGE, BONUS_NUMBER_RANGE)
        main_gap_delta, bonus_gap_delta = dm.calculate_gap_delta_features(data_augmented)
        freq_features = dm.calculate_frequency_features(data_augmented)
        date_features = dm.calculate_date_features(data_augmented)
        main_hot_cold, bonus_hot_cold = dm.calculate_hot_cold_features(data_augmented)
        main_affinity, bonus_affinity = dm.calculate_cooccurrence_features(data_augmented)
        global_features = dm.calculate_global_features(data_augmented)

        # New advanced features
        main_var_ent, bonus_var_ent = dm.calculate_variance_entropy_features(data_augmented)
        main_momentum, bonus_momentum = dm.calculate_momentum_features(data_augmented)
        spread_features = dm.calculate_spread_features(data_augmented)

        # 2. Stack Features (MUST match order in _prepare_sequences)
        # Main: gaps, gap_deltas, freqs, date, hot_cold, affinity, global, var_ent, momentum, spread
        main_feats_list = [main_gaps]
        for w in sorted(main_gap_delta.keys()):
            main_feats_list.append(main_gap_delta[w])
        for w in [10, 50, 100]:
            main_feats_list.append(freq_features[f'main_freq_{w}'])
        main_feats_list.append(date_features)
        main_feats_list.append(main_hot_cold)
        main_feats_list.append(main_affinity)
        main_feats_list.append(global_features)
        main_feats_list.append(main_var_ent)
        main_feats_list.append(main_momentum)
        main_feats_list.append(spread_features)
        main_features = np.hstack(main_feats_list)

        # Bonus: gaps, gap_deltas, freqs, date, hot_cold, affinity, global, var_ent, momentum, spread
        bonus_feats_list = [bonus_gaps]
        for w in sorted(bonus_gap_delta.keys()):
            bonus_feats_list.append(bonus_gap_delta[w])
        for w in [10, 50, 100]:
            bonus_feats_list.append(freq_features[f'bonus_freq_{w}'])
        bonus_feats_list.append(date_features)
        bonus_feats_list.append(bonus_hot_cold)
        bonus_feats_list.append(bonus_affinity)
        bonus_feats_list.append(global_features)
        bonus_feats_list.append(bonus_var_ent)
        bonus_feats_list.append(bonus_momentum)
        bonus_feats_list.append(spread_features)
        bonus_features = np.hstack(bonus_feats_list)
        
        seq_len = self.params['sequence_length']
        if len(main_features) < seq_len:
            raise ValueError(f"Not enough data for prediction. Need at least {seq_len} draws.")
            
        current_main_seq = main_features[-seq_len:].reshape(1, seq_len, -1)
        current_bonus_seq = bonus_features[-seq_len:].reshape(1, seq_len, -1)
        
        predictions = []
        
        for _ in range(num_predictions):
            # Predict Main
            main_probs_raw = self.main_model.predict(current_main_seq, verbose=0)[0]
            main_probs = self._apply_sampling_bias(
                main_probs_raw,
                top_k=self.sampling.get("top_k_main"),
                temperature=self.sampling.get("temperature_main"),
                top_p=self.sampling.get("top_p_main"),
                required_nonzero=N_MAIN
            )
            main_prob_vector = main_probs.copy()

            main_nums = np.random.choice(
                np.arange(1, MAIN_NUMBER_RANGE + 1),
                size=N_MAIN,
                replace=False,
                p=main_probs
            )

            # Predict Bonus
            bonus_probs_raw = self.bonus_model.predict(current_bonus_seq, verbose=0)[0]
            bonus_probs = self._apply_sampling_bias(
                bonus_probs_raw,
                top_k=self.sampling.get("top_k_bonus"),
                temperature=self.sampling.get("temperature_bonus"),
                top_p=self.sampling.get("top_p_bonus"),
                required_nonzero=N_BONUS
            )
            bonus_prob_vector = bonus_probs.copy()
            
            bonus_nums = np.random.choice(
                np.arange(1, BONUS_NUMBER_RANGE + 1),
                size=N_BONUS,
                replace=False,
                p=bonus_probs
            )
            
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
                'source': 'Neural_Transformer'
            })
            
        return predictions
