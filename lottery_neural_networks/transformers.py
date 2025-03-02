#!/usr/bin/env python3
"""
Complete Lottery Predictor
An enhanced lottery prediction system using transformer-based architecture.

Usage:
  # Step 1: Create a sample file if you don't have lottery data
    python transformers.py --mode create_sample --draws 500

    # Step 2: Optimize hyperparameters (most important step for accuracy)
    python transformers.py --mode optimize --trials 20 --input lottery_numbers.txt

    # Step 3: Train using ensemble models with the optimized parameters
    python transformers.py --mode train_and_predict --ensemble --epochs 150 --input lottery_numbers.txt

    # Step 4: Evaluate model quality with extensive backtesting
    python transformers.py --mode validate --backtest 30 --input lottery_numbers.txt

    # Step 5: Make predictions using your trained model
    python transformers.py --mode predict_only --input lottery_numbers.txt
"""

import tensorflow as tf
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Dense, Dropout
from tensorflow.keras.layers import Input, BatchNormalization, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l1_l2
import numpy as np
import os
import logging
import argparse
from sklearn.preprocessing import StandardScaler
from collections import Counter
import traceback
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error
import json
from datetime import datetime

# Set up structured logger
def setup_logger(name='LotteryPredictor', log_dir='logs'):
    """Setup and return a configured logger"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers if any
    if logger.handlers:
        logger.handlers = []
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create directory for logs if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Create unique log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f'{name}_{timestamp}.log')
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

# Initialize logger
logger = setup_logger()

class TransformerBlock(tf.keras.layers.Layer):
    """
    Streamlined Transformer block with multi-head attention and feed-forward network
    """
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        # Store params as instance variables
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="gelu"),
            Dropout(rate),
            Dense(embed_dim),
        ])
        
        # Layer normalizations with improved epsilon
        self.layernorm1 = LayerNormalization(epsilon=1e-5)
        self.layernorm2 = LayerNormalization(epsilon=1e-5)
        
        # Dropouts
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)
    
    # Add get_config method
    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "rate": self.rate
        })
        return config
    
    def call(self, inputs, training=True):
        # Attention mechanism
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        # Feed-forward network
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        
        return self.layernorm2(out1 + ffn_output)

class EnhancedLotteryPredictor:
    """
    Improved lottery number predictor with more efficient architecture and better metrics
    """
    def __init__(self, 
                 sequence_length=15,  # Reduced from 20
                 test_size=0.2, 
                 random_state=42,
                 main_numbers=5, 
                 bonus_numbers=2, 
                 main_number_range=(1, 50), 
                 bonus_number_range=(1, 12),
                 model_dir='enhanced_lottery_model',
                 embed_dim=32,        # Reduced from 64
                 num_heads=4,         # Reduced from 8
                 ff_dim=64,           # Reduced from 128
                 transformer_blocks=2):  # Reduced from 3
        """
        Initialize the enhanced predictor with more efficient parameters
        """
        self.sequence_length = sequence_length
        self.test_size = test_size
        self.random_state = random_state
        self.main_numbers = main_numbers
        self.bonus_numbers = bonus_numbers
        self.main_min_number, self.main_max_number = main_number_range
        self.bonus_min_number, self.bonus_max_number = bonus_number_range
        self.model_dir = model_dir
        self.numbers_per_draw = main_numbers + bonus_numbers
        
        # Model hyperparameters
        self.embed_dim = embed_dim 
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_transformer_blocks = transformer_blocks
        
        # Enhanced scalers - using StandardScaler for better normalization
        self.scaler_X_main = StandardScaler()
        self.scaler_y_main = StandardScaler()
        self.scaler_X_bonus = StandardScaler()
        self.scaler_y_bonus = StandardScaler()
        
        # Initialize models
        self.main_model = None
        self.bonus_model = None
        self.ensemble = None
        
        # Set random seeds for reproducibility
        np.random.seed(random_state)
        tf.random.set_seed(random_state)
        
        # Create model directory with timestamp for better organization
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.model_dir = f"{model_dir}_{timestamp}"
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Metadata to track performance and parameters
        self.metadata = {
            'creation_date': datetime.now().isoformat(),
            'params': {
                'sequence_length': sequence_length,
                'main_numbers': main_numbers,
                'bonus_numbers': bonus_numbers,
                'main_range': main_number_range,
                'bonus_range': bonus_number_range,
                'embed_dim': embed_dim,
                'num_heads': num_heads,
                'ff_dim': ff_dim,
                'transformer_blocks': transformer_blocks
            },
            'performance': {},
            'predictions': []
        }

    def separate_main_and_bonus(self, numbers):
        """
        Separate the flat list of numbers into main and bonus sequences
        
        Args:
            numbers: Flat list of all lottery numbers
            
        Returns:
            main_numbers: List of main numbers only
            bonus_numbers: List of bonus numbers only
            draws: List of complete drawings
        """
        logger.info(f"Separating main and bonus numbers (assuming {self.main_numbers} main and {self.bonus_numbers} bonus per draw)")
        
        # Ensure we have complete draws
        if len(numbers) % self.numbers_per_draw != 0:
            logger.warning(f"Number of elements ({len(numbers)}) is not divisible by numbers per draw ({self.numbers_per_draw})")
            # Trim the list to get complete draws
            numbers = numbers[:-(len(numbers) % self.numbers_per_draw)]
        
        # Calculate how many complete draws we have
        num_draws = len(numbers) // self.numbers_per_draw
        logger.info(f"Found {num_draws} complete draws in the data")
        
        # Handle the case of very few draws - IMPROVED with better synthetic data
        if num_draws < 10:
            logger.warning("Very few draws available, creating additional synthetic draws for training")
            
            # Generate synthetic draws using historical frequencies if available
            if num_draws > 0:
                # Use historical data to inform synthesis
                draws_array = np.array(numbers).reshape(num_draws, self.numbers_per_draw)
                historical_main = draws_array[:, :self.main_numbers].flatten()
                historical_bonus = draws_array[:, self.main_numbers:].flatten()
                
                # Calculate probabilities based on historical frequencies
                main_counts = Counter(historical_main)
                bonus_counts = Counter(historical_bonus)
                
                # Convert to probabilities
                main_probs = {num: count/len(historical_main) for num, count in main_counts.items()}
                bonus_probs = {num: count/len(historical_bonus) for num, count in bonus_counts.items()}
                
                # Fill in missing numbers with uniform probability
                for i in range(self.main_min_number, self.main_max_number + 1):
                    if i not in main_probs:
                        main_probs[i] = 1 / (self.main_max_number - self.main_min_number + 1)
                
                for i in range(self.bonus_min_number, self.bonus_max_number + 1):
                    if i not in bonus_probs:
                        bonus_probs[i] = 1 / (self.bonus_max_number - self.bonus_min_number + 1)
                
                # Normalize probabilities
                main_sum = sum(main_probs.values())
                bonus_sum = sum(bonus_probs.values())
                
                main_probs = {num: prob/main_sum for num, prob in main_probs.items()}
                bonus_probs = {num: prob/bonus_sum for num, prob in bonus_probs.items()}
                
                # Generate synthetic draws using weighted probabilities
                synthetic_draws = []
                for _ in range(max(20 - num_draws, 0)):
                    main_nums = sorted(np.random.choice(
                        list(main_probs.keys()), 
                        self.main_numbers, 
                        replace=False, 
                        p=list(main_probs.values())
                    ))
                    
                    bonus_nums = sorted(np.random.choice(
                        list(bonus_probs.keys()), 
                        self.bonus_numbers, 
                        replace=False, 
                        p=list(bonus_probs.values())
                    ))
                    
                    synthetic_draws.extend(main_nums + bonus_nums)
            else:
                # If no historical data, use uniform distribution
                synthetic_draws = []
                for _ in range(20):
                    main_nums = sorted(np.random.choice(range(self.main_min_number, self.main_max_number + 1), 
                                                     self.main_numbers, replace=False))
                    bonus_nums = sorted(np.random.choice(range(self.bonus_min_number, self.bonus_max_number + 1), 
                                                       self.bonus_numbers, replace=False))
                    synthetic_draws.extend(main_nums + bonus_nums)
            
            # Prepend synthetic draws to the real ones
            numbers = synthetic_draws + numbers
            num_draws = len(numbers) // self.numbers_per_draw
            logger.info(f"Added synthetic draws, now have {num_draws} draws")
        
        # Reshape the list into a 2D array where each row is a draw
        draws_array = np.array(numbers).reshape(num_draws, self.numbers_per_draw)
        
        # Separate main and bonus numbers
        main_numbers = draws_array[:, :self.main_numbers].flatten().tolist()
        bonus_numbers = draws_array[:, self.main_numbers:].flatten().tolist()
        
        # Create a list of complete drawings for reference
        draws = [
            {
                'main': draws_array[i, :self.main_numbers].tolist(),
                'bonus': draws_array[i, self.main_numbers:].tolist()
            } 
            for i in range(num_draws)
        ]
        
        return main_numbers, bonus_numbers, draws

    def create_features(self, numbers, target_size=1):
        """
        Create optimized input features with better statistical indicators
        
        Args:
            numbers: List of lottery numbers
            target_size: Number of target values to predict
            
        Returns:
            X: Input sequences with features
            y: Target values
        """
        if len(numbers) < self.sequence_length + target_size:
            logger.error(f"Not enough numbers for sequences. Need at least {self.sequence_length + target_size}")
            return np.array([]), np.array([])
        
        X, y = [], []
        
        # Calculate rolling statistics with better metrics
        series = pd.Series(numbers)
        rolling_mean = series.rolling(window=self.sequence_length).mean()
        rolling_std = series.rolling(window=self.sequence_length).std()
        rolling_min = series.rolling(window=self.sequence_length).min()
        rolling_max = series.rolling(window=self.sequence_length).max()
        rolling_median = series.rolling(window=self.sequence_length).median()
        rolling_skew = series.rolling(window=self.sequence_length).skew()
        rolling_kurt = series.rolling(window=self.sequence_length).kurt()
        
        # Calculate exponentially weighted metrics (gives more weight to recent values)
        ewm_mean = series.ewm(span=self.sequence_length).mean()
        ewm_std = series.ewm(span=self.sequence_length).std()
        
        # Calculate frequency within window
        def get_frequency_features(sequence):
            """Calculate enhanced frequency-based features from a sequence"""
            counter = Counter(sequence)
            
            # Get counts for all possible numbers
            range_min = self.main_min_number if min(sequence) >= self.main_min_number else self.bonus_min_number
            range_max = self.main_max_number if max(sequence) <= self.main_max_number else self.bonus_max_number
            
            all_counts = {i: counter.get(i, 0) for i in range(range_min, range_max + 1)}
            
            # Frequency statistics
            most_common = counter.most_common(1)[0][0] if counter else 0
            most_common_count = counter.most_common(1)[0][1] if counter else 0
            least_common = counter.most_common()[-1][0] if counter else 0
            least_common_count = counter.most_common()[-1][1] if counter else 0
            
            # Get counts of odd and even numbers
            odd_count = sum(1 for num in sequence if num % 2 == 1)
            even_count = len(sequence) - odd_count
            
            # Get counts by range bins (useful for lottery patterns)
            range_bins = 5  # Number of bins to divide the range into
            bin_size = (range_max - range_min + 1) // range_bins
            bin_counts = [0] * range_bins
            
            for num in sequence:
                bin_idx = min(range_bins - 1, (num - range_min) // bin_size)
                bin_counts[bin_idx] += 1
            
            # Entropy of distribution (measure of randomness)
            probs = [count/len(sequence) for count in counter.values()]
            entropy = -sum(p * np.log(p) if p > 0 else 0 for p in probs)
            
            # Calculate gap statistics (distance between consecutive numbers)
            sorted_seq = sorted(sequence)
            gaps = [sorted_seq[i+1] - sorted_seq[i] for i in range(len(sorted_seq)-1)]
            avg_gap = np.mean(gaps) if gaps else 0
            max_gap = max(gaps) if gaps else 0
            min_gap = min(gaps) if gaps else 0
            
            return {
                'most_common': most_common,
                'most_common_count': most_common_count,
                'least_common': least_common,
                'least_common_count': least_common_count,
                'odd_count': odd_count,
                'even_count': even_count,
                'bin_counts': bin_counts,
                'entropy': entropy,
                'avg_gap': avg_gap,
                'max_gap': max_gap,
                'min_gap': min_gap
            }
        
        # Create sequences with enhanced features
        for i in range(len(numbers) - self.sequence_length - target_size + 1):
            # Base sequence
            sequence = numbers[i:i+self.sequence_length]
            
            # Target values
            targets = numbers[i+self.sequence_length:i+self.sequence_length+target_size]
            
            # Get rolling statistics at this point
            stats_index = i + self.sequence_length - 1  # End of current sequence
            
            if stats_index >= self.sequence_length:  # Ensure we have enough history for stats
                # Get frequency info
                freq_features = get_frequency_features(sequence)
                
                # Extract sequence data
                sequence_data = np.array(sequence)
                
                # Calculate differences with lag-1, lag-2, and lag-3
                diffs_lag1 = np.diff(sequence_data, prepend=sequence_data[0])
                diffs_lag2 = np.diff(diffs_lag1, prepend=diffs_lag1[0])
                diffs_lag3 = np.diff(diffs_lag2, prepend=diffs_lag2[0])
                
                # Calculate recency-weighted sequence (more recent = more weight)
                weights = np.linspace(0.5, 1.5, self.sequence_length)
                weighted_sequence = sequence_data * weights
                
                # Apply nonlinear transformations
                log_sequence = np.log1p(sequence_data - min(sequence_data) + 1)  # Log transform
                
                # Calculate sequences relative to min/max
                normalized_sequence = (sequence_data - np.min(sequence_data)) / (np.max(sequence_data) - np.min(sequence_data) + 1e-5)
                
                # Calculate autocorrelation (relationship between number and its lag)
                autocorr_lag1 = pd.Series(sequence_data).autocorr(lag=1) if len(sequence_data) > 1 else 0
                autocorr_lag2 = pd.Series(sequence_data).autocorr(lag=2) if len(sequence_data) > 2 else 0
                
                # Robust statistics
                try:
                    stats = np.array([
                        rolling_mean.iloc[stats_index],
                        rolling_std.iloc[stats_index],
                        rolling_min.iloc[stats_index],
                        rolling_max.iloc[stats_index],
                        rolling_median.iloc[stats_index],
                        rolling_skew.iloc[stats_index] if not np.isnan(rolling_skew.iloc[stats_index]) else 0,
                        rolling_kurt.iloc[stats_index] if not np.isnan(rolling_kurt.iloc[stats_index]) else 0,
                        ewm_mean.iloc[stats_index],
                        ewm_std.iloc[stats_index],
                        freq_features['most_common'],
                        freq_features['most_common_count'],
                        freq_features['least_common'],
                        freq_features['least_common_count'],
                        freq_features['odd_count'],
                        freq_features['even_count'],
                        *freq_features['bin_counts'],
                        freq_features['entropy'],
                        freq_features['avg_gap'],
                        freq_features['max_gap'],
                        freq_features['min_gap'],
                        autocorr_lag1,
                        autocorr_lag2,
                        np.percentile(sequence, 25),  # 1st quartile
                        np.percentile(sequence, 75)   # 3rd quartile
                    ])
                except Exception as e:
                    # Fallback if stats calculation fails
                    logger.warning(f"Stats calculation failed: {e}. Using basic stats instead.")
                    stats = np.array([
                        np.mean(sequence),
                        np.std(sequence),
                        np.min(sequence),
                        np.max(sequence),
                        np.median(sequence),
                        0,  # placeholder for skew
                        0,  # placeholder for kurtosis
                        np.mean(sequence),  # placeholder for ewm_mean
                        np.std(sequence),   # placeholder for ewm_std
                        freq_features['most_common'],
                        freq_features['most_common_count'],
                        freq_features['least_common'],
                        freq_features['least_common_count'],
                        freq_features['odd_count'],
                        freq_features['even_count'],
                        *freq_features['bin_counts'],
                        0,  # placeholder for entropy
                        0,  # placeholder for avg_gap
                        0,  # placeholder for max_gap
                        0,  # placeholder for min_gap
                        0,  # placeholder for autocorr_lag1
                        0,  # placeholder for autocorr_lag2
                        np.percentile(sequence, 25),
                        np.percentile(sequence, 75)
                    ])
                
                # Combine features into a fixed-length representation
                feature_vector = np.concatenate([
                    sequence_data,               # Original sequence
                    diffs_lag1,                  # 1st order differences
                    diffs_lag2[:5],              # First 5 2nd order differences
                    diffs_lag3[:5],              # First 5 3rd order differences
                    weighted_sequence[:5],       # First 5 weighted values
                    log_sequence[:5],            # First 5 log-transformed values
                    normalized_sequence[:5],     # First 5 normalized values
                    stats                        # Statistical features
                ])
                
                X.append(feature_vector)
                y.append(targets)
        
        return np.array(X), np.array(y)

    def prepare_data(self, main_numbers, bonus_numbers):
        """
        Prepare and split data for training and validation with improved handling
        
        Args:
            main_numbers: List of main lottery numbers
            bonus_numbers: List of bonus numbers
            
        Returns:
            Prepared data for main and bonus number predictions
        """
        # Create sequences
        logger.info("Creating features...")
        X_main, y_main = self.create_features(main_numbers)
        X_bonus, y_bonus = self.create_features(bonus_numbers)
        
        if X_main.size == 0 or y_main.size == 0 or X_bonus.size == 0 or y_bonus.size == 0:
            logger.error("Failed to create sequences")
            return None
        
        logger.info(f"Created {len(X_main)} main sequences and {len(X_bonus)} bonus sequences")
        
        # Use time-based split instead of random to avoid data leakage
        # More recent data goes to validation
        min_val_samples = max(5, int(0.1 * len(X_main)))
        
        # Main numbers split
        train_size = max(len(X_main) - min_val_samples, int(len(X_main) * (1 - self.test_size)))
        X_main_train, X_main_val = X_main[:train_size], X_main[train_size:]
        y_main_train, y_main_val = y_main[:train_size], y_main[train_size:]
        
        # Bonus numbers split
        train_size_bonus = max(len(X_bonus) - min_val_samples, int(len(X_bonus) * (1 - self.test_size)))
        X_bonus_train, X_bonus_val = X_bonus[:train_size_bonus], X_bonus[train_size_bonus:]
        y_bonus_train, y_bonus_val = y_bonus[:train_size_bonus], y_bonus[train_size_bonus:]
        
        logger.info(f"Split data - Main: {len(X_main_train)} train, {len(X_main_val)} validation")
        logger.info(f"Split data - Bonus: {len(X_bonus_train)} train, {len(X_bonus_val)} validation")
        
        # Augment data - more conservative augmentation to avoid overfitting
        X_main_train, y_main_train = self.augment_data(
            X_main_train, y_main_train, 
            num_samples=min(300, max(10, len(X_main_train)))
        )
        
        X_bonus_train, y_bonus_train = self.augment_data(
            X_bonus_train, y_bonus_train, 
            num_samples=min(300, max(10, len(X_bonus_train))), 
            is_bonus=True
        )
        
        logger.info(f"After augmentation - Main: {len(X_main_train)} train, Bonus: {len(X_bonus_train)} train")
        
        # Scale data with try-except blocks for robustness
        try:
            X_main_train_scaled = self.scaler_X_main.fit_transform(X_main_train)
            X_main_val_scaled = self.scaler_X_main.transform(X_main_val)
            
            y_main_train_scaled = self.scaler_y_main.fit_transform(y_main_train)
            y_main_val_scaled = self.scaler_y_main.transform(y_main_val)
        except Exception as e:
            logger.error(f"Error scaling main number data: {e}")
            return None
            
        try:
            X_bonus_train_scaled = self.scaler_X_bonus.fit_transform(X_bonus_train)
            X_bonus_val_scaled = self.scaler_X_bonus.transform(X_bonus_val)
            
            y_bonus_train_scaled = self.scaler_y_bonus.fit_transform(y_bonus_train)
            y_bonus_val_scaled = self.scaler_y_bonus.transform(y_bonus_val)
        except Exception as e:
            logger.error(f"Error scaling bonus number data: {e}")
            return None
        
        # Reshape for neural network input - more memory efficient reshape
        feature_dim = 1
        
        # Reshape directly without using unnecessary memory
        X_main_train_reshaped = X_main_train_scaled.reshape(X_main_train_scaled.shape[0], -1, feature_dim)
        X_main_val_reshaped = X_main_val_scaled.reshape(X_main_val_scaled.shape[0], -1, feature_dim)
        
        X_bonus_train_reshaped = X_bonus_train_scaled.reshape(X_bonus_train_scaled.shape[0], -1, feature_dim)
        X_bonus_val_reshaped = X_bonus_val_scaled.reshape(X_bonus_val_scaled.shape[0], -1, feature_dim)
        
        logger.info(f"Final data shapes - Main train: {X_main_train_reshaped.shape}, val: {X_main_val_reshaped.shape}")
        logger.info(f"Final data shapes - Bonus train: {X_bonus_train_reshaped.shape}, val: {X_bonus_val_reshaped.shape}")
        
        return {
            'main': {
                'X_train': X_main_train, 'X_val': X_main_val,
                'y_train': y_main_train, 'y_val': y_main_val,
                'X_train_scaled': X_main_train_reshaped, 'X_val_scaled': X_main_val_reshaped,
                'y_train_scaled': y_main_train_scaled, 'y_val_scaled': y_main_val_scaled
            },
            'bonus': {
                'X_train': X_bonus_train, 'X_val': X_bonus_val,
                'y_train': y_bonus_train, 'y_val': y_bonus_val,
                'X_train_scaled': X_bonus_train_reshaped, 'X_val_scaled': X_bonus_val_reshaped,
                'y_train_scaled': y_bonus_train_scaled, 'y_val_scaled': y_bonus_val_scaled
            }
        }

    def augment_data(self, X, y, num_samples=100, is_bonus=False):
        """
        Generate synthetic data to improve model generalization with better control
        
        Args:
            X: Original feature data
            y: Original target data
            num_samples: Number of synthetic samples to generate
            is_bonus: Whether this is for bonus numbers
            
        Returns:
            Augmented X and y data
        """
        augmented_X, augmented_y = X.copy(), y.copy()
        
        # Set valid range based on number type
        min_num = self.bonus_min_number if is_bonus else self.main_min_number
        max_num = self.bonus_max_number if is_bonus else self.main_max_number
        
        # Get data dimensions
        n_samples, feature_size = X.shape
        
        # Smart augmentation - focus on recent samples more
        sample_weights = np.linspace(0.5, 1.5, n_samples)
        sample_weights = sample_weights / sample_weights.sum()
        
        # Generate synthetic examples
        for _ in range(num_samples):
            # Sample with weights that favor recent draws
            idx = np.random.choice(range(n_samples), p=sample_weights)
            perturbed_X = X[idx].copy()
            
            # Limited perturbation - avoid changing too many features
            noise_scale = 0.03  # Reduced from 0.5
            num_to_adjust = np.random.randint(int(0.05 * feature_size), int(0.2 * feature_size))
            indices_to_adjust = np.random.choice(range(feature_size), num_to_adjust, replace=False)
            
            # Add small random noise
            for i in indices_to_adjust:
                perturbed_X[i] += np.random.normal(0, noise_scale)
            
            # Target will be original with slight variation - more conservative
            target_y = y[idx].copy()
            
            # Smaller chance of target modification
            if np.random.random() < 0.15:  # Reduced from 0.3
                noise = np.random.normal(0, 0.5, target_y.shape)  # Reduced from 1.0
                target_y = target_y + noise
                target_y = np.clip(target_y, min_num, max_num)
                target_y = np.round(target_y)
            
            augmented_X = np.vstack([augmented_X, perturbed_X.reshape(1, -1)])
            augmented_y = np.vstack([augmented_y, target_y.reshape(1, -1)])
        
        return augmented_X, augmented_y

    def build_model(self, input_shape, for_bonus=False, dropout_rate=None, lr_multiplier=1.0):
        """
        Build a streamlined transformer-based model for sequence prediction
        
        Args:
            input_shape: Shape of input data
            for_bonus: Whether this is for bonus numbers
            dropout_rate: Optional custom dropout rate
            lr_multiplier: Learning rate multiplier
            
        Returns:
            Transformer-based model with improved architecture
        """
        # Set default dropout rate if not provided
        if dropout_rate is None:
            dropout_rate = 0.1 if not for_bonus else 0.15
        
        # Input layer
        inputs = Input(shape=input_shape)
        
        # Add embeddings
        x = Dense(self.embed_dim)(inputs)
        
        # Add positional encoding - static implementation for efficiency
        seq_length = input_shape[0]
        position = tf.range(float(seq_length))
        position = tf.expand_dims(position, axis=0)
        position = tf.expand_dims(position, axis=2)
        
        # Create Transformer blocks
        for i in range(self.num_transformer_blocks):
            block_dropout = dropout_rate + (i * 0.03)  # Gradual increase
            block = TransformerBlock(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                ff_dim=self.ff_dim,
                rate=block_dropout
            )
            x = block(x)
        
        # Use global average pooling for more robust feature aggregation
        x = GlobalAveragePooling1D()(x)
        
        # Simplified dense layers with appropriate regularization
        x = Dense(32, activation="gelu", kernel_regularizer=l1_l2(l1=1e-6, l2=1e-5))(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        
        # Different final layer size based on number type
        if for_bonus:
            # Smaller network for bonus numbers
            hidden_size = 16
        else:
            hidden_size = 24
            
        x = Dense(hidden_size, activation="gelu", kernel_regularizer=l1_l2(l1=1e-6, l2=1e-5))(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate * 0.75)(x)  # Slightly less dropout before output
        
        # Output layer
        outputs = Dense(1)(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs)
        
        # Compile model with appropriate hyperparameters
        lr = 0.001 * lr_multiplier
        optimizer = Adam(learning_rate=lr, clipnorm=0.5)  # Gradient clipping for stability
        
        # Huber loss is more robust to outliers than MSE
        model.compile(
            optimizer=optimizer, 
            loss=tf.keras.losses.Huber(delta=1.0), 
            metrics=['mae']
        )
        
        model_type = "Bonus" if for_bonus else "Main"
        logger.info(f"Built {model_type} model with {self.num_transformer_blocks} transformer blocks, dropout={dropout_rate:.2f}, lr={lr:.6f}")
        
        return model

    def train_models(self, prepared_data, epochs=100, batch_size=16, patience=15):
        """
        Train both main and bonus number models with improved training parameters
        
        Args:
            prepared_data: Prepared data for training
            epochs: Maximum number of training epochs
            batch_size: Batch size for training
            patience: Early stopping patience
            
        Returns:
            Training history for both models
        """
        # Adjust batch size based on dataset size
        actual_batch_size = min(batch_size, max(8, len(prepared_data['main']['X_train_scaled']) // 10))
        
        # Define callbacks with better parameters
        main_callbacks = [
            EarlyStopping(
                monitor='val_loss', 
                patience=patience, 
                restore_best_weights=True, 
                verbose=1,
                min_delta=0.001  # Minimum improvement required
            ),
            ModelCheckpoint(
                os.path.join(self.model_dir, 'main_model.keras'), 
                monitor='val_loss', 
                save_best_only=True, 
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss', 
                factor=0.5, 
                patience=patience//2, 
                min_lr=0.0001, 
                verbose=1
            )
        ]
        
        # Similar for bonus but with adjusted parameters
        bonus_callbacks = [
            EarlyStopping(
                monitor='val_loss', 
                patience=int(patience*1.5), 
                restore_best_weights=True, 
                verbose=1,
                min_delta=0.001
            ),
            ModelCheckpoint(
                os.path.join(self.model_dir, 'bonus_model.keras'), 
                monitor='val_loss', 
                save_best_only=True, 
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss', 
                factor=0.5, 
                patience=patience//2, 
                min_lr=0.0001, 
                verbose=1
            )
        ]
        
        # Get input shapes
        main_input_shape = prepared_data['main']['X_train_scaled'].shape[1:]
        bonus_input_shape = prepared_data['bonus']['X_train_scaled'].shape[1:]
        
        # Build and train main model
        logger.info(f"Training main numbers model with batch size {actual_batch_size}...")
        self.main_model = self.build_model(main_input_shape, for_bonus=False)
        
        # Train with sample weights that favor recent data
        main_sample_weights = np.linspace(0.7, 1.3, len(prepared_data['main']['X_train_scaled']))
        main_sample_weights = main_sample_weights / np.mean(main_sample_weights)  # Normalize
        
        main_history = self.main_model.fit(
            prepared_data['main']['X_train_scaled'], 
            prepared_data['main']['y_train_scaled'],
            epochs=epochs,
            batch_size=actual_batch_size,
            validation_data=(
                prepared_data['main']['X_val_scaled'], 
                prepared_data['main']['y_val_scaled']
            ),
            callbacks=main_callbacks,
            sample_weight=main_sample_weights,
            verbose=1
        )
        
        # Evaluate on validation set
        main_val_loss, main_val_mae = self.main_model.evaluate(
            prepared_data['main']['X_val_scaled'],
            prepared_data['main']['y_val_scaled'],
            verbose=0
        )
        logger.info(f"Main model validation - Loss: {main_val_loss:.4f}, MAE: {main_val_mae:.4f}")
        
        # Build and train bonus model
        logger.info("Training bonus numbers model...")
        self.bonus_model = self.build_model(bonus_input_shape, for_bonus=True)
        
        # More pronounced weighting for bonus
        bonus_sample_weights = np.linspace(0.6, 1.4, len(prepared_data['bonus']['X_train_scaled']))
        bonus_sample_weights = bonus_sample_weights / np.mean(bonus_sample_weights)  # Normalize
        
        bonus_history = self.bonus_model.fit(
            prepared_data['bonus']['X_train_scaled'], 
            prepared_data['bonus']['y_train_scaled'],
            epochs=int(epochs * 1.2),  # More epochs for bonus
            batch_size=actual_batch_size,
            validation_data=(
                prepared_data['bonus']['X_val_scaled'], 
                prepared_data['bonus']['y_val_scaled']
            ),
            callbacks=bonus_callbacks,
            sample_weight=bonus_sample_weights,
            verbose=1
        )
        
        # Evaluate on validation set
        bonus_val_loss, bonus_val_mae = self.bonus_model.evaluate(
            prepared_data['bonus']['X_val_scaled'],
            prepared_data['bonus']['y_val_scaled'],
            verbose=0
        )
        logger.info(f"Bonus model validation - Loss: {bonus_val_loss:.4f}, MAE: {bonus_val_mae:.4f}")
        
        # Save validation metrics to metadata
        self.metadata['performance']['main_val_loss'] = float(main_val_loss)
        self.metadata['performance']['main_val_mae'] = float(main_val_mae)
        self.metadata['performance']['bonus_val_loss'] = float(bonus_val_loss)
        self.metadata['performance']['bonus_val_mae'] = float(bonus_val_mae)
        
        # Save metadata
        with open(os.path.join(self.model_dir, 'model_metadata.json'), 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        return {
            'main': main_history,
            'bonus': bonus_history
        }
    
    def create_ensemble_models(self, prepared_data, num_models=5, epochs=100):
        """
        Create an ensemble of models for improved prediction stability
        
        Args:
            prepared_data: Prepared data for training
            num_models: Number of models to train in ensemble
            epochs: Training epochs per model
            
        Returns:
            Dictionary of ensemble models
        """
        logger.info(f"Training ensemble of {num_models} models")
        
        main_models = []
        bonus_models = []
        
        for i in range(num_models):
            logger.info(f"Training ensemble model {i+1}/{num_models}")
            
            # Slightly different random seed for each model
            tf.random.set_seed(self.random_state + i)
            np.random.seed(self.random_state + i)
            
            # Get input shapes
            main_input_shape = prepared_data['main']['X_train_scaled'].shape[1:]
            bonus_input_shape = prepared_data['bonus']['X_train_scaled'].shape[1:]
            
            # Build models with slightly different architectures
            main_model = self.build_model(
                main_input_shape, 
                for_bonus=False,
                dropout_rate=0.1 + (i * 0.05),  # Vary dropout
                lr_multiplier=0.8 + (i * 0.1)   # Vary learning rate
            )
            
            bonus_model = self.build_model(
                bonus_input_shape, 
                for_bonus=True,
                dropout_rate=0.15 + (i * 0.05),  # Vary dropout
                lr_multiplier=0.8 + (i * 0.1)    # Vary learning rate
            )
            
            # Set up early stopping callbacks
            early_stop = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=0
            )
            
            # Train with slightly different subsets of data for diversity
            train_idx = np.random.choice(
                len(prepared_data['main']['X_train_scaled']),
                size=int(0.9 * len(prepared_data['main']['X_train_scaled'])),
                replace=False
            )
            
            main_model.fit(
                prepared_data['main']['X_train_scaled'][train_idx], 
                prepared_data['main']['y_train_scaled'][train_idx],
                epochs=epochs,
                batch_size=16,
                validation_data=(
                    prepared_data['main']['X_val_scaled'], 
                    prepared_data['main']['y_val_scaled']
                ),
                callbacks=[early_stop],
                verbose=0
            )
            
            bonus_train_idx = np.random.choice(
                len(prepared_data['bonus']['X_train_scaled']),
                size=int(0.9 * len(prepared_data['bonus']['X_train_scaled'])),
                replace=False
            )
            
            bonus_model.fit(
                prepared_data['bonus']['X_train_scaled'][bonus_train_idx], 
                prepared_data['bonus']['y_train_scaled'][bonus_train_idx],
                epochs=epochs,
                batch_size=16,
                validation_data=(
                    prepared_data['bonus']['X_val_scaled'], 
                    prepared_data['bonus']['y_val_scaled']
                ),
                callbacks=[early_stop],
                verbose=0
            )
            
            # Evaluate the models
            main_val_loss = main_model.evaluate(
                prepared_data['main']['X_val_scaled'],
                prepared_data['main']['y_val_scaled'],
                verbose=0
            )[0]
            
            bonus_val_loss = bonus_model.evaluate(
                prepared_data['bonus']['X_val_scaled'],
                prepared_data['bonus']['y_val_scaled'],
                verbose=0
            )[0]
            
            logger.info(f"Ensemble model {i+1} - Main loss: {main_val_loss:.4f}, Bonus loss: {bonus_val_loss:.4f}")
            
            main_models.append(main_model)
            bonus_models.append(bonus_model)
        
        # Save the ensemble models
        os.makedirs(os.path.join(self.model_dir, 'ensemble'), exist_ok=True)
        
        for i, model in enumerate(main_models):
            model.save(os.path.join(self.model_dir, f'ensemble/main_model_{i}.keras'))
            
        for i, model in enumerate(bonus_models):
            model.save(os.path.join(self.model_dir, f'ensemble/bonus_model_{i}.keras'))
        
        logger.info(f"Ensemble of {num_models} models created and saved")
        
        return {
            'main_models': main_models,
            'bonus_models': bonus_models
        }

    def hyperparameter_search(self, main_numbers, bonus_numbers, trials=10):
        """
        Perform hyperparameter search to find optimal model parameters with improved error handling
        
        Args:
            main_numbers: List of main lottery numbers
            bonus_numbers: List of bonus numbers
            trials: Number of hyperparameter combinations to try
            
        Returns:
            Dictionary of best hyperparameters
        """
        logger.info(f"Starting hyperparameter search with {trials} trials")
        
        # Define search space
        param_space = {
            'sequence_length': [10, 15, 20, 25],
            'embed_dim': [16, 24, 32, 48],
            'num_heads': [2, 4, 6, 8],
            'ff_dim': [32, 48, 64, 96],
            'num_transformer_blocks': [1, 2, 3]
        }
        
        best_score = -np.inf
        best_params = {}
        
        # Default fallback parameters in case all trials fail
        fallback_params = {
            'sequence_length': 15,
            'embed_dim': 32,
            'num_heads': 4,
            'ff_dim': 64,
            'num_transformer_blocks': 2
        }
        
        # Store original parameters to restore later
        original_params = {
            'sequence_length': self.sequence_length,
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'num_transformer_blocks': self.num_transformer_blocks
        }
        
        successful_trials = 0
        
        # Run trials
        for trial in range(trials):
            logger.info(f"Trial {trial+1}/{trials}")
            
            # Sample random parameters from space
            params = {
                'sequence_length': np.random.choice(param_space['sequence_length']),
                'embed_dim': np.random.choice(param_space['embed_dim']),
                'num_heads': np.random.choice(param_space['num_heads']),
                'ff_dim': np.random.choice(param_space['ff_dim']),
                'num_transformer_blocks': np.random.choice(param_space['num_transformer_blocks'])
            }
            
            logger.info(f"Testing parameters: {params}")
            
            # Apply parameters
            for param, value in params.items():
                setattr(self, param, value)
            
            # Prepare data with these parameters
            try:
                prepared_data = self.prepare_data(main_numbers, bonus_numbers)
                
                if prepared_data:
                    # Train mini-models with fewer epochs for speed
                    try:
                        mini_history = self.train_models(
                            prepared_data, 
                            epochs=30,  # Fewer epochs for search
                            batch_size=16,
                            patience=10
                        )
                        
                        # Check if history contains required keys
                        if (mini_history and 'main' in mini_history and 'bonus' in mini_history and
                            'val_loss' in mini_history['main'].history and 
                            'val_loss' in mini_history['bonus'].history):
                            
                            # Evaluate using validation loss as score
                            main_val_loss = min(mini_history['main'].history['val_loss'])
                            bonus_val_loss = min(mini_history['bonus'].history['val_loss'])
                            
                            # Combined score (weighted average)
                            score = -(main_val_loss * 0.6 + bonus_val_loss * 0.4)  # Negative because lower loss is better
                            
                            logger.info(f"Trial score: {score}")
                            successful_trials += 1
                            
                            if score > best_score:
                                best_score = score
                                best_params = params.copy()
                                logger.info(f"New best parameters found: {best_params}, score: {score}")
                        else:
                            logger.warning(f"Trial {trial+1} did not produce valid history data")
                    except Exception as e:
                        logger.error(f"Error during trial training: {e}")
                        logger.error(traceback.format_exc())
                else:
                    logger.warning(f"Trial {trial+1} failed to prepare data")
            except Exception as e:
                logger.error(f"Error during trial data preparation: {e}")
                logger.error(traceback.format_exc())
        
        # Restore original parameters
        for param, value in original_params.items():
            setattr(self, param, value)
        
        # If no successful trials, use fallback parameters
        if successful_trials == 0 or not best_params:
            logger.warning("No successful trials completed. Using fallback parameters.")
            best_params = fallback_params
            best_score = 0  # Neutral score for fallback
        
        logger.info(f"Hyperparameter search complete. Best parameters: {best_params}")
        
        # Save results to file
        try:
            with open('hyperparameter_search_results.json', 'w') as f:
                json.dump({
                    'best_params': best_params,
                    'best_score': float(best_score),
                    'successful_trials': successful_trials,
                    'total_trials': trials,
                    'search_date': datetime.now().isoformat()
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving hyperparameter results: {e}")
        
        return best_params

    def predict_next_drawing(self, recent_numbers, num_simulations=1000):
        """
        Predict the next drawing with improved ensemble approach
        
        Args:
            recent_numbers: Recent lottery numbers
            num_simulations: Number of simulations to run
            
        Returns:
            Dictionary with predicted main and bonus numbers, and probability distributions
        """
        # Check if we have ensemble models available
        has_ensemble = hasattr(self, 'ensemble') and self.ensemble is not None and \
                       'main_models' in self.ensemble and 'bonus_models' in self.ensemble and \
                       len(self.ensemble['main_models']) > 0 and len(self.ensemble['bonus_models']) > 0
        
        if not has_ensemble and (self.main_model is None or self.bonus_model is None):
            logger.error("Models not trained or loaded yet")
            return None
        
        # Require at least some minimum data
        min_required = max(self.sequence_length, 10)
        if len(recent_numbers) < min_required:
            logger.error(f"Need at least {min_required} recent numbers")
            return None
        
        try:
            # Separate main and bonus numbers
            main_numbers, bonus_numbers, _ = self.separate_main_and_bonus(recent_numbers)
            
            # Create features
            main_features, _ = self.create_features(main_numbers)
            bonus_features, _ = self.create_features(bonus_numbers)
            
            if len(main_features) == 0 or len(bonus_features) == 0:
                logger.error("Failed to create input features for prediction")
                return None
            
            # Use the last sample for prediction
            main_input = main_features[-1].reshape(1, -1)
            bonus_input = bonus_features[-1].reshape(1, -1)
            
            # Scale inputs
            main_input_scaled = self.scaler_X_main.transform(main_input)
            bonus_input_scaled = self.scaler_X_bonus.transform(bonus_input)
            
            # Reshape for model
            if has_ensemble:
                # Get input shape from first model
                main_input_shape = self.ensemble['main_models'][0].input_shape[1:]
                bonus_input_shape = self.ensemble['bonus_models'][0].input_shape[1:]
            else:
                main_input_shape = self.main_model.input_shape[1:]
                bonus_input_shape = self.bonus_model.input_shape[1:]
            
            main_input_scaled = main_input_scaled.reshape(1, main_input_shape[0], main_input_shape[1])
            bonus_input_scaled = bonus_input_scaled.reshape(1, bonus_input_shape[0], bonus_input_shape[1])
            
            # Predict with ensemble or single model approach
            if has_ensemble:
                logger.info("Using ensemble prediction with Monte Carlo dropout")
                main_predictions, main_probs = self.predict_with_ensemble(
                    main_input_scaled, 
                    self.ensemble['main_models'], 
                    self.scaler_y_main, 
                    num_to_predict=self.main_numbers, 
                    is_bonus=False,
                    num_simulations=num_simulations
                )
                
                bonus_predictions, bonus_probs = self.predict_with_ensemble(
                    bonus_input_scaled, 
                    self.ensemble['bonus_models'], 
                    self.scaler_y_bonus, 
                    num_to_predict=self.bonus_numbers, 
                    is_bonus=True,
                    num_simulations=num_simulations
                )
            else:
                logger.info("Using single model prediction with Monte Carlo dropout")
                main_predictions, main_probs = self._ensemble_predict(
                    main_input_scaled, 
                    self.main_model, 
                    self.scaler_y_main, 
                    num_to_predict=self.main_numbers, 
                    is_bonus=False,
                    num_simulations=num_simulations
                )
                
                bonus_predictions, bonus_probs = self._ensemble_predict(
                    bonus_input_scaled, 
                    self.bonus_model, 
                    self.scaler_y_bonus, 
                    num_to_predict=self.bonus_numbers, 
                    is_bonus=True,
                    num_simulations=num_simulations
                )
            
            # Save prediction to metadata
            prediction_entry = {
                'date': datetime.now().isoformat(),
                'main_numbers': main_predictions,
                'bonus_numbers': bonus_predictions
            }
            self.metadata['predictions'].append(prediction_entry)
            
            # Update metadata file
            with open(os.path.join(self.model_dir, 'model_metadata.json'), 'w') as f:
                json.dump(self.metadata, f, indent=2)
            
            return {
                'main': main_predictions,
                'bonus': bonus_predictions,
                'main_probs': main_probs,
                'bonus_probs': bonus_probs
            }
                
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            logger.error(traceback.format_exc())
            return None

    def _ensemble_predict(self, input_scaled, model, scaler_y, num_to_predict=5, is_bonus=False, num_simulations=1000):
        """
        Improved prediction with ensemble approach, Monte Carlo dropout, and statistical combination - fixed to avoid retracing
        
        Args:
            input_scaled: Scaled input features
            model: Model to use for prediction
            scaler_y: Target scaler
            num_to_predict: Number of predictions to make
            is_bonus: Whether this is for bonus numbers
            num_simulations: Number of simulations
            
        Returns:
            List of predicted numbers and probabilities
        """
        # Set valid range
        min_num = self.bonus_min_number if is_bonus else self.main_min_number
        max_num = self.bonus_max_number if is_bonus else self.main_max_number
        
        # Create a function that enables dropout during inference - MOVED OUTSIDE LOOP
        @tf.function(reduce_retracing=True)
        def predict_with_dropout(x, training=True):
            return model(x, training=training)
        
        # Run multiple predictions with various noise levels and dropout
        all_preds = []
        
        # Different noise scales for different simulations
        noise_scales = np.linspace(0.01, 0.08, 5)
        
        for _ in range(num_simulations):
            # Pick a random noise scale
            noise_scale = np.random.choice(noise_scales)
            
            # Add noise to input
            noise = np.random.normal(0, noise_scale, input_scaled.shape)
            noisy_input = input_scaled + noise
            
            # Predict with dropout enabled
            pred_scaled = predict_with_dropout(noisy_input, training=True).numpy()
            
            # Transform back to original scale
            pred = scaler_y.inverse_transform(pred_scaled)[0][0]
            
            # Round and ensure valid range
            pred_round = int(round(pred))
            pred_round = max(min_num, min(max_num, pred_round))
            
            all_preds.append(pred_round)
        
        # Calculate probability for each possible number
        number_counts = Counter(all_preds)
        
        # Get all possible numbers in the valid range
        all_possible = list(range(min_num, max_num + 1))
        
        # Ensure every possible number has a count (even if zero)
        for num in all_possible:
            if num not in number_counts:
                number_counts[num] = 0
        
        # Convert to probabilities with Bayesian smoothing
        total = sum(number_counts.values())
        prior = 1.0 / len(all_possible)  # Uniform prior
        
        smoothed_probs = {}
        for num in all_possible:
            # Add a small prior to avoid overconfidence
            smoothed_probs[num] = (number_counts[num] + prior) / (total + 1) 
        
        # Get most likely numbers
        predicted_numbers = [num for num, _ in number_counts.most_common(num_to_predict)]
        
        # Ensure we have enough unique numbers
        if len(predicted_numbers) < num_to_predict:
            remaining = [num for num in all_possible if num not in predicted_numbers]
            # Add randomly from remaining until we have enough
            np.random.shuffle(remaining)
            predicted_numbers.extend(remaining[:num_to_predict - len(predicted_numbers)])
        
        # Sort numbers
        predicted_numbers.sort()
        
        return predicted_numbers, smoothed_probs
    
    def predict_with_ensemble(self, input_scaled, models, scaler_y, num_to_predict=5, is_bonus=False, num_simulations=500):
        """
        Make predictions using an ensemble of models with Monte Carlo dropout - improved to avoid tf.function retracing
        
        Args:
            input_scaled: Scaled input features
            models: List of trained models to use
            scaler_y: Target scaler
            num_to_predict: Number of predictions to make
            is_bonus: Whether this is for bonus numbers
            num_simulations: Number of Monte Carlo simulations per model
            
        Returns:
            List of predicted numbers and probabilities
        """
        # Set valid range
        min_num = self.bonus_min_number if is_bonus else self.main_min_number
        max_num = self.bonus_max_number if is_bonus else self.main_max_number
        
        # Collect all predictions
        all_predictions = []
        
        # Different noise scales for various simulations
        noise_scales = np.linspace(0.01, 0.08, 5)
        
        # Define prediction function with dropout OUTSIDE the loop - FIX FOR RETRACING ISSUE
        @tf.function(reduce_retracing=True)
        def predict_with_dropout(model, x, training=True):
            return model(x, training=training)
        
        # Make predictions with each model
        for model_idx, model in enumerate(models):
            logger.debug(f"Making predictions with {'bonus' if is_bonus else 'main'} model {model_idx+1}/{len(models)}")
            
            # Base prediction
            pred_scaled = model.predict(input_scaled, verbose=0)
            pred = scaler_y.inverse_transform(pred_scaled)[0][0]
            pred_round = int(round(pred))
            pred_round = max(min_num, min(max_num, pred_round))
            all_predictions.append(pred_round)
            
            # Add Monte Carlo dropout predictions for each model
            simulations_per_model = num_simulations // len(models)
            
            for _ in range(simulations_per_model):
                # Pick a random noise scale
                noise_scale = np.random.choice(noise_scales)
                
                # Add noise to input
                noise = np.random.normal(0, noise_scale, input_scaled.shape)
                noisy_input = input_scaled + noise
                
                # Predict with dropout enabled - using the function defined outside the loop
                mc_pred_scaled = predict_with_dropout(model, noisy_input, training=True).numpy()
                mc_pred = scaler_y.inverse_transform(mc_pred_scaled)[0][0]
                mc_pred_round = int(round(mc_pred))
                mc_pred_round = max(min_num, min(max_num, mc_pred_round))
                all_predictions.append(mc_pred_round)
        
        # Calculate probability for each possible number
        number_counts = Counter(all_predictions)
        
        # Get all possible numbers in the valid range
        all_possible = list(range(min_num, max_num + 1))
        
        # Ensure every possible number has a count (even if zero)
        for num in all_possible:
            if num not in number_counts:
                number_counts[num] = 0
        
        # Convert to probabilities
        total = sum(number_counts.values())
        probabilities = {num: count/total for num, count in number_counts.items()}
        
        # Apply Bayesian smoothing to avoid overconfidence
        prior = 1.0 / len(all_possible)  # Uniform prior
        smoothed_probs = {}
        
        for num in all_possible:
            # Smoothed probability formula (weighted average with prior)
            smoothed_probs[num] = (number_counts[num] + prior) / (total + 1)
        
        # Get most likely numbers
        predicted_numbers = [num for num, _ in number_counts.most_common(num_to_predict)]
        
        # Ensure we have enough unique numbers
        if len(predicted_numbers) < num_to_predict:
            remaining = [num for num in all_possible if num not in predicted_numbers]
            # Add randomly from remaining until we have enough
            np.random.shuffle(remaining)
            predicted_numbers.extend(remaining[:num_to_predict - len(predicted_numbers)])
        
        # Sort numbers
        predicted_numbers.sort()
        
        return predicted_numbers, smoothed_probs

    def plot_learning_curves(self, history):
        """
        Plot learning curves with improved styling and metrics
        
        Args:
            history: Dictionary containing training history for main and bonus models
        """
        # Set up plot style
        plt.style.use('seaborn-darkgrid')

        
        # Create figure for learning curves
        fig, axes = plt.subplots(2, 2, figsize=(15, 10), dpi=100)
        
        # Plot main model loss
        axes[0, 0].plot(history['main'].history['loss'], label='Training', color='#3498db', linewidth=2)
        axes[0, 0].plot(history['main'].history['val_loss'], label='Validation', color='#e74c3c', linewidth=2)
        axes[0, 0].set_title('Main Numbers Model Loss', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch', fontsize=12)
        axes[0, 0].set_ylabel('Loss', fontsize=12)
        axes[0, 0].legend(fontsize=10)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot main model MAE
        axes[0, 1].plot(history['main'].history['mae'], label='Training', color='#3498db', linewidth=2)
        axes[0, 1].plot(history['main'].history['val_mae'], label='Validation', color='#e74c3c', linewidth=2)
        axes[0, 1].set_title('Main Numbers Model MAE', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch', fontsize=12)
        axes[0, 1].set_ylabel('MAE', fontsize=12)
        axes[0, 1].legend(fontsize=10)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot bonus model loss
        axes[1, 0].plot(history['bonus'].history['loss'], label='Training', color='#2ecc71', linewidth=2)
        axes[1, 0].plot(history['bonus'].history['val_loss'], label='Validation', color='#e74c3c', linewidth=2)
        axes[1, 0].set_title('Bonus Numbers Model Loss', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch', fontsize=12)
        axes[1, 0].set_ylabel('Loss', fontsize=12)
        axes[1, 0].legend(fontsize=10)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot bonus model MAE
        axes[1, 1].plot(history['bonus'].history['mae'], label='Training', color='#2ecc71', linewidth=2)
        axes[1, 1].plot(history['bonus'].history['val_mae'], label='Validation', color='#e74c3c', linewidth=2)
        axes[1, 1].set_title('Bonus Numbers Model MAE', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch', fontsize=12)
        axes[1, 1].set_ylabel('MAE', fontsize=12)
        axes[1, 1].legend(fontsize=10)
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add overall title
        plt.suptitle('Model Training Performance', fontsize=16, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        
        # Save figure
        figure_path = os.path.join(self.model_dir, 'learning_curves.png')
        plt.savefig(figure_path, dpi=120)
        logger.info(f"Learning curves saved to {figure_path}")
        
        # Also save a summary of final metrics
        final_epochs_main = len(history['main'].history['loss'])
        final_epochs_bonus = len(history['bonus'].history['loss'])
        
        with open(os.path.join(self.model_dir, 'training_summary.txt'), 'w') as f:
            f.write("===== TRAINING SUMMARY =====\n\n")
            f.write(f"Main Numbers Model (epochs: {final_epochs_main})\n")
            f.write(f"  Final training loss: {history['main'].history['loss'][-1]:.4f}\n")
            f.write(f"  Final validation loss: {history['main'].history['val_loss'][-1]:.4f}\n")
            f.write(f"  Final training MAE: {history['main'].history['mae'][-1]:.4f}\n")
            f.write(f"  Final validation MAE: {history['main'].history['val_mae'][-1]:.4f}\n\n")
            
            f.write(f"Bonus Numbers Model (epochs: {final_epochs_bonus})\n")
            f.write(f"  Final training loss: {history['bonus'].history['loss'][-1]:.4f}\n")
            f.write(f"  Final validation loss: {history['bonus'].history['val_loss'][-1]:.4f}\n")
            f.write(f"  Final training MAE: {history['bonus'].history['mae'][-1]:.4f}\n")
            f.write(f"  Final validation MAE: {history['bonus'].history['val_mae'][-1]:.4f}\n")

    def create_frequency_heatmap(self, draws, title="Number Frequency Analysis"):
        """
        Create an improved heatmap visualizing number frequency patterns
        
        Args:
            draws: List of historical draws
            title: Plot title
        """
        # Extract all numbers
        main_nums = []
        bonus_nums = []
        
        for draw in draws:
            main_nums.extend(draw['main'])
            bonus_nums.extend(draw['bonus'])
        
        # Count frequencies
        main_counts = Counter(main_nums)
        bonus_counts = Counter(bonus_nums)
        
        # Calculate statistics
        main_mean = np.mean(list(main_counts.values()))
        main_std = np.std(list(main_counts.values()))
        bonus_mean = np.mean(list(bonus_counts.values()))
        bonus_std = np.std(list(bonus_counts.values()))
        
        # Prepare matrices for visualization
        main_matrix = np.zeros((5, 10))
        
        for num in range(1, 51):
            row = (num - 1) // 10
            col = (num - 1) % 10
            main_matrix[row, col] = main_counts.get(num, 0)
        
        bonus_matrix = np.zeros((2, 6))
        
        for num in range(1, 13):
            row = (num - 1) // 6
            col = (num - 1) % 6
            bonus_matrix[row, col] = bonus_counts.get(num, 0)
        
        # Calculate z-scores to find statistical outliers
        main_zscores = np.zeros_like(main_matrix)
        for i in range(main_matrix.shape[0]):
            for j in range(main_matrix.shape[1]):
                val = main_matrix[i, j]
                main_zscores[i, j] = (val - main_mean) / (main_std if main_std > 0 else 1)
        
        bonus_zscores = np.zeros_like(bonus_matrix)
        for i in range(bonus_matrix.shape[0]):
            for j in range(bonus_matrix.shape[1]):
                val = bonus_matrix[i, j]
                bonus_zscores[i, j] = (val - bonus_mean) / (bonus_std if bonus_std > 0 else 1)
        
        # Set up plot style
        plt.style.use('seaborn-darkgrid')

        
        # Create figure with custom style
        fig, axes = plt.subplots(3, 1, figsize=(14, 14), 
                                 gridspec_kw={'height_ratios': [3, 1, 0.5]})
        
        # Create custom labels
        main_row_labels = [f"{i*10+1}-{i*10+10}" for i in range(5)]
        main_col_labels = [str(i+1) for i in range(10)]
        
        bonus_row_labels = ["1-6", "7-12"]
        bonus_col_labels = [str(i+1) for i in range(6)]
        
        # Calculate max value for color normalization
        main_max = np.max(main_matrix)
        bonus_max = np.max(bonus_matrix)
        
        # Main numbers heatmap with annotations
        sns.heatmap(main_matrix, 
                    annot=True, 
                    fmt=".0f", 
                    cmap="YlGnBu", 
                    ax=axes[0],
                    cbar_kws={"label": "Frequency"})
        
        axes[0].set_title("Main Numbers Frequency Distribution", fontsize=16, pad=20, fontweight='bold')
        axes[0].set_yticklabels(main_row_labels, fontsize=10, rotation=0)
        axes[0].set_xticklabels(main_col_labels, fontsize=10)
        
        # Mark statistical outliers
        for i in range(main_matrix.shape[0]):
            for j in range(main_matrix.shape[1]):
                zscore = main_zscores[i, j]
                if abs(zscore) > 1.5:  # Significant deviation
                    color = 'red' if zscore > 0 else 'blue'
                    axes[0].add_patch(plt.Rectangle((j, i), 1, 1, fill=False, 
                                                  edgecolor=color, lw=2))
        
        # Bonus numbers heatmap
        sns.heatmap(bonus_matrix, 
                    annot=True, 
                    fmt=".0f", 
                    cmap="YlOrRd", 
                    ax=axes[1],
                    cbar_kws={"label": "Frequency"})
        
        axes[1].set_title("Bonus Numbers Frequency Distribution", fontsize=16, pad=20, fontweight='bold')
        axes[1].set_yticklabels(bonus_row_labels, fontsize=10, rotation=0)
        axes[1].set_xticklabels(bonus_col_labels, fontsize=10)
        
        # Mark statistical outliers for bonus
        for i in range(bonus_matrix.shape[0]):
            for j in range(bonus_matrix.shape[1]):
                zscore = bonus_zscores[i, j]
                if abs(zscore) > 1.5:  # Significant deviation
                    color = 'red' if zscore > 0 else 'blue'
                    axes[1].add_patch(plt.Rectangle((j, i), 1, 1, fill=False, 
                                                  edgecolor=color, lw=2))
        
        # Add a text explanation in the third subplot
        axes[2].axis('off')
        explanation = (
            "This heatmap shows the frequency distribution of lottery numbers. "
            "Numbers with red borders appear significantly more frequently than average, "
            "while numbers with blue borders appear significantly less frequently. "
            "However, past frequency is not necessarily predictive of future draws, "
            "as lottery drawings are designed to be random."
        )
        axes[2].text(0.5, 0.5, explanation, 
                    ha='center', va='center', 
                    fontsize=12, 
                    wrap=True,
                    bbox=dict(facecolor='white', alpha=0.8, 
                             boxstyle='round,pad=1'))
        
        plt.tight_layout()
        
        # Save figure
        figure_path = os.path.join(self.model_dir, 'frequency_heatmap.png')
        plt.savefig(figure_path, dpi=120, bbox_inches='tight')
        logger.info(f"Frequency heatmap saved to {figure_path}")
        
        # Save statistical summary
        with open(os.path.join(self.model_dir, 'frequency_stats.txt'), 'w') as f:
            f.write("===== NUMBER FREQUENCY STATISTICS =====\n\n")
            f.write("Main Numbers:\n")
            f.write(f"  Average frequency: {main_mean:.2f}\n")
            f.write(f"  Standard deviation: {main_std:.2f}\n")
            
            # Most and least frequent main numbers
            most_frequent_main = main_counts.most_common(5)
            least_frequent_main = main_counts.most_common()[:-6:-1]
            
            f.write("\n  Most frequent main numbers:\n")
            for num, freq in most_frequent_main:
                zscore = (freq - main_mean) / (main_std if main_std > 0 else 1)
                f.write(f"    {num}: {freq} occurrences (z-score: {zscore:.2f})\n")
                
            f.write("\n  Least frequent main numbers:\n")
            for num, freq in least_frequent_main:
                zscore = (freq - main_mean) / (main_std if main_std > 0 else 1)
                f.write(f"    {num}: {freq} occurrences (z-score: {zscore:.2f})\n")
            
            f.write("\nBonus Numbers:\n")
            f.write(f"  Average frequency: {bonus_mean:.2f}\n")
            f.write(f"  Standard deviation: {bonus_std:.2f}\n")
            
            # Most and least frequent bonus numbers
            most_frequent_bonus = bonus_counts.most_common(3)
            least_frequent_bonus = bonus_counts.most_common()[:-4:-1]
            
            f.write("\n  Most frequent bonus numbers:\n")
            for num, freq in most_frequent_bonus:
                zscore = (freq - bonus_mean) / (bonus_std if bonus_std > 0 else 1)
                f.write(f"    {num}: {freq} occurrences (z-score: {zscore:.2f})\n")
                
            f.write("\n  Least frequent bonus numbers:\n")
            for num, freq in least_frequent_bonus:
                zscore = (freq - bonus_mean) / (bonus_std if bonus_std > 0 else 1)
                f.write(f"    {num}: {freq} occurrences (z-score: {zscore:.2f})\n")
            
            f.write("\nNOTE: Past frequency patterns do not necessarily predict future draws.\n")
            f.write("Lottery drawings are designed to be random and independent events.\n")
        
        return {
            'main_stats': {
                'mean': main_mean,
                'std': main_std,
                'most_frequent': most_frequent_main,
                'least_frequent': least_frequent_main
            },
            'bonus_stats': {
                'mean': bonus_mean,
                'std': bonus_std,
                'most_frequent': most_frequent_bonus,
                'least_frequent': least_frequent_bonus
            }
        }

    def visualize_predictions(self, prediction_data):
        """
        Create high-quality visualizations of prediction probabilities
        
        Args:
            prediction_data: Dictionary with prediction data from predict_next_drawing
            
        Returns:
            Dictionary with paths to visualization files
        """
        if not prediction_data:
            logger.error("No prediction data to visualize")
            return None
        
        # Unpack prediction data
        main_numbers = prediction_data['main']
        bonus_numbers = prediction_data['bonus']
        main_probs = prediction_data['main_probs']
        bonus_probs = prediction_data['bonus_probs']
        
        # Set up plot style
        plt.style.use('seaborn-darkgrid')
        
        # Create figure for probabilities
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), dpi=100)
        
        # Main numbers probability distribution - improved visualization
        main_nums = sorted(main_probs.keys())
        main_values = [main_probs[num] for num in main_nums]
        
        # Set color based on probability (higher = darker)
        main_colors = plt.cm.Blues(np.array(main_values) / max(main_values))
        
        # Highlight selected numbers with different color
        bars = ax1.bar(main_nums, main_values, color=main_colors, alpha=0.7)
        
        # Mark selected numbers
        for i, num in enumerate(main_nums):
            if num in main_numbers:
                bars[i].set_color('#3498db')
                bars[i].set_alpha(1.0)
                ax1.text(num, main_probs[num] + 0.01, '✓', 
                        ha='center', fontsize=12, fontweight='bold')
        
        ax1.set_title('Main Numbers Probability Distribution', fontsize=16, fontweight='bold')
        ax1.set_xlabel('Number', fontsize=12)
        ax1.set_ylabel('Probability', fontsize=12)
        ax1.set_xticks(np.arange(self.main_min_number, self.main_max_number+1, 5))
        ax1.set_xlim(self.main_min_number-0.5, self.main_max_number+0.5)
        ax1.grid(axis='y', linestyle='--', alpha=0.3)
        
        # Add horizontal line for uniform probability
        uniform_prob = 1.0 / (self.main_max_number - self.main_min_number + 1)
        ax1.axhline(y=uniform_prob, color='red', linestyle='--', alpha=0.5, 
                label=f'Uniform ({uniform_prob:.4f})')
        
        # Add mean probability line
        mean_prob = np.mean(list(main_probs.values()))
        ax1.axhline(y=mean_prob, color='green', linestyle='-.', alpha=0.5,
                label=f'Mean ({mean_prob:.4f})')
        
        ax1.legend()
        
        # Bonus numbers probability distribution
        bonus_nums = sorted(bonus_probs.keys())
        bonus_values = [bonus_probs[num] for num in bonus_nums]
        
        # Set color based on probability (higher = darker)
        bonus_colors = plt.cm.Reds(np.array(bonus_values) / max(bonus_values))
        
        # Highlight selected numbers with different color
        bars = ax2.bar(bonus_nums, bonus_values, color=bonus_colors, alpha=0.7)
        
        # Mark selected numbers
        for i, num in enumerate(bonus_nums):
            if num in bonus_numbers:
                bars[i].set_color('#e74c3c')
                bars[i].set_alpha(1.0)
                ax2.text(num, bonus_probs[num] + 0.01, '✓', 
                        ha='center', fontsize=12, fontweight='bold')
        
        ax2.set_title('Bonus Numbers Probability Distribution', fontsize=16, fontweight='bold')
        ax2.set_xlabel('Number', fontsize=12)
        ax2.set_ylabel('Probability', fontsize=12)
        ax2.set_xticks(np.arange(self.bonus_min_number, self.bonus_max_number+1, 1))
        ax2.set_xlim(self.bonus_min_number-0.5, self.bonus_max_number+0.5)
        ax2.grid(axis='y', linestyle='--', alpha=0.3)
        
        # Add horizontal line for uniform probability
        uniform_prob = 1.0 / (self.bonus_max_number - self.bonus_min_number + 1)
        ax2.axhline(y=uniform_prob, color='red', linestyle='--', alpha=0.5,
                label=f'Uniform ({uniform_prob:.4f})')
        
        # Add mean probability line
        mean_prob = np.mean(list(bonus_probs.values()))
        ax2.axhline(y=mean_prob, color='green', linestyle='-.', alpha=0.5,
                label=f'Mean ({mean_prob:.4f})')
        
        ax2.legend()
        
        # Add overall title with prediction numbers
        main_str = ', '.join(map(str, main_numbers))
        bonus_str = ', '.join(map(str, bonus_numbers))
        plt.suptitle(f'Predicted Drawing: Main [{main_str}] Bonus [{bonus_str}]', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        
        # Save figure
        prob_figure_path = os.path.join(self.model_dir, 'prediction_probabilities.png')
        plt.savefig(prob_figure_path, dpi=120)
        logger.info(f"Prediction probabilities saved to {prob_figure_path}")
        
        # Create additional visualization - Heatmap of probabilities
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7), dpi=100,
                                    gridspec_kw={'width_ratios': [4, 1]})
        
        # Create heatmap data for main numbers
        main_heatmap = np.zeros((5, 10))
        for num in range(1, 51):
            row = (num - 1) // 10
            col = (num - 1) % 10
            main_heatmap[row, col] = main_probs.get(num, 0)
        
        # Create heatmap for bonus numbers
        bonus_heatmap = np.zeros((2, 6))
        for num in range(1, 13):
            row = (num - 1) // 6
            col = (num - 1) % 6
            bonus_heatmap[row, col] = bonus_probs.get(num, 0)
        
        # Create pre-formatted annotation matrices
        main_annot = np.empty_like(main_heatmap, dtype=object)
        for i in range(main_heatmap.shape[0]):
            for j in range(main_heatmap.shape[1]):
                val = main_heatmap[i, j]
                main_annot[i, j] = f'{val*100:.1f}%' if val > 0.01 else ''
        
        bonus_annot = np.empty_like(bonus_heatmap, dtype=object)
        for i in range(bonus_heatmap.shape[0]):
            for j in range(bonus_heatmap.shape[1]):
                val = bonus_heatmap[i, j]
                bonus_annot[i, j] = f'{val*100:.1f}%' if val > 0.01 else ''
        
        # Main numbers heatmap
        sns.heatmap(main_heatmap, 
                annot=main_annot, 
                fmt='',  # Empty since values are pre-formatted
                cmap="YlGnBu", 
                ax=ax1,
                cbar_kws={"label": "Probability"})
        
        ax1.set_title("Main Numbers Probability Heatmap", fontsize=14, fontweight='bold')
        
        # Add row and column labels
        ax1.set_yticklabels([f"{i*10+1}-{i*10+10}" for i in range(5)], fontsize=10, rotation=0)
        ax1.set_xticklabels([str(i+1) for i in range(10)], fontsize=10)
        
        # Mark predicted numbers on heatmap
        for num in main_numbers:
            row = (num - 1) // 10
            col = (num - 1) % 10
            ax1.add_patch(plt.Rectangle((col, row), 1, 1, fill=False, 
                                    edgecolor='red', lw=2))
        
        # Bonus numbers heatmap
        sns.heatmap(bonus_heatmap, 
                annot=bonus_annot, 
                fmt='',  # Empty since values are pre-formatted
                cmap="YlOrRd", 
                ax=ax2,
                cbar_kws={"label": "Probability"})
        
        ax2.set_title("Bonus Numbers Probability Heatmap", fontsize=14, fontweight='bold')
        
        # Add row and column labels
        ax2.set_yticklabels(["1-6", "7-12"], fontsize=10, rotation=0)
        ax2.set_xticklabels([str(i+1) for i in range(6)], fontsize=10)
        
        # Mark predicted numbers on heatmap
        for num in bonus_numbers:
            row = (num - 1) // 6
            col = (num - 1) % 6
            ax2.add_patch(plt.Rectangle((col, row), 1, 1, fill=False, 
                                    edgecolor='red', lw=2))
        
        plt.tight_layout()
        
        # Save heatmap figure
        heatmap_path = os.path.join(self.model_dir, 'prediction_heatmap.png')
        plt.savefig(heatmap_path, dpi=120)
        logger.info(f"Prediction heatmap saved to {heatmap_path}")
        
        return {
            'probabilities_chart': prob_figure_path,
            'heatmap_chart': heatmap_path
        }

    def validate_with_backtesting(self, test_draws, num_backtest=10):
        """
        Improved validation with more robust backtesting and metrics
        
        Args:
            test_draws: List of historical draws to test against
            num_backtest: Number of historical draws to backtest
            
        Returns:
            Dictionary with detailed validation metrics
        """
        logger.info(f"Performing enhanced backtesting on {num_backtest} historical draws")
        
        if len(test_draws) < num_backtest + 5:  # Need some extra draws for context
            logger.warning(f"Not enough historical draws for robust backtesting, using {len(test_draws)} draws")
            num_backtest = max(1, len(test_draws) - 5)
        
        # Convert draws to flat list for prediction
        flat_numbers = []
        for draw in test_draws:
            flat_numbers.extend(draw['main'] + draw['bonus'])
        
        # Store results for each backtested prediction
        results = []
        
        # Calculate baseline metrics for random prediction
        random_metrics = self._calculate_random_baseline(num_backtest)
        
        # Perform backtesting with progress updates
        for i in range(num_backtest):
            logger.info(f"Backtesting draw {i+1} of {num_backtest}")
            
            # Calculate cut-off index for this prediction
            cutoff = len(flat_numbers) - (i + 1) * self.numbers_per_draw
            
            if cutoff <= self.sequence_length * 3:  # Need enough data for prediction
                logger.warning(f"Not enough data for prediction at step {i}")
                continue
            
            # Get numbers up to cutoff for prediction
            history = flat_numbers[:cutoff]
            
            # Get actual draw that happened
            actual_draw = {
                'main': flat_numbers[cutoff:cutoff+self.main_numbers],
                'bonus': flat_numbers[cutoff+self.main_numbers:cutoff+self.numbers_per_draw]
            }
            
            # Make prediction
            try:
                prediction = self.predict_next_drawing(history)
                
                if prediction:
                    # Calculate metrics
                    main_matches = len(set(prediction['main']) & set(actual_draw['main']))
                    bonus_matches = len(set(prediction['bonus']) & set(actual_draw['bonus']))
                    
                    # Calculate precision (percentage of predicted numbers that were correct)
                    main_precision = main_matches / len(prediction['main']) if prediction['main'] else 0
                    bonus_precision = bonus_matches / len(prediction['bonus']) if prediction['bonus'] else 0
                    
                    # Calculate recall (percentage of actual numbers that were predicted)
                    main_recall = main_matches / len(actual_draw['main']) if actual_draw['main'] else 0
                    bonus_recall = bonus_matches / len(actual_draw['bonus']) if actual_draw['bonus'] else 0
                    
                    # Calculate F1 score (harmonic mean of precision and recall)
                    main_f1 = 2 * (main_precision * main_recall) / (main_precision + main_recall) if (main_precision + main_recall) > 0 else 0
                    bonus_f1 = 2 * (bonus_precision * bonus_recall) / (bonus_precision + bonus_recall) if (bonus_precision + bonus_recall) > 0 else 0
                    
                    # Evaluate probability calibration for main numbers
                    main_prob_score = 0
                    for num in actual_draw['main']:
                        if num in prediction['main_probs']:
                            main_prob_score += prediction['main_probs'][num]
                    
                    # Evaluate probability calibration for bonus numbers
                    bonus_prob_score = 0
                    for num in actual_draw['bonus']:
                        if num in prediction['bonus_probs']:
                            bonus_prob_score += prediction['bonus_probs'][num]
                    
                    # Store detailed results
                    results.append({
                        'backtest_index': i,
                        'prediction': {
                            'main': prediction['main'],
                            'bonus': prediction['bonus']
                        },
                        'actual': actual_draw,
                        'main_matches': main_matches,
                        'bonus_matches': bonus_matches,
                        'main_precision': main_precision,
                        'bonus_precision': bonus_precision,
                        'main_recall': main_recall,
                        'bonus_recall': bonus_recall,
                        'main_f1': main_f1,
                        'bonus_f1': bonus_f1,
                        'main_prob_score': main_prob_score,
                        'bonus_prob_score': bonus_prob_score,
                        'total_score': main_matches + bonus_matches * 1.5  # Bonus matches weighted more
                    })
            except Exception as e:
                logger.error(f"Error during backtesting at step {i}: {e}")
                logger.error(traceback.format_exc())
        
        # Calculate aggregate metrics with confidence intervals
        if results:
            # Calculate all metrics
            metrics = {
                'avg_main_matches': np.mean([r['main_matches'] for r in results]),
                'std_main_matches': np.std([r['main_matches'] for r in results]),
                'avg_bonus_matches': np.mean([r['bonus_matches'] for r in results]),
                'std_bonus_matches': np.std([r['bonus_matches'] for r in results]),
                'avg_main_precision': np.mean([r['main_precision'] for r in results]),
                'avg_bonus_precision': np.mean([r['bonus_precision'] for r in results]),
                'avg_main_recall': np.mean([r['main_recall'] for r in results]),
                'avg_bonus_recall': np.mean([r['bonus_recall'] for r in results]),
                'avg_main_f1': np.mean([r['main_f1'] for r in results]),
                'avg_bonus_f1': np.mean([r['bonus_f1'] for r in results]),
                'avg_main_prob_score': np.mean([r['main_prob_score'] for r in results]),
                'avg_bonus_prob_score': np.mean([r['bonus_prob_score'] for r in results]),
                'avg_score': np.mean([r['total_score'] for r in results]),
                'std_score': np.std([r['total_score'] for r in results]),
                'random_baseline': random_metrics
            }
            
            # Calculate 95% confidence intervals for key metrics
            n = len(results)
            t_value = 1.96  # Approximate 95% CI
            
            metrics['main_matches_ci'] = t_value * metrics['std_main_matches'] / np.sqrt(n)
            metrics['bonus_matches_ci'] = t_value * metrics['std_bonus_matches'] / np.sqrt(n)
            metrics['total_score_ci'] = t_value * metrics['std_score'] / np.sqrt(n)
            
            # Calculate improvement over random with confidence
            if random_metrics['total_score'] > 0:
                improvement = (metrics['avg_score'] / random_metrics['total_score'] - 1) * 100
                metrics['improvement'] = improvement
                
                # Calculate whether improvement is statistically significant
                metrics['significant_improvement'] = (
                    metrics['avg_score'] - metrics['total_score_ci'] > 
                    random_metrics['total_score']
                )
            else:
                metrics['improvement'] = 0
                metrics['significant_improvement'] = False
            
            # Log results
            logger.info(f"Backtesting results (avg over {len(results)} predictions):")
            logger.info(f"  Main numbers - Avg matches: {metrics['avg_main_matches']:.2f} ± {metrics['main_matches_ci']:.2f}/{self.main_numbers} " +
                    f"(Random: {random_metrics['main_matches']:.2f})")
            logger.info(f"  Bonus numbers - Avg matches: {metrics['avg_bonus_matches']:.2f} ± {metrics['bonus_matches_ci']:.2f}/{self.bonus_numbers} " +
                    f"(Random: {random_metrics['bonus_matches']:.2f})")
            logger.info(f"  Overall score: {metrics['avg_score']:.2f} ± {metrics['total_score_ci']:.2f} (Random: {random_metrics['total_score']:.2f})")
            logger.info(f"  Improvement over random: {metrics['improvement']:.1f}%")
            logger.info(f"  Statistically significant improvement: {'Yes' if metrics['significant_improvement'] else 'No'}")
            
            # Save results to JSON
            validation_result = {
                'results': results,
                'metrics': metrics
            }

            # Helper function to convert NumPy types to Python native types
            def numpy_to_python(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.bool_):
                    return bool(obj)
                elif isinstance(obj, dict):
                    return {k: numpy_to_python(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [numpy_to_python(i) for i in obj]
                return obj
            
            with open(os.path.join(self.model_dir, 'validation_results.json'), 'w') as f:
                # Use the helper function to convert NumPy types to Python native types
                json_data = numpy_to_python({
                    'results': results,
                    'metrics': {k: v for k, v in metrics.items() if k != 'random_baseline'},
                    'random_baseline': random_metrics
                })
                json.dump(json_data, f, indent=2)
            
            # Save detailed text report
            with open(os.path.join(self.model_dir, 'validation_report.txt'), 'w') as f:
                f.write("===== BACKTESTING VALIDATION REPORT =====\n\n")
                f.write(f"Number of backtest draws: {len(results)}\n\n")
                
                f.write("MAIN NUMBERS METRICS:\n")
                f.write(f"  Average matches: {metrics['avg_main_matches']:.2f} ± {metrics['main_matches_ci']:.2f} out of {self.main_numbers}\n")
                f.write(f"  Random baseline: {random_metrics['main_matches']:.2f}\n")
                f.write(f"  Precision: {metrics['avg_main_precision']:.2f}\n")
                f.write(f"  Recall: {metrics['avg_main_recall']:.2f}\n")
                f.write(f"  F1 Score: {metrics['avg_main_f1']:.2f}\n")
                f.write(f"  Probability calibration score: {metrics['avg_main_prob_score']:.3f}\n\n")
                
                f.write("BONUS NUMBERS METRICS:\n")
                f.write(f"  Average matches: {metrics['avg_bonus_matches']:.2f} ± {metrics['bonus_matches_ci']:.2f} out of {self.bonus_numbers}\n")
                f.write(f"  Random baseline: {random_metrics['bonus_matches']:.2f}\n")
                f.write(f"  Precision: {metrics['avg_bonus_precision']:.2f}\n")
                f.write(f"  Recall: {metrics['avg_bonus_recall']:.2f}\n")
                f.write(f"  F1 Score: {metrics['avg_bonus_f1']:.2f}\n")
                f.write(f"  Probability calibration score: {metrics['avg_bonus_prob_score']:.3f}\n\n")
                
                f.write("OVERALL PERFORMANCE:\n")
                f.write(f"  Total score: {metrics['avg_score']:.2f} ± {metrics['total_score_ci']:.2f}\n")
                f.write(f"  Random baseline: {random_metrics['total_score']:.2f}\n")
                f.write(f"  Improvement over random: {metrics['improvement']:.1f}%\n")
                f.write(f"  Statistically significant improvement: {'Yes' if metrics['significant_improvement'] else 'No'}\n\n")
                
                f.write("INDIVIDUAL BACKTESTS:\n")
                for i, result in enumerate(results):
                    f.write(f"  Backtest #{i+1}:\n")
                    f.write(f"    Predicted main: {result['prediction']['main']}\n")
                    f.write(f"    Actual main: {result['actual']['main']}\n")
                    f.write(f"    Main matches: {result['main_matches']}\n\n")
                    f.write(f"    Predicted bonus: {result['prediction']['bonus']}\n")
                    f.write(f"    Actual bonus: {result['actual']['bonus']}\n")
                    f.write(f"    Bonus matches: {result['bonus_matches']}\n\n")
                    f.write(f"    Total score: {result['total_score']:.2f}\n\n")
                
                f.write("\nIMPORTANT NOTE: Lottery drawings are designed to be random and unpredictable.\n")
                f.write("Even with advanced modeling techniques, significant long-term prediction accuracy is unlikely.\n")
                f.write("This model should be used for educational purposes only.\n")
            
            # Update metadata with validation results
            self.metadata['performance']['validation'] = {
                'avg_main_matches': float(metrics['avg_main_matches']),
                'avg_bonus_matches': float(metrics['avg_bonus_matches']),
                'avg_score': float(metrics['avg_score']),
                'improvement': float(metrics['improvement']),
                'significant_improvement': bool(metrics['significant_improvement'])
            }
            
            with open(os.path.join(self.model_dir, 'model_metadata.json'), 'w') as f:
                json.dump(numpy_to_python(self.metadata), f, indent=2)
            
            return validation_result
        else:
            logger.error("No valid backtesting results")
            return None

    def _calculate_random_baseline(self, num_trials=10000):
        """
        Calculate baseline metrics for random prediction more accurately
        
        Args:
            num_trials: Number of random trials to simulate
            
        Returns:
            Dictionary with baseline metrics
        """
        # Simulate random draws with Monte Carlo approach
        main_matches_sum = 0
        bonus_matches_sum = 0
        
        for _ in range(num_trials):
            # Random prediction
            pred_main = sorted(np.random.choice(range(self.main_min_number, self.main_max_number+1), 
                                              self.main_numbers, replace=False))
            pred_bonus = sorted(np.random.choice(range(self.bonus_min_number, self.bonus_max_number+1), 
                                               self.bonus_numbers, replace=False))
            
            # Random actual draw
            actual_main = np.random.choice(range(self.main_min_number, self.main_max_number+1), 
                                         self.main_numbers, replace=False)
            actual_bonus = np.random.choice(range(self.bonus_min_number, self.bonus_max_number+1), 
                                          self.bonus_numbers, replace=False)
            
            # Count matches
            main_matches = len(set(pred_main) & set(actual_main))
            bonus_matches = len(set(pred_bonus) & set(actual_bonus))
            
            main_matches_sum += main_matches
            bonus_matches_sum += bonus_matches
        
        # Calculate averages
        avg_main_matches = main_matches_sum / num_trials
        avg_bonus_matches = bonus_matches_sum / num_trials
        avg_score = avg_main_matches + avg_bonus_matches * 1.5
        
        # Calculate theoretical expectations for verification
        # Expected matches = (numbers selected) / (total possible numbers)
        theoretical_main_matches = self.main_numbers * self.main_numbers / (self.main_max_number - self.main_min_number + 1)
        theoretical_bonus_matches = self.bonus_numbers * self.bonus_numbers / (self.bonus_max_number - self.bonus_min_number + 1)
        
        logger.info(f"Random baseline - Monte Carlo simulation ({num_trials} trials):")
        logger.info(f"  Main matches: {avg_main_matches:.4f} (theoretical: {theoretical_main_matches:.4f})")
        logger.info(f"  Bonus matches: {avg_bonus_matches:.4f} (theoretical: {theoretical_bonus_matches:.4f})")
        
        return {
            'main_matches': avg_main_matches,
            'bonus_matches': avg_bonus_matches,
            'total_score': avg_score,
            'theoretical_main': theoretical_main_matches,
            'theoretical_bonus': theoretical_bonus_matches
        }

    def plot_backtesting_results(self, validation_results):
        """
        Create enhanced visualizations of backtesting results
        
        Args:
            validation_results: Results from validate_with_backtesting
        """
        if not validation_results or 'results' not in validation_results or not validation_results['results']:
            logger.error("No backtest results to plot")
            return
        
        # Extract metrics for readability
        metrics = validation_results['metrics']
        random_baseline = metrics['random_baseline']
        
        # Create DataFrame from results
        results = validation_results['results']
        df = pd.DataFrame([{
            'backtest_index': r['backtest_index'],
            'main_matches': r['main_matches'],
            'bonus_matches': r['bonus_matches'],
            'total_score': r['total_score'],
            'main_precision': r['main_precision'],
            'bonus_precision': r['bonus_precision'],
            'main_f1': r['main_f1'],
            'bonus_f1': r['bonus_f1']
        } for r in results])
        
        # Set up plot style
        plt.style.use('seaborn-darkgrid')
        
        # Create comprehensive figure
        fig = plt.figure(figsize=(15, 18), dpi=100)
        
        # Define grid for plots
        gs = fig.add_gridspec(5, 2, height_ratios=[3, 3, 3, 3, 1])
        
        # Plot main matches
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(df['backtest_index'], df['main_matches'], 'o-', color='#3498db', linewidth=2, 
               label='Model')
        ax1.axhline(y=random_baseline['main_matches'], color='gray', linestyle='--', 
                  label=f'Random ({random_baseline["main_matches"]:.2f})')
        
        # Add confidence interval
        ax1.axhline(y=metrics['avg_main_matches'], color='#3498db', linestyle='-', alpha=0.5)
        ax1.fill_between(df['backtest_index'], 
                        metrics['avg_main_matches'] - metrics['main_matches_ci'],
                        metrics['avg_main_matches'] + metrics['main_matches_ci'],
                        color='#3498db', alpha=0.2)
        
        ax1.set_title(f'Main Number Matches (Avg: {metrics["avg_main_matches"]:.2f} ± {metrics["main_matches_ci"]:.2f})', 
                    fontsize=14, fontweight='bold')
        ax1.set_xlabel('Backtest Index', fontsize=12)
        ax1.set_ylabel('Matches', fontsize=12)
        ax1.set_ylim(-0.5, self.main_numbers + 0.5)
        ax1.set_yticks(range(self.main_numbers + 1))
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Plot bonus matches
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(df['backtest_index'], df['bonus_matches'], 'o-', color='#2ecc71', linewidth=2,
               label='Model')
        ax2.axhline(y=random_baseline['bonus_matches'], color='gray', linestyle='--',
                  label=f'Random ({random_baseline["bonus_matches"]:.2f})')
        
        # Add confidence interval
        ax2.axhline(y=metrics['avg_bonus_matches'], color='#2ecc71', linestyle='-', alpha=0.5)
        ax2.fill_between(df['backtest_index'], 
                        metrics['avg_bonus_matches'] - metrics['bonus_matches_ci'],
                        metrics['avg_bonus_matches'] + metrics['bonus_matches_ci'],
                        color='#2ecc71', alpha=0.2)
        
        ax2.set_title(f'Bonus Number Matches (Avg: {metrics["avg_bonus_matches"]:.2f} ± {metrics["bonus_matches_ci"]:.2f})', 
                    fontsize=14, fontweight='bold')
        ax2.set_xlabel('Backtest Index', fontsize=12)
        ax2.set_ylabel('Matches', fontsize=12)
        ax2.set_ylim(-0.5, self.bonus_numbers + 0.5)
        ax2.set_yticks(range(self.bonus_numbers + 1))
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # Plot total score
        ax3 = fig.add_subplot(gs[1, :])
        ax3.plot(df['backtest_index'], df['total_score'], 'o-', color='#9b59b6', linewidth=2,
               label='Model')
        ax3.axhline(y=random_baseline['total_score'], color='gray', linestyle='--',
                  label=f'Random ({random_baseline["total_score"]:.2f})')
        
        # Add confidence interval
        ax3.axhline(y=metrics['avg_score'], color='#9b59b6', linestyle='-', alpha=0.5)
        ax3.fill_between(df['backtest_index'], 
                        metrics['avg_score'] - metrics['total_score_ci'],
                        metrics['avg_score'] + metrics['total_score_ci'],
                        color='#9b59b6', alpha=0.2)
        
        significance_str = "Statistically Significant" if metrics['significant_improvement'] else "Not Statistically Significant"
        
        ax3.set_title(f'Total Score (Avg: {metrics["avg_score"]:.2f} ± {metrics["total_score_ci"]:.2f}, ' + 
                    f'Improvement: {metrics["improvement"]:.1f}% - {significance_str})', 
                    fontsize=14, fontweight='bold')
        ax3.set_xlabel('Backtest Index', fontsize=12)
        ax3.set_ylabel('Score', fontsize=12)
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        # Plot precision metrics
        ax4 = fig.add_subplot(gs[2, 0])
        ax4.plot(df['backtest_index'], df['main_precision'], 'o-', color='#e74c3c', linewidth=2,
               label='Main Numbers')
        ax4.plot(df['backtest_index'], df['bonus_precision'], 's-', color='#f39c12', linewidth=2,
               label='Bonus Numbers')
        
        # Add horizontal lines for average precision
        ax4.axhline(y=metrics['avg_main_precision'], color='#e74c3c', linestyle='--', alpha=0.5)
        ax4.axhline(y=metrics['avg_bonus_precision'], color='#f39c12', linestyle='--', alpha=0.5)
        
        # Add horizontal line for random precision
        main_random_precision = self.main_numbers / (self.main_max_number - self.main_min_number + 1)
        bonus_random_precision = self.bonus_numbers / (self.bonus_max_number - self.bonus_min_number + 1)
        
        ax4.axhline(y=main_random_precision, color='gray', linestyle=':', alpha=0.5,
                  label=f'Random Main ({main_random_precision:.3f})')
        ax4.axhline(y=bonus_random_precision, color='black', linestyle=':', alpha=0.5,
                  label=f'Random Bonus ({bonus_random_precision:.3f})')
        
        ax4.set_title('Precision (% of Predicted Numbers That Were Correct)', 
                    fontsize=14, fontweight='bold')
        ax4.set_xlabel('Backtest Index', fontsize=12)
        ax4.set_ylabel('Precision', fontsize=12)
        ax4.set_ylim(-0.05, 1.05)
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3)
        
        # Plot F1 score metrics
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.plot(df['backtest_index'], df['main_f1'], 'o-', color='#1abc9c', linewidth=2,
               label='Main Numbers')
        ax5.plot(df['backtest_index'], df['bonus_f1'], 's-', color='#f1c40f', linewidth=2,
               label='Bonus Numbers')
        
        # Add horizontal lines for average F1
        ax5.axhline(y=metrics['avg_main_f1'], color='#1abc9c', linestyle='--', alpha=0.5)
        ax5.axhline(y=metrics['avg_bonus_f1'], color='#f1c40f', linestyle='--', alpha=0.5)
        
        ax5.set_title('F1 Score (Harmonic Mean of Precision and Recall)', 
                    fontsize=14, fontweight='bold')
        ax5.set_xlabel('Backtest Index', fontsize=12)
        ax5.set_ylabel('F1 Score', fontsize=12)
        ax5.set_ylim(-0.05, 1.05)
        ax5.legend(fontsize=10)
        ax5.grid(True, alpha=0.3)
        
        # Add distribution comparison
        ax6 = fig.add_subplot(gs[3, :])
        
        # Create histograms comparing model vs random
        model_scores = df['total_score'].values
        
        # Generate random scores for comparison
        np.random.seed(42)  # For reproducibility
        random_scores = []
        for _ in range(1000):  # Generate more samples for smoother distribution
            main_match = np.random.binomial(self.main_numbers, 
                                          self.main_numbers/(self.main_max_number-self.main_min_number+1))
            bonus_match = np.random.binomial(self.bonus_numbers, 
                                           self.bonus_numbers/(self.bonus_max_number-self.bonus_min_number+1))
            random_scores.append(main_match + bonus_match * 1.5)
        
        # Plot histograms
        bins = np.linspace(0, max(np.max(model_scores), np.max(random_scores)) + 0.5, 10)
        ax6.hist(random_scores, bins=bins, alpha=0.5, color='gray', label='Random Model')
        ax6.hist(model_scores, bins=bins, alpha=0.7, color='#9b59b6', label='Enhanced Model')
        
        # Add vertical lines for means
        ax6.axvline(x=np.mean(random_scores), color='gray', linestyle='--', linewidth=2,
                  label=f'Random Mean: {np.mean(random_scores):.2f}')
        ax6.axvline(x=np.mean(model_scores), color='#9b59b6', linestyle='--', linewidth=2,
                  label=f'Model Mean: {np.mean(model_scores):.2f}')
        
        ax6.set_title('Score Distribution Comparison', fontsize=14, fontweight='bold')
        ax6.set_xlabel('Total Score', fontsize=12)
        ax6.set_ylabel('Frequency', fontsize=12)
        ax6.legend(fontsize=10)
        ax6.grid(True, alpha=0.3)
        
        # Add summary text
        ax7 = fig.add_subplot(gs[4, :])
        ax7.axis('off')
        
        # Create informative text box
        summary_text = (
            f"BACKTESTING SUMMARY\n\n"
            f"Main Numbers: Avg {metrics['avg_main_matches']:.2f} matches (random: {random_baseline['main_matches']:.2f})\n"
            f"Bonus Numbers: Avg {metrics['avg_bonus_matches']:.2f} matches (random: {random_baseline['bonus_matches']:.2f})\n"
            f"Overall Score: {metrics['avg_score']:.2f} ± {metrics['total_score_ci']:.2f} "
            f"(random: {random_baseline['total_score']:.2f})\n"
            f"Improvement: {metrics['improvement']:.1f}% - "
            f"{'Statistically Significant' if metrics['significant_improvement'] else 'Not Statistically Significant'}\n\n"
            f"IMPORTANT NOTE: Lottery drawings are designed to be random and unpredictable."
        )
        
        ax7.text(0.5, 0.5, summary_text, ha='center', va='center', fontsize=12,
               bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=1'),
               transform=ax7.transAxes)
        
        # Add overall title
        plt.suptitle('Enhanced Lottery Predictor - Backtesting Results', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        
        # Save figure
        figure_path = os.path.join(self.model_dir, 'backtesting_results.png')
        plt.savefig(figure_path, dpi=120)
        logger.info(f"Backtesting results visualization saved to {figure_path}")
        
        return figure_path

    def generate_prediction_report(self, prediction_data, recent_draws=None):
        """
        Generate a comprehensive prediction report with visualizations
        
        Args:
            prediction_data: Dictionary with prediction data
            recent_draws: Optional list of recent draws for comparison
            
        Returns:
            Path to the generated report
        """
        if not prediction_data:
            logger.error("No prediction data available")
            return None
        
        # Extract prediction information
        main_numbers = prediction_data['main']
        bonus_numbers = prediction_data['bonus']
        
        # Create visualizations
        viz_paths = self.visualize_predictions(prediction_data)
        
        # Create report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(self.model_dir, f'prediction_report_{timestamp}.txt')
        
        with open(report_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("ENHANCED LOTTERY PREDICTION REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("PREDICTION SUMMARY\n")
            f.write("-" * 30 + "\n")
            f.write(f"Main Numbers: {', '.join(map(str, main_numbers))}\n")
            f.write(f"Bonus Numbers: {', '.join(map(str, bonus_numbers))}\n\n")
            
            # Include recent draws if provided
            if recent_draws and len(recent_draws) > 0:
                f.write("RECENT DRAWS\n")
                f.write("-" * 30 + "\n")
                for i, draw in enumerate(recent_draws[-5:]):  # Show last 5 draws
                    f.write(f"Draw #{len(recent_draws)-len(recent_draws[-5:])+i+1}: " + 
                           f"Main: {sorted(draw['main'])}, Bonus: {sorted(draw['bonus'])}\n")
                f.write("\n")
            
            # Include validation information if available
            if hasattr(self, 'metadata') and 'performance' in self.metadata and 'validation' in self.metadata['performance']:
                val = self.metadata['performance']['validation']
                f.write("MODEL VALIDATION\n")
                f.write("-" * 30 + "\n")
                f.write(f"Average Main Matches: {val['avg_main_matches']:.2f}/{self.main_numbers}\n")
                f.write(f"Average Bonus Matches: {val['avg_bonus_matches']:.2f}/{self.bonus_numbers}\n")
                f.write(f"Overall Improvement: {val['improvement']:.1f}%\n")
                f.write(f"Statistically Significant: {'Yes' if val['significant_improvement'] else 'No'}\n\n")
            
            # Include probability summaries
            f.write("PROBABILITY ANALYSIS\n")
            f.write("-" * 30 + "\n")
            
            # Sort numbers by probability
            main_probs = prediction_data['main_probs']
            bonus_probs = prediction_data['bonus_probs']
            
            # Top 10 main numbers by probability
            f.write("Top 10 Main Numbers by Probability:\n")
            sorted_main = sorted(main_probs.items(), key=lambda x: x[1], reverse=True)[:10]
            for num, prob in sorted_main:
                mark = "✓" if num in main_numbers else " "
                f.write(f"  {mark} {num}: {prob*100:.2f}%\n")
                
            f.write("\nAll Bonus Numbers by Probability:\n")
            sorted_bonus = sorted(bonus_probs.items(), key=lambda x: x[1], reverse=True)
            for num, prob in sorted_bonus:
                mark = "✓" if num in bonus_numbers else " "
                f.write(f"  {mark} {num}: {prob*100:.2f}%\n")
            
            f.write("\nVISUALIZATIONS\n")
            f.write("-" * 30 + "\n")
            if viz_paths:
                for name, path in viz_paths.items():
                    f.write(f"{name.replace('_', ' ').title()}: {path}\n")
            
            f.write("\n" + "=" * 60 + "\n")
            f.write("IMPORTANT DISCLAIMER\n")
            f.write("=" * 60 + "\n")
            f.write("This prediction is based on statistical analysis and machine learning techniques.\n")
            f.write("However, lottery drawings are designed to be random and unpredictable events.\n")
            f.write("No model can reliably predict lottery numbers with significant accuracy.\n")
            f.write("This tool is provided for educational and entertainment purposes only.\n")
        
        logger.info(f"Prediction report generated: {report_path}")
        return report_path

    def save_models(self):
        """Save trained models and configuration"""
        if self.main_model and self.bonus_model:
            try:
                # Save models
                main_model_path = os.path.join(self.model_dir, 'main_model.keras')
                bonus_model_path = os.path.join(self.model_dir, 'bonus_model.keras')
                
                self.main_model.save(main_model_path)
                self.bonus_model.save(bonus_model_path)
                
                # Save configuration
                config = {
                    'sequence_length': self.sequence_length,
                    'main_numbers': self.main_numbers,
                    'bonus_numbers': self.bonus_numbers,
                    'main_number_range': (self.main_min_number, self.main_max_number),
                    'bonus_number_range': (self.bonus_min_number, self.bonus_max_number),
                    'embed_dim': self.embed_dim,
                    'num_heads': self.num_heads,
                    'ff_dim': self.ff_dim,
                    'transformer_blocks': self.num_transformer_blocks,
                    'saved_date': datetime.now().isoformat()
                }
                
                with open(os.path.join(self.model_dir, 'model_config.json'), 'w') as f:
                    json.dump(config, f, indent=2)
                
                # Save scalers
                np.save(os.path.join(self.model_dir, 'scaler_X_main.npy'), 
                       self.scaler_X_main.__dict__)
                np.save(os.path.join(self.model_dir, 'scaler_y_main.npy'), 
                       self.scaler_y_main.__dict__)
                np.save(os.path.join(self.model_dir, 'scaler_X_bonus.npy'), 
                       self.scaler_X_bonus.__dict__)
                np.save(os.path.join(self.model_dir, 'scaler_y_bonus.npy'), 
                       self.scaler_y_bonus.__dict__)
                
                logger.info(f"Models and configuration saved to {self.model_dir}")
                return True
            except Exception as e:
                logger.error(f"Error saving models: {e}")
                logger.error(traceback.format_exc())
                return False

    def load_models(self, model_dir):
        """
        Load trained models and configuration
        
        Args:
            model_dir: Directory with saved models
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load configuration
            with open(os.path.join(model_dir, 'model_config.json'), 'r') as f:
                config = json.load(f)
            
            # Update attributes
            self.sequence_length = config['sequence_length']
            self.main_numbers = config['main_numbers']
            self.bonus_numbers = config['bonus_numbers']
            self.main_min_number, self.main_max_number = config['main_number_range']
            self.bonus_min_number, self.bonus_max_number = config['bonus_number_range']
            self.embed_dim = config['embed_dim']
            self.num_heads = config['num_heads']
            self.ff_dim = config['ff_dim']
            self.num_transformer_blocks = config['transformer_blocks']
            self.model_dir = model_dir
            
            # Load models
            self.main_model = tf.keras.models.load_model(
                os.path.join(model_dir, 'main_model.keras')
            )
            
            self.bonus_model = tf.keras.models.load_model(
                os.path.join(model_dir, 'bonus_model.keras')
            )
            
            # Load scalers
            self.scaler_X_main = StandardScaler()
            self.scaler_y_main = StandardScaler()
            self.scaler_X_bonus = StandardScaler()
            self.scaler_y_bonus = StandardScaler()
            
            # Load scaler parameters
            scaler_X_main_dict = np.load(os.path.join(model_dir, 'scaler_X_main.npy'), 
                                       allow_pickle=True).item()
            scaler_y_main_dict = np.load(os.path.join(model_dir, 'scaler_y_main.npy'), 
                                       allow_pickle=True).item()
            scaler_X_bonus_dict = np.load(os.path.join(model_dir, 'scaler_X_bonus.npy'), 
                                        allow_pickle=True).item()
            scaler_y_bonus_dict = np.load(os.path.join(model_dir, 'scaler_y_bonus.npy'), 
                                        allow_pickle=True).item()
            
            # Apply parameters to scalers
            for k, v in scaler_X_main_dict.items():
                if k != 'n_samples_seen_' and not k.startswith('_'):
                    setattr(self.scaler_X_main, k, v)
            
            for k, v in scaler_y_main_dict.items():
                if k != 'n_samples_seen_' and not k.startswith('_'):
                    setattr(self.scaler_y_main, k, v)
                    
            for k, v in scaler_X_bonus_dict.items():
                if k != 'n_samples_seen_' and not k.startswith('_'):
                    setattr(self.scaler_X_bonus, k, v)
                    
            for k, v in scaler_y_bonus_dict.items():
                if k != 'n_samples_seen_' and not k.startswith('_'):
                    setattr(self.scaler_y_bonus, k, v)
            
            # Set n_samples_seen_ attribute separately (it's a numpy int64 which needs special handling)
            self.scaler_X_main.n_samples_seen_ = int(scaler_X_main_dict.get('n_samples_seen_', 0))
            self.scaler_y_main.n_samples_seen_ = int(scaler_y_main_dict.get('n_samples_seen_', 0))
            self.scaler_X_bonus.n_samples_seen_ = int(scaler_X_bonus_dict.get('n_samples_seen_', 0))
            self.scaler_y_bonus.n_samples_seen_ = int(scaler_y_bonus_dict.get('n_samples_seen_', 0))
            
            logger.info(f"Models and configuration loaded from {model_dir}")
            
            # Load metadata if it exists
            if os.path.exists(os.path.join(model_dir, 'model_metadata.json')):
                with open(os.path.join(model_dir, 'model_metadata.json'), 'r') as f:
                    self.metadata = json.load(f)
            else:
                # Create new metadata
                self.metadata = {
                    'creation_date': datetime.now().isoformat(),
                    'params': config,
                    'performance': {},
                    'predictions': []
                }
            
            # Check for ensemble models
            ensemble_dir = os.path.join(model_dir, 'ensemble')
            if os.path.exists(ensemble_dir):
                main_model_files = sorted([f for f in os.listdir(ensemble_dir) if f.startswith('main_model_')])
                bonus_model_files = sorted([f for f in os.listdir(ensemble_dir) if f.startswith('bonus_model_')])
                
                if main_model_files and bonus_model_files:
                    main_models = []
                    bonus_models = []
                    
                    for main_file in main_model_files:
                        main_model = tf.keras.models.load_model(os.path.join(ensemble_dir, main_file))
                        main_models.append(main_model)
                    
                    for bonus_file in bonus_model_files:
                        bonus_model = tf.keras.models.load_model(os.path.join(ensemble_dir, bonus_file))
                        bonus_models.append(bonus_model)
                    
                    self.ensemble = {
                        'main_models': main_models,
                        'bonus_models': bonus_models
                    }
                    
                    logger.info(f"Loaded ensemble with {len(main_models)} models")
            
            return True
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            logger.error(traceback.format_exc())
            return False


def extract_numbers_to_list(input_file):
    """
    Extract lottery numbers from a text file using the exact function provided by the user.
    
    Args:
        input_file: Path to text file containing lottery numbers
        
    Returns:
        List of extracted numbers
    """
    all_numbers = []
    current_set = []
    
    with open(input_file, 'r') as f:
        for line in f:
            if 'th' in line:
                if current_set:
                    all_numbers = current_set + all_numbers
                current_set = []
                continue
                
            try:
                num = int(line.strip())
                if 0 < num <= 50: # Ensure numbers are between 1 and 50
                    current_set.append(num)
            except ValueError:
                continue
                
    if current_set:
        all_numbers = current_set + all_numbers
        
    return all_numbers


def create_sample_lottery_file(input_file, num_draws=200):
    """
    Create a sample lottery file for testing with realistic patterns.
    Ensures that the output file doesn't overwrite the input file.
    
    Args:
        input_file: Input file name (to avoid overwriting)
        num_draws: Number of draws to generate
        
    Returns:
        Path to the created sample file
    """
    # Generate a unique output filename to avoid overwriting input
    base_name = os.path.splitext(input_file)[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{base_name}_sample_{timestamp}.txt"
    
    # Ensure we don't accidentally overwrite the input file
    if output_file == input_file:
        output_file = f"lottery_sample_{timestamp}.txt"
    
    # Initialize random number generators with different seeds for main and bonus
    main_rng = np.random.RandomState(42)
    bonus_rng = np.random.RandomState(24)
    
    with open(output_file, 'w') as f:
        for i in range(num_draws, 0, -1):
            f.write(f"{i}th draw:\n")
            
            # Generate 5 main numbers (1-50) with slight weighting to make some numbers appear more often
            weights = np.ones(50) / 50
            # Give slightly higher weight to some numbers
            for num in [7, 12, 23, 33, 42]:
                weights[num-1] *= 1.2
            # And slightly lower weight to others
            for num in [1, 15, 27, 39, 49]:
                weights[num-1] *= 0.8
                
            # Normalize weights
            weights = weights / np.sum(weights)
            
            # Generate main numbers with these weights
            main_numbers = []
            while len(main_numbers) < 5:
                num = main_rng.choice(range(1, 51), p=weights)
                if num not in main_numbers:
                    main_numbers.append(num)
            
            # Sort and write main numbers
            for num in sorted(main_numbers):
                f.write(f"{num}\n")
            
            # Generate 2 bonus numbers (1-12) with slight weighting
            bonus_weights = np.ones(12) / 12
            # Give slightly higher weight to some numbers
            for num in [3, 8]:
                bonus_weights[num-1] *= 1.3
            # And slightly lower weight to others
            for num in [1, 11]:
                bonus_weights[num-1] *= 0.7
                
            # Normalize weights
            bonus_weights = bonus_weights / np.sum(bonus_weights)
            
            # Generate bonus numbers with these weights
            bonus_numbers = []
            while len(bonus_numbers) < 2:
                num = bonus_rng.choice(range(1, 13), p=bonus_weights)
                if num not in bonus_numbers:
                    bonus_numbers.append(num)
            
            # Sort and write bonus numbers
            for num in sorted(bonus_numbers):
                f.write(f"{num}\n")
    
    logger.info(f"Created sample lottery file with {num_draws} draws at {output_file}")
    return output_file


def main():
    """
    Main function to run the enhanced lottery predictor
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Enhanced Lottery Predictor')
    parser.add_argument('--input', '-i', type=str, default='lottery_numbers.txt',
                        help='Input file with lottery numbers (default: lottery_numbers.txt)')
    parser.add_argument('--mode', '-m', type=str, default='train_and_predict',
                        choices=['train_and_predict', 'predict_only', 'validate', 'create_sample', 'optimize'],
                        help='Operation mode (default: train_and_predict)')
    parser.add_argument('--draws', '-d', type=int, default=200,
                        help='Number of draws to generate if creating sample file (default: 200)')
    parser.add_argument('--epochs', '-e', type=int, default=100,
                        help='Number of training epochs (default: 100)')
    parser.add_argument('--backtest', '-b', type=int, default=10,
                        help='Number of draws to use for backtesting (default: 10)')
    parser.add_argument('--ensemble', action='store_true', 
                        help='Use ensemble of models for prediction')
    parser.add_argument('--trials', '-t', type=int, default=10,
                        help='Number of trials for hyperparameter optimization (default: 10)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create sample file if requested
    if args.mode == 'create_sample':
        output_file = create_sample_lottery_file(args.input, args.draws)
        logger.info(f"Created sample file with {args.draws} draws at {output_file}")
        return
    
    # Check if input file exists
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        logger.info("Use --mode create_sample to generate a sample file")
        return
    
    # Extract numbers
    all_numbers = extract_numbers_to_list(args.input)
    
    if not all_numbers:
        logger.error("No numbers extracted from input file")
        return
    
    # Create predictor instance
    predictor = EnhancedLotteryPredictor(
        sequence_length=15,
        main_numbers=5,
        bonus_numbers=2,
        main_number_range=(1, 50),
        bonus_number_range=(1, 12)
    )
    
    # Separate main and bonus numbers
    main_numbers, bonus_numbers, draws = predictor.separate_main_and_bonus(all_numbers)
    
    # Create frequency visualization
    predictor.create_frequency_heatmap(draws)
    
    # Handle optimization mode
    if args.mode == 'optimize':
        logger.info("Starting hyperparameter optimization...")
        best_params = predictor.hyperparameter_search(main_numbers, bonus_numbers, trials=args.trials)
        
        # Apply best parameters
        for param, value in best_params.items():
            if hasattr(predictor, param):
                setattr(predictor, param, value)
        
        logger.info(f"Applied optimized parameters: {best_params}")
    
    # Handle different modes
    if args.mode in ['train_and_predict', 'optimize']:
        # Prepare data and train model
        logger.info("Preparing data and training model...")
        prepared_data = predictor.prepare_data(main_numbers, bonus_numbers)
        
        if prepared_data:
            # Train models (regular or ensemble)
            if args.ensemble:
                logger.info("Training ensemble of models...")
                ensemble = predictor.create_ensemble_models(prepared_data, epochs=args.epochs)
                predictor.ensemble = ensemble
            else:
                # Train single models
                history = predictor.train_models(
                    prepared_data, 
                    epochs=args.epochs,
                    batch_size=16, 
                    patience=20
                )
                
                # Plot learning curves
                predictor.plot_learning_curves(history)
            
            # Validate with backtesting
            num_backtest = min(args.backtest, len(draws)-5)
            logger.info(f"Validating with backtesting on {num_backtest} draws...")
            validation_results = predictor.validate_with_backtesting(draws, num_backtest)
            
            if validation_results:
                predictor.plot_backtesting_results(validation_results)
            
            # Make prediction
            logger.info("Generating prediction...")
            prediction = predictor.predict_next_drawing(all_numbers)
            
            if prediction:
                # Visualize prediction
                predictor.visualize_predictions(prediction)
                
                # Generate report
                report_path = predictor.generate_prediction_report(prediction, draws[-5:])
                
                # Save models
                predictor.save_models()
                
                # Print prediction
                print("\n===== ENHANCED LOTTERY PREDICTION =====")
                print(f"Main numbers: {prediction['main']}")
                print(f"Bonus numbers: {prediction['bonus']}")
                print(f"Full report: {report_path}")
                print(f"Model directory: {predictor.model_dir}")
            else:
                logger.error("Failed to generate prediction")
        else:
            logger.error("Failed to prepare data")
    
    elif args.mode == 'predict_only':
        # Find existing models
        model_dirs = [d for d in os.listdir('.') if os.path.isdir(d) and d.startswith('enhanced_lottery_model_')]
        
        if not model_dirs:
            logger.error("No trained models found. Please run in 'train_and_predict' mode first.")
            return
        
        # Sort by modification time to get the most recent
        model_dirs.sort(key=lambda d: os.path.getmtime(d), reverse=True)
        latest_model_dir = model_dirs[0]
        
        # Load models
        logger.info(f"Loading model from {latest_model_dir}...")
        if predictor.load_models(latest_model_dir):
            # Make prediction
            logger.info("Generating prediction...")
            prediction = predictor.predict_next_drawing(all_numbers)
            
            if prediction:
                # Visualize prediction
                predictor.visualize_predictions(prediction)
                
                # Generate report
                report_path = predictor.generate_prediction_report(prediction, draws[-5:])
                
                # Print prediction
                print("\n===== ENHANCED LOTTERY PREDICTION =====")
                print(f"Using model from: {latest_model_dir}")
                print(f"Main numbers: {prediction['main']}")
                print(f"Bonus numbers: {prediction['bonus']}")
                print(f"Full report: {report_path}")
            else:
                logger.error("Failed to generate prediction")
        else:
            logger.error(f"Failed to load models from {latest_model_dir}")
    
    elif args.mode == 'validate':
        # Prepare data
        logger.info("Preparing data and training model for validation...")
        prepared_data = predictor.prepare_data(main_numbers, bonus_numbers)
        
        if prepared_data:
            # Train models
            history = predictor.train_models(
                prepared_data, 
                epochs=args.epochs,
                batch_size=16, 
                patience=20
            )
            
            # Plot learning curves
            predictor.plot_learning_curves(history)
            
            # Run extensive validation
            num_backtest = min(args.backtest, len(draws)-5)
            logger.info(f"Running extensive validation on {num_backtest} draws...")
            validation_results = predictor.validate_with_backtesting(draws, num_backtest)
            
            if validation_results:
                predictor.plot_backtesting_results(validation_results)
                
                # Print validation summary
                metrics = validation_results['metrics']
                print("\n===== VALIDATION RESULTS =====")
                print(f"Main matches: {metrics['avg_main_matches']:.2f} ± {metrics['main_matches_ci']:.2f}/{predictor.main_numbers}")
                print(f"Bonus matches: {metrics['avg_bonus_matches']:.2f} ± {metrics['bonus_matches_ci']:.2f}/{predictor.bonus_numbers}")
                print(f"Overall score: {metrics['avg_score']:.2f} ± {metrics['total_score_ci']:.2f}")
                print(f"Improvement over random: {metrics['improvement']:.2f}%")
                print(f"Statistically significant: {'Yes' if metrics['significant_improvement'] else 'No'}")
                print(f"Validation plots saved to: {predictor.model_dir}")
            else:
                logger.error("Validation failed")
        else:
            logger.error("Failed to prepare data")


if __name__ == "__main__":
    main()