#!/usr/bin/env python
"""
Default command
python neural_pred.py --file lottery_numbers.txt --predictions 30 --evaluate

Step 1: Optimize hyperparameters, this will find the best hyperparameters and save them to transformer_params.json.
python neural_pred.py --file lottery_numbers.txt --optimize --trials 50

Step 2: Train models and generate predictions using optimized parameters
python neural_pred.py --file lottery_numbers.txt --params transformer_params.json --ensemble --num_models 7 --predictions 30 --evaluate


"""

import os
import sys
import re
import json
import random
import argparse
import logging
import traceback
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, MultiHeadAttention, LayerNormalization,
    GlobalAveragePooling1D, Concatenate, GRU, Conv1D,
    Bidirectional, BatchNormalization, Activation, LSTM, SimpleRNN
)
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import gc

#######################
# CONSTANTS
#######################

# Random seeds for reproducibility
RANDOM_SEED = 42

# Lottery number ranges
MAIN_NUM_MIN = 1
MAIN_NUM_MAX = 50
MAIN_NUM_COUNT = 5
BONUS_NUM_MIN = 1
BONUS_NUM_MAX = 12
BONUS_NUM_COUNT = 2

# Model parameters - improved defaults based on empirical results
DEFAULT_SEQUENCE_LENGTH = 20
DEFAULT_EMBED_DIM = 48  # Adjusted from 64 for better balance
DEFAULT_NUM_HEADS = 3  # Adjusted from 4 to better suit the task
DEFAULT_FF_DIM = 96  # Reduced from 128 to prevent overfitting
DEFAULT_DROPOUT_RATE = 0.35  # Increased from 0.3 for stronger regularization
DEFAULT_BATCH_SIZE = 32  # Increased from 16 for better stability
DEFAULT_EPOCHS = 80  # Reduced from 100 to avoid overfitting
DEFAULT_LEARNING_RATE = 0.0008  # Reduced from 0.001 for better convergence
DEFAULT_TRANSFORMER_BLOCKS = 2
DEFAULT_CONV_FILTERS = 24  # Optimized from 32
DEFAULT_PATIENCE = 10  # Reduced from 15 for earlier stopping

# Confidence calibration factors - refined based on statistical analysis
MAIN_CONF_SCALE = 0.55  # Reduced from 0.6 for more conservative estimations
MAIN_CONF_OFFSET = 0.18  # Reduced from 0.2 for more conservative estimations
BONUS_CONF_SCALE = 0.65  # Reduced from 0.7 for more conservative estimations
BONUS_CONF_OFFSET = 0.12  # Reduced from 0.15 for more conservative estimations

# Feature selection - increased to include more features
MAX_FEATURES = 400  # Increased from 300

# Temperature scaling for sampling - more dynamic approach
DEFAULT_TEMPERATURE = 0.75  # Base temperature adjusted from 0.8
MIN_TEMPERATURE = 0.6  # Minimum temperature for high confidence predictions
MAX_TEMPERATURE = 1.2  # Maximum temperature for low confidence predictions
DIVERSITY_FACTOR = 0.85  # Increased from 0.8 for more diverse results

# Memory management constants - optimized frequency
CLEAN_MEMORY_FREQUENCY = 8  # Reduced from 10 for more frequent cleanup with less impact

# Early stopping threshold for loss improvement
MIN_DELTA = 0.001  # Minimum change in the monitored quantity to qualify as improvement

# Configure GPU if available
def configure_gpu():
    """Configure GPU with proper memory growth if available."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Configure GPU memory growth to avoid taking all memory at once
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Set TensorFlow memory allocator
            tf.config.experimental.set_memory_growth(gpus[0], True)
            
            # Set TensorFlow to only log errors
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
            
            return f"Found {len(gpus)} GPU(s), configured for optimal memory usage"
        except RuntimeError as e:
            return f"GPU configuration error: {e}"
    else:
        return "No GPU found. Using CPU."

# Set up consistent seeds
def set_seeds(seed=RANDOM_SEED):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

# Set up logging with a configurable level
def setup_logging(level=logging.INFO):
    """Set up logging with file and console handlers."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("enhanced_transformer.log", mode='w'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    # Hide TensorFlow warnings
    tf.get_logger().setLevel('ERROR')
    return logger

# Initialize logger
logger = setup_logging()

# Print GPU configuration
logger.info(configure_gpu())

# Set seeds
set_seeds()

#######################
# ERROR HANDLING
#######################

class ErrorHandler:
    """Centralized error handling system."""
    
    @staticmethod
    def handle_exception(logger, operation_name, fallback_result=None, log_traceback=True):
        """Decorator for handling exceptions in a consistent way."""
        def decorator(func):
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    logger.error(f"Error in {operation_name}: {str(e)}")
                    if log_traceback:
                        logger.error(traceback.format_exc())
                    
                    # Return fallback if provided, or raise the exception
                    if fallback_result is not None:
                        return fallback_result() if callable(fallback_result) else fallback_result
                    raise
            return wrapper
        return decorator

#######################
# UTILITIES
#######################

class Utilities:
    """Unified utility class for shared functionality."""
    
    # Track memory cleaning operations to avoid redundant calls
    _last_clean_time = 0
    _clean_counter = 0
    
    @staticmethod
    def calculate_optimal_sequence_length(data_size):
        """Dynamically calculate optimal sequence length based on data size."""
        # Use a heuristic approach based on data size
        if data_size < 200:
            return 8  # Small dataset -> short sequence
        elif data_size < 500:
            return 12  # Medium dataset -> medium sequence
        elif data_size < 1000:
            return 16  # Large dataset -> longer sequence
        else:
            return 20  # Very large dataset -> longest sequence
    
    @staticmethod
    def clean_memory(force=False):
        """Force garbage collection to free memory, with intelligent frequency control."""
        # Only clean memory if forced or enough operations have occurred
        Utilities._clean_counter += 1
        current_time = datetime.now().timestamp()
        
        # Clean if forced, counter threshold reached, or significant time passed
        if (force or 
            Utilities._clean_counter >= CLEAN_MEMORY_FREQUENCY or 
            current_time - Utilities._last_clean_time > 60):  # At least every 60 seconds
            
            try:
                # Force garbage collection
                gc.collect()
                
                # If running in TensorFlow environment, clear session
                tf.keras.backend.clear_session()
                
                logger.info("Memory cleanup performed")
                
                # Reset counters
                Utilities._clean_counter = 0
                Utilities._last_clean_time = current_time
                
            except Exception as e:
                logger.warning(f"Memory cleanup failed: {e}")
    
    @staticmethod
    def calculate_dynamic_temperature(confidence, min_temp=MIN_TEMPERATURE, max_temp=MAX_TEMPERATURE):
        """Calculate temperature dynamically based on confidence score."""
        # Scale temperature inversely with confidence
        # Higher confidence -> lower temperature (more deterministic sampling)
        # Lower confidence -> higher temperature (more exploration)
        return max_temp - (confidence * (max_temp - min_temp))
    
    @staticmethod
    def sample_numbers(probs, available_nums, num_to_select, 
                      used_nums=None, diversity_sampling=True, 
                      diversity_factor=DIVERSITY_FACTOR, draw_idx=0, temperature=DEFAULT_TEMPERATURE,
                      confidence=None):
        """Unified sampling functionality for number selection with improved diversity."""
        selected_numbers = []
        available = list(available_nums)
        
        # If confidence is provided, adjust temperature dynamically
        if confidence is not None:
            temperature = Utilities.calculate_dynamic_temperature(confidence)
        
        for i in range(num_to_select):
            # Get probability distribution for this position
            position_probs = probs[i][0]
            
            # Apply temperature scaling
            position_probs = np.power(position_probs, 1/max(0.1, temperature))
            sum_probs = np.sum(position_probs)
            if sum_probs > 0:
                position_probs = position_probs / sum_probs
            
            # Adjust probabilities to only include available numbers
            adjusted_probs = np.zeros(len(position_probs))
            for num in available:
                if 1 <= num <= len(position_probs):  # Ensure index is valid
                    adjusted_probs[num-1] = position_probs[num-1]
                    
                    # Apply progressive diversity penalty - scales with draw_idx
                    if diversity_sampling and used_nums and num in used_nums and draw_idx > 0:
                        penalty = diversity_factor ** (1 + (draw_idx * 0.1))  # Stronger penalty for later draws
                        adjusted_probs[num-1] *= penalty
            
            # Normalize probabilities with proper error handling
            sum_adjusted = np.sum(adjusted_probs)
            if sum_adjusted > 0:
                adjusted_probs = adjusted_probs / sum_adjusted
            else:
                # Fallback to uniform distribution
                valid_indices = [j-1 for j in available if 1 <= j <= len(position_probs)]
                if valid_indices:
                    adjusted_probs[valid_indices] = 1.0 / len(valid_indices)
                else:
                    # Emergency fallback - no valid indices
                    continue
            
            # Sample a number based on adjusted probabilities
            try:
                num_idx = np.random.choice(len(position_probs), p=adjusted_probs)
                num = num_idx + 1  # Convert back to 1-based indexing
            except ValueError:
                # Fallback to random selection
                if available:
                    num = random.choice(available)
                else:
                    # Emergency fallback - should not happen
                    remaining = set(range(1, min(51, len(position_probs) + 1))) - set(selected_numbers)
                    if not remaining:
                        remaining = set(range(1, min(51, len(position_probs) + 1)))
                    num = random.choice(list(remaining)) if remaining else 1
            
            selected_numbers.append(int(num))
            if num in available:
                available.remove(int(num))
            
            # Update used numbers set if provided
            if used_nums is not None:
                used_nums.add(int(num))
        
        # Sort the selected numbers
        return sorted(selected_numbers)
    
    @staticmethod
    def calculate_pattern_score(main_numbers, bonus_numbers=None):
        """Calculate pattern score based on historical patterns with improved metrics."""
        # Simple pattern checks
        # 1. Check distribution across range
        main_bins = [0, 0, 0, 0, 0]  # 1-10, 11-20, 21-30, 31-40, 41-50
        for num in main_numbers:
            bin_idx = (num - 1) // 10
            if 0 <= bin_idx < 5:
                main_bins[bin_idx] += 1
        
        # Ideal distribution tends to be more spread out (1-2 numbers per bin)
        # Calculate the variance of the bins - lower variance is better
        bin_variance = np.var(main_bins)
        distribution_score = 1.0 / (1.0 + bin_variance)  # Higher variance -> lower score
        
        # 2. Check for consecutive numbers
        consecutive_count = 0
        for i in range(len(main_numbers) - 1):
            if main_numbers[i + 1] - main_numbers[i] == 1:
                consecutive_count += 1
        
        # Calculate average gap between numbers
        gaps = [main_numbers[i+1] - main_numbers[i] for i in range(len(main_numbers)-1)]
        avg_gap = np.mean(gaps) if gaps else 0
        
        # Balance between not too many consecutive and reasonable gap size
        if consecutive_count <= 1 and 5 <= avg_gap <= 15:
            consecutive_score = 0.9
        elif consecutive_count == 2 or (3 <= avg_gap <= 18):
            consecutive_score = 0.7
        else:
            consecutive_score = 0.5
        
        # 3. Check for even/odd balance
        even_count = sum(1 for n in main_numbers if n % 2 == 0)
        
        # Ideal balance is 2-3 or 3-2
        if even_count in [2, 3]:
            balance_score = 0.9
        elif even_count in [1, 4]:
            balance_score = 0.7
        else:
            balance_score = 0.5  # All even or all odd is rare
            
        # 4. Check sum of numbers - most common sums fall in certain ranges
        num_sum = sum(main_numbers)
        if 120 <= num_sum <= 180:  # Most common range
            sum_score = 0.9
        elif 100 <= num_sum <= 200:  # Wider common range
            sum_score = 0.7
        else:
            sum_score = 0.5  # Uncommon sum
        
        # 5. Check low-high balance (numbers below and above 25)
        low_count = sum(1 for n in main_numbers if n <= 25)
        if 2 <= low_count <= 3:  # Balanced
            low_high_score = 0.9
        elif low_count in [1, 4]:  # Slightly imbalanced
            low_high_score = 0.7
        else:
            low_high_score = 0.5  # Very imbalanced
        
        # Combine scores with weights
        pattern_score = (
            0.25 * distribution_score + 
            0.2 * consecutive_score + 
            0.2 * balance_score + 
            0.15 * sum_score + 
            0.2 * low_high_score
        )
        
        # Bonus number pattern (if provided)
        if bonus_numbers and len(bonus_numbers) >= 2:
            # Check if bonus numbers are close or far apart
            bonus_gap = abs(bonus_numbers[1] - bonus_numbers[0])
            if 2 <= bonus_gap <= 6:  # Most common gap
                bonus_score = 0.9
            elif 1 <= bonus_gap <= 8:  # Wider common range
                bonus_score = 0.7
            else:
                bonus_score = 0.5  # Uncommon gap
            
            # Combine with main score
            pattern_score = 0.85 * pattern_score + 0.15 * bonus_score
            
        return pattern_score
    
    @staticmethod
    def calculate_frequency_score(main_numbers, bonus_numbers, data):
        """Calculate score based on historical frequency of numbers with improved metrics."""
        if data is None or len(data) < 20:
            return 0.5  # Not enough historical data
        
        # Calculate historical frequency for main numbers
        main_counts = np.zeros(MAIN_NUM_MAX)
        for _, row in data.iterrows():
            for num in row["main_numbers"]:
                if MAIN_NUM_MIN <= num <= MAIN_NUM_MAX:
                    main_counts[num-1] += 1
        
        # Normalize to get probability
        main_freq = main_counts / len(data)
        
        # Calculate average frequency for selected main numbers
        main_freqs = [main_freq[num-1] for num in main_numbers]
        main_avg = np.mean(main_freqs)
        
        # Calculate variance of frequencies - lower variance is better
        main_var = np.var(main_freqs)
        
        # Combine average and variance for a balanced score
        # We want high average frequency but also balanced variance
        main_score = main_avg * (1 - main_var * 5)  # Penalize high variance
        
        # Normalize score to 0-1 range (clip to handle extreme values)
        main_score_norm = np.clip(main_score / 0.05, 0, 1)  # Adjusted normalization factor
        
        # Same for bonus numbers
        bonus_counts = np.zeros(BONUS_NUM_MAX)
        for _, row in data.iterrows():
            for num in row["bonus_numbers"]:
                if BONUS_NUM_MIN <= num <= BONUS_NUM_MAX:
                    bonus_counts[num-1] += 1
        
        bonus_freq = bonus_counts / len(data)
        bonus_freqs = [bonus_freq[num-1] for num in bonus_numbers]
        bonus_avg = np.mean(bonus_freqs)
        bonus_var = np.var(bonus_freqs)
        
        bonus_score = bonus_avg * (1 - bonus_var * 5)
        bonus_score_norm = np.clip(bonus_score / 0.085, 0, 1)  # Adjusted for bonus number range
        
        # Calculate recency bias score - more emphasis on recent drawings
        # Get last 50 draws
        recent_draws = data.iloc[-50:] if len(data) >= 50 else data
        
        # Calculate recency scores
        main_recent_score = 0
        bonus_recent_score = 0
        
        # Count recent occurrences with decay factor
        for i, (_, row) in enumerate(recent_draws.iterrows()):
            # More recent draws get higher weight
            recency_weight = 1 - (i / len(recent_draws))
            
            # Check main numbers
            for num in main_numbers:
                if num in row["main_numbers"]:
                    main_recent_score += recency_weight * 0.05
            
            # Check bonus numbers
            for num in bonus_numbers:
                if num in row["bonus_numbers"]:
                    bonus_recent_score += recency_weight * 0.05
        
        # Normalize recency scores
        main_recent_score_norm = np.clip(main_recent_score, 0, 1)
        bonus_recent_score_norm = np.clip(bonus_recent_score, 0, 1)
        
        # Combine scores with weights
        # More emphasis on long-term frequency but some weight to recency
        combined_main_score = 0.7 * main_score_norm + 0.3 * main_recent_score_norm
        combined_bonus_score = 0.7 * bonus_score_norm + 0.3 * bonus_recent_score_norm
        
        return (combined_main_score + combined_bonus_score) / 2
    
    @staticmethod
    def calculate_partial_match_score(prediction, actual_draw):
        """Calculate score for partial matches between prediction and actual draw."""
        if not actual_draw or not prediction:
            return 0.0
            
        main_pred = set(prediction["main_numbers"])
        main_actual = set(actual_draw["main_numbers"])
        bonus_pred = set(prediction["bonus_numbers"])
        bonus_actual = set(actual_draw["bonus_numbers"])
        
        # Count matching numbers
        main_matches = len(main_pred.intersection(main_actual))
        bonus_matches = len(bonus_pred.intersection(bonus_actual))
        
        # Calculate scores with proper weighting
        # Main numbers are worth 80% of total score
        main_score = (main_matches / MAIN_NUM_COUNT) * 0.8
        
        # Bonus numbers are worth 20% of total score
        bonus_score = (bonus_matches / BONUS_NUM_COUNT) * 0.2
        
        # Combine scores
        total_score = main_score + bonus_score
        
        # Add position accuracy bonus for main numbers if appropriate
        if main_matches > 0 and "main_number_positions" in prediction and "main_number_positions" in actual_draw:
            position_matches = sum(1 for num in main_pred.intersection(main_actual) 
                                if prediction["main_number_positions"].get(num) == actual_draw["main_number_positions"].get(num))
            position_bonus = (position_matches / max(1, main_matches)) * 0.05
            total_score += position_bonus
        
        return total_score
    
    @staticmethod
    def get_default_params():
        """Get default parameters with documentation."""
        return {
            'learning_rate': DEFAULT_LEARNING_RATE,
            'batch_size': DEFAULT_BATCH_SIZE,
            'epochs': DEFAULT_EPOCHS,
            'dropout_rate': DEFAULT_DROPOUT_RATE,
            'num_heads': DEFAULT_NUM_HEADS,
            'ff_dim': DEFAULT_FF_DIM,
            'embed_dim': DEFAULT_EMBED_DIM,
            'use_gru': True,
            'conv_filters': DEFAULT_CONV_FILTERS,
            'optimizer': 'adam',
            'num_transformer_blocks': DEFAULT_TRANSFORMER_BLOCKS,
            'sequence_length': DEFAULT_SEQUENCE_LENGTH,
            'l2_regularization': 0.0001,  # New parameter for L2 regularization
            'model_type': 'transformer',  # New parameter for model architecture type
            'lstm_units': 32,  # New parameter for LSTM-based models
            'min_delta': MIN_DELTA,  # New parameter for early stopping threshold
            'use_residual': True,  # New parameter for using residual connections
            'use_layer_scaling': True,  # New parameter for layer scaling in transformer
        }
    
    @staticmethod
    @ErrorHandler.handle_exception(logger, "parameter loading")
    def load_params(file_path, default_params=None):
        """Load model parameters from JSON file with proper error handling."""
        if default_params is None:
            default_params = Utilities.get_default_params()
            
        if file_path and os.path.exists(file_path):
            with open(file_path, 'r') as f:
                params = json.load(f)
            logger.info(f"Loaded parameters from {file_path}")
            
            # Validate and fill in missing parameters
            for key, value in default_params.items():
                if key not in params:
                    params[key] = value
                    logger.info(f"Added missing parameter {key}={value}")
            
            return params
        else:
            logger.warning(f"Parameter file {file_path} not found, using defaults")
            return default_params
    
    @staticmethod
    @ErrorHandler.handle_exception(logger, "parameter saving")
    def save_params(params, file_path):
        """Save model parameters to JSON file with proper error handling."""
        # Create directory if it doesn't exist
        directory = os.path.dirname(os.path.abspath(file_path))
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        
        with open(file_path, 'w') as f:
            json.dump(params, f, indent=4)
        logger.info(f"Saved parameters to {file_path}")
        return True
    
    @staticmethod
    def get_model_callbacks(model_types=None, patience=DEFAULT_PATIENCE, include_checkpoint=True, 
                         base_path="models/best", include_timestamp=True, min_delta=MIN_DELTA):
        """Get consistent callbacks for multiple model types at once."""
        if model_types is None:
            model_types = ["main"]  # Default to just main model
        elif isinstance(model_types, str):
            model_types = [model_types]  # Convert single string to list
        
        # Common callbacks with same configuration
        common_callbacks = [
            # Early stopping - improved with min_delta parameter
            EarlyStopping(
                patience=patience,
                restore_best_weights=True,
                monitor='val_loss',
                min_delta=min_delta,
                verbose=1
            ),
            # Learning rate scheduler - improved with more gradual reduction
            ReduceLROnPlateau(
                factor=0.6,  # More gradual reduction than 0.7
                patience=max(3, patience // 3),  # More responsive to plateaus
                min_lr=5e-7,  # Lower limit to allow more exploration
                monitor='val_loss',
                verbose=1
            )
        ]
        
        # Model-specific callbacks with checkpoints
        model_callbacks = {}
        checkpoint_paths = {}
        
        # Add model-specific callbacks if needed
        for model_type in model_types:
            model_callbacks[model_type] = common_callbacks.copy()
            
            # Add checkpoint callback if requested
            if include_checkpoint:
                if include_timestamp:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filepath = f"{base_path}_{model_type}_{timestamp}.keras"
                else:
                    filepath = f"{base_path}_{model_type}.keras"
                
                # Create directory if it doesn't exist
                directory = os.path.dirname(os.path.abspath(filepath))
                if directory and not os.path.exists(directory):
                    os.makedirs(directory, exist_ok=True)
                    
                checkpoint_callback = ModelCheckpoint(
                    filepath=filepath,
                    save_best_only=True,
                    monitor='val_loss',
                    mode='min',
                    verbose=1
                )
                model_callbacks[model_type].append(checkpoint_callback)
                checkpoint_paths[model_type] = filepath
        
        # If only one model type, return directly for backward compatibility
        if len(model_types) == 1:
            return model_callbacks[model_types[0]], checkpoint_paths.get(model_types[0])
        
        # Otherwise return dictionaries for all models
        return model_callbacks, checkpoint_paths

#######################
# DATA PROCESSING
#######################

class LotteryDataProcessor:
    """Enhanced processor for lottery data from text files."""
    
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.expanded_data = None
        self.features = None
        self.feature_engineer = None
        self.sequence_length = DEFAULT_SEQUENCE_LENGTH
    
    def set_sequence_length(self, length):
        """Set the sequence length for data processing."""
        self.sequence_length = length
        return self
        
    @ErrorHandler.handle_exception(logger, "lottery data parsing", pd.DataFrame())
    def parse_file(self):
        """Parse the lottery data file into a structured DataFrame."""
        logger.info(f"Parsing lottery data from {self.file_path}")
        
        if not os.path.exists(self.file_path):
            logger.error(f"Lottery data file not found: {self.file_path}")
            return pd.DataFrame()
            
        with open(self.file_path, 'r') as file:
            content = file.read()
        
        # Improved regex pattern - more robust to variations in formatting
        draw_pattern = r"((?:\w+)\s+\d+(?:st|nd|rd|th)?\s+(?:\w+)\s+\d{4})\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(€[\d,]+)\s+(Roll|Won)"
        draws = re.findall(draw_pattern, content)
        
        if not draws:
            logger.error(f"No valid draws found in {self.file_path}")
            return pd.DataFrame()
        
        # Process extracted data
        structured_data = []
        
        for draw in draws:
            try:
                date_str = draw[0]
                main_numbers = [int(draw[i]) for i in range(1, 6)]
                bonus_numbers = [int(draw[i]) for i in range(6, 8)]
                jackpot = draw[8]
                result = draw[9]
                
                # Improved date parsing
                clean_date_str = re.sub(r'(\d+)(st|nd|rd|th)', r'\1', date_str)
                try:
                    date = datetime.strptime(clean_date_str, "%A %d %B %Y")
                except ValueError:
                    # Try alternative format
                    try:
                        date = datetime.strptime(clean_date_str, "%a %d %b %Y")
                    except ValueError:
                        # If both fail, extract components manually
                        parts = clean_date_str.split()
                        if len(parts) >= 4:
                            day = int(parts[1])
                            month_map = {"January": 1, "February": 2, "March": 3, "April": 4, 
                                    "May": 5, "June": 6, "July": 7, "August": 8, 
                                    "September": 9, "October": 10, "November": 11, "December": 12}
                            month = month_map.get(parts[2], 1)  # Default to January if not found
                            year = int(parts[3])
                            date = datetime(year, month, day)
                        else:
                            # Last resort, create a dummy date
                            date = datetime(2000, 1, 1)
                            logger.warning(f"Could not parse date: {date_str}, using default")
            
                # Create main numbers position map for position accuracy metrics
                main_positions = {num: idx for idx, num in enumerate(main_numbers)}
                
                structured_data.append({
                    "date": date,
                    "day_of_week": date.strftime("%A"),
                    "main_numbers": sorted(main_numbers),
                    "main_number_positions": main_positions,
                    "bonus_numbers": sorted(bonus_numbers),
                    "jackpot": jackpot,
                    "result": result
                })
            except Exception as e:
                logger.warning(f"Could not parse draw: {draw}, error: {e}")
                continue
        
        # Convert to DataFrame and sort by date
        df = pd.DataFrame(structured_data)
        
        if df.empty:
            logger.error("No valid draws could be parsed")
            return df
            
        df = df.sort_values("date")
        
        # Extract jackpot value as numeric
        df["jackpot_value"] = df["jackpot"].str.replace("€", "").str.replace(",", "").astype(float)
        df["is_won"] = df["result"] == "Won"
        
        # Determine sequence length if not already set
        if self.sequence_length is None:
            optimal_length = Utilities.calculate_optimal_sequence_length(len(df))
            self.sequence_length = optimal_length
            logger.info(f"Dynamically set sequence length to {optimal_length} based on data size")
        else:
            logger.info(f"Using sequence length: {self.sequence_length}")
        
        self.data = df
        logger.info(f"Successfully parsed {len(df)} draws")
        
        return df
    
    @ErrorHandler.handle_exception(logger, "number expansion", pd.DataFrame())
    def expand_numbers(self):
        """Expand main and bonus numbers into individual columns with enhanced metadata."""
        if self.data is None or self.data.empty:
            logger.error("No data available. Parse the file first.")
            return pd.DataFrame()
        
        df = self.data.copy()
        
        # Expand main numbers (1-50)
        for i in range(MAIN_NUM_COUNT):
            df[f"main_{i+1}"] = df["main_numbers"].apply(lambda x: x[i] if i < len(x) else None)
        
        # Expand bonus numbers (1-12)
        for i in range(BONUS_NUM_COUNT):
            df[f"bonus_{i+1}"] = df["bonus_numbers"].apply(lambda x: x[i] if i < len(x) else None)
        
        # Add metadata about the draw
        df["draw_index"] = range(len(df))
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month
        df["day"] = df["date"].dt.day
        df["day_of_week_num"] = df["date"].dt.dayofweek
        
        # Handle the isocalendar() approach
        try:
            df["week_of_year"] = df["date"].dt.isocalendar().week
        except:
            try:
                df["week_of_year"] = df["date"].dt.weekofyear
            except:
                # Manual calculation for week of year
                df["week_of_year"] = df["date"].apply(lambda x: x.isocalendar()[1])
        
        # Holiday seasons and special periods
        df["is_holiday_season"] = ((df["month"] == 12) & (df["day"] >= 15)) | ((df["month"] == 1) & (df["day"] <= 15))
        df["is_summer"] = (df["month"] >= 6) & (df["month"] <= 8)
        df["is_weekend"] = df["day_of_week_num"].isin([5, 6])  # Saturday and Sunday
        
        # Calculate days since last draw
        df["days_since_last_draw"] = (df["date"] - df["date"].shift(1)).dt.days
        df["days_since_last_draw"].fillna(0, inplace=True)
        
        # Number range metrics
        df["main_number_range"] = df["main_numbers"].apply(lambda x: max(x) - min(x) if x else 0)
        df["main_number_sum"] = df["main_numbers"].apply(lambda x: sum(x) if x else 0)
        df["bonus_number_diff"] = df["bonus_numbers"].apply(lambda x: x[1] - x[0] if len(x) >= 2 else 0)
        
        # Distribution metrics
        df["main_even_count"] = df["main_numbers"].apply(lambda x: sum(1 for n in x if n % 2 == 0))
        df["main_odd_count"] = df["main_numbers"].apply(lambda x: sum(1 for n in x if n % 2 == 1))
        
        # Decade distribution (1-10, 11-20, etc.)
        for decade in range(5):
            start = decade * 10 + 1
            end = start + 9
            df[f"main_decade_{decade+1}_count"] = df["main_numbers"].apply(
                lambda x: sum(1 for n in x if start <= n <= end)
            )
        
        self.expanded_data = df
        return df
    
    def create_features(self):
        """Generate all features for lottery prediction."""
        if self.expanded_data is None or self.expanded_data.empty:
            self.expanded_data = self.expand_numbers()
            if self.expanded_data.empty:
                return pd.DataFrame()
                
        # Create feature engineering instance
        self.feature_engineer = FeatureEngineering(self.expanded_data)
        self.features = self.feature_engineer.create_enhanced_features()
        return self.features

#######################
# FEATURE ENGINEERING
#######################

class FeatureEngineering:
    """Advanced feature engineering for lottery prediction."""
    
    def __init__(self, data):
        self.data = data
        self.historical_counts = None
        self.pattern_clusters = None
        self._initialize_number_statistics()
    
    @ErrorHandler.handle_exception(logger, "number statistics initialization")
    def _initialize_number_statistics(self):
        """Initialize historical statistics for each number."""
        if self.data.empty:
            raise ValueError("Empty dataset provided")
            
        main_counts = np.zeros(MAIN_NUM_MAX)
        bonus_counts = np.zeros(BONUS_NUM_MAX)
        
        for _, row in self.data.iterrows():
            for num in row["main_numbers"]:
                if MAIN_NUM_MIN <= num <= MAIN_NUM_MAX:
                    main_counts[num-1] += 1
            for num in row["bonus_numbers"]:
                if BONUS_NUM_MIN <= num <= BONUS_NUM_MAX:
                    bonus_counts[num-1] += 1
        
        self.historical_counts = {
            "main": main_counts / len(self.data) if len(self.data) > 0 else main_counts,
            "bonus": bonus_counts / len(self.data) if len(self.data) > 0 else bonus_counts
        }
    
    @ErrorHandler.handle_exception(logger, "feature engineering", pd.DataFrame())
    def create_enhanced_features(self):
        """Create and combine all advanced features."""
        logger.info("Generating enhanced features for lottery prediction")
        
        # Generate all features in one efficient pass
        feature_sets = self._create_all_feature_sets()
        
        # Combine all feature sets
        all_features = pd.concat(feature_sets, axis=1)
        
        # Handle missing and infinite values
        all_features = all_features.fillna(0)
        all_features = self._fix_infinite_values(all_features)
        
        # Select top features to avoid feature explosion
        all_features = self._select_top_features(all_features)
        
        logger.info(f"Created {all_features.shape[1]} enhanced features")
        return all_features
    
    def _create_all_feature_sets(self):
        """Create all feature sets in one function to avoid multiple passes."""
        feature_sets = []
        
        # Calculate time series features with window-based metrics
        ts_features = self._calculate_time_series_features()
        feature_sets.append(ts_features)
        
        # Calculate number relationships
        relationship_features = self._calculate_number_relationships()
        feature_sets.append(relationship_features)
        
        # Calculate pattern features
        pattern_features = self._calculate_pattern_features()
        feature_sets.append(pattern_features)
        
        # Calculate cyclical time features
        cyclical_features = self._calculate_cyclical_features()
        feature_sets.append(cyclical_features)
        
        # Calculate auto-correlation features
        autocorr_features = self._calculate_autocorrelation_features()
        feature_sets.append(autocorr_features)
        
        # Enhanced statistical features
        stat_features = self._calculate_statistical_features()
        feature_sets.append(stat_features)
        
        return [df for df in feature_sets if not df.empty]
    
    def _fix_infinite_values(self, data_frame):
        """Handle infinite or extremely large values in the DataFrame."""
        try:
            # Replace infinities with NaN
            data_frame.replace([np.inf, -np.inf], np.nan, inplace=True)
            
            # Fill NaN values with 0
            data_frame.fillna(0, inplace=True)
            
            # Clip extremely large values to a reasonable range
            numeric_cols = data_frame.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                try:
                    # Use quantile-based clipping for more robust outlier handling
                    q_lo = data_frame[col].quantile(0.01)
                    q_hi = data_frame[col].quantile(0.99)
                    
                    # Use a more conservative approach for clipping
                    range_val = q_hi - q_lo
                    if range_val > 0:  # Prevent division by zero
                        data_frame[col] = data_frame[col].clip(q_lo - 3*range_val, q_hi + 3*range_val)
                except Exception as e:
                    logger.warning(f"Error clipping values in column {col}: {e}")
                    # If clipping fails, just continue
                    continue
            
            return data_frame
        except Exception as e:
            logger.error(f"Error fixing infinite values: {e}")
            # Return original DataFrame if fixing fails
            return data_frame
    
    def _select_top_features(self, features, max_features=MAX_FEATURES):
        """Select top features based on variance and correlation."""
        try:
            if features.shape[1] <= max_features:
                return features
                
            # Calculate variance for each feature
            variances = features.var().sort_values(ascending=False)
            
            # Select features with highest variance
            high_var_features = list(variances.head(max_features * 2).index)
            
            # If we have enough features with variance, do correlation filtering
            if len(high_var_features) > max_features:
                # Calculate correlation matrix for high variance features
                corr_matrix = features[high_var_features].corr().abs()
                
                # Select features iteratively, removing highly correlated ones
                selected_features = [high_var_features[0]]  # Start with highest variance feature
                
                for feature in high_var_features[1:]:
                    # Check correlation with already selected features
                    correlated = False
                    for selected in selected_features:
                        if corr_matrix.loc[feature, selected] > 0.85:  # Correlation threshold
                            correlated = True
                            break
                            
                    if not correlated:
                        selected_features.append(feature)
                        
                    # Stop if we have enough features
                    if len(selected_features) >= max_features:
                        break
                        
                return features[selected_features]
            else:
                # Not enough features for correlation filtering
                return features[high_var_features]
                
        except Exception as e:
            logger.warning(f"Error selecting top features: {e}")
            # Return original features if selection fails
            return features
    
    def _calculate_time_series_features(self, window_sizes=[5, 10, 20, 30]):
        """Calculate time series features with improved windowing."""
        df = self.data.copy()
        
        # Create empty dictionary for time series features
        ts_features = {}
        
        # Create indicators for ALL main and bonus numbers
        indicators = {}
        
        # Track ALL main numbers (not just every 5th)
        for num in range(1, MAIN_NUM_MAX+1):
            indicators[f"main_has_{num}"] = df["main_numbers"].apply(lambda x: num in x).astype(int)
            
        # Track ALL bonus numbers (not just every 3rd)
        for num in range(1, BONUS_NUM_MAX+1):
            indicators[f"bonus_has_{num}"] = df["bonus_numbers"].apply(lambda x: num in x).astype(int)
        
        # Convert indicators to DataFrame
        indicators_df = pd.DataFrame(indicators, index=df.index)
        
        # Adjust window sizes based on data length
        valid_window_sizes = [w for w in window_sizes if w < len(df)]
        
        # Calculate rolling statistics with adaptive windows
        for window in valid_window_sizes:
            # Use every 3rd main number to reduce feature count but maintain coverage
            for num in range(1, MAIN_NUM_MAX+1, 3):
                # Exponentially weighted moving average (more weight to recent draws)
                ts_features[f"main_{num}_ewma_{window}"] = indicators_df[f"main_has_{num}"].ewm(span=window, adjust=False).mean()
                
                # Mean reversion indicator - deviation from historical mean
                ts_features[f"main_{num}_mean_rev_{window}"] = (
                    indicators_df[f"main_has_{num}"].rolling(window=window).mean() - 
                    self.historical_counts["main"][num-1]
                )
                
                # Rolling standard deviation - to detect changes in volatility
                ts_features[f"main_{num}_std_{window}"] = indicators_df[f"main_has_{num}"].rolling(window=window).std().fillna(0)
                
            # Use every 2nd bonus number for better coverage
            for num in range(1, BONUS_NUM_MAX+1, 2):
                ts_features[f"bonus_{num}_ewma_{window}"] = indicators_df[f"bonus_has_{num}"].ewm(span=window, adjust=False).mean()
                
                # Mean reversion indicator
                ts_features[f"bonus_{num}_mean_rev_{window}"] = (
                    indicators_df[f"bonus_has_{num}"].rolling(window=window).mean() - 
                    self.historical_counts["bonus"][num-1]
                )
                
                # Rolling standard deviation
                ts_features[f"bonus_{num}_std_{window}"] = indicators_df[f"bonus_has_{num}"].rolling(window=window).std().fillna(0)
        
        # Convert to DataFrame and handle NaN values
        ts_df = pd.DataFrame(ts_features, index=df.index)
        return ts_df.fillna(0)
    
    def _calculate_number_relationships(self):
        """Calculate enhanced features based on relationships between numbers."""
        df = self.data.copy()
        
        # Vectorized calculation of statistics for all draws
        main_nums_array = np.array([np.array(lst) for lst in df["main_numbers"].tolist()])
        bonus_nums_array = np.array([np.array(lst) for lst in df["bonus_numbers"].tolist()])
        
        # Pre-allocate arrays for better performance
        n_samples = len(df)
        main_mean_diff = np.zeros(n_samples)
        main_std_diff = np.zeros(n_samples)
        main_min_diff = np.zeros(n_samples)
        main_max_diff = np.zeros(n_samples)
        main_range = np.zeros(n_samples)
        main_sum = np.zeros(n_samples)
        main_mean = np.zeros(n_samples)
        main_low_count = np.zeros(n_samples)
        main_high_count = np.zeros(n_samples)
        main_odd_count = np.zeros(n_samples)
        main_even_count = np.zeros(n_samples)
        main_consecutive_count = np.zeros(n_samples)
        main_median = np.zeros(n_samples)
        main_mode = np.zeros(n_samples)
        
        bonus_diff = np.zeros(n_samples)
        bonus_sum = np.zeros(n_samples)
        bonus_mean = np.zeros(n_samples)
        bonus_odd_count = np.zeros(n_samples)
        bonus_even_count = np.zeros(n_samples)
        
        # Calculate statistics in a loop to handle variable-length arrays
        for i, (main_nums, bonus_nums) in enumerate(zip(main_nums_array, bonus_nums_array)):
            # Main numbers statistics
            if len(main_nums) >= MAIN_NUM_COUNT:
                try:
                    # Calculate differences between consecutive numbers
                    diffs = np.diff(main_nums)
                    
                    # Store statistics about differences
                    main_mean_diff[i] = np.mean(diffs)
                    main_std_diff[i] = np.std(diffs)
                    main_min_diff[i] = np.min(diffs)
                    main_max_diff[i] = np.max(diffs)
                    main_range[i] = main_nums[-1] - main_nums[0]
                    
                    # Calculate sum, mean, median, mode
                    main_sum[i] = np.sum(main_nums)
                    main_mean[i] = np.mean(main_nums)
                    main_median[i] = np.median(main_nums)
                    
                    # Simplified mode calculation
                    try:
                        # This won't work for all cases but provides a basic estimate
                        # Real mode calculation would need a full frequency count
                        main_mode[i] = np.mean([num for num in main_nums if num > 20 and num < 40])
                    except:
                        main_mode[i] = main_mean[i]  # Fallback to mean
                    
                    # Distribution statistics
                    main_low_count[i] = np.sum(main_nums <= 25)
                    main_high_count[i] = np.sum(main_nums > 25)
                    main_odd_count[i] = np.sum(main_nums % 2 == 1)
                    main_even_count[i] = np.sum(main_nums % 2 == 0)
                    
                    # Consecutive numbers
                    main_consecutive_count[i] = np.sum(diffs == 1)
                except Exception as e:
                    # Keep zeros for failed calculations
                    continue
            
            # Bonus numbers statistics
            if len(bonus_nums) >= BONUS_NUM_COUNT:
                try:
                    bonus_diff[i] = bonus_nums[1] - bonus_nums[0] if len(bonus_nums) > 1 else 0
                    bonus_sum[i] = np.sum(bonus_nums)
                    bonus_mean[i] = np.mean(bonus_nums)
                    bonus_odd_count[i] = np.sum(bonus_nums % 2 == 1)
                    bonus_even_count[i] = np.sum(bonus_nums % 2 == 0)
                except Exception:
                    # Keep zeros for failed calculations
                    continue
        
        # Compute derived features
        # Main-bonus relationships
        main_bonus_ratio = np.zeros(n_samples)
        main_bonus_correlation = np.zeros(n_samples)
        
        for i, (main_nums, bonus_nums) in enumerate(zip(main_nums_array, bonus_nums_array)):
            if len(main_nums) >= MAIN_NUM_COUNT and len(bonus_nums) >= BONUS_NUM_COUNT:
                try:
                    # Ratio of main numbers mean to bonus numbers mean
                    main_bonus_ratio[i] = np.mean(main_nums) / np.mean(bonus_nums) if np.mean(bonus_nums) > 0 else 0
                    
                    # Simple correlation estimate between main and bonus numbers
                    # (not a true correlation but a proxy for relationship)
                    main_bonus_correlation[i] = (
                        np.sum(main_nums <= 25) / len(main_nums) - 
                        np.sum(bonus_nums <= 6) / len(bonus_nums)
                    )
                except:
                    continue
        
        # Convert to DataFrame
        relationship_features = {
            'main_mean_diff': main_mean_diff,
            'main_std_diff': main_std_diff,
            'main_min_diff': main_min_diff,
            'main_max_diff': main_max_diff,
            'main_range': main_range,
            'main_sum': main_sum,
            'main_mean': main_mean,
            'main_median': main_median,
            'main_mode': main_mode,
            'main_low_count': main_low_count,
            'main_high_count': main_high_count,
            'main_odd_count': main_odd_count,
            'main_even_count': main_even_count,
            'main_consecutive_count': main_consecutive_count,
            'bonus_diff': bonus_diff,
            'bonus_sum': bonus_sum,
            'bonus_mean': bonus_mean,
            'bonus_odd_count': bonus_odd_count,
            'bonus_even_count': bonus_even_count,
            'main_bonus_ratio': main_bonus_ratio,
            'main_bonus_correlation': main_bonus_correlation
        }
        
        return pd.DataFrame(relationship_features, index=df.index)
    
    def _calculate_pattern_features(self):
        """Calculate features based on number patterns and clusters with improved metrics."""
        df = self.data.copy()
        
        # Pre-allocate pattern features
        n_samples = len(df)
        pattern_cluster = np.zeros(n_samples)
        pattern_distance = np.zeros(n_samples)
        digit_sum_score = np.zeros(n_samples)
        adjacent_count = np.zeros(n_samples)
        decade_spread = np.zeros(n_samples)
        
        # Create a signature for each draw's main numbers
        signatures = []
        valid_indices = []
        
        # Enhanced pattern features
        for i, (_, row) in enumerate(df.iterrows()):
            try:
                main_nums = row["main_numbers"]
                
                # Create a histogram of the numbers (count in each decade)
                bins = [1, 11, 21, 31, 41, 51]  # 1-10, 11-20, ..., 41-50
                hist, _ = np.histogram(main_nums, bins=bins)
                
                # Calculate digit sum (sum of individual digits in each number)
                # This is a common pattern analyzed in lottery systems
                digit_sum = sum(sum(int(digit) for digit in str(num)) for num in main_nums)
                digit_sum_score[i] = digit_sum / 50  # Normalize score
                
                # Count adjacent numbers
                sorted_nums = sorted(main_nums)
                adjacent_count[i] = sum(1 for j in range(len(sorted_nums)-1) if sorted_nums[j+1] - sorted_nums[j] == 1)
                
                # Calculate decade spread (how many decades are represented)
                decades_present = sum(1 for count in hist if count > 0)
                decade_spread[i] = decades_present / 5  # Normalize by total decades
                
                # Create pattern signature with decade distribution
                if len(main_nums) >= 2:
                    # Normalize the histogram
                    hist_norm = hist/np.sum(hist) if np.sum(hist) > 0 else hist
                    
                    # Add signature with draw index
                    signatures.append(hist_norm)
                    valid_indices.append(i)
            except Exception:
                continue
        
        # Create clusters of similar patterns if we have enough data
        min_signatures_for_clustering = max(8, min(50, n_samples // 8))
        if len(signatures) >= min_signatures_for_clustering:
            try:
                # Convert to array
                signatures_array = np.array(signatures)
                
                # Reduce dimensions with PCA first if we have enough samples
                if signatures_array.shape[0] > 2 and signatures_array.shape[1] > 1:
                    n_components = min(3, signatures_array.shape[1])
                    pca = PCA(n_components=n_components)
                    signatures_pca = pca.fit_transform(signatures_array)
                    
                    # Create clusters with optimal number based on data size
                    n_clusters = max(3, min(8, len(signatures) // 12))
                    
                    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED, n_init=10)
                    cluster_labels = kmeans.fit_predict(signatures_pca)
                    
                    # Save cluster information for later use
                    self.pattern_clusters = {
                        'kmeans': kmeans,
                        'pca': pca
                    }
                    
                    # Assign clusters and calculate distances
                    for idx, (orig_idx, cluster) in enumerate(zip(valid_indices, cluster_labels)):
                        pattern_cluster[orig_idx] = cluster
                        
                        # Distance to cluster center (normalized)
                        center = kmeans.cluster_centers_[cluster]
                        dist = np.linalg.norm(signatures_pca[idx] - center)
                        pattern_distance[orig_idx] = dist / np.sqrt(n_components)  # Normalize by dimensionality
                else:
                    # Just use simple clustering directly on signatures if dimensions not sufficient
                    kmeans = KMeans(n_clusters=2, random_state=RANDOM_SEED, n_init=10)
                    cluster_labels = kmeans.fit_predict(signatures_array)
                    
                    for idx, (orig_idx, cluster) in enumerate(zip(valid_indices, cluster_labels)):
                        pattern_cluster[orig_idx] = cluster
            except Exception as e:
                logger.warning(f"Pattern clustering failed: {e}")
        
        # Create the pattern feature DataFrame
        pattern_features = {
            'main_pattern_cluster': pattern_cluster,
            'main_pattern_distance': pattern_distance,
            'digit_sum_score': digit_sum_score,
            'adjacent_count': adjacent_count,
            'decade_spread': decade_spread
        }
        
        return pd.DataFrame(pattern_features, index=df.index)
    
    def _calculate_cyclical_features(self):
        """Calculate cyclical time features using sine/cosine transformations."""
        df = self.data.copy()
        
        # Vectorized implementation for efficiency
        dates = pd.Series(df["date"])
        
        # Day of week (0-6)
        day_of_week = dates.dt.dayofweek
        day_of_week_sin = np.sin(2 * np.pi * day_of_week / 7)
        day_of_week_cos = np.cos(2 * np.pi * day_of_week / 7)
        
        # Day of month (1-31)
        day_of_month = dates.dt.day
        day_of_month_sin = np.sin(2 * np.pi * day_of_month / 31)
        day_of_month_cos = np.cos(2 * np.pi * day_of_month / 31)
        
        # Month (1-12)
        month = dates.dt.month
        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)
        
        # Week of year (1-53)
        try:
            week_of_year = dates.dt.isocalendar().week
        except:
            try:
                # Fallback for older pandas versions
                week_of_year = dates.dt.weekofyear
            except:
                # Manual calculation for week of year
                week_of_year = dates.apply(lambda x: x.isocalendar()[1])
                
        week_of_year_sin = np.sin(2 * np.pi * week_of_year / 53)
        week_of_year_cos = np.cos(2 * np.pi * week_of_year / 53)
        
        # Quarter (1-4)
        quarter = dates.dt.quarter
        quarter_sin = np.sin(2 * np.pi * quarter / 4)
        quarter_cos = np.cos(2 * np.pi * quarter / 4)
        
        # Year progress (0-1)
        year_progress = (month - 1) / 12 + (day_of_month - 1) / 365
        year_progress_sin = np.sin(2 * np.pi * year_progress)
        year_progress_cos = np.cos(2 * np.pi * year_progress)
        
        # Create the cyclical features DataFrame
        cyclical_features = {
            'day_of_week_sin': day_of_week_sin,
            'day_of_week_cos': day_of_week_cos,
            'day_of_month_sin': day_of_month_sin,
            'day_of_month_cos': day_of_month_cos,
            'month_sin': month_sin,
            'month_cos': month_cos,
            'week_of_year_sin': week_of_year_sin,
            'week_of_year_cos': week_of_year_cos,
            'quarter_sin': quarter_sin,
            'quarter_cos': quarter_cos,
            'year_progress_sin': year_progress_sin,
            'year_progress_cos': year_progress_cos
        }
        
        return pd.DataFrame(cyclical_features, index=df.index)
    
    def _calculate_autocorrelation_features(self, max_lag=5):
        """Calculate autocorrelation features for number patterns."""
        df = self.data.copy()
        
        # Prepare data structures
        n_samples = len(df)
        
        # Return empty DataFrame if not enough samples
        if n_samples < max_lag + 5:
            return pd.DataFrame(index=df.index)
            
        autocorr_features = {}
        
        # Calculate autocorrelation for key metrics
        metrics = ['main_sum', 'main_range', 'main_consecutive_count']
        
        try:
            for metric in metrics:
                if metric in df.columns:
                    # Get metric series
                    series = df[metric].values
                    
                    # Normalize the series
                    series_norm = (series - np.mean(series)) / np.std(series) if np.std(series) > 0 else series
                    
                    # Calculate autocorrelation for different lags
                    for lag in range(1, min(max_lag + 1, n_samples // 3)):
                        # Shift the series
                        lagged = np.roll(series_norm, lag)
                        
                        # Calculate correlation
                        valid_idx = np.arange(lag, n_samples)
                        corr = np.corrcoef(series_norm[valid_idx], lagged[valid_idx])[0, 1]
                        
                        # Handle NaN values
                        if np.isnan(corr):
                            corr = 0
                            
                        # Create lagged version of the feature
                        autocorr_features[f'{metric}_lag_{lag}'] = np.zeros(n_samples)
                        autocorr_features[f'{metric}_lag_{lag}'][lag:] = series[:-lag]
                        
                        # Create autocorrelation feature
                        autocorr_features[f'{metric}_autocorr_{lag}'] = np.ones(n_samples) * corr
        except Exception as e:
            logger.warning(f"Error calculating autocorrelation features: {e}")
            
        # Create a DataFrame from the features
        autocorr_df = pd.DataFrame(autocorr_features, index=df.index)
        
        # Fill NaN values
        autocorr_df.fillna(0, inplace=True)
        
        return autocorr_df
    
    def _calculate_statistical_features(self):
        """Calculate additional statistical features for prediction enhancement."""
        df = self.data.copy()
        
        # Prepare data structures
        n_samples = len(df)
        
        # Create features dictionary
        stat_features = {}
        
        try:
            # Calculate number frequency distributions
            main_num_freq = np.zeros((n_samples, MAIN_NUM_MAX))
            bonus_num_freq = np.zeros((n_samples, BONUS_NUM_MAX))
            
            # Fill frequency matrices with historical data
            for i in range(1, min(n_samples, 100)):  # Look back up to 100 draws
                for j in range(i, n_samples):
                    for num in df.iloc[j-i]["main_numbers"]:
                        if 1 <= num <= MAIN_NUM_MAX:
                            main_num_freq[j, num-1] += 1 / i  # Weighted by recency
                    
                    for num in df.iloc[j-i]["bonus_numbers"]:
                        if 1 <= num <= BONUS_NUM_MAX:
                            bonus_num_freq[j, num-1] += 1 / i  # Weighted by recency
            
            # Normalize frequencies
            for i in range(n_samples):
                if np.sum(main_num_freq[i]) > 0:
                    main_num_freq[i] = main_num_freq[i] / np.sum(main_num_freq[i])
                if np.sum(bonus_num_freq[i]) > 0:
                    bonus_num_freq[i] = bonus_num_freq[i] / np.sum(bonus_num_freq[i])
            
            # Create aggregate frequency features
            # Frequency momentum (increasing or decreasing trend)
            main_momentum = np.zeros(n_samples)
            bonus_momentum = np.zeros(n_samples)
            
            # Calculate momentum for the latest draw's numbers compared to previous
            for i in range(5, n_samples):
                # Main number momentum
                current_main_nums = df.iloc[i]["main_numbers"]
                
                # Calculate average frequency change for current numbers
                main_change = 0
                for num in current_main_nums:
                    if 1 <= num <= MAIN_NUM_MAX:
                        if i > 0 and i-5 >= 0:
                            main_change += (main_num_freq[i, num-1] - main_num_freq[i-5, num-1])
                
                main_momentum[i] = main_change / len(current_main_nums) if current_main_nums else 0
                
                # Bonus number momentum
                current_bonus_nums = df.iloc[i]["bonus_numbers"]
                
                # Calculate average frequency change for current numbers
                bonus_change = 0
                for num in current_bonus_nums:
                    if 1 <= num <= BONUS_NUM_MAX:
                        if i > 0 and i-5 >= 0:
                            bonus_change += (bonus_num_freq[i, num-1] - bonus_num_freq[i-5, num-1])
                
                bonus_momentum[i] = bonus_change / len(current_bonus_nums) if current_bonus_nums else 0
            
            # Add momentum features
            stat_features['main_frequency_momentum'] = main_momentum
            stat_features['bonus_frequency_momentum'] = bonus_momentum
            
            # Calculate statistical features from the expanded data
            
            # Sum features at decade level (1-10, 11-20, etc.)
            for decade in range(5):
                decade_col = f'main_decade_{decade+1}_count'
                if decade_col in df.columns:
                    # Rolling statistics for each decade
                    window_sizes = [5, 10]
                    for window in window_sizes:
                        if n_samples > window:
                            # Rolling mean
                            stat_features[f'{decade_col}_mean_{window}'] = df[decade_col].rolling(window=window).mean().fillna(0)
                            # Rolling std
                            stat_features[f'{decade_col}_std_{window}'] = df[decade_col].rolling(window=window).std().fillna(0)
            
            # Number of draws since each number last appeared
            for num in range(1, MAIN_NUM_MAX+1, 5):  # Every 5th main number for efficiency
                last_seen = np.zeros(n_samples)
                counter = 0
                
                for i in range(n_samples):
                    if num in df.iloc[i]["main_numbers"]:
                        counter = 0
                    else:
                        counter += 1
                    last_seen[i] = counter
                
                stat_features[f'main_{num}_draws_since_last'] = last_seen
            
            for num in range(1, BONUS_NUM_MAX+1, 2):  # Every 2nd bonus number
                last_seen = np.zeros(n_samples)
                counter = 0
                
                for i in range(n_samples):
                    if num in df.iloc[i]["bonus_numbers"]:
                        counter = 0
                    else:
                        counter += 1
                    last_seen[i] = counter
                
                stat_features[f'bonus_{num}_draws_since_last'] = last_seen
        
        except Exception as e:
            logger.warning(f"Error calculating statistical features: {e}")
            
        # Create DataFrame and handle missing values
        stat_df = pd.DataFrame(stat_features, index=df.index)
        stat_df.fillna(0, inplace=True)
        
        return stat_df

#######################
# TRANSFORMER MODEL COMPONENTS
#######################

class TransformerBlock(tf.keras.layers.Layer):
    """Transformer block with multi-head attention and feed-forward network."""
    
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=DEFAULT_DROPOUT_RATE, 
                use_residual=True, use_layer_scaling=True):
        super(TransformerBlock, self).__init__()
        
        # Store configuration
        self.embed_dim = embed_dim
        self.use_residual = use_residual
        self.use_layer_scaling = use_layer_scaling
        
        # Pre-compute key dimension for efficiency
        key_dim = embed_dim // num_heads
        
        # Multi-head attention with improved configuration
        self.att = MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=key_dim, 
            dropout=dropout,
            use_bias=True,  # Enable bias for better expressivity
        )
        
        # Feed-forward network with GeLU activation (better performance than ReLU)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="gelu"),
            Dropout(dropout),
            Dense(embed_dim)
        ])
        
        # Layer normalization with improved epsilon
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        
        # Layer scaling factors (if enabled)
        if use_layer_scaling:
            self.layer_scale1 = self.add_weight(
                name="layer_scale1",
                shape=(embed_dim,),
                initializer=tf.constant_initializer(1.0),
                trainable=True,
            )
            self.layer_scale2 = self.add_weight(
                name="layer_scale2",
                shape=(embed_dim,),
                initializer=tf.constant_initializer(1.0),
                trainable=True,
            )
        
        # Dropout layers with shared rate
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        
    def call(self, inputs, training=False):
        # Pre-norm architecture (more stable training)
        x = self.layernorm1(inputs)
        attn_output = self.att(x, x)
        attn_output = self.dropout1(attn_output, training=training)
        
        # Apply layer scaling if enabled
        if self.use_layer_scaling:
            attn_output = attn_output * self.layer_scale1
            
        # Apply residual connection if enabled
        if self.use_residual:
            out1 = inputs + attn_output
        else:
            out1 = attn_output
        
        # Feed-forward network with pre-norm
        x = self.layernorm2(out1)
        ffn_output = self.ffn(x)
        ffn_output = self.dropout2(ffn_output, training=training)
        
        # Apply layer scaling if enabled
        if self.use_layer_scaling:
            ffn_output = ffn_output * self.layer_scale2
            
        # Apply residual connection if enabled
        if self.use_residual:
            return out1 + ffn_output
        else:
            return ffn_output

class PositionalEncoding(tf.keras.layers.Layer):
    """Improved positional encoding layer for transformer models."""
    
    def __init__(self, max_position, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(max_position, d_model)
        
    def get_angles(self, position, i, d_model):
        # Vectorized angle calculation
        angles = 1 / tf.pow(10000.0, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return position * angles
        
    def positional_encoding(self, max_position, d_model):
        # Vectorized position encoding calculation
        position_range = tf.range(max_position, dtype=tf.float32)[:, tf.newaxis]
        dimension_range = tf.range(d_model, dtype=tf.float32)[tf.newaxis, :]
        
        angle_rads = self.get_angles(position_range, dimension_range, d_model)
        
        # Apply sin to even indices, cos to odd indices
        sin_indices = tf.range(0, d_model, 2, dtype=tf.int32)
        cos_indices = tf.range(1, d_model, 2, dtype=tf.int32)
        
        # Optimized calculation using tf.tensor_scatter_nd_update
        pe = tf.zeros([max_position, d_model], dtype=tf.float32)
        
        # Update sin positions
        if tf.size(sin_indices) > 0:
            updates_sin = tf.sin(tf.gather(angle_rads, sin_indices, axis=1))
            indices_sin = tf.stack([
                tf.repeat(tf.range(max_position), tf.size(sin_indices)),
                tf.tile(sin_indices, [max_position])
            ], axis=1)
            pe = tf.tensor_scatter_nd_update(pe, indices_sin, tf.reshape(updates_sin, [-1]))
        
        # Update cos positions
        if tf.size(cos_indices) > 0:
            updates_cos = tf.cos(tf.gather(angle_rads, cos_indices, axis=1))
            indices_cos = tf.stack([
                tf.repeat(tf.range(max_position), tf.size(cos_indices)),
                tf.tile(cos_indices, [max_position])
            ], axis=1)
            pe = tf.tensor_scatter_nd_update(pe, indices_cos, tf.reshape(updates_cos, [-1]))
        
        # Add batch dimension
        pe = tf.expand_dims(pe, 0)
        
        return tf.cast(pe, tf.float32)
        
    def call(self, inputs):
        # Add positional encoding to input embeddings (using efficient slicing)
        seq_len = tf.shape(inputs)[1]
        return inputs + self.pos_encoding[:, :seq_len, :]

#######################
# MODEL ARCHITECTURE
#######################

class ModelBuilder:
    """Unified model building for lottery prediction models."""
    
    @staticmethod
    @ErrorHandler.handle_exception(logger, "model building")
    def build_transformer_model(input_dim, seq_length, params, model_config):
        """Build transformer model for lottery number prediction with flexible configuration."""
        # Extract model configuration
        num_outputs = model_config['num_outputs'] 
        output_size = model_config['output_size']
        sequence_size = model_config['sequence_size']
        name_prefix = model_config['name_prefix']
        
        # Extract parameters with defaults
        l2_reg = params.get('l2_regularization', 0.0001)
        use_residual = params.get('use_residual', True)
        use_layer_scaling = params.get('use_layer_scaling', True)
        model_type = params.get('model_type', 'transformer')
        lstm_units = params.get('lstm_units', 32)
        
        # Input layers - ensure consistent naming
        feature_input = Input(shape=(input_dim,), name=f"{name_prefix}_feature_input")
        sequence_input = Input(shape=(seq_length, sequence_size), name=f"{name_prefix}_sequence_input")
        
        # Process feature input with regularization
        x_features = Dense(
            params.get('embed_dim', DEFAULT_EMBED_DIM), 
            activation="gelu",
            kernel_regularizer=tf.keras.regularizers.l2(l2_reg)
        )(feature_input)
        x_features = BatchNormalization()(x_features)
        x_features = Dropout(params.get('dropout_rate', DEFAULT_DROPOUT_RATE))(x_features)
        
        x_features = Dense(
            params.get('embed_dim', DEFAULT_EMBED_DIM) // 2, 
            activation="gelu",
            kernel_regularizer=tf.keras.regularizers.l2(l2_reg)
        )(x_features)
        
        # Process sequence input based on model type
        if model_type == 'transformer':
            # Transformer-based architecture (default)
            
            # Fix: Ensure conv_filters is at least 1
            conv_filters = max(1, params.get('conv_filters', DEFAULT_CONV_FILTERS))
            
            # Optional convolutional layer for sequence processing
            if conv_filters > 0:  # This check is redundant now but kept for clarity
                x_seq = Conv1D(
                    conv_filters, 
                    kernel_size=3, 
                    padding='same', 
                    activation="gelu",
                    kernel_regularizer=tf.keras.regularizers.l2(l2_reg)
                )(sequence_input)
                x_seq = BatchNormalization()(x_seq)
            else:
                x_seq = sequence_input
                    
            # Embedding layer
            x_seq = Dense(
                params.get('embed_dim', DEFAULT_EMBED_DIM),
                kernel_regularizer=tf.keras.regularizers.l2(l2_reg)
            )(x_seq)
            
            # Apply positional encoding
            x_seq = PositionalEncoding(
                max_position=seq_length + 5,  # Add padding for safety
                d_model=params.get('embed_dim', DEFAULT_EMBED_DIM)
            )(x_seq)
            
            # Apply transformer blocks
            for _ in range(params.get('num_transformer_blocks', DEFAULT_TRANSFORMER_BLOCKS)):
                x_seq = TransformerBlock(
                    embed_dim=params.get('embed_dim', DEFAULT_EMBED_DIM),
                    num_heads=params.get('num_heads', DEFAULT_NUM_HEADS),
                    ff_dim=params.get('ff_dim', DEFAULT_FF_DIM),
                    dropout=params.get('dropout_rate', DEFAULT_DROPOUT_RATE),
                    use_residual=use_residual,
                    use_layer_scaling=use_layer_scaling
                )(x_seq)
            
            # Process sequence with GRU or global pooling
            if params.get('use_gru', True):
                x_seq = Bidirectional(GRU(
                    params.get('embed_dim', DEFAULT_EMBED_DIM) // 2, 
                    return_sequences=False,
                    kernel_regularizer=tf.keras.regularizers.l2(l2_reg)
                ))(x_seq)
            else:
                x_seq = GlobalAveragePooling1D()(x_seq)
                    
        elif model_type == 'lstm':
            # LSTM-based architecture
            # Fix: Ensure conv_filters is at least 1
            conv_filters = max(1, params.get('conv_filters', DEFAULT_CONV_FILTERS))
            
            x_seq = Conv1D(
                conv_filters, 
                kernel_size=3, 
                padding='same', 
                activation="gelu",
                kernel_regularizer=tf.keras.regularizers.l2(l2_reg)
            )(sequence_input)
            
            x_seq = BatchNormalization()(x_seq)
            
            # Bidirectional LSTM
            x_seq = Bidirectional(LSTM(
                lstm_units,
                return_sequences=True,
                kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
                recurrent_regularizer=tf.keras.regularizers.l2(l2_reg),
                dropout=params.get('dropout_rate', DEFAULT_DROPOUT_RATE) * 0.5,
                recurrent_dropout=params.get('dropout_rate', DEFAULT_DROPOUT_RATE) * 0.5
            ))(x_seq)
            
            # Another LSTM layer
            x_seq = LSTM(
                lstm_units,
                return_sequences=False,
                kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
                recurrent_regularizer=tf.keras.regularizers.l2(l2_reg),
                dropout=params.get('dropout_rate', DEFAULT_DROPOUT_RATE) * 0.5,
                recurrent_dropout=params.get('dropout_rate', DEFAULT_DROPOUT_RATE) * 0.5
            )(x_seq)
            
        elif model_type == 'rnn':
            # Simple RNN architecture (lighter weight)
            x_seq = SimpleRNN(
                params.get('embed_dim', DEFAULT_EMBED_DIM),
                return_sequences=True,
                kernel_regularizer=tf.keras.regularizers.l2(l2_reg)
            )(sequence_input)
            
            x_seq = Dropout(params.get('dropout_rate', DEFAULT_DROPOUT_RATE))(x_seq)
            
            x_seq = SimpleRNN(
                params.get('embed_dim', DEFAULT_EMBED_DIM) // 2,
                return_sequences=False,
                kernel_regularizer=tf.keras.regularizers.l2(l2_reg)
            )(x_seq)
        
        else:
            # Fallback to CNN architecture
            # Fix: Ensure conv_filters is at least 1
            conv_filters = max(1, params.get('conv_filters', DEFAULT_CONV_FILTERS))
            
            x_seq = Conv1D(
                conv_filters * 2, 
                kernel_size=5, 
                padding='same', 
                activation="gelu",
                kernel_regularizer=tf.keras.regularizers.l2(l2_reg)
            )(sequence_input)
            
            x_seq = BatchNormalization()(x_seq)
            x_seq = Dropout(params.get('dropout_rate', DEFAULT_DROPOUT_RATE))(x_seq)
            
            x_seq = Conv1D(
                conv_filters, 
                kernel_size=3, 
                padding='same', 
                activation="gelu",
                kernel_regularizer=tf.keras.regularizers.l2(l2_reg)
            )(x_seq)
            
            x_seq = GlobalAveragePooling1D()(x_seq)
        
        # Combine feature and sequence representations
        combined = Concatenate()([x_features, x_seq])
        
        # Dense layers for combined processing with regularization
        combined = Dense(
            params.get('ff_dim', DEFAULT_FF_DIM), 
            activation="gelu",
            kernel_regularizer=tf.keras.regularizers.l2(l2_reg)
        )(combined)
        combined = BatchNormalization()(combined)
        combined = Dropout(params.get('dropout_rate', DEFAULT_DROPOUT_RATE))(combined)
        
        combined = Dense(
            params.get('ff_dim', DEFAULT_FF_DIM) // 2, 
            activation="gelu",
            kernel_regularizer=tf.keras.regularizers.l2(l2_reg)
        )(combined)
        combined = Dropout(params.get('dropout_rate', DEFAULT_DROPOUT_RATE) / 2)(combined)
        
        # Output layers (one for each position)
        outputs = []
        for i in range(num_outputs):
            # Position-specific processing
            position_specific = Dense(
                params.get('ff_dim', DEFAULT_FF_DIM) // 4, 
                activation="gelu",
                kernel_regularizer=tf.keras.regularizers.l2(l2_reg)
            )(combined)
            
            logits = Dense(
                output_size, 
                activation=None, 
                name=f"{name_prefix}_logits_{i+1}",
                kernel_regularizer=tf.keras.regularizers.l2(l2_reg)
            )(position_specific)
            
            output = Activation('softmax', name=f"{name_prefix}_{i+1}")(logits)
            outputs.append(output)
        
        # Create model with explicit list inputs
        model = Model(inputs=[feature_input, sequence_input], outputs=outputs)
        
        # Compile model with metrics
        metrics_dict = {f"{name_prefix}_{i+1}": "accuracy" for i in range(num_outputs)}
        
        # Configure optimizer
        optimizer_name = params.get('optimizer', 'adam').lower()
        learning_rate = params.get('learning_rate', DEFAULT_LEARNING_RATE)
        
        if optimizer_name == 'adam':
            optimizer = Adam(learning_rate=learning_rate)
        elif optimizer_name == 'sgd':
            optimizer = SGD(learning_rate=learning_rate, momentum=0.9)
        elif optimizer_name == 'rmsprop':
            optimizer = RMSprop(learning_rate=learning_rate)
        else:
            optimizer = Adam(learning_rate=learning_rate)
        
        model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=metrics_dict
        )
        
        return model

#######################
# LOTTERY PREDICTOR
#######################

class LotteryModel:
    """Unified model for lottery prediction."""
    
    def __init__(self, params=None):
        """Initialize the lottery model."""
        # Default parameters
        self.params = Utilities.get_default_params() if params is None else params
        self.main_model = None
        self.bonus_model = None
        
    def build_models(self, input_dim, seq_length):
        """Build both main and bonus number models using the same framework."""
        # Define model configurations
        main_config = {
            'num_outputs': MAIN_NUM_COUNT,
            'output_size': MAIN_NUM_MAX,
            'sequence_size': MAIN_NUM_MAX,
            'name_prefix': 'main'
        }
        
        bonus_config = {
            'num_outputs': BONUS_NUM_COUNT,
            'output_size': BONUS_NUM_MAX,
            'sequence_size': BONUS_NUM_MAX,
            'name_prefix': 'bonus'
        }
        
        # Build models with shared architecture but different configurations
        self.main_model = ModelBuilder.build_transformer_model(
            input_dim=input_dim,
            seq_length=seq_length,
            params=self.params,
            model_config=main_config
        )
        
        self.bonus_model = ModelBuilder.build_transformer_model(
            input_dim=input_dim,
            seq_length=seq_length,
            params=self.params,
            model_config=bonus_config
        )
        
        logger.info("Built transformer models for main and bonus numbers")
        return self.main_model, self.bonus_model
    
    @ErrorHandler.handle_exception(logger, "model training")
    def train_models(self, X_train, main_seq_train, bonus_seq_train, 
                    y_main_train, y_bonus_train, validation_split=0.1, 
                    epochs=None, batch_size=None):
        """Train both models with unified approach."""
        if self.main_model is None or self.bonus_model is None:
            raise ValueError("Models not built. Call build_models() first.")
        
        # Use parameter values or defaults
        actual_epochs = epochs if epochs is not None else self.params.get('epochs', DEFAULT_EPOCHS)
        actual_batch_size = batch_size if batch_size is not None else self.params.get('batch_size', DEFAULT_BATCH_SIZE)
        min_delta = self.params.get('min_delta', MIN_DELTA)
        
        # Get callbacks for both models at once
        callbacks_dict, _ = Utilities.get_model_callbacks(
            model_types=["main", "bonus"], 
            patience=DEFAULT_PATIENCE,
            min_delta=min_delta
        )
        
        # Flatten target arrays
        y_main_train_flat = [y_main_train[:, j].flatten() for j in range(MAIN_NUM_COUNT)]
        
        # Ensure inputs are numpy arrays with correct type
        X_train_arr = np.asarray(X_train)
        main_seq_train_arr = np.asarray(main_seq_train)
        
        # Train main model with explicitly formatted inputs
        logger.info("Training main numbers model")
        main_history = self.main_model.fit(
            [X_train_arr, main_seq_train_arr],  # Pass as list instead of dict
            y_main_train_flat,
            epochs=actual_epochs,
            batch_size=actual_batch_size,
            validation_split=validation_split,
            callbacks=callbacks_dict["main"],
            verbose=1
        )
        
        # Save training history
        try:
            with open("main_model_history.json", "w") as f:
                json.dump({
                    "loss": [float(x) for x in main_history.history["loss"]],
                    "val_loss": [float(x) for x in main_history.history["val_loss"]]
                }, f, indent=4)
        except Exception as e:
            logger.warning(f"Could not save main model history: {e}")
        
        # Clean memory before training bonus model
        Utilities.clean_memory()
        
        # Flatten bonus target arrays
        y_bonus_train_flat = [y_bonus_train[:, j].flatten() for j in range(BONUS_NUM_COUNT)]
        
        # Ensure bonus inputs are numpy arrays
        bonus_seq_train_arr = np.asarray(bonus_seq_train)
        
        # Train bonus model with explicitly formatted inputs
        logger.info("Training bonus numbers model")
        bonus_history = self.bonus_model.fit(
            [X_train_arr, bonus_seq_train_arr],  # Pass as list instead of dict
            y_bonus_train_flat,
            epochs=actual_epochs,
            batch_size=actual_batch_size,
            validation_split=validation_split,
            callbacks=callbacks_dict["bonus"],
            verbose=1
        )
        
        # Save training history
        try:
            with open("bonus_model_history.json", "w") as f:
                json.dump({
                    "loss": [float(x) for x in bonus_history.history["loss"]],
                    "val_loss": [float(x) for x in bonus_history.history["val_loss"]]
                }, f, indent=4)
        except Exception as e:
            logger.warning(f"Could not save bonus model history: {e}")
        
        # Clean memory after training
        Utilities.clean_memory(force=True)
        
        return {"main": main_history, "bonus": bonus_history}
    
    @ErrorHandler.handle_exception(logger, "prediction", [])
    def predict(self, features, main_sequence, bonus_sequence, num_draws=5, 
            temperature=DEFAULT_TEMPERATURE, diversity_sampling=True):
        """Generate predictions using the model."""
        if self.main_model is None or self.bonus_model is None:
            raise ValueError("Models not trained. Train models first.")
        
        # Ensure inputs are numpy arrays
        features_arr = np.asarray(features)
        main_sequence_arr = np.asarray(main_sequence)
        bonus_sequence_arr = np.asarray(bonus_sequence)
        
        # Track used numbers for diversity
        used_main_numbers = set()
        used_bonus_numbers = set()
        
        # Generate predictions
        predictions = []
        
        for draw_idx in range(num_draws):
            # Predict main numbers with explicit list inputs
            main_probs = self.main_model.predict([features_arr, main_sequence_arr], verbose=0)
            
            # Predict bonus numbers with explicit list inputs
            bonus_probs = self.bonus_model.predict([features_arr, bonus_sequence_arr], verbose=0)
            
            # Calculate confidence scores for dynamic temperature
            main_confidence = np.mean([np.max(main_probs[i][0]) for i in range(MAIN_NUM_COUNT)])
            bonus_confidence = np.mean([np.max(bonus_probs[i][0]) for i in range(BONUS_NUM_COUNT)])
            
            # Calculate overall confidence for temperature adjustment
            overall_confidence = (main_confidence + bonus_confidence) / 2
            
            # Apply dynamic temperature based on confidence
            dynamic_temp = Utilities.calculate_dynamic_temperature(overall_confidence, 
                                                                min_temp=MIN_TEMPERATURE,
                                                                max_temp=MAX_TEMPERATURE)
            
            # Use original temperature for first draw, then use dynamic temperature
            actual_temp = temperature if draw_idx == 0 else dynamic_temp
            
            # Log temperature adjustment for debugging
            if draw_idx > 0 and abs(actual_temp - temperature) > 0.1:
                logger.info(f"Dynamic temperature adjustment: {temperature} -> {actual_temp} (confidence: {overall_confidence:.4f})")
            
            # Sample main numbers
            main_numbers = Utilities.sample_numbers(
                probs=main_probs,
                available_nums=range(MAIN_NUM_MIN, MAIN_NUM_MAX+1),
                num_to_select=MAIN_NUM_COUNT,
                used_nums=used_main_numbers,
                diversity_sampling=diversity_sampling,
                draw_idx=draw_idx,
                temperature=actual_temp,
                confidence=overall_confidence
            )
            
            # Sample bonus numbers
            bonus_numbers = Utilities.sample_numbers(
                probs=bonus_probs,
                available_nums=range(BONUS_NUM_MIN, BONUS_NUM_MAX+1),
                num_to_select=BONUS_NUM_COUNT,
                used_nums=used_bonus_numbers,
                diversity_sampling=diversity_sampling,
                draw_idx=draw_idx,
                temperature=actual_temp,
                confidence=overall_confidence
            )
            
            # Store position information for evaluation metrics
            main_positions = {num: idx for idx, num in enumerate(main_numbers)}
            
            # Calibrate confidence scores with improved calibration
            calibrated_main_conf = MAIN_CONF_SCALE * main_confidence + MAIN_CONF_OFFSET
            calibrated_bonus_conf = BONUS_CONF_SCALE * bonus_confidence + BONUS_CONF_OFFSET
            overall_confidence = (calibrated_main_conf + calibrated_bonus_conf) / 2
            
            predictions.append({
                "main_numbers": main_numbers,
                "main_number_positions": main_positions,
                "bonus_numbers": bonus_numbers,
                "confidence": {
                    "overall": float(overall_confidence),
                    "main_numbers": float(calibrated_main_conf),
                    "bonus_numbers": float(calibrated_bonus_conf)
                },
                "method": "transformer",
                "temperature": float(actual_temp)
            })
            
            # Clean memory less often - only every CLEAN_MEMORY_FREQUENCY predictions
            if draw_idx > 0 and (draw_idx % CLEAN_MEMORY_FREQUENCY) == 0:
                Utilities.clean_memory()
        
        return predictions
    
#######################
# ENSEMBLE PREDICTOR
#######################

class EnsemblePredictor:
    """Enhanced ensemble predictor that combines multiple diverse models for better predictions."""
    
    def __init__(self, file_path, num_models=5, params=None):
        """Initialize the ensemble predictor."""
        self.file_path = file_path
        self.num_models = num_models
        self.params = params if params is not None else Utilities.get_default_params()
        self.base_predictors = []
        self.processor = LotteryDataProcessor(file_path)
        self.data_dict = None
        self.model_architectures = ['transformer', 'lstm', 'rnn']  # Multiple architecture types
        
    @ErrorHandler.handle_exception(logger, "ensemble training")
    def train(self):
        """Train multiple diverse base models with different architectures."""
        logger.info(f"Training {self.num_models} diverse base models for ensemble")
        
        # First, train a main predictor to use its data
        main_predictor = LotteryPredictionSystem(self.file_path, self.params)
        self.data_dict = main_predictor.prepare_data()
        
        # Train the main predictor
        main_predictor.train_model()
        self.base_predictors.append(main_predictor)
        
        # Train additional diverse models - only if more than one requested
        if self.num_models > 1:
            # Create and initialize shared data structures to avoid redundant processing
            input_dim = self.data_dict["X"].shape[1]
            n_samples = len(self.data_dict["X"])
            
            # Use different architectures for diversity
            for i in range(1, self.num_models):
                try:
                    logger.info(f"Training base model {i+1}/{self.num_models}")
                    
                    # Create diverse parameters with different architectures
                    model_params = self._create_diverse_params(
                        self.params, 
                        i, 
                        architecture=self.model_architectures[i % len(self.model_architectures)]
                    )
                    
                    # Create a new prediction system, reusing the data from the main predictor
                    predictor = LotteryPredictionSystem(self.file_path, model_params)
                    
                    # Share data to avoid reprocessing
                    predictor.data_prepared = True
                    predictor.X_scaled = main_predictor.X_scaled
                    predictor.main_sequences = main_predictor.main_sequences
                    predictor.bonus_sequences = main_predictor.bonus_sequences
                    predictor.processor = main_predictor.processor  # Share the processor to avoid duplicate data loading
                    
                    # Create the model with the diversity parameters
                    predictor.model = LotteryModel(model_params)
                    predictor.sequence_length = model_params.get('sequence_length', DEFAULT_SEQUENCE_LENGTH)
                    predictor.model.build_models(input_dim, predictor.sequence_length)
                    
                    # Set different random seed for each model
                    seed = i * 100 + RANDOM_SEED
                    set_seeds(seed)
                    
                    # Different sampling strategies for diversity
                    if i % 3 == 0:
                        # Standard training
                        bootstrap_indices = np.arange(n_samples)
                    elif i % 3 == 1:
                        # Bootstrap sampling - sampling with replacement
                        bootstrap_indices = np.random.choice(n_samples, n_samples, replace=True)
                    else:
                        # Bagging with reduced dataset - sampling subset without replacement
                        subset_size = int(n_samples * 0.8)  # Use 80% of data
                        bootstrap_indices = np.random.choice(n_samples, subset_size, replace=False)
                    
                    X_bootstrap = self.data_dict["X"][bootstrap_indices]
                    main_seq_bootstrap = self.data_dict["main_sequences"][bootstrap_indices]
                    bonus_seq_bootstrap = self.data_dict["bonus_sequences"][bootstrap_indices]
                    y_main_bootstrap = self.data_dict["y_main"][bootstrap_indices]
                    y_bonus_bootstrap = self.data_dict["y_bonus"][bootstrap_indices]
                    
                    # Train with different epochs to introduce diversity
                    # Create a unique validation split for each model
                    val_split = 0.1 + (i % 5) * 0.02  # Varies from 0.1 to 0.18
                    
                    # Adjust epochs based on model index
                    epochs_factor = 0.7 + (i % 3) * 0.15  # Varies from 0.7 to 1.0
                    adjusted_epochs = int(50 * epochs_factor)  # Base of 50 epochs
                    
                    predictor.model.train_models(
                        X_train=X_bootstrap,
                        main_seq_train=main_seq_bootstrap,
                        bonus_seq_train=bonus_seq_bootstrap,
                        y_main_train=y_main_bootstrap,
                        y_bonus_train=y_bonus_bootstrap,
                        validation_split=val_split,
                        epochs=adjusted_epochs
                    )
                    
                    # Add to ensemble
                    self.base_predictors.append(predictor)
                    
                    # Clean memory less frequently for efficiency
                    if i % 2 == 0:
                        Utilities.clean_memory()
                    
                except Exception as e:
                    logger.error(f"Error training model {i+1}: {e}")
                    # Continue with other models even if one fails
                    continue
        
        # Reset random seeds for consistency
        set_seeds()
        
        logger.info(f"Ensemble training complete with {len(self.base_predictors)} models")
        Utilities.clean_memory(force=True)
        return self.base_predictors
    
    def _create_diverse_params(self, base_params, model_index, architecture='transformer'):
        """Create diverse parameters for ensemble models with architecture variations."""
        diverse_params = base_params.copy()
        
        # Set model architecture type
        diverse_params['model_type'] = architecture
        
        # Apply diversity techniques based on architecture
        if architecture == 'transformer':
            # Transformer-specific parameters
            diverse_params['dropout_rate'] = base_params.get('dropout_rate', DEFAULT_DROPOUT_RATE) * (0.8 + 0.5 * random.random())
            diverse_params['num_heads'] = random.choice([2, 3, 4, 8])
            diverse_params['ff_dim'] = random.choice([64, 96, 128, 192])
            diverse_params['num_transformer_blocks'] = random.choice([1, 2, 3])
            diverse_params['use_residual'] = random.choice([True, False])
            diverse_params['use_layer_scaling'] = random.choice([True, False])
            # Fix: Ensure conv_filters is at least 1
            diverse_params['conv_filters'] = random.choice([16, 24, 32])
            
        elif architecture == 'lstm':
            # LSTM-specific parameters
            diverse_params['dropout_rate'] = base_params.get('dropout_rate', DEFAULT_DROPOUT_RATE) * (0.9 + 0.3 * random.random())
            diverse_params['lstm_units'] = random.choice([24, 32, 48, 64])
            # Fix: Ensure conv_filters is at least 1
            diverse_params['conv_filters'] = random.choice([16, 24, 32])
            
        elif architecture == 'rnn':
            # RNN-specific parameters
            diverse_params['dropout_rate'] = base_params.get('dropout_rate', DEFAULT_DROPOUT_RATE) * (0.7 + 0.6 * random.random())
            diverse_params['embed_dim'] = random.choice([32, 48, 64])
            # Fix: Ensure conv_filters is at least 1
            diverse_params['conv_filters'] = random.choice([8, 16, 24])
            
        # Common parameter diversity
        # Learning rate diversity - wider range for exploration
        diverse_params['learning_rate'] = base_params.get('learning_rate', DEFAULT_LEARNING_RATE) * (0.5 + 1.0 * random.random())
        
        # Fix: Use consistent sequence length
        diverse_params['sequence_length'] = base_params.get('sequence_length', 20)  # Use existing or default to 20
        
        # Optimizer diversity
        diverse_params['optimizer'] = random.choice(['adam', 'rmsprop', 'sgd'])
        
        # L2 regularization diversity
        diverse_params['l2_regularization'] = base_params.get('l2_regularization', 0.0001) * (0.5 + 1.5 * random.random())
        
        return diverse_params
    
    @ErrorHandler.handle_exception(logger, "ensemble prediction")    
    def predict(self, num_draws=5, temperature=DEFAULT_TEMPERATURE, diversity_sampling=True):
        """Generate ensemble predictions by combining model outputs with improved weighting."""
        logger.info(f"Generating {num_draws} ensemble predictions")
        
        if not self.base_predictors:
            logger.error("No base predictors. Call train() first.")
            # Fallback to single predictor
            predictor = LotteryPredictionSystem(self.file_path, self.params)
            predictor.train_model()
            return predictor.predict(num_draws)
        
        # Get the first predictor as reference
        main_predictor = self.base_predictors[0]
        
        # Get latest data for prediction
        latest_features = main_predictor.X_scaled.iloc[-1:].values
        
        # Track used numbers for diversity
        used_main_numbers = set()
        used_bonus_numbers = set()
        
        # Generate predictions
        predictions = []
        
        # Get and store model weights based on validation performance
        model_weights = self._calculate_model_weights()
        
        for draw_idx in range(num_draws):
            # Initialize arrays for weighted averaging of probabilities from all models
            main_probs = [np.zeros((1, MAIN_NUM_MAX)) for _ in range(MAIN_NUM_COUNT)]
            bonus_probs = [np.zeros((1, BONUS_NUM_MAX)) for _ in range(BONUS_NUM_COUNT)]
            
            # Track weights used for normalization
            total_weight = 0
            
            # Get predictions from each model and average them with weights
            for idx, predictor in enumerate(self.base_predictors):
                try:
                    if predictor.model is None:
                        continue
                        
                    # Get appropriate sequence data for this model
                    # (Models might have different sequence lengths)
                    pred_seq_length = predictor.sequence_length
                    
                    if idx == 0 or pred_seq_length == main_predictor.sequence_length:
                        # Use the same sequence data as main predictor
                        main_seq = main_predictor.main_sequences[-1:]
                        bonus_seq = main_predictor.bonus_sequences[-1:]
                    else:
                        # We need to use appropriate sequence length for this model
                        # This is handled by accessing the right indices from the processor
                        # Note: In practice, you might need to create sequences of the right length
                        # This is a simplification for demonstration
                        main_seq = main_predictor.main_sequences[-1:, -pred_seq_length:, :]
                        bonus_seq = main_predictor.bonus_sequences[-1:, -pred_seq_length:, :]
                    
                    # Predict main numbers
                    model_main_probs = predictor.model.main_model.predict([latest_features, main_seq], verbose=0)
                    
                    # Apply model weight to these probabilities
                    weight = model_weights.get(idx, 1.0)
                    total_weight += weight
                    
                    for i in range(MAIN_NUM_COUNT):
                        main_probs[i] += model_main_probs[i] * weight
                    
                    # Predict bonus numbers
                    model_bonus_probs = predictor.model.bonus_model.predict([latest_features, bonus_seq], verbose=0)
                    for i in range(BONUS_NUM_COUNT):
                        bonus_probs[i] += model_bonus_probs[i] * weight
                        
                except Exception as e:
                    logger.warning(f"Error getting ensemble predictions from model {idx}: {e}")
                    continue
            
            # Make sure we have at least one valid model prediction
            if total_weight == 0:
                logger.error("No valid ensemble models for prediction")
                return main_predictor.generate_fallback_predictions(num_draws)
            
            # Normalize the probabilities by total weight
            for i in range(MAIN_NUM_COUNT):
                main_probs[i] /= total_weight
            
            for i in range(BONUS_NUM_COUNT):
                bonus_probs[i] /= total_weight
            
            # Calculate confidence for dynamic temperature
            main_confidence = np.mean([np.max(main_probs[i][0]) for i in range(MAIN_NUM_COUNT)])
            bonus_confidence = np.mean([np.max(bonus_probs[i][0]) for i in range(BONUS_NUM_COUNT)])
            overall_confidence = (main_confidence + bonus_confidence) / 2
            
            # Apply dynamic temperature based on confidence
            dynamic_temp = Utilities.calculate_dynamic_temperature(overall_confidence, 
                                                                 min_temp=MIN_TEMPERATURE,
                                                                 max_temp=MAX_TEMPERATURE)
            
            # Use original temperature for first draw, then use dynamic temperature
            actual_temp = temperature if draw_idx == 0 else dynamic_temp
            
            # Sample main numbers with diversity handling
            main_numbers = Utilities.sample_numbers(
                probs=main_probs,
                available_nums=range(MAIN_NUM_MIN, MAIN_NUM_MAX+1),
                num_to_select=MAIN_NUM_COUNT,
                used_nums=used_main_numbers,
                diversity_sampling=diversity_sampling,
                draw_idx=draw_idx,
                temperature=actual_temp,
                confidence=overall_confidence
            )
            
            # Sample bonus numbers
            bonus_numbers = Utilities.sample_numbers(
                probs=bonus_probs,
                available_nums=range(BONUS_NUM_MIN, BONUS_NUM_MAX+1),
                num_to_select=BONUS_NUM_COUNT,
                used_nums=used_bonus_numbers,
                diversity_sampling=diversity_sampling,
                draw_idx=draw_idx,
                temperature=actual_temp,
                confidence=overall_confidence
            )
            
            # Store position information for evaluation metrics
            main_positions = {num: idx for idx, num in enumerate(main_numbers)}
            
            # Calibrate confidence scores with proper scaling
            calibrated_main_conf = MAIN_CONF_SCALE * main_confidence + MAIN_CONF_OFFSET
            calibrated_bonus_conf = BONUS_CONF_SCALE * bonus_confidence + BONUS_CONF_OFFSET
            
            # Add pattern and frequency scores
            pattern_score = Utilities.calculate_pattern_score(main_numbers, bonus_numbers)
            frequency_score = Utilities.calculate_frequency_score(
                main_numbers, bonus_numbers, main_predictor.processor.data
            )
            
            # Overall confidence with ensemble bonus - ensemble predictions are typically more reliable
            ensemble_bonus = 0.05  # Small bonus for ensemble predictions
            overall_confidence = ((calibrated_main_conf + calibrated_bonus_conf) / 2 + 
                                pattern_score + frequency_score) / 3 + ensemble_bonus
            
            predictions.append({
                "main_numbers": main_numbers,
                "main_number_positions": main_positions,
                "bonus_numbers": bonus_numbers,
                "confidence": {
                    "overall": float(min(1.0, overall_confidence)),  # Cap at 1.0
                    "main_numbers": float(calibrated_main_conf),
                    "bonus_numbers": float(calibrated_bonus_conf),
                    "pattern_score": float(pattern_score),
                    "frequency_score": float(frequency_score)
                },
                "method": "ensemble",
                "model_count": len(self.base_predictors),
                "temperature": float(actual_temp)
            })
            
            # Clean memory less frequently
            if draw_idx > 0 and (draw_idx % CLEAN_MEMORY_FREQUENCY) == 0:
                Utilities.clean_memory()
        
        return predictions
    
    def _calculate_model_weights(self):
        """Calculate weights for ensemble models based on validation performance."""
        weights = {}
        
        # If we have no models or just one, use equal weights
        if len(self.base_predictors) <= 1:
            weights = {0: 1.0}
            return weights
            
        try:
            # Get validation losses from each model if available
            val_losses = []
            
            for idx, predictor in enumerate(self.base_predictors):
                if predictor.model is None:
                    val_losses.append(None)
                    continue
                
                # Try to get validation loss
                main_history_file = "main_model_history.json"
                model_val_loss = None
                
                try:
                    if os.path.exists(main_history_file):
                        with open(main_history_file, 'r') as f:
                            history = json.load(f)
                            if "val_loss" in history and history["val_loss"]:
                                # Get the best (lowest) validation loss
                                model_val_loss = min(history["val_loss"])
                except Exception:
                    # If loading fails, assign None
                    model_val_loss = None
                
                val_losses.append(model_val_loss)
            
            # Remove None values
            val_losses = [x for x in val_losses if x is not None]
            
            # If we have no valid losses, use equal weights
            if not val_losses:
                for idx in range(len(self.base_predictors)):
                    weights[idx] = 1.0
                return weights
            
            # Calculate weights inversely proportional to validation loss
            # First, invert losses (lower loss = higher weight)
            mean_loss = np.mean(val_losses)
            inverted_losses = [mean_loss / max(val, 0.0001) for val in val_losses]
            
            # Normalize to sum to the number of models for proper averaging
            total_inverted = sum(inverted_losses)
            if total_inverted > 0:
                model_count = len(self.base_predictors)
                normalized_weights = [model_count * inv / total_inverted for inv in inverted_losses]
            else:
                # Fallback to equal weights
                normalized_weights = [1.0] * len(val_losses)
            
            # Assign weights to valid models
            valid_idx = 0
            for idx, predictor in enumerate(self.base_predictors):
                if predictor.model is not None:
                    if valid_idx < len(normalized_weights):
                        weights[idx] = normalized_weights[valid_idx]
                        valid_idx += 1
                    else:
                        weights[idx] = 1.0  # Fallback
            
        except Exception as e:
            logger.warning(f"Error calculating model weights: {e}")
            # Fallback to equal weights
            for idx in range(len(self.base_predictors)):
                weights[idx] = 1.0
                
        return weights

#######################
# HYPERPARAMETER OPTIMIZATION
#######################

class HyperparameterOptimizer:
    """Enhanced hyperparameter optimization using Bayesian methods with Optuna."""
    
    def __init__(self, file_path, n_trials=30):
        """Initialize the optimizer."""
        self.file_path = file_path
        self.n_trials = n_trials
        self.study = None
        self.best_params = None
        self.evaluator = None
        self.data_cache = None  # Cache for data reuse across trials
    
    @ErrorHandler.handle_exception(logger, "optimization trial")
    def objective(self, trial):
        """Objective function for hyperparameter optimization with expanded search space."""
        # Define expanded parameter ranges
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 0.0001, 0.003, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
            'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
            'num_heads': trial.suggest_categorical('num_heads', [2, 3, 4, 8]),
            'ff_dim': trial.suggest_categorical('ff_dim', [64, 96, 128, 192, 256]),
            'embed_dim': trial.suggest_categorical('embed_dim', [32, 48, 64, 96, 128]),
            'use_gru': trial.suggest_categorical('use_gru', [True, False]),
            # Fix 1: Ensure conv_filters is at least 1
            'conv_filters': trial.suggest_int('conv_filters', 1, 64, step=16),  # Changed minimum from 0 to 1
            'num_transformer_blocks': trial.suggest_int('num_transformer_blocks', 1, 3),
            'optimizer': trial.suggest_categorical('optimizer', ['adam', 'rmsprop', 'sgd']),
            # Fix 2: Fix sequence length to match data preparation
            'sequence_length': 20,  # Fixed to 20 instead of varying
            'l2_regularization': trial.suggest_float('l2_regularization', 1e-5, 1e-3, log=True),
            'model_type': trial.suggest_categorical('model_type', ['transformer', 'lstm', 'rnn']),
            'use_residual': trial.suggest_categorical('use_residual', [True, False]),
            'use_layer_scaling': trial.suggest_categorical('use_layer_scaling', [True, False]),
            'min_delta': trial.suggest_float('min_delta', 0.0005, 0.005, log=True)
        }
        
        # Add model-specific parameters based on model type
        if params['model_type'] == 'lstm':
            params['lstm_units'] = trial.suggest_categorical('lstm_units', [24, 32, 48, 64])
        
        # Reuse the same evaluator for all trials to avoid reloading data
        if self.evaluator is None:
            # Initialize with shared data cache if available
            self.evaluator = CrossValidationEvaluator(self.file_path, params=None, folds=3, data_cache=self.data_cache)
            # Store data cache for future use
            if self.data_cache is None:
                self.data_cache = self.evaluator.data_cache
        else:
            # Update evaluator parameters for this trial
            self.evaluator.params = params
        
        # Run evaluation with multiple metrics
        results = self.evaluator.evaluate()
        
        # Use a balanced objective considering both accuracy and partial matches
        trial_accuracy = results['avg_overall_accuracy']
        partial_match_score = results.get('avg_partial_match_score', 0)
        
        # Combined objective for optimization (70% accuracy, 30% partial matches)
        combined_score = 0.7 * trial_accuracy + 0.3 * partial_match_score
        
        # Store additional metrics in the trial
        trial.set_user_attr('accuracy', float(trial_accuracy))
        trial.set_user_attr('partial_match', float(partial_match_score))
        trial.set_user_attr('main_accuracy', float(results['avg_main_accuracy']))
        trial.set_user_attr('bonus_accuracy', float(results['avg_bonus_accuracy']))
        
        # Clean up memory
        Utilities.clean_memory()
        
        return combined_score
    
    @ErrorHandler.handle_exception(logger, "hyperparameter optimization", Utilities.get_default_params())
    def optimize(self):
        """Run Bayesian optimization to find best hyperparameters."""
        logger.info(f"Starting hyperparameter optimization with {self.n_trials} trials")
        
        # Create Optuna study with improved configuration
        self.study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=RANDOM_SEED, n_startup_trials=10),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        )
        
        # Run optimization
        self.study.optimize(self.objective, n_trials=self.n_trials)
        
        # Get best parameters
        self.best_params = self.study.best_params
        logger.info(f"Optimization complete. Best score: {self.study.best_value:.4f}")
        logger.info(f"Best parameters: {self.best_params}")
        
        # Add derived parameters that may be missing
        if 'lstm_units' not in self.best_params and self.best_params.get('model_type') == 'lstm':
            self.best_params['lstm_units'] = 32  # Default value
            
        return self.best_params
    
    @ErrorHandler.handle_exception(logger, "optimization plotting", None)
    def plot_optimization_history(self, filename="optimization_history.png"):
        """Plot optimization history with enhanced visualization."""
        if self.study is None:
            logger.error("No optimization study. Call optimize() first.")
            return None
        
        # Create figure with three subplots
        plt.figure(figsize=(18, 12))
        
        # 1. Plot optimization history
        plt.subplot(2, 2, 1)
        
        # Get trial data
        trial_numbers = [t.number for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        values = [t.value for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        best_values = []
        current_best = float('-inf')
        
        for i, value in zip(trial_numbers, values):
            if value > current_best:
                current_best = value
            best_values.append(current_best)
        
        # Plot optimization history
        plt.plot(trial_numbers, values, 'o-', markersize=4, alpha=0.7, label='Trial Value')
        plt.plot(trial_numbers, best_values, 'r-', alpha=0.7, label='Best Value')
        plt.xlabel('Trial Number')
        plt.ylabel('Objective Value')
        plt.title('Optimization History')
        plt.legend()
        plt.grid(alpha=0.3)
        
        # 2. Plot parameter importances
        plt.subplot(2, 2, 2)
        
        # Calculate parameter importances manually
        importances = {}
        for param_name in self.best_params.keys():
            try:
                param_values = []
                param_scores = []
                
                # Collect all values for this parameter
                for trial in self.study.trials:
                    if trial.state == optuna.trial.TrialState.COMPLETE and param_name in trial.params:
                        param_values.append(trial.params[param_name])
                        param_scores.append(trial.value)
                
                # Calculate variance of scores for different parameter values
                unique_values = list(set(param_values))
                if len(unique_values) > 1:
                    # Calculate mean score for each unique value
                    value_scores = {value: [] for value in unique_values}
                    for value, score in zip(param_values, param_scores):
                        value_scores[value].append(score)
                    
                    mean_scores = [np.mean(value_scores[value]) for value in unique_values]
                    importance = np.var(mean_scores)
                    importances[param_name] = importance
            except Exception:
                continue
        
        # Sort parameters by importance
        sorted_importances = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        
        # Plot top 10 parameters
        top_params = sorted_importances[:10]
        param_names = [param[0] for param in top_params]
        importance_values = [param[1] for param in top_params]
        
        # Normalize importances for better visualization
        if importance_values:
            importance_values = [value / max(importance_values) for value in importance_values]
            
        plt.barh(param_names, importance_values)
        plt.xlabel('Relative Importance')
        plt.title('Parameter Importances')
        plt.grid(alpha=0.3)
        
        # 3. Plot accuracy metrics
        plt.subplot(2, 2, 3)
        
        # Extract metrics
        accuracies = [t.user_attrs.get('accuracy', 0) for t in self.study.trials 
                      if t.state == optuna.trial.TrialState.COMPLETE]
        main_accuracies = [t.user_attrs.get('main_accuracy', 0) for t in self.study.trials 
                           if t.state == optuna.trial.TrialState.COMPLETE]
        bonus_accuracies = [t.user_attrs.get('bonus_accuracy', 0) for t in self.study.trials 
                            if t.state == optuna.trial.TrialState.COMPLETE]
        partial_matches = [t.user_attrs.get('partial_match', 0) for t in self.study.trials 
                           if t.state == optuna.trial.TrialState.COMPLETE]
        
        # Plot metrics over trials
        plt.plot(trial_numbers, accuracies, 'b-', alpha=0.7, label='Overall Accuracy')
        plt.plot(trial_numbers, main_accuracies, 'g-', alpha=0.7, label='Main Accuracy')
        plt.plot(trial_numbers, bonus_accuracies, 'm-', alpha=0.7, label='Bonus Accuracy')
        plt.plot(trial_numbers, partial_matches, 'c-', alpha=0.7, label='Partial Match Score')
        plt.xlabel('Trial Number')
        plt.ylabel('Score')
        plt.title('Performance Metrics')
        plt.legend()
        plt.grid(alpha=0.3)
        
        # 4. Plot parameter distributions for best trials
        plt.subplot(2, 2, 4)
        
        # Get top 5 trials
        top_trials = sorted(self.study.trials, key=lambda t: t.value if t.value is not None else float('-inf'), reverse=True)[:5]
        
        # Select a subset of important parameters to visualize
        key_params = ['model_type', 'learning_rate', 'dropout_rate', 'sequence_length']
        
        # Create a table-like visualization
        cell_text = []
        for trial in top_trials:
            row = [f"Trial {trial.number}"]
            for param in key_params:
                if param in trial.params:
                    value = trial.params[param]
                    # Format float values
                    if isinstance(value, float):
                        row.append(f"{value:.5f}")
                    else:
                        row.append(str(value))
                else:
                    row.append("N/A")
            row.append(f"{trial.value:.5f}")
            cell_text.append(row)
        
        # Create table
        plt.axis('off')
        table = plt.table(
            cellText=cell_text,
            colLabels=["Trial"] + key_params + ["Score"],
            loc='center',
            cellLoc='center',
            colWidths=[0.15] * (len(key_params) + 2)
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        plt.title('Top 5 Trials')
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()
        
        logger.info(f"Optimization plots saved to {filename}")
        return filename

class CrossValidationEvaluator:
    """Enhanced evaluator for time-series cross validation of lottery prediction."""
    
    def __init__(self, file_path, params=None, folds=5, data_cache=None):
        """Initialize the cross-validation evaluator."""
        self.file_path = file_path
        self.params = params if params is not None else Utilities.get_default_params()
        self.folds = folds
        self.data_dict = None
        self.system = None
        self.data_cache = data_cache  # For reusing data across trials
    
    @ErrorHandler.handle_exception(logger, "cross-validation", {
        'fold_metrics': [], 'avg_main_accuracy': 0.0, 
        'avg_bonus_accuracy': 0.0, 'avg_overall_accuracy': 0.0,
        'avg_partial_match_score': 0.0
    })
    def evaluate(self):
        """Perform time-series cross-validation with enhanced metrics."""
        logger.info(f"Performing {self.folds}-fold time-series cross-validation")
        
        # Create predictor system and prepare data once - reuse data if cached
        if self.data_cache is not None:
            self.data_dict = self.data_cache
            logger.info("Using cached data for evaluation")
        elif self.system is None or self.data_dict is None:
            # Initialize system with sequence length from parameters
            self.system = LotteryPredictionSystem(self.file_path, self.params)
            self.data_dict = self.system.prepare_data()
            # Store for later reuse
            self.data_cache = self.data_dict
        else:
            # Update parameters if they've changed
            self.system.params = self.params
        
        # Calculate total data size and validate
        total_samples = len(self.data_dict["X"])
        if total_samples < self.folds * 2:
            logger.warning(f"Not enough data ({total_samples} samples) for {self.folds} folds")
            # Adjust folds to a reasonable number
            self.folds = max(2, total_samples // 3)
            logger.info(f"Adjusted to {self.folds} folds")
            
        fold_size = total_samples // self.folds
    
        # Initialize metrics
        fold_metrics = []
        
        # Use time-series cross-validation with forward chaining
        # This ensures proper temporal ordering is maintained
        for fold in range(self.folds - 1):  # Last fold is reserved for testing
            try:
                logger.info(f"Training fold {fold+1}/{self.folds}")
                
                # Calculate fold indices for forward chaining
                # Each fold uses all previous data for training
                train_end = (fold + 1) * fold_size
                test_start = train_end
                test_end = min(test_start + fold_size, total_samples)
                
                # Validate indices
                if train_end <= 0 or test_start >= test_end or test_start >= total_samples:
                    logger.warning(f"Invalid fold indices: train end={train_end}, test={test_start}:{test_end} (total: {total_samples})")
                    continue
                
                # Split data for this fold
                train_indices = list(range(0, train_end))
                test_indices = list(range(test_start, test_end))
                
                if len(train_indices) < 10 or len(test_indices) < 2:
                    logger.warning(f"Fold {fold+1} has insufficient samples: {len(train_indices)} train, {len(test_indices)} test")
                    continue
                
                # Extract data for this fold
                X_train = self.data_dict["X"][train_indices]
                X_test = self.data_dict["X"][test_indices]
                
                main_seq_train = self.data_dict["main_sequences"][train_indices]
                main_seq_test = self.data_dict["main_sequences"][test_indices]
                
                bonus_seq_train = self.data_dict["bonus_sequences"][train_indices]
                bonus_seq_test = self.data_dict["bonus_sequences"][test_indices]
                
                y_main_train = self.data_dict["y_main"][train_indices]
                y_main_test = self.data_dict["y_main"][test_indices]
                
                y_bonus_train = self.data_dict["y_bonus"][train_indices]
                y_bonus_test = self.data_dict["y_bonus"][test_indices]
                
                # Create model for this fold
                model = LotteryModel(self.params)
                sequence_length = self.params.get('sequence_length', DEFAULT_SEQUENCE_LENGTH)
                model.build_models(X_train.shape[1], sequence_length)
                
                # Train with reduced epochs for CV
                model.train_models(
                    X_train=X_train,
                    main_seq_train=main_seq_train,
                    bonus_seq_train=bonus_seq_train,
                    y_main_train=y_main_train,
                    y_bonus_train=y_bonus_train,
                    epochs=30,  # Reduced epochs for CV
                    validation_split=0.1
                )
                
                # Evaluate on test set
                main_preds = model.main_model.predict([X_test, main_seq_test], verbose=0)
                bonus_preds = model.bonus_model.predict([X_test, bonus_seq_test], verbose=0)
            
                # Convert to class predictions (vectorized operation)
                main_class_preds = [np.argmax(main_preds[i], axis=1) for i in range(MAIN_NUM_COUNT)]
                bonus_class_preds = [np.argmax(bonus_preds[i], axis=1) for i in range(BONUS_NUM_COUNT)]
                
                # Calculate accuracy for each position (vectorized operation)
                main_position_acc = [float(np.mean(main_class_preds[i] == y_main_test[:, i])) for i in range(MAIN_NUM_COUNT)]
                bonus_position_acc = [float(np.mean(bonus_class_preds[i] == y_bonus_test[:, i])) for i in range(BONUS_NUM_COUNT)]
                
                # Calculate average accuracy
                main_avg_acc = float(np.mean(main_position_acc))
                bonus_avg_acc = float(np.mean(bonus_position_acc))
                overall_avg_acc = float((main_avg_acc + bonus_avg_acc) / 2)
                
                # Calculate partial match score - new metric
                partial_match_scores = []
                
                for i in range(len(test_indices)):
                    # Create actual draw from test data
                    actual_main = [int(y_main_test[i, j]) + 1 for j in range(MAIN_NUM_COUNT)]  # Add 1 to convert back to 1-indexed
                    actual_bonus = [int(y_bonus_test[i, j]) + 1 for j in range(BONUS_NUM_COUNT)]
                    
                    # Create prediction from model
                    pred_main = [int(main_class_preds[j][i]) + 1 for j in range(MAIN_NUM_COUNT)]
                    pred_bonus = [int(bonus_class_preds[j][i]) + 1 for j in range(BONUS_NUM_COUNT)]
                    
                    # Create dictionaries for score calculation
                    actual_draw = {
                        "main_numbers": sorted(actual_main),
                        "bonus_numbers": sorted(actual_bonus),
                        "main_number_positions": {num: idx for idx, num in enumerate(actual_main)}
                    }
                    
                    prediction = {
                        "main_numbers": sorted(pred_main),
                        "bonus_numbers": sorted(pred_bonus),
                        "main_number_positions": {num: idx for idx, num in enumerate(pred_main)}
                    }
                    
                    # Calculate score
                    score = Utilities.calculate_partial_match_score(prediction, actual_draw)
                    partial_match_scores.append(score)
                
                # Average partial match score
                avg_partial_match = float(np.mean(partial_match_scores)) if partial_match_scores else 0.0
                
                # Store metrics
                fold_metrics.append({
                    'fold': fold+1,
                    'main_accuracy': main_avg_acc,
                    'bonus_accuracy': bonus_avg_acc,
                    'overall_accuracy': overall_avg_acc,
                    'partial_match_score': avg_partial_match,
                    'main_position_acc': main_position_acc,
                    'bonus_position_acc': bonus_position_acc
                })
                
                logger.info(f"Fold {fold+1} metrics: Main acc={main_avg_acc:.4f}, Bonus acc={bonus_avg_acc:.4f}, Overall={overall_avg_acc:.4f}, Partial={avg_partial_match:.4f}")
                
                # Clean memory at a reasonable frequency
                if fold % 2 == 1:  # Clean every other fold
                    Utilities.clean_memory()
                
            except Exception as e:
                logger.error(f"Error processing fold {fold+1}: {e}")
                # Add a placeholder with zero accuracy for this fold
                fold_metrics.append({
                    'fold': fold+1,
                    'main_accuracy': 0.0,
                    'bonus_accuracy': 0.0,
                    'overall_accuracy': 0.0,
                    'partial_match_score': 0.0,
                    'error': str(e)
                })
        
        # Calculate average metrics if we have any valid folds
        if fold_metrics:
            avg_main_acc = float(np.mean([m['main_accuracy'] for m in fold_metrics]))
            avg_bonus_acc = float(np.mean([m['bonus_accuracy'] for m in fold_metrics]))
            avg_overall_acc = float(np.mean([m['overall_accuracy'] for m in fold_metrics]))
            avg_partial_match = float(np.mean([m.get('partial_match_score', 0.0) for m in fold_metrics]))
        else:
            avg_main_acc = 0.0
            avg_bonus_acc = 0.0
            avg_overall_acc = 0.0
            avg_partial_match = 0.0
        
        logger.info(f"Average metrics across {len(fold_metrics)} folds:")
        logger.info(f"Main numbers accuracy: {avg_main_acc:.4f}")
        logger.info(f"Bonus numbers accuracy: {avg_bonus_acc:.4f}")
        logger.info(f"Overall accuracy: {avg_overall_acc:.4f}")
        logger.info(f"Partial match score: {avg_partial_match:.4f}")
        
        # Clean memory after all folds
        Utilities.clean_memory(force=True)
        
        return {
            'fold_metrics': fold_metrics,
            'avg_main_accuracy': avg_main_acc,
            'avg_bonus_accuracy': avg_bonus_acc,
            'avg_overall_accuracy': avg_overall_acc,
            'avg_partial_match_score': avg_partial_match
        }
    
#######################
# UNIFIED PREDICTION SYSTEM
#######################

class LotteryPredictionSystem:
    """Enhanced unified prediction system with optimized data processing and model management."""
    
    def __init__(self, file_path, params=None):
        """Initialize the prediction system with proper sequence length handling."""
        self.file_path = file_path
        self.params = params if params is not None else Utilities.get_default_params()
        
        # Get sequence length from params if available
        self.sequence_length = self.params.get('sequence_length', DEFAULT_SEQUENCE_LENGTH)
        
        # Initialize processor with the same sequence length
        self.processor = LotteryDataProcessor(file_path)
        self.processor.set_sequence_length(self.sequence_length)
        
        self.model = None
        self.feature_scaler = StandardScaler()
        self.X_scaled = None
        self.main_sequences = None
        self.bonus_sequences = None
        self.data_prepared = False
        
    @ErrorHandler.handle_exception(logger, "data preparation")
    def prepare_data(self, force_reload=False):
        """Load, process, and prepare data for model training with improved caching."""
        # Return cached data if already prepared and not forcing reload
        if self.data_prepared and not force_reload and self.X_scaled is not None:
            logger.info("Using cached prepared data")
            
            # Get latest data for creating target variables
            expanded_data = self.processor.expanded_data
            
            # Create target variables (next draw's numbers)
            # Subtract 1 from each number to use as index (0-49 for main, 0-11 for bonus)
            y_main = np.array([
                expanded_data.iloc[self.sequence_length:][f"main_{i+1}"].values - 1 for i in range(MAIN_NUM_COUNT)
            ]).T
            
            y_bonus = np.array([
                expanded_data.iloc[self.sequence_length:][f"bonus_{i+1}"].values - 1 for i in range(BONUS_NUM_COUNT)
            ]).T
            
            # Match feature data to sequence data
            X_scaled_matched = self.X_scaled.iloc[self.sequence_length:]
            
            return {
                "X": X_scaled_matched.values,
                "main_sequences": self.main_sequences,
                "bonus_sequences": self.bonus_sequences,
                "y_main": y_main,
                "y_bonus": y_bonus,
                "data": self.processor.data  # Original data for reference
            }
        
        logger.info("Preparing data for lottery prediction")
        
        # Load and process lottery data
        data = self.processor.parse_file()
        if data.empty:
            raise ValueError("Failed to parse lottery data")
            
        # Ensure processor is using the correct sequence length
        self.processor.set_sequence_length(self.sequence_length)
            
        expanded_data = self.processor.expand_numbers()
        features = self.processor.create_features()
        
        if features.empty:
            raise ValueError("Feature engineering failed")
        
        # Scale features
        X_raw = features.select_dtypes(include=[np.number])
        X_raw = X_raw.fillna(0)
        
        # Scale features - with improved standardization and outlier handling
        # Robust scaling for better handling of outliers
        X_scaled_array = self.feature_scaler.fit_transform(X_raw)
        
        # Check for extreme values after scaling and clip them
        X_scaled_array = np.clip(X_scaled_array, -10, 10)  # Clip to reasonable range
        
        # Convert back to DataFrame
        self.X_scaled = pd.DataFrame(
            X_scaled_array,
            columns=X_raw.columns,
            index=X_raw.index
        )
        
        # Create sequence features for transformer
        self._create_sequences(expanded_data)
        
        # Create target variables (next draw's numbers)
        # Subtract 1 from each number to use as index (0-49 for main, 0-11 for bonus)
        y_main = np.array([
            expanded_data.iloc[self.sequence_length:][f"main_{i+1}"].values - 1 for i in range(MAIN_NUM_COUNT)
        ]).T
        
        y_bonus = np.array([
            expanded_data.iloc[self.sequence_length:][f"bonus_{i+1}"].values - 1 for i in range(BONUS_NUM_COUNT)
        ]).T
        
        # Match feature data to sequence data
        X_scaled_matched = self.X_scaled.iloc[self.sequence_length:]
        
        # Mark as prepared
        self.data_prepared = True
        
        logger.info(f"Data preparation complete. Features: {X_scaled_matched.shape}")
        
        return {
            "X": X_scaled_matched.values,
            "main_sequences": self.main_sequences,
            "bonus_sequences": self.bonus_sequences,
            "y_main": y_main,
            "y_bonus": y_bonus,
            "data": data  # Original data for reference
        }
    
    @ErrorHandler.handle_exception(logger, "sequence creation")
    def _create_sequences(self, expanded_data):
        """Create sequence data for transformer model using optimized vectorized operations."""
        logger.info(f"Creating sequence data for transformer model (length: {self.sequence_length})")
        
        # Process in sliding windows
        num_samples = len(expanded_data) - self.sequence_length
        
        # Pre-allocate arrays for efficiency
        main_sequences = np.zeros((num_samples, self.sequence_length, MAIN_NUM_MAX))
        bonus_sequences = np.zeros((num_samples, self.sequence_length, BONUS_NUM_MAX))
        
        # Extract ranges more efficiently using list comprehensions
        main_cols = [f"main_{i+1}" for i in range(MAIN_NUM_COUNT)]
        bonus_cols = [f"bonus_{i+1}" for i in range(BONUS_NUM_COUNT)]
        
        # Check for NaN values in columns
        if expanded_data[main_cols + bonus_cols].isna().any().any():
            logger.warning("NaN values detected in number columns. Filling with -1 for processing.")
            expanded_data[main_cols + bonus_cols] = expanded_data[main_cols + bonus_cols].fillna(-1)
        
        # Process each window more efficiently
        for i in range(num_samples):
            # Get the window of previous draws
            window = expanded_data.iloc[i:i+self.sequence_length]
            
            # For each draw in the window, set the corresponding values in the sequences
            for j, (_, row) in enumerate(window.iterrows()):
                # Convert main numbers to one-hot encoding in one operation
                main_nums = np.array([row[col] for col in main_cols if not pd.isna(row[col])]).astype(int)
                main_nums = main_nums[main_nums > 0] - 1  # Convert to 0-based indices
                if len(main_nums) > 0:
                    main_sequences[i, j, main_nums] = 1
                
                # Same for bonus numbers
                bonus_nums = np.array([row[col] for col in bonus_cols if not pd.isna(row[col])]).astype(int)
                bonus_nums = bonus_nums[bonus_nums > 0] - 1  # Convert to 0-based indices
                if len(bonus_nums) > 0:
                    bonus_sequences[i, j, bonus_nums] = 1
        
        self.main_sequences = main_sequences
        self.bonus_sequences = bonus_sequences
        
        logger.info(f"Created sequences: Main shape: {self.main_sequences.shape}, Bonus shape: {self.bonus_sequences.shape}")
    
    @ErrorHandler.handle_exception(logger, "model training")
    def train_model(self, validation_split=0.1):
        """Train the lottery prediction model with improved methodology."""
        logger.info("Training lottery prediction model")
        
        # Prepare data
        data_dict = self.prepare_data()
        
        # Create and build model
        self.model = LotteryModel(self.params)
        
        # Build models with the correct sequence length
        input_dim = data_dict["X"].shape[1]
        self.model.build_models(input_dim, self.sequence_length)
        
        # Train models with dynamic batch size based on data size
        # Smaller batch sizes for smaller datasets
        n_samples = len(data_dict["X"])
        if n_samples < 500:
            batch_size = min(16, max(4, n_samples // 20))
        elif n_samples < 1000:
            batch_size = min(32, max(8, n_samples // 30))
        else:
            batch_size = self.params.get('batch_size', DEFAULT_BATCH_SIZE)
            
        # Ensure batch size is not larger than dataset with validation split
        max_batch = int(n_samples * (1 - validation_split))
        batch_size = min(batch_size, max_batch)
        
        history = self.model.train_models(
            X_train=data_dict["X"],
            main_seq_train=data_dict["main_sequences"],
            bonus_seq_train=data_dict["bonus_sequences"],
            y_main_train=data_dict["y_main"],
            y_bonus_train=data_dict["y_bonus"],
            validation_split=validation_split,
            batch_size=batch_size
        )
        
        logger.info("Model training complete")
        Utilities.clean_memory(force=True)
        return history
    
    @ErrorHandler.handle_exception(logger, "prediction", lambda self, num_draws, **kwargs: self.generate_fallback_predictions(num_draws))
    def predict(self, num_draws=5, temperature=DEFAULT_TEMPERATURE, diversity_sampling=True):
        """Generate lottery predictions with improved confidence calibration."""
        logger.info(f"Generating {num_draws} lottery predictions")
        
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        # Prepare the latest data for prediction
        latest_features = self.X_scaled.iloc[-1:].values
        latest_main_seq = self.main_sequences[-1:] 
        latest_bonus_seq = self.bonus_sequences[-1:]
        
        # Generate predictions
        predictions = self.model.predict(
            features=latest_features,
            main_sequence=latest_main_seq,
            bonus_sequence=latest_bonus_seq,
            num_draws=num_draws,
            temperature=temperature,
            diversity_sampling=diversity_sampling
        )
        
        # Add pattern and frequency scores
        data = self.processor.data
        if data is not None and not data.empty:
            for pred in predictions:
                pattern_score = Utilities.calculate_pattern_score(
                    pred["main_numbers"], pred["bonus_numbers"]
                )
                frequency_score = Utilities.calculate_frequency_score(
                    pred["main_numbers"], pred["bonus_numbers"], data
                )
                
                pred["confidence"]["pattern_score"] = float(pattern_score)
                pred["confidence"]["frequency_score"] = float(frequency_score)
                
                # Update overall confidence with a balanced approach
                pred["confidence"]["overall"] = float(
                    (pred["confidence"]["overall"] + pattern_score + frequency_score) / 3
                )
        
        return predictions
    
    def generate_fallback_predictions(self, num_draws=5):
        """Generate fallback predictions using improved statistical approach."""
        logger.warning("Using enhanced fallback prediction method")
        
        try:
            data = self.processor.parse_file()
            
            # Calculate historical frequency with recency bias
            main_counts = np.zeros(MAIN_NUM_MAX)
            bonus_counts = np.zeros(BONUS_NUM_MAX)
            
            if data is not None and not data.empty:
                n_draws = len(data)
                
                # Apply recency bias - more recent draws have higher weight
                for i, (_, row) in enumerate(data.iterrows()):
                    # Calculate weight based on position (more recent draws have higher weight)
                    recency_weight = 0.5 + 0.5 * (i / n_draws)  # Weight ranges from 0.5 to 1.0
                    
                    for num in row["main_numbers"]:
                        if MAIN_NUM_MIN <= num <= MAIN_NUM_MAX:
                            main_counts[num-1] += recency_weight
                            
                    for num in row["bonus_numbers"]:
                        if BONUS_NUM_MIN <= num <= BONUS_NUM_MAX:
                            bonus_counts[num-1] += recency_weight
                            
                # Normalize to probabilities
                main_sum = np.sum(main_counts)
                bonus_sum = np.sum(bonus_counts)
                
                main_probs = main_counts / main_sum if main_sum > 0 else np.ones(MAIN_NUM_MAX) / MAIN_NUM_MAX
                bonus_probs = bonus_counts / bonus_sum if bonus_sum > 0 else np.ones(BONUS_NUM_MAX) / BONUS_NUM_MAX
                
                # Small smoothing to avoid zero probabilities
                main_probs = 0.95 * main_probs + 0.05 / MAIN_NUM_MAX
                bonus_probs = 0.95 * bonus_probs + 0.05 / BONUS_NUM_MAX
            else:
                # If no data, use uniform distribution
                main_probs = np.ones(MAIN_NUM_MAX) / MAIN_NUM_MAX
                bonus_probs = np.ones(BONUS_NUM_MAX) / BONUS_NUM_MAX
            
            # Generate predictions using the same sampling logic as the model
            predictions = []
            used_main_numbers = set()
            used_bonus_numbers = set()
            
            for draw_idx in range(num_draws):
                # Convert frequencies to the same format expected by sample_numbers
                main_position_probs = [[main_probs] for _ in range(MAIN_NUM_COUNT)]
                bonus_position_probs = [[bonus_probs] for _ in range(BONUS_NUM_COUNT)]
                
                # Dynamic temperature based on draw index
                draw_temp = DEFAULT_TEMPERATURE * (1.0 + 0.1 * draw_idx)  # Increase temperature for later draws
                
                # Use the unified sampling method
                main_numbers = Utilities.sample_numbers(
                    probs=main_position_probs,
                    available_nums=range(MAIN_NUM_MIN, MAIN_NUM_MAX+1),
                    num_to_select=MAIN_NUM_COUNT,
                    used_nums=used_main_numbers,
                    diversity_sampling=True,
                    draw_idx=draw_idx,
                    temperature=draw_temp
                )
                
                bonus_numbers = Utilities.sample_numbers(
                    probs=bonus_position_probs,
                    available_nums=range(BONUS_NUM_MIN, BONUS_NUM_MAX+1),
                    num_to_select=BONUS_NUM_COUNT,
                    used_nums=used_bonus_numbers,
                    diversity_sampling=True,
                    draw_idx=draw_idx,
                    temperature=draw_temp
                )
                
                # Store position information for evaluation metrics
                main_positions = {num: idx for idx, num in enumerate(main_numbers)}
                
                # Calculate pattern score for these numbers
                pattern_score = Utilities.calculate_pattern_score(main_numbers, bonus_numbers)
                frequency_score = 0.5  # Neutral frequency score
                
                if data is not None and not data.empty:
                    frequency_score = Utilities.calculate_frequency_score(main_numbers, bonus_numbers, data)
                
                # Calculate confidence based on sampling method
                base_confidence = 0.35  # Higher base for frequency-based sampling vs pure random
                
                predictions.append({
                    "main_numbers": main_numbers,
                    "main_number_positions": main_positions,
                    "bonus_numbers": bonus_numbers,
                    "confidence": {
                        "overall": float((base_confidence + pattern_score + frequency_score) / 3),
                        "main_numbers": float(base_confidence),
                        "bonus_numbers": float(base_confidence),
                        "pattern_score": float(pattern_score),
                        "frequency_score": float(frequency_score)
                    },
                    "method": "frequency_based_fallback",
                    "temperature": float(draw_temp)
                })
                
                # Clean memory less frequently
                if draw_idx > 0 and (draw_idx % CLEAN_MEMORY_FREQUENCY) == 0:
                    Utilities.clean_memory()
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error in fallback prediction: {str(e)}")
            # Fallback to truly random predictions
            return self._generate_random_predictions(num_draws)
    
    def _generate_random_predictions(self, num_draws):
        """Generate completely random predictions as a last resort with improved diversity."""
        predictions = []
        
        used_main_numbers = set()
        used_bonus_numbers = set()
        
        for draw_idx in range(num_draws):
            try:
                # Use more sophisticated sampling for diversity
                if draw_idx == 0 or random.random() < 0.7:
                    # Standard random sampling
                    main_numbers = sorted(random.sample(range(MAIN_NUM_MIN, MAIN_NUM_MAX+1), MAIN_NUM_COUNT))
                    bonus_numbers = sorted(random.sample(range(BONUS_NUM_MIN, BONUS_NUM_MAX+1), BONUS_NUM_COUNT))
                else:
                    # Diversity-aware sampling
                    available_main = list(set(range(MAIN_NUM_MIN, MAIN_NUM_MAX+1)) - used_main_numbers)
                    available_bonus = list(set(range(BONUS_NUM_MIN, BONUS_NUM_MAX+1)) - used_bonus_numbers)
                    
                    # Ensure enough numbers are available
                    if len(available_main) < MAIN_NUM_COUNT:
                        available_main = list(range(MAIN_NUM_MIN, MAIN_NUM_MAX+1))
                    if len(available_bonus) < BONUS_NUM_COUNT:
                        available_bonus = list(range(BONUS_NUM_MIN, BONUS_NUM_MAX+1))
                    
                    main_numbers = sorted(random.sample(available_main, MAIN_NUM_COUNT))
                    bonus_numbers = sorted(random.sample(available_bonus, BONUS_NUM_COUNT))
                
                # Update used numbers sets
                used_main_numbers.update(main_numbers)
                used_bonus_numbers.update(bonus_numbers)
                
                # Store position information
                main_positions = {num: idx for idx, num in enumerate(main_numbers)}
                
                # Calculate pattern score for random numbers
                pattern_score = Utilities.calculate_pattern_score(main_numbers, bonus_numbers)
                
                predictions.append({
                    "main_numbers": main_numbers,
                    "main_number_positions": main_positions,
                    "bonus_numbers": bonus_numbers,
                    "confidence": {
                        "overall": 0.15,
                        "main_numbers": 0.15,
                        "bonus_numbers": 0.15,
                        "pattern_score": float(pattern_score),
                        "frequency_score": 0.15
                    },
                    "method": "pure_random_fallback",
                    "temperature": 1.0
                })
                
            except Exception:
                # Ultimate fallback if even random.sample fails
                try:
                    main_numbers = sorted([random.randint(MAIN_NUM_MIN, MAIN_NUM_MAX) for _ in range(MAIN_NUM_COUNT)])
                    bonus_numbers = sorted([random.randint(BONUS_NUM_MIN, BONUS_NUM_MAX) for _ in range(BONUS_NUM_COUNT)])
                    
                    predictions.append({
                        "main_numbers": main_numbers,
                        "bonus_numbers": bonus_numbers,
                        "confidence": {
                            "overall": 0.1,
                            "main_numbers": 0.1,
                            "bonus_numbers": 0.1,
                            "pattern_score": 0.1,
                            "frequency_score": 0.1
                        },
                        "method": "emergency_random_fallback",
                        "temperature": 1.5
                    })
                except Exception as e:
                    logger.error(f"Emergency fallback failed: {e}")
                    # Skip this prediction in the worst case
                    continue
        
        return predictions

#######################
# VISUALIZATION FUNCTIONS
#######################

@ErrorHandler.handle_exception(logger, "visualization", None)
def generate_visualizations(predictions, file_path, include_historical=True):
    """Generate enhanced visualizations for lottery predictions with better analytics."""
    logger.info("Generating visualizations for lottery predictions")
    
    # Create processor to get historical data
    processor = LotteryDataProcessor(file_path)
    data = processor.parse_file()
    
    if data.empty and include_historical:
        logger.error("No data available for visualization")
        include_historical = False
    
    # Set up the figure for main visualization
    plt.figure(figsize=(16, 12))
    
    # Plot 1: Frequency of main numbers (historical vs predicted)
    plt.subplot(2, 2, 1)
    
    if include_historical:
        # Historical frequency - vectorized calculation
        main_freq = np.zeros(MAIN_NUM_MAX)
        all_main_numbers = [num for row in data["main_numbers"] for num in row]
        for num in all_main_numbers:
            if MAIN_NUM_MIN <= num <= MAIN_NUM_MAX:
                main_freq[num-1] += 1
        main_freq = main_freq / len(data) if len(data) > 0 else np.zeros(MAIN_NUM_MAX)
    else:
        # Create a uniform distribution as baseline if no historical data
        main_freq = np.ones(MAIN_NUM_MAX) / MAIN_NUM_MAX
    
    # Predicted frequency - vectorized calculation
    pred_main_freq = np.zeros(MAIN_NUM_MAX)
    all_pred_main_numbers = [num for pred in predictions for num in pred["main_numbers"]]
    for num in all_pred_main_numbers:
        if MAIN_NUM_MIN <= num <= MAIN_NUM_MAX:
            pred_main_freq[num-1] += 1
    pred_main_freq = pred_main_freq / len(predictions) if len(predictions) > 0 else np.zeros(MAIN_NUM_MAX)
    
    # Create the plot
    x = np.arange(1, MAIN_NUM_MAX+1)
    width = 0.35
    
    plt.bar(x - width/2, main_freq, width, label='Historical' if include_historical else 'Baseline', 
           alpha=0.7, color='royalblue')
    plt.bar(x + width/2, pred_main_freq, width, label='Predicted', alpha=0.7, color='seagreen')
    
    plt.title("Main Numbers Frequency Comparison", fontsize=14)
    plt.xlabel("Number", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.xticks(range(0, MAIN_NUM_MAX+1, 5))
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # Plot 2: Frequency of bonus numbers (historical vs predicted)
    plt.subplot(2, 2, 2)
    
    if include_historical:
        # Historical frequency - vectorized calculation
        bonus_freq = np.zeros(BONUS_NUM_MAX)
        all_bonus_numbers = [num for row in data["bonus_numbers"] for num in row]
        for num in all_bonus_numbers:
            if BONUS_NUM_MIN <= num <= BONUS_NUM_MAX:
                bonus_freq[num-1] += 1
        bonus_freq = bonus_freq / len(data) if len(data) > 0 else np.zeros(BONUS_NUM_MAX)
    else:
        # Create a uniform distribution as baseline if no historical data
        bonus_freq = np.ones(BONUS_NUM_MAX) / BONUS_NUM_MAX
    
    # Predicted frequency - vectorized calculation
    pred_bonus_freq = np.zeros(BONUS_NUM_MAX)
    all_pred_bonus_numbers = [num for pred in predictions for num in pred["bonus_numbers"]]
    for num in all_pred_bonus_numbers:
        if BONUS_NUM_MIN <= num <= BONUS_NUM_MAX:
            pred_bonus_freq[num-1] += 1
    pred_bonus_freq = pred_bonus_freq / len(predictions) if len(predictions) > 0 else np.zeros(BONUS_NUM_MAX)
    
    # Create the plot
    x = np.arange(1, BONUS_NUM_MAX+1)
    plt.bar(x - width/2, bonus_freq, width, label='Historical' if include_historical else 'Baseline', 
           alpha=0.7, color='royalblue')
    plt.bar(x + width/2, pred_bonus_freq, width, label='Predicted', alpha=0.7, color='seagreen')
    
    plt.title("Bonus Numbers Frequency Comparison", fontsize=14)
    plt.xlabel("Number", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.xticks(range(1, BONUS_NUM_MAX+1))
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # Plot 3: Prediction confidence distribution
    plt.subplot(2, 2, 3)
    confidences = np.array([pred["confidence"]["overall"] * 100 for pred in predictions])
    
    # Enhanced histogram with better binning
    num_bins = min(10, len(confidences) // 2) if len(confidences) > 0 else 5
    hist, bins = np.histogram(confidences, bins=num_bins)
    center = (bins[:-1] + bins[1:]) / 2
    width = 0.8 * (bins[1] - bins[0])
    plt.bar(center, hist, align='center', width=width, color='purple', alpha=0.7)
    
    plt.title("Prediction Confidence Distribution", fontsize=14)
    plt.xlabel("Confidence (%)", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    
    # Add vertical line for average confidence
    avg_conf = np.mean(confidences) if len(confidences) > 0 else 0
    plt.axvline(x=avg_conf, color='r', linestyle='--', alpha=0.7, label=f"Average ({avg_conf:.2f}%)")
    plt.legend()
    
    # Plot 4: Enhanced Pattern Analysis (Number Range + Metrics)
    plt.subplot(2, 2, 4)
    
    if include_historical:
        # Create pattern analysis for predicted draws
        ranges = ["1-10", "11-20", "21-30", "31-40", "41-50"]
        hist_range_counts = np.zeros(5)
        pred_range_counts = np.zeros(5)
        
        # Historical distribution - vectorized
        for num in all_main_numbers:
            range_idx = (num - 1) // 10
            if 0 <= range_idx < 5:
                hist_range_counts[range_idx] += 1
    else:
        # Create baseline distribution
        ranges = ["1-10", "11-20", "21-30", "31-40", "41-50"]
        hist_range_counts = np.ones(5) * 100  # Uniform distribution
        pred_range_counts = np.zeros(5)
    
    # Predicted distribution - vectorized
    for num in all_pred_main_numbers:
        range_idx = (num - 1) // 10
        if 0 <= range_idx < 5:
            pred_range_counts[range_idx] += 1
    
    # Normalize
    hist_sum = np.sum(hist_range_counts)
    pred_sum = np.sum(pred_range_counts)
    hist_range_counts = hist_range_counts / hist_sum if hist_sum > 0 else hist_range_counts
    pred_range_counts = pred_range_counts / pred_sum if pred_sum > 0 else pred_range_counts
    
    # Plot as bar chart
    x = np.arange(len(ranges))
    plt.bar(x - width/2, hist_range_counts, width, label='Historical' if include_historical else 'Baseline', 
           alpha=0.7, color='royalblue')
    plt.bar(x + width/2, pred_range_counts, width, label='Predicted', alpha=0.7, color='seagreen')
    
    plt.title("Number Range Distribution", fontsize=14)
    plt.xlabel("Number Range", fontsize=12)
    plt.ylabel("Proportion", fontsize=12)
    plt.xticks(x, ranges)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # Save figure
    plt.tight_layout()
    plt.savefig("lottery_predictions.png", dpi=300)
    plt.close()
    
    # Create enhanced pattern analysis visualization
    plt.figure(figsize=(16, 12))
    
    # Plot 1: Odd/Even Distribution
    plt.subplot(2, 2, 1)
    
    # Calculate odd/even stats
    if include_historical:
        hist_odd = sum(1 for num in all_main_numbers if num % 2 == 1)
        hist_even = sum(1 for num in all_main_numbers if num % 2 == 0)
    else:
        # Baseline is 50/50
        hist_odd = 100
        hist_even = 100
    
    pred_odd = sum(1 for num in all_pred_main_numbers if num % 2 == 1)
    pred_even = sum(1 for num in all_pred_main_numbers if num % 2 == 0)
    
    # Normalize
    hist_total = hist_odd + hist_even
    pred_total = pred_odd + pred_even
    
    if hist_total > 0:
        hist_odd_pct = hist_odd / hist_total * 100
        hist_even_pct = hist_even / hist_total * 100
    else:
        hist_odd_pct = 50
        hist_even_pct = 50
        
    if pred_total > 0:
        pred_odd_pct = pred_odd / pred_total * 100
        pred_even_pct = pred_even / pred_total * 100
    else:
        pred_odd_pct = 0
        pred_even_pct = 0
    
    # Create the plot
    categories = ['Odd', 'Even']
    hist_values = [hist_odd_pct, hist_even_pct]
    pred_values = [pred_odd_pct, pred_even_pct]
    
    x = np.arange(len(categories))
    plt.bar(x - width/2, hist_values, width, label='Historical' if include_historical else 'Baseline', 
           alpha=0.7, color='royalblue')
    plt.bar(x + width/2, pred_values, width, label='Predicted', alpha=0.7, color='seagreen')
    
    plt.title("Odd/Even Distribution", fontsize=14)
    plt.xlabel("Number Type", fontsize=12)
    plt.ylabel("Percentage (%)", fontsize=12)
    plt.xticks(x, categories)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # Plot 2: Sum Range Distribution
    plt.subplot(2, 2, 2)
    
    # Calculate sum ranges
    sum_ranges = ["70-100", "101-130", "131-160", "161-190", "191-220", "221+"]
    sum_bins = [70, 101, 131, 161, 191, 221, float('inf')]
    
    if include_historical:
        # Historical sums
        hist_sums = [sum(row) for row in data["main_numbers"]]
        hist_sum_counts = np.zeros(len(sum_ranges))
        
        for s in hist_sums:
            for i, (lower, upper) in enumerate(zip(sum_bins[:-1], sum_bins[1:])):
                if lower <= s < upper:
                    hist_sum_counts[i] += 1
                    break
    else:
        # Baseline - uniform distribution
        hist_sum_counts = np.ones(len(sum_ranges))
    
    # Predicted sums
    pred_sums = [sum(pred["main_numbers"]) for pred in predictions]
    pred_sum_counts = np.zeros(len(sum_ranges))
    
    for s in pred_sums:
        for i, (lower, upper) in enumerate(zip(sum_bins[:-1], sum_bins[1:])):
            if lower <= s < upper:
                pred_sum_counts[i] += 1
                break
    
    # Normalize
    hist_sum_total = np.sum(hist_sum_counts)
    pred_sum_total = np.sum(pred_sum_counts)
    
    if hist_sum_total > 0:
        hist_sum_counts = hist_sum_counts / hist_sum_total
    
    if pred_sum_total > 0:
        pred_sum_counts = pred_sum_counts / pred_sum_total
    
    # Create the plot
    x = np.arange(len(sum_ranges))
    plt.bar(x - width/2, hist_sum_counts, width, label='Historical' if include_historical else 'Baseline', 
           alpha=0.7, color='royalblue')
    plt.bar(x + width/2, pred_sum_counts, width, label='Predicted', alpha=0.7, color='seagreen')
    
    plt.title("Sum Range Distribution", fontsize=14)
    plt.xlabel("Sum Range", fontsize=12)
    plt.ylabel("Proportion", fontsize=12)
    plt.xticks(x, sum_ranges)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # Plot 3: Consecutive Numbers Analysis
    plt.subplot(2, 2, 3)
    
    # Count consecutive pairs in numbers
    if include_historical:
        hist_consecutive_counts = []
        for row in data["main_numbers"]:
            sorted_nums = sorted(row)
            count = sum(1 for i in range(len(sorted_nums)-1) if sorted_nums[i+1] - sorted_nums[i] == 1)
            hist_consecutive_counts.append(count)
    else:
        # Baseline - random distribution around 0-1 pairs
        hist_consecutive_counts = [0, 0, 1, 1, 1, 0, 0] * 20
    
    pred_consecutive_counts = []
    for pred in predictions:
        sorted_nums = sorted(pred["main_numbers"])
        count = sum(1 for i in range(len(sorted_nums)-1) if sorted_nums[i+1] - sorted_nums[i] == 1)
        pred_consecutive_counts.append(count)
    
    # Create histograms
    max_count = max(max(hist_consecutive_counts) if hist_consecutive_counts else 0, 
                   max(pred_consecutive_counts) if pred_consecutive_counts else 0)
    bins = np.arange(-0.5, max_count + 1.5, 1)
    
    plt.hist(hist_consecutive_counts, bins=bins, alpha=0.5, label='Historical' if include_historical else 'Baseline', 
            color='royalblue')
    plt.hist(pred_consecutive_counts, bins=bins, alpha=0.5, label='Predicted', color='seagreen')
    
    plt.title("Consecutive Number Pairs Distribution", fontsize=14)
    plt.xlabel("Number of Consecutive Pairs", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.xticks(range(0, max_count + 1))
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # Plot 4: Average Gap Analysis
    plt.subplot(2, 2, 4)
    
    # Calculate average gap between numbers
    if include_historical:
        hist_gaps = []
        for row in data["main_numbers"]:
            sorted_nums = sorted(row)
            gaps = [sorted_nums[i+1] - sorted_nums[i] for i in range(len(sorted_nums)-1)]
            hist_gaps.append(np.mean(gaps) if gaps else 0)
    else:
        # Baseline - theoretical random distribution
        hist_gaps = np.random.normal(10, 2, 100)  # Centered around 10 with some variation
    
    pred_gaps = []
    for pred in predictions:
        sorted_nums = sorted(pred["main_numbers"])
        gaps = [sorted_nums[i+1] - sorted_nums[i] for i in range(len(sorted_nums)-1)]
        pred_gaps.append(np.mean(gaps) if gaps else 0)
    
    # Create the plots - kernel density estimation for smoother visualization
    hist_gaps = np.array(hist_gaps)
    pred_gaps = np.array(pred_gaps)
    
    from scipy.stats import gaussian_kde
    
    # Only perform KDE if we have sufficient data points
    if len(hist_gaps) > 5 and np.std(hist_gaps) > 0:
        hist_kde = gaussian_kde(hist_gaps)
        x_hist = np.linspace(min(hist_gaps), max(hist_gaps), 100)
        plt.plot(x_hist, hist_kde(x_hist), label='Historical' if include_historical else 'Baseline', color='royalblue')
    else:
        # Fallback to histogram
        plt.hist(hist_gaps, bins=10, alpha=0.5, label='Historical' if include_historical else 'Baseline', 
                color='royalblue', density=True)
    
    if len(pred_gaps) > 5 and np.std(pred_gaps) > 0:
        pred_kde = gaussian_kde(pred_gaps)
        x_pred = np.linspace(min(pred_gaps), max(pred_gaps), 100)
        plt.plot(x_pred, pred_kde(x_pred), label='Predicted', color='seagreen')
    else:
        # Fallback to histogram
        plt.hist(pred_gaps, bins=10, alpha=0.5, label='Predicted', color='seagreen', density=True)
    
    plt.title("Average Gap Between Numbers", fontsize=14)
    plt.xlabel("Average Gap", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Save the pattern analysis
    plt.tight_layout()
    plt.savefig("pattern_analysis.png", dpi=300)
    plt.close()
    
    # Create correlation/heatmap visualization
    plt.figure(figsize=(14, 10))
    
    # Create heatmap of number co-occurrences in predictions
    co_occurrence = np.zeros((MAIN_NUM_MAX, MAIN_NUM_MAX))
    
    for pred in predictions:
        for i in pred["main_numbers"]:
            for j in pred["main_numbers"]:
                if 1 <= i <= MAIN_NUM_MAX and 1 <= j <= MAIN_NUM_MAX:
                    co_occurrence[i-1, j-1] += 1
    
    # Normalize by diagonal values
    for i in range(MAIN_NUM_MAX):
        if co_occurrence[i, i] > 0:
            co_occurrence[:, i] = co_occurrence[:, i] / co_occurrence[i, i]
    
    # Plot reduced size heatmap to make it more viewable
    # Focus on numbers 1-25
    plt.subplot(1, 2, 1)
    plt.imshow(co_occurrence[:25, :25], cmap='viridis', origin='lower')
    plt.colorbar(label='Normalized Co-occurrence')
    plt.title("Number Co-occurrence Heatmap (1-25)", fontsize=14)
    plt.xlabel("Number", fontsize=12)
    plt.ylabel("Number", fontsize=12)
    plt.xticks(range(0, 25, 5), range(1, 26, 5))
    plt.yticks(range(0, 25, 5), range(1, 26, 5))
    
    # Focus on numbers 26-50
    plt.subplot(1, 2, 2)
    plt.imshow(co_occurrence[25:, 25:], cmap='viridis', origin='lower')
    plt.colorbar(label='Normalized Co-occurrence')
    plt.title("Number Co-occurrence Heatmap (26-50)", fontsize=14)
    plt.xlabel("Number", fontsize=12)
    plt.ylabel("Number", fontsize=12)
    plt.xticks(range(0, 25, 5), range(26, 51, 5))
    plt.yticks(range(0, 25, 5), range(26, 51, 5))
    
    # Save the heatmap
    plt.tight_layout()
    plt.savefig("number_correlation.png", dpi=300)
    plt.close()
    
    logger.info("Enhanced visualizations saved to lottery_predictions.png, pattern_analysis.png, and number_correlation.png")
    return ["lottery_predictions.png", "pattern_analysis.png", "number_correlation.png"]


def configure_tensorflow():
    """Configure TensorFlow to reduce warnings and optimize performance."""
    # Reduce threading warnings
    try:
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)
    except:
        pass
    
    # Set logging level to suppress warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    # Disable eager execution for better performance with graph mode
    try:
        tf.compat.v1.disable_eager_execution()
    except:
        pass
    
    # Set GPU memory growth
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
    except:
        pass
    
    logger.info("TensorFlow configured for optimal performance")


#######################
# MAIN FUNCTION
#######################

def main():
    """Main function to run the improved lottery prediction system."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Optimized Transformer-Based EuroMillions Prediction System")
    parser.add_argument("--file", default="lottery_numbers.txt", help="Path to lottery data file")
    parser.add_argument("--optimize", action="store_true", help="Run hyperparameter optimization before prediction")
    parser.add_argument("--trials", type=int, default=30, help="Number of optimization trials")
    parser.add_argument("--predictions", type=int, default=20, help="Number of predictions to generate")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate model performance with cross-validation")
    parser.add_argument("--ensemble", action="store_true", help="Use ensemble for improved predictions")
    parser.add_argument("--num_models", type=int, default=5, help="Number of models in ensemble")
    parser.add_argument("--params", default="transformer_params.json", help="Path to parameters file")
    parser.add_argument("--sequence_length", type=int, default=20, help="Sequence length for historical data")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="Temperature for sampling")
    parser.add_argument("--output", default="predictions.json", help="Output file for predictions")

    args = parser.parse_args()
    
    try:
        configure_tensorflow()
        # Print header
        print("\n" + "="*80)
        print("ENHANCED TRANSFORMER-BASED EUROMILLIONS LOTTERY PREDICTION SYSTEM".center(80))
        print("="*80 + "\n")
        print("This system uses transformer models with advanced feature engineering")
        print("and ensemble techniques to enhance prediction accuracy.")
        print("\nDISCLAIMER: Lottery outcomes are primarily random events and no")
        print("prediction system can guarantee winning numbers.")
        print("="*80 + "\n")
        
        # Check if data file exists
        if not os.path.exists(args.file):
            print(f"Error: Lottery data file '{args.file}' not found.")
            sys.exit(1)
        
        # Load parameters with unified error handling
        params = Utilities.load_params(args.params, Utilities.get_default_params())
        
        # Override sequence length if provided
        if args.sequence_length is not None:
            params['sequence_length'] = args.sequence_length
            print(f"Using custom sequence length: {args.sequence_length}")
        
        # Handle optimization if requested
        if args.optimize:
            print(f"Running hyperparameter optimization with {args.trials} trials...")
            optimizer = HyperparameterOptimizer(args.file, args.trials)
            params = optimizer.optimize()
            Utilities.save_params(params, args.params)
            print(f"Optimization complete. Best parameters saved to {args.params}")
            
            # Generate optimization plots
            plot_file = optimizer.plot_optimization_history()
            if plot_file:
                print(f"Optimization history plot saved to {plot_file}")
        
        # Evaluate model if requested
        if args.evaluate:
            print("\nEvaluating model performance with cross-validation...")
            evaluator = CrossValidationEvaluator(args.file, params)
            performance = evaluator.evaluate()
            
            if 'error' in performance:
                print(f"\nError during evaluation: {performance['error']}")
            
            print("\nCross-Validation Performance:")
            print(f"Overall Accuracy: {performance['avg_overall_accuracy']*100:.2f}%")
            print(f"Main Numbers Accuracy: {performance['avg_main_accuracy']*100:.2f}%")
            print(f"Bonus Numbers Accuracy: {performance['avg_bonus_accuracy']*100:.2f}%")
            print(f"Partial Match Score: {performance['avg_partial_match_score']*100:.2f}%")
            
            # Clean memory
            Utilities.clean_memory(force=True)
        
        # Generate predictions
        predictions = None
        if args.ensemble:
            print(f"\nTraining ensemble with {args.num_models} diverse models...")
            ensemble = EnsemblePredictor(args.file, args.num_models, params)
            ensemble.train()
            
            print(f"\nGenerating {args.predictions} ensemble predictions...")
            predictions = ensemble.predict(args.predictions, temperature=args.temperature)
        else:
            print("\nTraining transformer model...")
            system = LotteryPredictionSystem(args.file, params)
            system.train_model()
            
            print(f"\nGenerating {args.predictions} predictions...")
            predictions = system.predict(args.predictions, temperature=args.temperature)
        
        # Check if we got valid predictions
        if not predictions or len(predictions) == 0:
            print("\nError: Failed to generate predictions.")
            sys.exit(1)
        
        # Save predictions to JSON file
        try:
            with open(args.output, 'w') as f:
                # Convert numpy values to Python types for JSON serialization
                cleaned_predictions = []
                for pred in predictions:
                    cleaned_pred = {}
                    for key, value in pred.items():
                        if isinstance(value, dict):
                            cleaned_pred[key] = {k: float(v) if isinstance(v, (np.float32, np.float64)) else v 
                                              for k, v in value.items()}
                        elif isinstance(value, (np.float32, np.float64)):
                            cleaned_pred[key] = float(value)
                        else:
                            cleaned_pred[key] = value
                    cleaned_predictions.append(cleaned_pred)
                
                json.dump(cleaned_predictions, f, indent=4)
            print(f"Predictions saved to {args.output}")
        except Exception as e:
            print(f"Error saving predictions to file: {e}")
        
        # Display predictions
        print("\nEuroMillions Predictions:")
        print("=====================\n")
        
        for i, pred in enumerate(predictions):
            main_numbers = pred["main_numbers"]
            bonus_numbers = pred["bonus_numbers"]
            confidence = pred["confidence"]["overall"] * 100
            
            print(f"Draw {i+1}:")
            print(f"Main Numbers: {main_numbers}")
            print(f"Lucky Stars: {bonus_numbers}")
            print(f"Confidence: {confidence:.2f}%")
            if "method" in pred:
                print(f"Method: {pred['method']}")
            if "temperature" in pred:
                print(f"Temperature: {pred['temperature']:.2f}")
            print("-" * 30)
        
        # Generate visualizations
        print("\nGenerating enhanced visualizations...")
        viz_files = generate_visualizations(predictions, args.file)
        if viz_files:
            print(f"Visualizations saved to: {', '.join(viz_files)}")
        
        # Print summary with optimized calculation
        try:
            # Efficient counting of number frequencies
            main_counts = {}
            bonus_counts = {}
            
            # Use collections.Counter for more efficient counting
            from collections import Counter
            all_main_numbers = [num for pred in predictions for num in pred["main_numbers"]]
            all_bonus_numbers = [num for pred in predictions for num in pred["bonus_numbers"]]
            
            main_counts = Counter(all_main_numbers)
            bonus_counts = Counter(all_bonus_numbers)
            
            print("\nPrediction Summary:")
            print("=================")
            
            print("\nMost frequent main numbers in predictions:")
            for num, count in sorted(main_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"Number {num}: appeared {count} times")
            
            print("\nMost frequent bonus numbers in predictions:")
            for num, count in sorted(bonus_counts.items(), key=lambda x: x[1], reverse=True)[:3]:
                print(f"Number {num}: appeared {count} times")
            
            # Calculate average confidence and other metrics
            avg_confidence = np.mean([pred["confidence"]["overall"] * 100 for pred in predictions])
            avg_pattern_score = np.mean([pred["confidence"].get("pattern_score", 0) * 100 for pred in predictions])
            avg_frequency_score = np.mean([pred["confidence"].get("frequency_score", 0) * 100 for pred in predictions])
            
            print(f"\nAverage prediction confidence: {avg_confidence:.2f}%")
            print(f"Average pattern score: {avg_pattern_score:.2f}%")
            print(f"Average frequency score: {avg_frequency_score:.2f}%")
            
            print("\nReminder: These predictions are based on statistical analysis and")
            print("are not guaranteed to win. Please gamble responsibly.")
            
        except Exception as e:
            logger.error(f"Error in summary generation: {str(e)}")
            print(f"\nError generating summary: {str(e)}")
    
    except KeyboardInterrupt:
        print("\nPrediction system stopped by user.")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        print(f"\nError: {str(e)}")
        print("Check enhanced_transformer.log for details.")

if __name__ == "__main__":
    main()