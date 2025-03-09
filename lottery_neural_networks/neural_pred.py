#!/usr/bin/env python
"""
Usage:

On remote for more computer:
#optimize hyper parameters
python neural_pred.py --file lottery_numbers.txt --optimize --trials 100 --evaluate --output remote_predictions.json
#train the full model
python neural_pred.py --file lottery_numbers.txt --params hybrid_best_params.json --force_train --predictions 0

locally:
#Generate predictions locally
python neural_pred.py --file lottery_numbers.txt --params hybrid_best_params.json --load_existing --evaluate --predictions 30 --match_recent
"""
import sys
import os
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
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, LSTM, Dropout, BatchNormalization, Bidirectional,
    GRU, Conv1D, GlobalAveragePooling1D, MultiHeadAttention, LayerNormalization,
    SimpleRNN, Activation, Concatenate
)
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)
from tensorflow.keras.regularizers import l2
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

# Model parameters - hybridized from both approaches
DEFAULT_SEQUENCE_LENGTH = 20
DEFAULT_EMBED_DIM = 56
DEFAULT_NUM_HEADS = 4
DEFAULT_FF_DIM = 112
DEFAULT_DROPOUT_RATE = 0.35
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 80
DEFAULT_LEARNING_RATE = 0.0008
DEFAULT_TRANSFORMER_BLOCKS = 2
DEFAULT_CONV_FILTERS = 28
DEFAULT_PATIENCE = 12

# Confidence calibration factors
MAIN_CONF_SCALE = 0.55
MAIN_CONF_OFFSET = 0.18
BONUS_CONF_SCALE = 0.65
BONUS_CONF_OFFSET = 0.12

# Feature selection
MAX_FEATURES = 450

# Temperature scaling for sampling
DEFAULT_TEMPERATURE = 0.75
MIN_TEMPERATURE = 0.6
MAX_TEMPERATURE = 1.2
DIVERSITY_FACTOR = 0.85

# Memory management constants
CLEAN_MEMORY_FREQUENCY = 8

# Early stopping threshold for loss improvement
MIN_DELTA = 0.001

# Directory constants
MODEL_DIR = "models"
CACHE_DIR = "cache"
LOG_DIR = "logs"
VISUALIZATION_DIR = "visualizations"

# Create necessary directories
for directory in [MODEL_DIR, CACHE_DIR, LOG_DIR, VISUALIZATION_DIR]:
    os.makedirs(directory, exist_ok=True)

#######################
# SETUP AND UTILITIES
#######################

# Configure GPU usage
def configure_gpu():
    """Configure TensorFlow to use one or more GPUs if available."""
    try:
        # Check for available GPUs
        physical_devices = tf.config.list_physical_devices('GPU')
        if len(physical_devices) > 0:
            # Enable memory growth for all GPUs to prevent memory allocation errors
            for device in physical_devices:
                try:
                    tf.config.experimental.set_memory_growth(device, True)
                    print(f"Memory growth enabled for {device}")
                except Exception as e:
                    print(f"Could not set memory growth for {device}: {e}")
            
            # Set TensorFlow to only log errors
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
            
            print(f"GPU support enabled - {len(physical_devices)} GPU(s) available")
            return len(physical_devices)
        else:
            print("No GPU detected. Using CPU for training.")
            return 0
    except Exception as e:
        print(f"Error configuring GPU: {str(e)}")
        print("Falling back to CPU training.")
        return 0
    
def create_distribution_strategy():
    """Create an appropriate distribution strategy based on available hardware."""
    num_gpus = configure_gpu()
    
    if num_gpus > 1:
        print(f"Creating MirroredStrategy for {num_gpus} GPUs")
        return tf.distribute.MirroredStrategy()
    elif num_gpus == 1:
        print("Creating OneDeviceStrategy for single GPU")
        return tf.distribute.OneDeviceStrategy(device="/gpu:0")
    else:
        print("Creating default strategy for CPU")
        return tf.distribute.get_strategy()

# Set up consistent seeds
def set_seeds(seed=RANDOM_SEED):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

def optimize_memory_usage():
    """Apply memory optimization settings for multi-GPU training."""
    # Limit TensorFlow memory growth
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(f"Memory growth setting failed: {e}")
        
        # Optional: Set memory limit per GPU (uncomment if needed)
        # for gpu in gpus:
        #    tf.config.experimental.set_virtual_device_configuration(
        #        gpu,
        #        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]
        #    )
    
    # TensorFlow mixed precision for faster training
    try:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        print("Mixed precision policy set to mixed_float16")
    except:
        print("Could not set mixed precision policy")


# Set up logging with a configurable level
def setup_logging(level=logging.INFO):
    """Set up logging with file and console handlers."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("hybrid_neural_lottery.log", mode='w'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    # Hide TensorFlow warnings
    tf.get_logger().setLevel('ERROR')
    return logger

def evaluate_predictions_against_recent(predictions, file_path, num_recent=5):
    """Evaluate predictions against the most recent draws.
    
    Args:
        predictions: List of prediction dictionaries
        file_path: Path to lottery data file
        num_recent: Number of recent draws to evaluate against
        
    Returns:
        Dictionary with evaluation metrics
    """
    print(f"Evaluating predictions against {num_recent} most recent draws...")
    
    # Load data
    processor = LotteryDataProcessor(file_path)
    data = processor.parse_file()
    
    if data.empty or len(data) < num_recent:
        print("Not enough historical data for evaluation")
        return {
            'success': False,
            'error': 'Insufficient historical data'
        }
    
    # Get recent draws
    recent_draws = data.tail(num_recent)
    
    # Evaluate each prediction against each recent draw
    results = []
    
    for i, prediction in enumerate(predictions):
        draw_scores = []
        
        for _, row in recent_draws.iterrows():
            # Format actual draw
            actual_draw = {
                "main_numbers": row["main_numbers"],
                "main_number_positions": {num: i for i, num in enumerate(row["main_numbers"])},
                "bonus_numbers": row["bonus_numbers"]
            }
            
            # Calculate match score
            match_score = Utilities.calculate_partial_match_score(prediction, actual_draw)
            
            draw_date = row["date"].strftime("%Y-%m-%d") if hasattr(row["date"], "strftime") else str(row["date"])
            
            draw_scores.append({
                "draw_date": draw_date,
                "match_score": match_score,
                "main_matches": len(set(prediction["main_numbers"]).intersection(set(row["main_numbers"]))),
                "bonus_matches": len(set(prediction["bonus_numbers"]).intersection(set(row["bonus_numbers"])))
            })
        
        # Find best matching draw
        best_match = max(draw_scores, key=lambda x: x["match_score"]) if draw_scores else None
        
        results.append({
            "prediction_index": i + 1,
            "main_numbers": prediction["main_numbers"],
            "bonus_numbers": prediction["bonus_numbers"],
            "confidence": prediction["confidence"]["overall"],
            "draw_scores": draw_scores,
            "best_match": best_match,
            "avg_score": np.mean([s["match_score"] for s in draw_scores]) if draw_scores else 0.0
        })
    
    # Calculate overall metrics
    avg_match_score = np.mean([r["avg_score"] for r in results]) if results else 0.0
    best_prediction = max(results, key=lambda x: x["avg_score"]) if results else None
    
    print(f"Evaluation complete: Average Match Score = {avg_match_score:.4f}")
    if best_prediction:
        print(f"Best prediction ({best_prediction['avg_score']:.4f}): {best_prediction['main_numbers']} + {best_prediction['bonus_numbers']}")
    
    return {
        'success': True,
        'avg_match_score': avg_match_score,
        'detailed_results': results,
        'best_prediction': best_prediction
    }

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
        
        # Convert probs to array format if needed
        if isinstance(probs, list) and isinstance(probs[0], np.ndarray):
            # Already in the expected format
            position_probs_list = probs
        else:
            # Convert to list of position probabilities
            position_probs_list = []
            for i in range(num_to_select):
                if isinstance(probs[i], np.ndarray):
                    position_probs_list.append(probs[i])
                else:
                    # Reshape to expected format
                    position_probs_list.append(np.array([probs[i]]))
        
        for i in range(num_to_select):
            # Get probability distribution for this position
            if i < len(position_probs_list):
                position_probs = position_probs_list[i][0]
            else:
                # Fallback to uniform distribution if position probs not available
                position_probs = np.ones(max(available_nums)) / max(available_nums)
            
            # Apply temperature scaling
            position_probs = np.power(position_probs, 1/max(0.1, temperature))
            sum_probs = np.sum(position_probs)
            if sum_probs > 0:
                position_probs = position_probs / sum_probs
            
            # Adjust probabilities to only include available numbers
            adjusted_probs = np.zeros(len(position_probs))
            for num in available:
                if 1 <= num <= len(position_probs):  # Ensure index is valid
                    # Handle different probabilities format
                    prob_value = position_probs[num-1]
                    # If prob_value is an array or sequence, take its first element
                    if hasattr(prob_value, '__len__') and not isinstance(prob_value, (str, bytes)):
                        prob_value = prob_value[0]
                    adjusted_probs[num-1] = prob_value
                    
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
        # Check distribution across range
        main_bins = [0] * 5  # 1-10, 11-20, 21-30, 31-40, 41-50
        for num in main_numbers:
            bin_idx = (num - 1) // 10
            if 0 <= bin_idx < 5:
                main_bins[bin_idx] += 1
        
        # Calculate distribution score - balanced distribution is better
        bin_variance = np.var(main_bins)
        distribution_score = 1.0 / (1.0 + bin_variance)
        
        # Check odd/even balance
        even_count = sum(1 for n in main_numbers if n % 2 == 0)
        
        # Ideal balance is 2-3 or 3-2
        if even_count in [2, 3]:
            balance_score = 0.9
        elif even_count in [1, 4]:
            balance_score = 0.7
        else:
            balance_score = 0.5  # All even or all odd is rare
            
        # Check sum of numbers - most common sums fall in certain ranges
        num_sum = sum(main_numbers)
        if 120 <= num_sum <= 170:
            sum_score = 0.9
        elif 100 <= num_sum <= 200:
            sum_score = 0.7
        else:
            sum_score = 0.5
            
        # Check for consecutive numbers - having 0 or 1 is better
        consecutive_count = 0
        sorted_nums = sorted(main_numbers)
        for i in range(len(sorted_nums) - 1):
            if sorted_nums[i + 1] - sorted_nums[i] == 1:
                consecutive_count += 1
                
        if consecutive_count <= 1:
            consecutive_score = 0.9
        elif consecutive_count == 2:
            consecutive_score = 0.7
        else:
            consecutive_score = 0.5
            
        # Calculate average gaps
        gaps = [sorted_nums[i+1] - sorted_nums[i] for i in range(len(sorted_nums)-1)]
        avg_gap = np.mean(gaps) if gaps else 0
        
        if 8 <= avg_gap <= 12:
            gap_score = 0.9
        elif 6 <= avg_gap <= 15:
            gap_score = 0.7
        else:
            gap_score = 0.5
            
        # Check low/high distribution (numbers below and above 25)
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
            0.2 * balance_score + 
            0.15 * sum_score + 
            0.15 * consecutive_score +
            0.1 * gap_score +
            0.15 * low_high_score
        )
        
        # Bonus number pattern (if provided)
        if bonus_numbers and len(bonus_numbers) >= 2:
            # Check bonus numbers distribution
            bonus_diff = abs(bonus_numbers[1] - bonus_numbers[0])
            if 2 <= bonus_diff <= 6:  # Most common gap
                bonus_score = 0.9
            elif 1 <= bonus_diff <= 8:  # Wider common range
                bonus_score = 0.7
            else:
                bonus_score = 0.5  # Uncommon gap
                
            # Check even/odd balance in bonus numbers
            bonus_even = sum(1 for n in bonus_numbers if n % 2 == 0)
            if bonus_even == 1:  # One even, one odd is most common
                bonus_balance_score = 0.9
            else:
                bonus_balance_score = 0.6  # Both even or both odd is less common
                
            bonus_combined_score = (bonus_score + bonus_balance_score) / 2
            
            # Combine with main score
            pattern_score = 0.85 * pattern_score + 0.15 * bonus_combined_score
            
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
        
        # Combine scores with weights - more emphasis on long-term frequency
        combined_main_score = 0.7 * main_score_norm + 0.3 * main_recent_score_norm
        combined_bonus_score = 0.7 * bonus_score_norm + 0.3 * bonus_recent_score_norm
        
        return (combined_main_score * 0.8 + combined_bonus_score * 0.2)
    
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
                                if prediction["main_number_positions"].get(str(num)) == actual_draw["main_number_positions"].get(str(num)))
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
            'l2_regularization': 0.0001,
            'model_type': 'transformer',
            'lstm_units': 32,
            'min_delta': MIN_DELTA,
            'use_residual': True,
            'use_layer_scaling': True,
            'use_frequency_model': True,
            'use_pattern_model': True,
            'use_transformer_model': True,
            'ensemble_weights': {
                'transformer': 0.4,
                'frequency': 0.3,
                'pattern': 0.3
            }
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
                restore_best_weights=True,  # Important: restore weights in memory
                monitor='val_loss',
                min_delta=min_delta,
                verbose=1
            ),
            # Learning rate scheduler - improved with more gradual reduction
            ReduceLROnPlateau(
                factor=0.7,
                patience=max(3, patience // 3),
                min_lr=1e-6,
                monitor='val_loss',
                verbose=1
            )
        ]
        
        # Model-specific callbacks with checkpoints
        model_callbacks = {}
        checkpoint_paths = {}
        
        # Use weights-only saving
        save_weights_only = True
        
        # Add model-specific callbacks if needed
        for model_type in model_types:
            model_callbacks[model_type] = common_callbacks.copy()
            
            # Add checkpoint callback if requested
            if include_checkpoint:
                # Use proper extension based on save_weights_only flag
                if save_weights_only:
                    extension = ".weights.h5"
                else:
                    extension = ".keras"
                    
                if include_timestamp:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filepath = f"{base_path}_{model_type}_{timestamp}{extension}"
                else:
                    filepath = f"{base_path}_{model_type}{extension}"
                
                # Create directory if it doesn't exist
                directory = os.path.dirname(os.path.abspath(filepath))
                if directory and not os.path.exists(directory):
                    os.makedirs(directory, exist_ok=True)
                    
                checkpoint_callback = ModelCheckpoint(
                    filepath=filepath,
                    save_best_only=True,
                    monitor='val_loss',
                    mode='min',
                    save_weights_only=save_weights_only,
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
    """Enhanced processor for lottery data from various sources."""
    
    def __init__(self, file_path=None, data=None):
        """Initialize with either a file path or direct data"""
        self.file_path = file_path
        self.raw_data = data
        self.data = None
        self.expanded_data = None
        self.features = None
        self.sequence_length = DEFAULT_SEQUENCE_LENGTH
        
        # Feature scalers
        self.feature_scaler = StandardScaler()
        
        # Cache
        self.cache_file = os.path.join(CACHE_DIR, "data_processor_cache.pkl")
    
    def set_sequence_length(self, length):
        """Set the sequence length for data processing."""
        self.sequence_length = length
        return self
    
    def load_cache(self):
        """Load cached data if available."""
        if os.path.exists(self.cache_file):
            try:
                cache = np.load(self.cache_file, allow_pickle=True).item()
                
                # Match file path or content hash to verify cache validity
                if ((self.file_path and cache.get('file_path') == self.file_path) or 
                    (self.raw_data is not None and cache.get('data_hash') == hash(str(self.raw_data)))):
                    logger.info("Using cached processed data")
                    self.data = cache.get('data')
                    self.expanded_data = cache.get('expanded_data')
                    self.features = cache.get('features')
                    self.feature_scaler = cache.get('feature_scaler')
                    return True
            except Exception as e:
                logger.warning(f"Error loading cache: {str(e)}")
                
        return False
    
    def save_cache(self):
        """Save processed data to cache."""
        try:
            # Create dir if not exists
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            
            cache = {
                'file_path': self.file_path,
                'data_hash': hash(str(self.raw_data)) if self.raw_data is not None else None,
                'data': self.data,
                'expanded_data': self.expanded_data,
                'features': self.features,
                'feature_scaler': self.feature_scaler
            }
            
            np.save(self.cache_file, cache)
            logger.info(f"Data processing cache saved to {self.cache_file}")
        except Exception as e:
            logger.warning(f"Could not save cache: {e}")
    
    @ErrorHandler.handle_exception(logger, "lottery data parsing", pd.DataFrame())
    def parse_file(self):
        """Parse lottery data from file or raw data with enhanced pattern matching."""
        # Check cache first
        if self.load_cache():
            return self.data
            
        logger.info("Parsing lottery data")
        
        if self.raw_data is not None:
            content = self.raw_data
        elif self.file_path and os.path.exists(self.file_path):
            with open(self.file_path, 'r') as file:
                content = file.read()
        else:
            raise ValueError("No valid data source provided")
            
        # Improved regex for more robust matching
        draw_pattern = r"((?:\w+)\s+\d+(?:st|nd|rd|th)?\s+(?:\w+)\s+\d{4})\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(€[\d,]+)\s+(Roll|Won)"
        draws = re.findall(draw_pattern, content)
        
        if not draws:
            raise ValueError("No valid draws found in data")
            
        structured_data = []
        
        for draw in draws:
            try:
                date_str = draw[0]
                main_numbers = [int(draw[i]) for i in range(1, 6)]
                bonus_numbers = [int(draw[i]) for i in range(6, 8)]
                jackpot_str = draw[8]
                result = draw[9]
                
                # Improved date parsing
                clean_date_str = re.sub(r'(\d+)(st|nd|rd|th)', r'\1', date_str)
                try:
                    date = datetime.strptime(clean_date_str, "%A %d %B %Y")
                except ValueError:
                    try:
                        date = datetime.strptime(clean_date_str, "%a %d %b %Y")
                    except ValueError:
                        # Extract parts manually as last resort
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
                            # Last resort, use current date
                            date = datetime.now()
                            logger.warning(f"Could not parse date: {date_str}, using default")
                
                # Extract jackpot value
                try:
                    jackpot_value = float(jackpot_str.replace('€', '').replace(',', ''))
                except ValueError:
                    jackpot_value = 0.0
                    
                # Create main numbers position map
                main_positions = {num: idx for idx, num in enumerate(main_numbers)}
                
                structured_data.append({
                    "date": date,
                    "day_of_week": date.strftime("%A"),
                    "main_numbers": sorted(main_numbers),
                    "main_number_positions": main_positions,
                    "bonus_numbers": sorted(bonus_numbers),
                    "jackpot": jackpot_str,
                    "jackpot_value": jackpot_value,
                    "result": result
                })
            except Exception as e:
                logger.warning(f"Error parsing draw: {str(e)}")
                continue
        
        # Convert to DataFrame and sort by date
        df = pd.DataFrame(structured_data)
        
        if df.empty:
            logger.error("No valid draws could be parsed")
            return df
            
        df = df.sort_values("date")
        
        # Determine sequence length if not already set
        if self.sequence_length is None:
            optimal_length = Utilities.calculate_optimal_sequence_length(len(df))
            self.sequence_length = optimal_length
            logger.info(f"Dynamic sequence length set to {optimal_length}")
        
        self.data = df
        logger.info(f"Successfully parsed {len(df)} draws")
        
        # Save to cache
        self.save_cache()
        
        return df
    
    @ErrorHandler.handle_exception(logger, "data expansion", pd.DataFrame())
    def expand_numbers(self):
        """Expand main and bonus numbers into individual columns with enhanced metadata."""
        if self.data is None or self.data.empty:
            logger.error("No data available. Parse the file first.")
            return pd.DataFrame()
        
        # Check cache first
        if self.expanded_data is not None:
            return self.expanded_data
            
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
        
        # Handle week of year consistently
        try:
            df["week_of_year"] = df["date"].dt.isocalendar().week
        except:
            try:
                df["week_of_year"] = df["date"].dt.weekofyear
            except:
                # Manual calculation as last resort
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
        df["main_number_mean"] = df["main_numbers"].apply(lambda x: np.mean(x) if x else 0)
        df["bonus_number_diff"] = df["bonus_numbers"].apply(lambda x: x[1] - x[0] if len(x) >= 2 else 0)
        
        # Distribution metrics
        df["main_even_count"] = df["main_numbers"].apply(lambda x: sum(1 for n in x if n % 2 == 0))
        df["main_odd_count"] = df["main_numbers"].apply(lambda x: sum(1 for n in x if n % 2 == 1))
        
        # Create one-hot encoding for main and bonus numbers
        for i in range(1, MAIN_NUM_MAX + 1):
            df[f'main_has_{i}'] = df['main_numbers'].apply(lambda x: 1 if i in x else 0)
            
        for i in range(1, BONUS_NUM_MAX + 1):
            df[f'bonus_has_{i}'] = df['bonus_numbers'].apply(lambda x: 1 if i in x else 0)
        
        # Decade distribution (1-10, 11-20, etc.)
        for decade in range(5):
            start = decade * 10 + 1
            end = start + 9
            df[f"main_decade_{decade+1}_count"] = df["main_numbers"].apply(
                lambda x: sum(1 for n in x if start <= n <= end)
            )
        
        # Calculate consecutive numbers
        df["main_consecutive_count"] = df["main_numbers"].apply(
            lambda x: sum(1 for i in range(len(x)-1) if sorted(x)[i+1] - sorted(x)[i] == 1)
        )
        
        # Low/high distribution
        df["main_low_count"] = df["main_numbers"].apply(lambda x: sum(1 for n in x if n <= 25))
        df["main_high_count"] = df["main_numbers"].apply(lambda x: sum(1 for n in x if n > 25))
        df["main_low_high_ratio"] = df["main_low_count"] / df["main_high_count"].replace(0, 0.5)
        
        # Cyclical encodings for temporal features
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
        df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week_num'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week_num'] / 7)
        
        self.expanded_data = df
        logger.info(f"Created expanded data with {df.shape[1]} columns")
        
        # Save to cache
        self.save_cache()
        
        return df
    
    @ErrorHandler.handle_exception(logger, "feature engineering", pd.DataFrame())
    def create_features(self):
        """Generate advanced features for lottery prediction."""
        if self.expanded_data is None or self.expanded_data.empty:
            self.expanded_data = self.expand_numbers()
            if self.expanded_data.empty:
                return pd.DataFrame()
        
        # Check cache first
        if self.features is not None:
            return self.features
                
        # Create feature engineering instance
        feature_engineer = FeatureEngineering(self.expanded_data)
        self.features = feature_engineer.create_enhanced_features()
        
        # Save to cache
        self.save_cache()
        
        return self.features
    
    def create_sequences(self):
        """Create sequence data for transformer model."""
        if self.expanded_data is None or self.expanded_data.empty:
            self.expanded_data = self.expand_numbers()
            if self.expanded_data.empty:
                return None, None, None, None
        
        logger.info(f"Creating sequence data (length: {self.sequence_length})")
        
        df = self.expanded_data
        
        # Process in sliding windows
        num_samples = len(df) - self.sequence_length
        
        if num_samples <= 0:
            logger.error(f"Not enough data for sequence length {self.sequence_length}")
            return None, None, None, None
        
        # Pre-allocate arrays for main and bonus sequences
        main_sequences = np.zeros((num_samples, self.sequence_length, MAIN_NUM_MAX))
        bonus_sequences = np.zeros((num_samples, self.sequence_length, BONUS_NUM_MAX))
        
        # Extract main and bonus columns more efficiently
        main_cols = [f"main_{i+1}" for i in range(MAIN_NUM_COUNT)]
        bonus_cols = [f"bonus_{i+1}" for i in range(BONUS_NUM_COUNT)]
        
        # Check for NaN values in columns
        if df[main_cols + bonus_cols].isna().any().any():
            logger.warning("NaN values detected in number columns. Filling with -1 for processing.")
            df[main_cols + bonus_cols] = df[main_cols + bonus_cols].fillna(-1)
        
        # Process each window
        for i in range(num_samples):
            # Get the window of previous draws
            window = df.iloc[i:i+self.sequence_length]
            
            # For each draw in the window
            for j, (_, row) in enumerate(window.iterrows()):
                # Convert main numbers to one-hot encoding
                main_nums = np.array([row[col] for col in main_cols if not pd.isna(row[col])]).astype(int)
                main_nums = main_nums[main_nums > 0] - 1  # Convert to 0-based indices
                if len(main_nums) > 0:
                    main_sequences[i, j, main_nums] = 1
                
                # Same for bonus numbers
                bonus_nums = np.array([row[col] for col in bonus_cols if not pd.isna(row[col])]).astype(int)
                bonus_nums = bonus_nums[bonus_nums > 0] - 1  # Convert to 0-based indices
                if len(bonus_nums) > 0:
                    bonus_sequences[i, j, bonus_nums] = 1
        
        # Create target variables (next draw's numbers)
        # Subtract 1 from each number to use as index (0-49 for main, 0-11 for bonus)
        y_main = np.array([
            df.iloc[self.sequence_length:][f"main_{i+1}"].values - 1 for i in range(MAIN_NUM_COUNT)
        ]).T
        
        y_bonus = np.array([
            df.iloc[self.sequence_length:][f"bonus_{i+1}"].values - 1 for i in range(BONUS_NUM_COUNT)
        ]).T
        
        logger.info(f"Created sequences: Main shape: {main_sequences.shape}, Bonus shape: {bonus_sequences.shape}")
        
        return main_sequences, bonus_sequences, y_main, y_bonus

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
        feature_sets = []
        
        # Time series features
        ts_features = self._calculate_time_series_features()
        feature_sets.append(ts_features)
        
        # Number relationships
        relationship_features = self._calculate_number_relationships()
        feature_sets.append(relationship_features)
        
        # Pattern features
        pattern_features = self._calculate_pattern_features()
        feature_sets.append(pattern_features)
        
        # Cyclical features
        cyclical_features = self._calculate_cyclical_features()
        feature_sets.append(cyclical_features)
        
        # Autocorrelation features
        autocorr_features = self._calculate_autocorrelation_features()
        feature_sets.append(autocorr_features)
        
        # Statistical features
        stat_features = self._calculate_statistical_features()
        feature_sets.append(stat_features)
        
        # Combine all feature sets
        valid_feature_sets = [fs for fs in feature_sets if not fs.empty]
        
        if not valid_feature_sets:
            logger.error("No valid features could be generated")
            return pd.DataFrame()
            
        all_features = pd.concat(valid_feature_sets, axis=1)
        
        # Handle missing values
        all_features = all_features.fillna(0)
        
        # Handle infinity values
        all_features = all_features.replace([np.inf, -np.inf], 0)
        
        # Select top features to avoid feature explosion
        all_features = self._select_top_features(all_features)
        
        logger.info(f"Created {all_features.shape[1]} enhanced features")
        return all_features
    
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
            
            # Calculate statistics for bonus numbers
            if len(bonus_nums) >= BONUS_NUM_COUNT:
                try:
                    sorted_bonus_nums = np.sort(bonus_nums)
                    
                    # Store difference between bonus numbers
                    if len(sorted_bonus_nums) >= 2:
                        bonus_diff[i] = sorted_bonus_nums[1] - sorted_bonus_nums[0]
                    
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
            except Exception as e:
                logger.debug(f"Error calculating pattern features for row {i}: {e}")
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
                        'pca': pca,
                        'valid_indices': valid_indices
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
                    n_clusters = min(2, len(signatures) // 10 + 1)  # Ensure at least 1 cluster
                    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED, n_init=10)
                    
                    if signatures_array.shape[0] > n_clusters:  # Ensure we have more samples than clusters
                        cluster_labels = kmeans.fit_predict(signatures_array)
                        
                        # Save simple cluster model
                        self.pattern_clusters = {
                            'kmeans': kmeans,
                            'pca': None,
                            'valid_indices': valid_indices,
                            'is_fallback': True
                        }
                        
                        for idx, (orig_idx, cluster) in enumerate(zip(valid_indices, cluster_labels)):
                            pattern_cluster[orig_idx] = cluster
            except Exception as e:
                logger.warning(f"Pattern clustering failed: {e}")
                logger.debug(traceback.format_exc())
        
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
# MODEL COMPONENTS
#######################

class TransformerBlock(tf.keras.layers.Layer):
    """Enhanced transformer block with multi-head attention and feed-forward network."""
    
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

class ModelBuilder:
    """Unified model building for lottery prediction models."""
    
    @staticmethod
    @ErrorHandler.handle_exception(logger, "model building")
    def build_transformer_model(input_dim, seq_length, params, model_config, distribution_strategy=None):
        """Build transformer model for lottery number prediction with flexible configuration."""
        # Import optimizers explicitly here to ensure they're in scope
        from tensorflow.keras.optimizers import Adam, SGD, RMSprop
        
        # Use provided strategy or get default
        if distribution_strategy is None:
            distribution_strategy = tf.distribute.get_strategy()  # Default strategy
        
        with distribution_strategy.scope():
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
            
            # Define conv_filters here to ensure it's always available
            conv_filters = max(1, params.get('conv_filters', DEFAULT_CONV_FILTERS))
            
            # Input layers - use correct dimensions and log them for debugging
            logger.info(f"Creating model with input dimensions: features={input_dim}, seq_length={seq_length}, sequence_size={sequence_size}")
            feature_input = Input(shape=(input_dim,), name=f"{name_prefix}_feature_input")
            sequence_input = Input(shape=(seq_length, sequence_size), name=f"{name_prefix}_sequence_input")
            
            # Process feature input
            x_features = Dense(
                params.get('embed_dim', DEFAULT_EMBED_DIM), 
                activation="gelu",
                kernel_regularizer=l2(l2_reg)
            )(feature_input)
            x_features = BatchNormalization()(x_features)
            x_features = Dropout(params.get('dropout_rate', DEFAULT_DROPOUT_RATE))(x_features)
            
            x_features = Dense(
                params.get('embed_dim', DEFAULT_EMBED_DIM) // 2, 
                activation="gelu",
                kernel_regularizer=l2(l2_reg)
            )(x_features)
            
            # Process sequence input based on model type
            if model_type == 'transformer':
                # Optional convolutional layer for sequence processing
                x_seq = Conv1D(
                    conv_filters, 
                    kernel_size=3, 
                    padding='same', 
                    activation="relu",
                    kernel_regularizer=l2(l2_reg)
                )(sequence_input)
                x_seq = BatchNormalization()(x_seq)
                    
                # Embedding layer
                x_seq = Dense(
                    params.get('embed_dim', DEFAULT_EMBED_DIM),
                    kernel_regularizer=l2(l2_reg)
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
                        kernel_regularizer=l2(l2_reg)
                    ))(x_seq)
                else:
                    x_seq = GlobalAveragePooling1D()(x_seq)
                        
            elif model_type == 'lstm':
                # LSTM-based architecture
                x_seq = Conv1D(
                    conv_filters, 
                    kernel_size=3, 
                    padding='same', 
                    activation="gelu",
                    kernel_regularizer=l2(l2_reg)
                )(sequence_input)
                
                x_seq = BatchNormalization()(x_seq)
                
                # Bidirectional LSTM
                x_seq = Bidirectional(LSTM(
                    lstm_units,
                    return_sequences=True,
                    kernel_regularizer=l2(l2_reg),
                    recurrent_regularizer=l2(l2_reg),
                    dropout=params.get('dropout_rate', DEFAULT_DROPOUT_RATE) * 0.5,
                    recurrent_dropout=params.get('dropout_rate', DEFAULT_DROPOUT_RATE) * 0.5
                ))(x_seq)
                
                # Another LSTM layer
                x_seq = LSTM(
                    lstm_units,
                    return_sequences=False,
                    kernel_regularizer=l2(l2_reg),
                    recurrent_regularizer=l2(l2_reg),
                    dropout=params.get('dropout_rate', DEFAULT_DROPOUT_RATE) * 0.5,
                    recurrent_dropout=params.get('dropout_rate', DEFAULT_DROPOUT_RATE) * 0.5
                )(x_seq)
            
            elif model_type == 'hybrid':
                # New hybrid architecture combining transformer and RNN elements
                x_seq = Conv1D(
                    conv_filters, 
                    kernel_size=3, 
                    padding='same', 
                    activation="gelu",
                    kernel_regularizer=l2(l2_reg)
                )(sequence_input)
                x_seq = BatchNormalization()(x_seq)
                    
                # First apply positional encoding
                x_seq = Dense(
                    params.get('embed_dim', DEFAULT_EMBED_DIM),
                    kernel_regularizer=l2(l2_reg)
                )(x_seq)
                
                x_seq = PositionalEncoding(
                    max_position=seq_length + 5,
                    d_model=params.get('embed_dim', DEFAULT_EMBED_DIM)
                )(x_seq)
                
                # Apply transformer block
                x_seq = TransformerBlock(
                    embed_dim=params.get('embed_dim', DEFAULT_EMBED_DIM),
                    num_heads=params.get('num_heads', DEFAULT_NUM_HEADS),
                    ff_dim=params.get('ff_dim', DEFAULT_FF_DIM),
                    dropout=params.get('dropout_rate', DEFAULT_DROPOUT_RATE),
                    use_residual=use_residual,
                    use_layer_scaling=use_layer_scaling
                )(x_seq)
                
                # Then process with LSTM or GRU
                gru_units = params.get('gru_units', 32)
                lstm_units = params.get('lstm_units', 32)
                
                # Split processing path
                path1 = GRU(
                    gru_units, 
                    return_sequences=True,
                    kernel_regularizer=l2(l2_reg)
                )(x_seq)
                
                path2 = LSTM(
                    lstm_units, 
                    return_sequences=True,
                    kernel_regularizer=l2(l2_reg)
                )(x_seq)
                
                # Combine paths
                x_seq = Concatenate()([path1, path2])
                
                # Final sequence processing
                x_seq = GlobalAveragePooling1D()(x_seq)
                
            elif model_type == 'rnn':
                # Simple RNN architecture
                x_seq = SimpleRNN(
                    params.get('embed_dim', DEFAULT_EMBED_DIM),
                    return_sequences=True,
                    kernel_regularizer=l2(l2_reg)
                )(sequence_input)
                
                x_seq = Dropout(params.get('dropout_rate', DEFAULT_DROPOUT_RATE))(x_seq)
                
                x_seq = SimpleRNN(
                    params.get('embed_dim', DEFAULT_EMBED_DIM) // 2,
                    return_sequences=False,
                    kernel_regularizer=l2(l2_reg)
                )(x_seq)
            
            else:
                # Fallback to CNN architecture
                x_seq = Conv1D(
                    conv_filters * 2, 
                    kernel_size=5, 
                    padding='same', 
                    activation="relu",
                    kernel_regularizer=l2(l2_reg)
                )(sequence_input)
                
                x_seq = BatchNormalization()(x_seq)
                x_seq = Dropout(params.get('dropout_rate', DEFAULT_DROPOUT_RATE))(x_seq)
                
                x_seq = Conv1D(
                    conv_filters, 
                    kernel_size=3, 
                    padding='same', 
                    activation="relu",
                    kernel_regularizer=l2(l2_reg)
                )(x_seq)
                
                x_seq = GlobalAveragePooling1D()(x_seq)
            
            # Combine feature and sequence representations
            combined = Concatenate()([x_features, x_seq])
            
            # Dense layers
            combined = Dense(
                params.get('ff_dim', DEFAULT_FF_DIM), 
                activation="gelu",
                kernel_regularizer=l2(l2_reg)
            )(combined)
            combined = BatchNormalization()(combined)
            combined = Dropout(params.get('dropout_rate', DEFAULT_DROPOUT_RATE))(combined)
            
            combined = Dense(
                params.get('ff_dim', DEFAULT_FF_DIM) // 2, 
                activation="gelu",
                kernel_regularizer=l2(l2_reg)
            )(combined)
            combined = Dropout(params.get('dropout_rate', DEFAULT_DROPOUT_RATE) / 2)(combined)
            
            # Output layers (one for each position)
            outputs = []
            for i in range(num_outputs):
                # Position-specific processing
                position_specific = Dense(
                    params.get('ff_dim', DEFAULT_FF_DIM) // 4, 
                    activation="gelu",
                    kernel_regularizer=l2(l2_reg)
                )(combined)
                
                # Logits
                logits = Dense(
                    output_size, 
                    activation=None, 
                    name=f"{name_prefix}_logits_{i+1}",
                    kernel_regularizer=l2(l2_reg)
                )(position_specific)
                
                # Apply softmax activation
                output = Activation('softmax', name=f"{name_prefix}_{i+1}")(logits)
                outputs.append(output)
            
            # Create model
            model = Model(inputs=[feature_input, sequence_input], outputs=outputs)
            
            # Configure metrics
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
            elif optimizer_name == 'adamw':
                # Add support for AdamW optimizer
                # Some versions of TF don't have AdamW directly
                try:
                    from tensorflow.keras.optimizers import AdamW
                    optimizer = AdamW(learning_rate=learning_rate, weight_decay=0.01)
                except ImportError:
                    # Fallback to Adam with L2 regularization if AdamW not available
                    logger.warning("AdamW optimizer not available, falling back to Adam")
                    optimizer = Adam(learning_rate=learning_rate)
            else:
                optimizer = Adam(learning_rate=learning_rate)

            # Create loss dictionary - one entry per output
            loss_dict = {f"{name_prefix}_{i+1}": "sparse_categorical_crossentropy" for i in range(num_outputs)}

            # Compile model
            model.compile(
                optimizer=optimizer,
                loss=loss_dict,
                metrics=metrics_dict
            )

            return model

#######################
# NEURAL NETWORK MODELS
#######################

class TransformerModel:
    """Neural sequence model for lottery prediction."""
    
    def __init__(self, input_shape, output_dim, params=None, distribution_strategy=None):
        """
        Initialize the sequence model with proper shape handling and distribution strategy.
        
        Args:
            input_shape: Tuple of (feature_dim, sequence_dim)
            output_dim: Output dimension (usually MAIN_NUM_MAX)
            params: Dictionary of model parameters
            distribution_strategy: TensorFlow distribution strategy for multi-GPU training
        """
        self.input_shape = input_shape
        self.seq_dim = input_shape[1]  # Store sequence dimension for clarity
        self.features_dim = input_shape[0]  # Store feature dimension
        self.output_dim = output_dim
        self.params = params or Utilities.get_default_params()
        self.model = None
        self.history = None
        self.checkpoint_path = os.path.join(MODEL_DIR, "transformer_model.keras")
        self.distribution_strategy = distribution_strategy or tf.distribute.get_strategy()
    
    def build_model(self):
        """Build neural sequence model with transformer architecture."""
        # Configure model - explicitly use the dimensions from initialization
        model_config = {
            'num_outputs': MAIN_NUM_COUNT,
            'output_size': self.output_dim,
            'sequence_size': self.seq_dim,  # Use stored sequence dimension
            'name_prefix': 'transformer'
        }
        
        # Ensure sequence_length is set correctly from params
        seq_length = self.params.get('sequence_length', DEFAULT_SEQUENCE_LENGTH)
        
        # Log the dimensions being used to build the model
        logger.info(f"Building transformer model with dimensions: features={self.features_dim}, sequence_length={seq_length}, sequence_size={self.seq_dim}")
        
        # Build model with the correct dimensions and distribution strategy
        self.model = ModelBuilder.build_transformer_model(
            input_dim=self.features_dim,
            seq_length=seq_length,
            params=self.params,
            model_config=model_config,
            distribution_strategy=self.distribution_strategy
        )
        
        return self.model
    
    def _load_best_weights(self):
        """Helper method to load the best weights after training."""
        if self.checkpoint_path and os.path.exists(self.checkpoint_path):
            try:
                logger.info(f"Loading best weights from {self.checkpoint_path}")
                self.model.load_weights(self.checkpoint_path)
                logger.info("Successfully loaded best weights")
                return True
            except Exception as e:
                logger.warning(f"Could not load weights from checkpoint: {e}")
                logger.warning("Using current model weights instead (from EarlyStopping callback)")
                
                # In case the checkpoint file is corrupted, try to remove it
                try:
                    os.remove(self.checkpoint_path)
                    logger.info(f"Removed potentially corrupted checkpoint file: {self.checkpoint_path}")
                except:
                    pass
        else:
            logger.info("No checkpoint file found. Using weights from EarlyStopping callback.")
        
        return False
    
    @ErrorHandler.handle_exception(logger, "transformer model training")
    def train(self, X_train, y_train, validation_split=0.2, epochs=None, batch_size=None):
        """Train the model with early stopping and learning rate scheduling."""
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        # Use parameter values or defaults
        actual_epochs = epochs if epochs is not None else self.params.get('epochs', DEFAULT_EPOCHS)
        base_batch_size = batch_size if batch_size is not None else self.params.get('batch_size', DEFAULT_BATCH_SIZE)
        
        # Scale batch size by number of replicas (GPUs)
        num_replicas = self.distribution_strategy.num_replicas_in_sync
        scaled_batch_size = base_batch_size * num_replicas
        logger.info(f"Scaling batch size from {base_batch_size} to {scaled_batch_size} for {num_replicas} GPU(s)")
        
        # Get callbacks
        callbacks, self.checkpoint_path = Utilities.get_model_callbacks(
            model_types="transformer",
            patience=DEFAULT_PATIENCE,
            min_delta=self.params.get('min_delta', MIN_DELTA)
        )
        
        # Ensure inputs are properly formatted
        if isinstance(X_train, tuple) and len(X_train) == 2:
            X_features, X_sequences = X_train
        else:
            X_features = X_train[0]
            X_sequences = X_train[1]
        
        # Verify and log data shapes
        logger.info(f"Initial data shapes: X_features={X_features.shape}, X_sequences={X_sequences.shape}")
        if isinstance(y_train, np.ndarray):
            logger.info(f"y_train shape: {y_train.shape}")
        elif isinstance(y_train, list) and len(y_train) > 0:
            logger.info(f"y_train is a list with {len(y_train)} elements. First element shape: {y_train[0].shape}")
            
        # Ensure all arrays have the same number of samples
        if X_features.shape[0] != X_sequences.shape[0]:
            min_samples = min(X_features.shape[0], X_sequences.shape[0])
            logger.warning(f"Input arrays have different sizes. Truncating to {min_samples} samples.")
            X_features = X_features[:min_samples]
            X_sequences = X_sequences[:min_samples]
        
        # Format y_train for multi-output model
        if isinstance(y_train, np.ndarray) and len(y_train.shape) == 2:
            # Ensure y_train has the right number of samples
            if y_train.shape[0] != X_features.shape[0]:
                min_samples = min(y_train.shape[0], X_features.shape[0])
                logger.warning(f"Target and input arrays have different sizes. Truncating all to {min_samples} samples.")
                X_features = X_features[:min_samples]
                X_sequences = X_sequences[:min_samples]
                y_train = y_train[:min_samples]
                
            # Convert 2D array to list of arrays, one for each output
            y_train_formatted = [y_train[:, i] for i in range(y_train.shape[1])]
        else:
            # If y_train is already a list of arrays
            y_train_formatted = y_train
            
            # Check if the first element has the right shape
            if isinstance(y_train_formatted, list) and len(y_train_formatted) > 0:
                first_y = y_train_formatted[0]
                if first_y.shape[0] != X_features.shape[0]:
                    min_samples = min(first_y.shape[0], X_features.shape[0])
                    logger.warning(f"Target and input arrays have different sizes. Truncating all to {min_samples} samples.")
                    X_features = X_features[:min_samples]
                    X_sequences = X_sequences[:min_samples]
                    y_train_formatted = [y[:min_samples] for y in y_train_formatted]
        
        # Log final data shapes after adjustments
        logger.info(f"Final data shapes: X_features={X_features.shape}, X_sequences={X_sequences.shape}")
        if isinstance(y_train_formatted, list) and len(y_train_formatted) > 0:
            logger.info(f"y_train_formatted first element shape: {y_train_formatted[0].shape}")
        
        # Create our own train/validation split to ensure consistency
        if validation_split > 0:
            split_idx = int(X_features.shape[0] * (1 - validation_split))
            
            # Training data
            X_train_features = X_features[:split_idx]
            X_train_sequences = X_sequences[:split_idx]
            y_train_formatted_split = [y[:split_idx] for y in y_train_formatted]
            
            # Validation data
            X_val_features = X_features[split_idx:]
            X_val_sequences = X_sequences[split_idx:]
            y_val_formatted = [y[split_idx:] for y in y_train_formatted]
            
            logger.info(f"Created manual train/validation split: {split_idx} training samples, "
                    f"{X_features.shape[0] - split_idx} validation samples")
            
            # Train with manual validation data
            self.history = self.model.fit(
                [X_train_features, X_train_sequences],
                y_train_formatted_split,
                epochs=actual_epochs,
                batch_size=scaled_batch_size,
                validation_data=([X_val_features, X_val_sequences], y_val_formatted),
                callbacks=callbacks,
                verbose=1
            )
        else:
            # Train without validation
            self.history = self.model.fit(
                [X_features, X_sequences],
                y_train_formatted,
                epochs=actual_epochs,
                batch_size=scaled_batch_size,
                callbacks=callbacks,
                verbose=1
            )
        
        self._load_best_weights()
                    
        # We're relying on EarlyStopping callback's restore_best_weights=True as backup
        logger.info("Training completed. Model is using the weights from best validation loss.")
        
        return self.history
    
    def predict(self, X):
        """Generate predictions for input data."""
        if self.model is None:
            raise ValueError("Model not trained")
            
        # Ensure X is in the right format
        if isinstance(X, tuple) and len(X) == 2:
            return self.model.predict(X, verbose=0)
        elif isinstance(X, list) and len(X) == 2:
            return self.model.predict(X, verbose=0)
        else:
            raise ValueError("Input must be a tuple or list of (features, sequences)")
    
    def save(self, path=None):
        """Save model to disk."""
        if self.model is None:
            raise ValueError("No model to save")
            
        save_path = path or self.checkpoint_path
        self.model.save(save_path)
        logger.info(f"Model saved to {save_path}")
        
    def load(self, path=None):
        """Load model from disk."""
        load_path = path or self.checkpoint_path
        if os.path.exists(load_path):
            self.model = tf.keras.models.load_model(load_path)
            logger.info(f"Model loaded from {load_path}")
            return True
        else:
            logger.warning(f"No saved model found at {load_path}")
            return False

class FrequencyModel:
    """Frequency-based prediction model."""
    
    def __init__(self, n_estimators=200, learning_rate=0.05, max_depth=5):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.models = []  # One model per position
        self.feature_importances = []
        self.checkpoint_path = os.path.join(MODEL_DIR, "frequency_model.pkl")
    
    def build_model(self, n_positions=MAIN_NUM_COUNT):
        """Build frequency models for each position."""
        for pos in range(n_positions):
            model = GradientBoostingRegressor(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                random_state=RANDOM_SEED,
                validation_fraction=0.2,
                n_iter_no_change=15,
                tol=0.0005
            )
            self.models.append(model)
        
        return self.models
    
    @ErrorHandler.handle_exception(logger, "frequency model training")
    def train(self, X_train, y_train_raw):
        """Train frequency model for each position in the draw."""
        if not self.models:
            raise ValueError("Models not built. Call build_model() first.")
        
        # Train a separate model for each position
        for pos in range(len(self.models)):
            # Extract target for this position
            y_pos = np.array([draw[pos] for draw in y_train_raw])
            
            # Train model
            self.models[pos].fit(X_train, y_pos)
            
            # Store feature importances
            self.feature_importances.append(self.models[pos].feature_importances_)
            
        # Save the trained model
        self.save()
        
        return self.models
    
    def predict(self, X):
        """Generate predictions for each position."""
        if not self.models:
            raise ValueError("Models not trained")
            
        # Predict for each position
        position_predictions = []
        
        for model in self.models:
            pred = model.predict(X)
            position_predictions.append(pred)
            
        return np.array(position_predictions).T
    
    def predict_probabilities(self, X, num_range=MAIN_NUM_MAX):
        """Convert predictions to probability distribution over all numbers."""
        position_preds = self.predict(X)
        
        # Create probability matrix for each number
        probabilities = np.zeros((len(X), num_range))
        
        for i, preds in enumerate(position_preds):
            # For each prediction, create a Gaussian probability distribution
            for pos, pred_num in enumerate(preds):
                # Center of distribution is the predicted number
                center = int(round(pred_num)) - 1  # Convert to 0-indexed
                
                # Ensure center is within bounds
                center = max(0, min(center, num_range - 1))
                
                # Create Gaussian distribution around predicted number
                for j in range(num_range):
                    # Distance from prediction center
                    dist = min(abs(j - center), num_range - abs(j - center))
                    # Add probability mass inversely proportional to distance
                    # Use a spreading factor that depends on position
                    spread = 2.0 + pos * 0.5  # Positions later in draw have wider spread
                    probabilities[i, j] += np.exp(-0.5 * (dist/spread)**2)
        
        # Normalize probabilities
        row_sums = probabilities.sum(axis=1, keepdims=True)
        probabilities = probabilities / row_sums
        
        return probabilities
    
    def get_top_features(self, n=10):
        """Get top n most important features for each position."""
        if not self.feature_importances:
            raise ValueError("No feature importances available. Train model first.")
            
        top_features = []
        
        for pos, importances in enumerate(self.feature_importances):
            # Get feature indices sorted by importance
            sorted_idx = np.argsort(importances)[::-1]
            
            # Get top n features
            top_n = sorted_idx[:n]
            
            top_features.append({
                'position': pos,
                'indices': top_n,
                'importances': importances[top_n]
            })
            
        return top_features
    
    def save(self, path=None):
        """Save models to disk."""
        if not self.models:
            raise ValueError("No models to save")
            
        save_path = path or self.checkpoint_path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        data_to_save = {
            'models': self.models,
            'feature_importances': self.feature_importances
        }
        
        with open(save_path, 'wb') as f:
            import pickle
            pickle.dump(data_to_save, f)
            
        logger.info(f"Frequency models saved to {save_path}")
        
    def load(self, path=None):
        """Load models from disk with enhanced error handling."""
        load_path = path or self.checkpoint_path
        
        if os.path.exists(load_path):
            try:
                with open(load_path, 'rb') as f:
                    import pickle
                    data = pickle.load(f)
                    
                # Validate the loaded data has the expected structure
                if not isinstance(data, dict) or 'models' not in data or 'feature_importances' not in data:
                    logger.error(f"Invalid frequency model format in {load_path}")
                    return False
                    
                self.models = data['models']
                self.feature_importances = data['feature_importances']
                
                # Verify models are valid
                if not all(hasattr(model, 'predict') for model in self.models):
                    logger.error(f"Loaded frequency models are not valid")
                    return False
                    
                logger.info(f"Frequency models loaded from {load_path}")
                return True
            except Exception as e:
                logger.error(f"Error loading frequency models: {e}")
                logger.error(traceback.format_exc())
                return False
        else:
            logger.warning(f"No saved models found at {load_path}")
            return False

class PatternModel:
    """Pattern recognition model for lottery numbers."""
    
    def __init__(self, n_estimators=150, num_range=MAIN_NUM_MAX):
        self.n_estimators = n_estimators
        self.num_range = num_range
        self.models = {}  # One model per number
        self.feature_importances = {}
        self.checkpoint_path = os.path.join(MODEL_DIR, "pattern_model.pkl")
        
    def build_model(self):
        """Build pattern model for each possible number."""
        for num in range(1, self.num_range + 1):
            model = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=8,
                min_samples_split=4,
                min_samples_leaf=2,
                random_state=num  # Different seed for each model
            )
            self.models[num] = model
            
        return self.models
    
    @ErrorHandler.handle_exception(logger, "pattern model training")
    def train(self, X_train, y_train):
        """Train pattern model for each number."""
        if not self.models:
            raise ValueError("Models not built. Call build_model() first.")
            
        # Train a model for each number
        for num in range(1, self.num_range + 1):
            # Target: 1 if number appears in draw, 0 otherwise
            y_num = np.array([1 if num in draw else 0 for draw in y_train])
            
            # Train model
            self.models[num].fit(X_train, y_num)
            
            # Store feature importances
            self.feature_importances[num] = self.models[num].feature_importances_
            
        # Save model
        self.save()
        
        return self.models
        
    def predict_probabilities(self, X):
        """Generate probability for each number appearing in the draw."""
        if not self.models:
            raise ValueError("Models not trained")
            
        # Get probabilities for each number
        probabilities = np.zeros((len(X), self.num_range))
        
        for num, model in self.models.items():
            # Predict probability of this number appearing
            prob = model.predict(X)
            
            # Store in probability matrix (convert to 0-indexed)
            probabilities[:, num-1] = prob
            
        # Normalize probabilities
        row_sums = probabilities.sum(axis=1, keepdims=True)
        if np.any(row_sums == 0):
            # Handle zero sums by setting equal probabilities
            zero_rows = (row_sums == 0).flatten()
            probabilities[zero_rows, :] = 1.0 / self.num_range
            # Recalculate sums
            row_sums = probabilities.sum(axis=1, keepdims=True)
            
        probabilities = probabilities / row_sums
        
        return probabilities
    
    def get_top_numbers(self, X, top_n=10):
        """Get top n most likely numbers for each input."""
        probabilities = self.predict_probabilities(X)
        
        # Get indices of top n numbers
        top_indices = np.argsort(probabilities, axis=1)[:, -top_n:]
        
        # Convert to actual numbers (1-indexed)
        top_numbers = top_indices + 1
        
        return top_numbers, probabilities[np.arange(len(X))[:, np.newaxis], top_indices]
    
    def get_pattern_score(self, numbers):
        """Calculate pattern score based on lottery best practices."""
        return Utilities.calculate_pattern_score(numbers)
    
    def save(self, path=None):
        """Save models to disk."""
        if not self.models:
            raise ValueError("No models to save")
            
        save_path = path or self.checkpoint_path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        data_to_save = {
            'models': self.models,
            'feature_importances': self.feature_importances
        }
        
        with open(save_path, 'wb') as f:
            import pickle
            pickle.dump(data_to_save, f)
            
        logger.info(f"Pattern models saved to {save_path}")
        
    def load(self, path=None):
        """Load models from disk."""
        load_path = path or self.checkpoint_path
        
        if os.path.exists(load_path):
            try:
                with open(load_path, 'rb') as f:
                    import pickle
                    data = pickle.load(f)
                    
                self.models = data['models']
                self.feature_importances = data['feature_importances']
                
                logger.info(f"Pattern models loaded from {load_path}")
                return True
            except Exception as e:
                logger.error(f"Error loading pattern models: {e}")
                return False
        else:
            logger.warning(f"No saved models found at {load_path}")
            return False

class BonusNumberModel:
    """Specialized model for bonus numbers."""
    
    def __init__(self, n_estimators=100, learning_rate=0.05, max_depth=4, num_range=BONUS_NUM_MAX):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.num_range = num_range
        self.position_models = []  # One model per position
        self.number_models = {}  # One model per possible number
        self.checkpoint_path = os.path.join(MODEL_DIR, "bonus_model.pkl")
        
    def build_model(self, n_positions=BONUS_NUM_COUNT):
        """Build position-based and number-based models."""
        # Position models - predict number at each position
        for pos in range(n_positions):
            model = GradientBoostingRegressor(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                random_state=RANDOM_SEED + pos
            )
            self.position_models.append(model)
            
        # Number models - predict probability of each number
        for num in range(1, self.num_range + 1):
            model = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=100 + num
            )
            self.number_models[num] = model
            
        return self.position_models, self.number_models
    
    @ErrorHandler.handle_exception(logger, "bonus model training")
    def train(self, X_train, y_train_raw):
        """Train both position and number models."""
        if not self.position_models or not self.number_models:
            raise ValueError("Models not built. Call build_model() first.")
            
        # Extract bonus numbers from raw targets
        bonus_numbers = []
        for draw_nums in y_train_raw:
            # Extract just the bonus numbers
            bonus_nums = draw_nums[-BONUS_NUM_COUNT:] if len(draw_nums) > MAIN_NUM_COUNT else draw_nums[:BONUS_NUM_COUNT]
            bonus_numbers.append(bonus_nums)
            
        # Train position models
        for pos in range(len(self.position_models)):
            # Extract target for this position
            y_pos = np.array([draw[pos] for draw in bonus_numbers if len(draw) > pos])
            
            # Train model
            self.position_models[pos].fit(X_train, y_pos)
            
        # Train number models
        for num in range(1, self.num_range + 1):
            # Target: 1 if number appears in bonus numbers, 0 otherwise
            y_num = np.array([1 if num in draw else 0 for draw in bonus_numbers])
            
            # Train model
            self.number_models[num].fit(X_train, y_num)
            
        # Save the trained model
        self.save()
        
        return self.position_models, self.number_models
    
    def predict_probabilities(self, X):
        """Generate probability distribution for bonus numbers."""
        if not self.position_models or not self.number_models:
            raise ValueError("Models not trained")
            
        # Get position-based probabilities
        position_probs = np.zeros((len(X), self.num_range))
        
        for pos, model in enumerate(self.position_models):
            # Predict position value
            pred_val = model.predict(X)
            
            # Convert to probability distribution
            for i, val in enumerate(pred_val):
                # Round to nearest integer
                center = int(round(val)) - 1  # Convert to 0-indexed
                
                # Ensure center is within valid range
                center = max(0, min(center, self.num_range - 1))
                
                # Create Gaussian distribution around predicted value
                for j in range(self.num_range):
                    dist = min(abs(j - center), self.num_range - abs(j - center))
                    position_probs[i, j] += np.exp(-0.5 * (dist/2.0)**2)
                    
        # Normalize position probabilities
        row_sums = position_probs.sum(axis=1, keepdims=True)
        position_probs = position_probs / row_sums
        
        # Get number-based probabilities
        number_probs = np.zeros((len(X), self.num_range))
        
        for num, model in self.number_models.items():
            # Predict probability of this number appearing
            prob = model.predict(X)
            
            # Store in probability matrix (convert to 0-indexed)
            number_probs[:, num-1] = prob
            
        # Normalize number probabilities
        row_sums = number_probs.sum(axis=1, keepdims=True)
        number_probs = number_probs / row_sums
        
        # Combine both probability types with ensemble weighting
        combined_probs = 0.4 * position_probs + 0.6 * number_probs
        
        # Final normalization
        row_sums = combined_probs.sum(axis=1, keepdims=True)
        combined_probs = combined_probs / row_sums
        
        return combined_probs
    
    def predict(self, X, n_positions=BONUS_NUM_COUNT):
        """Generate concrete bonus number predictions."""
        probabilities = self.predict_probabilities(X)
        
        predictions = []
        for i in range(len(X)):
            # Sample n_positions without replacement
            selected_indices = []
            remaining_probs = probabilities[i].copy()
            
            for _ in range(n_positions):
                # Normalize remaining probabilities
                if np.sum(remaining_probs) > 0:
                    remaining_probs = remaining_probs / np.sum(remaining_probs)
                    
                # Sample one number
                selected_idx = np.random.choice(self.num_range, p=remaining_probs)
                selected_indices.append(selected_idx)
                
                # Set probability to zero to avoid resampling
                remaining_probs[selected_idx] = 0
            
            # Convert to actual numbers (1-indexed)
            selected_numbers = [idx + 1 for idx in selected_indices]
            predictions.append(sorted(selected_numbers))
            
        return predictions
    
    def save(self, path=None):
        """Save models to disk."""
        if not self.position_models or not self.number_models:
            raise ValueError("No models to save")
            
        save_path = path or self.checkpoint_path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        data_to_save = {
            'position_models': self.position_models,
            'number_models': self.number_models
        }
        
        with open(save_path, 'wb') as f:
            import pickle
            pickle.dump(data_to_save, f)
            
        logger.info(f"Bonus number models saved to {save_path}")
        
    def load(self, path=None):
        """Load models from disk."""
        load_path = path or self.checkpoint_path
        
        if os.path.exists(load_path):
            try:
                with open(load_path, 'rb') as f:
                    import pickle
                    data = pickle.load(f)
                    
                self.position_models = data['position_models']
                self.number_models = data['number_models']
                
                logger.info(f"Bonus number models loaded from {load_path}")
                return True
            except Exception as e:
                logger.error(f"Error loading bonus models: {e}")
                return False
        else:
            logger.warning(f"No saved models found at {load_path}")
            return False

#######################
# ENSEMBLE MODEL
#######################

class EnsembleCalibrator:
    """Probability calibration for ensemble predictions."""
    
    def __init__(self, temperature_range=(0.5, 1.5), calibration_factor=0.85):
        self.temperature_range = temperature_range
        self.calibration_factor = calibration_factor
        self.temperature_history = []
        
    def calibrate_ensemble(self, model_probs, confidence=None, iteration=0, diversity_factor=0.8):
        """Calibrate and combine probabilities from different models.
        
        Args:
            model_probs: Dictionary of model probabilities {'model': probabilities}
            confidence: Optional confidence score to adjust calibration
            iteration: Current prediction iteration (for diversity)
            diversity_factor: How much to penalize previously selected numbers
        """
        # Check if model_probs is empty to prevent StopIteration
        if not model_probs:
            logger.warning("Empty model probabilities received in calibrate_ensemble")
            # Return a default probability distribution
            return np.ones((1, 50)) / 50  # Adjust 50 to match your output dimension
            
        # Calculate dynamic temperature based on confidence and iteration
        base_temp = self.temperature_range[0] + 0.2 * iteration  # Increase temperature for later iterations
        
        if confidence is not None:
            # Higher confidence -> lower temperature (more deterministic)
            temp_adjustment = (1.0 - confidence) * (self.temperature_range[1] - self.temperature_range[0])
            temperature = base_temp + temp_adjustment
        else:
            temperature = base_temp
            
        # Record temperature for analysis
        self.temperature_history.append(temperature)
        
        # Apply temperature scaling to each model's probabilities
        scaled_probs = {}
        for model_name, probs in model_probs.items():
            # Different models get slightly different temperatures
            if model_name == 'transformer':
                # Transformer model gets slightly lower temperature
                model_temp = temperature * 0.9
            elif model_name == 'pattern':
                # Pattern model gets slightly higher temperature
                model_temp = temperature * 1.1
            else:
                model_temp = temperature
            
            # Ensure probs is properly shaped before scaling
            # If probs is a list of arrays, use the first array
            if isinstance(probs, list) and len(probs) > 0 and isinstance(probs[0], np.ndarray):
                probs_array = probs[0]
            else:
                probs_array = probs
                
            # Apply temperature scaling
            scaled_probs[model_name] = self._apply_temperature(probs_array, model_temp)
        
        # Check if scaled_probs is empty before accessing (fix for StopIteration)
        if not scaled_probs:
            logger.warning("No valid model probabilities for ensemble calibration.")
            # Return a default probability distribution
            return np.ones((1, 50)) / 50  # Adjust 50 to match your output dimension
        
        # Combine using learned weights that evolve with iterations
        # First draw - trust transformer model more
        if iteration == 0:
            weights = {
                'transformer': 0.4,
                'frequency': 0.35,
                'pattern': 0.25
            }
        # Middle draws - balanced weights
        elif iteration < 3:
            weights = {
                'transformer': 0.35,
                'frequency': 0.35,
                'pattern': 0.3
            }
        # Later draws - trust pattern model more for diversity
        else:
            weights = {
                'transformer': 0.25,
                'frequency': 0.35,
                'pattern': 0.4
            }
        
        # Combine probabilities with weights
        # Get a sample shape from one of the probability arrays
        sample_probs = next(iter(scaled_probs.values()))
        ensemble_probs = np.zeros_like(sample_probs)
        
        for model_name, probs in scaled_probs.items():
            if model_name in weights:
                # Ensure shapes match
                if probs.shape == ensemble_probs.shape:
                    ensemble_probs += probs * weights[model_name]
                else:
                    # Handle shape mismatch (might need reshaping)
                    try:
                        # Try basic reshaping if dimensions are compatible
                        reshaped_probs = np.reshape(probs, ensemble_probs.shape)
                        ensemble_probs += reshaped_probs * weights[model_name]
                    except:
                        # Log the issue and skip this model's contribution
                        print(f"Shape mismatch in {model_name} model: {probs.shape} vs {ensemble_probs.shape}")
        
        # Normalize
        row_sums = ensemble_probs.sum(axis=1, keepdims=True)
        if np.any(row_sums == 0):
            # Handle zero sums
            zero_rows = (row_sums == 0).flatten()
            ensemble_probs[zero_rows, :] = 1.0 / ensemble_probs.shape[1]
            row_sums = ensemble_probs.sum(axis=1, keepdims=True)
            
        ensemble_probs = ensemble_probs / row_sums
            
        return ensemble_probs
    
    def _apply_temperature(self, probabilities, temperature):
        """Apply temperature scaling to probabilities."""
        # Avoid division by zero or other numerical issues
        epsilon = 1e-10
        
        # Ensure probabilities is a numpy array
        if not isinstance(probabilities, np.ndarray):
            try:
                probabilities = np.array(probabilities)
            except:
                # If conversion fails, try to get the first element if it's a sequence
                if hasattr(probabilities, '__len__') and len(probabilities) > 0:
                    probabilities = np.array(probabilities[0])
                else:
                    # Last resort
                    return probabilities
        
        # Ensure we have at least 2D array [batch, probabilities]
        if len(probabilities.shape) == 1:
            probabilities = probabilities.reshape(1, -1)
        
        valid_probs = np.clip(probabilities, epsilon, 1.0 - epsilon)
        
        # Apply temperature scaling
        scaled_probs = np.power(valid_probs, 1/max(0.1, temperature))
        
        # Renormalize
        row_sums = scaled_probs.sum(axis=1, keepdims=True)
        # Avoid division by zero
        row_sums = np.maximum(row_sums, epsilon)
        scaled_probs = scaled_probs / row_sums
        
        return scaled_probs
    
    def adjust_for_diversity(self, probabilities, used_numbers, diversity_factor=0.8):
        """Adjust probabilities to encourage diversity in selections."""
        if not used_numbers:
            return probabilities
            
        # Ensure probabilities is a numpy array and properly shaped
        if not isinstance(probabilities, np.ndarray):
            try:
                probabilities = np.array(probabilities)
            except:
                # If conversion fails, try to get the first element if it's a sequence
                if hasattr(probabilities, '__len__') and len(probabilities) > 0:
                    probabilities = np.array(probabilities[0])
                else:
                    # Last resort: return original
                    return probabilities
        
        # Ensure we have at least 2D array [batch, probabilities]
        if len(probabilities.shape) == 1:
            probabilities = probabilities.reshape(1, -1)
        
        adjusted_probs = probabilities.copy()
        
        # Reduce probabilities for already used numbers
        for num in used_numbers:
            if 1 <= num <= adjusted_probs.shape[1]:
                adjusted_probs[:, num-1] *= diversity_factor
        
        # Renormalize
        row_sums = adjusted_probs.sum(axis=1, keepdims=True)
        # Avoid division by zero
        epsilon = 1e-10
        row_sums = np.maximum(row_sums, epsilon)
        adjusted_probs = adjusted_probs / row_sums
        
        return adjusted_probs
    
    def calculate_calibrated_confidence(self, probabilities, selected_indices, pattern_score):
        """Calculate calibrated confidence score based on probabilities and pattern."""
        # Ensure probabilities is properly shaped
        if not isinstance(probabilities, np.ndarray):
            try:
                probabilities = np.array(probabilities)
            except:
                # If conversion fails, try to get the first element if it's a sequence
                if hasattr(probabilities, '__len__') and len(probabilities) > 0:
                    probabilities = np.array(probabilities[0])
                else:
                    # Last resort: use a default confidence value
                    return 0.5
        
        # Ensure we have at least 2D array [batch, probabilities]
        if len(probabilities.shape) == 1:
            probabilities = probabilities.reshape(1, -1)
        
        # Extract probabilities of selected numbers
        selected_probs = []
        for idx in selected_indices:
            if 0 <= idx < probabilities.shape[1]:  # Ensure index is valid
                # Get the probability for this index (first batch)
                prob = probabilities[0, idx]
                if hasattr(prob, '__len__'):  # If it's a sequence, take first element
                    prob = prob[0] if len(prob) > 0 else 0.0
                selected_probs.append(float(prob))
        
        # Calculate mean probability
        mean_prob = np.mean(selected_probs) if selected_probs else 0.0
        
        # Scale mean probability to favor values over random chance
        # For 5/50 numbers, random chance is 0.1, so scale to stretch the range
        prob_factor = (mean_prob - 0.1) / 0.9 if mean_prob > 0.1 else 0.0
        scaled_prob = 0.5 + 0.5 * prob_factor  # Scale to 0.5-1.0 range for probs above random
        
        # Combine with pattern score using calibration factor
        combined_score = self.calibration_factor * scaled_prob + (1 - self.calibration_factor) * pattern_score
        
        # Apply final sigmoid calibration for better scaling
        confidence = 1.0 / (1.0 + np.exp(-5.0 * (combined_score - 0.5)))
        
        return min(0.99, confidence)  # Cap at 0.99 for realism

#######################
# HYBRID SYSTEM
#######################

class HybridNeuralSystem:
    """Unified hybrid lottery prediction system combining multiple models."""
    
    def __init__(self, file_path=None, data=None, params=None):
        """Initialize the hybrid lottery prediction system."""
        self.file_path = file_path
        self.raw_data = data
        self.params = params if params is not None else Utilities.get_default_params()
        
        # Create distribution strategy
        self.distribution_strategy = create_distribution_strategy()
        
        # Initialize processor
        self.processor = LotteryDataProcessor(file_path, data)
        self.processor.set_sequence_length(self.params.get('sequence_length', DEFAULT_SEQUENCE_LENGTH))
        
        # Initialize models
        self.transformer_model = None
        self.frequency_model = None
        self.pattern_model = None
        self.bonus_model = None
        
        # Data attributes
        self.features = None
        self.features_scaled = None
        self.main_sequences = None
        self.bonus_sequences = None
        self.y_main = None
        self.y_bonus = None
        self.y_main_raw = None
        self.y_bonus_raw = None
    
        # Calibrator for ensemble predictions
        self.calibrator = EnsembleCalibrator()

    def backtest(self, num_past_draws=20):
        """Backtest predictions against historical draws.
        
        Args:
            num_past_draws: Number of past draws to use for backtesting
            
        Returns:
            Dictionary with backtesting metrics
        """
        logger.info(f"Running backtesting against {num_past_draws} historical draws")
        
        # Ensure data is loaded
        if self.processor.data is None:
            self.prepare_data()
            
        # Get historical draws
        if self.processor.data is None or len(self.processor.data) < num_past_draws:
            logger.error(f"Not enough historical data for backtesting (need at least {num_past_draws} draws)")
            return {
                'success': False,
                'error': 'Insufficient historical data'
            }
            
        # Get the latest draws for testing
        test_draws = self.processor.data.tail(num_past_draws).copy()
        
        # Remove test draws from training data
        train_data = self.processor.data.iloc[:-num_past_draws].copy()
        
        # Create a new processor with just the training data
        train_processor = LotteryDataProcessor(data=train_data)
        
        # Create a temporary system with only training data
        temp_system = HybridNeuralSystem(data=train_data, params=self.params)
        temp_system.processor = train_processor
        
        # Prepare data, build and train models
        logger.info("Preparing data for backtesting...")
        temp_system.prepare_data()
        
        logger.info("Building models for backtesting...")
        temp_system.build_models()
        
        logger.info("Training models for backtesting...")
        temp_system.train_models()
        
        # Generate predictions for comparison with actual draws
        backtest_results = []
        total_score = 0.0
        
        logger.info("Generating backtesting predictions...")
        
        for idx, (_, row) in enumerate(test_draws.iterrows()):
            # Generate a prediction that would have been made before this draw
            prediction = temp_system.predict(num_draws=1)[0]
            
            # Format actual draw
            actual_draw = {
                "main_numbers": row["main_numbers"],
                "main_number_positions": {num: i for i, num in enumerate(row["main_numbers"])},
                "bonus_numbers": row["bonus_numbers"]
            }
            
            # Calculate match score
            match_score = Utilities.calculate_partial_match_score(prediction, actual_draw)
            total_score += match_score
            
            # Store detailed results
            backtest_results.append({
                "draw_index": idx + 1,
                "draw_date": row["date"].strftime("%Y-%m-%d") if hasattr(row["date"], "strftime") else str(row["date"]),
                "predicted_main": prediction["main_numbers"],
                "actual_main": row["main_numbers"],
                "predicted_bonus": prediction["bonus_numbers"],
                "actual_bonus": row["bonus_numbers"],
                "match_score": match_score,
                "confidence": prediction["confidence"]["overall"]
            })
            
            logger.info(f"Backtest draw {idx+1}: Match Score = {match_score:.4f}")
        
        # Calculate overall metrics
        avg_match_score = total_score / len(backtest_results) if backtest_results else 0.0
        
        # Count different match levels
        match_counts = {
            '0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0,  # main numbers
            '0+0': 0, '0+1': 0, '0+2': 0,  # 0 main + bonus
            '1+0': 0, '1+1': 0, '1+2': 0,  # 1 main + bonus
            '2+0': 0, '2+1': 0, '2+2': 0,  # 2 main + bonus
            '3+0': 0, '3+1': 0, '3+2': 0,  # 3 main + bonus
            '4+0': 0, '4+1': 0, '4+2': 0,  # 4 main + bonus
            '5+0': 0, '5+1': 0, '5+2': 0   # 5 main + bonus
        }
        
        # Count matches by category
        for result in backtest_results:
            main_matches = len(set(result["predicted_main"]).intersection(set(result["actual_main"])))
            bonus_matches = len(set(result["predicted_bonus"]).intersection(set(result["actual_bonus"])))
            
            # Update main count
            if str(main_matches) in match_counts:
                match_counts[str(main_matches)] += 1
                
            # Update combined count
            combined_key = f"{main_matches}+{bonus_matches}"
            if combined_key in match_counts:
                match_counts[combined_key] += 1
        
        # Calculate expected prize winnings based on match categories
        # This is a very simplified model and actual prizes vary significantly
        prize_expectations = {
            '2+0': 0.0,      # No prize typically
            '2+1': 4.30,     # Estimated average prize
            '2+2': 7.50,
            '3+0': 6.00,
            '3+1': 10.70,
            '3+2': 43.30,
            '4+0': 25.60,
            '4+1': 123.00,
            '4+2': 844.70,
            '5+0': 33466.40,
            '5+1': 256213.60,
            '5+2': 17000000.00  # Jackpot - very rough estimate
        }
        
        # Calculate expected return
        expected_return = 0.0
        for category, count in match_counts.items():
            if category in prize_expectations:
                expected_return += count * prize_expectations[category]
        
        # Assuming each prediction costs 2.50 EUR
        prediction_cost = len(backtest_results) * 2.50
        net_return = expected_return - prediction_cost
        roi = (net_return / prediction_cost) if prediction_cost > 0 else 0.0
        
        logger.info(f"Backtesting complete: Avg Match Score = {avg_match_score:.4f}")
        logger.info(f"Expected Return: €{expected_return:.2f}, ROI: {roi*100:.2f}%")
        
        return {
            'success': True,
            'avg_match_score': avg_match_score,
            'detailed_results': backtest_results,
            'match_counts': match_counts,
            'expected_return': expected_return,
            'cost': prediction_cost,
            'net_return': net_return,
            'roi': roi
        }
        
    @ErrorHandler.handle_exception(logger, "data preparation")
    def prepare_data(self, force_reprocess=False):
        """Prepare data for model training and prediction."""
        logger.info("Preparing data for model training")
        
        # Load and process data
        df = self.processor.parse_file()
        if df.empty:
            raise ValueError("Failed to load lottery data")
                
        # Create expanded data
        expanded_df = self.processor.expand_numbers()
        
        # Generate features
        self.features = self.processor.create_features()
        
        # Create sequences
        self.main_sequences, self.bonus_sequences, self.y_main, self.y_bonus = self.processor.create_sequences()
        
        # Get actual number of samples in sequence data
        if self.main_sequences is not None:
            seq_length = self.processor.sequence_length
            num_seq_samples = self.main_sequences.shape[0]
            
            # Important: Trim features to exactly match sequence data
            # The sequences start from seq_length onwards, so features should too
            feature_start_idx = seq_length
            feature_end_idx = feature_start_idx + num_seq_samples
            logger.info(f"Adjusting features from index {feature_start_idx} to {feature_end_idx} "
                      f"to match sequence data with {num_seq_samples} samples")
            
            if isinstance(self.features, pd.DataFrame) and feature_end_idx <= len(self.features):
                self.features = self.features.iloc[feature_start_idx:feature_end_idx].reset_index(drop=True)
            else:
                logger.warning(f"Could not properly subset features. Features shape: {self.features.shape}, "
                            f"Desired range: {feature_start_idx}:{feature_end_idx}")
            
            # Create raw targets for models that need them
            self.y_main_raw = expanded_df.iloc[seq_length:seq_length+num_seq_samples]['main_numbers'].values
            self.y_bonus_raw = expanded_df.iloc[seq_length:seq_length+num_seq_samples]['bonus_numbers'].values
            
            # Scale features
            scaler = StandardScaler()
            self.features_scaled = scaler.fit_transform(self.features.select_dtypes(include=[np.number]))
            
            # Convert back to DataFrame for easier access
            self.features_scaled = pd.DataFrame(
                self.features_scaled,
                columns=self.features.select_dtypes(include=[np.number]).columns,
                index=self.features.index
            )
            
            # Double check dimensions
            logger.info(f"Final data dimensions:")
            logger.info(f"  Features: {self.features_scaled.shape}")
            logger.info(f"  Main sequences: {self.main_sequences.shape}")
            logger.info(f"  Y main: {self.y_main.shape}")
            logger.info(f"  Y main raw: {len(self.y_main_raw)}")
            
            if self.features_scaled.shape[0] != self.main_sequences.shape[0]:
                logger.warning("Data dimension mismatch after preparation! This may cause training errors.")
                
            logger.info(f"Data preparation complete: {len(self.features_scaled)} samples")
        else:
            # Handle case where sequence data could not be created
            logger.warning("Sequence data not available. Some models may not function correctly.")
            
            # Scale features anyway
            scaler = StandardScaler()
            self.features_scaled = scaler.fit_transform(self.features.select_dtypes(include=[np.number]))
            
            # Set raw targets to None
            self.y_main_raw = None
            self.y_bonus_raw = None
            
            logger.info(f"Data preparation complete: {len(self.features_scaled)} samples (no sequences)")
        
        return {
            "features": self.features,
            "features_scaled": self.features_scaled,
            "main_sequences": self.main_sequences,
            "bonus_sequences": self.bonus_sequences,
            "y_main": self.y_main,
            "y_bonus": self.y_bonus,
            "y_main_raw": self.y_main_raw,
            "y_bonus_raw": self.y_bonus_raw,
            "data": self.processor.data
        }
    
    @ErrorHandler.handle_exception(logger, "model building")
    def build_models(self):
        """Build all prediction models."""
        logger.info("Building prediction models")
        
        # Ensure data is prepared
        if self.features_scaled is None:
            self.prepare_data()
        
        # Make sure we have sequence data
        if self.main_sequences is None:
            raise ValueError("No sequence data available. Cannot build models.")
        
        # Build transformer model
        if self.params.get('use_transformer_model', True):
            logger.info("Building transformer model")
            # Get the actual sequence dimensions from the data
            actual_seq_length = self.main_sequences.shape[1]
            actual_seq_dim = self.main_sequences.shape[2]
            
            # Update the sequence_length in params to match the actual data
            self.params['sequence_length'] = actual_seq_length
            
            # Build sequence model with correct dimensions and distribution strategy
            input_shape = (self.features_scaled.shape[1], actual_seq_dim)
            self.transformer_model = TransformerModel(
                input_shape=input_shape,
                output_dim=MAIN_NUM_MAX,
                params=self.params,
                distribution_strategy=self.distribution_strategy
            )
            self.transformer_model.build_model()
        
        # Build frequency model
        if self.params.get('use_frequency_model', True):
            logger.info("Building frequency model")
            self.frequency_model = FrequencyModel(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=5
            )
            # Try to load existing model first if available
            if hasattr(self, 'load_existing') and self.load_existing:
                if not self.frequency_model.load():
                    logger.info("Could not load frequency model, building new one")
                    self.frequency_model.build_model(n_positions=MAIN_NUM_COUNT)
            else:
                self.frequency_model.build_model(n_positions=MAIN_NUM_COUNT)
        
        # Build pattern model
        if self.params.get('use_pattern_model', True):
            logger.info("Building pattern model")
            self.pattern_model = PatternModel(
                n_estimators=150,
                num_range=MAIN_NUM_MAX
            )
            # Try to load existing model first if available
            if hasattr(self, 'load_existing') and self.load_existing:
                if not self.pattern_model.load():
                    logger.info("Could not load pattern model, building new one")
                    self.pattern_model.build_model()
            else:
                self.pattern_model.build_model()
        
        # Build bonus model
        logger.info("Building bonus model")
        self.bonus_model = BonusNumberModel(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=4,
            num_range=BONUS_NUM_MAX
        )
        # Try to load existing model first if available
        if hasattr(self, 'load_existing') and self.load_existing:
            if not self.bonus_model.load():
                logger.info("Could not load bonus model, building new one")
                self.bonus_model.build_model(n_positions=BONUS_NUM_COUNT)
        else:
            self.bonus_model.build_model(n_positions=BONUS_NUM_COUNT)
        
        logger.info("All models built successfully")
        
        return {
            'transformer_model': self.transformer_model,
            'frequency_model': self.frequency_model,
            'pattern_model': self.pattern_model,
            'bonus_model': self.bonus_model
        }
    
    @ErrorHandler.handle_exception(logger, "model training")
    def train_models(self, validation_split=0.2):
        """Train all prediction models with improved error handling."""
        logger.info("Training prediction models")
        
        # Ensure models are built
        if self.transformer_model is None:
            self.build_models()
        
        # Ensure data is prepared
        if self.features_scaled is None:
            self.prepare_data()
        
        # Prepare arrays
        X_features = self.features_scaled.values
        
        # Train models with individual try-except blocks for each
        transformer_model_trained = False
        
        # Try to train transformer model, but don't fail the whole process if it fails
        if self.params.get('use_transformer_model', True) and self.transformer_model is not None:
            try:
                logger.info("Training transformer model")
                self.transformer_model.train(
                    X_train=(X_features, self.main_sequences),
                    y_train=self.y_main,
                    validation_split=validation_split
                )
                transformer_model_trained = True
                logger.info("Transformer model training completed successfully")
            except Exception as e:
                logger.error(f"Error training transformer model: {e}")
                logger.error(traceback.format_exc())
                logger.warning("Will continue with other models")
        
        # Clean memory
        Utilities.clean_memory()
        
        # Train frequency model
        if self.params.get('use_frequency_model', True) and self.frequency_model is not None:
            try:
                logger.info("Training frequency model")
                self.frequency_model.train(
                    X_train=X_features,
                    y_train_raw=self.y_main_raw
                )
                logger.info("Frequency model training completed successfully")
            except Exception as e:
                logger.error(f"Error training frequency model: {e}")
                logger.error(traceback.format_exc())
        
        # Clean memory
        Utilities.clean_memory()
        
        # Train pattern model
        if self.params.get('use_pattern_model', True) and self.pattern_model is not None:
            try:
                logger.info("Training pattern model")
                self.pattern_model.train(
                    X_train=X_features,
                    y_train=self.y_main_raw
                )
                logger.info("Pattern model training completed successfully")
            except Exception as e:
                logger.error(f"Error training pattern model: {e}")
                logger.error(traceback.format_exc())
        
        # Clean memory
        Utilities.clean_memory()
        
        # Train bonus model
        try:
            logger.info("Training bonus model")
            self.bonus_model.train(
                X_train=X_features,
                y_train_raw=self.y_main_raw  # Use main numbers as context
            )
            logger.info("Bonus model training completed successfully")
        except Exception as e:
            logger.error(f"Error training bonus model: {e}")
            logger.error(traceback.format_exc())
        
        # Clean memory
        Utilities.clean_memory(force=True)
        
        return True
    
    @ErrorHandler.handle_exception(logger, "prediction", lambda self, num_draws, **kwargs: self.generate_fallback_predictions(num_draws))
    def predict(self, num_draws=5, temperature=DEFAULT_TEMPERATURE, diversity_sampling=True):
        """Generate predictions using ensemble of models."""
        logger.info(f"Generating {num_draws} lottery predictions")
        
        # Ensure at least one model is trained
        if (self.transformer_model is None and self.frequency_model is None and 
            self.pattern_model is None and self.bonus_model is None):
            raise ValueError("No models are trained. Call train_models() first.")
        
        # Verify frequency model is properly fitted
        if self.frequency_model is not None:
            try:
                # Check if models are empty or not fitted
                if not self.frequency_model.models or not hasattr(self.frequency_model.models[0], 'feature_importances_'):
                    logger.warning("Frequency model not properly trained or loaded. Attempting to train now...")
                    # Check if we have the data needed to train
                    if self.features_scaled is not None and self.y_main_raw is not None:
                        X_features = self.features_scaled.values if hasattr(self.features_scaled, 'values') else self.features_scaled
                        self.frequency_model.train(X_train=X_features, y_train_raw=self.y_main_raw)
                        logger.info("Frequency model trained successfully")
                    else:
                        logger.error("Cannot train frequency model: training data not available")
                        logger.info("Disabling frequency model for predictions")
                        self.frequency_model = None
            except Exception as e:
                logger.error(f"Error verifying/training frequency model: {e}")
                logger.error(traceback.format_exc())
                logger.info("Disabling frequency model for predictions")
                self.frequency_model = None
        
        # Similarly verify pattern model
        if self.pattern_model is not None:
            try:
                # Check if models are empty or not fitted
                test_model = next(iter(self.pattern_model.models.values())) if self.pattern_model.models else None
                if not self.pattern_model.models or not hasattr(test_model, 'feature_importances_'):
                    logger.warning("Pattern model not properly trained or loaded. Attempting to train now...")
                    # Check if we have the data needed to train
                    if self.features_scaled is not None and self.y_main_raw is not None:
                        X_features = self.features_scaled.values if hasattr(self.features_scaled, 'values') else self.features_scaled
                        self.pattern_model.train(X_train=X_features, y_train=self.y_main_raw)
                        logger.info("Pattern model trained successfully")
                    else:
                        logger.error("Cannot train pattern model: training data not available")
                        logger.info("Disabling pattern model for predictions")
                        self.pattern_model = None
            except Exception as e:
                logger.error(f"Error verifying/training pattern model: {e}")
                logger.error(traceback.format_exc())
                logger.info("Disabling pattern model for predictions")
                self.pattern_model = None
        
        # Get latest data point for prediction
        latest_features = self.features_scaled.values[-1:] if hasattr(self.features_scaled, 'values') else self.features_scaled[-1:]
        latest_main_seq = self.main_sequences[-1:] if self.main_sequences is not None else None
        latest_bonus_seq = self.bonus_sequences[-1:] if self.bonus_sequences is not None else None
        
        # Handle case where sequences are None (not enough data)
        if latest_main_seq is None or latest_bonus_seq is None:
            logger.warning("Not enough sequence data. Using fallback predictions.")
            return self.generate_fallback_predictions(num_draws)
        
        # Track used numbers for diversity
        used_main_numbers = set()
        used_bonus_numbers = set()
        
        # Generate predictions
        predictions = []
        
        # Get ensemble weights from parameters
        ensemble_weights = self.params.get('ensemble_weights', {
            'transformer': 0.4,
            'frequency': 0.3,
            'pattern': 0.3
        })
        
        # Adjust weights based on which models are available
        if self.transformer_model is None:
            ensemble_weights['transformer'] = 0
        if self.frequency_model is None:
            ensemble_weights['frequency'] = 0
        if self.pattern_model is None:
            ensemble_weights['pattern'] = 0
            
        # Renormalize weights if needed
        weight_sum = sum(ensemble_weights.values())
        if weight_sum > 0:
            ensemble_weights = {k: v / weight_sum for k, v in ensemble_weights.items()}
        
        for draw_idx in range(num_draws):
            try:
                model_probs = {}
                
                # Get transformer model predictions
                if self.transformer_model is not None:
                    transformer_probs = self.transformer_model.predict((latest_features, latest_main_seq))
                    model_probs['transformer'] = transformer_probs
                
                # Get frequency model predictions
                if self.frequency_model is not None:
                    freq_probs = self.frequency_model.predict_probabilities(latest_features)
                    model_probs['frequency'] = freq_probs.reshape(1, -1)
                
                # Get pattern model predictions
                if self.pattern_model is not None:
                    pattern_probs = self.pattern_model.predict_probabilities(latest_features)
                    model_probs['pattern'] = pattern_probs
                
                # Calculate overall confidence for temperature adjustment
                if draw_idx > 0 and predictions:
                    avg_confidence = np.mean([p["confidence"]["overall"] for p in predictions])
                else:
                    avg_confidence = 0.5  # Default for first draw
                
                # Apply calibration and ensemble weighting
                ensemble_probs = self.calibrator.calibrate_ensemble(
                    model_probs,
                    confidence=avg_confidence,
                    iteration=draw_idx,
                    diversity_factor=DIVERSITY_FACTOR
                )
                
                # Adjust for diversity
                if draw_idx > 0 and diversity_sampling and used_main_numbers:
                    ensemble_probs = self.calibrator.adjust_for_diversity(
                        ensemble_probs,
                        used_main_numbers,
                        diversity_factor=DIVERSITY_FACTOR
                    )
                
                # Sample main numbers
                main_numbers = Utilities.sample_numbers(
                    probs=[ensemble_probs],
                    available_nums=range(MAIN_NUM_MIN, MAIN_NUM_MAX+1),
                    num_to_select=MAIN_NUM_COUNT,
                    used_nums=used_main_numbers,
                    diversity_sampling=diversity_sampling,
                    draw_idx=draw_idx,
                    temperature=temperature
                )
                
                # Get bonus number predictions
                bonus_probs = self.bonus_model.predict_probabilities(latest_features)
                
                # Adjust for diversity
                if draw_idx > 0 and diversity_sampling and used_bonus_numbers:
                    bonus_probs = self.calibrator.adjust_for_diversity(
                        bonus_probs,
                        used_bonus_numbers,
                        diversity_factor=DIVERSITY_FACTOR
                    )
                
                # Sample bonus numbers
                bonus_numbers = Utilities.sample_numbers(
                    probs=[bonus_probs],
                    available_nums=range(BONUS_NUM_MIN, BONUS_NUM_MAX+1),
                    num_to_select=BONUS_NUM_COUNT,
                    used_nums=used_bonus_numbers,
                    diversity_sampling=diversity_sampling,
                    draw_idx=draw_idx,
                    temperature=temperature
                )
                
                # Store position information for analysis
                main_positions = {num: idx for idx, num in enumerate(main_numbers)}
                
                # Calculate pattern score
                pattern_score = Utilities.calculate_pattern_score(main_numbers, bonus_numbers)
                
                # Calculate frequency score if historical data is available
                frequency_score = 0.5  # Default
                if self.processor.data is not None and not self.processor.data.empty:
                    frequency_score = Utilities.calculate_frequency_score(
                        main_numbers, bonus_numbers, self.processor.data
                    )
                
                # Calculate calibrated confidence
                # Extract indices for selected numbers (0-based)
                selected_indices = [num - 1 for num in main_numbers]
                confidence = self.calibrator.calculate_calibrated_confidence(
                    ensemble_probs,
                    selected_indices,
                    pattern_score
                )
                
                # Add prediction
                predictions.append({
                    "main_numbers": main_numbers,
                    "main_number_positions": main_positions,
                    "bonus_numbers": bonus_numbers,
                    "confidence": {
                        "overall": float(confidence),
                        "main_numbers": float(min(0.95, confidence + 0.05 * (1.0 - pattern_score))),
                        "bonus_numbers": float(min(0.90, 0.75 * confidence)),
                        "pattern_score": float(pattern_score),
                        "frequency_score": float(frequency_score)
                    },
                    "method": "hybrid_ensemble",
                    "temperature": float(self.calibrator.temperature_history[-1] if self.calibrator.temperature_history else temperature)
                })
                
                # Add used numbers to sets
                used_main_numbers.update(main_numbers)
                used_bonus_numbers.update(bonus_numbers)
                
                # Clean memory occasionally
                if draw_idx > 0 and draw_idx % CLEAN_MEMORY_FREQUENCY == 0:
                    Utilities.clean_memory()
                    
            except Exception as e:
                logger.error(f"Error generating prediction {draw_idx+1}: {e}")
                logger.error(traceback.format_exc())
                
                # Generate a fallback prediction for this draw
                fallback_pred = self._generate_single_fallback_prediction(draw_idx)
                if fallback_pred:
                    predictions.append(fallback_pred)
        
        logger.info(f"Generated {len(predictions)} predictions successfully")
        return predictions

    def _generate_single_fallback_prediction(self, draw_idx=0):
        """Generate a single fallback prediction."""
        try:
            # Try to generate a reasonable prediction based on historical data
            if self.processor.data is not None and not self.processor.data.empty:
                # Calculate historical frequencies
                main_counts = np.zeros(MAIN_NUM_MAX)
                bonus_counts = np.zeros(BONUS_NUM_MAX)
                
                for _, row in self.processor.data.iterrows():
                    for num in row["main_numbers"]:
                        if MAIN_NUM_MIN <= num <= MAIN_NUM_MAX:
                            main_counts[num-1] += 1
                    for num in row["bonus_numbers"]:
                        if BONUS_NUM_MIN <= num <= BONUS_NUM_MAX:
                            bonus_counts[num-1] += 1
                
                # Add recency bias - more recent draws get higher weight
                recent_draws = min(30, len(self.processor.data))
                for i in range(recent_draws):
                    idx = len(self.processor.data) - i - 1
                    weight = 2.0 * (recent_draws - i) / recent_draws
                    
                    row = self.processor.data.iloc[idx]
                    for num in row["main_numbers"]:
                        if MAIN_NUM_MIN <= num <= MAIN_NUM_MAX:
                            main_counts[num-1] += weight
                    for num in row["bonus_numbers"]:
                        if BONUS_NUM_MIN <= num <= BONUS_NUM_MAX:
                            bonus_counts[num-1] += weight
                
                # Normalize to probabilities
                main_probs = main_counts / np.sum(main_counts) if np.sum(main_counts) > 0 else np.ones(MAIN_NUM_MAX) / MAIN_NUM_MAX
                bonus_probs = bonus_counts / np.sum(bonus_counts) if np.sum(bonus_counts) > 0 else np.ones(BONUS_NUM_MAX) / BONUS_NUM_MAX
                
                # Add small random noise for diversity
                main_probs += np.random.random(MAIN_NUM_MAX) * 0.2 * np.mean(main_probs)
                bonus_probs += np.random.random(BONUS_NUM_MAX) * 0.2 * np.mean(bonus_probs)
                
                # Renormalize
                main_probs = main_probs / np.sum(main_probs)
                bonus_probs = bonus_probs / np.sum(bonus_probs)
                
                # Sample numbers using the utility function
                main_numbers = Utilities.sample_numbers(
                    probs=[main_probs.reshape(1, -1)],
                    available_nums=range(MAIN_NUM_MIN, MAIN_NUM_MAX+1),
                    num_to_select=MAIN_NUM_COUNT,
                    temperature=DEFAULT_TEMPERATURE + (draw_idx * 0.1)  # Increase temperature for diversity
                )
                
                bonus_numbers = Utilities.sample_numbers(
                    probs=[bonus_probs.reshape(1, -1)],
                    available_nums=range(BONUS_NUM_MIN, BONUS_NUM_MAX+1),
                    num_to_select=BONUS_NUM_COUNT,
                    temperature=DEFAULT_TEMPERATURE + (draw_idx * 0.1)
                )
                
                # Calculate pattern score
                pattern_score = Utilities.calculate_pattern_score(main_numbers, bonus_numbers)
                
                # Calculate frequency score
                frequency_score = Utilities.calculate_frequency_score(
                    main_numbers, bonus_numbers, self.processor.data
                )
                
                # Store position information
                main_positions = {num: idx for idx, num in enumerate(main_numbers)}
                
                # Create prediction with reasonable confidence
                return {
                    "main_numbers": main_numbers,
                    "main_number_positions": main_positions,
                    "bonus_numbers": bonus_numbers,
                    "confidence": {
                        "overall": 0.4,
                        "main_numbers": 0.45,
                        "bonus_numbers": 0.35,
                        "pattern_score": float(pattern_score),
                        "frequency_score": float(frequency_score)
                    },
                    "method": "fallback_frequency",
                    "temperature": DEFAULT_TEMPERATURE + (draw_idx * 0.1)
                }
            else:
                # Pure random if no historical data
                main_numbers = sorted(random.sample(range(MAIN_NUM_MIN, MAIN_NUM_MAX+1), MAIN_NUM_COUNT))
                bonus_numbers = sorted(random.sample(range(BONUS_NUM_MIN, BONUS_NUM_MAX+1), BONUS_NUM_COUNT))
                
                # Calculate pattern score
                pattern_score = Utilities.calculate_pattern_score(main_numbers, bonus_numbers)
                
                # Store position information
                main_positions = {num: idx for idx, num in enumerate(main_numbers)}
                
                return {
                    "main_numbers": main_numbers,
                    "main_number_positions": main_positions, 
                    "bonus_numbers": bonus_numbers,
                    "confidence": {
                        "overall": 0.2,
                        "main_numbers": 0.2,
                        "bonus_numbers": 0.2,
                        "pattern_score": float(pattern_score),
                        "frequency_score": 0.2
                    },
                    "method": "fallback_random",
                    "temperature": 1.0
                }
        except Exception as e:
            logger.error(f"Error in fallback prediction: {e}")
            return None
    
    def generate_fallback_predictions(self, num_draws=5):
        """Generate fallback predictions when models are not available."""
        logger.warning("Using fallback prediction method")
        
        predictions = []
        for i in range(num_draws):
            pred = self._generate_single_fallback_prediction(i)
            if pred:
                predictions.append(pred)
                
        return predictions

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
        self.data_cache = None  # Cache for data reuse
    
    @ErrorHandler.handle_exception(logger, "optimization trial")
    def objective(self, trial):
        """Objective function for hyperparameter optimization."""
        # Define expanded parameter ranges with model_type first
        params = {
            # Define model_type early to avoid conflicts with dynamic parameters
            'model_type': trial.suggest_categorical('model_type', ['transformer', 'lstm', 'rnn', 'hybrid']),
            
            # Learning rate with finer granularity
            'learning_rate': trial.suggest_float('learning_rate', 0.0001, 0.003, log=True),
            
            # More batch size options
            'batch_size': trial.suggest_categorical('batch_size', [16, 24, 32, 48, 64, 96]),
            
            # More granular dropout options
            'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5, step=0.05),
            
            # Expanded attention head options
            'num_heads': trial.suggest_categorical('num_heads', [2, 3, 4, 6, 8, 12]),
            
            # More feed-forward dimension options
            'ff_dim': trial.suggest_categorical('ff_dim', [64, 96, 128, 160, 192, 224, 256, 320]),
            
            # More embedding dimension options
            'embed_dim': trial.suggest_categorical('embed_dim', [32, 48, 64, 80, 96, 112, 128, 144]),
            
            # GRU usage
            'use_gru': trial.suggest_categorical('use_gru', [True, False]),
            
            # More granular conv filter options
            'conv_filters': trial.suggest_int('conv_filters', 8, 64, step=8),
            
            # More transformer block options
            'num_transformer_blocks': trial.suggest_int('num_transformer_blocks', 1, 4),
            
            # Additional optimizer options
            'optimizer': trial.suggest_categorical('optimizer', ['adam', 'rmsprop', 'sgd', 'adamw']),
            
            # More sequence length options
            'sequence_length': trial.suggest_categorical('sequence_length', [12, 16, 20, 24, 28, 32]),
            
            # More regularization options
            'l2_regularization': trial.suggest_float('l2_regularization', 1e-6, 1e-3, log=True),
            
            # Architecture options
            'use_residual': trial.suggest_categorical('use_residual', [True, False]),
            'use_layer_scaling': trial.suggest_categorical('use_layer_scaling', [True, False]),
            
            # Early stopping control
            'min_delta': trial.suggest_float('min_delta', 0.0001, 0.005, log=True),
            
            # Model component usage
            'use_frequency_model': trial.suggest_categorical('use_frequency_model', [True, False]),
            'use_pattern_model': trial.suggest_categorical('use_pattern_model', [True, False]),
            'use_transformer_model': trial.suggest_categorical('use_transformer_model', [True, True, False]),  # 2/3 chance of using transformer
        }
        
        # Add model-specific parameters
        if model_type == 'lstm':
            params['lstm_units'] = lstm_units
        elif model_type == 'hybrid':
            params['lstm_units_hybrid'] = lstm_units
            params['gru_units'] = gru_units
        
        try:
            # Initialize hybrid system with these parameters
            system = HybridNeuralSystem(self.file_path, params=params)
            
            # Reuse data cache if available
            if self.data_cache is not None:
                # Apply cached data
                system.features = self.data_cache['features']
                system.features_scaled = self.data_cache['features_scaled']
                system.processor.data = self.data_cache['data']
                
                # Only reuse sequence data if the sequence length matches
                if self.data_cache.get('sequence_length') == params['sequence_length']:
                    system.main_sequences = self.data_cache['main_sequences']
                    system.bonus_sequences = self.data_cache['bonus_sequences']
                    system.y_main = self.data_cache['y_main']
                    system.y_bonus = self.data_cache['y_bonus']
                    system.y_main_raw = self.data_cache['y_main_raw']
                    system.y_bonus_raw = self.data_cache['y_bonus_raw']
                else:
                    # Set the correct sequence length and regenerate sequences
                    system.processor.set_sequence_length(params['sequence_length'])
                    data_dict = system.prepare_data()
                    
                    # Update cache with new sequence data
                    self.data_cache['sequence_length'] = params['sequence_length']
                    self.data_cache.update(data_dict)
            else:
                # Prepare data and cache it
                data_dict = system.prepare_data()
                data_dict['sequence_length'] = params['sequence_length']  # Store sequence_length in cache
                self.data_cache = data_dict
            
            # Build models
            system.build_models()
            
            # Calculate metrics through cross-validation
            evaluator = CrossValidationEvaluator(system, folds=3)
            scores = evaluator.evaluate()
            
            # Use a balanced objective considering both accuracy and pattern score
            trial_score = scores['avg_overall_acc'] * 0.6 + scores['avg_pattern_score'] * 0.4
            
            # Store additional metrics in the trial
            trial.set_user_attr('accuracy', float(scores['avg_overall_acc']))
            trial.set_user_attr('pattern_score', float(scores['avg_pattern_score']))
            
            # Clean memory
            Utilities.clean_memory()
            
            return trial_score
            
        except Exception as e:
            logger.error(f"Error in trial: {e}")
            # Return a very low score for failed trials
            return -1.0
    
    @ErrorHandler.handle_exception(logger, "hyperparameter optimization", Utilities.get_default_params())
    def optimize(self):
        """Run Bayesian optimization to find best hyperparameters."""
        logger.info(f"Starting hyperparameter optimization with {self.n_trials} trials")
        
        # Create Optuna study with improved configuration
        self.study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=RANDOM_SEED, n_startup_trials=25),  # Increased from 10
            pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=20)  # Relaxed parameters
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
        
        # Set reasonable ensemble weights based on which models are enabled
        ensemble_weights = {}
        models_enabled = 0
        
        if self.best_params.get('use_transformer_model', True):
            ensemble_weights['transformer'] = 0.4
            models_enabled += 1
        
        if self.best_params.get('use_frequency_model', True):
            ensemble_weights['frequency'] = 0.3
            models_enabled += 1
        
        if self.best_params.get('use_pattern_model', True):
            ensemble_weights['pattern'] = 0.3
            models_enabled += 1
            
        # Adjust weights if not all models are enabled
        if models_enabled < 3:
            weight_sum = sum(ensemble_weights.values())
            if weight_sum > 0:
                ensemble_weights = {k: v / weight_sum for k, v in ensemble_weights.items()}
                
        self.best_params['ensemble_weights'] = ensemble_weights
            
        # Save best parameters to file
        Utilities.save_params(self.best_params, "hybrid_best_params.json")
        
        return self.best_params

    @ErrorHandler.handle_exception(logger, "optimization plotting", None)
    def plot_optimization_history(self, filename="optimization_history.png"):
        """Plot optimization history with enhanced visualization."""
        if self.study is None:
            logger.error("No optimization study. Call optimize() first.")
            return None
        
        # Create figure with subplots
        plt.figure(figsize=(16, 12))
        
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
        
        # Calculate parameter importances if possible
        try:
            importances = optuna.importance.get_param_importances(self.study)
            param_names = list(importances.keys())
            importance_values = list(importances.values())
            
            # Plot top 10 parameters
            if param_names:
                plt.barh(param_names[-10:], importance_values[-10:])
                plt.xlabel('Relative Importance')
                plt.title('Parameter Importances')
                plt.grid(alpha=0.3)
        except Exception as e:
            plt.text(0.5, 0.5, f"Could not calculate parameter importances:\n{str(e)}",
                    ha='center', va='center', fontsize=12)
        
        # 3. Plot validation accuracy
        plt.subplot(2, 2, 3)
        
        # Extract metrics
        accuracies = [t.user_attrs.get('accuracy', float('nan')) for t in self.study.trials 
                     if t.state == optuna.trial.TrialState.COMPLETE]
        pattern_scores = [t.user_attrs.get('pattern_score', float('nan')) for t in self.study.trials 
                         if t.state == optuna.trial.TrialState.COMPLETE]
        
        # Plot metrics over trials
        plt.plot(trial_numbers, accuracies, 'b-', alpha=0.7, label='Accuracy')
        plt.plot(trial_numbers, pattern_scores, 'g-', alpha=0.7, label='Pattern Score')
        plt.xlabel('Trial Number')
        plt.ylabel('Metric Value')
        plt.title('Validation Metrics')
        plt.legend()
        plt.grid(alpha=0.3)
        
        # 4. Plot parallel coordinate plot for top trials
        plt.subplot(2, 2, 4)
        
        try:
            # Get most important parameters
            important_params = list(importances.keys())[:5]
            optuna.visualization.matplotlib.plot_parallel_coordinate(
                self.study, 
                params=important_params
            )
            plt.title('Parallel Coordinate Plot')
        except Exception as e:
            plt.text(0.5, 0.5, f"Could not create parallel coordinate plot:\n{str(e)}",
                    ha='center', va='center', fontsize=12)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()
        
        logger.info(f"Optimization plots saved to {filename}")
        return filename

class CrossValidationEvaluator:
    """Evaluate model performance using time-series cross-validation."""
    
    def __init__(self, system, folds=5):
        """Initialize with a hybrid system instance."""
        self.system = system
        self.folds = folds
    
    @ErrorHandler.handle_exception(logger, "cross-validation", {
    'fold_metrics': [], 'avg_overall_acc': 0.0, 
    'avg_pattern_score': 0.0, 'avg_score': 0.0
    })
    def evaluate(self):
        """Perform time-series cross-validation with enhanced scoring."""
        logger.info(f"Performing {self.folds}-fold time-series cross-validation")
        
        # Ensure data is prepared
        if self.system.features_scaled is None:
            self.system.prepare_data()
        
        # Check if we have enough data
        total_samples = len(self.system.features_scaled)
        if total_samples < self.folds * 2:
            logger.warning(f"Not enough data ({total_samples} samples) for {self.folds} folds")
            # Adjust folds to a reasonable number
            self.folds = max(2, total_samples // 3)
            logger.info(f"Adjusted to {self.folds} folds")
            
        fold_size = total_samples // self.folds
        
        # Initialize metrics
        fold_metrics = []
        
        # Prepare arrays for all data
        X_features = self.system.features_scaled.values if hasattr(self.system.features_scaled, 'values') else self.system.features_scaled
        main_sequences = self.system.main_sequences
        bonus_sequences = self.system.bonus_sequences
        y_main = self.system.y_main
        y_main_raw = self.system.y_main_raw
        
        # Use time-series cross-validation (each fold uses only past data)
        for fold in range(self.folds):
            try:
                # Calculate test split for this fold
                test_start = fold * fold_size
                test_end = min((fold + 1) * fold_size, total_samples)
                
                # For first fold, we can't have a training set (need at least 1 sample)
                if test_start == 0:
                    test_start = 1
                
                # Calculate training indices (all data before test split)
                train_indices = list(range(0, test_start))
                test_indices = list(range(test_start, test_end))
                
                # Skip if we don't have enough data
                if len(train_indices) < 5 or len(test_indices) < 2:
                    logger.info(f"Skipping fold {fold+1} due to insufficient data")
                    continue
                
                # Extract data for this fold
                X_train = X_features[train_indices]
                X_test = X_features[test_indices]
                
                main_seq_train = main_sequences[train_indices]
                main_seq_test = main_sequences[test_indices]
                
                y_main_train = y_main[train_indices]
                y_main_test = y_main[test_indices]
                
                y_main_raw_train = y_main_raw[train_indices]
                y_main_raw_test = y_main_raw[test_indices]
                
                # Train a transformer model for this fold
                params = self.system.params.copy()
                input_shape = (X_train.shape[1], main_sequences.shape[2])
                
                # Create and train models
                transformer_model = None
                frequency_model = None
                pattern_model = None
                
                # Only train enabled models - INITIALIZE MODEL FLAGS
                transformer_enabled = params.get('use_transformer_model', True)
                frequency_enabled = params.get('use_frequency_model', True)
                pattern_enabled = params.get('use_pattern_model', True)
                
                # Ensure at least one model is enabled
                if not (transformer_enabled or frequency_enabled or pattern_enabled):
                    logger.warning("No models enabled for ensemble, enabling frequency model as fallback")
                    frequency_enabled = True
                    params['use_frequency_model'] = True
                
                if transformer_enabled:
                    transformer_model = TransformerModel(input_shape, MAIN_NUM_MAX, params)
                    transformer_model.build_model()
                    
                    # Train with reduced epochs for CV
                    transformer_model.train(
                        X_train=(X_train, main_seq_train),
                        y_train=y_main_train,
                        epochs=20,  # Reduced epochs
                        validation_split=0.1
                    )
                
                if frequency_enabled:
                    frequency_model = FrequencyModel()
                    frequency_model.build_model()
                    frequency_model.train(X_train, y_main_raw_train)
                
                if pattern_enabled:
                    pattern_model = PatternModel()
                    pattern_model.build_model()
                    pattern_model.train(X_train, y_main_raw_train)
                
                # Predict on test set
                predictions = []
                
                for i in range(len(X_test)):
                    # Get predictions from each model
                    model_probs = {}
                    
                    # Get transformer model predictions only if model is enabled and trained
                    if transformer_model is not None:
                        transformer_probs = transformer_model.predict((X_test[i:i+1], main_seq_test[i:i+1]))
                        model_probs['transformer'] = transformer_probs
                    
                    # Get frequency model predictions only if model is enabled and trained
                    if frequency_model is not None:
                        freq_probs = frequency_model.predict_probabilities(X_test[i:i+1])
                        model_probs['frequency'] = freq_probs
                    
                    # Get pattern model predictions only if model is enabled and trained
                    if pattern_model is not None:
                        pattern_probs = pattern_model.predict_probabilities(X_test[i:i+1])
                        model_probs['pattern'] = pattern_probs
                    
                    # Check if we have any model predictions before proceeding
                    if not model_probs:
                        logger.warning(f"No valid model predictions for sample {i} in fold {fold+1}")
                        continue
                    
                    # Combine model predictions based on ensemble weights
                    ensemble_weights = params.get('ensemble_weights', {
                        'transformer': 0.4,
                        'frequency': 0.3,
                        'pattern': 0.3
                    })
                    
                    # Only use weights for models that are available
                    ensemble_weights = {k: v for k, v in ensemble_weights.items() if k in model_probs}
                    
                    # Normalize weights if any models are missing
                    weight_sum = sum(ensemble_weights.values())
                    if weight_sum > 0:
                        ensemble_weights = {k: v / weight_sum for k, v in ensemble_weights.items()}
                    
                    # Create calibrator
                    calibrator = EnsembleCalibrator()
                    
                    # Apply calibration
                    ensemble_probs = calibrator.calibrate_ensemble(model_probs, None, 0)
                    
                    # Sample numbers
                    main_numbers = Utilities.sample_numbers(
                        probs=[ensemble_probs],
                        available_nums=range(MAIN_NUM_MIN, MAIN_NUM_MAX+1),
                        num_to_select=MAIN_NUM_COUNT,
                        temperature=DEFAULT_TEMPERATURE
                    )
                    
                    # Create position map for prediction
                    main_positions = {num: idx for idx, num in enumerate(main_numbers)}
                    
                    # Get actual numbers and their positions
                    actual_numbers = y_main_raw_test[i]
                    actual_positions = {num: idx for idx, num in enumerate(actual_numbers)}
                    
                    # Generate a simple bonus number prediction
                    bonus_probs = np.ones(BONUS_NUM_MAX) / BONUS_NUM_MAX  # Simple uniform distribution for CV
                    bonus_numbers = Utilities.sample_numbers(
                        probs=[bonus_probs.reshape(1, -1)],
                        available_nums=range(BONUS_NUM_MIN, BONUS_NUM_MAX+1),
                        num_to_select=BONUS_NUM_COUNT,
                        temperature=DEFAULT_TEMPERATURE
                    )
                    
                    # Get actual bonus numbers if available (or use empty list)
                    actual_bonus = []  # We may not have actual bonus numbers in CV
                    
                    # Create prediction dict
                    prediction = {
                        "main_numbers": main_numbers,
                        "main_number_positions": main_positions,
                        "bonus_numbers": bonus_numbers
                    }
            
                    # Create actual draw dict
                    actual_draw = {
                        "main_numbers": actual_numbers,
                        "main_number_positions": actual_positions,
                        "bonus_numbers": actual_bonus
                    }
                    
                    # Calculate partial match score
                    match_score = Utilities.calculate_partial_match_score(prediction, actual_draw)
                    
                    # Calculate pattern score
                    pattern_score = Utilities.calculate_pattern_score(main_numbers)
                    
                    predictions.append({
                        'predicted': main_numbers,
                        'actual': actual_numbers,
                        'partial_match_score': match_score,
                        'pattern_score': pattern_score
                    })
                
                # Calculate fold metrics
                if predictions:
                    # Use the new partial match score instead of simple match rate
                    avg_match_score = np.mean([p['partial_match_score'] for p in predictions])
                    avg_pattern_score = np.mean([p['pattern_score'] for p in predictions])
                    
                    # Store metrics
                    fold_metrics.append({
                        'fold': fold + 1,
                        'match_score': avg_match_score,
                        'pattern_score': avg_pattern_score,
                        'score': 0.7 * avg_match_score + 0.3 * avg_pattern_score
                    })
                    
                    logger.info(f"Fold {fold+1}: Match Score = {avg_match_score:.4f}, Pattern Score = {avg_pattern_score:.4f}")
                
                # Clean memory between folds
                Utilities.clean_memory()
                
            except Exception as e:
                logger.error(f"Error evaluating fold {fold+1}: {e}")
                logger.error(traceback.format_exc())
                continue
        
        # Calculate average metrics
        if fold_metrics:
            avg_match_score = np.mean([m['match_score'] for m in fold_metrics])
            avg_pattern_score = np.mean([m['pattern_score'] for m in fold_metrics])
            avg_score = np.mean([m['score'] for m in fold_metrics])
        else:
            avg_match_score = 0.0
            avg_pattern_score = 0.0
            avg_score = 0.0
        
        logger.info(f"Cross-validation complete:")
        logger.info(f"  Average Match Score: {avg_match_score:.4f}")
        logger.info(f"  Average Pattern Score: {avg_pattern_score:.4f}")
        logger.info(f"  Average Combined Score: {avg_score:.4f}")
        
        return {
            'fold_metrics': fold_metrics,
            'avg_overall_acc': avg_match_score,
            'avg_pattern_score': avg_pattern_score,
            'avg_score': avg_score
        }

#######################
# VISUALIZATION FUNCTIONS
#######################

@ErrorHandler.handle_exception(logger, "visualization", None)
def generate_visualizations(predictions, file_path, output_dir=VISUALIZATION_DIR):
    """Generate enhanced visualizations for lottery predictions."""
    logger.info("Generating visualizations for lottery predictions")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load historical data for comparison
    include_historical = False
    try:
        processor = LotteryDataProcessor(file_path)
        data = processor.parse_file()
        include_historical = not data.empty
    except Exception as e:
        logger.warning(f"Could not load historical data for visualization: {e}")
        data = None
    
    # Set up the figure for main visualization
    plt.figure(figsize=(16, 12))
    
    # Plot 1: Frequency of main numbers
    plt.subplot(2, 2, 1)
    
    if include_historical:
        # Historical frequency
        main_freq = np.zeros(MAIN_NUM_MAX)
        all_main_numbers = [num for row in data["main_numbers"] for num in row]
        for num in all_main_numbers:
            if MAIN_NUM_MIN <= num <= MAIN_NUM_MAX:
                main_freq[num-1] += 1
        main_freq = main_freq / len(data) if len(data) > 0 else np.zeros(MAIN_NUM_MAX)
    else:
        # Create a uniform distribution as baseline
        main_freq = np.ones(MAIN_NUM_MAX) / MAIN_NUM_MAX
    
    # Predicted frequency
    pred_main_freq = np.zeros(MAIN_NUM_MAX)
    all_pred_main_numbers = [num for pred in predictions for num in pred["main_numbers"]]
    for num in all_pred_main_numbers:
        if MAIN_NUM_MIN <= num <= MAIN_NUM_MAX:
            pred_main_freq[num-1] += 1
    pred_main_freq = pred_main_freq / len(predictions) if len(predictions) > 0 else np.zeros(MAIN_NUM_MAX)
    
    # Create the bar chart
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
    
    # Plot 2: Frequency of bonus numbers
    plt.subplot(2, 2, 2)
    
    if include_historical:
        # Historical frequency
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
    
    # Save first figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "lottery_predictions.png"), dpi=300)
    plt.close()
    
    # Create second figure for additional analysis
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
    
    # Plot 4: Method Analysis
    plt.subplot(2, 2, 4)
    
    # Count predictions by method
    method_counts = {}
    for pred in predictions:
        method = pred.get("method", "unknown")
        if method in method_counts:
            method_counts[method] += 1
        else:
            method_counts[method] = 1
    
    # Create pie chart
    methods = list(method_counts.keys())
    counts = list(method_counts.values())
    
    if methods:
        plt.pie(counts, labels=methods, autopct='%1.1f%%', startangle=90, 
               colors=plt.cm.tab10.colors[:len(methods)])
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        plt.title("Prediction Methods Distribution", fontsize=14)
    else:
        plt.text(0.5, 0.5, "No method data available",
               horizontalalignment='center', verticalalignment='center')
    
    # Save the second figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pattern_analysis.png"), dpi=300)
    plt.close()
    
    # Create a third figure for confidence analysis
    plt.figure(figsize=(16, 8))
    
    # Plot 1: Confidence vs Pattern Score
    plt.subplot(1, 2, 1)
    
    # Extract confidence and pattern scores
    overall_confidences = [pred["confidence"]["overall"] for pred in predictions]
    pattern_scores = [pred["confidence"].get("pattern_score", 0) for pred in predictions]
    
    plt.scatter(pattern_scores, overall_confidences, alpha=0.7)
    plt.title("Confidence vs. Pattern Score", fontsize=14)
    plt.xlabel("Pattern Score", fontsize=12)
    plt.ylabel("Overall Confidence", fontsize=12)
    plt.grid(alpha=0.3)
    
    # Add trend line
    if len(overall_confidences) > 1:
        z = np.polyfit(pattern_scores, overall_confidences, 1)
        p = np.poly1d(z)
        plt.plot(sorted(pattern_scores), p(sorted(pattern_scores)), "r--", alpha=0.7)
    
    # Plot 2: Number occurrence heatmap
    plt.subplot(1, 2, 2)
    
    # Create matrix of frequency counts for numbers appearing together
    cooccurrence = np.zeros((10, 5))  # 5 decades x 5 positions
    
    # Count occurrences of each number by decade and position
    for pred in predictions:
        main_nums = pred["main_numbers"]
        for i, num in enumerate(sorted(main_nums)):
            # Determine decade (0-4)
            decade = (num - 1) // 10
            if 0 <= decade < 5:
                cooccurrence[decade*2, i] += 1
                # Also bump adjacent cells for smoother visualization
                if decade*2 + 1 < 10:
                    cooccurrence[decade*2 + 1, i] += 0.5
    
    # Normalize
    row_sums = cooccurrence.sum(axis=1, keepdims=True)
    if np.all(row_sums > 0):
        cooccurrence = cooccurrence / row_sums
    
    # Create heatmap
    plt.imshow(cooccurrence, cmap='viridis', aspect='auto')
    plt.colorbar(label='Normalized Frequency')
    plt.title("Number Decade vs. Position Heatmap", fontsize=14)
    plt.xlabel("Position (Left to Right)", fontsize=12)
    plt.ylabel("Number Decade", fontsize=12)
    
    # Set y-ticks for decades
    decades = ['1-10', '11-20', '21-30', '31-40', '41-50']
    plt.yticks(np.arange(0, 10, 2), decades)
    
    # Set x-ticks for positions
    plt.xticks(np.arange(5), [f"Pos {i+1}" for i in range(5)])
    
    # Save the third figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confidence_analysis.png"), dpi=300)
    plt.close()
    
    logger.info(f"Visualizations saved to {output_dir}")
    
    # Return paths to visualization files
    return [
        os.path.join(output_dir, "lottery_predictions.png"),
        os.path.join(output_dir, "pattern_analysis.png"),
        os.path.join(output_dir, "confidence_analysis.png")
    ]


#######################
# MAIN FUNCTION
#######################

def main():
    """Main function to run the hybrid neural lottery prediction system."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Hybrid Neural Lottery Prediction System")
    parser.add_argument("--file", default="lottery_numbers.txt", help="Path to lottery data file")
    parser.add_argument("--optimize", action="store_true", help="Run hyperparameter optimization before prediction")
    parser.add_argument("--trials", type=int, default=30, help="Number of optimization trials")
    parser.add_argument("--predictions", type=int, default=20, help="Number of predictions to generate")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate model performance with cross-validation")
    parser.add_argument("--ensemble", action="store_true", help="Use ensemble for improved predictions")
    parser.add_argument("--params", default="hybrid_params.json", help="Path to parameters file")
    parser.add_argument("--sequence_length", type=int, default=None, help="Override sequence length for historical data")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="Temperature for sampling")
    parser.add_argument("--output", default="predictions.json", help="Output file for predictions")
    parser.add_argument("--no_visualize", action="store_true", help="Skip visualization generation")
    parser.add_argument("--load_existing", action="store_true", help="Load existing models instead of training")
    parser.add_argument("--force_train", action="store_true", help="Force training even with load_existing")
    parser.add_argument("--gpus", type=int, default=None, help="Number of GPUs to use (default: use all available)")
    parser.add_argument("--match_recent", action="store_true", help="Evaluate predictions against recent results")
    parser.add_argument("--recent_draws", type=int, default=5, help="Number of recent draws to use for evaluation")

    args = parser.parse_args()
    
    try:
        # Configure GPU usage
        if args.gpus is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(args.gpus))
            print(f"Limited to {args.gpus} GPU(s)")
        
        # Apply memory optimizations
        optimize_memory_usage()

        # Print header
        print("\n" + "="*80)
        print("HYBRID NEURAL LOTTERY PREDICTION SYSTEM".center(80))
        print("="*80 + "\n")
        print("This system combines transformer models, frequency analysis, and pattern recognition")
        print("to enhance lottery prediction accuracy.")
        print("\nDISCLAIMER: Lottery outcomes are primarily random events and no")
        print("prediction system can guarantee winning numbers.")
        print("="*80 + "\n")

        
        # Check if data file exists
        if not os.path.exists(args.file):
            print(f"Error: Lottery data file '{args.file}' not found.")
            sys.exit(1)
        
        # Load parameters
        params = Utilities.load_params(args.params, Utilities.get_default_params())
        
        # Override sequence length if provided
        if args.sequence_length is not None:
            params['sequence_length'] = args.sequence_length
            print(f"Using custom sequence length: {args.sequence_length}")
        
        # Handle optimization if requested
        if args.optimize:
            print(f"\nRunning hyperparameter optimization with {args.trials} trials...")
            optimizer = HyperparameterOptimizer(args.file, args.trials)
            params = optimizer.optimize()
            Utilities.save_params(params, args.params)
            print(f"Optimization complete. Best parameters saved to {args.params}")
            
            # Generate optimization plots
            plot_file = optimizer.plot_optimization_history()
            if plot_file:
                print(f"Optimization history plot saved to {plot_file}")
            
            # Clean memory
            Utilities.clean_memory(force=True)
        
        # Create the prediction system
        print("\nInitializing hybrid neural prediction system...")
        system = HybridNeuralSystem(args.file, params=params)
        system.load_existing = args.load_existing
        
        # Prepare data
        print("Preparing data...")
        system.prepare_data()
        
        # Evaluate model if requested
        if args.evaluate:
            print("\nEvaluating model performance with cross-validation...")
            evaluator = CrossValidationEvaluator(system, folds=5)
            performance = evaluator.evaluate()
            
            print("\nCross-Validation Performance:")
            print(f"Overall Accuracy: {performance['avg_overall_acc']*100:.2f}%")
            print(f"Pattern Score: {performance['avg_pattern_score']*100:.2f}%")
            print(f"Combined Score: {performance['avg_score']*100:.2f}%")
            
            # Clean memory
            Utilities.clean_memory(force=True)
        
        # Build models
        print("\nBuilding prediction models...")
        system.build_models()
        
        # Train models if not loading existing or if force_train is specified
        if not args.load_existing or args.force_train:
            print("\nTraining prediction models (this may take some time)...")
            system.train_models()
            print("Model training completed successfully.")
        else:
            print("\nUsing existing trained models...")
            # Could add a check here to verify models exist
        
        # Generate predictions
        print(f"\nGenerating {args.predictions} predictions...")
        predictions = system.predict(args.predictions, temperature=args.temperature)
        
        # Check if we have valid predictions
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
            print("-" * 30)
                
        # Evaluate predictions against recent draws if requested
        if args.match_recent and system.processor.data is not None:
            print(f"\nEvaluating predictions against {args.recent_draws} most recent draws...")
            
            # Get the most recent draws
            recent_draws = system.processor.data.tail(args.recent_draws)
            
            # Calculate match scores for each prediction against each recent draw
            match_results = []
            
            for i, pred in enumerate(predictions):
                best_score = 0
                best_draw_info = None
                
                for _, row in recent_draws.iterrows():
                    # Format actual draw
                    actual_draw = {
                        "main_numbers": row["main_numbers"],
                        "main_number_positions": {num: idx for idx, num in enumerate(row["main_numbers"])},
                        "bonus_numbers": row["bonus_numbers"]
                    }
                    
                    # Calculate match score using our utility function
                    match_score = Utilities.calculate_partial_match_score(pred, actual_draw)
                    
                    # Track best match
                    if match_score > best_score:
                        best_score = match_score
                        draw_date = row["date"].strftime("%Y-%m-%d") if hasattr(row["date"], "strftime") else str(row["date"])
                        main_matches = len(set(pred["main_numbers"]) & set(row["main_numbers"]))
                        bonus_matches = len(set(pred["bonus_numbers"]) & set(row["bonus_numbers"]))
                        
                        best_draw_info = {
                            "date": draw_date,
                            "score": match_score,
                            "main_matches": main_matches,
                            "bonus_matches": bonus_matches
                        }
                
                match_results.append({
                    "prediction": i+1,
                    "main_numbers": pred["main_numbers"],
                    "best_match": best_draw_info,
                    "score": best_score
                })
            
            # Sort results by match score
            match_results.sort(key=lambda x: x["score"], reverse=True)
            
            # Display top matching predictions
            print("\nPredictions Ranked by Match Score:")
            print("=" * 50)
            
            for i, result in enumerate(match_results[:min(5, len(match_results))]):
                print(f"Rank {i+1} (Score: {result['score']:.4f}):")
                print(f"  Prediction: Main Numbers {result['main_numbers']}")
                if result["best_match"]:
                    print(f"  Best Match: {result['best_match']['date']}")
                    print(f"  Main Matches: {result['best_match']['main_matches']}, Bonus Matches: {result['best_match']['bonus_matches']}")
                print("-" * 40)
        
        # Generate visualizations
        if not args.no_visualize:
            print("\nGenerating enhanced visualizations...")
            viz_files = generate_visualizations(predictions, args.file)
            if viz_files:
                print(f"Visualizations saved: {', '.join(viz_files)}")
        
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
            logging.error(f"Error in summary generation: {str(e)}")
            print(f"\nError generating summary: {str(e)}")
    
    except KeyboardInterrupt:
        print("\nPrediction system stopped by user.")
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        print(f"\nError: {str(e)}")
        print("Check hybrid_neural_lottery.log for details.")

if __name__ == "__main__":
    main()