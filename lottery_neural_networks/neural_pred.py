#!/usr/bin/env python
"""
Optimized Transformer-Based EuroMillions Prediction System

Usage:
    python optimized_transformers.py --file lottery_numbers.txt --predictions 20
    python optimized_transformers.py --file lottery_numbers.txt --optimize --trials 30 --ensemble --num_models 7 --predictions 20 --evaluate
    python optimized_transformers.py --file lottery_numbers.txt --params your_params.json --ensemble --num_models 7 --evaluate --predictions 20
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
from typing import Dict, List, Tuple, Union, Optional, Any, Set
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, MultiHeadAttention, LayerNormalization,
    GlobalAveragePooling1D, Concatenate, Reshape, Add, GRU, Conv1D,
    Bidirectional, BatchNormalization, Activation
)
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l1_l2
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import gc
from functools import lru_cache

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

# Model parameters
DEFAULT_SEQUENCE_LENGTH = 20
DEFAULT_EMBED_DIM = 64
DEFAULT_NUM_HEADS = 4
DEFAULT_FF_DIM = 128
DEFAULT_DROPOUT_RATE = 0.3
DEFAULT_BATCH_SIZE = 16
DEFAULT_EPOCHS = 100
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_TRANSFORMER_BLOCKS = 2
DEFAULT_CONV_FILTERS = 32
DEFAULT_PATIENCE = 15

# Confidence calibration factors
MAIN_CONF_SCALE = 0.6
MAIN_CONF_OFFSET = 0.2
BONUS_CONF_SCALE = 0.7
BONUS_CONF_OFFSET = 0.15

# Feature selection
MAX_FEATURES = 300

# Temperature scaling for sampling
DEFAULT_TEMPERATURE = 0.8
DIVERSITY_FACTOR = 0.8

# Memory management
MEMORY_CLEANUP_FREQUENCY = 5  # Standardized cleanup frequency

# Configure GPU if available
def configure_gpu():
    """Configure GPU with proper memory growth if available."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Configure GPU memory growth to avoid taking all memory at once
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            return f"Found {len(gpus)} GPU(s)"
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
    def log_and_raise(logger, error_msg, exception=None):
        """Log an error and return details."""
        if exception:
            logger.error(f"{error_msg}: {str(exception)}")
            logger.error(traceback.format_exc())
        else:
            logger.error(error_msg)
        return error_msg
    
    @staticmethod
    def handle_with_fallback(logger, operation_name, func, fallback_result, *args, **kwargs):
        """Run a function with proper error handling and fallback."""
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {operation_name}: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Call fallback if it's a function, otherwise return it directly
            if callable(fallback_result):
                return fallback_result()
            return fallback_result

#######################
# UTILITIES
#######################

class Utilities:
    """Unified utility class for shared functionality."""
    
    _memory_cleanup_counter = 0
    
    @staticmethod
    def clean_memory(force=False):
        """Force garbage collection to free memory based on standardized frequency."""
        try:
            # Increment counter
            Utilities._memory_cleanup_counter += 1
            
            # Only perform cleanup at specified frequency or when forced
            if force or Utilities._memory_cleanup_counter >= MEMORY_CLEANUP_FREQUENCY:
                # Reset counter
                Utilities._memory_cleanup_counter = 0
                
                # Force garbage collection
                gc.collect()
                
                # If running in TensorFlow environment, clear session
                tf.keras.backend.clear_session()
                
                logger.info("Memory cleanup performed")
        except Exception as e:
            logger.warning(f"Memory cleanup failed: {e}")
    
    @staticmethod
    def sample_numbers(probs, available_nums, num_to_select, 
                      used_nums=None, diversity_sampling=True, 
                      diversity_factor=DIVERSITY_FACTOR, draw_idx=0, temperature=DEFAULT_TEMPERATURE):
        """Centralized sampling functionality for number selection."""
        selected_numbers = []
        available = list(available_nums)
        
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
                    
                    # Apply diversity penalty
                    if diversity_sampling and used_nums and num in used_nums and draw_idx > 0:
                        adjusted_probs[num-1] *= diversity_factor
            
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
    def calculate_pattern_score(main_numbers, bonus_numbers, data=None):
        """Calculate pattern score based on historical patterns."""
        # Simple pattern checks
        # 1. Check distribution across range
        main_bins = [0, 0, 0, 0, 0]  # 1-10, 11-20, 21-30, 31-40, 41-50
        for num in main_numbers:
            bin_idx = (num - 1) // 10
            if 0 <= bin_idx < 5:
                main_bins[bin_idx] += 1
        
        # Perfect distribution would have 1 number in each bin
        distribution_score = 1.0 - sum(abs(x - 1) for x in main_bins) / 5
        
        # 2. Check for consecutive numbers
        consecutive_count = 0
        for i in range(len(main_numbers) - 1):
            if main_numbers[i + 1] - main_numbers[i] == 1:
                consecutive_count += 1
        
        # Realistic patterns usually have 0-1 consecutive pairs
        if consecutive_count <= 1:
            consecutive_score = 0.9
        elif consecutive_count == 2:
            consecutive_score = 0.7
        else:
            consecutive_score = 0.5  # More than 2 is rare
        
        # 3. Check for even/odd balance
        even_count = sum(1 for n in main_numbers if n % 2 == 0)
        odd_count = MAIN_NUM_COUNT - even_count
        
        # Ideal balance is 2-3 or 3-2
        if even_count in [2, 3]:
            balance_score = 0.9
        elif even_count in [1, 4]:
            balance_score = 0.7
        else:
            balance_score = 0.5  # All even or all odd is rare
        
        # Combine scores with weights
        pattern_score = (0.4 * distribution_score + 0.3 * consecutive_score + 0.3 * balance_score)
        return pattern_score
    
    @staticmethod
    def calculate_frequency_score(main_numbers, bonus_numbers, data):
        """Calculate score based on historical frequency of numbers."""
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
        
        # Calculate frequency score for selected main numbers
        main_score = np.mean([main_freq[num-1] for num in main_numbers])
        
        # Normalize score to 0-1 range
        main_score_norm = min(1.0, main_score / 0.1)
        
        # Same for bonus numbers
        bonus_counts = np.zeros(BONUS_NUM_MAX)
        for _, row in data.iterrows():
            for num in row["bonus_numbers"]:
                if BONUS_NUM_MIN <= num <= BONUS_NUM_MAX:
                    bonus_counts[num-1] += 1
        
        bonus_freq = bonus_counts / len(data)
        bonus_score = np.mean([bonus_freq[num-1] for num in bonus_numbers])
        
        # Normalize (expected value for random selection is 0.167 (2/12))
        bonus_score_norm = min(1.0, bonus_score / 0.167)
        
        # Combine scores
        return (main_score_norm + bonus_score_norm) / 2
    
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
            'num_transformer_blocks': DEFAULT_TRANSFORMER_BLOCKS
        }
    
    @staticmethod
    def normalize_params(params, default_params=None):
        """Centralized parameter normalization to avoid duplication."""
        if default_params is None:
            default_params = Utilities.get_default_params()
            
        # Create a new dict with default values
        normalized_params = default_params.copy()
        
        # Update with provided params (if any)
        if params:
            for key, value in params.items():
                if key in normalized_params:
                    normalized_params[key] = value
        
        return normalized_params
    
    @staticmethod
    def load_params(file_path, default_params=None):
        """Load model parameters from JSON file with proper error handling."""
        if default_params is None:
            default_params = Utilities.get_default_params()
            
        try:
            if file_path and os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    params = json.load(f)
                logger.info(f"Loaded parameters from {file_path}")
                
                # Use the centralized parameter normalization
                return Utilities.normalize_params(params, default_params)
            else:
                logger.warning(f"Parameter file {file_path} not found, using defaults")
                return default_params
        except Exception as e:
            ErrorHandler.log_and_raise(logger, f"Error loading parameters: {str(e)}")
            return default_params
    
    @staticmethod
    def save_params(params, file_path):
        """Save model parameters to JSON file with proper error handling."""
        try:
            # Create directory if it doesn't exist
            directory = os.path.dirname(os.path.abspath(file_path))
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
            
            with open(file_path, 'w') as f:
                json.dump(params, f, indent=4)
            logger.info(f"Saved parameters to {file_path}")
            return True
        except Exception as e:
            ErrorHandler.log_and_raise(logger, f"Error saving parameters: {str(e)}")
            return False
    
    @staticmethod
    def get_model_callbacks(model_prefix="model", patience=DEFAULT_PATIENCE, 
                          include_checkpoint=True, base_path="models/best", 
                          include_timestamp=True):
        """Unified callbacks for model training to avoid duplication."""
        callbacks = []
        
        # Early stopping
        callbacks.append(EarlyStopping(
            patience=patience,
            restore_best_weights=True,
            monitor='val_loss'
        ))
        
        # Learning rate scheduler
        callbacks.append(ReduceLROnPlateau(
            factor=0.7,
            patience=patience // 2,
            min_lr=1e-6,
            monitor='val_loss',
            verbose=1
        ))
        
        # Model checkpoint
        checkpoint_path = None
        if include_checkpoint:
            if include_timestamp:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filepath = f"{base_path}_{model_prefix}_{timestamp}.keras"
            else:
                filepath = f"{base_path}_{model_prefix}.keras"
                
            # Create directory if it doesn't exist
            directory = os.path.dirname(os.path.abspath(filepath))
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
                
            callbacks.append(ModelCheckpoint(
                filepath=filepath,
                save_best_only=True,
                monitor='val_loss',
                mode='min',
                verbose=1
            ))
            checkpoint_path = filepath
        
        return callbacks, checkpoint_path

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
        # Cache for historical counts to avoid recalculation
        self._main_counts_cache = None
        self._bonus_counts_cache = None
        
    def parse_file(self):
        """Parse the lottery data file into a structured DataFrame."""
        logger.info(f"Parsing lottery data from {self.file_path}")
        
        if not os.path.exists(self.file_path):
            ErrorHandler.log_and_raise(logger, f"Lottery data file not found: {self.file_path}")
            return pd.DataFrame()
            
        try:
            with open(self.file_path, 'r') as file:
                content = file.read()
            
            # Improved regex pattern - more robust to variations in formatting
            draw_pattern = r"((?:\w+)\s+\d+(?:st|nd|rd|th)?\s+(?:\w+)\s+\d{4})\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(€[\d,]+)\s+(Roll|Won)"
            draws = re.findall(draw_pattern, content)
            
            if not draws:
                ErrorHandler.log_and_raise(logger, f"No valid draws found in {self.file_path}")
                return pd.DataFrame()
            
            # Pre-allocate lists for better performance
            dates = []
            day_of_weeks = []
            main_numbers_list = []
            bonus_numbers_list = []
            jackpots = []
            results = []
            
            # Process extracted data in one pass
            for draw in draws:
                try:
                    date_str = draw[0]
                    main_numbers = [int(draw[i]) for i in range(1, 6)]
                    bonus_numbers = [int(draw[i]) for i in range(6, 8)]
                    jackpot = draw[8]
                    result = draw[9]
                    
                    # Improved date parsing
                    clean_date_str = re.sub(r'(\d+)(st|nd|rd|th)', r'\1', date_str)
                    date = datetime.strptime(clean_date_str, "%A %d %B %Y")
                    
                    # Append to lists
                    dates.append(date)
                    day_of_weeks.append(date.strftime("%A"))
                    main_numbers_list.append(sorted(main_numbers))
                    bonus_numbers_list.append(sorted(bonus_numbers))
                    jackpots.append(jackpot)
                    results.append(result)
                except Exception as e:
                    logger.warning(f"Could not parse draw: {draw}, error: {e}")
                    continue
            
            # Convert to DataFrame in one step
            df = pd.DataFrame({
                "date": dates,
                "day_of_week": day_of_weeks,
                "main_numbers": main_numbers_list,
                "bonus_numbers": bonus_numbers_list,
                "jackpot": jackpots,
                "result": results
            }

#######################
# VISUALIZATION FUNCTIONS
#######################

def generate_visualizations(predictions, file_path):
    """Generate visualizations for lottery predictions."""
    logger.info("Generating visualizations for lottery predictions")
    
    try:
        # Create processor to get historical data
        processor = LotteryDataProcessor(file_path)
        data = processor.parse_file()
        
        if data.empty:
            logger.error("No data available for visualization")
            return None
        
        # Set up the figure for main visualization
        plt.figure(figsize=(16, 12))
        
        # Pre-calculate historical frequencies to avoid redundant calculations
        main_freq = np.zeros(MAIN_NUM_MAX)
        bonus_freq = np.zeros(BONUS_NUM_MAX)
        hist_odd = 0
        hist_even = 0
        hist_range_counts = np.zeros(5)
        
        # Process historical data in a single pass
        for _, row in data.iterrows():
            # Main numbers
            for num in row["main_numbers"]:
                if MAIN_NUM_MIN <= num <= MAIN_NUM_MAX:
                    main_freq[num-1] += 1
                    
                    # Count odd/even
                    if num % 2 == 0:
                        hist_even += 1
                    else:
                        hist_odd += 1
                    
                    # Count by range
                    range_idx = (num - 1) // 10
                    if 0 <= range_idx < 5:
                        hist_range_counts[range_idx] += 1
            
            # Bonus numbers            
            for num in row["bonus_numbers"]:
                if BONUS_NUM_MIN <= num <= BONUS_NUM_MAX:
                    bonus_freq[num-1] += 1
        
        # Normalize historical frequencies
        main_freq = main_freq / len(data) if len(data) > 0 else np.zeros(MAIN_NUM_MAX)
        bonus_freq = bonus_freq / len(data) if len(data) > 0 else np.zeros(BONUS_NUM_MAX)
        
        # Calculate predicted frequencies in one pass
        pred_main_freq = np.zeros(MAIN_NUM_MAX)
        pred_bonus_freq = np.zeros(BONUS_NUM_MAX)
        pred_odd = 0
        pred_even = 0
        pred_range_counts = np.zeros(5)
        
        for pred in predictions:
            for num in pred["main_numbers"]:
                if MAIN_NUM_MIN <= num <= MAIN_NUM_MAX:
                    pred_main_freq[num-1] += 1
                    
                    # Count odd/even
                    if num % 2 == 0:
                        pred_even += 1
                    else:
                        pred_odd += 1
                    
                    # Count by range
                    range_idx = (num - 1) // 10
                    if 0 <= range_idx < 5:
                        pred_range_counts[range_idx] += 1
            
            for num in pred["bonus_numbers"]:
                if BONUS_NUM_MIN <= num <= BONUS_NUM_MAX:
                    pred_bonus_freq[num-1] += 1
        
        # Normalize predicted frequencies
        pred_main_freq = pred_main_freq / len(predictions) if len(predictions) > 0 else np.zeros(MAIN_NUM_MAX)
        pred_bonus_freq = pred_bonus_freq / len(predictions) if len(predictions) > 0 else np.zeros(BONUS_NUM_MAX)
        
        # Plot 1: Frequency of main numbers (historical vs predicted)
        plt.subplot(2, 2, 1)
        x = np.arange(1, MAIN_NUM_MAX+1)
        width = 0.35
        plt.bar(x - width/2, main_freq, width, label='Historical', alpha=0.7, color='royalblue')
        plt.bar(x + width/2, pred_main_freq, width, label='Predicted', alpha=0.7, color='seagreen')
        
        plt.title("Main Numbers Frequency Comparison", fontsize=14)
        plt.xlabel("Number", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.xticks(range(0, MAIN_NUM_MAX+1, 5))
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        
        # Plot 2: Frequency of bonus numbers (historical vs predicted)
        plt.subplot(2, 2, 2)
        x = np.arange(1, BONUS_NUM_MAX+1)
        plt.bar(x - width/2, bonus_freq, width, label='Historical', alpha=0.7, color='royalblue')
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
        
        # Create histogram
        plt.hist(confidences, bins=10, color='purple', alpha=0.7)
        
        plt.title("Prediction Confidence Distribution", fontsize=14)
        plt.xlabel("Confidence (%)", fontsize=12)
        plt.ylabel("Count", fontsize=12)
        plt.grid(axis='y', alpha=0.3)
        
        # Add vertical line for average confidence
        avg_conf = np.mean(confidences) if len(confidences) > 0 else 0
        plt.axvline(x=avg_conf, color='r', linestyle='--', alpha=0.7, label=f"Average ({avg_conf:.2f}%)")
        plt.legend()
        
        # Plot 4: Pattern Analysis (Number Range Distribution)
        plt.subplot(2, 2, 4)
        ranges = ["1-10", "11-20", "21-30", "31-40", "41-50"]
        
        # Normalize
        hist_sum = np.sum(hist_range_counts)
        pred_sum = np.sum(pred_range_counts)
        hist_range_counts = hist_range_counts / hist_sum if hist_sum > 0 else hist_range_counts
        pred_range_counts = pred_range_counts / pred_sum if pred_sum > 0 else pred_range_counts
        
        # Plot as bar chart
        x = np.arange(len(ranges))
        plt.bar(x - width/2, hist_range_counts, width, label='Historical', alpha=0.7, color='royalblue')
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
        
        # Create a separate figure for pattern analysis
        plt.figure(figsize=(10, 6))
        
        # Normalize odd/even counts
        hist_total = hist_odd + hist_even
        pred_total = pred_odd + pred_even
        
        if hist_total > 0:
            hist_odd_pct = hist_odd / hist_total * 100
            hist_even_pct = hist_even / hist_total * 100
        else:
            hist_odd_pct = 0
            hist_even_pct = 0
            
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
        plt.bar(x - width/2, hist_values, width, label='Historical', alpha=0.7, color='royalblue')
        plt.bar(x + width/2, pred_values, width, label='Predicted', alpha=0.7, color='seagreen')
        
        plt.title("Odd/Even Distribution", fontsize=14)
        plt.xlabel("Number Type", fontsize=12)
        plt.ylabel("Percentage (%)", fontsize=12)
        plt.xticks(x, categories)
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        
        # Save the pattern analysis
        plt.tight_layout()
        plt.savefig("pattern_analysis.png", dpi=300)
        plt.close()
        
        logger.info("Visualizations saved to lottery_predictions.png and pattern_analysis.png")
        return ["lottery_predictions.png", "pattern_analysis.png"]
    except Exception as e:
        ErrorHandler.log_and_raise(logger, "Error generating visualizations", e)
        return None)
            
            if df.empty:
                ErrorHandler.log_and_raise(logger, "No valid draws could be parsed")
                return df
                
            df = df.sort_values("date")
            
            # Extract jackpot value as numeric
            df["jackpot_value"] = df["jackpot"].str.replace("€", "").str.replace(",", "").astype(float)
            df["is_won"] = df["result"] == "Won"
            
            self.data = df
            logger.info(f"Successfully parsed {len(df)} draws")
            
            return df
            
        except Exception as e:
            ErrorHandler.log_and_raise(logger, "Error parsing lottery data file", e)
            return pd.DataFrame()
    
    def expand_numbers(self):
        """Expand main and bonus numbers into individual columns with enhanced metadata."""
        if self.data is None or self.data.empty:
            ErrorHandler.log_and_raise(logger, "No data available. Parse the file first.")
            return pd.DataFrame()
        
        try:
            df = self.data.copy()
            
            # Vectorized approach for expanding numbers
            main_df = pd.DataFrame(df["main_numbers"].tolist(), 
                                 index=df.index, 
                                 columns=[f"main_{i+1}" for i in range(MAIN_NUM_COUNT)])
            
            bonus_df = pd.DataFrame(df["bonus_numbers"].tolist(), 
                                  index=df.index, 
                                  columns=[f"bonus_{i+1}" for i in range(BONUS_NUM_COUNT)])
            
            # Concatenate with original dataframe
            df = pd.concat([df, main_df, bonus_df], axis=1)
            
            # Add metadata about the draw more efficiently
            df["draw_index"] = range(len(df))
            df["year"] = df["date"].dt.year
            df["month"] = df["date"].dt.month
            df["day"] = df["date"].dt.day
            df["day_of_week_num"] = df["date"].dt.dayofweek
            
            # Handle the isocalendar() approach
            try:
                df["week_of_year"] = df["date"].dt.isocalendar().week
            except:
                df["week_of_year"] = df["date"].dt.weekofyear
            
            # Vectorized holiday season calculation
            df["is_holiday_season"] = ((df["month"] == 12) & (df["day"] >= 15)) | ((df["month"] == 1) & (df["day"] <= 15))
            
            # Calculate days since last draw
            df["days_since_last_draw"] = (df["date"] - df["date"].shift(1)).dt.days
            df["days_since_last_draw"].fillna(0, inplace=True)
            
            self.expanded_data = df
            return df
        except Exception as e:
            ErrorHandler.log_and_raise(logger, "Error expanding numbers", e)
            return pd.DataFrame()
    
    def get_historical_counts(self, force_recalculate=False):
        """Get historical counts with caching to avoid recalculation."""
        if self.data is None or self.data.empty:
            return None

class CrossValidationEvaluator:
    """Evaluator for time-series cross validation of lottery prediction."""
    
    def __init__(self, file_path, params=None, folds=5):
        """Initialize the cross-validation evaluator."""
        self.file_path = file_path
        self.params = Utilities.normalize_params(params)
        self.folds = folds
        self.data_dict = None
        self.system = None
    
    def evaluate(self):
        """Perform time-series cross-validation with metrics."""
        logger.info(f"Performing {self.folds}-fold time-series cross-validation")
        
        try:
            # Create predictor system and prepare data once if not already provided
            if self.data_dict is None:
                if self.system is None:
                    self.system = LotteryPredictionSystem(self.file_path, self.params)
                self.data_dict = self.system.prepare_data()
            else:
                # If data_dict was provided externally, ensure system is updated
                if self.system is None:
                    self.system = LotteryPredictionSystem(self.file_path, self.params)
                    self.system._data_prepared = True
                    # Share data with the system
                    X_scaled = pd.DataFrame(self.data_dict["X"])
                    self.system.X_scaled = X_scaled
                    self.system.main_sequences = self.data_dict["main_sequences"]
                    self.system.bonus_sequences = self.data_dict["bonus_sequences"]
            
            # Calculate total data size and validate
            total_samples = len(self.data_dict["X"])
            if total_samples < self.folds * 2:
                logger.warning(f"Not enough data ({total_samples} samples) for {self.folds} folds")
                # Adjust folds to a reasonable number
                self.folds = max(2, total_samples // 2)
                logger.info(f"Adjusted to {self.folds} folds")
            
            fold_size = total_samples // self.folds
            
            # Initialize metrics
            fold_metrics = []
            
            # Pre-compute all fold indices to avoid redundant calculations
            fold_indices = []
            for fold in range(self.folds):
                test_start = fold * fold_size
                test_end = min((fold + 1) * fold_size, total_samples)
                
                # Skip invalid folds
                if test_start >= test_end or test_start >= total_samples:
                    continue
                    
                train_indices = list(range(0, test_start)) + list(range(test_end, total_samples))
                test_indices = list(range(test_start, test_end))
                
                if not train_indices or not test_indices:
                    continue
                    
                fold_indices.append((train_indices, test_indices))
            
            # Perform time-series cross-validation for each valid fold
            for fold, (train_indices, test_indices) in enumerate(fold_indices):
                try:
                    logger.info(f"Training fold {fold+1}/{len(fold_indices)}")
                    
                    # Split data for this fold
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
                    model.build_models(X_train.shape[1], self.system.sequence_length)
                    
                    # Train with reduced epochs for CV
                    model.train_models(
                        X_train=X_train,
                        main_seq_train=main_seq_train,
                        bonus_seq_train=bonus_seq_train,
                        y_main_train=y_main_train,
                        y_bonus_train=y_bonus_train,
                        epochs=30  # Reduced epochs for CV
                    )
                    
                    # Evaluate on test set
                    main_preds = model.main_model.predict([X_test, main_seq_test])
                    bonus_preds = model.bonus_model.predict([X_test, bonus_seq_test])
                
                    # Convert to class predictions
                    main_class_preds = []
                    for i in range(MAIN_NUM_COUNT):
                        main_class_preds.append(np.argmax(main_preds[i], axis=1))
                    
                    bonus_class_preds = []
                    for i in range(BONUS_NUM_COUNT):
                        bonus_class_preds.append(np.argmax(bonus_preds[i], axis=1))
                    
                    # Calculate accuracy for each position
                    main_position_acc = []
                    for i in range(MAIN_NUM_COUNT):
                        main_position_acc.append(float(np.mean(main_class_preds[i] == y_main_test[:, i])))
                    
                    bonus_position_acc = []
                    for i in range(BONUS_NUM_COUNT):
                        bonus_position_acc.append(float(np.mean(bonus_class_preds[i] == y_bonus_test[:, i])))
                    
                    # Calculate average accuracy
                    main_avg_acc = float(np.mean(main_position_acc))
                    bonus_avg_acc = float(np.mean(bonus_position_acc))
                    overall_avg_acc = float((main_avg_acc + bonus_avg_acc) / 2)
                    
                    # Store metrics
                    fold_metrics.append({
                        'fold': fold+1,
                        'main_accuracy': main_avg_acc,
                        'bonus_accuracy': bonus_avg_acc,
                        'overall_accuracy': overall_avg_acc,
                        'main_position_acc': main_position_acc,
                        'bonus_position_acc': bonus_position_acc
                    })
                    
                    logger.info(f"Fold {fold+1} metrics: Main acc={main_avg_acc:.4f}, Bonus acc={bonus_avg_acc:.4f}, Overall={overall_avg_acc:.4f}")
                    
                except Exception as e:
                    logger.error(f"Error processing fold {fold+1}: {e}")
                    # Add a placeholder with zero accuracy for this fold
                    fold_metrics.append({
                        'fold': fold+1,
                        'main_accuracy': 0.0,
                        'bonus_accuracy': 0.0,
                        'overall_accuracy': 0.0,
                        'error': str(e)
                    })
                
                # Clean memory after each fold
                Utilities.clean_memory()
            
            # Calculate average metrics if we have any valid folds
            if fold_metrics:
                avg_main_acc = float(np.mean([m['main_accuracy'] for m in fold_metrics]))
                avg_bonus_acc = float(np.mean([m['bonus_accuracy'] for m in fold_metrics]))
                avg_overall_acc = float(np.mean([m['overall_accuracy'] for m in fold_metrics]))
            else:
                avg_main_acc = 0.0
                avg_bonus_acc = 0.0
                avg_overall_acc = 0.0
            
            logger.info(f"Average metrics across {len(fold_metrics)} folds:")
            logger.info(f"Main numbers accuracy: {avg_main_acc:.4f}")
            logger.info(f"Bonus numbers accuracy: {avg_bonus_acc:.4f}")
            logger.info(f"Overall accuracy: {avg_overall_acc:.4f}")
            
            return {
                'fold_metrics': fold_metrics,
                'avg_main_accuracy': avg_main_acc,
                'avg_bonus_accuracy': avg_bonus_acc,
                'avg_overall_accuracy': avg_overall_acc
            }
            
        except Exception as e:
            ErrorHandler.log_and_raise(logger, "Critical error in cross-validation", e)
            return {
                'fold_metrics': [],
                'avg_main_accuracy': 0.0,
                'avg_bonus_accuracy': 0.0,
                'avg_overall_accuracy': 0.0,
                'error': str(e)
            }, None
            
        # Return cached values if available
        if not force_recalculate and self._main_counts_cache is not None and self._bonus_counts_cache is not None:
            return self._main_counts_cache, self._bonus_counts_cache
            
        # Calculate counts
        main_counts = np.zeros(MAIN_NUM_MAX)
        bonus_counts = np.zeros(BONUS_NUM_MAX)
        
        for _, row in self.data.iterrows():
            for num in row["main_numbers"]:
                if MAIN_NUM_MIN <= num <= MAIN_NUM_MAX:
                    main_counts[num-1] += 1
            for num in row["bonus_numbers"]:
                if BONUS_NUM_MIN <= num <= BONUS_NUM_MAX:
                    bonus_counts[num-1] += 1
        
        # Cache the results
        self._main_counts_cache = main_counts
        self._bonus_counts_cache = bonus_counts
        
        return main_counts, bonus_counts
            
    def create_features(self):
        """Generate all features for lottery prediction."""
        if self.expanded_data is None or self.expanded_data.empty:
            self.expanded_data = self.expand_numbers()
            if self.expanded_data.empty:
                return pd.DataFrame()
                
        # Create feature engineering instance
        self.feature_engineer = FeatureEngineering(self.expanded_data)
        
        # Pass historical counts to avoid recalculation
        main_counts, bonus_counts = self.get_historical_counts()
        self.feature_engineer.set_historical_counts(main_counts, bonus_counts)
        
        # Create features
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
    
    def _initialize_number_statistics(self):
        """Initialize historical statistics for each number."""
        try:
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
        except Exception as e:
            logger.error(f"Error initializing number statistics: {e}")
            # Set default values
            self.historical_counts = {
                "main": np.ones(MAIN_NUM_MAX) / MAIN_NUM_MAX,
                "bonus": np.ones(BONUS_NUM_MAX) / BONUS_NUM_MAX
            }
    
    def set_historical_counts(self, main_counts, bonus_counts):
        """Set historical counts from external source to avoid recalculation."""
        if main_counts is not None and bonus_counts is not None:
            self.historical_counts = {
                "main": main_counts / len(self.data) if len(self.data) > 0 else main_counts,
                "bonus": bonus_counts / len(self.data) if len(self.data) > 0 else bonus_counts
            }
            return True
        return False
    
    def create_enhanced_features(self):
        """Create and combine all advanced features."""
        logger.info("Generating enhanced features for lottery prediction")
        
        try:
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
            
        except Exception as e:
            ErrorHandler.log_and_raise(logger, f"Error creating enhanced features: {e}")
            # Return empty DataFrame with correct index
            return pd.DataFrame(index=self.data.index)
    
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
        
        return [df for df in feature_sets if not df.empty]
    
    def _select_top_features(self, features, max_features=MAX_FEATURES):
        """Select top features to avoid dimensionality issues."""
        # Skip if we're already under the feature limit (optimization)
        if features.shape[1] <= max_features:
            return features
            
        try:
            # Use mutual information to select important features
            X_raw = features.select_dtypes(include=[np.number])
            
            # Create a target variable (any of the main numbers)
            y = self.data["main_1"].values if "main_1" in self.data.columns else None
            
            if y is None or len(y) != len(X_raw):
                # Fallback to keeping all features
                logger.warning("Feature selection failed - keeping all features")
                return features
                
            # Select top features
            selector = SelectKBest(mutual_info_regression, k=max_features)
            X_selected = selector.fit_transform(X_raw, y)
            
            # Get selected feature names
            selected_indices = selector.get_support(indices=True)
            selected_cols = [X_raw.columns[i] for i in selected_indices]
            
            # Create new DataFrame with selected features
            selected_features = pd.DataFrame(
                X_selected, 
                columns=selected_cols,
                index=features.index
            )
            
            logger.info(f"Selected {len(selected_cols)} features from {features.shape[1]} total")
            return selected_features
            
        except Exception as e:
            logger.warning(f"Feature selection failed: {e} - keeping all features")
            return features
    
    def _fix_infinite_values(self, data_frame):
        """Handle infinite or extremely large values in the DataFrame."""
        try:
            # Replace infinities with NaN
            data_frame.replace([np.inf, -np.inf], np.nan, inplace=True)
            
            # Fill NaN values with 0
            data_frame.fillna(0, inplace=True)
            
            # Get list of numeric columns more efficiently
            numeric_cols = data_frame.select_dtypes(include=[np.number]).columns
            
            # Calculate percentiles once for all columns
            df_percentiles = data_frame[numeric_cols].quantile([0.01, 0.99])
            
            # Clip all columns at once for better performance
            for col in numeric_cols:
                try:
                    q_lo = df_percentiles.loc[0.01, col]
                    q_hi = df_percentiles.loc[0.99, col]
                    range_val = q_hi - q_lo
                    data_frame[col] = data_frame[col].clip(q_lo - 5*range_val, q_hi + 5*range_val)
                except Exception as e:
                    # Skip problematic columns
                    continue
            
            return data_frame
        except Exception as e:
            logger.error(f"Error fixing infinite values: {e}")
            # Return original DataFrame if fixing fails
            return data_frame
    
    def _calculate_time_series_features(self, window_sizes=[5, 10, 20]):
        """Calculate time series features with adaptive windowing."""
        df = self.data.copy()
        
        # Create empty dictionary for time series features
        ts_features = {}
        
        # Create indicators for important numbers (subset to avoid feature explosion)
        indicators = {}
        
        # Main numbers - select a subset of numbers to track (every 5th number)
        for num in range(1, MAIN_NUM_MAX+1, 5):
            indicators[f"main_has_{num}"] = df["main_numbers"].apply(lambda x: num in x).astype(int)
            
        # Bonus numbers - select a subset (every 3rd number)
        for num in range(1, BONUS_NUM_MAX+1, 3):
            indicators[f"bonus_has_{num}"] = df["bonus_numbers"].apply(lambda x: num in x).astype(int)
        
        # Convert indicators to DataFrame
        indicators_df = pd.DataFrame(indicators, index=df.index)
        
        # Calculate rolling statistics with adaptive windows
        for window in window_sizes:
            if window >= len(df):
                continue
                
            # Only compute for selected numbers
            for num in range(1, MAIN_NUM_MAX+1, 5):
                # Exponentially weighted moving average
                ts_features[f"main_{num}_ewma_{window}"] = indicators_df[f"main_has_{num}"].ewm(span=window).mean()
                
                # Mean reversion indicator - deviation from historical mean
                ts_features[f"main_{num}_mean_rev_{window}"] = (
                    indicators_df[f"main_has_{num}"].rolling(window=window).mean() - 
                    self.historical_counts["main"][num-1]
                )
                
            # Similar calculations for bonus numbers
            for num in range(1, BONUS_NUM_MAX+1, 3):
                ts_features[f"bonus_{num}_ewma_{window}"] = indicators_df[f"bonus_has_{num}"].ewm(span=window).mean()
                
                # Mean reversion indicator
                ts_features[f"bonus_{num}_mean_rev_{window}"] = (
                    indicators_df[f"bonus_has_{num}"].rolling(window=window).mean() - 
                    self.historical_counts["bonus"][num-1]
                )
        
        # Convert to DataFrame and handle NaN values
        ts_df = pd.DataFrame(ts_features, index=df.index)
        return ts_df.fillna(0)
    
    def _calculate_number_relationships(self):
        """Calculate enhanced features based on relationships between numbers."""
        df = self.data.copy()
        
        # Process all draws at once for efficiency
        main_nums_list = df["main_numbers"].tolist()
        bonus_nums_list = df["bonus_numbers"].tolist()
        
        num_draws = len(df)
        
        # Pre-allocate arrays for better performance
        main_mean_diff = np.zeros(num_draws)
        main_std_diff = np.zeros(num_draws)
        main_min_diff = np.zeros(num_draws)
        main_max_diff = np.zeros(num_draws)
        main_range = np.zeros(num_draws)
        main_sum = np.zeros(num_draws)
        main_mean = np.zeros(num_draws)
        main_low_count = np.zeros(num_draws)
        main_high_count = np.zeros(num_draws)
        main_odd_count = np.zeros(num_draws)
        main_even_count = np.zeros(num_draws)
        main_consecutive_count = np.zeros(num_draws)
        
        bonus_diff = np.zeros(num_draws)
        bonus_sum = np.zeros(num_draws)
        bonus_odd_count = np.zeros(num_draws)
        bonus_even_count = np.zeros(num_draws)
        
        # Use NumPy vectorization where possible
        for i in range(num_draws):
            main_nums = main_nums_list[i]
            bonus_nums = bonus_nums_list[i]
            
            if len(main_nums) >= MAIN_NUM_COUNT:
                try:
                    # Calculate differences between consecutive numbers (vectorized)
                    main_nums_array = np.array(main_nums)
                    diffs = main_nums_array[1:] - main_nums_array[:-1]
                    
                    # Store statistics about differences
                    main_mean_diff[i] = np.mean(diffs)
                    main_std_diff[i] = np.std(diffs)
                    main_min_diff[i] = np.min(diffs)
                    main_max_diff[i] = np.max(diffs)
                    main_range[i] = np.max(main_nums_array) - np.min(main_nums_array)
                    
                    # Calculate sum and mean
                    main_sum[i] = np.sum(main_nums_array)
                    main_mean[i] = np.mean(main_nums_array)
                    
                    # Distribution statistics
                    main_low_count[i] = np.sum(main_nums_array <= 25)
                    main_high_count[i] = np.sum(main_nums_array > 25)
                    main_odd_count[i] = np.sum(main_nums_array % 2 == 1)
                    main_even_count[i] = np.sum(main_nums_array % 2 == 0)
                    
                    # Consecutive numbers - calculate via diffs array
                    main_consecutive_count[i] = np.sum(diffs == 1)
                    
                except Exception:
                    # Keep zeros for failed calculations
                    pass
            
            # Similar calculations for bonus numbers
            if len(bonus_nums) >= BONUS_NUM_COUNT:
                try:
                    bonus_nums_array = np.array(bonus_nums)
                    bonus_diff[i] = bonus_nums[1] - bonus_nums[0]
                    bonus_sum[i] = np.sum(bonus_nums_array)
                    bonus_odd_count[i] = np.sum(bonus_nums_array % 2 == 1)
                    bonus_even_count[i] = np.sum(bonus_nums_array % 2 == 0)
                except Exception:
                    # Keep zeros for failed calculations
                    pass
        
        # Create the relationship features DataFrame in one step
        relationship_features = pd.DataFrame({
            'main_mean_diff': main_mean_diff,
            'main_std_diff': main_std_diff,
            'main_min_diff': main_min_diff,
            'main_max_diff': main_max_diff,
            'main_range': main_range,
            'main_sum': main_sum,
            'main_mean': main_mean,
            'main_low_count': main_low_count,
            'main_high_count': main_high_count,
            'main_odd_count': main_odd_count,
            'main_even_count': main_even_count,
            'main_consecutive_count': main_consecutive_count,
            'bonus_diff': bonus_diff,
            'bonus_sum': bonus_sum,
            'bonus_odd_count': bonus_odd_count,
            'bonus_even_count': bonus_even_count
        }, index=df.index)
        
        return relationship_features
    
    def _calculate_pattern_features(self):
        """Calculate features based on number patterns and clusters."""
        df = self.data.copy()
        
        num_samples = len(df)
        
        # Pre-allocate pattern features
        pattern_cluster = np.zeros(num_samples)
        pattern_distance = np.zeros(num_samples)
        
        # Create a signature for each draw's main numbers
        signatures = []
        valid_indices = []
        
        # Define bins once
        bins = [1, 11, 21, 31, 41, 51]  # 1-10, 11-20, ..., 41-50
        
        for i, (_, row) in enumerate(df.iterrows()):
            try:
                main_nums = row["main_numbers"]
                
                # Create a histogram of the numbers (count in each decade)
                # Use NumPy's histogram function more efficiently
                hist, _ = np.histogram(main_nums, bins=bins)
                
                # Create pattern signature with decade distribution
                if len(main_nums) >= 2:
                    # Normalize the histogram
                    hist_sum = np.sum(hist)
                    hist_norm = hist/hist_sum if hist_sum > 0 else hist
                    
                    # Add signature with draw index
                    signatures.append(hist_norm)
                    valid_indices.append(i)
            except Exception:
                continue
        
        # Create clusters of similar patterns if we have enough data
        min_signatures_for_clustering = max(10, min(100, num_samples // 4))
        if len(signatures) >= min_signatures_for_clustering:
            try:
                # Convert to array
                signatures_array = np.array(signatures)
                
                # Reduce dimensions with PCA first if we have enough samples
                if signatures_array.shape[0] > 2 and signatures_array.shape[1] > 1:
                    n_components = min(3, signatures_array.shape[1])
                    pca = PCA(n_components=n_components)
                    signatures_pca = pca.fit_transform(signatures_array)
                    
                    # Create clusters
                    n_clusters = min(5, len(signatures) // 4)
                    n_clusters = max(2, n_clusters)  # At least 2 clusters
                    
                    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED)
                    cluster_labels = kmeans.fit_predict(signatures_pca)
                    
                    # Save cluster information for later
                    self.pattern_clusters = {
                        'kmeans': kmeans,
                        'pca': pca
                    }
                    
                    # Calculate distances vectorized where possible
                    for idx, (orig_idx, cluster) in enumerate(zip(valid_indices, cluster_labels)):
                        pattern_cluster[orig_idx] = cluster
                        
                        # Distance to cluster center
                        center = kmeans.cluster_centers_[cluster]
                        pattern_distance[orig_idx] = np.linalg.norm(signatures_pca[idx] - center)
                else:
                    # Just use simple clustering directly on signatures if dimensions not sufficient
                    kmeans = KMeans(n_clusters=2, random_state=RANDOM_SEED)
                    cluster_labels = kmeans.fit_predict(signatures_array)
                    
                    for idx, (orig_idx, cluster) in enumerate(zip(valid_indices, cluster_labels)):
                        pattern_cluster[orig_idx] = cluster
            except Exception as e:
                logger.warning(f"Pattern clustering failed: {e}")
        
        # Create the pattern feature DataFrame
        pattern_features = pd.DataFrame({
            'main_pattern_cluster': pattern_cluster,
            'main_pattern_distance': pattern_distance
        }, index=df.index)
        
        return pattern_features
    
    def _calculate_cyclical_features(self):
        """Calculate cyclical time features using sine/cosine transformations."""
        df = self.data.copy()
        
        num_samples = len(df)
        
        # Extract date information vectorized
        day_of_week = df["date"].dt.weekday.values
        month = df["date"].dt.month.values
        
        # Get week of year
        try:
            week_of_year = df["date"].dt.isocalendar().week.values
        except:
            # Fallback for older pandas versions
            week_of_year = df["date"].dt.weekofyear.values
        
        # Calculate cyclical features more efficiently using NumPy operations
        day_of_week_sin = np.sin(2 * np.pi * day_of_week / 7)
        day_of_week_cos = np.cos(2 * np.pi * day_of_week / 7)
        
        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)
        
        week_of_year_sin = np.sin(2 * np.pi * week_of_year / 53)
        week_of_year_cos = np.cos(2 * np.pi * week_of_year / 53)
        
        # Create the cyclical features DataFrame in one step
        cyclical_features = pd.DataFrame({
            'day_of_week_sin': day_of_week_sin,
            'day_of_week_cos': day_of_week_cos,
            'month_sin': month_sin,
            'month_cos': month_cos,
            'week_of_year_sin': week_of_year_sin,
            'week_of_year_cos': week_of_year_cos
        }, index=df.index)
        
        return cyclical_features

#######################
# TRANSFORMER MODEL COMPONENTS
#######################

class TransformerBlock(tf.keras.layers.Layer):
    """Transformer block with multi-head attention and feed-forward network."""
    
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=DEFAULT_DROPOUT_RATE):
        super(TransformerBlock, self).__init__()
        
        # More efficient key_dim calculation
        key_dim = embed_dim // num_heads
        
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, dropout=dropout)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="gelu"),
            Dropout(dropout),
            Dense(embed_dim)
        ])
        
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        
    def call(self, inputs, training=False):
        # Self-attention with pre-norm architecture
        x = self.layernorm1(inputs)
        attn_output = self.att(x, x)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = inputs + attn_output
        
        # Feed-forward network
        x = self.layernorm2(out1)
        ffn_output = self.ffn(x)
        ffn_output = self.dropout2(ffn_output, training=training)
        
        return out1 + ffn_output

class PositionalEncoding(tf.keras.layers.Layer):
    """Positional encoding layer for transformer models."""
    
    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)
        
    def positional_encoding(self, position, d_model):
        """Optimized positional encoding computation."""
        # Pre-compute position indices and divisors in single operations
        positions = tf.range(position, dtype=tf.float32)[:, tf.newaxis]
        div_term = tf.pow(10000.0, tf.range(0, d_model, 2, dtype=tf.float32) / tf.cast(d_model, tf.float32))
        
        # Calculate angles in one step
        angles = positions / div_term
        
        # Apply sin to even indices, cos to odd indices
        pos_encoding = tf.zeros((position, d_model), dtype=tf.float32)
        pos_encoding = tf.tensor_scatter_nd_update(
            pos_encoding, 
            tf.stack([tf.repeat(tf.range(position), d_model//2), 
                     tf.tile(tf.range(0, d_model, 2), [position])], axis=1),
            tf.reshape(tf.sin(angles), [-1])
        )
        pos_encoding = tf.tensor_scatter_nd_update(
            pos_encoding, 
            tf.stack([tf.repeat(tf.range(position), d_model//2), 
                     tf.tile(tf.range(1, d_model, 2), [position])], axis=1),
            tf.reshape(tf.cos(angles), [-1])
        )
        
        # Add batch dimension
        pos_encoding = pos_encoding[tf.newaxis, ...]
        
        return tf.cast(pos_encoding, tf.float32)
        
    def call(self, inputs):
        # Add positional encoding to input embeddings
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

#######################
# MODEL ARCHITECTURE
#######################

class ModelBuilder:
    """Unified model building for lottery prediction models."""
    
    @staticmethod
    def build_transformer_model(input_dim, seq_length, params, model_config):
        """Build transformer model for lottery number prediction with flexible configuration."""
        # Extract model configuration
        num_outputs = model_config['num_outputs'] 
        output_size = model_config['output_size']
        sequence_size = model_config['sequence_size']
        name_prefix = model_config['name_prefix']
        
        # Input layers
        feature_input = Input(shape=(input_dim,), name=f"{name_prefix}_feature_input")
        sequence_input = Input(shape=(seq_length, sequence_size), name=f"{name_prefix}_sequence_input")
        
        # Process feature input
        x_features = Dense(params.get('embed_dim', DEFAULT_EMBED_DIM), activation="gelu")(feature_input)
        x_features = BatchNormalization()(x_features)
        x_features = Dropout(params.get('dropout_rate', DEFAULT_DROPOUT_RATE))(x_features)
        
        x_features = Dense(params.get('embed_dim', DEFAULT_EMBED_DIM) // 2, activation="gelu")(x_features)
        
        # Process sequence input
        if params.get('conv_filters', DEFAULT_CONV_FILTERS) > 0:
            x_seq = Conv1D(params.get('conv_filters', DEFAULT_CONV_FILTERS), kernel_size=3, padding='same', activation="gelu")(sequence_input)
        else:
            x_seq = sequence_input
            
        x_seq = Dense(params.get('embed_dim', DEFAULT_EMBED_DIM))(x_seq)
        
        # Apply positional encoding
        x_seq = PositionalEncoding(seq_length, params.get('embed_dim', DEFAULT_EMBED_DIM))(x_seq)
        
        # Apply transformer blocks
        for _ in range(params.get('num_transformer_blocks', DEFAULT_TRANSFORMER_BLOCKS)):
            x_seq = TransformerBlock(
                params.get('embed_dim', DEFAULT_EMBED_DIM),
                params.get('num_heads', DEFAULT_NUM_HEADS),
                params.get('ff_dim', DEFAULT_FF_DIM),
                params.get('dropout_rate', DEFAULT_DROPOUT_RATE)
            )(x_seq)
        
        # Process sequence with GRU or global pooling
        if params.get('use_gru', True):
            x_seq = Bidirectional(GRU(params.get('embed_dim', DEFAULT_EMBED_DIM) // 2, return_sequences=False))(x_seq)
        else:
            x_seq = GlobalAveragePooling1D()(x_seq)
        
        # Combine feature and sequence representations
        combined = Concatenate()([x_features, x_seq])
        
        # Dense layers for combined processing
        combined = Dense(params.get('ff_dim', DEFAULT_FF_DIM), activation="gelu")(combined)
        combined = BatchNormalization()(combined)
        combined = Dropout(params.get('dropout_rate', DEFAULT_DROPOUT_RATE))(combined)
        
        combined = Dense(params.get('ff_dim', DEFAULT_FF_DIM) // 2, activation="gelu")(combined)
        combined = Dropout(params.get('dropout_rate', DEFAULT_DROPOUT_RATE) / 2)(combined)
        
        # Output layers (one for each position)
        outputs = []
        for i in range(num_outputs):
            # Position-specific processing
            position_specific = Dense(params.get('ff_dim', DEFAULT_FF_DIM) // 4, activation="gelu")(combined)
            
            logits = Dense(output_size, activation=None, name=f"{name_prefix}_logits_{i+1}")(position_specific)
            output = Activation('softmax', name=f"{name_prefix}_{i+1}")(logits)
            outputs.append(output)
        
        # Create model
        model = Model(inputs=[feature_input, sequence_input], outputs=outputs)
        
        # Compile model with metrics
        metrics_dict = {f"{name_prefix}_{i+1}": "accuracy" for i in range(num_outputs)}
        
        # Configure optimizer
        if params.get('optimizer', 'adam').lower() == 'adam':
            optimizer = Adam(learning_rate=params.get('learning_rate', DEFAULT_LEARNING_RATE))
        elif params.get('optimizer', 'adam').lower() == 'sgd':
            optimizer = SGD(learning_rate=params.get('learning_rate', DEFAULT_LEARNING_RATE), momentum=0.9)
        elif params.get('optimizer', 'adam').lower() == 'rmsprop':
            optimizer = RMSprop(learning_rate=params.get('learning_rate', DEFAULT_LEARNING_RATE))
        else:
            optimizer = Adam(learning_rate=params.get('learning_rate', DEFAULT_LEARNING_RATE))
        
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
        self.params = Utilities.normalize_params(params)
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
    
    def train_models(self, X_train, main_seq_train, bonus_seq_train, 
                    y_main_train, y_bonus_train, validation_split=0.1, 
                    epochs=None, batch_size=None):
        """Train both models with unified approach."""
        if self.main_model is None or self.bonus_model is None:
            raise ValueError("Models not built. Call build_models() first.")
        
        # Use parameter values or defaults
        actual_epochs = epochs if epochs is not None else self.params.get('epochs', DEFAULT_EPOCHS)
        actual_batch_size = batch_size if batch_size is not None else self.params.get('batch_size', DEFAULT_BATCH_SIZE)
        
        # Get callbacks only once using unified function
        main_callbacks, _ = Utilities.get_model_callbacks(model_prefix="main", patience=DEFAULT_PATIENCE)
        
        # Flatten target arrays
        y_main_train_flat = [y_main_train[:, j].flatten() for j in range(MAIN_NUM_COUNT)]
        
        # Train main model
        logger.info("Training main numbers model")
        main_history = self.main_model.fit(
            [X_train, main_seq_train],
            y_main_train_flat,
            epochs=actual_epochs,
            batch_size=actual_batch_size,
            validation_split=validation_split,
            callbacks=main_callbacks,
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
        Utilities.clean_memory(force=True)
        
        # Get callbacks for bonus model - reuse same config but with bonus prefix
        bonus_callbacks, _ = Utilities.get_model_callbacks(model_prefix="bonus", patience=DEFAULT_PATIENCE)
        
        # Flatten bonus target arrays
        y_bonus_train_flat = [y_bonus_train[:, j].flatten() for j in range(BONUS_NUM_COUNT)]
        
        # Train bonus model
        logger.info("Training bonus numbers model")
        bonus_history = self.bonus_model.fit(
            [X_train, bonus_seq_train],
            y_bonus_train_flat,
            epochs=actual_epochs,
            batch_size=actual_batch_size,
            validation_split=validation_split,
            callbacks=bonus_callbacks,
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
    
    def predict(self, features, main_sequence, bonus_sequence, num_draws=5, 
               temperature=DEFAULT_TEMPERATURE, diversity_sampling=True):
        """Generate predictions using the model."""
        if self.main_model is None or self.bonus_model is None:
            raise ValueError("Models not trained. Train models first.")
        
        try:
            # Track used numbers for diversity
            used_main_numbers = set()
            used_bonus_numbers = set()
            
            # Generate predictions in batches for memory efficiency
            predictions = []
            batch_size = min(10, num_draws)  # Process in batches of 10 or fewer
            
            for batch_start in range(0, num_draws, batch_size):
                batch_end = min(batch_start + batch_size, num_draws)
                batch_draws = batch_end - batch_start
                
                # Predict main and bonus numbers only once per batch
                main_probs = self.main_model.predict([features, main_sequence])
                bonus_probs = self.bonus_model.predict([features, bonus_sequence])
                
                # Generate each prediction in the batch
                for i in range(batch_draws):
                    draw_idx = batch_start + i
                    
                    # Sample main numbers using centralized sampling function
                    main_numbers = Utilities.sample_numbers(
                        probs=main_probs,
                        available_nums=range(MAIN_NUM_MIN, MAIN_NUM_MAX+1),
                        num_to_select=MAIN_NUM_COUNT,
                        used_nums=used_main_numbers,
                        diversity_sampling=diversity_sampling,
                        draw_idx=draw_idx,
                        temperature=temperature
                    )
                    
                    # Sample bonus numbers
                    bonus_numbers = Utilities.sample_numbers(
                        probs=bonus_probs,
                        available_nums=range(BONUS_NUM_MIN, BONUS_NUM_MAX+1),
                        num_to_select=BONUS_NUM_COUNT,
                        used_nums=used_bonus_numbers,
                        diversity_sampling=diversity_sampling,
                        draw_idx=draw_idx,
                        temperature=temperature
                    )
                    
                    # Calculate confidence scores
                    main_confidence = np.mean([np.max(main_probs[j][0]) for j in range(MAIN_NUM_COUNT)])
                    bonus_confidence = np.mean([np.max(bonus_probs[j][0]) for j in range(BONUS_NUM_COUNT)])
                    
                    # Calibrate confidence scores
                    calibrated_main_conf = MAIN_CONF_SCALE * main_confidence + MAIN_CONF_OFFSET
                    calibrated_bonus_conf = BONUS_CONF_SCALE * bonus_confidence + BONUS_CONF_OFFSET
                    overall_confidence = (calibrated_main_conf + calibrated_bonus_conf) / 2
                    
                    predictions.append({
                        "main_numbers": main_numbers,
                        "bonus_numbers": bonus_numbers,
                        "confidence": {
                            "overall": float(overall_confidence),
                            "main_numbers": float(calibrated_main_conf),
                            "bonus_numbers": float(calibrated_bonus_conf)
                        },
                        "method": "transformer"
                    })
                
                # Clean memory at the end of each batch
                Utilities.clean_memory()
            
            return predictions
        except Exception as e:
            ErrorHandler.log_and_raise(logger, "Error generating predictions", e)
            return []

class LotteryPredictionSystem:
    """Unified prediction system that handles data processing and model management."""
    
    def __init__(self, file_path, params=None):
        """Initialize the prediction system."""
        self.file_path = file_path
        self.params = Utilities.normalize_params(params)
        self.processor = LotteryDataProcessor(file_path)
        self.model = None
        self.sequence_length = DEFAULT_SEQUENCE_LENGTH
        self.feature_scaler = StandardScaler()
        self.X_scaled = None
        self.main_sequences = None
        self.bonus_sequences = None
        self._data_prepared = False
        
    def prepare_data(self):
        """Load, process, and prepare data for model training."""
        # Skip if data already prepared
        if self._data_prepared and self.X_scaled is not None and self.main_sequences is not None:
            logger.info("Using cached prepared data")
            return {
                "X": self.X_scaled.iloc[self.sequence_length:].values,
                "main_sequences": self.main_sequences,
                "bonus_sequences": self.bonus_sequences,
                "y_main": self.y_main,
                "y_bonus": self.y_bonus,
                "data": self.processor.data
            }
            
        logger.info("Preparing data for lottery prediction")
        
        # Load and process lottery data
        data = self.processor.parse_file()
        if data.empty:
            raise ValueError("Failed to parse lottery data")
            
        expanded_data = self.processor.expand_numbers()
        features = self.processor.create_features()
        
        if features.empty:
            raise ValueError("Feature engineering failed")
        
        # Scale features
        X_raw = features.select_dtypes(include=[np.number])
        X_raw = X_raw.fillna(0)
        
        # Scale features
        self.X_scaled = pd.DataFrame(
            self.feature_scaler.fit_transform(X_raw),
            columns=X_raw.columns,
            index=X_raw.index
        )
        
        # Create sequence features for transformer
        self._create_sequences(expanded_data)
        
        # Create target variables (next draw's numbers)
        # Subtract 1 from each number to use as index (0-49 for main, 0-11 for bonus)
        self.y_main = np.array([
            expanded_data.iloc[self.sequence_length:][f"main_{i+1}"].values - 1 for i in range(MAIN_NUM_COUNT)
        ]).T
        
        self.y_bonus = np.array([
            expanded_data.iloc[self.sequence_length:][f"bonus_{i+1}"].values - 1 for i in range(BONUS_NUM_COUNT)
        ]).T
        
        # Match feature data to sequence data
        X_scaled_matched = self.X_scaled.iloc[self.sequence_length:]
        
        logger.info(f"Data preparation complete. Features: {X_scaled_matched.shape}")
        
        # Mark data as prepared
        self._data_prepared = True
        
        return {
            "X": X_scaled_matched.values,
            "main_sequences": self.main_sequences,
            "bonus_sequences": self.bonus_sequences,
            "y_main": self.y_main,
            "y_bonus": self.y_bonus,
            "data": data  # Original data for reference
        }
    
    def _create_sequences(self, expanded_data):
        """Create sequence data for transformer model using vectorized operations."""
        logger.info("Creating sequence data for transformer model")
        
        # Get number of samples
        num_samples = len(expanded_data) - self.sequence_length
        
        # Pre-allocate arrays for efficiency
        main_sequences = np.zeros((num_samples, self.sequence_length, MAIN_NUM_MAX))
        bonus_sequences = np.zeros((num_samples, self.sequence_length, BONUS_NUM_MAX))
        
        # Process each window more efficiently
        for i in range(num_samples):
            # Get the window of previous draws
            window = expanded_data.iloc[i:i+self.sequence_length]
            
            # Extract main and bonus numbers for the entire window
            for j, (_, row) in enumerate(window.iterrows()):
                # Use vectorized operations to set one-hot encoding
                main_indices = np.array(row["main_numbers"]) - 1
                bonus_indices = np.array(row["bonus_numbers"]) - 1
                
                # Set valid indices to 1
                main_valid_mask = (main_indices >= 0) & (main_indices < MAIN_NUM_MAX)
                bonus_valid_mask = (bonus_indices >= 0) & (bonus_indices < BONUS_NUM_MAX)
                
                main_indices = main_indices[main_valid_mask]
                bonus_indices = bonus_indices[bonus_valid_mask]
                
                # Set values in one operation
                if len(main_indices) > 0:
                    main_sequences[i, j, main_indices] = 1
                
                if len(bonus_indices) > 0:
                    bonus_sequences[i, j, bonus_indices] = 1
        
        self.main_sequences = main_sequences
        self.bonus_sequences = bonus_sequences
        
        logger.info(f"Created sequences: Main shape: {self.main_sequences.shape}, Bonus shape: {self.bonus_sequences.shape}")
    
    def train_model(self, validation_split=0.1):
        """Train the lottery prediction model."""
        logger.info("Training lottery prediction model")
        
        # Prepare data
        data_dict = self.prepare_data()
        
        # Create and build model
        self.model = LotteryModel(self.params)
        
        # Build models
        input_dim = data_dict["X"].shape[1]
        self.model.build_models(input_dim, self.sequence_length)
        
        # Train models
        history = self.model.train_models(
            X_train=data_dict["X"],
            main_seq_train=data_dict["main_sequences"],
            bonus_seq_train=data_dict["bonus_sequences"],
            y_main_train=data_dict["y_main"],
            y_bonus_train=data_dict["y_bonus"],
            validation_split=validation_split
        )
        
        logger.info("Model training complete")
        return history
    
    def predict(self, num_draws=5, temperature=DEFAULT_TEMPERATURE, diversity_sampling=True):
        """Generate lottery predictions."""
        logger.info(f"Generating {num_draws} lottery predictions")
        
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        if self.X_scaled is None or self.main_sequences is None or self.bonus_sequences is None:
            data_dict = self.prepare_data()
        
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
                
                # Update overall confidence
                pred["confidence"]["overall"] = float(
                    (pred["confidence"]["overall"] + pattern_score + frequency_score) / 3
                )
        
        return predictions
    
    def generate_fallback_predictions(self, num_draws=5):
        """Generate fallback predictions if model-based prediction fails."""
        logger.warning("Using fallback prediction method")
        
        try:
            data = self.processor.data
            
            # Get historical counts with caching
            main_counts, bonus_counts = self.processor.get_historical_counts()
            
            if data is not None and not data.empty and main_counts is not None and bonus_counts is not None:
                # Normalize to probabilities
                main_sum = np.sum(main_counts)
                bonus_sum = np.sum(bonus_counts)
                
                main_probs = main_counts / main_sum if main_sum > 0 else np.ones(MAIN_NUM_MAX) / MAIN_NUM_MAX
                bonus_probs = bonus_counts / bonus_sum if bonus_sum > 0 else np.ones(BONUS_NUM_MAX) / BONUS_NUM_MAX
            else:
                # If no data, use uniform distribution
                main_probs = np.ones(MAIN_NUM_MAX) / MAIN_NUM_MAX
                bonus_probs = np.ones(BONUS_NUM_MAX) / BONUS_NUM_MAX
            
            # Generate predictions
            predictions = []
            
            # Simulate model outputs for sampling function
            main_model_probs = [np.array([main_probs]).reshape(1, -1) for _ in range(MAIN_NUM_COUNT)]
            bonus_model_probs = [np.array([bonus_probs]).reshape(1, -1) for _ in range(BONUS_NUM_COUNT)]
            
            # Track used numbers for diversity
            used_main_numbers = set()
            used_bonus_numbers = set()
            
            for i in range(num_draws):
                # Use the centralized sampling function for consistency
                main_numbers = Utilities.sample_numbers(
                    probs=main_model_probs,
                    available_nums=range(MAIN_NUM_MIN, MAIN_NUM_MAX+1),
                    num_to_select=MAIN_NUM_COUNT,
                    used_nums=used_main_numbers,
                    diversity_sampling=True,
                    draw_idx=i
                )
                
                bonus_numbers = Utilities.sample_numbers(
                    probs=bonus_model_probs,
                    available_nums=range(BONUS_NUM_MIN, BONUS_NUM_MAX+1),
                    num_to_select=BONUS_NUM_COUNT,
                    used_nums=used_bonus_numbers,
                    diversity_sampling=True,
                    draw_idx=i
                )
                
                predictions.append({
                    "main_numbers": main_numbers,
                    "bonus_numbers": bonus_numbers,
                    "confidence": {
                        "overall": 0.3,  # Lower confidence for fallback
                        "main_numbers": 0.3,
                        "bonus_numbers": 0.3,
                        "pattern_score": 0.5,
                        "frequency_score": 0.5
                    },
                    "method": "frequency_based_fallback"
                })
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error in fallback prediction: {str(e)}")
            
            # Last resort: truly random
            return self._generate_random_predictions(num_draws)
    
    def _generate_random_predictions(self, num_draws):
        """Generate completely random predictions as a last resort."""
        predictions = []
        for _ in range(num_draws):
            try:
                main_numbers = sorted(random.sample(range(MAIN_NUM_MIN, MAIN_NUM_MAX+1), MAIN_NUM_COUNT))
                bonus_numbers = sorted(random.sample(range(BONUS_NUM_MIN, BONUS_NUM_MAX+1), BONUS_NUM_COUNT))
            except Exception:
                # Ultimate fallback if even random.sample fails
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
                "method": "pure_random_fallback"
            })
        
        return predictions

#######################
# ENSEMBLE PREDICTOR
#######################

class EnsemblePredictor:
    """Ensemble predictor that combines multiple models for better predictions."""
    
    def __init__(self, file_path, num_models=5, params=None):
        """Initialize the ensemble predictor."""
        self.file_path = file_path
        self.num_models = num_models
        self.params = Utilities.normalize_params(params)
        self.base_predictors = []
        self.processor = LotteryDataProcessor(file_path)
        self.data_dict = None
        self._main_predictor = None
        
    def train(self):
        """Train multiple diverse base models."""
        logger.info(f"Training {self.num_models} diverse base models for ensemble")
        
        # First, train a main predictor to use its data
        self._main_predictor = LotteryPredictionSystem(self.file_path, self.params)
        self.data_dict = self._main_predictor.prepare_data()
        
        # Train the main predictor
        self._main_predictor.train_model()
        self.base_predictors.append(self._main_predictor)
        
        # Train additional diverse models - only if more than one requested
        if self.num_models > 1:
            # Create and initialize shared data structures to avoid redundant processing
            input_dim = self.data_dict["X"].shape[1]
            n_samples = len(self.data_dict["X"])
            
            for i in range(1, self.num_models):
                try:
                    logger.info(f"Training base model {i+1}/{self.num_models}")
                    
                    # Create model with diversity parameters
                    model_params = self._create_diverse_params(self.params, i)
                    
                    # Create a new prediction system, reusing the data from the main predictor
                    predictor = LotteryPredictionSystem(self.file_path, model_params)
                    
                    # Share data and processor to avoid duplication
                    predictor.X_scaled = self._main_predictor.X_scaled
                    predictor.main_sequences = self._main_predictor.main_sequences
                    predictor.bonus_sequences = self._main_predictor.bonus_sequences
                    predictor.processor = self._main_predictor.processor
                    predictor._data_prepared = True
                    predictor.y_main = self._main_predictor.y_main
                    predictor.y_bonus = self._main_predictor.y_bonus
                    
                    # Create the model with the diversity parameters
                    predictor.model = LotteryModel(model_params)
                    predictor.model.build_models(input_dim, predictor.sequence_length)
                    
                    # Set different random seed for each model
                    seed = i * 100 + RANDOM_SEED
                    set_seeds(seed)
                    
                    # Train with bootstrapping - sampling with replacement
                    bootstrap_indices = np.random.choice(n_samples, n_samples, replace=True)
                    
                    X_bootstrap = self.data_dict["X"][bootstrap_indices]
                    main_seq_bootstrap = self.data_dict["main_sequences"][bootstrap_indices]
                    bonus_seq_bootstrap = self.data_dict["bonus_sequences"][bootstrap_indices]
                    y_main_bootstrap = self.data_dict["y_main"][bootstrap_indices]
                    y_bonus_bootstrap = self.data_dict["y_bonus"][bootstrap_indices]
                    
                    # Train with shorter epochs for ensemble diversity
                    predictor.model.train_models(
                        X_train=X_bootstrap,
                        main_seq_train=main_seq_bootstrap,
                        bonus_seq_train=bonus_seq_bootstrap,
                        y_main_train=y_main_bootstrap,
                        y_bonus_train=y_bonus_bootstrap,
                        epochs=50  # Reduced epochs for ensemble models
                    )
                    
                    # Add to ensemble
                    self.base_predictors.append(predictor)
                    
                    # Clean memory according to standard frequency
                    Utilities.clean_memory()
                    
                except Exception as e:
                    logger.error(f"Error training model {i+1}: {e}")
                    continue
        
        # Reset random seeds
        set_seeds()
        
        logger.info(f"Ensemble training complete with {len(self.base_predictors)} models")
        return self.base_predictors
    
    def _create_diverse_params(self, base_params, model_index):
        """Create diverse parameters for ensemble models."""
        diverse_params = base_params.copy()
        
        # Apply diversity techniques
        diverse_params['dropout_rate'] = base_params.get('dropout_rate', DEFAULT_DROPOUT_RATE) * (0.8 + 0.6 * random.random())
        diverse_params['learning_rate'] = base_params.get('learning_rate', DEFAULT_LEARNING_RATE) * (0.7 + 0.6 * random.random())
        
        # Use a formula based on model_index to systematically vary parameters
        heads_options = [2, 4, 8]
        ff_dim_options = [64, 128, 256]
        optimizer_options = ['adam', 'rmsprop', 'sgd']
        
        diverse_params['num_heads'] = heads_options[model_index % len(heads_options)]
        diverse_params['ff_dim'] = ff_dim_options[(model_index // 2) % len(ff_dim_options)]
        diverse_params['optimizer'] = optimizer_options[(model_index // 3) % len(optimizer_options)]
        diverse_params['use_gru'] = model_index % 2 == 0  # Alternate between True and False
        
        return diverse_params
        
    def predict(self, num_draws=5, temperature=DEFAULT_TEMPERATURE, diversity_sampling=True):
        """Generate ensemble predictions by combining model outputs."""
        logger.info(f"Generating {num_draws} ensemble predictions")
        
        if not self.base_predictors:
            logger.error("No base predictors. Call train() first.")
            # Fallback to single predictor
            predictor = LotteryPredictionSystem(self.file_path, self.params)
            predictor.train_model()
            return predictor.predict(num_draws)
        
        try:
            # Get the first predictor as reference
            main_predictor = self.base_predictors[0]
            
            # Get latest data for prediction
            latest_features = main_predictor.X_scaled.iloc[-1:].values
            main_seq_latest = main_predictor.main_sequences[-1:]
            bonus_seq_latest = main_predictor.bonus_sequences[-1:]
            
            # Track used numbers for diversity
            used_main_numbers = set()
            used_bonus_numbers = set()
            
            # Generate predictions in batches for memory efficiency
            predictions = []
            batch_size = min(10, num_draws)  # Process in batches of 10 or fewer
            
            for batch_start in range(0, num_draws, batch_size):
                batch_end = min(batch_start + batch_size, num_draws)
                batch_draws = batch_end - batch_start
                
                # Get predictions from all models once per batch
                all_main_probs = []
                all_bonus_probs = []
                valid_models = 0
                
                for predictor in self.base_predictors:
                    try:
                        if predictor.model is None:
                            continue
                            
                        # Predict main and bonus numbers
                        main_probs = predictor.model.main_model.predict([latest_features, main_seq_latest])
                        bonus_probs = predictor.model.bonus_model.predict([latest_features, bonus_seq_latest])
                        
                        all_main_probs.append(main_probs)
                        all_bonus_probs.append(bonus_probs)
                        valid_models += 1
                    except Exception as e:
                        logger.warning(f"Error getting ensemble predictions from a model: {e}")
                        continue
                
                # Make sure we have at least one valid model prediction
                if valid_models == 0:
                    logger.error("No valid ensemble models for prediction")
                    return main_predictor.generate_fallback_predictions(num_draws)
                
                # Combine model predictions into averaged probability distributions
                for draw_idx in range(batch_draws):
                    # Initialize arrays for averaging
                    avg_main_probs = [np.zeros((1, MAIN_NUM_MAX)) for _ in range(MAIN_NUM_COUNT)]
                    avg_bonus_probs = [np.zeros((1, BONUS_NUM_MAX)) for _ in range(BONUS_NUM_COUNT)]
                    
                    # Sum probabilities from all models
                    for model_idx in range(valid_models):
                        model_main_probs = all_main_probs[model_idx]
                        model_bonus_probs = all_bonus_probs[model_idx]
                        
                        for i in range(MAIN_NUM_COUNT):
                            avg_main_probs[i] += model_main_probs[i]
                        
                        for i in range(BONUS_NUM_COUNT):
                            avg_bonus_probs[i] += model_bonus_probs[i]
                    
                    # Normalize by dividing by number of models
                    for i in range(MAIN_NUM_COUNT):
                        avg_main_probs[i] /= valid_models
                    
                    for i in range(BONUS_NUM_COUNT):
                        avg_bonus_probs[i] /= valid_models
                    
                    # Use centralized sampling function for consistency
                    global_draw_idx = batch_start + draw_idx
                    
                    main_numbers = Utilities.sample_numbers(
                        probs=avg_main_probs,
                        available_nums=range(MAIN_NUM_MIN, MAIN_NUM_MAX+1),
                        num_to_select=MAIN_NUM_COUNT,
                        used_nums=used_main_numbers,
                        diversity_sampling=diversity_sampling,
                        draw_idx=global_draw_idx,
                        temperature=temperature
                    )
                    
                    bonus_numbers = Utilities.sample_numbers(
                        probs=avg_bonus_probs,
                        available_nums=range(BONUS_NUM_MIN, BONUS_NUM_MAX+1),
                        num_to_select=BONUS_NUM_COUNT,
                        used_nums=used_bonus_numbers,
                        diversity_sampling=diversity_sampling,
                        draw_idx=global_draw_idx,
                        temperature=temperature
                    )
                    
                    # Calculate confidence with proper calibration
                    main_confidence = np.mean([np.max(avg_main_probs[i][0]) for i in range(MAIN_NUM_COUNT)])
                    bonus_confidence = np.mean([np.max(avg_bonus_probs[i][0]) for i in range(BONUS_NUM_COUNT)])
                    
                    # Calibrate confidence scores
                    calibrated_main_conf = MAIN_CONF_SCALE * main_confidence + MAIN_CONF_OFFSET
                    calibrated_bonus_conf = BONUS_CONF_SCALE * bonus_confidence + BONUS_CONF_OFFSET
                    
                    # Add pattern and frequency scores
                    pattern_score = Utilities.calculate_pattern_score(main_numbers, bonus_numbers)
                    frequency_score = Utilities.calculate_frequency_score(
                        main_numbers, bonus_numbers, main_predictor.processor.data
                    )
                    
                    # Overall confidence
                    overall_confidence = (calibrated_main_conf + calibrated_bonus_conf + pattern_score + frequency_score) / 4
                    
                    predictions.append({
                        "main_numbers": main_numbers,
                        "bonus_numbers": bonus_numbers,
                        "confidence": {
                            "overall": float(overall_confidence),
                            "main_numbers": float(calibrated_main_conf),
                            "bonus_numbers": float(calibrated_bonus_conf),
                            "pattern_score": float(pattern_score),
                            "frequency_score": float(frequency_score)
                        },
                        "method": "ensemble"
                    })
                
                # Clean memory at the end of each batch
                Utilities.clean_memory()
            
            return predictions
        except Exception as e:
            logger.error(f"Error in ensemble prediction: {str(e)}")
            
            # Simplified fallback cascade - go directly to main predictor
            if self._main_predictor:
                return self._main_predictor.predict(num_draws)
            elif self.base_predictors:
                return self.base_predictors[0].predict(num_draws)
            else:
                predictor = LotteryPredictionSystem(self.file_path, self.params)
                return predictor.generate_fallback_predictions(num_draws)

#######################
# HYPERPARAMETER OPTIMIZATION
#######################

class HyperparameterOptimizer:
    """Hyperparameter optimization using Bayesian methods with Optuna."""
    
    def __init__(self, file_path, n_trials=30):
        """Initialize the optimizer."""
        self.file_path = file_path
        self.n_trials = n_trials
        self.study = None
        self.best_params = None
        self.evaluator = None
        self.data_dict = None
    
    def objective(self, trial):
        """Objective function for hyperparameter optimization."""
        # Define parameter ranges
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 0.0001, 0.003, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
            'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
            'num_heads': trial.suggest_categorical('num_heads', [2, 4, 8]),
            'ff_dim': trial.suggest_categorical('ff_dim', [64, 128, 256]),
            'embed_dim': trial.suggest_categorical('embed_dim', [32, 64, 128]),
            'use_gru': trial.suggest_categorical('use_gru', [True, False]),
            'conv_filters': trial.suggest_int('conv_filters', 0, 64, step=16),
            'num_transformer_blocks': trial.suggest_int('num_transformer_blocks', 1, 3),
            'optimizer': trial.suggest_categorical('optimizer', ['adam', 'rmsprop'])
        }
        
        try:
            # Initialize evaluator if needed
            if self.evaluator is None:
                self.evaluator = CrossValidationEvaluator(self.file_path, params=None, folds=3)
                
                # Load data once and cache it for reuse
                if self.data_dict is None:
                    system = LotteryPredictionSystem(self.file_path)
                    self.data_dict = system.prepare_data()
                    self.evaluator.data_dict = self.data_dict
            
            # Update evaluator parameters for this trial
            self.evaluator.params = params
            
            # Run evaluation
            results = self.evaluator.evaluate()
            
            # Return the overall accuracy as the objective
            trial_accuracy = results['avg_overall_accuracy']
            
            # Clean up memory
            Utilities.clean_memory()
            
            return trial_accuracy
        except Exception as e:
            logger.error(f"Error in optimization trial: {e}")
            # Return a poor score for failed trials
            raise optuna.exceptions.TrialPruned()
    
    def optimize(self):
        """Run Bayesian optimization to find best hyperparameters."""
        logger.info(f"Starting hyperparameter optimization with {self.n_trials} trials")
        
        try:
            # Create Optuna study
            self.study = optuna.create_study(
                direction="maximize",
                sampler=TPESampler(seed=RANDOM_SEED),
                pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=5)
            )
            
            # Run optimization
            self.study.optimize(self.objective, n_trials=self.n_trials)
            
            # Get best parameters
            self.best_params = self.study.best_params
            logger.info(f"Optimization complete. Best accuracy: {self.study.best_value:.4f}")
            logger.info(f"Best parameters: {self.best_params}")
            
            return self.best_params
        except Exception as e:
            ErrorHandler.log_and_raise(logger, "Error in hyperparameter optimization", e)
            # Return default parameters as fallback
            return Utilities.get_default_params()
    
    def get_optimization_results(self):
        """Get detailed optimization results."""
        if self.study is None:
            ErrorHandler.log_and_raise(logger, "No optimization study. Call optimize() first.")
            return []
        
        try:
            # Get all trials
            trials = self.study.trials
            
            # Extract parameters and scores
            results = []
            for trial in trials:
                if trial.state == optuna.trial.TrialState.COMPLETE:
                    trial_params = trial.params.copy()
                    trial_params['value'] = trial.value
                    results.append(trial_params)
            
            # Sort by value (descending)
            results.sort(key=lambda x: x['value'], reverse=True)
            
            return results
        except Exception as e:
            ErrorHandler.log_and_raise(logger, "Error retrieving optimization results", e)
            return []
    
    def plot_optimization_history(self, filename="optimization_history.png"):
        """Plot optimization history."""
        if self.study is None:
            ErrorHandler.log_and_raise(logger, "No optimization study. Call optimize() first.")
            return None
        
        try:
            # Try to import visualization tools but handle if they're not available
            try:
                from optuna.visualization import plot_optimization_history, plot_param_importances
                has_optuna_viz = True
            except ImportError:
                logger.warning("Optuna visualization tools not available. Using basic plotting.")
                has_optuna_viz = False
            
            plt.figure(figsize=(16, 6))
            
            if has_optuna_viz:
                # Create figure with two subplots
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
                
                # Plot optimization history
                plot_optimization_history(self.study, ax=ax1)
                ax1.set_title("Optimization History")
                
                # Plot parameter importances
                plot_param_importances(self.study, ax=ax2)
                ax2.set_title("Parameter Importances")
            else:
                # Create basic plots without optuna visualizations
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
                
                # Get trial data
                trial_numbers = [t.number for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
                values = [t.value for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
                best_values = [self.study.best_value if i >= self.study.best_trial.number else float('nan') 
                              for i in trial_numbers]
                
                # Plot optimization history
                ax1.plot(trial_numbers, values, marker='o', markersize=4, linestyle='-', alpha=0.7)
                ax1.plot(trial_numbers, best_values, marker=None, linestyle='-', color='r')
                ax1.set_xlabel('Trial Number')
                ax1.set_ylabel('Objective Value')
                ax1.set_title('Optimization History')
                
                # Plot crude parameter importances (just show best parameters)
                params = list(self.best_params.keys())
                importances = list(range(len(params), 0, -1))  # Dummy importances
                ax2.barh(params, importances)
                ax2.set_xlabel('Importance')
                ax2.set_title('Best Parameters')
                for i, v in enumerate(importances):
                    ax2.text(v, i, f"{self.best_params[params[i]]}")
            
            plt.tight_layout()
            plt.savefig(filename, dpi=300)
            plt.close()
            
            logger.info(f"Optimization plots saved to {filename}")
            return filename
        except Exception as e:
            ErrorHandler.log_and_raise(logger, f"Error plotting optimization history: {e}")
            return None

#######################
# MAIN FUNCTION
#######################

def main():
    """Main function to run the lottery prediction system."""
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

    args = parser.parse_args()
    
    try:
        # Print header
        print("\n" + "="*80)
        print("OPTIMIZED TRANSFORMER-BASED EUROMILLIONS LOTTERY PREDICTION SYSTEM".center(80))
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
        
        # Load parameters
        params = Utilities.load_params(args.params)
        if params != Utilities.get_default_params():
            print(f"Loaded parameters from {args.params}")
        
        # Create a data processor to be shared across components to avoid duplication
        shared_processor = LotteryDataProcessor(args.file)
        data = shared_processor.parse_file()
        
        # Handle optimization if requested
        if args.optimize:
            print(f"Running hyperparameter optimization with {args.trials} trials...")
            try:
                optimizer = HyperparameterOptimizer(args.file, args.trials)
                params = optimizer.optimize()
                Utilities.save_params(params, args.params)
                print(f"Optimization complete. Best parameters saved to {args.params}")
                
                # Generate optimization plots
                plot_file = optimizer.plot_optimization_history()
                if plot_file:
                    print(f"Optimization history plot saved to {plot_file}")
            except Exception as e:
                print(f"Optimization error: {e}")
                print("Using default or previously loaded parameters")
                
            # Clean memory after optimization
            Utilities.clean_memory(force=True)
        
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
            
            # Clean memory
            Utilities.clean_memory(force=True)
        
        # Create prediction system
        predictor = None
        
        # Generate predictions
        predictions = None
        if args.ensemble:
            print(f"\nTraining ensemble with {args.num_models} base models...")
            ensemble = EnsemblePredictor(args.file, args.num_models, params)
            ensemble.processor = shared_processor  # Share the processor
            ensemble.train()
            
            print(f"\nGenerating {args.predictions} ensemble predictions...")
            predictions = ensemble.predict(args.predictions)
            predictor = ensemble
        else:
            print("\nTraining transformer model...")
            system = LotteryPredictionSystem(args.file, params)
            system.processor = shared_processor  # Share the processor
            system.train_model()
            
            print(f"\nGenerating {args.predictions} predictions...")
            predictions = system.predict(args.predictions)
            predictor = system
        
        # Check if we got valid predictions
        if not predictions or len(predictions) == 0:
            print("\nError: Failed to generate predictions.")
            sys.exit(1)
        
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
        
        # Generate visualizations
        print("\nGenerating visualizations...")
        viz_files = generate_visualizations(predictions, args.file)
        if viz_files:
            print(f"Visualizations saved to: {', '.join(viz_files)}")
        
        # Print summary
        try:
            # Count occurrences of each number in the predictions
            main_counts = {}
            bonus_counts = {}
            
            for pred in predictions:
                for num in pred["main_numbers"]:
                    main_counts[num] = main_counts.get(num, 0) + 1
                
                for num in pred["bonus_numbers"]:
                    bonus_counts[num] = bonus_counts.get(num, 0) + 1
            
            print("\nPrediction Summary:")
            print("=================")
            
            print("\nMost frequent main numbers in predictions:")
            for num, count in sorted(main_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"Number {num}: appeared {count} times")
            
            print("\nMost frequent bonus numbers in predictions:")
            for num, count in sorted(bonus_counts.items(), key=lambda x: x[1], reverse=True)[:3]:
                print(f"Number {num}: appeared {count} times")
            
            # Calculate average confidence
            avg_confidence = np.mean([pred["confidence"]["overall"] * 100 for pred in predictions])
            print(f"\nAverage prediction confidence: {avg_confidence:.2f}%")
            
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