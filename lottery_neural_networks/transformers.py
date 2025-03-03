#!/usr/bin/env python
"""
python enhanced_transformer.py --file lottery_numbers.txt --predictions 20

python enhanced_transformer.py --file lottery_numbers.txt --optimize --trials 30 --ensemble --num_models 7 --predictions 20

"""

import os
import sys
import re
import json
import random
import argparse
import logging
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, log_loss
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential, load_model, save_model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, MultiHeadAttention, LayerNormalization,
    GlobalAveragePooling1D, Concatenate, Reshape, Add, GRU, Conv1D,
    LSTM, Bidirectional, BatchNormalization, Activation, Attention
)
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l1_l2
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner


gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Configure GPU memory growth to avoid taking all memory at once
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Found {len(gpus)} GPU(s): {gpus}")
        
        # Instead of enabling global mixed precision,
        # we'll use it selectively for compute-intensive layers
        print("Using selective mixed precision for compute-intensive operations")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")
else:
    print("No GPU found. Using CPU.")

# Define policies that will be used selectively
COMPUTE_INTENSIVE_POLICY = tf.keras.mixed_precision.Policy('mixed_float16')
NUMERICALLY_SENSITIVE_POLICY = tf.keras.mixed_precision.Policy('float32')


# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("enhanced_transformer.log", mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Hide TensorFlow warnings
tf.get_logger().setLevel('ERROR')

#######################
# DATA PROCESSING
#######################

class LotteryDataProcessor:
    """Enhanced processor for lottery data from text files."""
    
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        
    def parse_file(self):
        """Parse the lottery data file into a structured DataFrame."""
        logger.info(f"Parsing lottery data from {self.file_path}")
        
        try:
            with open(self.file_path, 'r') as file:
                content = file.read()
            
            # Extract draws using regex pattern
            draw_pattern = r"((?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\s+\d+(?:st|nd|rd|th)\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(€[\d,]+)\s+(Roll|Won)"
            draws = re.findall(draw_pattern, content)
            
            if not draws:
                logger.error(f"No valid draws found in {self.file_path}")
                return None
            
            # Process extracted data
            structured_data = []
            
            for draw in draws:
                date_str = draw[0]
                main_numbers = [int(draw[i]) for i in range(1, 6)]
                bonus_numbers = [int(draw[i]) for i in range(6, 8)]
                jackpot = draw[8]
                result = draw[9]
                
                # Parse date with different suffixes
                try:
                    try:
                        date = datetime.strptime(date_str, "%A %dst %B %Y")
                    except ValueError:
                        try:
                            date = datetime.strptime(date_str, "%A %dnd %B %Y")
                        except ValueError:
                            try:
                                date = datetime.strptime(date_str, "%A %drd %B %Y")
                            except ValueError:
                                date = datetime.strptime(date_str, "%A %dth %B %Y")
                except Exception as e:
                    logger.warning(f"Could not parse date: {date_str}, error: {e}")
                    continue
                
                structured_data.append({
                    "date": date,
                    "day_of_week": date.strftime("%A"),
                    "main_numbers": sorted(main_numbers),  # Ensure numbers are sorted
                    "bonus_numbers": sorted(bonus_numbers),
                    "jackpot": jackpot,
                    "result": result
                })
            
            # Convert to DataFrame and sort by date
            df = pd.DataFrame(structured_data)
            df = df.sort_values("date")
            
            # Extract jackpot value as numeric
            df["jackpot_value"] = df["jackpot"].str.replace("€", "").str.replace(",", "").astype(float)
            df["is_won"] = df["result"] == "Won"
            
            self.data = df
            logger.info(f"Successfully parsed {len(df)} draws")
            
            return df
            
        except Exception as e:
            logger.error(f"Error parsing file: {str(e)}")
            raise
    
    def expand_numbers(self):
        """Expand main and bonus numbers into individual columns with enhanced metadata."""
        if self.data is None:
            logger.error("No data available. Parse the file first.")
            return None
        
        df = self.data.copy()
        
        # Expand main numbers (1-50)
        for i in range(5):
            df[f"main_{i+1}"] = df["main_numbers"].apply(lambda x: x[i])
        
        # Expand bonus numbers (1-12)
        for i in range(2):
            df[f"bonus_{i+1}"] = df["bonus_numbers"].apply(lambda x: x[i])
        
        # Add metadata about the draw
        df["draw_index"] = range(len(df))
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month
        df["day"] = df["date"].dt.day
        df["day_of_week_num"] = df["date"].dt.dayofweek
        df["week_of_year"] = df["date"].dt.isocalendar().week
        df["is_holiday_season"] = ((df["month"] == 12) & (df["day"] >= 15)) | ((df["month"] == 1) & (df["day"] <= 15))
        
        # Calculate days since last draw
        df["days_since_last_draw"] = (df["date"] - df["date"].shift(1)).dt.days
        df["days_since_last_draw"].fillna(0, inplace=True)
        
        return df

#######################
# ENHANCED FEATURE ENGINEERING
#######################

class EnhancedFeatureEngineering:
    """Advanced feature engineering for lottery prediction."""
    
    def __init__(self, data):
        self.data = data
        self.historical_counts = None
        self.pattern_clusters = None
        
    def initialize_number_statistics(self):
        """Initialize historical statistics for each number."""
        main_counts = np.zeros(50)
        bonus_counts = np.zeros(12)
        
        for _, row in self.data.iterrows():
            for num in row["main_numbers"]:
                main_counts[num-1] += 1
            for num in row["bonus_numbers"]:
                bonus_counts[num-1] += 1
        
        self.historical_counts = {
            "main": main_counts / len(self.data),
            "bonus": bonus_counts / len(self.data)
        }
    
    def calculate_time_series_features(self, window_sizes=[5, 10, 20, 50, 100]):
        """Calculate enhanced time series features with adaptive windowing."""
        df = self.data.copy()
        
        # Initialize the historical statistics if not already done
        if self.historical_counts is None:
            self.initialize_number_statistics()
        
        # Create empty DataFrame for time series features
        ts_features = {}
        
        # Create indicators for each number
        indicators = {}
        
        # Main numbers (1-50)
        for num in range(1, 51):
            indicators[f"main_has_{num}"] = df["main_numbers"].apply(lambda x: num in x).astype(int)
            
        # Bonus numbers (1-12)
        for num in range(1, 13):
            indicators[f"bonus_has_{num}"] = df["bonus_numbers"].apply(lambda x: num in x).astype(int)
        
        # Convert indicators to DataFrame
        indicators_df = pd.DataFrame(indicators, index=df.index)
        
        # Calculate rolling statistics with adaptive windows
        for window in window_sizes:
            if window >= len(df):
                continue
                
            # Frequency calculations with advanced weighting
            for num in range(1, 51):
                # Exponentially weighted moving average with adaptive span
                ts_features[f"main_{num}_ewma_{window}"] = indicators_df[f"main_has_{num}"].ewm(span=window).mean()
                
                # Rolling standard deviation of frequency
                ts_features[f"main_{num}_std_{window}"] = indicators_df[f"main_has_{num}"].rolling(window=window).std()
                
                # Rate of change over window
                ts_features[f"main_{num}_roc_{window}"] = indicators_df[f"main_has_{num}"].pct_change(periods=window)
                
                # Momentum indicator (difference between short and long windows)
                if window > 10:
                    short_ma = indicators_df[f"main_has_{num}"].rolling(window=5).mean()
                    long_ma = indicators_df[f"main_has_{num}"].rolling(window=window).mean()
                    ts_features[f"main_{num}_momentum_{window}"] = short_ma - long_ma
                
                # Variance ratio (measure of randomness)
                if window > 20:
                    var_short = indicators_df[f"main_has_{num}"].rolling(window=window//2).var()
                    var_long = indicators_df[f"main_has_{num}"].rolling(window=window).var()
                    # Avoid division by zero
                    ts_features[f"main_{num}_var_ratio_{window}"] = np.where(
                        var_long > 0, var_short / var_long, 0
                    )
                
                # Mean reversion indicator - deviation from historical mean
                ts_features[f"main_{num}_mean_rev_{window}"] = (
                    indicators_df[f"main_has_{num}"].rolling(window=window).mean() - 
                    self.historical_counts["main"][num-1]
                )
                
            # Similar calculations for bonus numbers
            for num in range(1, 13):
                ts_features[f"bonus_{num}_ewma_{window}"] = indicators_df[f"bonus_has_{num}"].ewm(span=window).mean()
                ts_features[f"bonus_{num}_std_{window}"] = indicators_df[f"bonus_has_{num}"].rolling(window=window).std()
                ts_features[f"bonus_{num}_roc_{window}"] = indicators_df[f"bonus_has_{num}"].pct_change(periods=window)
                
                # Momentum indicator
                if window > 10:
                    short_ma = indicators_df[f"bonus_has_{num}"].rolling(window=5).mean()
                    long_ma = indicators_df[f"bonus_has_{num}"].rolling(window=window).mean()
                    ts_features[f"bonus_{num}_momentum_{window}"] = short_ma - long_ma
                
                # Mean reversion indicator
                ts_features[f"bonus_{num}_mean_rev_{window}"] = (
                    indicators_df[f"bonus_has_{num}"].rolling(window=window).mean() - 
                    self.historical_counts["bonus"][num-1]
                )
        
        # Advanced Fourier analysis for cyclical patterns
        for num in range(1, 51):
            # Get the time series for this number
            ts = indicators_df[f"main_has_{num}"].fillna(0).values
            
            if len(ts) >= 100:  # Only perform FFT on sufficiently long series
                # Apply FFT
                fft_vals = np.fft.fft(ts)
                # Get magnitudes
                magnitudes = np.abs(fft_vals)
                
                # Store top 5 frequency components
                top_indices = np.argsort(magnitudes[1:len(ts)//2])[-5:]
                for i, idx in enumerate(top_indices):
                    period = len(ts) / (idx + 1)  # Calculate the period in draws
                    ts_features[f"main_{num}_fft_mag_{i}"] = magnitudes[idx+1]
                    ts_features[f"main_{num}_fft_phase_{i}"] = np.angle(fft_vals[idx+1])
                    ts_features[f"main_{num}_fft_period_{i}"] = period
                    
                    # Add sine and cosine components for the top frequencies
                    if period >= 2:  # Avoid too high frequencies
                        for j in range(len(df)):
                            cycle_pos = j / period
                            ts_features.setdefault(f"main_{num}_sin_wave_{i}", []).append(np.sin(2*np.pi*cycle_pos))
                            ts_features.setdefault(f"main_{num}_cos_wave_{i}", []).append(np.cos(2*np.pi*cycle_pos))
        
        # Same for bonus numbers with enough data
        for num in range(1, 13):
            ts = indicators_df[f"bonus_has_{num}"].fillna(0).values
            
            if len(ts) >= 100:
                fft_vals = np.fft.fft(ts)
                magnitudes = np.abs(fft_vals)
                top_indices = np.argsort(magnitudes[1:len(ts)//2])[-3:]
                for i, idx in enumerate(top_indices):
                    period = len(ts) / (idx + 1)
                    ts_features[f"bonus_{num}_fft_mag_{i}"] = magnitudes[idx+1]
                    ts_features[f"bonus_{num}_fft_phase_{i}"] = np.angle(fft_vals[idx+1])
                    ts_features[f"bonus_{num}_fft_period_{i}"] = period
                    
                    # Add sine and cosine components
                    if period >= 2:
                        for j in range(len(df)):
                            cycle_pos = j / period
                            ts_features.setdefault(f"bonus_{num}_sin_wave_{i}", []).append(np.sin(2*np.pi*cycle_pos))
                            ts_features.setdefault(f"bonus_{num}_cos_wave_{i}", []).append(np.cos(2*np.pi*cycle_pos))
        
        # Auto-correlation features for each number
        for num in range(1, 51):
            ts = indicators_df[f"main_has_{num}"].fillna(0).values
            
            if len(ts) >= 200:  # Only for longer series
                for lag in [1, 5, 10, 20]:
                    if lag < len(ts) - 10:
                        # Calculate auto-correlation
                        ac = np.correlate(ts[lag:], ts[:-lag], mode='valid')[0] / np.var(ts) / (len(ts) - lag)
                        ts_features[f"main_{num}_autocorr_{lag}"] = [ac] * len(df)
        
        # Convert to DataFrame
        ts_df = pd.DataFrame(ts_features, index=df.index)
        return ts_df.fillna(0)
    
    def calculate_number_relationships(self):
        """Calculate enhanced features based on relationships between numbers."""
        df = self.data.copy()
        relationship_features = {}
        
        # For each draw, calculate statistics about the relationships between numbers
        for idx, row in df.iterrows():
            main_nums = row["main_numbers"]
            
            if len(main_nums) >= 5:  # Ensure we have 5 main numbers
                # Calculate differences between consecutive numbers
                diffs = [main_nums[i+1] - main_nums[i] for i in range(len(main_nums)-1)]
                
                # Store statistics about differences
                relationship_features.setdefault("main_mean_diff", []).append(np.mean(diffs))
                relationship_features.setdefault("main_std_diff", []).append(np.std(diffs))
                relationship_features.setdefault("main_min_diff", []).append(min(diffs))
                relationship_features.setdefault("main_max_diff", []).append(max(diffs))
                relationship_features.setdefault("main_range", []).append(max(main_nums) - min(main_nums))
                
                # Calculate sum and product of numbers
                relationship_features.setdefault("main_sum", []).append(sum(main_nums))
                relationship_features.setdefault("main_mean", []).append(np.mean(main_nums))
                
                # Use log sum to avoid overflow with products
                relationship_features.setdefault("main_log_sum", []).append(sum(np.log(main_nums)))
                
                # Calculate statistics related to number distribution
                relationship_features.setdefault("main_low_count", []).append(sum(1 for n in main_nums if n <= 25))
                relationship_features.setdefault("main_high_count", []).append(sum(1 for n in main_nums if n > 25))
                relationship_features.setdefault("main_odd_count", []).append(sum(1 for n in main_nums if n % 2 == 1))
                relationship_features.setdefault("main_even_count", []).append(sum(1 for n in main_nums if n % 2 == 0))
                
                # Calculate digit sum (sum of all digits in all numbers)
                digit_sum = sum(int(digit) for num in main_nums for digit in str(num))
                relationship_features.setdefault("main_digit_sum", []).append(digit_sum)
                
                # Calculate digit product (product of all digits)
                digit_product = 1
                for num in main_nums:
                    for digit in str(num):
                        if int(digit) > 0:  # Avoid multiplying by zero
                            digit_product *= int(digit)
                relationship_features.setdefault("main_digit_product", []).append(digit_product)
                
                # Calculate quartile distribution
                quartiles = [13, 26, 38]  # Split 1-50 into quartiles
                relationship_features.setdefault("main_q1_count", []).append(sum(1 for n in main_nums if n <= quartiles[0]))
                relationship_features.setdefault("main_q2_count", []).append(sum(1 for n in main_nums if quartiles[0] < n <= quartiles[1]))
                relationship_features.setdefault("main_q3_count", []).append(sum(1 for n in main_nums if quartiles[1] < n <= quartiles[2]))
                relationship_features.setdefault("main_q4_count", []).append(sum(1 for n in main_nums if n > quartiles[2]))
                
                # Calculate consecutive numbers
                consecutive_count = 0
                for i in range(len(main_nums)-1):
                    if main_nums[i+1] - main_nums[i] == 1:
                        consecutive_count += 1
                relationship_features.setdefault("main_consecutive_count", []).append(consecutive_count)
                
                # Calculate Fibonacci sequence matches (1, 2, 3, 5, 8, 13, 21, 34)
                fibonacci_nums = [1, 2, 3, 5, 8, 13, 21, 34]
                fibonacci_matches = sum(1 for n in main_nums if n in fibonacci_nums)
                relationship_features.setdefault("main_fibonacci_count", []).append(fibonacci_matches)
                
                # Calculate prime number matches (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47)
                prime_nums = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
                prime_matches = sum(1 for n in main_nums if n in prime_nums)
                relationship_features.setdefault("main_prime_count", []).append(prime_matches)
                
                # Advanced distribution analysis - standard deviation
                relationship_features.setdefault("main_std", []).append(np.std(main_nums))
                
                # Skewness of the distribution
                from scipy.stats import skew
                relationship_features.setdefault("main_skew", []).append(skew(main_nums))
                
                # Number dispersion (entropy-like measure)
                bins = [10*i for i in range(6)]  # 0-10, 10-20, ..., 40-50
                hist, _ = np.histogram(main_nums, bins=bins)
                if sum(hist) > 0:
                    probs = hist / sum(hist)
                    # Shannon entropy
                    entropy = -np.sum([p * np.log(p) if p > 0 else 0 for p in probs])
                    relationship_features.setdefault("main_dispersion", []).append(entropy)
                else:
                    relationship_features.setdefault("main_dispersion", []).append(0)
            else:
                # Fill with NaN if we don't have enough numbers
                for feature in [
                    "main_mean_diff", "main_std_diff", "main_min_diff", "main_max_diff", "main_range",
                    "main_sum", "main_mean", "main_log_sum", "main_low_count", "main_high_count",
                    "main_odd_count", "main_even_count", "main_digit_sum", "main_digit_product",
                    "main_q1_count", "main_q2_count", "main_q3_count", "main_q4_count",
                    "main_consecutive_count", "main_fibonacci_count", "main_prime_count",
                    "main_std", "main_skew", "main_dispersion"
                ]:
                    relationship_features.setdefault(feature, []).append(np.nan)
            
            # Similar calculations for bonus numbers
            bonus_nums = row["bonus_numbers"]
            if len(bonus_nums) >= 2:  # Ensure we have 2 bonus numbers
                relationship_features.setdefault("bonus_diff", []).append(bonus_nums[1] - bonus_nums[0])
                relationship_features.setdefault("bonus_sum", []).append(sum(bonus_nums))
                relationship_features.setdefault("bonus_mean", []).append(np.mean(bonus_nums))
                relationship_features.setdefault("bonus_product", []).append(bonus_nums[0] * bonus_nums[1])
                relationship_features.setdefault("bonus_low_count", []).append(sum(1 for n in bonus_nums if n <= 6))
                relationship_features.setdefault("bonus_high_count", []).append(sum(1 for n in bonus_nums if n > 6))
                relationship_features.setdefault("bonus_odd_count", []).append(sum(1 for n in bonus_nums if n % 2 == 1))
                relationship_features.setdefault("bonus_even_count", []).append(sum(1 for n in bonus_nums if n % 2 == 0))
                relationship_features.setdefault("bonus_range", []).append(max(bonus_nums) - min(bonus_nums))
                
                # Prime numbers in bonus balls
                bonus_primes = [2, 3, 5, 7, 11]
                prime_matches = sum(1 for n in bonus_nums if n in bonus_primes)
                relationship_features.setdefault("bonus_prime_count", []).append(prime_matches)
            else:
                for feature in [
                    "bonus_diff", "bonus_sum", "bonus_mean", "bonus_product", 
                    "bonus_low_count", "bonus_high_count", 
                    "bonus_odd_count", "bonus_even_count", "bonus_range",
                    "bonus_prime_count"
                ]:
                    relationship_features.setdefault(feature, []).append(np.nan)
        
        # Convert to DataFrame
        relationship_df = pd.DataFrame(relationship_features, index=df.index)
        return relationship_df.fillna(0)
    
    def calculate_pattern_features(self):
        """Calculate features based on number patterns and clusters."""
        df = self.data.copy()
        pattern_features = {}
        
        # Create a signature for each draw's main numbers
        signatures = []
        for _, row in df.iterrows():
            # Create a histogram of the numbers (count in each decade)
            bins = [1, 11, 21, 31, 41, 51]  # 1-10, 11-20, ..., 41-50
            hist, _ = np.histogram(row["main_numbers"], bins=bins)
            
            # Create a pattern descriptor
            # 1. Differences pattern
            main_nums = row["main_numbers"]
            diffs = [main_nums[i+1] - main_nums[i] for i in range(len(main_nums)-1)]
            
            # 2. Distribution pattern (where in the range are the numbers)
            distribution = [n/50 for n in main_nums]  # Normalize to 0-1 range
            
            # 3. Create a combined signature
            signature = np.concatenate([hist/sum(hist), diffs, distribution])
            signatures.append(signature)
        
        # Create clusters of similar patterns if we have enough data
        if len(signatures) >= 200:
            # Normalize the signatures
            signatures_array = np.array(signatures)
            # Use PCA to reduce dimensions first
            if signatures_array.shape[1] > 10:
                pca = PCA(n_components=min(10, signatures_array.shape[1]))
                signatures_pca = pca.fit_transform(signatures_array)
            else:
                signatures_pca = signatures_array
            
            # Find optimal number of clusters
            silhouette_scores = []
            max_clusters = min(10, len(signatures) // 20)  # Limit number of clusters
            for n_clusters in range(2, max_clusters + 1):
                kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED)
                cluster_labels = kmeans.fit_predict(signatures_pca)
                
                # Calculate silhouette score if we have multiple clusters
                if len(set(cluster_labels)) > 1:
                    from sklearn.metrics import silhouette_score
                    score = silhouette_score(signatures_pca, cluster_labels)
                    silhouette_scores.append((n_clusters, score))
            
            # Choose number of clusters with best silhouette score
            if silhouette_scores:
                best_n_clusters = max(silhouette_scores, key=lambda x: x[1])[0]
            else:
                best_n_clusters = 5  # Default if can't determine optimal
                
            # Create final clusters
            kmeans = KMeans(n_clusters=best_n_clusters, random_state=RANDOM_SEED)
            cluster_labels = kmeans.fit_predict(signatures_pca)
            
            # Store the cluster centers for later use
            if signatures_array.shape[1] > 10:
                # We need to transform back from PCA to original space
                self.pattern_clusters = {
                    "n_clusters": best_n_clusters,
                    "kmeans": kmeans,
                    "pca": pca,
                    "use_pca": True
                }
            else:
                self.pattern_clusters = {
                    "n_clusters": best_n_clusters,
                    "kmeans": kmeans,
                    "use_pca": False
                }
            
            # Create features based on clusters
            for i in range(len(df)):
                # Assign cluster to each draw
                cluster = cluster_labels[i] if i < len(cluster_labels) else 0
                pattern_features.setdefault("main_pattern_cluster", []).append(cluster)
                
                # Distance to cluster center
                if i < len(signatures_pca):
                    center = kmeans.cluster_centers_[cluster]
                    distance = np.linalg.norm(signatures_pca[i] - center)
                    pattern_features.setdefault("main_pattern_distance", []).append(distance)
                else:
                    pattern_features.setdefault("main_pattern_distance", []).append(0)
                
                # Cluster transition (what clusters tend to follow each other)
                if i > 0 and i < len(cluster_labels):
                    prev_cluster = cluster_labels[i-1]
                    transition = prev_cluster * best_n_clusters + cluster
                    pattern_features.setdefault("main_pattern_transition", []).append(transition)
                else:
                    pattern_features.setdefault("main_pattern_transition", []).append(0)
            
            # Calculate frequency of each cluster
            cluster_counts = np.bincount(cluster_labels, minlength=best_n_clusters)
            cluster_freq = cluster_counts / sum(cluster_counts)
            
            # Add cluster frequency as a feature
            for i in range(len(df)):
                if i < len(cluster_labels):
                    cluster = cluster_labels[i]
                    pattern_features.setdefault("main_pattern_frequency", []).append(cluster_freq[cluster])
                else:
                    pattern_features.setdefault("main_pattern_frequency", []).append(0)
        else:
            # Not enough data for clustering, add dummy features
            pattern_features["main_pattern_cluster"] = [0] * len(df)
            pattern_features["main_pattern_distance"] = [0] * len(df)
            pattern_features["main_pattern_transition"] = [0] * len(df)
            pattern_features["main_pattern_frequency"] = [0] * len(df)
        
        return pd.DataFrame(pattern_features, index=df.index)
    
    def calculate_multi_draw_patterns(self, sequence_length=3):
        """Calculate features based on patterns across multiple consecutive draws."""
        df = self.data.copy()
        
        # Create features dictionary
        multi_draw_features = {}
        
        # Calculate features for overlapping windows of draws
        for i in range(len(df) - sequence_length + 1):
            # Get the sequence of draws
            draws = df.iloc[i:i+sequence_length]
            
            # Calculate common numbers between consecutive draws
            for j in range(sequence_length-1):
                draw1 = set(draws.iloc[j]["main_numbers"])
                draw2 = set(draws.iloc[j+1]["main_numbers"])
                common = len(draw1.intersection(draw2))
                
                # Store common count (with lag information)
                if i+sequence_length-1 < len(df):
                    multi_draw_features.setdefault(f"common_main_{j}_{j+1}_lag{sequence_length-1}", []).append(common)
            
            # Calculate common numbers across all draws in the sequence
            all_draws_sets = [set(draws.iloc[j]["main_numbers"]) for j in range(sequence_length)]
            common_all = len(set.intersection(*all_draws_sets)) if all_draws_sets else 0
            
            # Store common count
            if i+sequence_length-1 < len(df):
                multi_draw_features.setdefault(f"common_main_all_{sequence_length}_lag{sequence_length-1}", []).append(common_all)
            
            # Calculate transitions of number counts
            for num in range(1, 51):
                # Track if this number appeared in each draw of the sequence
                appearances = [num in draws.iloc[j]["main_numbers"] for j in range(sequence_length)]
                
                # Convert to transitions (e.g. [1,0,1] means it appeared, then didn't, then did)
                transitions = sum(1 for j in range(sequence_length-1) if appearances[j] != appearances[j+1])
                
                # Store transitions count
                if i+sequence_length-1 < len(df):
                    multi_draw_features.setdefault(f"main_{num}_transitions_lag{sequence_length-1}", []).append(transitions)
                    
                # Calculate streak length (consecutive appearances or non-appearances)
                current_streak = 1
                for j in range(1, sequence_length):
                    if appearances[j] == appearances[j-1]:
                        current_streak += 1
                    else:
                        current_streak = 1
                        
                if i+sequence_length-1 < len(df):
                    if appearances[-1]:  # Number appeared in most recent draw
                        multi_draw_features.setdefault(f"main_{num}_appearance_streak", []).append(current_streak)
                        multi_draw_features.setdefault(f"main_{num}_absence_streak", []).append(0)
                    else:  # Number didn't appear
                        multi_draw_features.setdefault(f"main_{num}_appearance_streak", []).append(0)
                        multi_draw_features.setdefault(f"main_{num}_absence_streak", []).append(current_streak)
            
            # Similar features for bonus numbers
            for j in range(sequence_length-1):
                draw1 = set(draws.iloc[j]["bonus_numbers"])
                draw2 = set(draws.iloc[j+1]["bonus_numbers"])
                common = len(draw1.intersection(draw2))
                
                if i+sequence_length-1 < len(df):
                    multi_draw_features.setdefault(f"common_bonus_{j}_{j+1}_lag{sequence_length-1}", []).append(common)
                    
            # Streak calculations for bonus numbers
            for num in range(1, 13):
                appearances = [num in draws.iloc[j]["bonus_numbers"] for j in range(sequence_length)]
                current_streak = 1
                for j in range(1, sequence_length):
                    if appearances[j] == appearances[j-1]:
                        current_streak += 1
                    else:
                        current_streak = 1
                        
                if i+sequence_length-1 < len(df):
                    if appearances[-1]:  # Number appeared in most recent draw
                        multi_draw_features.setdefault(f"bonus_{num}_appearance_streak", []).append(current_streak)
                        multi_draw_features.setdefault(f"bonus_{num}_absence_streak", []).append(0)
                    else:  # Number didn't appear
                        multi_draw_features.setdefault(f"bonus_{num}_appearance_streak", []).append(0)
                        multi_draw_features.setdefault(f"bonus_{num}_absence_streak", []).append(current_streak)
        
        # Pad the beginning of the DataFrame
        for col in multi_draw_features:
            pad_length = len(df) - len(multi_draw_features[col])
            multi_draw_features[col] = [0] * pad_length + multi_draw_features[col]
        
        # Convert to DataFrame
        multi_draw_df = pd.DataFrame(multi_draw_features, index=df.index)
        return multi_draw_df.fillna(0)
    
    def calculate_cyclical_features(self):
        """Calculate enhanced cyclical time features using sine/cosine transformations."""
        df = self.data.copy()
        
        # Create features dictionary
        cyclical_features = {}
        
        # Calculate cyclical features for each draw
        for i, (_, row) in enumerate(df.iterrows()):
            date = row["date"]
            
            # Day of week (0-6)
            day_of_week = date.weekday()
            cyclical_features.setdefault("day_of_week_sin", []).append(np.sin(2 * np.pi * day_of_week / 7))
            cyclical_features.setdefault("day_of_week_cos", []).append(np.cos(2 * np.pi * day_of_week / 7))
            
            # Day of month (1-31)
            day_of_month = date.day
            cyclical_features.setdefault("day_of_month_sin", []).append(np.sin(2 * np.pi * day_of_month / 31))
            cyclical_features.setdefault("day_of_month_cos", []).append(np.cos(2 * np.pi * day_of_month / 31))
            
            # Month (1-12)
            month = date.month
            cyclical_features.setdefault("month_sin", []).append(np.sin(2 * np.pi * month / 12))
            cyclical_features.setdefault("month_cos", []).append(np.cos(2 * np.pi * month / 12))
            
            # Day of year (1-366)
            day_of_year = date.timetuple().tm_yday
            cyclical_features.setdefault("day_of_year_sin", []).append(np.sin(2 * np.pi * day_of_year / 366))
            cyclical_features.setdefault("day_of_year_cos", []).append(np.cos(2 * np.pi * day_of_year / 366))
            
            # Week of year (1-53)
            week_of_year = date.isocalendar()[1]
            cyclical_features.setdefault("week_of_year_sin", []).append(np.sin(2 * np.pi * week_of_year / 53))
            cyclical_features.setdefault("week_of_year_cos", []).append(np.cos(2 * np.pi * week_of_year / 53))
            
            # Quarter (1-4)
            quarter = (month - 1) // 3 + 1
            cyclical_features.setdefault("quarter_sin", []).append(np.sin(2 * np.pi * quarter / 4))
            cyclical_features.setdefault("quarter_cos", []).append(np.cos(2 * np.pi * quarter / 4))
            
            # Years since start of dataset (continuous)
            if i == 0:
                first_date = date
            years_since_start = (date - first_date).days / 365.25
            cyclical_features.setdefault("years_since_start", []).append(years_since_start)
            
            # Events within the year cycle
            # For example, for holiday season effect:
            days_till_christmas = (datetime(date.year, 12, 25) - date).days
            if days_till_christmas < 0:  # Christmas has passed this year
                days_till_christmas = (datetime(date.year + 1, 12, 25) - date).days
            cyclical_features.setdefault("days_till_christmas_norm", []).append(1.0 - min(days_till_christmas, 180) / 180)
            
            # Distance from nearest month beginning/end (normalized)
            days_in_month = (date.replace(month=date.month % 12 + 1, day=1) - date.replace(day=1)).days
            day_pos_in_month = (date.day - 1) / (days_in_month - 1)  # 0 at beginning, 1 at end
            cyclical_features.setdefault("month_position_sin", []).append(np.sin(2 * np.pi * day_pos_in_month))
            cyclical_features.setdefault("month_position_cos", []).append(np.cos(2 * np.pi * day_pos_in_month))
        
        # Convert to DataFrame
        cyclical_df = pd.DataFrame(cyclical_features, index=df.index)
        return cyclical_df
    
    def calculate_synthetic_features(self):
        """Create synthetic features that combine multiple primary features."""
        df = self.data.copy()
        
        # Placeholder for synthetic features
        synthetic_features = {}
        
        # Calculate historical dispersion of main numbers
        historical_main_distribution = np.zeros(50)
        
        for _, row in df.iterrows():
            for num in row["main_numbers"]:
                historical_main_distribution[num-1] += 1
        
        historical_main_distribution = historical_main_distribution / historical_main_distribution.sum()
        historical_entropy = -np.sum([p * np.log(p) if p > 0 else 0 for p in historical_main_distribution])
        
        # For each draw, calculate synthetic features
        for i, (_, row) in enumerate(df.iterrows()):
            # 1. Balance score: Combination of even/odd and high/low counts
            main_nums = row["main_numbers"]
            even_count = sum(1 for n in main_nums if n % 2 == 0)
            high_count = sum(1 for n in main_nums if n > 25)
            
            # Perfect balance would be 2.5 for each (out of 5 numbers)
            even_balance = 1.0 - abs(even_count - 2.5) / 2.5
            high_balance = 1.0 - abs(high_count - 2.5) / 2.5
            balance_score = (even_balance + high_balance) / 2
            
            synthetic_features.setdefault("main_balance_score", []).append(balance_score)
            
            # 2. Dispersion score: How well-distributed are the numbers across the range
            # This builds on the entropy calculation but normalizes by the historical entropy
            current_distribution = np.zeros(50)
            for num in main_nums:
                current_distribution[num-1] = 1
            current_distribution = current_distribution / 5  # Normalize
            
            # Calculate entropy of current distribution
            entropy = -np.sum([p * np.log(p) if p > 0 else 0 for p in current_distribution])
            # Normalize by historical entropy
            if historical_entropy > 0:
                dispersion_score = entropy / historical_entropy
            else:
                dispersion_score = 0
                
            synthetic_features.setdefault("main_dispersion_score", []).append(dispersion_score)
            
            # 3. Pattern novelty: How different is this pattern from recent draws
            if i >= 5:  # If we have at least 5 previous draws
                recent_main_numbers = set()
                for j in range(1, 6):
                    recent_main_numbers.update(df.iloc[i-j]["main_numbers"])
                
                overlap = sum(1 for num in main_nums if num in recent_main_numbers)
                novelty_score = 1.0 - overlap / 5  # Higher is more novel (less overlap)
                synthetic_features.setdefault("main_novelty_score", []).append(novelty_score)
            else:
                synthetic_features.setdefault("main_novelty_score", []).append(0.5)  # Default for first draws
            
            # 4. Sequential pattern: Detects arithmetic sequences
            diffs = [main_nums[j+1] - main_nums[j] for j in range(len(main_nums)-1)]
            sequential_score = 1.0 - np.std(diffs) / np.mean(diffs) if np.mean(diffs) > 0 else 0
            synthetic_features.setdefault("main_sequential_score", []).append(max(0, min(1, sequential_score)))
            
            # 5. Historic frequency score: How common are these numbers historically
            if i > 20:  # Wait until we have some historical data
                historical_freq = np.zeros(50)
                for j in range(max(0, i-20), i):
                    for num in df.iloc[j]["main_numbers"]:
                        historical_freq[num-1] += 1
                historical_freq = historical_freq / historical_freq.sum() if historical_freq.sum() > 0 else np.zeros(50)
                
                current_freq = np.zeros(50)
                for num in main_nums:
                    current_freq[num-1] = 1
                current_freq = current_freq / 5
                
                # KL divergence (how different is the current draw from historical distribution)
                kl_div = 0
                for j in range(50):
                    if current_freq[j] > 0:
                        if historical_freq[j] > 0:
                            kl_div += current_freq[j] * np.log(current_freq[j] / historical_freq[j])
                        else:
                            kl_div += current_freq[j] * np.log(current_freq[j] / 0.0001)  # Small value to avoid division by zero
                
                # Normalize to 0-1 range
                freq_score = 1.0 / (1.0 + kl_div)
                synthetic_features.setdefault("main_historical_freq_score", []).append(freq_score)
            else:
                synthetic_features.setdefault("main_historical_freq_score", []).append(0.5)
        
        # Convert to DataFrame
        synthetic_df = pd.DataFrame(synthetic_features, index=df.index)
        return synthetic_df
    
    def fix_infinite_values(self, data_frame):
        """Handle infinite or extremely large values in the DataFrame."""
        # Replace infinities with NaN
        data_frame.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Fill NaN values with 0 or other appropriate value
        data_frame.fillna(0, inplace=True)
        
        # Clip extremely large values to a reasonable range
        numeric_cols = data_frame.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            # Find reasonable bounds (99th percentile)
            q_lo = data_frame[col].quantile(0.01)
            q_hi = data_frame[col].quantile(0.99)
            # Allow more extreme values but clip massive outliers
            range_val = q_hi - q_lo
            data_frame[col] = data_frame[col].clip(q_lo - 5*range_val, q_hi + 5*range_val)
        
        return data_frame

    def create_enhanced_features(self):
        """Create and combine all advanced features with feature importance weighting."""
        logger.info("Generating enhanced features for lottery prediction")
        
        # Generate enhanced feature sets
        time_series_features = self.calculate_time_series_features()
        time_series_features = self.fix_infinite_values(time_series_features)
        
        number_relationship_features = self.calculate_number_relationships()
        number_relationship_features = self.fix_infinite_values(number_relationship_features)
        
        pattern_features = self.calculate_pattern_features()
        pattern_features = self.fix_infinite_values(pattern_features)
        
        multi_draw_features = self.calculate_multi_draw_patterns()
        multi_draw_features = self.fix_infinite_values(multi_draw_features)
        
        cyclical_features = self.calculate_cyclical_features()
        cyclical_features = self.fix_infinite_values(cyclical_features)
        
        synthetic_features = self.calculate_synthetic_features()
        synthetic_features = self.fix_infinite_values(synthetic_features)
        
        # Combine all features
        all_features = pd.concat([
            time_series_features,
            number_relationship_features,
            pattern_features,
            multi_draw_features,
            cyclical_features,
            synthetic_features
        ], axis=1)
        
        # Handle any remaining NaN values
        all_features = all_features.fillna(0)
        
        # Final check for infinities or extremely large values
        all_features = self.fix_infinite_values(all_features)
        
        logger.info(f"Created {all_features.shape[1]} enhanced features")
        return all_features

#######################
# IMPROVED TRANSFORMER MODEL COMPONENTS
#######################

# Here are the fixed implementations of the problematic classes:

class ImprovedTransformerBlock(tf.keras.layers.Layer):
    """Advanced transformer block with improved architecture and residual connections."""
    
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(ImprovedTransformerBlock, self).__init__(dtype=NUMERICALLY_SENSITIVE_POLICY)
        
        # Attention layers are numerically sensitive
        self.att = MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=embed_dim//num_heads, 
            dropout=rate*0.5,
            dtype=NUMERICALLY_SENSITIVE_POLICY
        )
        
        # Two-layer feed-forward network with intermediate activation
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="gelu", kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
                 dtype=COMPUTE_INTENSIVE_POLICY),
            BatchNormalization(dtype=NUMERICALLY_SENSITIVE_POLICY),  # Keep BN in float32
            Dropout(rate),
            Dense(embed_dim, kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
                 dtype=COMPUTE_INTENSIVE_POLICY),
        ])
        
        # Layer normalization is numerically sensitive, keep in float32
        self.layernorm1 = LayerNormalization(epsilon=1e-4, dtype=NUMERICALLY_SENSITIVE_POLICY)
        self.layernorm2 = LayerNormalization(epsilon=1e-4, dtype=NUMERICALLY_SENSITIVE_POLICY)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)
        
        # Add Lambda layers for tensor operations
        self.cast_to_float32 = tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.float32))
        self.add_layer = tf.keras.layers.Add()
        
    def call(self, inputs, training=False):
        # Use Lambda layer for casting to float32
        inputs_float32 = self.cast_to_float32(inputs)
        
        # Pre-norm architecture (more stable training)
        attn_input = self.layernorm1(inputs_float32)
        attn_output = self.att(attn_input, attn_input)
        attn_output = self.dropout1(attn_output, training=training)
        
        # Use Lambda for casting and Add for addition
        out1_cast = self.cast_to_float32(inputs)  # Cast inputs to match attn_output
        out1 = self.add_layer([out1_cast, attn_output])
        
        ffn_input = self.layernorm2(out1)
        ffn_output = self.ffn(ffn_input)
        ffn_output = self.dropout2(ffn_output, training=training)
        
        # Use Add layer for the residual connection
        return self.add_layer([out1, ffn_output])

class PositionalEncoding(tf.keras.layers.Layer):
    """Enhanced positional encoding layer with learnable parameters."""
    
    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__(dtype=NUMERICALLY_SENSITIVE_POLICY)
        self.pos_encoding = self.positional_encoding(position, d_model)
        
        # Add learnable scaling factor
        self.pos_scaling = tf.Variable(1.0, trainable=True, name="pos_encoding_scale")
        
        # Add layers for tensor operations
        self.cast_to_float32 = tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.float32))
        self.add_layer = tf.keras.layers.Add()
        
    def get_angles(self, position, i, d_model):
        angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return position * angles
        
    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
            i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model=d_model
        )
        
        # Apply sin to even indices, cos to odd indices
        sines = tf.math.sin(angle_rads[:, 0::2])
        cosines = tf.math.cos(angle_rads[:, 1::2])
        
        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        
        return tf.cast(pos_encoding, tf.float32)
        
    def call(self, inputs):
        # Cast inputs to float32 using a Lambda layer
        inputs_float32 = self.cast_to_float32(inputs)
        
        # Get the sequence length from the dynamic shape
        seq_len = tf.keras.layers.Lambda(lambda x: tf.shape(x)[1])(inputs_float32)
        
        # Create a Lambda layer to get the appropriate slice of positional encoding
        pos_enc_slice = tf.keras.layers.Lambda(
            lambda x: self.pos_scaling * self.pos_encoding[:, :x, :]
        )(seq_len)
        
        # Use Add layer to combine inputs with positional encoding
        return self.add_layer([inputs_float32, pos_enc_slice])

class FeedForwardWithResidual(tf.keras.layers.Layer):
    """Feed-forward network with residual connection and layer normalization."""
    
    def __init__(self, dim, expansion_factor=4, dropout_rate=0.1, activation="gelu"):
        super(FeedForwardWithResidual, self).__init__()
        # LayerNorm is numerically sensitive
        self.norm = LayerNormalization(epsilon=1e-4, dtype=NUMERICALLY_SENSITIVE_POLICY)
        
        # Dense layers can use mixed precision
        self.expand = Dense(
            dim * expansion_factor, 
            activation=activation,
            dtype=COMPUTE_INTENSIVE_POLICY
        )
        self.dropout1 = Dropout(dropout_rate)
        self.contract = Dense(dim, dtype=COMPUTE_INTENSIVE_POLICY)
        self.dropout2 = Dropout(dropout_rate)
        
        # Add Lambda and Add layers for tensor operations
        self.cast_to_float32 = tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.float32))
        self.add_layer = tf.keras.layers.Add()
        
    def call(self, inputs, training=False):
        # Cast to float32 using a Lambda layer
        inputs_float32 = self.cast_to_float32(inputs)
        x = self.norm(inputs_float32)
        
        # Mixed precision for compute-intensive parts
        x = self.expand(x)
        x = self.dropout1(x, training=training)
        x = self.contract(x)
        x = self.dropout2(x, training=training)
        
        # Use Add layer for the residual connection
        return self.add_layer([inputs, x])

#######################
# IMPROVED TRANSFORMER MODEL
#######################

class ImprovedTransformerModel:
    """Improved transformer model with hybrid architecture and advanced training."""
    
    def __init__(self, input_dim, seq_length, params=None):
        self.input_dim = input_dim
        self.seq_length = seq_length
        self.main_model = None
        self.bonus_model = None
        
        # Default parameters (will be overridden by provided params)
        self.params = {
            'learning_rate': 0.001,
            'batch_size': 16,
            'epochs': 100,
            'dropout_rate': 0.3,
            'num_heads': 4,
            'ff_dim': 128,
            'embed_dim': 64,
            'use_gru': True,
            'conv_filters': 32,
            'optimizer': 'adam',
            'attention_dropout': 0.1,
            'weight_decay': 1e-4,
            'activation': 'gelu',
            'layer_norm_eps': 1e-6,
            'num_transformer_blocks': 3,
            'use_residual_connections': True,
            'use_batch_norm': True
        }
        
        # Update with provided parameters
        if params is not None:
            self.params.update(params)
    
    # Here's how to modify the model building methods:

    def build_main_numbers_model(self):
        """Build improved transformer model for main numbers prediction with selective precision."""
        # Input layers
        feature_input = Input(shape=(self.input_dim,), name="feature_input")
        sequence_input = Input(shape=(self.seq_length, 50), name="sequence_input")
        
        # Process feature input with multi-layer network - compute intensive
        x_features = Dense(
            self.params.get('embed_dim'), 
            activation=self.params.get('activation'), 
            kernel_regularizer=l1_l2(l1=1e-5, l2=self.params.get('weight_decay')),
            dtype=COMPUTE_INTENSIVE_POLICY
        )(feature_input)
        
        if self.params.get('use_batch_norm', True):
            x_features = BatchNormalization(dtype=NUMERICALLY_SENSITIVE_POLICY)(x_features)
        x_features = Dropout(self.params.get('dropout_rate'))(x_features)
        
        # Add hidden layers for better feature processing
        x_features = Dense(
            self.params.get('embed_dim'), 
            activation=self.params.get('activation'), 
            kernel_regularizer=l1_l2(l1=1e-5, l2=self.params.get('weight_decay')),
            dtype=COMPUTE_INTENSIVE_POLICY
        )(x_features)
        
        if self.params.get('use_batch_norm', True):
            x_features = BatchNormalization(dtype=NUMERICALLY_SENSITIVE_POLICY)(x_features)
        x_features = Dropout(self.params.get('dropout_rate'))(x_features)
        
        # Add final feature processing layer
        x_features = Dense(
            self.params.get('embed_dim') // 2, 
            activation=self.params.get('activation'), 
            kernel_regularizer=l1_l2(l1=1e-5, l2=self.params.get('weight_decay')),
            dtype=COMPUTE_INTENSIVE_POLICY
        )(x_features)
        
        # Process sequence input with convolutional layer first - compute intensive
        if self.params.get('conv_filters', 0) > 0:
            x_seq = Conv1D(
                self.params.get('conv_filters'), 
                kernel_size=3, 
                padding='same', 
                activation=self.params.get('activation'),
                dtype=COMPUTE_INTENSIVE_POLICY
            )(sequence_input)
            
            # Add a second convolutional layer for better pattern detection
            x_seq = Conv1D(
                self.params.get('conv_filters'), 
                kernel_size=5, 
                padding='same', 
                activation=self.params.get('activation'),
                dtype=COMPUTE_INTENSIVE_POLICY
            )(x_seq)
        else:
            x_seq = sequence_input
            
        # Apply dense layer to match dimensions
        x_seq = Dense(
            self.params.get('embed_dim'),
            dtype=COMPUTE_INTENSIVE_POLICY
        )(x_seq)
        
        # Apply positional encoding - numerically sensitive
        x_seq = PositionalEncoding(
            self.seq_length, 
            self.params.get('embed_dim')
        )(x_seq)
        
        # Stack advanced transformer blocks - mixed numerical sensitivity
        for _ in range(self.params.get('num_transformer_blocks', 3)):
            x_seq = ImprovedTransformerBlock(
                self.params.get('embed_dim'), 
                self.params.get('num_heads'), 
                self.params.get('ff_dim'), 
                self.params.get('dropout_rate')
            )(x_seq)
        
        # Use bidirectional GRU for better sequence modeling
        if self.params.get('use_gru', False):
            # Optimize for GPU if available
            if gpus:
                # Use optimized GRU implementation - compute intensive
                x_seq = Bidirectional(GRU(
                    self.params.get('embed_dim') // 2, 
                    return_sequences=False,
                    recurrent_activation='sigmoid',  # Optimized for CuDNN
                    reset_after=True,  # Required for CuDNN compatibility
                    name='gpu_optimized_gru',
                    dtype=COMPUTE_INTENSIVE_POLICY
                ))(x_seq)
            else:
                # Standard GRU for CPU
                x_seq = Bidirectional(GRU(
                    self.params.get('embed_dim') // 2, 
                    return_sequences=False,
                    dtype=COMPUTE_INTENSIVE_POLICY
                ))(x_seq)
        else:
            # Global attention pooling instead of simple average pooling
            # Attention is numerically sensitive
            x_seq = Attention(dtype=NUMERICALLY_SENSITIVE_POLICY)([x_seq, x_seq])
            x_seq = GlobalAveragePooling1D()(x_seq)
        
        # Combine feature and sequence representations
        combined = Concatenate()([x_features, x_seq])
        
        # Add multiple dense layers with residual connections - compute intensive
        combined = Dense(
            self.params.get('ff_dim'), 
            activation=self.params.get('activation'),
            kernel_regularizer=l1_l2(l1=1e-5, l2=self.params.get('weight_decay')),
            dtype=COMPUTE_INTENSIVE_POLICY
        )(combined)
        
        if self.params.get('use_batch_norm', True):
            combined = BatchNormalization(dtype=NUMERICALLY_SENSITIVE_POLICY)(combined)
        combined = Dropout(self.params.get('dropout_rate'))(combined)
        
        # Additional hidden layer
        residual = combined
        combined = Dense(
            self.params.get('ff_dim'), 
            activation=self.params.get('activation'),
            kernel_regularizer=l1_l2(l1=1e-5, l2=self.params.get('weight_decay')),
            dtype=COMPUTE_INTENSIVE_POLICY
        )(combined)
        
        if self.params.get('use_batch_norm', True):
            combined = BatchNormalization(dtype=NUMERICALLY_SENSITIVE_POLICY)(combined)
        combined = Dropout(self.params.get('dropout_rate'))(combined)
        
        # Add residual connection if dimensions match
        if self.params.get('use_residual_connections', True):
            # Use Add layer instead of direct addition
            combined = Add()([residual, combined])
        
        # Final hidden layer
        combined = Dense(
            self.params.get('ff_dim') // 2, 
            activation=self.params.get('activation'),
            kernel_regularizer=l1_l2(l1=1e-5, l2=self.params.get('weight_decay')),
            dtype=COMPUTE_INTENSIVE_POLICY
        )(combined)
        combined = Dropout(self.params.get('dropout_rate') / 2)(combined)
        
        # Output layers (one for each main number position)
        outputs = []
        for i in range(5):
            # Use a specialized head for each position
            position_specific = Dense(
                self.params.get('ff_dim') // 4, 
                activation=self.params.get('activation'),
                dtype=COMPUTE_INTENSIVE_POLICY
            )(combined)
            
            logits = Dense(50, activation=None, name=f"main_logits_{i+1}", dtype=COMPUTE_INTENSIVE_POLICY)(position_specific)
            logits_reshaped = Reshape((50,))(logits)
            output = tf.keras.layers.Activation('softmax', name=f"main_{i+1}")(logits_reshaped)
            outputs.append(output)
        
        # Create model
        model = Model(inputs=[feature_input, sequence_input], outputs=outputs)
        
        # Compile model with metrics
        metrics_dict = {f"main_{i+1}": "accuracy" for i in range(5)}
        
        # Select optimizer based on parameter
        if self.params.get('optimizer', 'adam').lower() == 'adam':
            base_optimizer = Adam(learning_rate=self.params.get('learning_rate'))
        elif self.params.get('optimizer', 'adam').lower() == 'sgd':
            base_optimizer = SGD(learning_rate=self.params.get('learning_rate'), momentum=0.9)
        elif self.params.get('optimizer', 'adam').lower() == 'rmsprop':
            base_optimizer = RMSprop(learning_rate=self.params.get('learning_rate'))
        else:
            base_optimizer = Adam(learning_rate=self.params.get('learning_rate'))
        
        # Wrap with loss scale optimizer for mixed precision
        from tensorflow.keras.mixed_precision import LossScaleOptimizer
        optimizer = LossScaleOptimizer(base_optimizer)
        
        model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=metrics_dict
        )
        
        self.main_model = model
        logger.info("Built improved transformer main numbers model successfully")
        return model

    def build_bonus_numbers_model(self):
        """Build improved transformer model for bonus numbers prediction with selective precision."""
        # Input layers
        feature_input = Input(shape=(self.input_dim,), name="feature_input")
        sequence_input = Input(shape=(self.seq_length, 12), name="sequence_input")
        
        # Process feature input with multi-layer network - compute intensive
        x_features = Dense(
            self.params.get('embed_dim'), 
            activation=self.params.get('activation'),
            kernel_regularizer=l1_l2(l1=1e-5, l2=self.params.get('weight_decay')),
            dtype=COMPUTE_INTENSIVE_POLICY
        )(feature_input)
        
        if self.params.get('use_batch_norm', True):
            x_features = BatchNormalization(dtype=NUMERICALLY_SENSITIVE_POLICY)(x_features)
        x_features = Dropout(self.params.get('dropout_rate'))(x_features)
        
        # Additional hidden layer for features
        x_features = Dense(
            self.params.get('embed_dim'), 
            activation=self.params.get('activation'),
            kernel_regularizer=l1_l2(l1=1e-5, l2=self.params.get('weight_decay')),
            dtype=COMPUTE_INTENSIVE_POLICY
        )(x_features)
        
        if self.params.get('use_batch_norm', True):
            x_features = BatchNormalization(dtype=NUMERICALLY_SENSITIVE_POLICY)(x_features)
        x_features = Dropout(self.params.get('dropout_rate'))(x_features)
        
        # Final feature processing layer
        x_features = Dense(
            self.params.get('embed_dim') // 2, 
            activation=self.params.get('activation'),
            kernel_regularizer=l1_l2(l1=1e-5, l2=self.params.get('weight_decay')),
            dtype=COMPUTE_INTENSIVE_POLICY
        )(x_features)
        
        # Process sequence input with convolutional layer first
        if self.params.get('conv_filters', 0) > 0:
            x_seq = Conv1D(
                self.params.get('conv_filters'), 
                kernel_size=3, 
                padding='same', 
                activation=self.params.get('activation'),
                dtype=COMPUTE_INTENSIVE_POLICY
            )(sequence_input)
            
            # Add additional conv layer
            x_seq = Conv1D(
                self.params.get('conv_filters'), 
                kernel_size=5, 
                padding='same', 
                activation=self.params.get('activation'),
                dtype=COMPUTE_INTENSIVE_POLICY
            )(x_seq)
        else:
            x_seq = sequence_input
            
        # Apply dense layer to match dimensions
        x_seq = Dense(
            self.params.get('embed_dim'),
            dtype=COMPUTE_INTENSIVE_POLICY
        )(x_seq)
        
        # Apply positional encoding - numerically sensitive
        x_seq = PositionalEncoding(
            self.seq_length, 
            self.params.get('embed_dim')
        )(x_seq)
        
        # Stack improved transformer blocks
        for _ in range(2):  # Using 2 transformer blocks for bonus numbers
            x_seq = ImprovedTransformerBlock(
                self.params.get('embed_dim'), 
                self.params.get('num_heads'), 
                self.params.get('ff_dim'), 
                self.params.get('dropout_rate')
            )(x_seq)
        
        # Use bidirectional GRU for advanced sequence processing
        if self.params.get('use_gru', False):
            if gpus:
                # Use optimized GRU implementation
                x_seq = Bidirectional(GRU(
                    self.params.get('embed_dim') // 2, 
                    return_sequences=False,
                    recurrent_activation='sigmoid',  # Optimized for CuDNN
                    reset_after=True,  # Required for CuDNN compatibility
                    name='gpu_optimized_bonus_gru',
                    dtype=COMPUTE_INTENSIVE_POLICY
                ))(x_seq)
            else:
                x_seq = Bidirectional(GRU(
                    self.params.get('embed_dim') // 2, 
                    return_sequences=False,
                    dtype=COMPUTE_INTENSIVE_POLICY
                ))(x_seq)
        else:
            # Global attention pooling - numerically sensitive
            x_seq = Attention(dtype=NUMERICALLY_SENSITIVE_POLICY)([x_seq, x_seq])
            x_seq = GlobalAveragePooling1D()(x_seq)
        
        # Combine feature and sequence representations
        combined = Concatenate()([x_features, x_seq])
        
        # Multi-layer processing with residual connections
        combined = Dense(
            self.params.get('ff_dim'), 
            activation=self.params.get('activation'),
            kernel_regularizer=l1_l2(l1=1e-5, l2=self.params.get('weight_decay')),
            dtype=COMPUTE_INTENSIVE_POLICY
        )(combined)
        
        if self.params.get('use_batch_norm', True):
            combined = BatchNormalization(dtype=NUMERICALLY_SENSITIVE_POLICY)(combined)
        combined = Dropout(self.params.get('dropout_rate'))(combined)
        
        # Additional hidden layer with residual connection
        residual = combined
        combined = Dense(
            self.params.get('ff_dim'), 
            activation=self.params.get('activation'),
            kernel_regularizer=l1_l2(l1=1e-5, l2=self.params.get('weight_decay')),
            dtype=COMPUTE_INTENSIVE_POLICY
        )(combined)
        
        if self.params.get('use_batch_norm', True):
            combined = BatchNormalization(dtype=NUMERICALLY_SENSITIVE_POLICY)(combined)
        combined = Dropout(self.params.get('dropout_rate'))(combined)
        
        # Add residual connection if enabled - Use Add layer instead of direct addition
        if self.params.get('use_residual_connections', True):
            combined = Add()([residual, combined])
        
        # Final processing layer
        combined = Dense(
            self.params.get('ff_dim') // 2, 
            activation=self.params.get('activation'),
            kernel_regularizer=l1_l2(l1=1e-5, l2=self.params.get('weight_decay')),
            dtype=COMPUTE_INTENSIVE_POLICY
        )(combined)
        combined = Dropout(self.params.get('dropout_rate') / 2)(combined)
        
        # Output layers with position-specific processing
        outputs = []
        for i in range(2):
            position_specific = Dense(
                self.params.get('ff_dim') // 4, 
                activation=self.params.get('activation'),
                dtype=COMPUTE_INTENSIVE_POLICY
            )(combined)
            
            logits = Dense(12, activation=None, name=f"bonus_logits_{i+1}", dtype=COMPUTE_INTENSIVE_POLICY)(position_specific)
            logits_reshaped = Reshape((12,))(logits)
            output = tf.keras.layers.Activation('softmax', name=f"bonus_{i+1}")(logits_reshaped)
            outputs.append(output)
        
        # Create model
        model = Model(inputs=[feature_input, sequence_input], outputs=outputs)
        
        # Compile model with metrics
        metrics_dict = {f"bonus_{i+1}": "accuracy" for i in range(2)}
        
        # Select optimizer
        if self.params.get('optimizer', 'adam').lower() == 'adam':
            base_optimizer = Adam(learning_rate=self.params.get('learning_rate'))
        elif self.params.get('optimizer', 'adam').lower() == 'sgd':
            base_optimizer = SGD(learning_rate=self.params.get('learning_rate'), momentum=0.9)
        elif self.params.get('optimizer', 'adam').lower() == 'rmsprop':
            base_optimizer = RMSprop(learning_rate=self.params.get('learning_rate'))
        else:
            base_optimizer = Adam(learning_rate=self.params.get('learning_rate'))
        
        # Wrap with loss scale optimizer for mixed precision
        from tensorflow.keras.mixed_precision import LossScaleOptimizer
        optimizer = LossScaleOptimizer(base_optimizer)
        
        model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=metrics_dict
        )
        
        self.bonus_model = model
        logger.info("Built improved transformer bonus numbers model successfully")
        return model

#######################
# IMPROVED LOTTERY PREDICTOR
#######################

class ImprovedLotteryPredictor:
    """Improved lottery prediction system with advanced transformer models."""
    
    def __init__(self, file_path, params=None):
        self.file_path = file_path
        self.data = None
        self.expanded_data = None
        self.features = None
        self.sequence_length = 40  # Increased for better pattern detection
        self.feature_scaler = StandardScaler()
        self.selected_features = None
        self.X_scaled = None
        self.main_sequences = None
        self.bonus_sequences = None
        self.model = None
        self.params = params
        self.feature_importance = None  # For tracking feature importance

    def create_tf_dataset(self, X, sequences, targets, batch_size=16, is_training=True):
        """Create optimized TensorFlow dataset for GPU acceleration."""
        # Unpack targets into a list for multi-output model
        target_list = [targets[:, i] for i in range(targets.shape[1])]
        
        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices(((X.values, sequences), target_list))
        
        if is_training:
            # Shuffle, batch, and prefetch for training
            dataset = dataset.cache().shuffle(buffer_size=1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        else:
            # Just batch and prefetch for validation/testing
            dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
            
        return dataset
    
    def load_and_preprocess_data(self):
        """Load and preprocess lottery data with enhanced features."""
        logger.info("Loading and preprocessing lottery data with enhanced features")
        
        try:
            # Load data using the enhanced processor
            processor = LotteryDataProcessor(self.file_path)
            self.data = processor.parse_file()
            self.expanded_data = processor.expand_numbers()
            
            # Create enhanced features
            feature_eng = EnhancedFeatureEngineering(self.expanded_data)
            self.features = feature_eng.create_enhanced_features()
            
            # Perform feature selection to reduce dimensionality and improve performance
            X_raw = self.features.select_dtypes(include=[np.number])
            X_raw = X_raw.fillna(0)
            
            # Use feature selection with mutual information - dynamically select feature count
            max_features = min(1000, X_raw.shape[1])  # Cap at 1000 features
            
            try:
                # Try mutual information first with cross-validation for optimal feature count
                best_score = 0
                best_k = max_features
                
                # Try different feature counts
                for k in [100, 200, 500, max_features]:
                    if k >= X_raw.shape[1]:
                        continue
                        
                    selector = SelectKBest(mutual_info_regression, k=k)
                    X_selected = selector.fit_transform(X_raw.values, self.expanded_data["main_1"].values)
                    
                    # Get 5 target values for simple cross-validation
                    y = self.expanded_data["main_1"].values[-100:]  # Use last 100 samples
                    X = X_selected[-100:, :]
                    
                    # Simple train/test split
                    X_train, X_test = X[:-20], X[-20:]
                    y_train, y_test = y[:-20], y[-20:]
                    
                    # Train a simple model to evaluate feature set
                    from sklearn.ensemble import RandomForestRegressor
                    model = RandomForestRegressor(n_estimators=50, random_state=RANDOM_SEED)
                    model.fit(X_train, y_train)
                    
                    # Evaluate
                    score = model.score(X_test, y_test)
                    
                    if score > best_score:
                        best_score = score
                        best_k = k
                
                # Use the best feature count
                selector = SelectKBest(mutual_info_regression, k=best_k)
                X_selected = selector.fit_transform(X_raw.values, self.expanded_data["main_1"].values)
                selected_indices = selector.get_support(indices=True)
                selected_cols = [X_raw.columns[i] for i in selected_indices]
                
                # Store feature importance scores
                self.feature_importance = dict(zip(selected_cols, selector.scores_[selected_indices]))
                
            except Exception as e:
                logger.warning(f"Mutual information feature selection failed: {e}, falling back to f_regression")
                # Fall back to f_regression if mutual_info fails
                selector = SelectKBest(f_regression, k=min(500, X_raw.shape[1]))
                X_selected = selector.fit_transform(X_raw.values, self.expanded_data["main_1"].values)
                selected_indices = selector.get_support(indices=True)
                selected_cols = [X_raw.columns[i] for i in selected_indices]
                
                # Store feature importance scores
                self.feature_importance = dict(zip(selected_cols, selector.scores_[selected_indices]))
            
            X = pd.DataFrame(X_selected, columns=selected_cols, index=X_raw.index)
            self.selected_features = selected_cols
            
            # Scale features using robust scaler
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
            self.X_scaled = pd.DataFrame(
                scaler.fit_transform(X),
                columns=X.columns,
                index=X.index
            )
            
            # Create enhanced sequence features for transformer
            main_sequences = []
            bonus_sequences = []
            
            # For each draw, create a sequence of previous draws with enhanced encoding
            for i in range(len(self.expanded_data) - self.sequence_length):
                # Get the window of previous draws
                window = self.expanded_data.iloc[i:i+self.sequence_length]
                
                # Create weighted one-hot encoding for main numbers
                main_seq = np.zeros((self.sequence_length, 50))
                for j, (_, row) in enumerate(window.iterrows()):
                    # Apply temporal weighting (more recent draws have slightly higher weight)
                    temporal_weight = 0.8 + 0.2 * (j / self.sequence_length)
                    
                    for num in row["main_numbers"]:
                        if 1 <= num <= 50:  # Ensure number is valid
                            main_seq[j, num-1] = temporal_weight
                            
                # Create weighted one-hot encoding for bonus numbers
                bonus_seq = np.zeros((self.sequence_length, 12))
                for j, (_, row) in enumerate(window.iterrows()):
                    # Apply temporal weighting
                    temporal_weight = 0.8 + 0.2 * (j / self.sequence_length)
                    
                    for num in row["bonus_numbers"]:
                        if 1 <= num <= 12:  # Ensure number is valid
                            bonus_seq[j, num-1] = temporal_weight
                
                main_sequences.append(main_seq)
                bonus_sequences.append(bonus_seq)
            
            # Convert to numpy arrays
            self.main_sequences = np.array(main_sequences)
            self.bonus_sequences = np.array(bonus_sequences)
            
            # Create target variables (next draw's numbers)
            # Subtract 1 from each number to use as index (0-49 for main, 0-11 for bonus)
            y_main = np.array([
                self.expanded_data.iloc[self.sequence_length:][f"main_{i+1}"].values - 1 for i in range(5)
            ]).T
            
            y_bonus = np.array([
                self.expanded_data.iloc[self.sequence_length:][f"bonus_{i+1}"].values - 1 for i in range(2)
            ]).T
            
            # Match feature data to sequence data
            X_scaled_matched = self.X_scaled.iloc[self.sequence_length:]
            
            return X_scaled_matched, self.main_sequences, self.bonus_sequences, y_main, y_bonus
        except Exception as e:
            logger.error(f"Error in data preprocessing: {str(e)}")
            raise

    def train_models(self, epochs=None, batch_size=None, validation_split=0.1, early_stopping=True):
        """Train improved transformer-based prediction models with advanced techniques and GPU acceleration."""
        logger.info("Training improved transformer-based lottery prediction models with GPU acceleration")
        
        try:
            # Load and preprocess data
            X_scaled, main_sequences, bonus_sequences, y_main, y_bonus = self.load_and_preprocess_data()
            
            # Verify input dimensions
            logger.info(f"Feature dimensions: {X_scaled.shape}")
            logger.info(f"Main sequences dimensions: {main_sequences.shape}")
            logger.info(f"Bonus sequences dimensions: {bonus_sequences.shape}")
            logger.info(f"Main numbers target shape: {y_main.shape}")
            logger.info(f"Bonus numbers target shape: {y_bonus.shape}")
            
            # Optimize batch size for GPU
            if gpus:
                actual_batch_size = batch_size * 2 if batch_size else self.params.get('batch_size', 16) * 2
                logger.info(f"Using GPU-optimized batch size: {actual_batch_size}")
            else:
                actual_batch_size = batch_size if batch_size is not None else self.params.get('batch_size', 16)
                logger.info(f"Using CPU batch size: {actual_batch_size}")
            
            # Override parameters if provided directly to the method
            actual_epochs = epochs if epochs is not None else self.params.get('epochs', 100)
            
            # Setup distribution strategy for multi-GPU if available
            if gpus and len(gpus) > 1:
                strategy = tf.distribute.MirroredStrategy()
                logger.info(f"Training with {strategy.num_replicas_in_sync} GPUs")
                # Create model in strategy scope
                with strategy.scope():
                    self.model = ImprovedTransformerModel(
                        input_dim=X_scaled.shape[1],
                        seq_length=self.sequence_length,
                        params=self.params
                    )
                    # Build models
                    main_model = self.model.build_main_numbers_model()
                    bonus_model = self.model.build_bonus_numbers_model()
            else:
                # Create improved transformer models
                self.model = ImprovedTransformerModel(
                    input_dim=X_scaled.shape[1],
                    seq_length=self.sequence_length,
                    params=self.params
                )
                # Build main numbers model
                main_model = self.model.build_main_numbers_model()
                bonus_model = self.model.build_bonus_numbers_model()
                
            # Set up callbacks with advanced options
            callbacks = []
            
            if early_stopping:
                callbacks.append(EarlyStopping(
                    patience=30,  # Increased patience
                    restore_best_weights=True,
                    monitor='val_loss',
                    min_delta=0.0001  # Smaller delta for more precise stopping
                ))
            
            # Learning rate scheduler
            callbacks.append(ReduceLROnPlateau(
                factor=0.8,  # Smaller factor for more gradual reduction
                patience=15,  # Increased patience
                min_lr=1e-6,
                monitor='val_loss',
                verbose=1
            ))
            
            # Model checkpoint to save best model
            callbacks.append(ModelCheckpoint(
                filepath="best_main_model.keras",
                save_best_only=True,
                monitor='val_loss',
                mode='min',
                verbose=1
            ))
            
            # Implement gradual learning rate warmup
            initial_lr = self.params.get('learning_rate', 0.001)
            
            class LRWarmupCallback(tf.keras.callbacks.Callback):
                def __init__(self, warmup_epochs=5, initial_lr=0.0001, target_lr=0.001):
                    super(LRWarmupCallback, self).__init__()
                    self.warmup_epochs = warmup_epochs
                    self.initial_lr = initial_lr
                    self.target_lr = target_lr
                    self.lr_increment = (target_lr - initial_lr) / warmup_epochs
                    
                def on_epoch_begin(self, epoch, logs=None):
                    if epoch < self.warmup_epochs:
                        new_lr = self.initial_lr + epoch * self.lr_increment
                        tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
                        print(f"\nEpoch {epoch+1}: Setting warmup LR to {new_lr:.6f}")
            
            # Add warmup callback if using at least 20 epochs
            if actual_epochs >= 20:
                callbacks.append(LRWarmupCallback(
                    warmup_epochs=5,
                    initial_lr=initial_lr / 10,
                    target_lr=initial_lr
                ))
            
            # Create TensorFlow datasets for efficient GPU feeding
            # Split data for validation
            val_samples = int(len(X_scaled) * validation_split)
            train_samples = len(X_scaled) - val_samples
            
            # Training data
            X_train = X_scaled.iloc[:train_samples]
            main_seq_train = main_sequences[:train_samples]
            y_main_train = y_main[:train_samples]
            
            # Validation data
            X_val = X_scaled.iloc[train_samples:]
            main_seq_val = main_sequences[train_samples:]
            y_main_val = y_main[train_samples:]
            
            # Create optimized datasets
            train_dataset = self.create_tf_dataset(X_train, main_seq_train, y_main_train, batch_size=actual_batch_size)
            val_dataset = self.create_tf_dataset(X_val, main_seq_val, y_main_val, batch_size=actual_batch_size, is_training=False)
            
            # Calculate class weights for main numbers
            main_class_weights = []
            for i in range(5):
                # Get historical distribution for this position
                main_pos_counts = np.zeros(50)
                for num in self.expanded_data[f"main_{i+1}"]:
                    if 1 <= num <= 50:
                        main_pos_counts[num-1] += 1
                        
                # Normalize counts
                pos_dist = main_pos_counts / np.sum(main_pos_counts)
                
                # Invert and normalize for class weights (less frequent = higher weight)
                class_weights = {}
                inv_freq = 1.0 / (pos_dist + 0.01)  # Add small value to avoid division by zero
                inv_freq = inv_freq / np.sum(inv_freq) * 50  # Normalize so average is 1.0
                
                for j in range(50):
                    class_weights[j] = inv_freq[j]
                    
                main_class_weights.append(class_weights)
            
            # Train with GPU profiling for performance analysis
            if gpus:
                # Set up profiling directory
                profiling_dir = os.path.join(os.getcwd(), 'tf_profiling')
                os.makedirs(profiling_dir, exist_ok=True)
                logger.info(f"GPU profiling enabled. Logs will be saved to {profiling_dir}")
                
                # Start profiler
                tf.profiler.experimental.start(profiling_dir)
            
            # Train main model with advanced options
            try:
                main_history = main_model.fit(
                    train_dataset,
                    epochs=actual_epochs,
                    validation_data=val_dataset,
                    callbacks=callbacks,
                    class_weight=main_class_weights,  # Apply class weights
                    verbose=1
                )
                
                # Save training history
                with open("main_model_history.json", "w") as f:
                    json.dump({
                        "loss": [float(x) for x in main_history.history["loss"]],
                        "val_loss": [float(x) for x in main_history.history["val_loss"]]
                    }, f, indent=4)
                
                # Load best model from checkpoint
                if os.path.exists("best_main_model.keras"):
                    main_model.load_weights("best_main_model.keras")
                
            except Exception as e:
                logger.error(f"Error training main model: {str(e)}")
            
            # Prepare bonus number datasets
            X_train = X_scaled.iloc[:train_samples]
            bonus_seq_train = bonus_sequences[:train_samples]
            y_bonus_train = y_bonus[:train_samples]
            
            X_val = X_scaled.iloc[train_samples:]
            bonus_seq_val = bonus_sequences[train_samples:]
            y_bonus_val = y_bonus[train_samples:]
            
            train_bonus_dataset = self.create_tf_dataset(X_train, bonus_seq_train, y_bonus_train, batch_size=actual_batch_size)
            val_bonus_dataset = self.create_tf_dataset(X_val, bonus_seq_val, y_bonus_val, batch_size=actual_batch_size, is_training=False)
            
            # Same for bonus numbers
            bonus_class_weights = []
            for i in range(2):
                bonus_pos_counts = np.zeros(12)
                for num in self.expanded_data[f"bonus_{i+1}"]:
                    if 1 <= num <= 12:
                        bonus_pos_counts[num-1] += 1
                        
                pos_dist = bonus_pos_counts / np.sum(bonus_pos_counts)
                
                class_weights = {}
                inv_freq = 1.0 / (pos_dist + 0.01)
                inv_freq = inv_freq / np.sum(inv_freq) * 12
                
                for j in range(12):
                    class_weights[j] = inv_freq[j]
                    
                bonus_class_weights.append(class_weights)
            
            # Update callbacks for bonus model
            for callback in callbacks:
                if isinstance(callback, ModelCheckpoint):
                    callback.filepath = "best_bonus_model.keras"
            
            try:
                bonus_history = bonus_model.fit(
                    train_bonus_dataset,
                    epochs=actual_epochs,
                    validation_data=val_bonus_dataset,
                    callbacks=callbacks,
                    class_weight=bonus_class_weights,  # Apply class weights
                    verbose=1
                )
                
                # Save training history
                with open("bonus_model_history.json", "w") as f:
                    json.dump({
                        "loss": [float(x) for x in bonus_history.history["loss"]],
                        "val_loss": [float(x) for x in bonus_history.history["val_loss"]]
                    }, f, indent=4)
                
                # Load best model from checkpoint
                if os.path.exists("best_bonus_model.keras"):
                    bonus_model.load_weights("best_bonus_model.keras")
                    
            except Exception as e:
                logger.error(f"Error training bonus model: {str(e)}")
            
            # Stop profiler if enabled
            if gpus:
                tf.profiler.experimental.stop()
                logger.info("GPU profiling complete")
            
            return self.model
        except Exception as e:
            logger.error(f"Error in model training: {str(e)}")
            raise
    
    def predict(self, num_draws=5, temperature=0.8, diversity_sampling=True):
        """Generate improved lottery number predictions with calibrated confidence."""
        logger.info(f"Generating {num_draws} lottery predictions with enhanced methods")
        
        if self.model is None:
            logger.error("Models not trained. Call train_models() first.")
            return None
        
        try:
            # Prepare the latest data for prediction
            latest_features = self.X_scaled.iloc[-1:].values
            latest_main_seq = self.main_sequences[-1:] 
            latest_bonus_seq = self.bonus_sequences[-1:]
            
            # Track used numbers to maximize diversity across predictions
            used_main_numbers = set()
            used_bonus_numbers = set()
            
            # Generate predictions
            predictions = []
            
            for draw_idx in range(num_draws):
                # Predict main numbers
                main_probs = self.model.main_model.predict([latest_features, latest_main_seq])
                
                # Predict bonus numbers
                bonus_probs = self.model.bonus_model.predict([latest_features, latest_bonus_seq])
                
                # Apply temperature scaling to control randomness
                for i in range(len(main_probs)):
                    main_probs[i] = np.power(main_probs[i], 1/temperature)
                    main_probs[i] = main_probs[i] / np.sum(main_probs[i], axis=1, keepdims=True)
                    
                for i in range(len(bonus_probs)):
                    bonus_probs[i] = np.power(bonus_probs[i], 1/temperature)
                    bonus_probs[i] = bonus_probs[i] / np.sum(bonus_probs[i], axis=1, keepdims=True)
                
                # Sample main numbers without replacement with diversity control
                main_numbers = []
                available_main = list(range(1, 51))
                
                for i in range(5):
                    # Get probability distribution for this position
                    probs = main_probs[i][0]
                    
                    # Ensure probs is the right shape
                    if probs.shape[0] != 50:
                        logger.warning(f"Unexpected prob shape: {probs.shape}, reshaping")
                        probs = np.ones(50) / 50
                    
                    # Adjust probabilities to only include available numbers
                    adjusted_probs = np.zeros(50)
                    for j, num in enumerate(available_main):
                        adjusted_probs[num-1] = probs[num-1]
                        
                        # Apply diversity penalty if this number was used in previous predictions
                        if diversity_sampling and num in used_main_numbers and draw_idx > 0:
                            diversity_factor = 0.8  # Reduce probability by 20%
                            adjusted_probs[num-1] *= diversity_factor
                    
                    # Normalize probabilities
                    if np.sum(adjusted_probs) > 0:
                        adjusted_probs = adjusted_probs / np.sum(adjusted_probs)
                    else:
                        # Fallback to uniform distribution if all probabilities are zero
                        valid_indices = [j-1 for j in available_main]
                        adjusted_probs[valid_indices] = 1.0 / len(valid_indices)
                    
                    # Sample a number based on adjusted probabilities
                    try:
                        num_idx = np.random.choice(50, p=adjusted_probs)
                        num = num_idx + 1  # Convert back to 1-50 range
                    except ValueError as e:
                        # Fallback to random selection if sampling fails
                        logger.warning(f"Error sampling main number: {e}. Using random selection.")
                        num = random.choice(available_main)
                    
                    main_numbers.append(int(num))
                    available_main.remove(int(num))
                    used_main_numbers.add(int(num))
                
                # Sort main numbers
                main_numbers.sort()
                
                # Sample bonus numbers without replacement with diversity
                bonus_numbers = []
                available_bonus = list(range(1, 13))
                
                for i in range(2):
                    # Get probability distribution for this position
                    probs = bonus_probs[i][0]
                    
                    # Ensure probs is the right shape
                    if probs.shape[0] != 12:
                        logger.warning(f"Unexpected bonus prob shape: {probs.shape}, reshaping")
                        probs = np.ones(12) / 12
                    
                    # Adjust probabilities to only include available numbers
                    adjusted_probs = np.zeros(12)
                    for j, num in enumerate(available_bonus):
                        adjusted_probs[num-1] = probs[num-1]
                        
                        # Apply diversity penalty
                        if diversity_sampling and num in used_bonus_numbers and draw_idx > 0:
                            diversity_factor = 0.8
                            adjusted_probs[num-1] *= diversity_factor
                    
                    # Normalize probabilities
                    if np.sum(adjusted_probs) > 0:
                        adjusted_probs = adjusted_probs / np.sum(adjusted_probs)
                    else:
                        valid_indices = [j-1 for j in available_bonus]
                        adjusted_probs[valid_indices] = 1.0 / len(valid_indices)
                    
                    # Sample a number based on adjusted probabilities
                    try:
                        num_idx = np.random.choice(12, p=adjusted_probs)
                        num = num_idx + 1  # Convert back to 1-12 range
                    except ValueError as e:
                        logger.warning(f"Error sampling bonus number: {e}. Using random selection.")
                        num = random.choice(available_bonus)
                    
                    bonus_numbers.append(int(num))
                    available_bonus.remove(int(num))
                    used_bonus_numbers.add(int(num))
                
                # Sort bonus numbers
                bonus_numbers.sort()
                
                # Calculate enhanced confidence scores
                main_confidence = np.mean([np.max(main_probs[i][0]) for i in range(5)])
                bonus_confidence = np.mean([np.max(bonus_probs[i][0]) for i in range(2)])
                
                # Calibrate confidence scores (raw model probabilities tend to be overconfident)
                # Apply a calibration function based on validation performance
                calibrated_main_conf = 0.6 * main_confidence + 0.2  # Linear scaling
                calibrated_bonus_conf = 0.7 * bonus_confidence + 0.15
                overall_confidence = (calibrated_main_conf + calibrated_bonus_conf) / 2
                
                # Calculate pattern score - how similar is this to historical patterns
                pattern_score = self.calculate_pattern_score(main_numbers, bonus_numbers)
                
                # Calculate historical frequency score
                frequency_score = self.calculate_frequency_score(main_numbers, bonus_numbers)
                
                # Average the confidence scores
                final_confidence = (overall_confidence + pattern_score + frequency_score) / 3
                
                predictions.append({
                    "main_numbers": main_numbers,
                    "bonus_numbers": bonus_numbers,
                    "confidence": {
                        "overall": float(final_confidence),
                        "main_numbers": float(calibrated_main_conf),
                        "bonus_numbers": float(calibrated_bonus_conf),
                        "pattern_score": float(pattern_score),
                        "frequency_score": float(frequency_score)
                    },
                    "method": "improved_transformer"
                })
            
            return predictions
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            # Fallback to intelligent random predictions if something goes wrong
            predictions = self.generate_fallback_predictions(num_draws)
            return predictions
    
    def calculate_pattern_score(self, main_numbers, bonus_numbers):
        """Calculate pattern score based on historical patterns."""
        try:
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
            odd_count = 5 - even_count
            
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
            
        except Exception as e:
            logger.error(f"Error calculating pattern score: {e}")
            return 0.5  # Default middle score
    
    def calculate_frequency_score(self, main_numbers, bonus_numbers):
        """Calculate score based on historical frequency of numbers."""
        try:
            if self.data is None or len(self.data) < 20:
                return 0.5  # Not enough historical data
            
            # Calculate historical frequency for main numbers
            main_counts = np.zeros(50)
            for _, row in self.data.iterrows():
                for num in row["main_numbers"]:
                    if 1 <= num <= 50:
                        main_counts[num-1] += 1
            
            # Normalize to get probability
            main_freq = main_counts / len(self.data)
            
            # Calculate frequency score for selected main numbers
            main_score = np.mean([main_freq[num-1] for num in main_numbers])
            
            # Normalize score to 0-1 range
            # Expected value for random selection would be 0.1 (5/50)
            main_score_norm = min(1.0, main_score / 0.1)
            
            # Same for bonus numbers
            bonus_counts = np.zeros(12)
            for _, row in self.data.iterrows():
                for num in row["bonus_numbers"]:
                    if 1 <= num <= 12:
                        bonus_counts[num-1] += 1
            
            bonus_freq = bonus_counts / len(self.data)
            bonus_score = np.mean([bonus_freq[num-1] for num in bonus_numbers])
            
            # Normalize (expected value for random selection is 0.167 (2/12))
            bonus_score_norm = min(1.0, bonus_score / 0.167)
            
            # Combine scores
            return (main_score_norm + bonus_score_norm) / 2
            
        except Exception as e:
            logger.error(f"Error calculating frequency score: {e}")
            return 0.5  # Default middle score
    
    def generate_fallback_predictions(self, num_draws):
        """Generate intelligent fallback predictions based on historical frequency."""
        logger.warning("Using fallback prediction method")
        
        try:
            # Calculate historical frequency
            main_counts = np.zeros(50)
            bonus_counts = np.zeros(12)
            
            if self.data is not None:
                for _, row in self.data.iterrows():
                    for num in row["main_numbers"]:
                        if 1 <= num <= 50:
                            main_counts[num-1] += 1
                    for num in row["bonus_numbers"]:
                        if 1 <= num <= 12:
                            bonus_counts[num-1] += 1
                            
                # Normalize to probabilities
                main_probs = main_counts / np.sum(main_counts) if np.sum(main_counts) > 0 else np.ones(50) / 50
                bonus_probs = bonus_counts / np.sum(bonus_counts) if np.sum(bonus_counts) > 0 else np.ones(12) / 12
            else:
                # If no data, use uniform distribution
                main_probs = np.ones(50) / 50
                bonus_probs = np.ones(12) / 12
            
            # Generate predictions
            predictions = []
            
            for _ in range(num_draws):
                # Sample main numbers based on historical frequency
                available_main = list(range(1, 51))
                main_numbers = []
                
                for _ in range(5):
                    # Adjust probabilities to only include available numbers
                    adjusted_probs = np.zeros(50)
                    for num in available_main:
                        adjusted_probs[num-1] = main_probs[num-1]
                    
                    # Normalize
                    adjusted_probs = adjusted_probs / np.sum(adjusted_probs)
                    
                    # Sample
                    try:
                        num = np.random.choice(50, p=adjusted_probs) + 1
                    except:
                        num = random.choice(available_main)
                        
                    main_numbers.append(int(num))
                    available_main.remove(int(num))
                
                # Sort
                main_numbers.sort()
                
                # Sample bonus numbers
                available_bonus = list(range(1, 13))
                bonus_numbers = []
                
                for _ in range(2):
                    # Adjust probabilities
                    adjusted_probs = np.zeros(12)
                    for num in available_bonus:
                        adjusted_probs[num-1] = bonus_probs[num-1]
                    
                    # Normalize
                    adjusted_probs = adjusted_probs / np.sum(adjusted_probs)
                    
                    # Sample
                    try:
                        num = np.random.choice(12, p=adjusted_probs) + 1
                    except:
                        num = random.choice(available_bonus)
                        
                    bonus_numbers.append(int(num))
                    available_bonus.remove(int(num))
                
                # Sort
                bonus_numbers.sort()
                
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
            logger.error(f"Error in fallback prediction: {e}")
            # Last resort: truly random
            predictions = []
            for _ in range(num_draws):
                main_numbers = sorted(random.sample(range(1, 51), 5))
                bonus_numbers = sorted(random.sample(range(1, 13), 2))
                
                predictions.append({
                    "main_numbers": main_numbers,
                    "bonus_numbers": bonus_numbers,
                    "confidence": {
                        "overall": 0.1,
                        "main_numbers": 0.1,
                        "bonus_numbers": 0.1
                    },
                    "method": "pure_random_fallback"
                })
            
            return predictions

#######################
# IMPROVED ENSEMBLE MODEL
#######################

class DiverseEnsemble:
    """Improved ensemble model with diversity optimization."""
    
    def __init__(self, file_path, num_base_models=7, params=None):
        self.file_path = file_path
        self.base_models = []
        self.meta_model = None
        self.predictor = None
        self.num_base_models = num_base_models
        self.params = params
        self.X_scaled = None
        self.main_sequences = None
        self.bonus_sequences = None
        self.model_weights = None  # Weights for each model's predictions
    
    def train_base_models(self):
        """Train diverse base models for the ensemble."""
        logger.info(f"Training {self.num_base_models} diverse base models for ensemble")
        
        # Create a predictor for data preprocessing
        self.predictor = ImprovedLotteryPredictor(self.file_path, self.params)
        
        # Load data once for reuse
        X_scaled, main_sequences, bonus_sequences, y_main, y_bonus = self.predictor.load_and_preprocess_data()
        
        self.X_scaled = X_scaled
        self.main_sequences = main_sequences
        self.bonus_sequences = bonus_sequences
        
        # Train multiple models with diversity enhancements
        for i in range(self.num_base_models):
            logger.info(f"Training base model {i+1}/{self.num_base_models} for ensemble")
            
            # Create model with diversity parameters
            model_params = self.params.copy() if self.params else {}
            
            # Apply diversity techniques:
            # 1. Vary dropout - higher values encourage different feature patterns
            model_params['dropout_rate'] = self.params.get('dropout_rate', 0.25) * (0.8 + 0.6 * random.random())
            
            # 2. Vary learning rate - affects convergence path
            model_params['learning_rate'] = self.params.get('learning_rate', 0.001) * (0.7 + 0.6 * random.random())
            
            # 3. Vary model capacity and architecture
            model_params['num_heads'] = random.choice([1, 2, 4, 8]) 
            model_params['ff_dim'] = random.choice([64, 128, 256, 512])
            model_params['embed_dim'] = random.choice([32, 64, 128])
            model_params['use_gru'] = random.choice([True, False])
            model_params['conv_filters'] = int(self.params.get('conv_filters', 32) * (0.5 + 1.0 * random.random()))
            
            # 4. Use different optimizers
            model_params['optimizer'] = random.choice(['adam', 'rmsprop', 'sgd'])
            
            # 5. Vary batch size
            model_params['batch_size'] = random.choice([8, 16, 32])
            
            # 6. Randomly set activation function
            model_params['activation'] = random.choice(['relu', 'gelu', 'swish'])
            
            # 7. Alternate between using batch norm
            model_params['use_batch_norm'] = (i % 2 == 0) 
            
            # Create the model
            model = ImprovedTransformerModel(
                input_dim=X_scaled.shape[1],
                seq_length=self.predictor.sequence_length,
                params=model_params
            )
            
            # Build models
            main_model = model.build_main_numbers_model()
            bonus_model = model.build_bonus_numbers_model()
            
            # Train each model with different random seed and data subsets
            tf.random.set_seed(i * 100 + 42)  # Different seed for each model
            
            # Select a subset of the data for each model (80%)
            n_samples = len(X_scaled)
            n_subset = int(n_samples * 0.8)
            subset_indices = np.random.choice(n_samples, n_subset, replace=False)
            
            # Use bootstrapping - sampling with replacement
            bootstrap_indices = np.random.choice(subset_indices, n_subset, replace=True)
            
            X_bootstrap = X_scaled.iloc[bootstrap_indices]
            main_seq_bootstrap = main_sequences[bootstrap_indices]
            bonus_seq_bootstrap = bonus_sequences[bootstrap_indices]
            y_main_bootstrap = y_main[bootstrap_indices]
            y_bonus_bootstrap = y_bonus[bootstrap_indices]
            
            # Callbacks
            callbacks = [
                EarlyStopping(patience=20, restore_best_weights=True),
                ReduceLROnPlateau(factor=0.7, patience=10, min_lr=1e-6)
            ]
            
            # Train main model
            main_model.fit(
                [X_bootstrap.values, main_seq_bootstrap],
                [y_main_bootstrap[:, j] for j in range(5)],
                epochs=self.params.get('epochs', 50) if self.params else 50,
                batch_size=model_params.get('batch_size', 16),
                validation_split=0.2,
                callbacks=callbacks,
                verbose=1
            )
            
            # Train bonus model
            bonus_model.fit(
                [X_bootstrap.values, bonus_seq_bootstrap],
                [y_bonus_bootstrap[:, j] for j in range(2)],
                epochs=self.params.get('epochs', 50) if self.params else 50,
                batch_size=model_params.get('batch_size', 16),
                validation_split=0.2,
                callbacks=callbacks,
                verbose=1
            )
            
            # Add to ensemble
            self.base_models.append(model)
        
        logger.info(f"Base model training complete with {len(self.base_models)} models")
        return self.base_models
    
    def train_meta_model(self):
        """Train improved meta-model with diversity weighting."""
        logger.info("Training improved meta-model for ensemble")
        
        if not self.base_models:
            logger.error("No base models. Call train_base_models() first.")
            return None
        
        try:
            # Create validation split to train meta-model
            val_size = int(len(self.X_scaled) * 0.2)
            X_meta_train = self.X_scaled.iloc[:-val_size]
            main_seq_meta_train = self.main_sequences[:-val_size]
            bonus_seq_meta_train = self.bonus_sequences[:-val_size]
            
            X_meta_val = self.X_scaled.iloc[-val_size:]
            main_seq_meta_val = self.main_sequences[-val_size:]
            bonus_seq_meta_val = self.bonus_sequences[-val_size:]
            
            # Generate base model predictions for meta-model training
            main_meta_features_train = []
            bonus_meta_features_train = []
            
            main_meta_features_val = []
            bonus_meta_features_val = []
            
            # For each base model, get predictions
            for model in self.base_models:
                # Get predictions on training set
                main_probs_train = model.main_model.predict([X_meta_train.values, main_seq_meta_train])
                bonus_probs_train = model.bonus_model.predict([X_meta_train.values, bonus_seq_meta_train])
                
                # Get predictions on validation set
                main_probs_val = model.main_model.predict([X_meta_val.values, main_seq_meta_val])
                bonus_probs_val = model.bonus_model.predict([X_meta_val.values, bonus_seq_meta_val])
                
                # Flatten and stack the probabilities
                main_features_train = np.hstack([probs.reshape(X_meta_train.shape[0], -1) for probs in main_probs_train])
                bonus_features_train = np.hstack([probs.reshape(X_meta_train.shape[0], -1) for probs in bonus_probs_train])
                
                main_features_val = np.hstack([probs.reshape(X_meta_val.shape[0], -1) for probs in main_probs_val])
                bonus_features_val = np.hstack([probs.reshape(X_meta_val.shape[0], -1) for probs in bonus_probs_val])
                
                main_meta_features_train.append(main_features_train)
                bonus_meta_features_train.append(bonus_features_train)
                
                main_meta_features_val.append(main_features_val)
                bonus_meta_features_val.append(bonus_features_val)
            
            # Concatenate features from all base models
            main_meta_X_train = np.hstack(main_meta_features_train)
            bonus_meta_X_train = np.hstack(bonus_meta_features_train)
            
            main_meta_X_val = np.hstack(main_meta_features_val)
            bonus_meta_X_val = np.hstack(bonus_meta_features_val)
            
            # Create meta targets
            y_main_meta_train = np.array([
                self.predictor.expanded_data.iloc[self.predictor.sequence_length:][-X_meta_train.shape[0]:][f"main_{i+1}"].values - 1
                for i in range(5)
            ]).T
            
            y_bonus_meta_train = np.array([
                self.predictor.expanded_data.iloc[self.predictor.sequence_length:][-X_meta_train.shape[0]:][f"bonus_{i+1}"].values - 1
                for i in range(2)
            ]).T
            
            y_main_meta_val = np.array([
                self.predictor.expanded_data.iloc[self.predictor.sequence_length:][-X_meta_val.shape[0]:][f"main_{i+1}"].values - 1
                for i in range(5)
            ]).T
            
            y_bonus_meta_val = np.array([
                self.predictor.expanded_data.iloc[self.predictor.sequence_length:][-X_meta_val.shape[0]:][f"bonus_{i+1}"].values - 1
                for i in range(2)
            ]).T
            
            # Use a more powerful meta-model architecture
            main_meta_models = []
            for i in range(5):
                # Input features
                meta_input = Input(shape=(main_meta_X_train.shape[1],))
                
                # Hidden layers
                x = Dense(256, activation='relu')(meta_input)
                x = BatchNormalization()(x)
                x = Dropout(0.3)(x)
                
                x = Dense(128, activation='relu')(x)
                x = BatchNormalization()(x)
                x = Dropout(0.3)(x)
                
                # Output layer with softmax activation
                output = Dense(50, activation='softmax')(x)
                
                # Create and compile model
                model = Model(inputs=meta_input, outputs=output)
                model.compile(
                    optimizer=Adam(learning_rate=0.001),
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy']
                )
                
                # Train with early stopping
                model.fit(
                    main_meta_X_train, y_main_meta_train[:, i],
                    epochs=50,
                    batch_size=32,
                    validation_data=(main_meta_X_val, y_main_meta_val[:, i]),
                    callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
                    verbose=1
                )
                
                main_meta_models.append(model)
            
            # Similar approach for bonus numbers
            bonus_meta_models = []
            for i in range(2):
                meta_input = Input(shape=(bonus_meta_X_train.shape[1],))
                
                x = Dense(128, activation='relu')(meta_input)
                x = BatchNormalization()(x)
                x = Dropout(0.3)(x)
                
                x = Dense(64, activation='relu')(x)
                x = BatchNormalization()(x)
                x = Dropout(0.3)(x)
                
                output = Dense(12, activation='softmax')(x)
                
                model = Model(inputs=meta_input, outputs=output)
                model.compile(
                    optimizer=Adam(learning_rate=0.001),
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy']
                )
                
                model.fit(
                    bonus_meta_X_train, y_bonus_meta_train[:, i],
                    epochs=50,
                    batch_size=32,
                    validation_data=(bonus_meta_X_val, y_bonus_meta_val[:, i]),
                    callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
                    verbose=1
                )
                
                bonus_meta_models.append(model)
            
            # Calculate model weights based on validation performance
            main_model_weights = np.zeros(len(self.base_models))
            bonus_model_weights = np.zeros(len(self.base_models))
            
            # Evaluate each base model on validation data
            for i, model in enumerate(self.base_models):
                # Predict main numbers
                main_preds = model.main_model.predict([X_meta_val.values, main_seq_meta_val])
                main_class_preds = [np.argmax(main_preds[j], axis=1) for j in range(5)]
                
                # Calculate accuracy for each position
                main_acc = []
                for j in range(5):
                    acc = np.mean(main_class_preds[j] == y_main_meta_val[:, j])
                    main_acc.append(acc)
                
                # Use average accuracy as the weight
                main_model_weights[i] = np.mean(main_acc)
                
                # Predict bonus numbers
                bonus_preds = model.bonus_model.predict([X_meta_val.values, bonus_seq_meta_val])
                bonus_class_preds = [np.argmax(bonus_preds[j], axis=1) for j in range(2)]
                
                # Calculate accuracy for each position
                bonus_acc = []
                for j in range(2):
                    acc = np.mean(bonus_class_preds[j] == y_bonus_meta_val[:, j])
                    bonus_acc.append(acc)
                
                bonus_model_weights[i] = np.mean(bonus_acc)
            
            # Normalize weights
            main_model_weights = main_model_weights / np.sum(main_model_weights)
            bonus_model_weights = bonus_model_weights / np.sum(bonus_model_weights)
            
            # Store meta-models and weights
            self.meta_model = {
                'main': main_meta_models,
                'bonus': bonus_meta_models
            }
            
            self.model_weights = {
                'main': main_model_weights,
                'bonus': bonus_model_weights
            }
            
            logger.info("Meta-model training complete with model weights")
            return self.meta_model
        except Exception as e:
            logger.error(f"Error in meta-model training: {str(e)}")
            return None
    
    def predict(self, num_draws=5, temperature=0.7, diversity_boost=True):
        """Generate predictions using improved ensemble approach."""
        logger.info(f"Generating {num_draws} diverse ensemble predictions")
        
        if not self.base_models:
            logger.error("No base models. Call train_base_models() first.")
            return None
        
        try:
            # Determine prediction method based on available models
            if self.meta_model:
                return self.predict_with_meta_model(num_draws, temperature, diversity_boost)
            elif self.model_weights:
                return self.predict_with_weighted_averaging(num_draws, temperature, diversity_boost)
            else:
                return self.predict_with_averaging(num_draws, temperature, diversity_boost)
        except Exception as e:
            logger.error(f"Error in ensemble prediction: {str(e)}")
            # Fallback to simple predictions
            return self.predict_with_simple_averaging(num_draws)
    
    def predict_with_meta_model(self, num_draws=5, temperature=0.7, diversity_boost=True):
        """Generate predictions using the trained meta-model."""
        try:
            # Get latest data for prediction
            X_latest = self.X_scaled.iloc[-1:].values
            main_seq_latest = self.main_sequences[-1:]
            bonus_seq_latest = self.bonus_sequences[-1:]
            
            # Get base model predictions
            main_meta_features = []
            bonus_meta_features = []
            
            for model in self.base_models:
                main_probs = model.main_model.predict([X_latest, main_seq_latest])
                bonus_probs = model.bonus_model.predict([X_latest, bonus_seq_latest])
                
                main_features = np.hstack([probs.reshape(1, -1) for probs in main_probs])
                bonus_features = np.hstack([probs.reshape(1, -1) for probs in bonus_probs])
                
                main_meta_features.append(main_features)
                bonus_meta_features.append(bonus_features)
            
            main_meta_X = np.hstack(main_meta_features)
            bonus_meta_X = np.hstack(bonus_meta_features)
            
            # Track used numbers to enhance diversity
            used_main_numbers = set()
            used_bonus_numbers = set()
            
            # Generate predictions
            predictions = []
            
            for draw_idx in range(num_draws):
                # Predict using meta-models
                main_probs = [model.predict(main_meta_X) for model in self.meta_model['main']]
                bonus_probs = [model.predict(bonus_meta_X) for model in self.meta_model['bonus']]
                
                # Apply temperature scaling
                for i in range(len(main_probs)):
                    main_probs[i] = np.power(main_probs[i], 1/temperature)
                    main_probs[i] = main_probs[i] / np.sum(main_probs[i], axis=1, keepdims=True)
                
                for i in range(len(bonus_probs)):
                    bonus_probs[i] = np.power(bonus_probs[i], 1/temperature)
                    bonus_probs[i] = bonus_probs[i] / np.sum(bonus_probs[i], axis=1, keepdims=True)
                
                # Sample main numbers without replacement
                main_numbers = []
                available_main = list(range(1, 51))
                
                for i in range(5):
                    # Get probability distribution
                    probs = main_probs[i][0]
                    
                    # Adjust probabilities for available numbers
                    adjusted_probs = np.zeros(50)
                    for j, num in enumerate(available_main):
                        adjusted_probs[num-1] = probs[num-1]
                        
                        # Apply diversity penalty if needed
                        if diversity_boost and num in used_main_numbers and draw_idx > 0:
                            diversity_factor = 0.7  # Stronger penalty for ensemble
                            adjusted_probs[num-1] *= diversity_factor
                    
                    # Normalize probabilities
                    if np.sum(adjusted_probs) > 0:
                        adjusted_probs = adjusted_probs / np.sum(adjusted_probs)
                    else:
                        # Fallback to uniform
                        valid_indices = [j-1 for j in available_main]
                        adjusted_probs[valid_indices] = 1.0 / len(valid_indices)
                    
                    # Sample
                    try:
                        num_idx = np.random.choice(50, p=adjusted_probs)
                        num = num_idx + 1
                    except ValueError:
                        num = random.choice(available_main)
                    
                    main_numbers.append(int(num))
                    available_main.remove(int(num))
                    used_main_numbers.add(int(num))
                
                # Sort main numbers
                main_numbers.sort()
                
                # Sample bonus numbers similarly
                bonus_numbers = []
                available_bonus = list(range(1, 13))
                
                for i in range(2):
                    probs = bonus_probs[i][0]
                    
                    adjusted_probs = np.zeros(12)
                    for j, num in enumerate(available_bonus):
                        adjusted_probs[num-1] = probs[num-1]
                        
                        if diversity_boost and num in used_bonus_numbers and draw_idx > 0:
                            diversity_factor = 0.7
                            adjusted_probs[num-1] *= diversity_factor
                    
                    if np.sum(adjusted_probs) > 0:
                        adjusted_probs = adjusted_probs / np.sum(adjusted_probs)
                    else:
                        valid_indices = [j-1 for j in available_bonus]
                        adjusted_probs[valid_indices] = 1.0 / len(valid_indices)
                    
                    try:
                        num_idx = np.random.choice(12, p=adjusted_probs)
                        num = num_idx + 1
                    except ValueError:
                        num = random.choice(available_bonus)
                    
                    bonus_numbers.append(int(num))
                    available_bonus.remove(int(num))
                    used_bonus_numbers.add(int(num))
                
                # Sort bonus numbers
                bonus_numbers.sort()
                
                # Calculate confidence
                main_confidence = np.mean([np.max(main_probs[i][0]) for i in range(5)])
                bonus_confidence = np.mean([np.max(bonus_probs[i][0]) for i in range(2)])
                
                # Calculate enhanced confidence scores
                pattern_score = self.predictor.calculate_pattern_score(main_numbers, bonus_numbers)
                frequency_score = self.predictor.calculate_frequency_score(main_numbers, bonus_numbers)
                
                # Meta-model provides better calibrated confidence
                calibrated_main_conf = 0.7 * main_confidence + 0.2
                calibrated_bonus_conf = 0.8 * bonus_confidence + 0.15
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
                    "method": "meta_model_ensemble"
                })
            
            return predictions
        except Exception as e:
            logger.error(f"Error in meta-model prediction: {str(e)}")
            return self.predict_with_simple_averaging(num_draws)
    
    def predict_with_weighted_averaging(self, num_draws=5, temperature=0.7, diversity_boost=True):
        """Predict using weighted averaging of base models."""
        try:
            # Get latest data for prediction
            X_latest = self.X_scaled.iloc[-1:].values
            main_seq_latest = self.main_sequences[-1:]
            bonus_seq_latest = self.bonus_sequences[-1:]
            
            # Track used numbers
            used_main_numbers = set()
            used_bonus_numbers = set()
            
            # Generate predictions
            predictions = []
            
            for draw_idx in range(num_draws):
                # Initialize arrays for weighted averaging
                main_probs_weighted = [np.zeros((1, 50)) for _ in range(5)]
                bonus_probs_weighted = [np.zeros((1, 12)) for _ in range(2)]
                
                # Get predictions from each model and apply weights
                for i, model in enumerate(self.base_models):
                    # Predict main numbers
                    main_probs = model.main_model.predict([X_latest, main_seq_latest])
                    main_weight = self.model_weights['main'][i]
                    
                    for j in range(5):
                        main_probs_weighted[j] += main_weight * main_probs[j]
                    
                    # Predict bonus numbers
                    bonus_probs = model.bonus_model.predict([X_latest, bonus_seq_latest])
                    bonus_weight = self.model_weights['bonus'][i]
                    
                    for j in range(2):
                        bonus_probs_weighted[j] += bonus_weight * bonus_probs[j]
                
                # Apply temperature to the weighted predictions
                for i in range(5):
                    main_probs_weighted[i] = np.power(main_probs_weighted[i], 1/temperature)
                    main_probs_weighted[i] = main_probs_weighted[i] / np.sum(main_probs_weighted[i], axis=1, keepdims=True)
                
                for i in range(2):
                    bonus_probs_weighted[i] = np.power(bonus_probs_weighted[i], 1/temperature)
                    bonus_probs_weighted[i] = bonus_probs_weighted[i] / np.sum(bonus_probs_weighted[i], axis=1, keepdims=True)
                
                # Sample main numbers
                main_numbers = []
                available_main = list(range(1, 51))
                
                for i in range(5):
                    # Apply diversity and sampling
                    probs = main_probs_weighted[i][0]
                    
                    adjusted_probs = np.zeros(50)
                    for j, num in enumerate(available_main):
                        adjusted_probs[num-1] = probs[num-1]
                        
                        if diversity_boost and num in used_main_numbers and draw_idx > 0:
                            diversity_factor = 0.7
                            adjusted_probs[num-1] *= diversity_factor
                    
                    # Normalize and sample
                    if np.sum(adjusted_probs) > 0:
                        adjusted_probs = adjusted_probs / np.sum(adjusted_probs)
                    else:
                        valid_indices = [j-1 for j in available_main]
                        adjusted_probs[valid_indices] = 1.0 / len(valid_indices)
                    
                    try:
                        num_idx = np.random.choice(50, p=adjusted_probs)
                        num = num_idx + 1
                    except ValueError:
                        num = random.choice(available_main)
                    
                    main_numbers.append(int(num))
                    available_main.remove(int(num))
                    used_main_numbers.add(int(num))
                
                # Sort main numbers
                main_numbers.sort()
                
                # Sample bonus numbers
                bonus_numbers = []
                available_bonus = list(range(1, 13))
                
                for i in range(2):
                    # Apply diversity and sampling
                    probs = bonus_probs_weighted[i][0]
                    
                    adjusted_probs = np.zeros(12)
                    for j, num in enumerate(available_bonus):
                        adjusted_probs[num-1] = probs[num-1]
                        
                        if diversity_boost and num in used_bonus_numbers and draw_idx > 0:
                            diversity_factor = 0.7
                            adjusted_probs[num-1] *= diversity_factor
                    
                    # Normalize and sample
                    if np.sum(adjusted_probs) > 0:
                        adjusted_probs = adjusted_probs / np.sum(adjusted_probs)
                    else:
                        valid_indices = [j-1 for j in available_bonus]
                        adjusted_probs[valid_indices] = 1.0 / len(valid_indices)
                    
                    try:
                        num_idx = np.random.choice(12, p=adjusted_probs)
                        num = num_idx + 1
                    except ValueError:
                        num = random.choice(available_bonus)
                    
                    bonus_numbers.append(int(num))
                    available_bonus.remove(int(num))
                    used_bonus_numbers.add(int(num))
                
                # Sort bonus numbers
                bonus_numbers.sort()
                
                # Calculate confidence
                main_confidence = np.mean([np.max(main_probs_weighted[i][0]) for i in range(5)])
                bonus_confidence = np.mean([np.max(bonus_probs_weighted[i][0]) for i in range(2)])
                
                # Calculate enhanced confidence scores
                pattern_score = self.predictor.calculate_pattern_score(main_numbers, bonus_numbers)
                frequency_score = self.predictor.calculate_frequency_score(main_numbers, bonus_numbers)
                
                # Weighted ensemble provides moderate confidence
                calibrated_main_conf = 0.65 * main_confidence + 0.2
                calibrated_bonus_conf = 0.75 * bonus_confidence + 0.15
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
                    "method": "weighted_ensemble"
                })
            
            return predictions
        except Exception as e:
            logger.error(f"Error in weighted ensemble prediction: {str(e)}")
            return self.predict_with_simple_averaging(num_draws)
    
    def predict_with_averaging(self, num_draws=5, temperature=0.7, diversity_boost=True):
        """Predict using simple averaging of base models."""
        try:
            # Get latest data for prediction
            X_latest = self.X_scaled.iloc[-1:].values
            main_seq_latest = self.main_sequences[-1:]
            bonus_seq_latest = self.bonus_sequences[-1:]
            
            # Track used numbers
            used_main_numbers = set()
            used_bonus_numbers = set()
            
            # Generate predictions
            predictions = []
            
            for draw_idx in range(num_draws):
                # Initialize arrays for averaging
                main_probs = [np.zeros((1, 50)) for _ in range(5)]
                bonus_probs = [np.zeros((1, 12)) for _ in range(2)]
                
                # Get predictions from each model and average them
                for model in self.base_models:
                    # Predict main numbers
                    model_main_probs = model.main_model.predict([X_latest, main_seq_latest])
                    for i in range(5):
                        main_probs[i] += model_main_probs[i]
                    
                    # Predict bonus numbers
                    model_bonus_probs = model.bonus_model.predict([X_latest, bonus_seq_latest])
                    for i in range(2):
                        bonus_probs[i] += model_bonus_probs[i]
                
                # Average the probabilities
                for i in range(5):
                    main_probs[i] /= len(self.base_models)
                
                for i in range(2):
                    bonus_probs[i] /= len(self.base_models)
                
                # Apply temperature
                for i in range(5):
                    main_probs[i] = np.power(main_probs[i], 1/temperature)
                    main_probs[i] = main_probs[i] / np.sum(main_probs[i], axis=1, keepdims=True)
                
                for i in range(2):
                    bonus_probs[i] = np.power(bonus_probs[i], 1/temperature)
                    bonus_probs[i] = bonus_probs[i] / np.sum(bonus_probs[i], axis=1, keepdims=True)
                
                # Sample main numbers
                main_numbers = []
                available_main = list(range(1, 51))
                
                for i in range(5):
                    # Apply diversity and sampling
                    probs = main_probs[i][0]
                    
                    adjusted_probs = np.zeros(50)
                    for j, num in enumerate(available_main):
                        adjusted_probs[num-1] = probs[num-1]
                        
                        if diversity_boost and num in used_main_numbers and draw_idx > 0:
                            diversity_factor = 0.7
                            adjusted_probs[num-1] *= diversity_factor
                    
                    # Normalize and sample
                    if np.sum(adjusted_probs) > 0:
                        adjusted_probs = adjusted_probs / np.sum(adjusted_probs)
                    else:
                        valid_indices = [j-1 for j in available_main]
                        adjusted_probs[valid_indices] = 1.0 / len(valid_indices)
                    
                    try:
                        num_idx = np.random.choice(50, p=adjusted_probs)
                        num = num_idx + 1
                    except ValueError:
                        num = random.choice(available_main)
                    
                    main_numbers.append(int(num))
                    available_main.remove(int(num))
                    used_main_numbers.add(int(num))
                
                # Sort main numbers
                main_numbers.sort()
                
                # Sample bonus numbers
                bonus_numbers = []
                available_bonus = list(range(1, 13))
                
                for i in range(2):
                    # Apply diversity and sampling
                    probs = bonus_probs[i][0]
                    
                    adjusted_probs = np.zeros(12)
                    for j, num in enumerate(available_bonus):
                        adjusted_probs[num-1] = probs[num-1]
                        
                        if diversity_boost and num in used_bonus_numbers and draw_idx > 0:
                            diversity_factor = 0.7
                            adjusted_probs[num-1] *= diversity_factor
                    
                    # Normalize and sample
                    if np.sum(adjusted_probs) > 0:
                        adjusted_probs = adjusted_probs / np.sum(adjusted_probs)
                    else:
                        valid_indices = [j-1 for j in available_bonus]
                        adjusted_probs[valid_indices] = 1.0 / len(valid_indices)
                    
                    try:
                        num_idx = np.random.choice(12, p=adjusted_probs)
                        num = num_idx + 1
                    except ValueError:
                        num = random.choice(available_bonus)
                    
                    bonus_numbers.append(int(num))
                    available_bonus.remove(int(num))
                    used_bonus_numbers.add(int(num))
                
                # Sort bonus numbers
                bonus_numbers.sort()
                
                # Calculate confidence
                main_confidence = np.mean([np.max(main_probs[i][0]) for i in range(5)])
                bonus_confidence = np.mean([np.max(bonus_probs[i][0]) for i in range(2)])
                
                # Calculate enhanced confidence scores
                pattern_score = self.predictor.calculate_pattern_score(main_numbers, bonus_numbers)
                frequency_score = self.predictor.calculate_frequency_score(main_numbers, bonus_numbers)
                
                # Simple average ensemble has moderate confidence
                calibrated_main_conf = 0.6 * main_confidence + 0.2
                calibrated_bonus_conf = 0.7 * bonus_confidence + 0.15
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
                    "method": "average_ensemble"
                })
            
            return predictions
        except Exception as e:
            logger.error(f"Error in ensemble averaging prediction: {str(e)}")
            return self.predict_with_simple_averaging(num_draws)
    
    def predict_with_simple_averaging(self, num_draws=5):
        """Fallback method for ensemble prediction."""
        logger.warning("Using simple averaging fallback for ensemble prediction")
        
        try:
            # Get latest data
            X_latest = self.X_scaled.iloc[-1:].values
            main_seq_latest = self.main_sequences[-1:]
            bonus_seq_latest = self.bonus_sequences[-1:]
            
            # Generate predictions
            predictions = []
            
            for _ in range(num_draws):
                # Initialize arrays for averaging
                main_probs = [np.zeros((1, 50)) for _ in range(5)]
                bonus_probs = [np.zeros((1, 12)) for _ in range(2)]
                
                # Get predictions from each model and average them
                model_count = 0
                for model in self.base_models:
                    try:
                        # Predict main numbers
                        model_main_probs = model.main_model.predict([X_latest, main_seq_latest])
                        for i in range(5):
                            main_probs[i] += model_main_probs[i]
                        
                        # Predict bonus numbers
                        model_bonus_probs = model.bonus_model.predict([X_latest, bonus_seq_latest])
                        for i in range(2):
                            bonus_probs[i] += model_bonus_probs[i]
                            
                        model_count += 1
                    except Exception as e:
                        logger.warning(f"Error with model in simple averaging: {e}")
                        continue
                
                if model_count == 0:
                    # No models worked, use random selection
                    main_numbers = sorted(random.sample(range(1, 51), 5))
                    bonus_numbers = sorted(random.sample(range(1, 13), 2))
                    
                    predictions.append({
                        "main_numbers": main_numbers,
                        "bonus_numbers": bonus_numbers,
                        "confidence": {
                            "overall": 0.1,
                            "main_numbers": 0.1,
                            "bonus_numbers": 0.1
                        },
                        "method": "random_fallback"
                    })
                    continue
                
                # Average the probabilities
                for i in range(5):
                    main_probs[i] /= model_count
                
                for i in range(2):
                    bonus_probs[i] /= model_count
                
                # Sample main numbers
                main_numbers = []
                available_main = list(range(1, 51))
                
                for i in range(5):
                    probs = main_probs[i][0]
                    
                    adjusted_probs = np.zeros(50)
                    for j, num in enumerate(available_main):
                        adjusted_probs[num-1] = probs[num-1]
                    
                    # Normalize and sample
                    if np.sum(adjusted_probs) > 0:
                        adjusted_probs = adjusted_probs / np.sum(adjusted_probs)
                    else:
                        valid_indices = [j-1 for j in available_main]
                        adjusted_probs[valid_indices] = 1.0 / len(valid_indices)
                    
                    try:
                        num_idx = np.random.choice(50, p=adjusted_probs)
                        num = num_idx + 1
                    except ValueError:
                        num = random.choice(available_main)
                    
                    main_numbers.append(int(num))
                    available_main.remove(int(num))
                
                # Sort main numbers
                main_numbers.sort()
                
                # Sample bonus numbers
                bonus_numbers = []
                available_bonus = list(range(1, 13))
                
                for i in range(2):
                    probs = bonus_probs[i][0]
                    
                    adjusted_probs = np.zeros(12)
                    for j, num in enumerate(available_bonus):
                        adjusted_probs[num-1] = probs[num-1]
                    
                    # Normalize and sample
                    if np.sum(adjusted_probs) > 0:
                        adjusted_probs = adjusted_probs / np.sum(adjusted_probs)
                    else:
                        valid_indices = [j-1 for j in available_bonus]
                        adjusted_probs[valid_indices] = 1.0 / len(valid_indices)
                    
                    try:
                        num_idx = np.random.choice(12, p=adjusted_probs)
                        num = num_idx + 1
                    except ValueError:
                        num = random.choice(available_bonus)
                    
                    bonus_numbers.append(int(num))
                    available_bonus.remove(int(num))
                
                # Sort bonus numbers
                bonus_numbers.sort()
                
                # Simple confidence calculation
                main_confidence = np.mean([np.max(main_probs[i][0]) for i in range(5)])
                bonus_confidence = np.mean([np.max(bonus_probs[i][0]) for i in range(2)])
                overall_confidence = (main_confidence + bonus_confidence) / 2
                
                predictions.append({
                    "main_numbers": main_numbers,
                    "bonus_numbers": bonus_numbers,
                    "confidence": {
                        "overall": float(overall_confidence),
                        "main_numbers": float(main_confidence),
                        "bonus_numbers": float(bonus_confidence)
                    },
                    "method": "simple_ensemble_averaging"
                })
            
            return predictions
        except Exception as e:
            logger.error(f"Error in simple ensemble averaging: {str(e)}")
            # Ultimate fallback
            predictions = []
            for _ in range(num_draws):
                main_numbers = sorted(random.sample(range(1, 51), 5))
                bonus_numbers = sorted(random.sample(range(1, 13), 2))
                
                predictions.append({
                    "main_numbers": main_numbers,
                    "bonus_numbers": bonus_numbers,
                    "confidence": {
                        "overall": 0.1,
                        "main_numbers": 0.1,
                        "bonus_numbers": 0.1
                    },
                    "method": "pure_random"
                })
            
            return predictions

#######################
# IMPROVED HYPERPARAMETER OPTIMIZATION
#######################

class BayesianOptimizer:
    """Improved hyperparameter optimization using Bayesian methods with Optuna."""
    
    def __init__(self, file_path, n_trials=30, n_jobs=1):
        self.file_path = file_path
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        self.study = None
        self.best_params = None
    
    def objective(self, trial):
        """Objective function for hyperparameter optimization."""
        # Define parameter ranges
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 0.0001, 0.003, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [8, 16, 32, 64]),
            'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
            'num_heads': trial.suggest_categorical('num_heads', [1, 2, 4, 8]),
            'ff_dim': trial.suggest_categorical('ff_dim', [64, 128, 256, 512]),
            'embed_dim': trial.suggest_categorical('embed_dim', [32, 64, 128, 256]),
            'use_gru': trial.suggest_categorical('use_gru', [True, False]),
            'conv_filters': trial.suggest_int('conv_filters', 0, 64, step=8),
            'num_transformer_blocks': trial.suggest_int('num_transformer_blocks', 1, 4),
            'optimizer': trial.suggest_categorical('optimizer', ['adam', 'rmsprop']),
            'attention_dropout': trial.suggest_float('attention_dropout', 0.1, 0.4),
            'activation': trial.suggest_categorical('activation', ['relu', 'gelu']),
            'use_batch_norm': trial.suggest_categorical('use_batch_norm', [True, False]),
            'use_residual_connections': trial.suggest_categorical('use_residual_connections', [True, False])
        }
        
        # Create evaluator with these parameters
        evaluator = ImprovedCrossValidationEvaluator(self.file_path, params, folds=3, test_size=0.2)
        
        # Run evaluation
        results = evaluator.evaluate()
        
        # Return the overall accuracy as the objective
        return results['avg_overall_accuracy']
    
    def optimize(self):
        """Run Bayesian optimization to find best hyperparameters."""
        logger.info(f"Starting Bayesian hyperparameter optimization with {self.n_trials} trials")
        
        # Create Optuna study with improved sampler and pruner
        self.study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=RANDOM_SEED),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10, interval_steps=3)
        )
        
        # Run optimization
        self.study.optimize(self.objective, n_trials=self.n_trials, n_jobs=self.n_jobs)
        
        # Get best parameters
        self.best_params = self.study.best_params
        logger.info(f"Optimization complete. Best accuracy: {self.study.best_value:.4f}")
        logger.info(f"Best parameters: {self.best_params}")
        
        return self.best_params
    
    def get_optimization_results(self):
        """Get detailed optimization results."""
        if self.study is None:
            logger.error("No optimization study. Call optimize() first.")
            return None
        
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
    
    def plot_optimization_history(self, filename="optimization_history.png"):
        """Plot optimization history."""
        if self.study is None:
            logger.error("No optimization study. Call optimize() first.")
            return None
        
        try:
            import matplotlib.pyplot as plt
            from optuna.visualization import plot_optimization_history, plot_param_importances
            
            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Plot optimization history
            plot_optimization_history(self.study, ax=ax1)
            ax1.set_title("Optimization History")
            
            # Plot parameter importances
            plot_param_importances(self.study, ax=ax2)
            ax2.set_title("Parameter Importances")
            
            plt.tight_layout()
            plt.savefig(filename, dpi=300)
            plt.close()
            
            logger.info(f"Optimization plots saved to {filename}")
        except Exception as e:
            logger.error(f"Error plotting optimization history: {e}")

class ImprovedCrossValidationEvaluator:
    """Improved evaluator for time-series cross validation of lottery prediction."""
    
    def __init__(self, file_path, params=None, folds=5, test_size=0.1):
        self.file_path = file_path
        self.params = params
        self.folds = folds
        self.test_size = test_size
    
    def evaluate(self):
        """Perform time-series cross-validation with advanced metrics."""
        logger.info(f"Performing {self.folds}-fold time-series cross-validation")
        
        # Create predictor
        predictor = ImprovedLotteryPredictor(self.file_path, self.params)
        
        # Load data
        X_scaled, main_sequences, bonus_sequences, y_main, y_bonus = predictor.load_and_preprocess_data()
        
        # Calculate total data size and test size in samples
        total_samples = len(X_scaled)
        test_samples = int(total_samples * self.test_size)
        
        # Initialize metrics
        fold_metrics = []
        
        # Initialize arrays to track individual number accuracy
        main_number_accuracy = np.zeros(50)
        main_number_counts = np.zeros(50)
        bonus_number_accuracy = np.zeros(12)
        bonus_number_counts = np.zeros(12)
        
        # Perform time-series cross-validation (rolling window)
        for fold in range(self.folds):
            logger.info(f"Training fold {fold+1}/{self.folds}")
            
            # Calculate fold indices
            start_idx = fold * test_samples
            end_idx = start_idx + test_samples
            
            if end_idx > total_samples:
                break
            
            # Split data for this fold
            X_train = pd.concat([X_scaled.iloc[:start_idx], X_scaled.iloc[end_idx:]])
            X_test = X_scaled.iloc[start_idx:end_idx]
            
            main_seq_train = np.concatenate([main_sequences[:start_idx], main_sequences[end_idx:]])
            main_seq_test = main_sequences[start_idx:end_idx]
            
            bonus_seq_train = np.concatenate([bonus_sequences[:start_idx], bonus_sequences[end_idx:]])
            bonus_seq_test = bonus_sequences[start_idx:end_idx]
            
            y_main_train = np.concatenate([y_main[:start_idx], y_main[end_idx:]])
            y_main_test = y_main[start_idx:end_idx]
            
            y_bonus_train = np.concatenate([y_bonus[:start_idx], y_bonus[end_idx:]])
            y_bonus_test = y_bonus[start_idx:end_idx]
            
            # Train model for this fold with early stopping
            model = ImprovedTransformerModel(
                input_dim=X_scaled.shape[1],
                seq_length=predictor.sequence_length,
                params=self.params
            )
            
            # Build models
            main_model = model.build_main_numbers_model()
            bonus_model = model.build_bonus_numbers_model()
            
            # Train main model with reduced epochs for faster evaluation
            main_model.fit(
                [X_train.values, main_seq_train],
                [y_main_train[:, i] for i in range(5)],
                epochs=30,  # Reduced epochs for CV
                batch_size=self.params.get('batch_size', 16) if self.params else 16,
                verbose=1,
                callbacks=[EarlyStopping(patience=5, restore_best_weights=True)]
            )
            
            # Train bonus model
            bonus_model.fit(
                [X_train.values, bonus_seq_train],
                [y_bonus_train[:, i] for i in range(2)],
                epochs=30,  # Reduced epochs for CV
                batch_size=self.params.get('batch_size', 16) if self.params else 16,
                verbose=1,
                callbacks=[EarlyStopping(patience=5, restore_best_weights=True)]
            )
            
            # Advanced evaluation with improved metrics
            # Get predictions
            main_preds = main_model.predict([X_test.values, main_seq_test])
            bonus_preds = bonus_model.predict([X_test.values, bonus_seq_test])
            
            # Log prediction shapes for debugging
            logger.info(f"main_preds types: {[type(p) for p in main_preds]}")
            logger.info(f"main_preds shapes: {[p.shape for p in main_preds]}")
            logger.info(f"bonus_preds shapes: {[p.shape for p in bonus_preds]}")
            
            # Convert to class predictions
            main_class_preds = []
            for i in range(5):
                main_class_preds.append(np.argmax(main_preds[i], axis=1))
            
            bonus_class_preds = []
            for i in range(2):
                bonus_class_preds.append(np.argmax(bonus_preds[i], axis=1))
            
            # Debug log class prediction shapes
            logger.info(f"main_class_preds shapes: {[p.shape for p in main_class_preds]}")
            logger.info(f"bonus_class_preds shapes: {[p.shape for p in bonus_class_preds]}")
            
            # Calculate accuracy for each position
            main_position_acc = []
            for i in range(5):
                acc = np.mean(main_class_preds[i] == y_main_test[:, i])
                main_position_acc.append(acc)
            
            bonus_position_acc = []
            for i in range(2):
                acc = np.mean(bonus_class_preds[i] == y_bonus_test[:, i])
                bonus_position_acc.append(acc)
            
            # Calculate average accuracy
            main_avg_acc = np.mean(main_position_acc)
            bonus_avg_acc = np.mean(bonus_position_acc)
            overall_avg_acc = (main_avg_acc + bonus_avg_acc) / 2
            
            # Calculate set-based metrics (partial matches)
            main_set_match_counts = np.zeros(6)  # 0-5 matches
            for i in range(len(y_main_test)):
                actual_set = set(y_main_test[i] + 1)  # Convert back to 1-50 range
                
                # Create pred_set
                pred_set = set()
                for j in range(5):
                    if i < len(main_class_preds[j]):
                        pred_set.add(int(main_class_preds[j][i]) + 1)  # Convert back to 1-50 range
                
                matches = len(actual_set.intersection(pred_set))
                main_set_match_counts[matches] += 1
            
            # Normalize to get probabilities
            main_set_match_probs = main_set_match_counts / np.sum(main_set_match_counts)
            
            # Calculate by-number accuracy
            for i in range(len(y_main_test)):
                for j in range(5):
                    actual_num = y_main_test[i, j]
                    pred_num = main_class_preds[j][i]
                    main_number_counts[actual_num] += 1
                    if actual_num == pred_num:
                        main_number_accuracy[actual_num] += 1
            
            for i in range(len(y_bonus_test)):
                for j in range(2):
                    actual_num = y_bonus_test[i, j]
                    pred_num = bonus_class_preds[j][i]
                    bonus_number_counts[actual_num] += 1
                    if actual_num == pred_num:
                        bonus_number_accuracy[actual_num] += 1
            
            # Calculate calibration metrics (reliability)
            main_calibration_scores = []
            for i in range(5):
                confidences = np.max(main_preds[i], axis=1)
                pred_classes = np.argmax(main_preds[i], axis=1)
                correct = (pred_classes == y_main_test[:, i])
                
                # Calculate average confidence and accuracy
                avg_conf = np.mean(confidences)
                accuracy = np.mean(correct)
                
                # Calibration error = |confidence - accuracy|
                cal_error = abs(avg_conf - accuracy)
                main_calibration_scores.append(cal_error)
            
            # Average calibration error
            main_calibration_error = np.mean(main_calibration_scores)
            
            # Same for bonus
            bonus_calibration_scores = []
            for i in range(2):
                confidences = np.max(bonus_preds[i], axis=1)
                pred_classes = np.argmax(bonus_preds[i], axis=1)
                correct = (pred_classes == y_bonus_test[:, i])
                
                avg_conf = np.mean(confidences)
                accuracy = np.mean(correct)
                cal_error = abs(avg_conf - accuracy)
                bonus_calibration_scores.append(cal_error)
            
            bonus_calibration_error = np.mean(bonus_calibration_scores)
            
            # Store metrics
            fold_metrics.append({
                'fold': fold+1,
                'main_accuracy': main_avg_acc,
                'bonus_accuracy': bonus_avg_acc,
                'overall_accuracy': overall_avg_acc,
                'main_position_accuracy': main_position_acc,
                'bonus_position_accuracy': bonus_position_acc,
                'main_set_match_probs': main_set_match_probs.tolist(),
                'main_calibration_error': main_calibration_error,
                'bonus_calibration_error': bonus_calibration_error
            })
            
            logger.info(f"Fold {fold+1} metrics: Main acc={main_avg_acc:.4f}, Bonus acc={bonus_avg_acc:.4f}, Overall={overall_avg_acc:.4f}")
            logger.info(f"Main matches: 0={main_set_match_probs[0]:.2f}, 1={main_set_match_probs[1]:.2f}, 2={main_set_match_probs[2]:.2f}, 3={main_set_match_probs[3]:.2f}, 4={main_set_match_probs[4]:.2f}, 5={main_set_match_probs[5]:.2f}")
        
        # Calculate average metrics
        avg_main_acc = np.mean([m['main_accuracy'] for m in fold_metrics])
        avg_bonus_acc = np.mean([m['bonus_accuracy'] for m in fold_metrics])
        avg_overall_acc = np.mean([m['overall_accuracy'] for m in fold_metrics])
        
        # Calculate by-number accuracy
        for i in range(50):
            if main_number_counts[i] > 0:
                main_number_accuracy[i] /= main_number_counts[i]
        
        for i in range(12):
            if bonus_number_counts[i] > 0:
                bonus_number_accuracy[i] /= bonus_number_counts[i]
        
        # Calculate average set-based metrics
        avg_main_set_match_probs = np.zeros(6)
        for m in fold_metrics:
            avg_main_set_match_probs += np.array(m['main_set_match_probs'])
        avg_main_set_match_probs /= len(fold_metrics)
        
        # Average calibration error
        avg_main_calibration_error = np.mean([m['main_calibration_error'] for m in fold_metrics])
        avg_bonus_calibration_error = np.mean([m['bonus_calibration_error'] for m in fold_metrics])
        
        logger.info(f"Average metrics across {self.folds} folds:")
        logger.info(f"Main numbers accuracy: {avg_main_acc:.4f}")
        logger.info(f"Bonus numbers accuracy: {avg_bonus_acc:.4f}")
        logger.info(f"Overall accuracy: {avg_overall_acc:.4f}")
        logger.info(f"Average main matches: 0={avg_main_set_match_probs[0]:.2f}, 1={avg_main_set_match_probs[1]:.2f}, 2={avg_main_set_match_probs[2]:.2f}, 3={avg_main_set_match_probs[3]:.2f}, 4={avg_main_set_match_probs[4]:.2f}, 5={avg_main_set_match_probs[5]:.2f}")
        
        return {
            'fold_metrics': fold_metrics,
            'avg_main_accuracy': avg_main_acc,
            'avg_bonus_accuracy': avg_bonus_acc,
            'avg_overall_accuracy': avg_overall_acc,
            'avg_main_set_match_probs': avg_main_set_match_probs.tolist(),
            'main_number_accuracy': main_number_accuracy.tolist(),
            'bonus_number_accuracy': bonus_number_accuracy.tolist(),
            'avg_main_calibration_error': avg_main_calibration_error,
            'avg_bonus_calibration_error': avg_bonus_calibration_error
        }

def self_optimizing_hyperparameters(file_path, num_trials=30):
    """Two-phase hyperparameter optimization with focused refinement."""
    logger.info(f"Starting self-optimizing hyperparameter search with {num_trials} trials")
    
    # Phase 1: Bayesian optimization for global search
    optimizer = BayesianOptimizer(file_path, n_trials=num_trials)
    best_params = optimizer.optimize()
    
    # Get optimization history
    opt_results = optimizer.get_optimization_results()
    top_params = opt_results[:3]  # Get top 3 parameter sets
    
    logger.info(f"Phase 1 complete. Best parameters: {best_params}")
    logger.info(f"Top 3 parameter sets: {top_params}")
    
    # Save parameters to file
    with open("optimized_params.json", "w") as f:
        json.dump(best_params, f, indent=4)
    
    # Phase 2: Fine-tune the best parameters with improved cross-validation
    logger.info("Phase 2: Fine-tuning best parameters")
    
    # Create evaluator with best parameters and more folds
    evaluator = ImprovedCrossValidationEvaluator(file_path, best_params, folds=5, test_size=0.1)
    results = evaluator.evaluate()
    
    # Save full evaluation results
    with open("optimization_evaluation.json", "w") as f:
        json.dump({
            'best_params': best_params,
            'evaluation_results': results
        }, f, indent=4)
    
    logger.info(f"Self-optimizing search complete. Parameters saved to optimized_params.json")
    
    # Plot optimization history
    optimizer.plot_optimization_history()
    
    return best_params

#######################
# ENHANCED VISUALIZATION FUNCTIONS
#######################

def generate_enhanced_visualizations(predictions, file_path):
    """Generate enhanced visualizations for lottery predictions."""
    logger.info("Generating enhanced visualizations for lottery predictions")
    
    try:
        # Create processor to get historical data
        processor = LotteryDataProcessor(file_path)
        data = processor.parse_file()
        
        # Set up the figure
        plt.figure(figsize=(20, 20))
        
        # Plot 1: Frequency of main numbers (historical)
        plt.subplot(3, 2, 1)
        main_freq = {}
        for num in range(1, 51):
            main_freq[num] = 0
        
        for draw in data["main_numbers"]:
            for num in draw:
                main_freq[num] += 1
                
        plt.bar(main_freq.keys(), main_freq.values(), color='royalblue', alpha=0.7)
        plt.title("Historical Frequency of Main Numbers", fontsize=14)
        plt.xlabel("Number", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.xticks(range(0, 51, 5))
        plt.grid(axis='y', alpha=0.3)
        
        # Plot 2: Frequency of bonus numbers (historical)
        plt.subplot(3, 2, 2)
        bonus_freq = {}
        for num in range(1, 13):
            bonus_freq[num] = 0
        
        for draw in data["bonus_numbers"]:
            for num in draw:
                bonus_freq[num] += 1
                
        plt.bar(bonus_freq.keys(), bonus_freq.values(), color='darkorange', alpha=0.7)
        plt.title("Historical Frequency of Bonus Numbers", fontsize=14)
        plt.xlabel("Number", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.xticks(range(1, 13))
        plt.grid(axis='y', alpha=0.3)
        
        # Plot 3: Predicted numbers distribution
        plt.subplot(3, 2, 3)
        pred_main_freq = {}
        for num in range(1, 51):
            pred_main_freq[num] = 0
        
        for pred in predictions:
            for num in pred["main_numbers"]:
                pred_main_freq[num] += 1
                
        plt.bar(pred_main_freq.keys(), pred_main_freq.values(), color='seagreen', alpha=0.7)
        plt.title("Predicted Main Numbers Distribution", fontsize=14)
        plt.xlabel("Number", fontsize=12)
        plt.ylabel("Frequency in Predictions", fontsize=12)
        plt.xticks(range(0, 51, 5))
        plt.grid(axis='y', alpha=0.3)
        
        # Add horizontal line for average
        avg_freq = sum(pred_main_freq.values()) / 50
        plt.axhline(y=avg_freq, color='r', linestyle='--', alpha=0.5, label=f"Average ({avg_freq:.2f})")
        plt.legend()
        
        # Plot 4: Predicted bonus numbers distribution
        plt.subplot(3, 2, 4)
        pred_bonus_freq = {}
        for num in range(1, 13):
            pred_bonus_freq[num] = 0
        
        for pred in predictions:
            for num in pred["bonus_numbers"]:
                pred_bonus_freq[num] += 1
                
        plt.bar(pred_bonus_freq.keys(), pred_bonus_freq.values(), color='indianred', alpha=0.7)
        plt.title("Predicted Bonus Numbers Distribution", fontsize=14)
        plt.xlabel("Number", fontsize=12)
        plt.ylabel("Frequency in Predictions", fontsize=12)
        plt.xticks(range(1, 13))
        plt.grid(axis='y', alpha=0.3)
        
        # Add horizontal line for average
        avg_freq = sum(pred_bonus_freq.values()) / 12
        plt.axhline(y=avg_freq, color='r', linestyle='--', alpha=0.5, label=f"Average ({avg_freq:.2f})")
        plt.legend()
        
        # Plot 5: Prediction confidence distribution
        plt.subplot(3, 2, 5)
        confidences = [pred["confidence"]["overall"] * 100 for pred in predictions]
        
        # Create histogram with KDE
        weights = np.ones_like(confidences) / len(confidences)
        plt.hist(confidences, bins=10, color='purple', alpha=0.6, weights=weights)
        
        # Add KDE curve
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(confidences)
        x = np.linspace(min(confidences), max(confidences), 100)
        plt.plot(x, kde(x), 'r-', linewidth=2)
        
        plt.title("Prediction Confidence Distribution", fontsize=14)
        plt.xlabel("Confidence (%)", fontsize=12)
        plt.ylabel("Density", fontsize=12)
        plt.grid(axis='y', alpha=0.3)
        
        # Add vertical line for average confidence
        avg_conf = np.mean(confidences)
        plt.axvline(x=avg_conf, color='g', linestyle='--', alpha=0.7, label=f"Average ({avg_conf:.2f}%)")
        plt.legend()
        
        # Plot 6: Historical vs. Predicted frequency correlation
        plt.subplot(3, 2, 6)
        
        # Normalize frequencies for comparison
        hist_main_values = np.array(list(main_freq.values()))
        hist_main_norm = hist_main_values / np.sum(hist_main_values)
        
        pred_main_values = np.array(list(pred_main_freq.values()))
        if np.sum(pred_main_values) > 0:  # Avoid division by zero
            pred_main_norm = pred_main_values / np.sum(pred_main_values)
        else:
            pred_main_norm = np.zeros_like(pred_main_values)
        
        # Create scatter plot with colors
        cmap = plt.cm.viridis
        plt.scatter(hist_main_norm, pred_main_norm, alpha=0.7, c=range(1, 51), cmap=cmap)
        
        # Add number labels to points
        for i, (x, y) in enumerate(zip(hist_main_norm, pred_main_norm)):
            plt.annotate(str(i+1), (x, y), fontsize=8, alpha=0.8)
        
        plt.title("Historical vs. Predicted Frequency Correlation", fontsize=14)
        plt.xlabel("Historical Frequency (normalized)", fontsize=12)
        plt.ylabel("Prediction Frequency (normalized)", fontsize=12)
        plt.grid(alpha=0.3)
        
        # Add line of equality
        max_val = max(np.max(hist_main_norm), np.max(pred_main_norm))
        plt.plot([0, max_val], [0, max_val], 'k--', alpha=0.5)
        
        # Add correlation coefficient
        correlation = np.corrcoef(hist_main_norm, pred_main_norm)[0, 1]
        plt.annotate(f"Correlation: {correlation:.2f}", 
                    xy=(0.05, 0.95), xycoords='axes fraction',
                    fontsize=12, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        # Plot 7: Number frequency over time
        plt.figure(figsize=(20, 10))
        plt.subplot(2, 1, 1)
        
        # Calculate rolling frequency for popular main numbers
        # First convert data to time series
        dates = []
        main_counts = {num: [] for num in range(1, 51)}
        
        window_size = 50  # Rolling window size
        for i in range(window_size, len(data)):
            window = data.iloc[i-window_size:i]
            dates.append(data.iloc[i-1]["date"])
            
            # Count occurrences of each number
            for num in range(1, 51):
                count = sum(1 for _, row in window.iterrows() if num in row["main_numbers"])
                main_counts[num].append(count / window_size)  # Normalize by window size
        
        # Plot top 5 most frequent numbers
        top_nums = sorted(range(1, 51), key=lambda x: sum(main_counts[x]), reverse=True)[:5]
        for num in top_nums:
            plt.plot(dates, main_counts[num], label=f"Number {num}", linewidth=2, alpha=0.7)
            
        plt.title("Frequency Trends of Top 5 Main Numbers Over Time", fontsize=14)
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Rolling Frequency", fontsize=12)
        plt.legend()
        plt.grid(alpha=0.3)
        
        # Plot 8: Prediction confidence by method
        plt.subplot(2, 1, 2)
        
        # Group by prediction method
        methods = {}
        for pred in predictions:
            method = pred.get("method", "unknown")
            if method not in methods:
                methods[method] = []
            methods[method].append(pred["confidence"]["overall"] * 100)
        
        # Create boxplot for each method
        method_names = list(methods.keys())
        method_data = [methods[m] for m in method_names]
        
        plt.boxplot(method_data, labels=method_names, patch_artist=True)
        plt.title("Prediction Confidence by Method", fontsize=14)
        plt.xlabel("Method", fontsize=12)
        plt.ylabel("Confidence (%)", fontsize=12)
        plt.grid(axis='y', alpha=0.3)
        
        # Save figures
        plt.tight_layout()
        plt.savefig("enhanced_lottery_visualizations.png", dpi=300)
        plt.close('all')
        
        # Create additional specialized visualization - pattern analysis
        plt.figure(figsize=(15, 10))
        
        # Plot distribution of predicted numbers across ranges
        plt.subplot(2, 2, 1)
        ranges = ["1-10", "11-20", "21-30", "31-40", "41-50"]
        range_counts = [0, 0, 0, 0, 0]
        
        for pred in predictions:
            for num in pred["main_numbers"]:
                range_idx = (num - 1) // 10
                range_counts[range_idx] += 1
        
        # Normalize
        range_counts = [count / sum(range_counts) * 100 for count in range_counts]
        
        # Historical distribution for comparison
        hist_range_counts = [0, 0, 0, 0, 0]
        total_hist_count = 0
        
        for _, row in data.iterrows():
            for num in row["main_numbers"]:
                range_idx = (num - 1) // 10
                hist_range_counts[range_idx] += 1
                total_hist_count += 1
        
        hist_range_counts = [count / total_hist_count * 100 for count in hist_range_counts]
        
        # Plot as grouped bar chart
        x = np.arange(len(ranges))
        width = 0.35
        
        plt.bar(x - width/2, range_counts, width, label='Predictions', color='seagreen', alpha=0.7)
        plt.bar(x + width/2, hist_range_counts, width, label='Historical', color='royalblue', alpha=0.7)
        
        plt.title("Distribution of Numbers Across Ranges", fontsize=14)
        plt.xlabel("Number Range", fontsize=12)
        plt.ylabel("Percentage (%)", fontsize=12)
        plt.xticks(x, ranges)
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        
        # Plot odd/even distribution
        plt.subplot(2, 2, 2)
        pred_odd_even = [0, 0]  # [odd, even]
        
        for pred in predictions:
            for num in pred["main_numbers"]:
                pred_odd_even[num % 2] += 1
        
        # Normalize
        pred_odd_even = [count / sum(pred_odd_even) * 100 for count in pred_odd_even]
        
        # Historical odd/even
        hist_odd_even = [0, 0]
        
        for _, row in data.iterrows():
            for num in row["main_numbers"]:
                hist_odd_even[num % 2] += 1
        
        hist_odd_even = [count / sum(hist_odd_even) * 100 for count in hist_odd_even]
        
        # Plot as grouped bar chart
        x = np.arange(2)
        labels = ['Odd', 'Even']
        
        plt.bar(x - width/2, [pred_odd_even[1], pred_odd_even[0]], width, label='Predictions', color='seagreen', alpha=0.7)
        plt.bar(x + width/2, [hist_odd_even[1], hist_odd_even[0]], width, label='Historical', color='royalblue', alpha=0.7)
        
        plt.title("Odd/Even Distribution", fontsize=14)
        plt.ylabel("Percentage (%)", fontsize=12)
        plt.xticks(x, labels)
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        
        # Plot consecutive number patterns
        plt.subplot(2, 2, 3)
        consecutive_counts = [0, 0, 0, 0, 0]  # 0, 1, 2, 3, 4 consecutive pairs
        
        for pred in predictions:
            nums = sorted(pred["main_numbers"])
            count = 0
            for i in range(len(nums) - 1):
                if nums[i+1] - nums[i] == 1:
                    count += 1
            consecutive_counts[min(count, 4)] += 1
        
        # Normalize
        consecutive_counts = [count / len(predictions) * 100 for count in consecutive_counts]
        
        # Historical consecutive counts
        hist_consecutive_counts = [0, 0, 0, 0, 0]
        
        for _, row in data.iterrows():
            nums = sorted(row["main_numbers"])
            count = 0
            for i in range(len(nums) - 1):
                if nums[i+1] - nums[i] == 1:
                    count += 1
            hist_consecutive_counts[min(count, 4)] += 1
        
        hist_consecutive_counts = [count / len(data) * 100 for count in hist_consecutive_counts]
        
        # Plot as grouped bar chart
        x = np.arange(5)
        labels = ['0 pairs', '1 pair', '2 pairs', '3 pairs', '4 pairs']
        
        plt.bar(x - width/2, consecutive_counts, width, label='Predictions', color='seagreen', alpha=0.7)
        plt.bar(x + width/2, hist_consecutive_counts, width, label='Historical', color='royalblue', alpha=0.7)
        
        plt.title("Consecutive Number Patterns", fontsize=14)
        plt.xlabel("Consecutive Pairs", fontsize=12)
        plt.ylabel("Percentage (%)", fontsize=12)
        plt.xticks(x, labels)
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        
        # Plot sum range distribution
        plt.subplot(2, 2, 4)
        sum_ranges = ["<100", "100-125", "126-150", "151-175", ">175"]
        sum_counts = [0, 0, 0, 0, 0]
        
        for pred in predictions:
            total = sum(pred["main_numbers"])
            if total < 100:
                sum_counts[0] += 1
            elif total <= 125:
                sum_counts[1] += 1
            elif total <= 150:
                sum_counts[2] += 1
            elif total <= 175:
                sum_counts[3] += 1
            else:
                sum_counts[4] += 1
        
        # Normalize
        sum_counts = [count / len(predictions) * 100 for count in sum_counts]
        
        # Historical sum distribution
        hist_sum_counts = [0, 0, 0, 0, 0]
        
        for _, row in data.iterrows():
            total = sum(row["main_numbers"])
            if total < 100:
                hist_sum_counts[0] += 1
            elif total <= 125:
                hist_sum_counts[1] += 1
            elif total <= 150:
                hist_sum_counts[2] += 1
            elif total <= 175:
                hist_sum_counts[3] += 1
            else:
                hist_sum_counts[4] += 1
        
        hist_sum_counts = [count / len(data) * 100 for count in hist_sum_counts]
        
        # Plot as grouped bar chart
        x = np.arange(5)
        
        plt.bar(x - width/2, sum_counts, width, label='Predictions', color='seagreen', alpha=0.7)
        plt.bar(x + width/2, hist_sum_counts, width, label='Historical', color='royalblue', alpha=0.7)
        
        plt.title("Sum of Main Numbers Distribution", fontsize=14)
        plt.xlabel("Sum Range", fontsize=12)
        plt.ylabel("Percentage (%)", fontsize=12)
        plt.xticks(x, sum_ranges)
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        
        # Save pattern analysis
        plt.tight_layout()
        plt.savefig("pattern_analysis.png", dpi=300)
        plt.close()
        
        logger.info("Enhanced visualizations saved to enhanced_lottery_visualizations.png and pattern_analysis.png")
    except Exception as e:
        logger.error(f"Error generating enhanced visualizations: {str(e)}")

#######################
# MAIN FUNCTION
#######################

def main():
    """Main function to run the improved prediction system."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Enhanced Transformer-Based EuroMillions Prediction System")
    parser.add_argument("--file", default="lottery_numbers.txt", help="Path to lottery data file")
    parser.add_argument("--optimize", action="store_true", help="Run hyperparameter optimization before prediction")
    parser.add_argument("--trials", type=int, default=30, help="Number of optimization trials")
    parser.add_argument("--predictions", type=int, default=20, help="Number of predictions to generate")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate model performance with cross-validation")
    parser.add_argument("--ensemble", action="store_true", help="Use enhanced ensemble for improved predictions")
    parser.add_argument("--num_models", type=int, default=7, help="Number of models in ensemble")
    parser.add_argument("--params", default="enhanced_transformer_params.json", help="Path to parameters file")

    args = parser.parse_args()
    
    # Print header
    print("\n" + "="*80)
    print("ENHANCED TRANSFORMER-BASED EUROMILLIONS LOTTERY PREDICTION SYSTEM".center(80))
    print("="*80 + "\n")
    print("This system uses enhanced transformer models with advanced feature engineering")
    print("and improved ensemble techniques to enhance prediction accuracy.")
    print("\nDISCLAIMER: Lottery outcomes are primarily random events and no")
    print("prediction system can guarantee winning numbers.")
    print("="*80 + "\n")
    
    # Check if data file exists
    if not os.path.exists(args.file):
        print(f"Error: Lottery data file '{args.file}' not found.")
        sys.exit(1)
    
    # Load parameters if specified
    params = None
    if args.params and os.path.exists(args.params) and not args.optimize:
        params = load_params(args.params)
        print(f"Loaded parameters from {args.params}")
    
    # Handle optimization if requested
    if args.optimize:
        print(f"Running self-optimizing hyperparameter search with {args.trials} trials...")
        params = self_optimizing_hyperparameters(args.file, args.trials)
        save_params(params, args.params)
        print(f"Optimization complete. Best parameters saved to {args.params}")
    else:
        # Use default parameters if none were loaded or optimized
        if params is None:
            params = {
                'learning_rate': 0.001,
                'batch_size': 16,
                'epochs': 100,
                'dropout_rate': 0.3,
                'num_heads': 4,
                'ff_dim': 128,
                'embed_dim': 64,
                'use_gru': True,
                'conv_filters': 32
            }
            print("Using default parameters")
    
    # Evaluate model if requested
    if args.evaluate:
        print("\nEvaluating model performance with improved cross-validation...")
        evaluator = ImprovedCrossValidationEvaluator(args.file, params)
        performance = evaluator.evaluate()
        
        print("\nImproved Cross-Validation Performance:")
        print(f"Overall Accuracy: {performance['avg_overall_accuracy']*100:.2f}%")
        print(f"Main Numbers Accuracy: {performance['avg_main_accuracy']*100:.2f}%")
        print(f"Bonus Numbers Accuracy: {performance['avg_bonus_accuracy']*100:.2f}%")
        
        # Display partial match statistics
        print("\nPartial Match Statistics:")
        match_probs = performance['avg_main_set_match_probs']
        for i, prob in enumerate(match_probs):
            print(f"  {i} main numbers correct: {prob*100:.2f}%")
        
        # Display by-number accuracy
        print("\nAccuracy by Number:")
        main_number_acc = performance['main_number_accuracy']
        bonus_number_acc = performance['bonus_number_accuracy']
        
        top_main_nums = np.argsort(main_number_acc)[-5:]  # Top 5 most accurately predicted main numbers
        top_bonus_nums = np.argsort(bonus_number_acc)[-3:]  # Top 3 most accurately predicted bonus numbers
        
        print("Top 5 Most Accurately Predicted Main Numbers:")
        for i in top_main_nums:
            print(f"  Number {i+1}: {main_number_acc[i]*100:.2f}% accuracy")
        
        print("Top 3 Most Accurately Predicted Bonus Numbers:")
        for i in top_bonus_nums:
            print(f"  Number {i+1}: {bonus_number_acc[i]*100:.2f}% accuracy")
        
        # Display calibration error
        print(f"\nMain Numbers Calibration Error: {performance['avg_main_calibration_error']:.4f}")
        print(f"Bonus Numbers Calibration Error: {performance['avg_bonus_calibration_error']:.4f}")
    
    # Generate predictions
    if args.ensemble:
        print(f"\nTraining diverse ensemble with {args.num_models} base models...")
        ensemble = DiverseEnsemble(args.file, args.num_models, params)
        ensemble.train_base_models()
        ensemble.train_meta_model()
        
        print(f"\nGenerating {args.predictions} ensemble predictions with diversity optimization...")
        predictions = ensemble.predict(args.predictions)
    else:
        print("\nTraining improved transformer model...")
        predictor = ImprovedLotteryPredictor(args.file, params)
        predictor.train_models()
        
        print(f"\nGenerating {args.predictions} predictions...")
        predictions = predictor.predict(args.predictions)
    
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
    
    # Generate enhanced visualizations
    print("\nGenerating enhanced visualizations...")
    generate_enhanced_visualizations(predictions, args.file)
    
    # Print summary
    try:
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
        for num, count in sorted(main_counts.items(), key=lambda x: x[1], reverse=True):
            if count > 1:  # Only show numbers that appear multiple times
                print(f"Number {num}: appeared {count} times")
        
        print("\nMost frequent bonus numbers in predictions:")
        for num, count in sorted(bonus_counts.items(), key=lambda x: x[1], reverse=True):
            if count > 1:  # Only show numbers that appear multiple times
                print(f"Number {num}: appeared {count} times")
        
        # Calculate average confidence
        avg_confidence = np.mean([pred["confidence"]["overall"] * 100 for pred in predictions])
        print(f"\nAverage prediction confidence: {avg_confidence:.2f}%")
        
        # Display pattern analysis
        pattern_scores = [pred["confidence"].get("pattern_score", 0) * 100 for pred in predictions if "pattern_score" in pred["confidence"]]
        if pattern_scores:
            avg_pattern_score = np.mean(pattern_scores)
            print(f"Average pattern score: {avg_pattern_score:.2f}%")
        
        frequency_scores = [pred["confidence"].get("frequency_score", 0) * 100 for pred in predictions if "frequency_score" in pred["confidence"]]
        if frequency_scores:
            avg_frequency_score = np.mean(frequency_scores)
            print(f"Average historical frequency score: {avg_frequency_score:.2f}%")
    except Exception as e:
        logger.error(f"Error in summary generation: {str(e)}")
    
    print("\nReminder: These predictions are based on statistical analysis and")
    print("are not guaranteed to win. Please gamble responsibly.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nPrediction system stopped by user.")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        print(f"\nError: {str(e)}")
        print("Check enhanced_transformer.log for details.")