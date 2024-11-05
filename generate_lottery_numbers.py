from collections import OrderedDict, Counter
from scipy.stats import dirichlet, entropy, norm
import numpy as np
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.model_selection import KFold
import warnings
from datetime import datetime
import joblib
import logging
from typing import List, Optional, Dict, Union, Callable, Any
from dataclasses import dataclass

# Suppress warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom exceptions
class LotteryError(Exception): pass
class LotteryDataError(LotteryError): pass
class LotteryConfigError(LotteryError): pass

@dataclass
class LotteryConfig:
    """Enhanced configuration using dataclass for better structure"""
    n_main: int = 5
    n_bonus: int = 2
    main_number_range: int = 50
    bonus_number_range: int = 12
    base_alpha: float = 1.0
    time_decay_factor: float = 0.95
    cache_size: int = 1000
    default_window_size: int = 10
    cv_splits: int = 5
    random_state: int = 42
    
    def validate(self) -> bool:
        """Validate configuration parameters"""
        try:
            assert self.n_main > 0
            assert self.n_bonus > 0
            assert self.main_number_range > self.n_main
            assert self.bonus_number_range > self.n_bonus
            assert 0 < self.time_decay_factor < 1
            assert self.cache_size > 0
            assert self.default_window_size > 0
            assert self.cv_splits > 1
            return True
        except AssertionError:
            raise LotteryConfigError("Invalid configuration parameters")

class ValidationMixin:
    """Centralized validation functionality"""
    
    @staticmethod
    def validate_number_range(numbers: Union[List[int], np.ndarray], min_val: int, max_val: int) -> bool:
        """Validate numbers are within specified range"""
        try:
            # Convert list to numpy array if necessary
            num_array = np.array(numbers)
            return np.all((num_array >= min_val) & (num_array <= max_val))
        except Exception:
            return False
    
    @staticmethod
    def validate_uniqueness(numbers: Union[List[int], np.ndarray]) -> bool:
        """Validate numbers are unique"""
        try:
            # Convert list to numpy array if necessary
            num_array = np.array(numbers)
            return len(np.unique(num_array)) == len(num_array)
        except Exception:
            return False
    
    @staticmethod
    def validate_number_count(numbers: Union[List[int], np.ndarray], expected_count: int) -> bool:
        """Validate correct number of numbers"""
        try:
            return len(numbers) == expected_count
        except Exception:
            return False
    
    @staticmethod
    def validate_distribution(numbers: Union[List[int], np.ndarray], number_range: int) -> bool:
        """Validate number distribution"""
        try:
            # Convert list to numpy array if necessary
            num_array = np.array(numbers)
            even_count = np.sum(num_array % 2 == 0)
            even_ratio = even_count / len(num_array)
            if not (0.3 <= even_ratio <= 0.7):
                return False
                
            low_count = np.sum(num_array <= number_range / 2)
            low_ratio = low_count / len(num_array)
            if not (0.3 <= low_ratio <= 0.7):
                return False
                
            return True
        except Exception:
            return False


class SharedAnalytics:
    """Shared analytical functionality"""
    
    @staticmethod
    def calculate_frequency(numbers: np.ndarray, number_range: int) -> np.ndarray:
        """Calculate frequency distribution"""
        freq = np.zeros(number_range)
        unique, counts = np.unique(numbers, return_counts=True)
        freq[unique - 1] = counts / len(numbers)
        return freq
    
    @staticmethod
    def calculate_spacing_metrics(numbers: np.ndarray) -> Dict[str, float]:
        """Calculate spacing-related metrics"""
        sorted_nums = np.sort(numbers)
        diffs = np.diff(sorted_nums)
        
        return {
            'mean_spacing': float(np.mean(diffs)) if len(diffs) > 0 else 0.0,
            'spacing_consistency': float(1 - (np.std(diffs) / (np.mean(diffs) + 1e-10))) if len(diffs) > 0 else 0.0
        }
    
    @staticmethod
    def calculate_balance_metrics(numbers: np.ndarray, number_range: int) -> Dict[str, float]:
        """Calculate balance-related metrics"""
        return {
            'low_high_ratio': float(np.sum(numbers <= number_range/2) / len(numbers)),
            'even_odd_ratio': float(np.sum(numbers % 2 == 0) / len(numbers))
        }
    
    @staticmethod
    def calculate_pattern_metrics(numbers: np.ndarray, 
                                historical_data: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate pattern-related metrics"""
        metrics = {
            'consecutive_count': 0,
            'pattern_similarity': 0.0,
            'distribution_score': 0.0
        }
        
        try:
            # Calculate consecutive numbers
            sorted_nums = np.sort(numbers)
            metrics['consecutive_count'] = np.sum(np.diff(sorted_nums) == 1)
            
            # Calculate distribution score
            unique_count = len(np.unique(numbers))
            metrics['distribution_score'] = unique_count / len(numbers)
            
            # Calculate pattern similarity if historical data is available
            if historical_data is not None:
                historical_freq = np.zeros(np.max(historical_data) + 1)
                number_freq = np.zeros_like(historical_freq)
                
                unique, counts = np.unique(historical_data, return_counts=True)
                historical_freq[unique] = counts / len(historical_data)
                
                unique, counts = np.unique(numbers, return_counts=True)
                number_freq[unique] = counts / len(numbers)
                
                # Calculate cosine similarity
                norm_product = np.linalg.norm(historical_freq) * np.linalg.norm(number_freq)
                if norm_product > 0:
                    metrics['pattern_similarity'] = np.dot(historical_freq, number_freq) / norm_product
            
        except Exception as e:
            logger.warning(f"Error calculating pattern metrics: {str(e)}")
        
        return metrics

class AnalysisCache:
    """Enhanced caching system with computation management"""
    def __init__(self, max_size: int = 1000):
        # Define the caches using consistent naming
        self.caches = {
            'patterns': OrderedDict(),
            'probabilities': OrderedDict(),
            'streaks': OrderedDict()
        }
        self.max_size = max_size
        self.last_update = None
    
    def _cleanup_cache(self, cache: OrderedDict) -> None:
        while len(cache) > self.max_size:
            cache.popitem(first=True)
    
    def get_or_compute(self, cache_type: str, key: str, compute_fn: Callable) -> Any:
        """Get from cache or compute value"""
        if cache_type not in self.caches:
            valid_types = ", ".join(self.caches.keys())
            raise ValueError(f"Invalid cache type: {cache_type}. Valid types are: {valid_types}")
            
        cache = self.caches[cache_type]
        if key in cache:
            cache.move_to_end(key)
            return cache[key]
        
        result = compute_fn()
        cache[key] = result
        self._cleanup_cache(cache)
        self.last_update = datetime.now()
        return result
    
    def save_state(self, filename: str) -> bool:
        """Save cache state"""
        try:
            state = {
                'caches': {name: dict(cache) for name, cache in self.caches.items()},
                'last_update': self.last_update
            }
            joblib.dump(state, filename)
            return True
        except Exception as e:
            logging.warning(f"Failed to save cache state: {str(e)}")
            return False
    
    def load_state(self, filename: str) -> bool:
        """Load cache state"""
        try:
            state = joblib.load(filename)
            self.caches = {
                name: OrderedDict(cache) for name, cache in state['caches'].items()
            }
            self.last_update = state['last_update']
            return True
        except Exception as e:
            logging.warning(f"Failed to load cache state: {str(e)}")
            return False

class DataManager(ValidationMixin):
    """Enhanced data management with shared validation"""
    def __init__(self, config: LotteryConfig):
        self.config = config
        self.data = None
        self.years = None
        self.days = None
        self.analytics = SharedAnalytics()
    
    def load_data(self, file_path: str) -> np.ndarray:
        """Load and validate lottery data"""
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
                structured_data = []
                current_numbers = []
                expected_numbers = self.config.n_main + self.config.n_bonus
                
                for line in lines:
                    try:
                        num = int(line.strip())
                        current_numbers.append(num)
                        
                        if len(current_numbers) == expected_numbers:
                            # Convert to numpy array for validation
                            numbers_array = np.array(current_numbers)
                            if self._validate_numbers(numbers_array):
                                structured_data.append(current_numbers)
                            current_numbers = []
                    except ValueError:
                        continue
                
                if not structured_data:
                    raise LotteryDataError("No valid lottery data found")
                
                self.data = np.array(structured_data)
                self._calculate_time_periods()
                return self.data
                
        except FileNotFoundError:
            raise LotteryDataError(f"Data file not found: {file_path}")
        except Exception as e:
            raise LotteryDataError(f"Error reading data: {str(e)}")
    
    def _validate_numbers(self, numbers: np.ndarray) -> bool:
        """Validate lottery numbers using shared validation"""
        try:
            main = numbers[:self.config.n_main]
            bonus = numbers[self.config.n_main:]
            
            return (
                self.validate_number_count(main, self.config.n_main) and
                self.validate_number_count(bonus, self.config.n_bonus) and
                self.validate_uniqueness(main) and
                self.validate_uniqueness(bonus) and
                self.validate_number_range(main, 1, self.config.main_number_range) and
                self.validate_number_range(bonus, 1, self.config.bonus_number_range)
            )
        except Exception:
            return False
    
    def _calculate_time_periods(self) -> None:
        """Calculate time periods for temporal analysis"""
        try:
            num_entries = len(self.data)
            num_years = num_entries // (2 * 52) + 1
            
            self.years = np.array([num_years - 1 - i // (2 * 52) for i in range(num_entries)])
            self.days = np.array([1 if i % 7 >= 5 else 0 for i in range(num_entries)])
            
            assert len(self.years) == num_entries
            assert len(self.days) == num_entries
            
        except Exception as e:
            logger.error(f"Error calculating time periods: {str(e)}")
            self.years = np.zeros(len(self.data), dtype=int)
            self.days = np.zeros(len(self.data), dtype=int)
    
    def get_main_numbers(self) -> np.ndarray:
        """Get main numbers array"""
        if self.data is None:
            raise LotteryDataError("No data loaded")
        return self.data[:, :self.config.n_main].flatten()
    
    def get_bonus_numbers(self) -> np.ndarray:
        """Get bonus numbers array"""
        if self.data is None:
            raise LotteryDataError("No data loaded")
        return self.data[:, self.config.n_main:].flatten()

class PatternAnalyzer:
    """Enhanced pattern analysis with shared analytics"""
    def __init__(self, config: LotteryConfig):
        self.config = config
        self.cache = AnalysisCache(config.cache_size)  # Using AnalysisCache instead of CacheManager
        self.analytics = SharedAnalytics()
        self.logger = logging.getLogger(__name__)
    
    def analyze_patterns(self, numbers: np.ndarray, is_bonus: bool = False) -> Dict:
        """Comprehensive pattern analysis"""
        cache_key = f"pattern_{'bonus' if is_bonus else 'main'}_{len(numbers)}"
        
        def compute_patterns():
            number_range = self.config.bonus_number_range if is_bonus else self.config.main_number_range
            n_numbers = self.config.n_bonus if is_bonus else self.config.n_main
            
            pattern_model = self._detect_patterns(numbers, number_range, n_numbers)
            pattern_metrics = self.analytics.calculate_pattern_metrics(numbers)
            spacing_metrics = self.analytics.calculate_spacing_metrics(numbers)
            balance_metrics = self.analytics.calculate_balance_metrics(numbers, number_range)
            
            return {
                'pattern_model': pattern_model,
                'metrics': {
                    **pattern_metrics,
                    **spacing_metrics,
                    **balance_metrics
                }
            }
        
        return self.cache.get_or_compute('patterns', cache_key, compute_patterns)
    
    def _detect_patterns(self, numbers: np.ndarray, number_range: int, n_numbers: int) -> Dict:
        """Enhanced pattern detection with multiple models"""
        try:
            shaped_numbers = numbers.reshape(-1, n_numbers)
            
            # Convert to enhanced feature space
            features = np.zeros((len(shaped_numbers), n_numbers * 2))
            angles = (shaped_numbers * 2 * np.pi) / number_range
            features[:, :n_numbers] = np.cos(angles)
            features[:, n_numbers:] = np.sin(angles)
            
            # Add additional features
            spacing_features = np.diff(np.sort(shaped_numbers, axis=1))
            range_features = np.max(shaped_numbers, axis=1) - np.min(shaped_numbers, axis=1)
            even_odd_features = np.sum(shaped_numbers % 2 == 0, axis=1)
            
            # Combine all features
            combined_features = np.column_stack([
                features,
                spacing_features,
                range_features.reshape(-1, 1),
                even_odd_features.reshape(-1, 1)
            ])
            
            # Fit models with correct feature dimensions
            gmm_regular = self._fit_gmm(combined_features, is_variational=False)
            gmm_variational = self._fit_gmm(combined_features, is_variational=True)
            
            return {
                'regular_gmm': gmm_regular,
                'variational_gmm': gmm_variational,
                'features': combined_features
            }
            
        except Exception as e:
            self.logger.warning(f"Pattern detection failed: {str(e)}")
            return {}

    def _fit_gmm(self, features: np.ndarray, is_variational: bool = False) -> Union[GaussianMixture, BayesianGaussianMixture]:
        """Fit GMM model with cross-validation"""
        models = []
        scores = []
        
        min_components = 2
        max_components = min(8, features.shape[0] // 5)
        
        for n_components in range(min_components, max_components):
            try:
                model_score = 0
                kf = KFold(n_splits=self.config.cv_splits, 
                        shuffle=True,
                        random_state=self.config.random_state)
                
                for train_idx, val_idx in kf.split(features):
                    if is_variational:
                        model = BayesianGaussianMixture(
                            n_components=n_components,
                            random_state=self.config.random_state,
                            weight_concentration_prior=1.0/n_components,
                            covariance_type='full',
                            max_iter=300,
                            n_init=3,
                            reg_covar=1e-6
                        )
                    else:
                        model = GaussianMixture(
                            n_components=n_components,
                            random_state=self.config.random_state,
                            covariance_type='full',
                            max_iter=300,
                            n_init=3,
                            reg_covar=1e-6
                        )
                    
                    model.fit(features[train_idx])
                    score = model.score(features[val_idx])
                    model_score += score
                
                avg_score = model_score / self.config.cv_splits
                models.append((n_components, model_score, model))
                scores.append(avg_score)
                
            except Exception as e:
                self.logger.warning(f"Error fitting model with {n_components} components: {str(e)}")
                continue
        
        if not models:
            if is_variational:
                default_model = BayesianGaussianMixture(
                    n_components=2,
                    random_state=self.config.random_state,
                    covariance_type='full',
                    reg_covar=1e-6
                )
            else:
                default_model = GaussianMixture(
                    n_components=2,
                    random_state=self.config.random_state,
                    covariance_type='full',
                    reg_covar=1e-6
                )
            default_model.fit(features)
            return default_model
        
        best_idx = np.argmax(scores)
        return models[best_idx][2]

class NumberGenerator(ValidationMixin):
    """Enhanced number generation with shared validation"""
    def __init__(self, config: LotteryConfig, pattern_analyzer: PatternAnalyzer):
        self.config = config
        self.pattern_analyzer = pattern_analyzer
        self.analytics = SharedAnalytics()
        self.min_attempts = 100
        self.max_attempts = 500
        self.target_spacing = 8.5
        self.spacing_tolerance = 2.0
        self.logger = logging.getLogger(__name__)

    def _validate_draw(self, draw: np.ndarray, historical_data: Optional[np.ndarray] = None) -> bool:
        """Final refinement of draw validation with stricter uniqueness constraints"""
        try:
            main_numbers = draw[:self.config.n_main]
            bonus_numbers = draw[self.config.n_main:]
            
            # Basic validation
            if not self._validate_main_numbers(main_numbers):
                return False
            
            if not self._validate_bonus_numbers(bonus_numbers):
                return False
            
            # Strict batch uniqueness check
            if hasattr(self, '_current_batch_numbers') and self._current_batch_numbers:
                # Track all used numbers in current batch
                batch_main_numbers = []
                batch_bonus_numbers = []
                
                for prev_draw in self._current_batch_numbers:
                    batch_main_numbers.extend(prev_draw[:self.config.n_main])
                    batch_bonus_numbers.extend(prev_draw[self.config.n_main:])
                
                # Check main number repetition
                main_number_counts = Counter(batch_main_numbers)
                for num in main_numbers:
                    if main_number_counts[num] >= 1:
                        return False
                
                # Check bonus number repetition
                bonus_number_counts = Counter(batch_bonus_numbers)
                for num in bonus_numbers:
                    if bonus_number_counts[num] >= 1:
                        return False
            
            # Modified spacing validation
            spacing_metrics = self.analytics.calculate_spacing_metrics(main_numbers)
            ideal_spacing = self.config.main_number_range / (self.config.n_main + 1)
            if abs(spacing_metrics['mean_spacing'] - ideal_spacing) > ideal_spacing * 0.4:
                return False
            
            # Enhanced distribution check
            segments = np.array_split(range(1, self.config.main_number_range + 1), 4)  # Changed to 4 segments
            numbers_in_segments = [
                sum(1 for num in main_numbers if num in segment)
                for segment in segments
            ]
            min_numbers_per_segment = 1
            max_numbers_per_segment = 2
            if not (min_numbers_per_segment <= max(numbers_in_segments) <= max_numbers_per_segment):
                return False
            
            # Historical pattern validation
            if historical_data is not None:
                recent_draws = historical_data[-30:] if len(historical_data) > 30 else historical_data
                for hist_draw in recent_draws:
                    hist_main = hist_draw[:self.config.n_main]
                    common_numbers = np.intersect1d(main_numbers, hist_main)
                    if len(common_numbers) > 3:
                        return False
            
            # Enhanced balance validation
            balance_metrics = self.analytics.calculate_balance_metrics(
                main_numbers,
                self.config.main_number_range
            )
            
            # Stricter low/high ratio
            if not (0.35 <= balance_metrics['low_high_ratio'] <= 0.65):
                return False
            
            # Even/Odd distribution
            even_count = sum(1 for x in main_numbers if x % 2 == 0)
            if not (2 <= even_count <= 3):
                return False
            
            return True
            
        except Exception as e:
            self.logger.warning(f"Draw validation failed: {str(e)}")
            return False
    
    def _adjust_batch_probabilities(self, probabilities: np.ndarray,
                                  current_batch: List[np.ndarray],
                                  is_bonus: bool = False) -> np.ndarray:
        """Enhanced probability adjustment with strict uniqueness control"""
        try:
            if not current_batch:
                return probabilities
            
            adjusted_probs = probabilities.copy()
            
            # Get index ranges based on number type
            start_idx = self.config.n_main if is_bonus else 0
            end_idx = None if is_bonus else self.config.n_main
            
            # Collect all used numbers in current batch
            used_numbers = set()
            for draw in current_batch:
                used_numbers.update(draw[start_idx:end_idx])
            
            # Set probability to near-zero for used numbers
            for num in used_numbers:
                if 0 <= num-1 < len(adjusted_probs):
                    adjusted_probs[num-1] = 1e-10
            
            # Boost probabilities for numbers in underutilized ranges
            number_range = len(adjusted_probs)
            segments = np.array_split(range(number_range), 4)
            
            for segment in segments:
                segment_numbers = set(segment)
                used_in_segment = segment_numbers.intersection(used_numbers)
                if len(used_in_segment) == 0:
                    # Boost unused segments
                    adjusted_probs[segment] *= 1.5
            
            # Ensure minimum probability and normalize
            min_prob = 1e-10
            adjusted_probs = np.maximum(adjusted_probs, min_prob)
            return adjusted_probs / np.sum(adjusted_probs)
            
        except Exception as e:
            self.logger.warning(f"Batch probability adjustment failed: {str(e)}")
            return probabilities
    
    def generate_numbers(self, n: int, main_probs: np.ndarray, bonus_probs: np.ndarray,
                        historical_data: Optional[np.ndarray] = None) -> np.ndarray:
        """Generate diverse sets of lottery numbers with improved variety control"""
        generated_lists = []
        previous_draws = set()
        self._current_batch_numbers = []  # Track numbers in current batch
        attempts = 0
        max_retries = self.max_attempts * n
        
        self.logger.info(f"Attempting to generate {n} number combinations")
        
        while len(generated_lists) < n and attempts < max_retries:
            try:
                # Adjust probabilities based on current batch
                adjusted_main_probs = self._adjust_batch_probabilities(
                    main_probs,
                    self._current_batch_numbers,
                    is_bonus=False
                )
                
                adjusted_bonus_probs = self._adjust_batch_probabilities(
                    bonus_probs,
                    self._current_batch_numbers,
                    is_bonus=True
                )
                
                # Generate main numbers
                main_numbers = self._generate_main_numbers(
                    adjusted_main_probs,
                    historical_data,
                    previous_draws
                )
                
                if main_numbers is None or not self._validate_main_numbers(main_numbers):
                    attempts += 1
                    continue
                
                # Generate bonus numbers
                bonus_numbers = self._generate_bonus_numbers(
                    adjusted_bonus_probs,
                    main_numbers,
                    historical_data,
                    previous_draws
                )
                
                if bonus_numbers is None or not self._validate_bonus_numbers(bonus_numbers):
                    attempts += 1
                    continue
                
                # Combine numbers
                complete_draw = np.concatenate((main_numbers, bonus_numbers))
                draw_tuple = tuple(complete_draw)
                
                if (draw_tuple not in previous_draws and 
                    self._validate_draw(complete_draw, historical_data)):
                    generated_lists.append(complete_draw)
                    self._current_batch_numbers.append(complete_draw)
                    previous_draws.add(draw_tuple)
                    self.logger.info(f"Generated valid combination: {complete_draw}")
                
            except Exception as e:
                self.logger.warning(f"Number generation attempt failed: {str(e)}")
            
            attempts += 1
        
        # Clean up
        if hasattr(self, '_current_batch_numbers'):
            delattr(self, '_current_batch_numbers')
        
        if len(generated_lists) == 0:
            self.logger.error("Failed to generate any valid number combinations")
            fallback = self._generate_fallback_numbers()
            if fallback is not None:
                generated_lists.append(fallback)
        
        if len(generated_lists) < n:
            self.logger.warning(f"Only generated {len(generated_lists)} of {n} requested combinations")
        
        return np.array(generated_lists)
    
    def _validate_main_numbers(self, numbers: np.ndarray) -> bool:
        """Validate main numbers using shared validation"""
        return (
            self.validate_number_count(numbers, self.config.n_main) and
            self.validate_uniqueness(numbers) and
            self.validate_number_range(numbers, 1, self.config.main_number_range) and
            self.validate_distribution(numbers, self.config.main_number_range)
        )
    
    def _validate_bonus_numbers(self, numbers: np.ndarray) -> bool:
        """Validate bonus numbers using shared validation"""
        return (
            self.validate_number_count(numbers, self.config.n_bonus) and
            self.validate_uniqueness(numbers) and
            self.validate_number_range(numbers, 1, self.config.bonus_number_range)
        )
    
    def _generate_main_numbers(self, probabilities: np.ndarray,
                             historical_data: Optional[np.ndarray],
                             previous_draws: set) -> Optional[np.ndarray]:
        """Generate main numbers with improved sampling"""
        best_numbers = None
        best_score = float('-inf')
        local_attempts = 0
        
        try:
            base_probs = self._add_entropy_to_probabilities(probabilities)
            
            while local_attempts < self.min_attempts:
                temperature = 1.0 + (local_attempts / self.min_attempts)
                adjusted_probs = np.power(base_probs, 1/temperature)
                adjusted_probs /= adjusted_probs.sum()
                
                candidate_numbers = self._sample_unique_numbers(
                    adjusted_probs,
                    self.config.n_main,
                    self.config.main_number_range
                )
                
                if candidate_numbers is None:
                    local_attempts += 1
                    continue
                
                score = self._evaluate_main_numbers(
                    candidate_numbers,
                    historical_data,
                    previous_draws
                )
                
                if score > best_score:
                    best_score = score
                    best_numbers = candidate_numbers
                
                local_attempts += 1
            
            return best_numbers
            
        except Exception as e:
            self.logger.warning(f"Main number generation failed: {str(e)}")
            return None
    
    def _generate_bonus_numbers(self, probabilities: np.ndarray,
                              main_numbers: np.ndarray,
                              historical_data: Optional[np.ndarray],
                              previous_draws: set) -> Optional[np.ndarray]:
        """Enhanced bonus number generation with improved distribution"""
        best_numbers = None
        best_score = float('-inf')
        local_attempts = 0
        
        try:
            # Track used bonus numbers in current batch
            used_bonus = set()
            if hasattr(self, '_current_batch_numbers'):
                for draw in self._current_batch_numbers:
                    used_bonus.update(draw[self.config.n_main:])
            
            adjusted_probs = self._adjust_bonus_probabilities(
                probabilities,
                main_numbers,
                previous_draws
            )
            
            # Further reduce probabilities for used bonus numbers
            for num in used_bonus:
                if 0 <= num-1 < len(adjusted_probs):
                    adjusted_probs[num-1] = 1e-10
            
            adjusted_probs = adjusted_probs / np.sum(adjusted_probs)
            
            while local_attempts < self.min_attempts:
                candidate_numbers = self._sample_unique_numbers(
                    adjusted_probs,
                    self.config.n_bonus,
                    self.config.bonus_number_range
                )
                
                if candidate_numbers is None:
                    local_attempts += 1
                    continue
                
                # Ensure no overlap with used bonus numbers
                if any(num in used_bonus for num in candidate_numbers):
                    local_attempts += 1
                    continue
                
                score = self._evaluate_bonus_numbers(
                    candidate_numbers,
                    main_numbers,
                    historical_data,
                    previous_draws
                )
                
                if score > best_score:
                    best_score = score
                    best_numbers = candidate_numbers
                
                local_attempts += 1
            
            return best_numbers
            
        except Exception as e:
            self.logger.warning(f"Bonus number generation failed: {str(e)}")
            return None
    
    def _add_entropy_to_probabilities(self, probabilities: np.ndarray,
                                    historical_data: Optional[np.ndarray] = None) -> np.ndarray:
        """Enhanced entropy addition with historical consideration"""
        try:
            min_prob = 0.2 / len(probabilities)
            smoothed_probs = probabilities + min_prob
            
            # Add controlled randomness
            noise = np.random.uniform(0, 0.1, size=len(probabilities))
            smoothed_probs += noise
            
            # If historical data is available, boost underused numbers
            if historical_data is not None:
                hist_freq = np.bincount(historical_data.flatten(), 
                                      minlength=len(probabilities)+1)[1:]
                hist_freq = hist_freq / np.sum(hist_freq)
                
                # Boost numbers that are historically underrepresented
                boost = 1 - (hist_freq / np.max(hist_freq))
                smoothed_probs *= (1 + 0.2 * boost)
            
            # Ensure non-negative probabilities
            smoothed_probs = np.maximum(smoothed_probs, 0)
            
            return smoothed_probs / np.sum(smoothed_probs)
            
        except Exception as e:
            self.logger.warning(f"Entropy addition failed: {str(e)}")
            return probabilities / np.sum(probabilities)
    
    def _adjust_bonus_probabilities(self, probabilities: np.ndarray,
                                  main_numbers: np.ndarray,
                                  previous_draws: set = None) -> np.ndarray:
        """More lenient bonus probability adjustment"""
        try:
            adjusted_probs = probabilities.copy()
            number_range = len(probabilities)
            
            # Add minimum probability
            min_prob = 0.05 / number_range
            adjusted_probs += min_prob
            
            # Reduce probabilities near main numbers - more gradual reduction
            for main_num in main_numbers:
                for i in range(len(adjusted_probs)):
                    num = i + 1
                    distance = abs(num - main_num)
                    if distance < 3:
                        adjusted_probs[i] *= 0.7
            
            # Reduce probabilities of recent bonus numbers - more gradual
            if previous_draws:
                recent_bonus = set()
                for draw in list(previous_draws)[-3:]:
                    recent_bonus.update(draw[self.config.n_main:])
                for num in recent_bonus:
                    if 0 <= num-1 < len(adjusted_probs):
                        adjusted_probs[num-1] *= 0.5
            
            # Boost underutilized ranges - more moderate boost
            quartiles = np.array_split(np.arange(number_range), 4)
            for quartile in quartiles:
                quartile_mean = np.mean(adjusted_probs[quartile])
                if quartile_mean < np.mean(adjusted_probs) * 0.8:
                    adjusted_probs[quartile] *= 1.2
            
            return adjusted_probs / np.sum(adjusted_probs)
            
        except Exception as e:
            self.logger.warning(f"Probability adjustment failed: {str(e)}")
            return probabilities / np.sum(probabilities)
    
    def _sample_unique_numbers(self, probabilities: np.ndarray,
                             n_numbers: int, number_range: int) -> Optional[np.ndarray]:
        """Sample unique numbers using rejection sampling"""
        max_retries = 50
        retries = 0
        
        while retries < max_retries:
            sample_size = min(n_numbers * 2, number_range)
            numbers = np.random.choice(
                np.arange(1, number_range + 1),
                size=sample_size,
                p=probabilities,
                replace=False
            )
            
            unique_numbers = np.unique(numbers)
            if len(unique_numbers) >= n_numbers:
                selected_indices = np.random.choice(
                    len(unique_numbers),
                    size=n_numbers,
                    replace=False
                )
                return np.sort(unique_numbers[selected_indices])
            
            retries += 1
        
        return None
    
    def _evaluate_main_numbers(self, numbers: np.ndarray,
                             historical_data: Optional[np.ndarray],
                             previous_draws: set) -> float:
        """Evaluate main numbers using shared analytics"""
        try:
            metrics = self.analytics.calculate_pattern_metrics(numbers, historical_data)
            spacing_metrics = self.analytics.calculate_spacing_metrics(numbers)
            balance_metrics = self.analytics.calculate_balance_metrics(
                numbers, self.config.main_number_range
            )
            
            # Combine scores with weights
            score = (
                0.3 * (1 - metrics['pattern_similarity']) +  # Prefer some novelty
                0.3 * spacing_metrics['spacing_consistency'] +
                0.2 * (1 - abs(balance_metrics['even_odd_ratio'] - 0.5)) +
                0.2 * (1 - abs(balance_metrics['low_high_ratio'] - 0.5))
            )
            
            return score
        except Exception as e:
            self.logger.warning(f"Main number evaluation failed: {str(e)}")
            return 0.0
    
    def _evaluate_bonus_numbers(self, numbers: np.ndarray,
                              main_numbers: np.ndarray,
                              historical_data: Optional[np.ndarray],
                              previous_draws: set) -> float:
        """Evaluate bonus numbers using shared analytics"""
        try:
            metrics = self.analytics.calculate_pattern_metrics(numbers, historical_data)
            spacing_metrics = self.analytics.calculate_spacing_metrics(numbers)
            balance_metrics = self.analytics.calculate_balance_metrics(
                numbers, self.config.bonus_number_range
            )
            
            # Calculate interaction with main numbers
            min_distance = float('inf')
            for bonus_num in numbers:
                distances = np.abs(bonus_num - main_numbers)
                min_dist = np.min(distances)
                min_distance = min(min_distance, min_dist)
            
            interaction_score = min_distance / self.config.bonus_number_range
            
            # Combine scores with weights
            score = (
                0.3 * interaction_score +
                0.3 * spacing_metrics['spacing_consistency'] +
                0.2 * (1 - metrics['pattern_similarity']) +
                0.2 * balance_metrics['low_high_ratio']
            )
            
            return score
        except Exception as e:
            self.logger.warning(f"Bonus number evaluation failed: {str(e)}")
            return 0.0
    
    def _generate_fallback_numbers(self) -> Optional[np.ndarray]:
        """Generate a simple valid combination as fallback"""
        try:
            # Generate main numbers
            main_numbers = []
            available_main = list(range(1, self.config.main_number_range + 1))
            
            while len(main_numbers) < self.config.n_main and available_main:
                num = np.random.choice(available_main)
                available_main.remove(num)
                if num-1 in available_main:
                    available_main.remove(num-1)
                if num+1 in available_main:
                    available_main.remove(num+1)
                main_numbers.append(num)
            
            if len(main_numbers) < self.config.n_main:
                return None
            
            # Generate bonus numbers
            bonus_numbers = []
            available_bonus = list(range(1, self.config.bonus_number_range + 1))
            
            while len(bonus_numbers) < self.config.n_bonus and available_bonus:
                num = np.random.choice(available_bonus)
                available_bonus.remove(num)
                bonus_numbers.append(num)
            
            if len(bonus_numbers) < self.config.n_bonus:
                return None
            
            return np.concatenate((np.array(sorted(main_numbers)), 
                                np.array(sorted(bonus_numbers))))
        except Exception as e:   
            self.logger.warning(f"Fallback generation failed: {str(e)}")
            return None

class ProbabilityEstimator:
    """Enhanced probability estimation with shared analytics"""
    def __init__(self, config: LotteryConfig, pattern_analyzer: PatternAnalyzer):
        self.config = config
        self.pattern_analyzer = pattern_analyzer
        self.analytics = SharedAnalytics()
        self.cache = AnalysisCache(config.cache_size)
        self.logger = logging.getLogger(__name__)

    def _combine_probabilities_dirichlet(self, probability_lists: List[np.ndarray]) -> np.ndarray:
        """Combine multiple probability distributions using Dirichlet distribution"""
        try:
            if not probability_lists:
                return None
                
            # Stack probabilities and calculate concentration parameters
            stacked_probs = np.stack(probability_lists)
            alpha = np.mean(stacked_probs, axis=0) * len(probability_lists) + self.config.base_alpha
            
            # Sample from Dirichlet distribution
            combined = dirichlet.mean(alpha)
            
            # Ensure proper normalization
            combined = np.maximum(combined, 1e-10)
            combined = combined / np.sum(combined)
            
            return combined
            
        except Exception as e:
            self.logger.warning(f"Probability combination failed: {str(e)}")
            if probability_lists:
                return probability_lists[0]
            return None
    
    def estimate_probabilities(self, numbers: np.ndarray, years: Optional[np.ndarray] = None,
                             days: Optional[np.ndarray] = None, is_bonus: bool = False) -> np.ndarray:
        """Estimate probabilities using enhanced methods"""
        cache_key = f"prob_{'bonus' if is_bonus else 'main'}_{len(numbers)}"
        
        def compute_probabilities():
            number_range = self.config.bonus_number_range if is_bonus else self.config.main_number_range
            
            probs = {
                'frequency': self._calculate_frequency_probabilities(numbers, number_range),
                'recent': self._calculate_recent_probabilities(numbers, number_range),
                'pattern': self._calculate_pattern_probabilities(numbers, is_bonus)
            }
            
            if years is not None and days is not None:
                probs['temporal'] = self._calculate_temporal_probabilities(
                    numbers, years, days, number_range
                )
            else:
                probs['temporal'] = np.ones(number_range) / number_range
            
            # Filter out None values
            valid_probs = [p for p in probs.values() if p is not None]
            
            if not valid_probs:
                return np.ones(number_range) / number_range
                
            combined_probs = self._combine_probabilities_dirichlet(valid_probs)
            if combined_probs is None:
                return np.ones(number_range) / number_range
                
            final_probs = self._ensure_minimum_probability(combined_probs)
            
            if not self._validate_probabilities(final_probs):
                self.logger.warning("Invalid probability distribution generated")
                return np.ones(number_range) / number_range
            
            return final_probs
            
        return self.cache.get_or_compute('probabilities', cache_key, compute_probabilities)
    
    def _calculate_frequency_probabilities(self, numbers: np.ndarray, 
                                        number_range: int) -> np.ndarray:
        """Calculate frequency-based probabilities"""
        freq = self.analytics.calculate_frequency(numbers, number_range)
        alpha = freq + self.config.base_alpha
        return dirichlet.mean(alpha)
    
    def _calculate_recent_probabilities(self, numbers: np.ndarray,
                                      number_range: int) -> np.ndarray:
        """Calculate probabilities based on recent history"""
        recent_window = min(100, len(numbers))
        recent_numbers = numbers[-recent_window:]
        freq = self.analytics.calculate_frequency(recent_numbers, number_range)
        
        # Inverse weighting for recent numbers
        inv_freq = 1 - freq
        alpha = inv_freq + self.config.base_alpha
        return dirichlet.mean(alpha)
    
    def _get_gmm_probabilities(self, model: Union[GaussianMixture, BayesianGaussianMixture],
                              number_range: int, n_numbers: int) -> np.ndarray:
        """Calculate probabilities from GMM model with correct feature dimensions"""
        try:
            # Create feature space for all possible numbers
            all_numbers = np.arange(1, number_range + 1)
            features = np.zeros((number_range, n_numbers * 2))
            
            # Calculate basic features
            angles = (all_numbers * 2 * np.pi) / number_range
            features[:, :n_numbers] = np.cos(angles)[:, np.newaxis]
            features[:, n_numbers:] = np.sin(angles)[:, np.newaxis]
            
            # Add spacing and range features (use means for single numbers)
            avg_spacing = np.mean(np.diff(np.sort(all_numbers)))
            avg_range = number_range / 2
            even_odd_ratio = 0.5
            
            # Create additional features
            spacing_features = np.full((number_range, n_numbers-1), avg_spacing)
            range_features = np.full((number_range, 1), avg_range)
            even_odd_features = np.full((number_range, 1), even_odd_ratio)
            
            # Combine all features
            combined_features = np.column_stack([
                features,
                spacing_features,
                range_features,
                even_odd_features
            ])
            
            # Get probabilities from model
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(combined_features)
                # Average across components if multiple
                if len(probs.shape) > 1:
                    probs = np.mean(probs, axis=1)
            else:
                # Fallback to log likelihood if predict_proba not available
                log_probs = model.score_samples(combined_features)
                probs = np.exp(log_probs)
                probs = probs / np.sum(probs)
            
            return probs
            
        except Exception as e:
            self.logger.warning(f"GMM probability calculation failed: {str(e)}")
            return np.ones(number_range) / number_range
    
    def _calculate_pattern_probabilities(self, numbers: np.ndarray,
                                       is_bonus: bool) -> np.ndarray:
        """Calculate pattern-based probabilities"""
        pattern_info = self.pattern_analyzer.analyze_patterns(numbers, is_bonus)
        number_range = self.config.bonus_number_range if is_bonus else self.config.main_number_range
        
        if not pattern_info or 'pattern_model' not in pattern_info:
            return np.ones(number_range) / number_range
        
        try:
            model = pattern_info['pattern_model']
            if isinstance(model, dict):
                probs = []
                if 'regular_gmm' in model:
                    p = self._get_gmm_probabilities(
                        model['regular_gmm'],
                        number_range,
                        self.config.n_bonus if is_bonus else self.config.n_main
                    )
                    if len(p) == number_range:
                        probs.append(p)
                
                if 'variational_gmm' in model:
                    p = self._get_gmm_probabilities(
                        model['variational_gmm'],
                        number_range,
                        self.config.n_bonus if is_bonus else self.config.n_main
                    )
                    if len(p) == number_range:
                        probs.append(p)
                
                if probs:
                    stacked_probs = np.stack(probs)
                    alpha = (stacked_probs.sum(axis=0) * len(probs) +
                            self.config.base_alpha)
                    return dirichlet.mean(alpha)
            
            return np.ones(number_range) / number_range
            
        except Exception as e:
            self.logger.warning(f"Pattern probability calculation failed: {str(e)}")
            return np.ones(number_range) / number_range
    
    def _calculate_temporal_probabilities(self, numbers: np.ndarray,
                                        years: np.ndarray, days: np.ndarray,
                                        number_range: int) -> np.ndarray:
        """Calculate temporal probabilities"""
        try:
            min_length = min(len(numbers), len(years), len(days))
            numbers = numbers[:min_length]
            years = years[:min_length]
            days = days[:min_length]
            
            decay_rate = self._optimize_decay(numbers)
            time_indices = np.arange(min_length)
            weights = decay_rate ** time_indices
            
            weekend_boost = 1 + 0.1 * days
            weights = weights * weekend_boost
            
            counts = np.zeros(number_range)
            for num, weight in zip(numbers, weights):
                if 1 <= num <= number_range:
                    counts[num-1] += weight
            
            alpha = counts + self.config.base_alpha
            return dirichlet.mean(alpha)
            
        except Exception as e:
            self.logger.warning(f"Temporal probability calculation failed: {str(e)}")
            return np.ones(number_range) / number_range
    
    def _optimize_decay(self, numbers: np.ndarray) -> float:
        """Optimize decay rate"""
        try:
            rates = np.linspace(0.8, 0.99, 20)
            best_rate = self.config.time_decay_factor
            best_score = float('-inf')

            for rate in rates:
                weights = rate ** np.arange(len(numbers)-1, -1, -1)
                score = np.sum(weights * numbers) / np.sum(weights)
                if score > best_score:
                    best_score = score
                    best_rate = rate

            return best_rate
            
        except Exception as e:
            self.logger.warning(f"Decay optimization failed: {str(e)}")
            return self.config.time_decay_factor
    
    def _ensure_minimum_probability(self, probabilities: np.ndarray) -> np.ndarray:
        """Ensure minimum probability with Dirichlet smoothing"""
        min_prob = 0.1 / len(probabilities)
        alpha = probabilities * 100 + min_prob
        return dirichlet.mean(alpha)
    
    def _validate_probabilities(self, probabilities: np.ndarray) -> bool:
        """Validate probability distribution"""
        try:
            if len(probabilities) == 0:
                return False
            
            if not np.all(np.isfinite(probabilities)):
                return False
            
            if not np.all(probabilities >= 0):
                return False
            
            if not np.isclose(np.sum(probabilities), 1.0, rtol=1e-5):
                return False
            
            return True
            
        except Exception:
            return False
        
class ResultAnalyzer:
    """Enhanced analysis of generated lottery numbers with confidence intervals"""
    def __init__(self, config: LotteryConfig, historical_data: Optional[np.ndarray] = None):
        self.config = config
        self.historical_data = historical_data
        self.analytics = SharedAnalytics()
        self.timestamp = datetime.now()
        self.logger = logging.getLogger(__name__)
    
    def _calculate_number_balance(self, numbers: np.ndarray) -> Dict:
        """Calculate number balance metrics for main and bonus numbers"""
        balance_metrics = {
            'main_numbers': {},
            'bonus_numbers': {}
        }
        
        try:
            # Calculate metrics for main numbers
            main_numbers = numbers[:, :self.config.n_main]
            main_metrics = self.analytics.calculate_balance_metrics(
                main_numbers.flatten(),
                self.config.main_number_range
            )
            main_spacing = self.analytics.calculate_spacing_metrics(
                main_numbers.flatten()
            )
            
            balance_metrics['main_numbers'].update(main_metrics)
            balance_metrics['main_numbers'].update(main_spacing)
            
            # Calculate metrics for bonus numbers
            bonus_numbers = numbers[:, self.config.n_main:]
            bonus_metrics = self.analytics.calculate_balance_metrics(
                bonus_numbers.flatten(),
                self.config.bonus_number_range
            )
            bonus_spacing = self.analytics.calculate_spacing_metrics(
                bonus_numbers.flatten()
            )
            
            balance_metrics['bonus_numbers'].update(bonus_metrics)
            balance_metrics['bonus_numbers'].update(bonus_spacing)
            
        except Exception as e:
            self.logger.warning(f"Error calculating number balance: {str(e)}")
            
        return balance_metrics
    
    def calculate_statistics(self, generated_numbers: np.ndarray) -> Dict:
        """Calculate comprehensive statistics"""
        stats = {
            'main_number_freq': np.zeros(self.config.main_number_range),
            'bonus_number_freq': np.zeros(self.config.bonus_number_range),
            'even_odd_ratios': [],
            'consecutive_numbers': 0,
            'pattern_similarity': 0,
            'confidence_intervals': {},
            'temporal_metrics': {},
            'number_balance': {
                'main_numbers': {},
                'bonus_numbers': {}
            },
            'timestamp': self.timestamp.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        try:
            # Calculate basic frequencies
            for draw in generated_numbers:
                main_numbers = draw[:self.config.n_main]
                bonus_numbers = draw[self.config.n_main:]
                
                # Update frequencies
                for num in main_numbers:
                    if 1 <= num <= self.config.main_number_range:
                        stats['main_number_freq'][num-1] += 1
                for num in bonus_numbers:
                    if 1 <= num <= self.config.bonus_number_range:
                        stats['bonus_number_freq'][num-1] += 1
                
                # Calculate even/odd ratio
                even_count = np.sum(main_numbers % 2 == 0)
                stats['even_odd_ratios'].append(even_count/self.config.n_main)
                
                # Check consecutive numbers
                sorted_numbers = np.sort(main_numbers)
                for i in range(len(sorted_numbers)-1):
                    if sorted_numbers[i+1] - sorted_numbers[i] == 1:
                        stats['consecutive_numbers'] += 1
            
            # Normalize frequencies
            stats['main_number_freq'] = stats['main_number_freq'] / len(generated_numbers)
            stats['bonus_number_freq'] = stats['bonus_number_freq'] / len(generated_numbers)
            
            # Calculate confidence intervals
            stats['confidence_intervals'] = self._calculate_confidence_intervals(
                generated_numbers, stats['main_number_freq'], stats['bonus_number_freq']
            )
            
            # Calculate temporal metrics if historical data is available
            if self.historical_data is not None:
                stats['temporal_metrics'] = self._calculate_temporal_metrics(generated_numbers)
                pattern_metrics = self._calculate_pattern_similarity(
                    stats['main_number_freq'],
                    stats['bonus_number_freq']
                )
                stats.update(pattern_metrics)
            
            # Calculate number balance metrics
            stats['number_balance'] = self._calculate_number_balance(generated_numbers)
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error calculating statistics: {str(e)}")
            return stats
    
    def _calculate_confidence_intervals(self, numbers: np.ndarray,
                                     main_freq: np.ndarray,
                                     bonus_freq: np.ndarray) -> Dict:
        """Calculate confidence intervals for metrics"""
        confidence_intervals = {}
        
        try:
            # Even/Odd ratio CI
            even_odd_ratios = []
            for draw in numbers:
                main_numbers = draw[:self.config.n_main]
                even_count = sum(1 for x in main_numbers if x % 2 == 0)
                even_odd_ratios.append(even_count/self.config.n_main)
            
            mean_eo = np.mean(even_odd_ratios)
            std_eo = np.std(even_odd_ratios)
            
            if len(numbers) > 1:
                ci_eo = norm.interval(0.95, loc=mean_eo, scale=std_eo/np.sqrt(len(numbers)))
                ci_lower = max(0, ci_eo[0])
                ci_upper = min(1, ci_eo[1])
            else:
                ci_lower = mean_eo
                ci_upper = mean_eo
                
            confidence_intervals['even_odd_ratio'] = {
                'mean': mean_eo,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper
            }
            
            # Frequency CIs
            for freq_type, freq_data in [('main_numbers', main_freq),
                                       ('bonus_numbers', bonus_freq)]:
                freq_std = np.std(freq_data)
                freq_mean = np.mean(freq_data)
                
                confidence_intervals[freq_type] = {
                    'mean_freq': freq_mean,
                    'ci_lower': np.maximum(0, freq_data - 1.96 * freq_std),
                    'ci_upper': np.minimum(1, freq_data + 1.96 * freq_std)
                }
            
        except Exception as e:
            self.logger.warning(f"Error calculating confidence intervals: {str(e)}")
            for metric in ['even_odd_ratio', 'main_numbers', 'bonus_numbers']:
                confidence_intervals[metric] = {
                    'mean': 0.5,
                    'ci_lower': 0.0,
                    'ci_upper': 1.0
                }
        
        return confidence_intervals
    
    def _calculate_temporal_metrics(self, generated_numbers: np.ndarray) -> Dict:
        """Calculate temporal metrics"""
        temporal_metrics = {
            'mean_spacing': {'value': 0, 'historical_mean': 0, 'historical_std': 0,
                           'z_score': 0, 'percentile': 0.5},
            'range_coverage': {'value': 0, 'historical_mean': 0, 'historical_std': 0,
                             'z_score': 0, 'percentile': 0.5},
            'even_odd_balance': {'value': 0.5, 'historical_mean': 0.5, 'historical_std': 0,
                               'z_score': 0, 'percentile': 0.5}
        }
        
        try:
            # Get metrics for generated numbers
            gen_stats = self._calculate_window_stats(generated_numbers)
            
            if self.historical_data is not None:
                # Calculate historical windows
                window_size = min(50, len(self.historical_data))
                historical_stats = []
                
                for i in range(len(self.historical_data) - window_size + 1):
                    window = self.historical_data[i:i+window_size]
                    stats = self._calculate_window_stats(window)
                    if stats:
                        historical_stats.append(stats)
                
                # Compare with historical patterns
                if historical_stats and gen_stats:
                    for metric in ['mean_spacing', 'range_coverage', 'even_odd_balance']:
                        hist_values = [h[metric] for h in historical_stats]
                        mean_hist = np.mean(hist_values)
                        std_hist = max(np.std(hist_values), 1e-10)
                        
                        z_score = (gen_stats[metric] - mean_hist) / std_hist
                        percentile = norm.cdf(z_score)
                        
                        temporal_metrics[metric].update({
                            'value': gen_stats[metric],
                            'historical_mean': mean_hist,
                            'historical_std': std_hist,
                            'z_score': z_score,
                            'percentile': percentile
                        })
            
        except Exception as e:
            self.logger.warning(f"Error calculating temporal metrics: {str(e)}")
        
        return temporal_metrics
    
    def _calculate_window_stats(self, numbers: np.ndarray) -> Dict:
        """Calculate statistics for a window of numbers"""
        try:
            if len(numbers.shape) == 1:
                numbers = numbers.reshape(-1, self.config.n_main + self.config.n_bonus)
            
            main_numbers = numbers[:, :self.config.n_main]
            
            # Calculate spacing metrics
            spacing_metrics = self.analytics.calculate_spacing_metrics(main_numbers)
            
            # Calculate balance metrics
            balance_metrics = self.analytics.calculate_balance_metrics(
                main_numbers, self.config.main_number_range
            )
            
            return {
                'mean_spacing': spacing_metrics['mean_spacing'],
                'range_coverage': balance_metrics['low_high_ratio'],
                'even_odd_balance': balance_metrics['even_odd_ratio']
            }
            
        except Exception as e:
            self.logger.warning(f"Error calculating window stats: {str(e)}")
            return None
    
    def _calculate_pattern_similarity(self, main_freq: np.ndarray,
                                   bonus_freq: np.ndarray) -> Dict:
        """Calculate pattern similarity between generated and historical numbers"""
        similarity_metrics = {
            'pattern_similarity': 0,
            'bonus_pattern_similarity': 0,
            'main_jensen_shannon': 0,
            'bonus_jensen_shannon': 0,
            'main_pattern_volatility': 0,
            'bonus_pattern_volatility': 0,
            'main_relative_entropy': 0,
            'bonus_relative_entropy': 0
        }
        
        try:
            if self.historical_data is not None:
                # Calculate historical frequencies
                historical_main_freq = np.zeros(self.config.main_number_range)
                historical_bonus_freq = np.zeros(self.config.bonus_number_range)
                
                for draw in self.historical_data:
                    main_numbers = draw[:self.config.n_main]
                    bonus_numbers = draw[self.config.n_main:]
                    
                    for num in main_numbers:
                        if 1 <= num <= self.config.main_number_range:
                            historical_main_freq[num-1] += 1
                    for num in bonus_numbers:
                        if 1 <= num <= self.config.bonus_number_range:
                            historical_bonus_freq[num-1] += 1
                
                # Normalize frequencies
                historical_main_freq = historical_main_freq / len(self.historical_data)
                historical_bonus_freq = historical_bonus_freq / len(self.historical_data)
                
                # Calculate similarities
                main_norm = np.linalg.norm(main_freq) * np.linalg.norm(historical_main_freq)
                if main_norm > 0:
                    similarity_metrics['pattern_similarity'] = np.dot(
                        main_freq, historical_main_freq
                    ) / main_norm
                
                bonus_norm = np.linalg.norm(bonus_freq) * np.linalg.norm(historical_bonus_freq)
                if bonus_norm > 0:
                    similarity_metrics['bonus_pattern_similarity'] = np.dot(
                        bonus_freq, historical_bonus_freq
                    ) / bonus_norm
                
                # Calculate Jensen-Shannon divergence
                main_m = 0.5 * (main_freq + historical_main_freq)
                main_js_div = 0.5 * (
                    entropy(main_freq + 1e-10, main_m + 1e-10) +
                    entropy(historical_main_freq + 1e-10, main_m + 1e-10)
                )
                similarity_metrics['main_jensen_shannon'] = 1 - main_js_div
                
                bonus_m = 0.5 * (bonus_freq + historical_bonus_freq)
                bonus_js_div = 0.5 * (
                    entropy(bonus_freq + 1e-10, bonus_m + 1e-10) +
                    entropy(historical_bonus_freq + 1e-10, bonus_m + 1e-10)
                )
                similarity_metrics['bonus_jensen_shannon'] = 1 - bonus_js_div
                
                # Calculate pattern volatility
                similarity_metrics['main_pattern_volatility'] = np.std(
                    main_freq - historical_main_freq
                )
                similarity_metrics['bonus_pattern_volatility'] = np.std(
                    bonus_freq - historical_bonus_freq
                )
                
                # Calculate relative entropy
                similarity_metrics['main_relative_entropy'] = entropy(
                    main_freq + 1e-10,
                    historical_main_freq + 1e-10
                )
                similarity_metrics['bonus_relative_entropy'] = entropy(
                    bonus_freq + 1e-10,
                    historical_bonus_freq + 1e-10
                )
            
        except Exception as e:
            self.logger.warning(f"Error calculating pattern similarity: {str(e)}")
        
        return similarity_metrics
    
    def format_results(self, generated_numbers: np.ndarray, stats: Dict) -> str:
        """Format analysis results into readable text"""
        formatted_data = [
            "# Lottery Number Analysis Report",
            f"Generated on: {stats['timestamp']}\n",
            "## Generated Numbers"
        ]
        
        # Format generated numbers
        for draw in generated_numbers:
            main_nums = draw[:self.config.n_main]
            bonus_nums = draw[self.config.n_main:]
            formatted_line = (
                " ".join(f"{num:02d}" for num in main_nums) +
                " | " +
                " ".join(f"{num:02d}" for num in bonus_nums)
            )
            formatted_data.append(formatted_line)
        
        # Pattern Analysis section
        formatted_data.extend([
            "\n## Pattern Analysis",
            f"Pattern Similarity Score: {stats.get('pattern_similarity', 0):.3f}",
            f"Bonus Pattern Similarity Score: {stats.get('bonus_pattern_similarity', 0):.3f}",
            f"Main Numbers Jensen-Shannon Similarity: {stats.get('main_jensen_shannon', 0):.3f}",
            f"Bonus Numbers Jensen-Shannon Similarity: {stats.get('bonus_jensen_shannon', 0):.3f}",
            f"Main Pattern Volatility: {stats.get('main_pattern_volatility', 0):.3f}",
            f"Bonus Pattern Volatility: {stats.get('bonus_pattern_volatility', 0):.3f}",
            f"Main Relative Entropy: {stats.get('main_relative_entropy', 0):.3f}",
            f"Bonus Relative Entropy: {stats.get('bonus_relative_entropy', 0):.3f}"
        ])
        
        # Confidence Intervals section
        formatted_data.extend([
            "\n### Confidence Intervals (95%)",
            f"Even/Odd Ratio: {stats['confidence_intervals']['even_odd_ratio']['mean']:.3f} "
            f"({stats['confidence_intervals']['even_odd_ratio']['ci_lower']:.3f} - "
            f"{stats['confidence_intervals']['even_odd_ratio']['ci_upper']:.3f})"
        ])
        
        # Number Balance section
        formatted_data.extend([
            "\n### Number Balance Metrics",
            "Main Numbers:",
            f"- Low/High Ratio: {stats['number_balance']['main_numbers'].get('low_high_ratio', 0):.3f}",
            f"- Spacing Consistency: {stats['number_balance']['main_numbers'].get('spacing_consistency', 0):.3f}",
            f"- Range Utilization: {len(np.nonzero(stats['main_number_freq'])[0]) / self.config.main_number_range:.3f}",
            "\nBonus Numbers:",
            f"- Low/High Ratio: {stats['number_balance']['bonus_numbers'].get('low_high_ratio', 0):.3f}",
            f"- Spacing Consistency: {stats['number_balance']['bonus_numbers'].get('spacing_consistency', 0):.3f}",
            f"- Range Utilization: {len(np.nonzero(stats['bonus_number_freq'])[0]) / self.config.bonus_number_range:.3f}"
        ])
        
        # Temporal Metrics section
        if 'temporal_metrics' in stats:
            formatted_data.extend([
                "\n### Temporal Analysis",
                "Mean Spacing:",
                f"- Value: {stats['temporal_metrics']['mean_spacing']['value']:.3f}",
                f"- Historical Mean: {stats['temporal_metrics']['mean_spacing']['historical_mean']:.3f}",
                f"- Z-score: {stats['temporal_metrics']['mean_spacing']['z_score']:.3f}",
                f"- Percentile: {stats['temporal_metrics']['mean_spacing']['percentile']:.3f}",
                "\nRange Coverage:",
                f"- Value: {stats['temporal_metrics']['range_coverage']['value']:.3f}",
                f"- Historical Mean: {stats['temporal_metrics']['range_coverage']['historical_mean']:.3f}",
                f"- Z-score: {stats['temporal_metrics']['range_coverage']['z_score']:.3f}",
                f"- Percentile: {stats['temporal_metrics']['range_coverage']['percentile']:.3f}",
                "\nEven/Odd Balance:",
                f"- Value: {stats['temporal_metrics']['even_odd_balance']['value']:.3f}",
                f"- Historical Mean: {stats['temporal_metrics']['even_odd_balance']['historical_mean']:.3f}",
                f"- Z-score: {stats['temporal_metrics']['even_odd_balance']['z_score']:.3f}",
                f"- Percentile: {stats['temporal_metrics']['even_odd_balance']['percentile']:.3f}"
            ])
        
        # Frequency Analysis section - Main Numbers
        formatted_data.extend([
            "\n### Frequency Analysis",
            "Main Numbers:",
            "Number | Frequency",
            "--------|----------"
        ])
        
        for i, freq in enumerate(stats['main_number_freq']):
            formatted_data.append(f"{i+1:6d} | {freq:.3f}")
        
        # Frequency Analysis section - Bonus Numbers
        formatted_data.extend([
            "\nBonus Numbers:",
            "Number | Frequency",
            "--------|----------"
        ])
        
        for i, freq in enumerate(stats['bonus_number_freq']):
            formatted_data.append(f"{i+1:6d} | {freq:.3f}")
        
        # Join all sections with newlines
        return "\n".join(formatted_data)

    def save_analysis(self, filename: str, generated_numbers: np.ndarray, stats: Dict) -> bool:
        """Save analysis results to file"""
        try:
            formatted_results = self.format_results(generated_numbers, stats)
            with open(filename, 'w') as f:
                f.write(formatted_results)
            return True
        except Exception as e:
            self.logger.error(f"Failed to save analysis: {str(e)}")
            return False
    
    def load_analysis(self, filename: str) -> Optional[str]:
        """Load previous analysis results"""
        try:
            with open(filename, 'r') as f:
                return f.read()
        except Exception as e:
            self.logger.error(f"Failed to load analysis: {str(e)}")
            return None

# Optional utility functions for command-line usage
def parse_arguments():
    """Parse command line arguments"""
    import argparse
    parser = argparse.ArgumentParser(description='Lottery Number Generator and Analyzer')
    parser.add_argument('--input', type=str, required=True, help='Input file with historical data')
    parser.add_argument('--output', type=str, default='lottery_analysis.txt', help='Output file for analysis')
    parser.add_argument('--draws', type=int, default=3, help='Number of draws to generate')
    parser.add_argument('--cache', type=str, default='pattern_cache.joblib', help='Cache file location')
    return parser.parse_args()

def run_from_command_line():
    """Run the lottery system from command line with comprehensive error handling"""
    args = parse_arguments()
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize configuration
        logger.info("Initializing lottery system configuration...")
        config = LotteryConfig()
        if not config.validate():
            logger.error("Invalid configuration parameters")
            return
        
        # Set up data management
        logger.info(f"Loading data from {args.input}...")
        data_manager = DataManager(config)
        try:
            data = data_manager.load_data(args.input)
        except LotteryDataError as e:
            logger.error(f"Data loading failed: {str(e)}")
            return
        except Exception as e:
            logger.error(f"Unexpected error loading data: {str(e)}")
            return
            
        if data is None or len(data) == 0:
            logger.error("No valid data loaded")
            return
        
        logger.info(f"Successfully loaded {len(data)} historical draws")
        
        # Initialize analysis components
        logger.info("Initializing analysis components...")
        pattern_analyzer = PatternAnalyzer(config)
        probability_estimator = ProbabilityEstimator(config, pattern_analyzer)
        number_generator = NumberGenerator(config, pattern_analyzer)
        result_analyzer = ResultAnalyzer(config, data)
        
        # Load cache if available
        logger.info(f"Loading pattern cache from {args.cache}...")
        cache_loaded = pattern_analyzer.cache.load_state(args.cache)
        if not cache_loaded:
            logger.warning("Failed to load cache, continuing with empty cache")
        
        # Get main and bonus numbers
        logger.info("Extracting main and bonus numbers...")
        try:
            main_numbers = data_manager.get_main_numbers()
            bonus_numbers = data_manager.get_bonus_numbers()
        except Exception as e:
            logger.error(f"Failed to extract numbers: {str(e)}")
            return
        
        # Calculate probabilities
        logger.info("Calculating number probabilities...")
        try:
            main_probs = probability_estimator.estimate_probabilities(
                main_numbers,
                years=data_manager.years,
                days=data_manager.days,
                is_bonus=False
            )
            
            bonus_probs = probability_estimator.estimate_probabilities(
                bonus_numbers,
                years=data_manager.years,
                days=data_manager.days,
                is_bonus=True
            )
            
            if main_probs is None or bonus_probs is None:
                logger.error("Failed to calculate probabilities")
                return
                
        except Exception as e:
            logger.error(f"Probability calculation failed: {str(e)}")
            return
        
        # Generate numbers
        logger.info(f"Generating {args.draws} number combinations...")
        try:
            generated_lists = number_generator.generate_numbers(
                args.draws,
                main_probs,
                bonus_probs,
                historical_data=data
            )
            
            if len(generated_lists) == 0:
                logger.error("Failed to generate any valid number combinations")
                return
            elif len(generated_lists) < args.draws:
                logger.warning(
                    f"Only generated {len(generated_lists)} of {args.draws} "
                    "requested combinations"
                )
            
        except Exception as e:
            logger.error(f"Number generation failed: {str(e)}")
            return
        
        # Calculate and save statistics
        logger.info("Calculating statistics for generated numbers...")
        try:
            stats = result_analyzer.calculate_statistics(generated_lists)
            if stats:
                success = result_analyzer.save_analysis(args.output, generated_lists, stats)
                if not success:
                    logger.error(f"Failed to save analysis to {args.output}")
                    return
            else:
                logger.error("Failed to calculate statistics")
                return
                
        except Exception as e:
            logger.error(f"Statistics calculation failed: {str(e)}")
            return
        
        # Save cache for future use
        logger.info(f"Saving pattern cache to {args.cache}...")
        try:
            cache_saved = pattern_analyzer.cache.save_state(args.cache)
            if not cache_saved:
                logger.warning("Failed to save cache state")
        except Exception as e:
            logger.warning(f"Cache saving failed: {str(e)}")
        
        # Display generated numbers
        logger.info("\nGenerated number combinations:")
        for i, numbers in enumerate(generated_lists, 1):
            main = numbers[:config.n_main]
            bonus = numbers[config.n_main:]
            logger.info(
                f"Combination {i}: "
                f"Main: {' '.join(f'{n:02d}' for n in main)} | "
                f"Bonus: {' '.join(f'{n:02d}' for n in bonus)}"
            )
        
        # Final success message
        logger.info("\nLottery number generation completed successfully")
        logger.info(f"Full analysis saved to: {args.output}")
        
        # Return generated numbers for potential further use
        return generated_lists
        
    except KeyboardInterrupt:
        logger.info("\nOperation cancelled by user")
        return None
    except MemoryError:
        logger.error("Insufficient memory to complete the operation")
        return None
    except Exception as e:
        logger.error(f"Unexpected error in main execution: {str(e)}")
        logger.debug("Stack trace:", exc_info=True)
        return None

if __name__ == '__main__':
    run_from_command_line()

# python lottery_system.py --input lottery_numbers.txt --output analysis.txt --draws 3