from pathlib import Path

# Lottery Rules
MAIN_NUMBER_RANGE = 50
BONUS_NUMBER_RANGE = 12
N_MAIN = 5
N_BONUS = 2

# File Paths
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
DATA_DIR = PROJECT_ROOT / "lottery_data"
DATA_FILE = DATA_DIR / "lottery_numbers.txt"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
PREDICTIONS_DIR = PROJECT_ROOT / "predictions"

# Ensure directories exist
for directory in [MODELS_DIR, LOGS_DIR, PREDICTIONS_DIR]:
    directory.mkdir(exist_ok=True)

# Model Configuration
NEURAL_MODEL_PARAMS = {
    "sequence_length": 20,
    "batch_size": 32,
    "epochs": 180,
    "learning_rate": 0.0005
}

# Transformer architecture scaling
NEURAL_ARCH_CONFIG = {
    "d_model": 256,          # balanced capacity for this dataset size
    "num_layers": 8,         # deep enough without excessive overfitting risk
    "num_heads": 8,          # used with per-head key_dim in neural model
    "ff_multiplier": 3.0,    # feed-forward expansion
    "dropout": 0.15,         # dropout inside blocks
    "label_smoothing": 0.05  # helps calibration and reduces overconfidence
}

STATISTICAL_CONFIG = {
    "recent_window": 100,
    "hot_cold_window": 50,
    "frequency_weight": 0.6,
    "recency_weight": 0.4,
    "elite_candidates": 12,
    "elite_main_step": 3,
    "elite_bonus_step": 1,
    "pattern_window": 240,
    "pair_window": 300,
    "main_top_pool": 16,
    "bonus_top_pool": 8,
    "main_diversity_penalty": 0.08,
    "bonus_diversity_penalty": 0.04,
}

PPO_PARAMS = {
    "learning_rate": 0.00005,
    "clip_ratio": 0.15,
    "gamma": 0.99,
    "lam": 0.95,
    "entropy_coef": 0.05,    # Increased from 0.02 for better exploration
    "batch_size": 32,
    "epochs": 80
}

# Sampling configuration (used for all model samplers)
SAMPLING_CONFIG = {
    # Balanced sampling: focus on high-probability numbers while maintaining diversity
    # Higher temperature = more exploration, lower = more exploitation
    "temperature_main": 0.65,      # Increased for better diversity (was 0.40)
    "temperature_bonus": 0.60,     # Increased for better diversity (was 0.40)
    "top_k_main": 12,              # Expanded candidate pool (was 4) - now 24% of range
    "top_k_bonus": 6,              # Expanded candidate pool (was 3) - now 50% of range
    "top_p_main": 0.85,            # Nucleus sampling: cumulative probability threshold
    "top_p_bonus": 0.80            # Alternative to top-k for adaptive cutoff
}
