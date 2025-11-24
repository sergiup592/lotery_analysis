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
    "sequence_length": 10,
    "batch_size": 32,
    "epochs": 120,
    "learning_rate": 0.001
}

# Transformer architecture scaling
NEURAL_ARCH_CONFIG = {
    "d_model": 256,          # base hidden size (was 192)
    "num_layers": 8,         # transformer encoder layers (was 6)
    "num_heads": 8,          # attention heads (was 4)
    "ff_multiplier": 2.0,    # feed-forward expansion
    "dropout": 0.1,          # dropout inside blocks
    "label_smoothing": 0.05  # helps calibration and reduces overconfidence
}

STATISTICAL_CONFIG = {
    "recent_window": 100,
    "hot_cold_window": 50
}

PPO_PARAMS = {
    "learning_rate": 0.00005,
    "clip_ratio": 0.2,
    "gamma": 0.99,
    "lam": 0.95,
    "entropy_coef": 0.01,
    "batch_size": 64
}

# Sampling configuration (used for all model samplers)
SAMPLING_CONFIG = {
    # Sharper sampling to focus on the highest-probability numbers while retaining some exploration
    "temperature_main": 0.50,
    "temperature_bonus": 0.50,
    "top_k_main": 5,   # limit draws to top-K highest probabilities
    "top_k_bonus": 3
}
