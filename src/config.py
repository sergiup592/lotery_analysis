from pathlib import Path

# EuroMillions game structure: pick 5 mains from 1-50 and 2 lucky stars from 1-12.
MAIN_NUMBER_RANGE = 50
BONUS_NUMBER_RANGE = 12
N_MAIN = 5
N_BONUS = 2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "lottery_data"
PREDICTIONS_DIR = PROJECT_ROOT / "predictions"

for directory in (DATA_DIR, PREDICTIONS_DIR):
    directory.mkdir(exist_ok=True)

# Ticket selection defaults (diversification across a generated set).
DEFAULT_OVERLAP_PENALTY = 0.90
DEFAULT_COVERAGE_BONUS = 0.85
DEFAULT_CANDIDATE_POOL_SIZE = 600
DEFAULT_POPULARITY_WEIGHT = 0.5

TICKET_COST = 2.50  # EUR per line

# The lucky-star range changed over the lottery's history:
# 1-9 at launch (2004), 1-11 from May 2011, 1-12 from 24 September 2016.
# Winner-count ingestion floors at this date so all 13 current tiers exist.
BONUS_ERA_START = "2016-09-24"
