# Lottery Number Analysis System

## Overview
A Python-based lottery number analysis system that provides tools for analyzing historical lottery data and generating number combinations. This system includes pattern analysis, probability estimation, and comprehensive statistical reporting capabilities.

**Important Disclaimer:** This system is for academic and entertainment purposes only. It cannot predict lottery numbers or increase chances of winning. All lottery draws are completely random and independent events.

## Features
- Historical data analysis
- Pattern recognition and analysis
- Probability estimation
- Number generation with configurable constraints
- Statistical analysis and reporting
- Caching system for performance optimization
- Comprehensive validation checks
- Detailed result analysis with confidence intervals

## Requirements
- Python 3.7+
- Required packages:
  - numpy
  - scipy
  - scikit-learn
  - joblib
  - logging
  - typing
  - dataclasses
  - collections

## Installation
1. Clone the repository:
```bash
git clone [repository-url]
```

2. Install required packages:
```bash
pip install numpy scipy scikit-learn joblib
```

## Usage

### Command Line Interface
```bash
python lottery_system.py --input lottery_numbers.txt --output analysis.txt --draws 3
```

Arguments:
- `--input`: Path to input file containing historical lottery data
- `--output`: Path for output analysis file (default: lottery_analysis.txt)
- `--draws`: Number of draws to generate (default: 3)
- `--cache`: Cache file location (default: pattern_cache.joblib)

### Python API
```python
from lottery_system import LotteryConfig, DataManager, PatternAnalyzer, ProbabilityEstimator, NumberGenerator, ResultAnalyzer

# Initialize components
config = LotteryConfig()
data_manager = DataManager(config)
pattern_analyzer = PatternAnalyzer(config)
probability_estimator = ProbabilityEstimator(config, pattern_analyzer)
number_generator = NumberGenerator(config, pattern_analyzer)
result_analyzer = ResultAnalyzer(config)

# Load and analyze data
data = data_manager.load_data('lottery_numbers.txt')
main_numbers = data_manager.get_main_numbers()
bonus_numbers = data_manager.get_bonus_numbers()

# Generate numbers
main_probs = probability_estimator.estimate_probabilities(main_numbers)
bonus_probs = probability_estimator.estimate_probabilities(bonus_numbers, is_bonus=True)
generated_numbers = number_generator.generate_numbers(3, main_probs, bonus_probs, historical_data=data)

# Analyze results
stats = result_analyzer.calculate_statistics(generated_numbers)
result_analyzer.save_analysis('analysis.txt', generated_numbers, stats)
```

## Configuration
The system can be configured using the `LotteryConfig` class:

```python
config = LotteryConfig(
    n_main=5,                  # Number of main numbers in each draw
    n_bonus=2,                 # Number of bonus numbers in each draw
    main_number_range=50,      # Range for main numbers
    bonus_number_range=12,     # Range for bonus numbers
    base_alpha=1.0,           # Base concentration parameter
    time_decay_factor=0.95,   # Time decay factor for historical data
    cache_size=1000,          # Maximum cache size
    default_window_size=10,   # Default analysis window size
    cv_splits=5,              # Cross-validation splits
    random_state=42           # Random seed for reproducibility
)
```

## Input File Format
The input file should contain lottery numbers in plain text format:
- One number per line
- Numbers for each draw should be in sequence
- Main numbers followed by bonus numbers
- Example:
```
7
12
23
31
45
3
8
```

## Output Analysis
The system generates a comprehensive analysis report including:
- Generated number combinations
- Pattern analysis
- Confidence intervals
- Number balance metrics
- Temporal analysis
- Frequency analysis for main and bonus numbers

## Error Handling
The system includes comprehensive error handling and validation:
- Input data validation
- Configuration validation
- Runtime error handling
- Detailed logging

## Caching
The system implements caching for performance optimization:
- Pattern analysis results
- Probability calculations
- Analysis results
