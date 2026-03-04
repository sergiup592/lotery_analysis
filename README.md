# Lottery Analysis System

This project builds and evaluates lottery ticket candidates for a fixed **5 + 2** format:

- Main numbers: choose 5 from `1..50`
- Bonus numbers: choose 2 from `1..12`
- Input data: [`lottery_data/lottery_numbers.txt`](lottery_data/lottery_numbers.txt)

It supports two generation modes:

- `hybrid`: model-based candidate generation + consensus ranking
- `coverage`: non-predictive combinatorial diversification

It also includes leak-safe backtesting, a readiness gate, abstain mode, and optional neural PPO fine-tuning.

## Important Reality Check

This repository is a research/engineering system for ranking ticket candidates, not a guaranteed way to predict lottery outcomes.  
Backtest improvements do not imply real-world certainty.

## Table of Contents

- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Data Format](#data-format)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Modes and Workflows](#modes-and-workflows)
- [Full CLI Reference](#full-cli-reference)
- [Output Files](#output-files)
- [Environment Variables and Acceleration](#environment-variables-and-acceleration)
- [Exit Codes](#exit-codes)

## How It Works

### Hybrid flow (`--strategy hybrid`)

1. Parse and normalize historical draws from `lottery_data/lottery_numbers.txt`.
2. Select model sources from `--model-profile` (and optional `--use-*` flags).
3. Run readiness gate by default (unless `--no-readiness-gate`).
4. Train/initialize enabled models:
   - Statistical
   - RandomForest (optional)
   - ExtraTrees (optional)
   - XGBoost (optional)
   - Neural Transformer (optional)
5. Generate candidate tickets from active sources.
6. Optionally generate ensemble candidates if at least 2 probabilistic sources are available.
7. Rank with adaptive consensus weights + statistical filters + diversity-aware selection.
8. Optionally apply live abstain filter.
9. Save to `predictions/hybrid_predictions.json`.

### Coverage flow (`--strategy coverage`)

- Generates tickets by maximizing coverage of number pairs and cross-pairs.
- Uses random candidate search per ticket (`--coverage-candidates`).
- Saves to `predictions/coverage_predictions.json`.

### Backtesting

- Walk-forward, leak-safe backtest over recent draws.
- Supports random baseline comparison, abstain mode, and consensus profile auto-tuning.
- Saves summary JSON + plot PNG artifacts under `predictions/`.

### Readiness gate (default before hybrid generation)

Runs an internal backtest and blocks hybrid generation unless criteria pass:

- minimum evaluated draws
- minimum evaluated tickets
- minimum average lift vs random baseline
- minimum latest rolling lift
- minimum confidence-interval lower bound for lift
- minimum ROI per ticket
- minimum ROI CI lower bound

If blocked and not overridden, process exits with code `2`.

## Project Structure

```text
.
├── main.py
├── requirements.txt
├── lottery_data/
│   └── lottery_numbers.txt
├── src/
│   ├── data.py            # parser + feature engineering
│   ├── statistical.py     # statistical source
│   ├── tree.py            # random forest + extra trees
│   ├── xgboost_model.py   # xgboost source
│   ├── neural.py          # transformer + PPO fine-tuning
│   ├── consensus.py       # ranking / diversity / blending
│   ├── backtest.py        # backtest + readiness + comparison
│   ├── coverage.py        # coverage optimizer
│   ├── filters.py         # statistical hard/soft filters
│   ├── acceleration.py    # TF/XGBoost device config helpers
│   └── config.py          # constants and paths
├── models/                # persisted neural model files
├── predictions/           # generated artifacts
└── logs/                  # runtime logs
```

## Data Format

The parser expects repeated draw blocks that begin with a weekday line and include a date, then 7 numbers, then jackpot/result.

Example:

```text
Tuesday
13th January 2026
6
10
18
44
47
2
10
€64,537,877    Roll
```

Notes:

- Dates support formats like `Tuesday 13 January 2026` and `Tuesday 13 Jan 2026`.
- Duplicate draw dates are deduplicated (latest block kept).
- Main/bonus counts and ranges are validated.

## Installation

1. Create and activate a virtual environment.
2. Install dependencies.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` includes:

- `pandas`, `numpy`, `scikit-learn`, `xgboost`, `tensorflow`, `scipy`, `matplotlib`

On macOS, you may need OpenMP:

```bash
brew install libomp
```

Optional Apple Silicon acceleration:

```bash
pip install tensorflow-macos tensorflow-metal
```

## Quick Start

Generate hybrid predictions with defaults:

```bash
python3 main.py
```

Generate coverage-only tickets:

```bash
python3 main.py --strategy coverage --predictions 10
```

Run backtest:

```bash
python3 main.py --backtest --backtest-window 50 --backtest-topk 2
```

Run baseline vs improved comparison:

```bash
python3 main.py --backtest --backtest-compare --backtest-window 50 --backtest-topk 2
```

## Modes and Workflows

### Model profiles (`hybrid`)

- `validated` (default): statistical only
- `balanced`: statistical + random forest
- `full`: statistical + neural + random forest + extra trees + xgboost

You can add sources explicitly with:

- `--use-neural`
- `--use-rf`
- `--use-et`
- `--use-xgb`

`--force-train` and `--rl-train` implicitly enable neural source.

### Neural training and PPO

Train neural weights from scratch:

```bash
python3 main.py --force-train
```

Run PPO fine-tuning (requires usable neural models):

```bash
python3 main.py --rl-train
```

Train + PPO + generate:

```bash
python3 main.py --force-train --rl-train --predictions 5
```

Neural artifacts are saved in `models/`:

- `models/main_transformer.keras`
- `models/bonus_transformer.keras`

### Readiness gate behavior

Default behavior for hybrid mode:

- readiness gate is ON
- readiness backtest uses abstain mode by default
- readiness backtest auto-tunes consensus profile by default
- recommended consensus profile from readiness is reused in live generation

Override controls:

```bash
python3 main.py --no-readiness-gate
python3 main.py --readiness-allow-unready
python3 main.py --readiness-no-abstain --readiness-no-auto-tune-consensus
```

### Live abstain mode

Skip weak live tickets instead of forcing output:

```bash
python3 main.py \
  --no-readiness-gate \
  --enable-abstain \
  --abstain-min-score 0.12 \
  --abstain-min-expected-main-prob 0.35
```

If all candidates fail thresholds, `hybrid_predictions.json` is saved as an empty list.

### Backtest speed controls

```bash
python3 main.py \
  --backtest \
  --backtest-fast-models \
  --backtest-fast-level ultrafast \
  --backtest-model-retrain-interval 3
```

## Full CLI Reference

### Core options

| Flag | Type / Choices | Default | Description |
|---|---|---|---|
| `--predictions` | `int` | `5` | Number of predictions to generate. |
| `--strategy` | `hybrid \| coverage` | `hybrid` | Prediction strategy. |
| `--model-profile` | `validated \| balanced \| full` | `validated` | Model set for hybrid strategy. |
| `--use-neural` | flag | `False` | Include neural model in hybrid prediction. |
| `--use-rf` | flag | `False` | Include random forest model in hybrid prediction. |
| `--use-et` | flag | `False` | Include extra trees model in hybrid prediction. |
| `--use-xgb` | flag | `False` | Include XGBoost model in hybrid prediction. |
| `--coverage-candidates` | `int` | `2000` | Random candidates per ticket in coverage mode. |
| `--force-train` | flag | `False` | Force retraining of neural models. |
| `--rl-train` | flag | `False` | Run PPO fine-tuning. |
| `--seed` | `int` | `42` | Reproducible random seed. |

### Backtest options

| Flag | Type / Choices | Default |
|---|---|---|
| `--backtest` | flag | `False` |
| `--backtest-compare` | flag | `False` |
| `--backtest-window` | `int` | `50` |
| `--backtest-topk` | `int` | `2` |
| `--backtest-use-neural` | flag | `False` |
| `--backtest-neural-mode` | `frozen \| rolling` | `frozen` |
| `--backtest-neural-retrain-interval` | `int` | `8` |
| `--backtest-neural-train-window` | `int` | `400` |
| `--backtest-neural-epochs` | `int` | `10` |
| `--backtest-neural-batch-size` | `int` | `32` |
| `--backtest-neural-min-retrains` | `int` | `2` |
| `--backtest-no-ensemble` | flag | `False` |
| `--backtest-no-filter-fit` | flag | `False` |
| `--backtest-skip-statistical` | flag | `False` |
| `--backtest-skip-rf` | flag | `False` |
| `--backtest-skip-et` | flag | `False` |
| `--backtest-skip-xgb` | flag | `False` |
| `--backtest-fast-models` | flag | `False` |
| `--backtest-fast-level` | `fast \| ultrafast` | `fast` |
| `--backtest-model-retrain-interval` | `int` | `1` |
| `--backtest-no-source-performance` | flag | `False` |
| `--backtest-source-performance-window` | `int` | `20` |
| `--backtest-source-performance-min-samples` | `int` | `4` |
| `--backtest-enable-abstain` | flag | `False` |
| `--backtest-abstain-min-score` | `float` | `0.10` |
| `--backtest-abstain-min-confidence` | `float` | `0.0` |
| `--backtest-abstain-min-expected-main-prob` | `float` | `0.0` |
| `--backtest-abstain-min-support-count` | `int` | `1` |
| `--backtest-auto-tune-consensus` | flag | `False` |
| `--backtest-auto-tune-interval` | `int` | `3` |
| `--backtest-auto-tune-window` | `int` | `24` |
| `--backtest-auto-tune-exploration` | `float` | `0.15` |

### Readiness gate options

| Flag | Type / Choices | Default |
|---|---|---|
| `--no-readiness-gate` | flag | `False` |
| `--readiness-allow-unready` | flag | `False` |
| `--readiness-window` | `int` | `20` |
| `--readiness-topk` | `int` | `2` |
| `--readiness-fast-level` | `default \| fast \| ultrafast` | `ultrafast` |
| `--readiness-model-retrain-interval` | `int` | `3` |
| `--readiness-min-draws` | `int` | `20` |
| `--readiness-min-avg-lift` | `float` | `0.03` |
| `--readiness-min-rolling-lift` | `float` | `0.02` |
| `--readiness-rolling-window` | `int` | `8` |
| `--readiness-confidence` | `float` | `0.90` |
| `--readiness-min-ci-lower` | `float` | `0.0` |
| `--readiness-report-tag` | `str` | `readiness` |
| `--readiness-min-tickets` | `int` | `10` |
| `--readiness-min-roi-per-ticket` | `float` | `0.0` |
| `--readiness-min-roi-ci-lower` | `float` | `0.0` |
| `--readiness-profit-confidence` | `float` | `0.90` |
| `--readiness-no-abstain` | flag | `False` |
| `--readiness-abstain-min-score` | `float` | `0.12` |
| `--readiness-abstain-min-confidence` | `float` | `0.0` |
| `--readiness-abstain-min-expected-main-prob` | `float` | `0.35` |
| `--readiness-abstain-min-support-count` | `int` | `1` |
| `--readiness-no-auto-tune-consensus` | flag | `False` |
| `--readiness-auto-tune-interval` | `int` | `3` |
| `--readiness-auto-tune-window` | `int` | `24` |
| `--readiness-auto-tune-exploration` | `float` | `0.15` |

### Live abstain options (hybrid generation)

| Flag | Type | Default |
|---|---|---|
| `--enable-abstain` | flag | `False` |
| `--abstain-min-score` | `float` | `0.12` |
| `--abstain-min-confidence` | `float` | `0.0` |
| `--abstain-min-expected-main-prob` | `float` | `0.35` |
| `--abstain-min-support-count` | `int` | `1` |

## Output Files

Generated under `predictions/`:

- `hybrid_predictions.json`: final hybrid picks (possibly empty if abstain filters all).
- `coverage_predictions.json`: coverage-optimized picks (coverage strategy runs).
- `backtest_summary.json`: summary from a standard backtest run.
- `backtest_accuracy.png`: rolling performance and hit distribution plot from a standard backtest run.
- `backtest_summary_<tag>.json`: tagged summaries (e.g., baseline/improved/readiness) when tagged runs are used.
- `backtest_accuracy_<tag>.png`: tagged plots (e.g., baseline/improved/readiness) when tagged runs are used.
- `backtest_comparison.json`: controlled baseline vs improved comparison output (`--backtest-compare`).
- `<report_tag>_report.json`: readiness gate report (`readiness_report.json` by default).

Other runtime outputs:

- `logs/lottery_system.log`: application logs.
- `models/main_transformer.keras`, `models/bonus_transformer.keras`: saved neural models.

## Environment Variables and Acceleration

- `XGBOOST_FORCE_CPU=1`: force XGBoost to CPU.
- `XGBOOST_FORCE_GPU=1`: force XGBoost GPU mode if CUDA build exists.
- `TF_MIXED_PRECISION=0`: disable TensorFlow mixed precision (enabled by default when GPU is available).

The app also sets `MPLCONFIGDIR` to `predictions/.mplconfig` to keep Matplotlib cache writable.

## Exit Codes

- `0`: successful completion.
- `1`: fatal runtime error.
- `2`: readiness gate blocked hybrid generation (unless overridden with `--readiness-allow-unready`).

## Additional Notes

- Backtest ROI uses a simplified hypothetical payout curve based on **main hits only** (`3->10`, `4->100`, `5->100000`) and ticket cost `1`.
- Ensemble candidates are generated only when at least 2 probabilistic sources are present.
- If neural weights are missing and not force-trained, neural source is skipped with a warning.
