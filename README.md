# How to Run

## Prerequisites
Ensure you have the required dependencies installed:
```bash
pip install pandas numpy tensorflow xgboost tensorflow-probability scikit-learn
```
*Note: On macOS, you might need `brew install libomp` for XGBoost.*

### Apple Silicon GPU Notes
On Apple Silicon, only TensorFlow can use the GPU (via Metal). XGBoost and scikit-learn run on CPU.

To enable the TensorFlow GPU on Apple Silicon:
```bash
pip3 install tensorflow-macos tensorflow-metal
```

If TensorFlow is not built with Metal support, it will fall back to CPU and the logs will report that no GPU is available.

## Basic Usage
To generate predictions using the trained models:
```bash
python3 main.py
```

## Advanced Options

### Retrain Models
To force retraining of the Neural Network (Transformer) and other models:
```bash
python3 main.py --force-train
```

### Reinforcement Learning (PPO)
To fine-tune the models using Proximal Policy Optimization (PPO) for better accuracy:
```bash
python3 main.py --rl-train
```

### Backtest + Accuracy Graph
Run a walk-forward backtest and save an accuracy graph to `predictions/backtest_accuracy.png`:
```bash
python3 main.py --backtest --backtest-window 50 --backtest-topk 2
```

### Generate More Predictions
To generate a specific number of predictions (e.g., 10):
```bash
python3 main.py --predictions 10
```

### Coverage Strategy (Non-Predictive)
Generate tickets that maximize coverage across your ticket budget:
```bash
python3 main.py --strategy coverage --predictions 10
```
Optional: increase candidate search per ticket for better coverage.
```bash
python3 main.py --strategy coverage --predictions 10 --coverage-candidates 4000
```

## Output
Predictions are saved to `predictions/hybrid_predictions.json` (hybrid strategy) or
`predictions/coverage_predictions.json` (coverage strategy) and displayed in the console.

## All-in-One Command
To run the full pipeline (Train -> PPO Fine-tune -> Predict):
```bash
python3 main.py --force-train --rl-train --predictions 5
```
