# How to Run

## Prerequisites
Ensure you have the required dependencies installed:
```bash
pip install pandas numpy tensorflow xgboost tensorflow-probability scikit-learn
```
*Note: On macOS, you might need `brew install libomp` for XGBoost.*

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

### Generate More Predictions
To generate a specific number of predictions (e.g., 10):
```bash
python3 main.py --predictions 10
```

## Output
Predictions are saved to `predictions/hybrid_predictions.json` and displayed in the console.

## All-in-One Command
To run the full pipeline (Train -> PPO Fine-tune -> Predict):
```bash
python3 main.py --force-train --rl-train --predictions 5
```
