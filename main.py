import argparse
import logging
import json
import sys
from src.config import LOGS_DIR, PREDICTIONS_DIR
from src.data import LotteryDataManager
from src.neural import NeuralModel
from src.statistical import StatisticalModel
from src.tree import RandomForestModel
from src.xgboost_model import XGBoostModel
from src.consensus import ConsensusEngine

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / "lottery_system.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Hybrid Lottery Number Generator")
    parser.add_argument("--predictions", type=int, default=5, help="Number of predictions to generate")
    parser.add_argument("--force-train", action="store_true", help="Force retraining of neural models")
    parser.add_argument("--rl-train", action="store_true", help="Fine-tune models using PPO (RL)")
    parser.add_argument("--backtest", action="store_true", help="Run backtesting framework")
    parser.add_argument("--backtest-window", type=int, default=50, help="Number of past draws to backtest")
    parser.add_argument("--backtest-topk", type=int, default=2, help="Number of top-ranked predictions to evaluate per draw during backtest")
    parser.add_argument("--backtest-use-neural", action="store_true", help="Include neural model (if weights exist) in backtest inference")
    args = parser.parse_args()
    
    try:
        logger.info("=== Starting Hybrid Lottery Analysis System ===")
        
        if args.backtest:
            from src.backtest import Backtester
            bt = Backtester(
                window=args.backtest_window,
                top_k=args.backtest_topk,
                use_neural=args.backtest_use_neural
            )
            bt.run()
            return
        
        # 1. Load Data
        logger.info("Loading data...")
        data_manager = LotteryDataManager()
        data = data_manager.load_data()
        
        # 2. Initialize Models
        logger.info("Initializing models...")
        neural_model = NeuralModel()
        stat_model = StatisticalModel()
        xgb_model = XGBoostModel()
        consensus = ConsensusEngine()
        
        # 3. Train/Load Models
        # Statistical model is fast, always train
        stat_model.train(data)
        
        # Random Forest model is fast, always train
        rf_model = RandomForestModel()
        rf_model.train(data)
        
        # XGBoost model is fast, always train
        xgb_model.train(data)
        
        # Neural model
        neural_model.build_models()
        if args.force_train or not neural_model.models_loaded:
            neural_model.train(data)
            
        # RL Fine-Tuning (PPO)
        if args.rl_train:
            neural_model.train_ppo(data)
            
        # 4. Generate Predictions
        logger.info(f"Generating {args.predictions} candidates per model...")
        
        # Generate more candidates internally for filtering
        n_candidates = max(args.predictions * 4, 20)
        
        neural_preds = neural_model.predict(data, num_predictions=n_candidates)
        stat_preds = stat_model.predict(data, num_predictions=n_candidates)
        rf_preds = rf_model.predict(data, num_predictions=n_candidates)
        xgb_preds = xgb_model.predict(data, num_predictions=n_candidates)
        
        all_preds = neural_preds + stat_preds + rf_preds + xgb_preds
        
        # 5. Consensus Ranking
        logger.info("Ranking candidates...")
        ranked_preds = consensus.rank_predictions(all_preds)
        
        # Select top N with diversity-aware chooser
        final_predictions = consensus.choose_diverse_top(ranked_preds, args.predictions)
        
        # 6. Output Results
        output_file = PREDICTIONS_DIR / "hybrid_predictions.json"
        with open(output_file, 'w') as f:
            json.dump(final_predictions, f, indent=4)
            
        print("\n=== Top Predictions ===")
        for i, pred in enumerate(final_predictions, 1):
            print(f"\nRank {i} (Score: {pred['final_score']:.4f} | Source: {pred['source']})")
            print(f"Main: {pred['main_numbers']}")
            print(f"Bonus: {pred['bonus_numbers']}")
            em = pred.get('expected_main_prob')
            eb = pred.get('expected_bonus_prob')
            if em is not None or eb is not None:
                em_str = f"{em:.3f}" if em is not None else "n/a"
                eb_str = f"{eb:.3f}" if eb is not None else "n/a"
                print(f"Prob mass (selected nums): Main {em_str} | Bonus {eb_str}")
            
        logger.info(f"Predictions saved to {output_file}")
        logger.info("=== Execution Complete ===")
        
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()

# python3 main.py --force-train --rl-train --predictions 5
# python3 main.py --backtest --backtest-window 50 --backtest-topk 3 --backtest-use-neural
