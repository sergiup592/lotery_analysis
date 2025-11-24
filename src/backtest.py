import pandas as pd
import numpy as np
import random
import logging
from typing import Dict
from .data import LotteryDataManager
from .xgboost_model import XGBoostModel
from .tree import RandomForestModel
from .statistical import StatisticalModel
from .consensus import ConsensusEngine

logger = logging.getLogger(__name__)

class Backtester:
    """Backtesting framework for Lottery Prediction System."""
    
    def __init__(self, window: int = 50, top_k: int = 1, use_neural: bool = False):
        self.window = window
        self.top_k = max(1, top_k)
        self.use_neural = use_neural
        self.results = []
        self.dm = LotteryDataManager()
        self.neural_model = None
        # Seed randomness for reproducibility of sampling during backtests
        random.seed(42)
        np.random.seed(42)
        if self.use_neural:
            try:
                # Lazy import to avoid loading TensorFlow when not requested
                from .neural import NeuralModel
                nm = NeuralModel()
                nm.build_models()
                if nm.models_loaded:
                    self.neural_model = nm
                    logger.info("Neural model loaded for backtest inference.")
                else:
                    logger.warning("Neural model not loaded; skipping neural predictions in backtest.")
            except Exception as e:
                logger.warning(f"Failed to initialize neural model for backtest: {e}")
        
    def run(self):
        """Run backtest over the specified window of recent draws."""
        logger.info(f"Starting Backtest over last {self.window} draws...")
        
        full_data = self.dm.load_data()
        total_draws = len(full_data)
        
        if total_draws < self.window + 100:
            logger.warning("Not enough data for requested backtest window. Adjusting...")
            self.window = max(1, total_draws - 100)
            
        start_idx = total_draws - self.window
        
        for i in range(start_idx, total_draws):
            # 1. Split Data
            # Training data: All draws UP TO i (exclusive)
            # Test data: Draw i
            train_data = full_data.iloc[:i].copy()
            target_draw = full_data.iloc[i]
            
            logger.info(f"Backtesting Draw {i+1}/{total_draws} (Date: {target_draw['date']})")
            
            # 2. Train Models (Fast mode: Retrain lightweight models, reuse heavy ones if possible)
            # For true accuracy we should retrain everything, but Neural is slow.
            # Let's retrain Statistical, RF, XGB. Neural we might skip or retrain every N steps.
            # For this implementation, we'll assume Neural is pre-trained or we just use it as is 
            # (risk of data leakage if not careful, but NeuralModel.train splits internally).
            # To be safe, we should ideally retrain. For speed, we'll just use the models as they are 
            # but we must ensure they don't "see" the future.
            # Actually, NeuralModel.train uses the passed dataframe. So if we pass `train_data`, it trains on that.
            
            stat_model = StatisticalModel()
            stat_model.train(train_data)
            
            rf_model = RandomForestModel()
            rf_model.train(train_data)
            
            xgb_model = XGBoostModel()
            xgb_model.train(train_data)
            
            # Neural Model - Retraining every step is too slow. 
            # Strategy: Train once at start of backtest, then maybe every 10 steps?
            # Or just use it in inference mode assuming it captures general patterns.
            # STRICT BACKTEST: Must retrain. 
            # COMPROMISE: We will skip Neural retraining for every single step to save time in this demo,
            # but in production you'd want to retrain.
            # We will initialize it but not force retrain every single time if it takes too long.
            # However, `NeuralModel` checks if models exist. We should probably force a retrain 
            # at the very beginning of the backtest on the initial `train_data`, then use it.
            # But that means it won't learn from recent history during the backtest window.
            # Let's just use the lightweight models for the "Fast" backtest to demonstrate the loop.
            # If user wants full, we can enable it.
            
            # 3. Generate Predictions
            # We need to predict for the "next" draw relative to train_data
            # The `predict` methods usually take the whole dataframe and predict for next.
            
            # Generate candidates
            n_candidates = max(40, self.top_k * 20)
            stat_preds = stat_model.predict(train_data, num_predictions=n_candidates)
            rf_preds = rf_model.predict(train_data, num_predictions=n_candidates)
            xgb_preds = xgb_model.predict(train_data, num_predictions=n_candidates)
            
            all_preds = stat_preds + rf_preds + xgb_preds
            if self.neural_model:
                try:
                    neural_preds = self.neural_model.predict(train_data, num_predictions=n_candidates)
                    all_preds += neural_preds
                except Exception as e:
                    logger.warning(f"Neural prediction failed in backtest: {e}")
            
            # Consensus
            consensus = ConsensusEngine()
            ranked_preds = consensus.rank_predictions(all_preds)
            top_preds = consensus.choose_diverse_top(ranked_preds, self.top_k) if ranked_preds else []
            
            # 4. Evaluate each selected rank
            for rank_idx, pred in enumerate(top_preds, 1):
                self._evaluate_prediction(pred, target_draw, i, rank_idx)
                
        self._generate_report()
        
    def _evaluate_prediction(self, prediction: Dict, actual_row: pd.Series, draw_idx: int, rank: int):
        """Compare prediction with actual result."""
        pred_main = set(prediction['main_numbers'])
        actual_main = set(actual_row['main_numbers'])
        
        pred_bonus = set(prediction['bonus_numbers'])
        actual_bonus = set(actual_row['bonus_numbers'])
        
        main_hits = len(pred_main & actual_main)
        bonus_hits = len(pred_bonus & actual_bonus)
        
        result = {
            'draw_idx': draw_idx,
            'date': actual_row['date'],
            'rank': rank,
            'main_hits': main_hits,
            'bonus_hits': bonus_hits,
            'pred_main': list(pred_main),
            'actual_main': list(actual_main),
            'confidence': prediction['confidence'],
            'source': prediction['source']
        }
        self.results.append(result)
        
        logger.info(f"Result: Main Hits={main_hits}, Bonus Hits={bonus_hits}")
        
    def _generate_report(self):
        """Summarize backtest results."""
        df = pd.DataFrame(self.results)
        if df.empty:
            logger.warning("No results to report.")
            return
            
        total_predictions = len(df)
        unique_draws = df['draw_idx'].nunique()
        hit_counts = df['main_hits'].value_counts().sort_index()
        
        report_lines = []
        report_lines.append("=== Backtest Report ===")
        report_lines.append(f"Draws Covered: {unique_draws}")
        report_lines.append(f"Tickets Evaluated: {total_predictions}")
        report_lines.append("Main Number Hits Distribution:")
        for hits, count in hit_counts.items():
            percentage = (count / total_predictions) * 100
            report_lines.append(f"{hits} Matches: {count} ({percentage:.1f}%)")
            
        avg_hits = df['main_hits'].mean()
        report_lines.append(f"Average Main Hits: {avg_hits:.2f}")
        
        # Rank-level diagnostics
        if 'rank' in df.columns:
            rank_hits = df.groupby('rank')['main_hits'].mean().sort_index()
            report_lines.append("Avg Main Hits by Rank:")
            for r, val in rank_hits.items():
                report_lines.append(f"  Rank {int(r)}: {val:.2f}")
        
        # Calculate "ROI" (Hypothetical)
        # Assume cost = 1 unit. 
        # Prize structure (approximate):
        # 3 matches = 10 units
        # 4 matches = 100 units
        # 5 matches = 100000 units
        cost = total_predictions
        winnings = 0
        for hits in df['main_hits']:
            if hits == 3: winnings += 10
            elif hits == 4: winnings += 100
            elif hits == 5: winnings += 100000
            
        report_lines.append("Hypothetical ROI:")
        report_lines.append(f"Cost: {cost}")
        report_lines.append(f"Winnings: {winnings}")
        report_lines.append(f"Net: {winnings - cost}")

        # Print to console and log to file for traceability
        print("\n".join(report_lines))
        logger.info(" | ".join(report_lines))
