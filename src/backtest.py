import pandas as pd
import numpy as np
import random
import logging
from typing import Dict
from .data import LotteryDataManager
from .xgboost_model import XGBoostModel
from .tree import RandomForestModel, ExtraTreesModel
from .statistical import StatisticalModel
from .consensus import ConsensusEngine
from .config import MAIN_NUMBER_RANGE, BONUS_NUMBER_RANGE, N_MAIN, N_BONUS, PREDICTIONS_DIR

logger = logging.getLogger(__name__)

class Backtester:
    """Backtesting framework for Lottery Prediction System."""
    
    def __init__(self, window: int = 50, top_k: int = 1, use_neural: bool = False):
        self.window = window
        self.top_k = max(1, top_k)
        self.use_neural = use_neural
        self.results = []
        self.baseline_results = []
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

            et_model = ExtraTreesModel()
            et_model.train(train_data)
            
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
            et_preds = et_model.predict(train_data, num_predictions=n_candidates)
            xgb_preds = xgb_model.predict(train_data, num_predictions=n_candidates)
            
            all_preds = stat_preds + rf_preds + et_preds + xgb_preds
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

            # Baseline: uniform random tickets for comparison
            for rank_idx in range(1, self.top_k + 1):
                baseline_pred = self._random_prediction()
                self._evaluate_prediction(baseline_pred, target_draw, i, rank_idx, self.baseline_results)
                
        self._generate_report()
        
    def _random_prediction(self) -> Dict:
        main_nums = np.random.choice(
            np.arange(1, MAIN_NUMBER_RANGE + 1),
            size=N_MAIN,
            replace=False
        )
        bonus_nums = np.random.choice(
            np.arange(1, BONUS_NUMBER_RANGE + 1),
            size=N_BONUS,
            replace=False
        )
        return {
            'main_numbers': sorted(main_nums.tolist()),
            'bonus_numbers': sorted(bonus_nums.tolist()),
            'confidence': 0.0,
            'source': 'RandomBaseline'
        }

    def _evaluate_prediction(self, prediction: Dict, actual_row: pd.Series, draw_idx: int, rank: int,
                             results_list: list = None):
        """Compare prediction with actual result."""
        if results_list is None:
            results_list = self.results

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
        results_list.append(result)
        
        if results_list is self.results:
            logger.info(f"Result: Main Hits={main_hits}, Bonus Hits={bonus_hits}")
        
    def _generate_report(self):
        """Summarize backtest results."""
        df = pd.DataFrame(self.results)
        if df.empty:
            logger.warning("No results to report.")
            return
        baseline_df = pd.DataFrame(self.baseline_results)
            
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

        # Baseline summary
        if not baseline_df.empty:
            baseline_avg = baseline_df['main_hits'].mean()
            report_lines.append("Random Baseline:")
            report_lines.append(f"Average Main Hits: {baseline_avg:.2f}")
            report_lines.append(f"Lift vs Baseline: {(avg_hits - baseline_avg):.2f}")

        # Print to console and log to file for traceability
        print("\n".join(report_lines))
        logger.info(" | ".join(report_lines))

        self._plot_report(df, baseline_df)

    def _plot_report(self, df: pd.DataFrame, baseline_df: pd.DataFrame):
        if df.empty:
            return
        try:
            import os
            mpl_dir = PREDICTIONS_DIR / ".mplconfig"
            mpl_dir.mkdir(exist_ok=True)
            os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as e:
            logger.warning(f"Could not import matplotlib; skipping plot: {e}")
            return

        # Rolling average of main hits by draw
        sys_series = df.groupby('draw_idx')['main_hits'].mean().sort_index()
        rolling_window = min(10, max(1, len(sys_series)))
        sys_roll = sys_series.rolling(window=rolling_window, min_periods=1).mean()

        base_series = None
        base_roll = None
        if not baseline_df.empty:
            base_series = baseline_df.groupby('draw_idx')['main_hits'].mean().sort_index()
            base_roll = base_series.rolling(window=rolling_window, min_periods=1).mean()

        fig, axes = plt.subplots(2, 1, figsize=(10, 8))

        axes[0].plot(sys_series.index, sys_roll, label='System (rolling avg)', color='tab:blue')
        if base_roll is not None:
            axes[0].plot(base_series.index, base_roll, label='Random baseline (rolling avg)', color='tab:orange')
        axes[0].set_title("Rolling Average Main Hits")
        axes[0].set_xlabel("Draw index")
        axes[0].set_ylabel("Avg main hits")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        bins = np.arange(0, N_MAIN + 2) - 0.5
        axes[1].hist(df['main_hits'], bins=bins, alpha=0.7, label='System', color='tab:blue')
        if not baseline_df.empty:
            axes[1].hist(baseline_df['main_hits'], bins=bins, alpha=0.7, label='Random baseline', color='tab:orange')
        axes[1].set_xticks(range(0, N_MAIN + 1))
        axes[1].set_title("Main Hits Distribution")
        axes[1].set_xlabel("Main hits per ticket")
        axes[1].set_ylabel("Count")
        axes[1].legend()

        fig.tight_layout()
        output_path = PREDICTIONS_DIR / "backtest_accuracy.png"
        fig.savefig(output_path, dpi=120)
        plt.close(fig)
        logger.info(f"Backtest plot saved to {output_path}")
