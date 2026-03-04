import argparse
import logging
import json
import sys
import random
import os
from pathlib import Path

# Ensure matplotlib cache is writable before importing project modules that may import matplotlib.
_project_dir = Path(__file__).resolve().parent
_mpl_cache_dir = _project_dir / "predictions" / ".mplconfig"
_mpl_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache_dir))

import numpy as np
from src.config import LOGS_DIR, PREDICTIONS_DIR
from src.coverage import CoverageOptimizer
from src.data import LotteryDataManager
from src.neural import NeuralModel
from src.statistical import StatisticalModel
from src.tree import RandomForestModel, ExtraTreesModel
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


def _passes_abstain_gate(
    pred: dict,
    min_score: float,
    min_confidence: float,
    min_expected_main_prob: float,
    min_support_count: int,
) -> bool:
    score = float(pred.get("final_score", pred.get("confidence", 0.0)) or 0.0)
    confidence = float(pred.get("confidence", 0.0) or 0.0)
    expected_main = float(pred.get("expected_main_prob", 0.0) or 0.0)
    support_count = int(pred.get("support_count", 1) or 1)
    return (
        score >= float(min_score)
        and confidence >= float(min_confidence)
        and expected_main >= float(min_expected_main_prob)
        and support_count >= int(min_support_count)
    )


def main():
    parser = argparse.ArgumentParser(description="Hybrid Lottery Number Generator")
    parser.add_argument("--predictions", type=int, default=5, help="Number of predictions to generate")
    parser.add_argument(
        "--strategy",
        choices=["hybrid", "coverage"],
        default="hybrid",
        help="Prediction strategy: model-based hybrid or coverage-optimized tickets",
    )
    parser.add_argument(
        "--model-profile",
        choices=["validated", "balanced", "full"],
        default="validated",
        help="Model set for hybrid strategy: validated (fast), balanced (adds RF), or full (all sources)",
    )
    parser.add_argument("--use-neural", action="store_true", help="Include neural model in hybrid prediction")
    parser.add_argument("--use-rf", action="store_true", help="Include Random Forest model in hybrid prediction")
    parser.add_argument("--use-et", action="store_true", help="Include Extra Trees model in hybrid prediction")
    parser.add_argument("--use-xgb", action="store_true", help="Include XGBoost model in hybrid prediction")
    parser.add_argument(
        "--coverage-candidates",
        type=int,
        default=2000,
        help="Random candidates per ticket for coverage strategy",
    )
    parser.add_argument("--force-train", action="store_true", help="Force retraining of neural models")
    parser.add_argument("--rl-train", action="store_true", help="Fine-tune models using PPO (RL)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible sampling")
    parser.add_argument("--backtest", action="store_true", help="Run backtesting framework")
    parser.add_argument("--backtest-compare", action="store_true", help="Run controlled baseline vs improved backtest comparison")
    parser.add_argument("--backtest-window", type=int, default=50, help="Number of past draws to backtest")
    parser.add_argument("--backtest-topk", type=int, default=2, help="Number of top-ranked predictions to evaluate per draw during backtest")
    parser.add_argument("--backtest-use-neural", action="store_true", help="Include neural model (if weights exist) in backtest inference")
    parser.add_argument(
        "--backtest-neural-mode",
        choices=["frozen", "rolling"],
        default="frozen",
        help="Neural backtest mode: frozen pre-trained model or rolling leak-safe retrain",
    )
    parser.add_argument(
        "--backtest-neural-retrain-interval",
        type=int,
        default=8,
        help="Draw interval between rolling neural retrains in backtest mode",
    )
    parser.add_argument(
        "--backtest-neural-train-window",
        type=int,
        default=400,
        help="Number of recent draws used for each rolling neural retrain",
    )
    parser.add_argument(
        "--backtest-neural-epochs",
        type=int,
        default=10,
        help="Epochs per rolling neural retrain",
    )
    parser.add_argument(
        "--backtest-neural-batch-size",
        type=int,
        default=32,
        help="Batch size for rolling neural retrain",
    )
    parser.add_argument(
        "--backtest-neural-min-retrains",
        type=int,
        default=2,
        help="Minimum successful rolling neural retrains before using neural predictions",
    )
    parser.add_argument(
        "--backtest-no-ensemble",
        action="store_true",
        help="Disable ensemble candidate generation in backtest",
    )
    parser.add_argument(
        "--backtest-no-filter-fit",
        action="store_true",
        help="Disable adaptive filter fitting in backtest",
    )
    parser.add_argument("--backtest-skip-statistical", action="store_true", help="Skip statistical model during backtest")
    parser.add_argument("--backtest-skip-rf", action="store_true", help="Skip random forest model during backtest")
    parser.add_argument("--backtest-skip-et", action="store_true", help="Skip extra trees model during backtest")
    parser.add_argument("--backtest-skip-xgb", action="store_true", help="Skip XGBoost model during backtest")
    parser.add_argument(
        "--backtest-fast-models",
        action="store_true",
        help="Use reduced-complexity RF/ET/XGBoost models for faster backtests",
    )
    parser.add_argument(
        "--backtest-fast-level",
        choices=["fast", "ultrafast"],
        default="fast",
        help="Speed preset used when --backtest-fast-models is enabled",
    )
    parser.add_argument(
        "--backtest-model-retrain-interval",
        type=int,
        default=1,
        help="Retrain classical models every N backtest draws (1 = retrain every draw)",
    )
    parser.add_argument(
        "--backtest-no-source-performance",
        action="store_true",
        help="Disable adaptive source weighting from rolling backtest performance",
    )
    parser.add_argument(
        "--backtest-source-performance-window",
        type=int,
        default=20,
        help="Rolling window of prior draws used for source performance weighting",
    )
    parser.add_argument(
        "--backtest-source-performance-min-samples",
        type=int,
        default=4,
        help="Minimum prior samples before source performance weighting is applied",
    )
    parser.add_argument(
        "--backtest-enable-abstain",
        action="store_true",
        help="Enable abstain mode in backtest when candidates do not meet quality thresholds",
    )
    parser.add_argument(
        "--backtest-abstain-min-score",
        type=float,
        default=0.10,
        help="Minimum final score required to place a backtest ticket in abstain mode",
    )
    parser.add_argument(
        "--backtest-abstain-min-confidence",
        type=float,
        default=0.0,
        help="Minimum confidence required to place a backtest ticket in abstain mode",
    )
    parser.add_argument(
        "--backtest-abstain-min-expected-main-prob",
        type=float,
        default=0.0,
        help="Minimum expected main probability mass required in backtest abstain mode",
    )
    parser.add_argument(
        "--backtest-abstain-min-support-count",
        type=int,
        default=1,
        help="Minimum cross-source support count required in backtest abstain mode",
    )
    parser.add_argument(
        "--backtest-auto-tune-consensus",
        action="store_true",
        help="Enable rolling prequential consensus profile auto-tuning in backtest",
    )
    parser.add_argument(
        "--backtest-auto-tune-interval",
        type=int,
        default=3,
        help="Draw interval between consensus auto-tune profile updates in backtest",
    )
    parser.add_argument(
        "--backtest-auto-tune-window",
        type=int,
        default=24,
        help="Rolling reward window used by backtest consensus auto-tuning",
    )
    parser.add_argument(
        "--backtest-auto-tune-exploration",
        type=float,
        default=0.15,
        help="Exploration strength for UCB-style backtest consensus auto-tuning",
    )
    parser.add_argument(
        "--no-readiness-gate",
        action="store_true",
        help="Disable readiness gate before hybrid prediction generation",
    )
    parser.add_argument(
        "--readiness-allow-unready",
        action="store_true",
        help="Allow hybrid prediction generation even when readiness gate fails",
    )
    parser.add_argument(
        "--readiness-window",
        type=int,
        default=20,
        help="Backtest window used by readiness gate",
    )
    parser.add_argument(
        "--readiness-topk",
        type=int,
        default=2,
        help="Top-k predictions per draw used by readiness gate",
    )
    parser.add_argument(
        "--readiness-fast-level",
        choices=["default", "fast", "ultrafast"],
        default="ultrafast",
        help="Model speed profile used by readiness gate",
    )
    parser.add_argument(
        "--readiness-model-retrain-interval",
        type=int,
        default=3,
        help="Model retrain interval used by readiness gate",
    )
    parser.add_argument(
        "--readiness-min-draws",
        type=int,
        default=20,
        help="Minimum evaluated draws required for readiness pass",
    )
    parser.add_argument(
        "--readiness-min-avg-lift",
        type=float,
        default=0.03,
        help="Minimum average lift vs baseline required to pass readiness gate",
    )
    parser.add_argument(
        "--readiness-min-rolling-lift",
        type=float,
        default=0.02,
        help="Minimum latest rolling lift required to pass readiness gate",
    )
    parser.add_argument(
        "--readiness-rolling-window",
        type=int,
        default=8,
        help="Rolling window size for readiness lift checks",
    )
    parser.add_argument(
        "--readiness-confidence",
        type=float,
        default=0.90,
        help="Two-sided confidence level for readiness lift interval",
    )
    parser.add_argument(
        "--readiness-min-ci-lower",
        type=float,
        default=0.0,
        help="Minimum confidence-interval lower bound for readiness lift",
    )
    parser.add_argument(
        "--readiness-report-tag",
        type=str,
        default="readiness",
        help="Report tag used for readiness artifacts",
    )
    parser.add_argument(
        "--readiness-min-tickets",
        type=int,
        default=10,
        help="Minimum evaluated tickets required for readiness pass",
    )
    parser.add_argument(
        "--readiness-min-roi-per-ticket",
        type=float,
        default=0.0,
        help="Minimum projected ROI per ticket required to pass readiness",
    )
    parser.add_argument(
        "--readiness-min-roi-ci-lower",
        type=float,
        default=0.0,
        help="Minimum lower confidence bound for ROI per ticket in readiness",
    )
    parser.add_argument(
        "--readiness-profit-confidence",
        type=float,
        default=0.90,
        help="Two-sided confidence level for ROI per-ticket readiness interval",
    )
    parser.add_argument(
        "--readiness-no-abstain",
        action="store_true",
        help="Disable abstain mode during readiness gate backtesting",
    )
    parser.add_argument(
        "--readiness-abstain-min-score",
        type=float,
        default=0.12,
        help="Minimum final score required to place a readiness ticket in abstain mode",
    )
    parser.add_argument(
        "--readiness-abstain-min-confidence",
        type=float,
        default=0.0,
        help="Minimum confidence required to place a readiness ticket in abstain mode",
    )
    parser.add_argument(
        "--readiness-abstain-min-expected-main-prob",
        type=float,
        default=0.35,
        help="Minimum expected main probability mass required in readiness abstain mode",
    )
    parser.add_argument(
        "--readiness-abstain-min-support-count",
        type=int,
        default=1,
        help="Minimum support count required in readiness abstain mode",
    )
    parser.add_argument(
        "--readiness-no-auto-tune-consensus",
        action="store_true",
        help="Disable rolling consensus auto-tuning during readiness backtest",
    )
    parser.add_argument(
        "--readiness-auto-tune-interval",
        type=int,
        default=3,
        help="Draw interval between readiness consensus auto-tune profile updates",
    )
    parser.add_argument(
        "--readiness-auto-tune-window",
        type=int,
        default=24,
        help="Rolling reward window used by readiness consensus auto-tuning",
    )
    parser.add_argument(
        "--readiness-auto-tune-exploration",
        type=float,
        default=0.15,
        help="Exploration strength for readiness consensus auto-tuning",
    )
    parser.add_argument(
        "--enable-abstain",
        action="store_true",
        help="Enable abstain mode for live hybrid generation",
    )
    parser.add_argument(
        "--abstain-min-score",
        type=float,
        default=0.12,
        help="Minimum final score required to output a live ticket in abstain mode",
    )
    parser.add_argument(
        "--abstain-min-confidence",
        type=float,
        default=0.0,
        help="Minimum confidence required to output a live ticket in abstain mode",
    )
    parser.add_argument(
        "--abstain-min-expected-main-prob",
        type=float,
        default=0.35,
        help="Minimum expected main probability mass required in live abstain mode",
    )
    parser.add_argument(
        "--abstain-min-support-count",
        type=int,
        default=1,
        help="Minimum support count required in live abstain mode",
    )
    args = parser.parse_args()
    
    try:
        logger.info("=== Starting Hybrid Lottery Analysis System ===")
        random.seed(args.seed)
        np.random.seed(args.seed)
        
        if args.backtest:
            from src.backtest import Backtester, run_controlled_comparison
            model_speed_profile = args.backtest_fast_level if args.backtest_fast_models else "default"
            if args.backtest_compare:
                run_controlled_comparison(
                    window=args.backtest_window,
                    top_k=args.backtest_topk,
                    seed=args.seed,
                    use_neural=args.backtest_use_neural,
                    neural_retrain_interval=args.backtest_neural_retrain_interval,
                    neural_train_window=args.backtest_neural_train_window,
                    neural_train_epochs=args.backtest_neural_epochs,
                    neural_batch_size=args.backtest_neural_batch_size,
                    neural_min_retrains_for_inference=args.backtest_neural_min_retrains,
                    include_statistical=not args.backtest_skip_statistical,
                    include_random_forest=not args.backtest_skip_rf,
                    include_extra_trees=not args.backtest_skip_et,
                    include_xgboost=not args.backtest_skip_xgb,
                    model_speed_profile=model_speed_profile,
                    model_retrain_interval=args.backtest_model_retrain_interval,
                    use_source_performance_weights=not args.backtest_no_source_performance,
                    source_performance_window=args.backtest_source_performance_window,
                    source_performance_min_samples=args.backtest_source_performance_min_samples,
                    enable_abstain=args.backtest_enable_abstain,
                    abstain_min_score=args.backtest_abstain_min_score,
                    abstain_min_confidence=args.backtest_abstain_min_confidence,
                    abstain_min_expected_main_prob=args.backtest_abstain_min_expected_main_prob,
                    abstain_min_support_count=args.backtest_abstain_min_support_count,
                    auto_tune_consensus=args.backtest_auto_tune_consensus,
                    auto_tune_interval=args.backtest_auto_tune_interval,
                    auto_tune_window=args.backtest_auto_tune_window,
                    auto_tune_exploration=args.backtest_auto_tune_exploration,
                )
            else:
                bt = Backtester(
                    window=args.backtest_window,
                    top_k=args.backtest_topk,
                    use_neural=args.backtest_use_neural,
                    seed=args.seed,
                    use_ensemble=not args.backtest_no_ensemble,
                    fit_filter_thresholds=not args.backtest_no_filter_fit,
                    neural_mode=args.backtest_neural_mode,
                    neural_retrain_interval=args.backtest_neural_retrain_interval,
                    neural_train_window=args.backtest_neural_train_window,
                    neural_train_epochs=args.backtest_neural_epochs,
                    neural_batch_size=args.backtest_neural_batch_size,
                    neural_min_retrains_for_inference=args.backtest_neural_min_retrains,
                    include_statistical=not args.backtest_skip_statistical,
                    include_random_forest=not args.backtest_skip_rf,
                    include_extra_trees=not args.backtest_skip_et,
                    include_xgboost=not args.backtest_skip_xgb,
                    model_speed_profile=model_speed_profile,
                    model_retrain_interval=args.backtest_model_retrain_interval,
                    use_source_performance_weights=not args.backtest_no_source_performance,
                    source_performance_window=args.backtest_source_performance_window,
                    source_performance_min_samples=args.backtest_source_performance_min_samples,
                    enable_abstain=args.backtest_enable_abstain,
                    abstain_min_score=args.backtest_abstain_min_score,
                    abstain_min_confidence=args.backtest_abstain_min_confidence,
                    abstain_min_expected_main_prob=args.backtest_abstain_min_expected_main_prob,
                    abstain_min_support_count=args.backtest_abstain_min_support_count,
                    auto_tune_consensus=args.backtest_auto_tune_consensus,
                    auto_tune_interval=args.backtest_auto_tune_interval,
                    auto_tune_window=args.backtest_auto_tune_window,
                    auto_tune_exploration=args.backtest_auto_tune_exploration,
                )
                bt.run()
            return

        if args.strategy == "coverage":
            optimizer = CoverageOptimizer()
            coverage_preds = optimizer.generate(
                args.predictions,
                candidates_per_ticket=args.coverage_candidates,
            )
            output_file = PREDICTIONS_DIR / "coverage_predictions.json"
            with open(output_file, "w") as f:
                json.dump(coverage_preds, f, indent=4)

            print("\n=== Coverage-Optimized Tickets ===")
            for i, pred in enumerate(coverage_preds, 1):
                score = pred.get("coverage_score", 0.0)
                print(f"\nPick {i} (Coverage Score: {score:.2f})")
                print(f"Main: {pred['main_numbers']}")
                print(f"Bonus: {pred['bonus_numbers']}")

            logger.info(f"Coverage picks saved to {output_file}")
            logger.info("=== Execution Complete ===")
            return
        
        # 1. Load Data
        logger.info("Loading data...")
        data_manager = LotteryDataManager()
        data = data_manager.load_data()

        profile_defaults = {
            "validated": {"stat": True, "neural": False, "rf": False, "et": False, "xgb": False},
            "balanced": {"stat": True, "neural": False, "rf": True, "et": False, "xgb": False},
            "full": {"stat": True, "neural": True, "rf": True, "et": True, "xgb": True},
        }
        selected_models = profile_defaults[args.model_profile].copy()

        # Force-train and RL fine-tuning both imply neural model usage.
        if args.force_train or args.rl_train:
            selected_models["neural"] = True

        # Explicit model flags allow opting in extra sources when using validated profile.
        if args.use_neural:
            selected_models["neural"] = True
        if args.use_rf:
            selected_models["rf"] = True
        if args.use_et:
            selected_models["et"] = True
        if args.use_xgb:
            selected_models["xgb"] = True

        logger.info("Hybrid model profile=%s selected_models=%s", args.model_profile, selected_models)
        recommended_consensus_profile = None

        if not args.no_readiness_gate:
            from src.backtest import evaluate_readiness_gate

            logger.info(
                "Running readiness gate (window=%d, top_k=%d, profile=%s)...",
                args.readiness_window,
                args.readiness_topk,
                args.readiness_fast_level,
            )
            readiness = evaluate_readiness_gate(
                window=args.readiness_window,
                top_k=args.readiness_topk,
                seed=args.seed,
                use_neural=selected_models["neural"],
                neural_mode="frozen",
                include_statistical=selected_models["stat"],
                include_random_forest=selected_models["rf"],
                include_extra_trees=selected_models["et"],
                include_xgboost=selected_models["xgb"],
                model_speed_profile=args.readiness_fast_level,
                model_retrain_interval=args.readiness_model_retrain_interval,
                min_avg_lift=args.readiness_min_avg_lift,
                min_rolling_lift=args.readiness_min_rolling_lift,
                rolling_window=args.readiness_rolling_window,
                confidence=args.readiness_confidence,
                min_ci_lower=args.readiness_min_ci_lower,
                min_draws=args.readiness_min_draws,
                min_tickets=args.readiness_min_tickets,
                min_roi_per_ticket=args.readiness_min_roi_per_ticket,
                min_roi_ci_lower=args.readiness_min_roi_ci_lower,
                profit_confidence=args.readiness_profit_confidence,
                enable_abstain=not args.readiness_no_abstain,
                abstain_min_score=args.readiness_abstain_min_score,
                abstain_min_confidence=args.readiness_abstain_min_confidence,
                abstain_min_expected_main_prob=args.readiness_abstain_min_expected_main_prob,
                abstain_min_support_count=args.readiness_abstain_min_support_count,
                auto_tune_consensus=not args.readiness_no_auto_tune_consensus,
                auto_tune_interval=args.readiness_auto_tune_interval,
                auto_tune_window=args.readiness_auto_tune_window,
                auto_tune_exploration=args.readiness_auto_tune_exploration,
                report_tag=args.readiness_report_tag,
            )
            passed = bool(readiness.get("passed"))
            metrics = readiness.get("metrics", {})
            checks = readiness.get("checks", {})
            recommended_consensus_profile = (
                readiness.get("summary", {}).get("consensus_profile_last")
            )
            logger.info(
                "Readiness gate verdict=%s checks=%s metrics={avg_lift=%.4f, rolling_lift=%.4f, ci_lower=%.4f, draws=%s}",
                "PASS" if passed else "FAIL",
                checks,
                float(metrics.get("avg_lift", 0.0)),
                float(metrics.get("rolling_lift", 0.0)),
                float(metrics.get("ci_lower", 0.0)),
                metrics.get("draws_evaluated", 0),
            )
            if not passed and not args.readiness_allow_unready:
                print("\n=== Readiness Gate: BLOCKED ===")
                print("Hybrid prediction generation stopped because readiness criteria were not met.")
                print(f"See report: {PREDICTIONS_DIR / f'{args.readiness_report_tag}_report.json'}")
                print("Use --readiness-allow-unready to override intentionally.")
                sys.exit(2)
            if not passed and args.readiness_allow_unready:
                logger.warning("Continuing despite readiness failure due to --readiness-allow-unready.")
        
        # 2. Initialize Models
        logger.info("Initializing models...")
        neural_model = NeuralModel() if selected_models["neural"] else None
        stat_model = StatisticalModel() if selected_models["stat"] else None
        rf_model = RandomForestModel() if selected_models["rf"] else None
        et_model = ExtraTreesModel() if selected_models["et"] else None
        xgb_model = XGBoostModel() if selected_models["xgb"] else None
        consensus = ConsensusEngine()
        consensus.fit_filters(data)
        if recommended_consensus_profile:
            profile_map = ConsensusEngine.tuning_profiles()
            selected_profile = profile_map.get(str(recommended_consensus_profile))
            if selected_profile:
                consensus.set_tuning_profile(selected_profile)
                logger.info(
                    "Applied readiness-selected consensus profile '%s' for live generation.",
                    recommended_consensus_profile,
                )
        
        # 3. Train/Load Models
        if stat_model is not None:
            stat_model.train(data)

        if rf_model is not None:
            rf_model.train(data)

        if et_model is not None:
            et_model.train(data)

        if xgb_model is not None:
            xgb_model.train(data)

        if neural_model is not None:
            neural_model.build_models()
            if args.force_train:
                neural_model.train(data)
            elif not neural_model.models_loaded:
                logger.warning(
                    "Neural weights not found; skipping neural source. Use --force-train to train from scratch."
                )
                neural_model = None
            
        # RL Fine-Tuning (PPO)
        if args.rl_train:
            if neural_model is None:
                logger.warning("Skipping PPO fine-tuning: neural model is unavailable.")
            else:
                neural_model.train_ppo(data)
            
        # 4. Generate Predictions
        logger.info(f"Generating {args.predictions} candidates per model...")
        
        # Generate more candidates internally for filtering
        n_candidates = max(args.predictions * 4, 20)
        
        all_preds = []
        if neural_model is not None:
            all_preds += neural_model.predict(data, num_predictions=n_candidates)
        if stat_model is not None:
            all_preds += stat_model.predict(data, num_predictions=n_candidates)
        if rf_model is not None:
            all_preds += rf_model.predict(data, num_predictions=n_candidates)
        if et_model is not None:
            all_preds += et_model.predict(data, num_predictions=n_candidates)
        if xgb_model is not None:
            all_preds += xgb_model.predict(data, num_predictions=n_candidates)

        if not all_preds:
            raise ValueError("No predictions were generated. Enable at least one model.")

        # 5. Generate Ensemble Predictions (combines probability distributions).
        # Skip when only one probabilistic source is present; blending a single source
        # with itself adds noise but no new information.
        prob_sources = {
            p.get("source", "Unknown")
            for p in all_preds
            if p.get("main_prob_vector") is not None and p.get("bonus_prob_vector") is not None
        }
        ensemble_preds = []
        if len(prob_sources) >= 2:
            logger.info("Generating ensemble predictions...")
            ensemble_preds = consensus.ensemble_predictions(all_preds, num_outputs=args.predictions)
        else:
            logger.info("Skipping ensemble generation: need >=2 probabilistic sources, got %d", len(prob_sources))

        # 6. Consensus Ranking
        logger.info("Ranking candidates...")
        ranked_preds = consensus.rank_predictions(all_preds + ensemble_preds)

        # Select top N with diversity-aware chooser
        final_predictions = consensus.choose_diverse_top(ranked_preds, args.predictions)

        if args.enable_abstain:
            before_count = len(final_predictions)
            final_predictions = [
                pred
                for pred in final_predictions
                if _passes_abstain_gate(
                    pred,
                    min_score=args.abstain_min_score,
                    min_confidence=args.abstain_min_confidence,
                    min_expected_main_prob=args.abstain_min_expected_main_prob,
                    min_support_count=args.abstain_min_support_count,
                )
            ]
            logger.info(
                "Live abstain filter retained %d/%d predictions.",
                len(final_predictions),
                before_count,
            )
            if not final_predictions:
                print("\n=== Abstain Signal ===")
                print("No tickets generated: all candidates failed abstain quality thresholds.")

        # 7. Output Results
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
# python3 main.py --backtest --backtest-window 50 --backtest-topk 2 --backtest-use-neural
