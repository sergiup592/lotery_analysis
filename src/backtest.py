import json
import logging
import random
from pathlib import Path
from statistics import NormalDist
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .config import BONUS_NUMBER_RANGE, MAIN_NUMBER_RANGE, N_BONUS, N_MAIN, PREDICTIONS_DIR
from .consensus import ConsensusEngine
from .data import LotteryDataManager
from .statistical import StatisticalModel
from .tree import ExtraTreesModel, RandomForestModel
from .xgboost_model import XGBoostModel

logger = logging.getLogger(__name__)


class Backtester:
    """Backtesting framework for Lottery Prediction System."""

    def __init__(
        self,
        window: int = 50,
        top_k: int = 1,
        use_neural: bool = False,
        seed: int = 42,
        use_ensemble: bool = True,
        fit_filter_thresholds: bool = True,
        neural_mode: str = "frozen",
        neural_retrain_interval: int = 10,
        neural_train_window: int = 400,
        neural_train_epochs: int = 12,
        neural_batch_size: int = 32,
        neural_min_retrains_for_inference: int = 2,
        include_statistical: bool = True,
        include_random_forest: bool = True,
        include_extra_trees: bool = True,
        include_xgboost: bool = True,
        model_speed_profile: str = "default",
        model_retrain_interval: int = 1,
        use_source_performance_weights: bool = False,
        source_performance_window: int = 20,
        source_performance_min_samples: int = 4,
        enable_abstain: bool = False,
        abstain_min_score: float = 0.10,
        abstain_min_confidence: float = 0.0,
        abstain_min_expected_main_prob: float = 0.0,
        abstain_min_support_count: int = 1,
        auto_tune_consensus: bool = False,
        auto_tune_interval: int = 3,
        auto_tune_window: int = 24,
        auto_tune_exploration: float = 0.15,
        report_tag: Optional[str] = None,
    ):
        self.window = max(1, int(window))
        self.top_k = max(1, int(top_k))
        self.use_neural = bool(use_neural)
        self.seed = int(seed)
        self.use_ensemble = bool(use_ensemble)
        self.fit_filter_thresholds = bool(fit_filter_thresholds)
        self.neural_mode = neural_mode
        self.neural_retrain_interval = max(1, int(neural_retrain_interval))
        self.neural_train_window = max(0, int(neural_train_window))
        self.neural_train_epochs = max(1, int(neural_train_epochs))
        self.neural_batch_size = max(1, int(neural_batch_size))
        self.neural_min_retrains_for_inference = max(1, int(neural_min_retrains_for_inference))
        self.include_statistical = bool(include_statistical)
        self.include_random_forest = bool(include_random_forest)
        self.include_extra_trees = bool(include_extra_trees)
        self.include_xgboost = bool(include_xgboost)
        self.model_speed_profile = model_speed_profile
        self.model_retrain_interval = max(1, int(model_retrain_interval))
        self.use_source_performance_weights = bool(use_source_performance_weights)
        self.source_performance_window = max(1, int(source_performance_window))
        self.source_performance_min_samples = max(1, int(source_performance_min_samples))
        self.enable_abstain = bool(enable_abstain)
        self.abstain_min_score = float(max(0.0, abstain_min_score))
        self.abstain_min_confidence = float(max(0.0, abstain_min_confidence))
        self.abstain_min_expected_main_prob = float(max(0.0, abstain_min_expected_main_prob))
        self.abstain_min_support_count = max(1, int(abstain_min_support_count))
        self.auto_tune_consensus = bool(auto_tune_consensus)
        self.auto_tune_interval = max(1, int(auto_tune_interval))
        self.auto_tune_window = max(1, int(auto_tune_window))
        self.auto_tune_exploration = float(max(0.0, auto_tune_exploration))
        self.report_tag = report_tag

        if self.neural_mode not in {"frozen", "rolling"}:
            raise ValueError("neural_mode must be one of: frozen, rolling")
        if self.model_speed_profile not in {"default", "fast", "ultrafast"}:
            raise ValueError("model_speed_profile must be one of: default, fast, ultrafast")

        self.results = []
        self.baseline_results = []
        self.dm = LotteryDataManager()
        self.neural_model = None
        self._rolling_retrain_count = 0
        self._source_hits_by_model: Dict[str, list] = {}
        self._baseline_rng = np.random.default_rng(self.seed + 17303)
        self._draws_evaluated = 0
        self._abstained_draws = 0
        self._tickets_target = 0
        self._active_consensus_profile = "balanced"
        self._consensus_profiles = ConsensusEngine.tuning_profiles()
        self._consensus_profile_rewards: Dict[str, list] = {
            name: [] for name in self._consensus_profiles
        }
        self._random_main_hit_baseline = (N_MAIN * N_MAIN) / float(MAIN_NUMBER_RANGE)
        self._seed_everything()

        # Frozen mode can preload existing weights for fast inference.
        if self.use_neural and self.neural_mode == "frozen":
            self._initialize_frozen_neural()

    def _seed_everything(self) -> None:
        random.seed(self.seed)
        np.random.seed(self.seed)
        try:
            import tensorflow as tf

            tf.random.set_seed(self.seed)
        except Exception:
            # TensorFlow is optional in non-neural runs.
            pass

    def _initialize_frozen_neural(self) -> None:
        try:
            from .neural import NeuralModel

            nm = NeuralModel()
            nm.build_models(load_existing=True)
            if nm.models_loaded:
                self.neural_model = nm
                logger.info("Neural model loaded for frozen backtest inference.")
            else:
                logger.warning("Neural model weights not found; neural predictions disabled in frozen mode.")
        except Exception as exc:
            logger.warning("Failed to initialize frozen neural model: %s", exc)

    def _maybe_retrain_rolling_neural(self, train_data: pd.DataFrame, step_idx: int) -> None:
        if not self.use_neural or self.neural_mode != "rolling":
            return

        if step_idx != 0 and (step_idx % self.neural_retrain_interval) != 0:
            return

        try:
            from .neural import NeuralModel
        except Exception as exc:
            logger.warning("TensorFlow/Neural imports unavailable; skipping rolling neural retrain: %s", exc)
            return

        if self.neural_model is None:
            self.neural_model = NeuralModel()
            # Critical for leak-safe backtests: do not load persisted models.
            self.neural_model.build_models(load_existing=False)

        seq_len = int(self.neural_model.params.get("sequence_length", 20))
        min_required = max(seq_len + 5, 30)
        if len(train_data) < min_required:
            logger.info(
                "Skipping rolling neural retrain (need >= %d draws, got %d).",
                min_required,
                len(train_data),
            )
            return

        train_slice = train_data.tail(self.neural_train_window).copy() if self.neural_train_window > 0 else train_data.copy()
        # Tail slices preserve original indices; reset to keep feature builders index-safe.
        train_slice.reset_index(drop=True, inplace=True)

        old_epochs = self.neural_model.params.get("epochs")
        old_batch = self.neural_model.params.get("batch_size")
        self.neural_model.params["epochs"] = self.neural_train_epochs
        self.neural_model.params["batch_size"] = self.neural_batch_size

        logger.info(
            "Rolling neural retrain at step=%d on %d draws (epochs=%d, batch=%d).",
            step_idx,
            len(train_slice),
            self.neural_train_epochs,
            self.neural_batch_size,
        )
        try:
            self.neural_model.train(train_slice, save_models=False)
            self._rolling_retrain_count += 1
            logger.info("Rolling neural retrain count: %d", self._rolling_retrain_count)
        except Exception as exc:
            logger.warning("Rolling neural retrain failed: %s", exc)
        finally:
            if old_epochs is not None:
                self.neural_model.params["epochs"] = old_epochs
            if old_batch is not None:
                self.neural_model.params["batch_size"] = old_batch

    def _passes_abstain_gate(self, pred: Dict[str, Any]) -> bool:
        """Return True if a candidate is strong enough to be played."""
        score = float(pred.get("final_score", pred.get("confidence", 0.0)) or 0.0)
        confidence = float(pred.get("confidence", 0.0) or 0.0)
        expected_main = float(pred.get("expected_main_prob", 0.0) or 0.0)
        support_count = int(pred.get("support_count", 1) or 1)
        return (
            score >= self.abstain_min_score
            and confidence >= self.abstain_min_confidence
            and expected_main >= self.abstain_min_expected_main_prob
            and support_count >= self.abstain_min_support_count
        )

    def _select_consensus_profile(self, step_idx: int) -> str:
        """
        Select a tuning profile using prequential online selection:
        initial exploration, then UCB over rolling rewards.
        """
        profile_names = list(self._consensus_profiles.keys())
        if not profile_names:
            self._active_consensus_profile = "balanced"
            return self._active_consensus_profile

        # Keep current profile between retune intervals.
        if step_idx > 0 and (step_idx % self.auto_tune_interval) != 0:
            return self._active_consensus_profile

        # Warm-up: force one initial sample per profile.
        for name in profile_names:
            if not self._consensus_profile_rewards.get(name):
                self._active_consensus_profile = name
                return name

        total_obs = sum(len(v) for v in self._consensus_profile_rewards.values())
        if total_obs <= 0:
            self._active_consensus_profile = profile_names[0]
            return self._active_consensus_profile

        best_name = self._active_consensus_profile
        best_score = -float("inf")
        for name in profile_names:
            history = self._consensus_profile_rewards.get(name, [])
            recent = history[-self.auto_tune_window :]
            n = len(recent)
            if n <= 0:
                score = float("inf")
            else:
                mean_reward = float(np.mean(recent))
                explore = self.auto_tune_exploration * float(np.sqrt(np.log(total_obs + 1.0) / n))
                score = mean_reward + explore
            if score > best_score:
                best_score = score
                best_name = name

        self._active_consensus_profile = best_name
        return best_name

    def _record_consensus_profile_reward(self, profile_name: str, reward: float) -> None:
        history = self._consensus_profile_rewards.setdefault(profile_name, [])
        history.append(float(reward))
        max_keep = max(self.auto_tune_window * 6, 24)
        if len(history) > max_keep:
            del history[: len(history) - max_keep]

    def _compute_draw_reward(self, predictions: list, actual_row: pd.Series) -> float:
        """
        Online reward used by consensus profile auto-tuning.
        Positive reward means better-than-random overlap per played ticket.
        """
        if not predictions:
            # Small abstain penalty avoids degenerate "always abstain" tuning.
            return -0.05
        actual_main = set(actual_row["main_numbers"])
        actual_bonus = set(actual_row["bonus_numbers"])
        main_hits = []
        bonus_hits = []
        for pred in predictions:
            main_hits.append(len(set(pred.get("main_numbers", [])) & actual_main))
            bonus_hits.append(len(set(pred.get("bonus_numbers", [])) & actual_bonus))
        avg_main = float(np.mean(main_hits)) if main_hits else 0.0
        avg_bonus = float(np.mean(bonus_hits)) if bonus_hits else 0.0
        edge = avg_main - self._random_main_hit_baseline
        return float(edge + (0.08 * avg_bonus))

    def run(self) -> Dict[str, Any]:
        """Run backtest over the specified window of recent draws and return summary metrics."""
        self.results = []
        self.baseline_results = []
        self._rolling_retrain_count = 0
        self._source_hits_by_model = {}
        self._baseline_rng = np.random.default_rng(self.seed + 17303)
        self._draws_evaluated = 0
        self._abstained_draws = 0
        self._tickets_target = 0
        self._active_consensus_profile = "balanced"
        self._consensus_profile_rewards = {name: [] for name in self._consensus_profiles}
        self._seed_everything()

        logger.info(
            "Starting Backtest: window=%d, top_k=%d, neural=%s(%s), ensemble=%s, fit_filters=%s, source_perf=%s, abstain=%s, auto_tune=%s, retrain_interval=%d, models=[stat=%s,rf=%s,et=%s,xgb=%s]",
            self.window,
            self.top_k,
            self.use_neural,
            self.neural_mode,
            self.use_ensemble,
            self.fit_filter_thresholds,
            self.use_source_performance_weights,
            self.enable_abstain,
            self.auto_tune_consensus,
            self.model_retrain_interval,
            self.include_statistical,
            self.include_random_forest,
            self.include_extra_trees,
            self.include_xgboost,
        )
        logger.info("Backtest model speed profile: %s", self.model_speed_profile)

        full_data = self.dm.load_data()
        total_draws = len(full_data)

        if total_draws < self.window + 100:
            logger.warning("Not enough data for requested backtest window. Adjusting...")
            self.window = max(1, total_draws - 100)

        start_idx = total_draws - self.window
        stat_model = None
        rf_model = None
        et_model = None
        xgb_model = None

        for i in range(start_idx, total_draws):
            train_data = full_data.iloc[:i].copy()
            target_draw = full_data.iloc[i]
            step_idx = i - start_idx
            self._draws_evaluated += 1
            self._tickets_target += self.top_k

            logger.info("Backtesting draw %d/%d (date=%s)", i + 1, total_draws, target_draw["date"])

            retrain_models_now = (
                step_idx == 0
                or (step_idx % self.model_retrain_interval) == 0
                or (
                    (self.include_statistical and stat_model is None)
                    or (self.include_random_forest and rf_model is None)
                    or (self.include_extra_trees and et_model is None)
                    or (self.include_xgboost and xgb_model is None)
                )
            )

            if retrain_models_now:
                if self.include_statistical:
                    stat_model = StatisticalModel()
                    stat_model.train(train_data)

                if self.include_random_forest:
                    rf_model = RandomForestModel(training_profile=self.model_speed_profile)
                    rf_model.train(train_data)

                if self.include_extra_trees:
                    et_model = ExtraTreesModel(training_profile=self.model_speed_profile)
                    et_model.train(train_data)

                if self.include_xgboost:
                    xgb_model = XGBoostModel(training_profile=self.model_speed_profile)
                    xgb_model.train(train_data)
            else:
                logger.debug(
                    "Reusing trained models at backtest step=%d (interval=%d).",
                    step_idx,
                    self.model_retrain_interval,
                )

            self._maybe_retrain_rolling_neural(train_data, step_idx)

            n_candidates = max(40, self.top_k * 20)
            all_preds = []
            if stat_model is not None:
                all_preds += stat_model.predict(train_data, num_predictions=n_candidates)
            if rf_model is not None:
                all_preds += rf_model.predict(train_data, num_predictions=n_candidates)
            if et_model is not None:
                all_preds += et_model.predict(train_data, num_predictions=n_candidates)
            if xgb_model is not None:
                all_preds += xgb_model.predict(train_data, num_predictions=n_candidates)

            neural_ready = self.use_neural and self.neural_model is not None and self.neural_model.models_loaded
            if self.neural_mode == "rolling":
                neural_ready = neural_ready and (
                    self._rolling_retrain_count >= self.neural_min_retrains_for_inference
                )

            if neural_ready:
                try:
                    neural_preds = self.neural_model.predict(train_data, num_predictions=n_candidates)
                    all_preds += neural_preds
                except Exception as exc:
                    logger.warning("Neural prediction failed in backtest: %s", exc)

            consensus = ConsensusEngine()
            active_profile = "balanced"
            if self.auto_tune_consensus:
                active_profile = self._select_consensus_profile(step_idx)
                consensus.set_tuning_profile(self._consensus_profiles.get(active_profile))
                logger.debug("Using consensus tuning profile '%s'.", active_profile)
            if self.fit_filter_thresholds:
                consensus.fit_filters(train_data)
            if self.use_source_performance_weights:
                source_hits, source_counts = self._get_source_performance_snapshot()
                if source_hits:
                    consensus.set_source_performance(source_hits, source_counts)

            ranking_pool = list(all_preds)
            use_ensemble_now = self.use_ensemble and self._can_build_ensemble(all_preds)
            if use_ensemble_now:
                ensemble_preds = consensus.ensemble_predictions(
                    all_preds,
                    num_outputs=max(self.top_k, 2),
                )
                ranking_pool += ensemble_preds

            ranked_preds = consensus.rank_predictions(ranking_pool)
            top_preds = consensus.choose_diverse_top(ranked_preds, self.top_k) if ranked_preds else []

            if self.enable_abstain:
                top_preds = [pred for pred in top_preds if self._passes_abstain_gate(pred)]
                if not top_preds:
                    self._abstained_draws += 1

            for rank_idx, pred in enumerate(top_preds, 1):
                pred["consensus_profile"] = active_profile
                self._evaluate_prediction(pred, target_draw, i, rank_idx)

            baseline_count = len(top_preds) if self.enable_abstain else self.top_k
            for rank_idx in range(1, baseline_count + 1):
                baseline_pred = self._random_prediction()
                self._evaluate_prediction(baseline_pred, target_draw, i, rank_idx, self.baseline_results)

            if self.use_source_performance_weights:
                self._update_source_performance_history(ranking_pool, target_draw)

            if self.auto_tune_consensus:
                reward = self._compute_draw_reward(top_preds, target_draw)
                self._record_consensus_profile_reward(active_profile, reward)

        return self._generate_report()

    @staticmethod
    def _can_build_ensemble(predictions: list) -> bool:
        """Require at least two probabilistic sources before blending."""
        source_ids = set()
        for pred in predictions:
            if pred.get("main_prob_vector") is None or pred.get("bonus_prob_vector") is None:
                continue
            source_ids.add(pred.get("source", "Unknown"))
            if len(source_ids) >= 2:
                return True
        return False

    def _get_source_performance_snapshot(self) -> Tuple[Dict[str, float], Dict[str, int]]:
        """Return rolling per-source average main hits from prior draws."""
        source_hits = {}
        source_counts = {}
        for source, history in self._source_hits_by_model.items():
            if len(history) < self.source_performance_min_samples:
                continue
            recent = history[-self.source_performance_window :]
            source_hits[source] = float(np.mean(recent))
            source_counts[source] = len(recent)
        return source_hits, source_counts

    def _update_source_performance_history(self, candidates: list, actual_row: pd.Series) -> None:
        """
        Track per-source main-hit quality using each source's strongest candidate.
        Uses current draw outcomes only for future draws (no leakage).
        """
        if not candidates:
            return

        best_by_source: Dict[str, Tuple[float, Dict[str, Any]]] = {}
        for pred in candidates:
            source = pred.get("source", "Unknown")
            confidence = float(pred.get("confidence", 0.0))
            expected_main = float(pred.get("expected_main_prob", 0.0))
            score = confidence + (0.15 * expected_main)

            prev = best_by_source.get(source)
            if prev is None or score > prev[0]:
                best_by_source[source] = (score, pred)

        actual_main = set(actual_row["main_numbers"])
        for source, (_, pred) in best_by_source.items():
            pred_main = set(pred.get("main_numbers", []))
            main_hits = int(len(pred_main & actual_main))
            history = self._source_hits_by_model.setdefault(source, [])
            history.append(main_hits)

            # Keep bounded history to avoid unbounded growth in long runs.
            max_keep = max(self.source_performance_window * 3, self.source_performance_min_samples * 3)
            if len(history) > max_keep:
                del history[: len(history) - max_keep]

    def _random_prediction(self) -> Dict[str, Any]:
        main_nums = self._baseline_rng.choice(np.arange(1, MAIN_NUMBER_RANGE + 1), size=N_MAIN, replace=False)
        bonus_nums = self._baseline_rng.choice(np.arange(1, BONUS_NUMBER_RANGE + 1), size=N_BONUS, replace=False)
        return {
            "main_numbers": sorted(main_nums.tolist()),
            "bonus_numbers": sorted(bonus_nums.tolist()),
            "confidence": 0.0,
            "source": "RandomBaseline",
        }

    @staticmethod
    def _ticket_payout(main_hits: int) -> float:
        """Simple hypothetical payout curve used for ROI tracking."""
        if main_hits == 3:
            return 10.0
        if main_hits == 4:
            return 100.0
        if main_hits == 5:
            return 100000.0
        return 0.0

    def _evaluate_prediction(
        self,
        prediction: Dict[str, Any],
        actual_row: pd.Series,
        draw_idx: int,
        rank: int,
        results_list: Optional[list] = None,
    ) -> None:
        if results_list is None:
            results_list = self.results

        pred_main = set(prediction["main_numbers"])
        actual_main = set(actual_row["main_numbers"])

        pred_bonus = set(prediction["bonus_numbers"])
        actual_bonus = set(actual_row["bonus_numbers"])

        main_hits = len(pred_main & actual_main)
        bonus_hits = len(pred_bonus & actual_bonus)
        payout = self._ticket_payout(main_hits)
        net_return = float(payout - 1.0)

        result = {
            "draw_idx": draw_idx,
            "date": actual_row["date"],
            "rank": rank,
            "main_hits": main_hits,
            "bonus_hits": bonus_hits,
            "pred_main": list(pred_main),
            "actual_main": list(actual_main),
            "confidence": float(prediction.get("confidence", 0.0)),
            "source": prediction.get("source", "Unknown"),
            "payout": float(payout),
            "net_return": net_return,
            "consensus_profile": prediction.get("consensus_profile"),
        }
        results_list.append(result)

        if results_list is self.results:
            logger.info("Result: main_hits=%d, bonus_hits=%d", main_hits, bonus_hits)

    def _build_summary(self, df: pd.DataFrame, baseline_df: pd.DataFrame) -> Dict[str, Any]:
        draws_covered = int(self._draws_evaluated) if self._draws_evaluated > 0 else int(df["draw_idx"].nunique()) if not df.empty else 0
        tickets_target = int(self._tickets_target) if self._tickets_target > 0 else int(max(0, self.top_k * draws_covered))
        if df.empty:
            return {
                "draws_covered": draws_covered,
                "tickets_evaluated": 0,
                "tickets_target": tickets_target,
                "abstained_draws": int(self._abstained_draws),
                "participation_rate": 0.0,
                "avg_main_hits": 0.0,
                "avg_bonus_hits": 0.0,
                "three_plus_main_rate": 0.0,
                "roi_net": 0.0,
                "roi_per_ticket": 0.0,
                "avg_net_per_draw": 0.0,
                "baseline_avg_main_hits": float(baseline_df["main_hits"].mean()) if not baseline_df.empty else None,
                "lift_vs_baseline": None,
                "consensus_profile_last": self._active_consensus_profile,
            }

        total_predictions = len(df)
        avg_main_hits = float(df["main_hits"].mean())
        avg_bonus_hits = float(df["bonus_hits"].mean())
        winnings = float(df["payout"].sum()) if "payout" in df.columns else 0.0
        roi_net = float(winnings - total_predictions)
        roi_per_ticket = float(roi_net / total_predictions) if total_predictions > 0 else 0.0

        baseline_avg = float(baseline_df["main_hits"].mean()) if not baseline_df.empty else None
        lift = (avg_main_hits - baseline_avg) if baseline_avg is not None else None

        return {
            "draws_covered": draws_covered,
            "tickets_evaluated": total_predictions,
            "tickets_target": tickets_target,
            "abstained_draws": int(self._abstained_draws),
            "participation_rate": float(total_predictions / max(1, tickets_target)),
            "avg_main_hits": avg_main_hits,
            "avg_bonus_hits": avg_bonus_hits,
            "three_plus_main_rate": float((df["main_hits"] >= 3).mean()),
            "roi_net": roi_net,
            "roi_per_ticket": roi_per_ticket,
            "avg_net_per_draw": float(roi_net / max(1, draws_covered)),
            "baseline_avg_main_hits": baseline_avg,
            "lift_vs_baseline": lift,
            "consensus_profile_last": self._active_consensus_profile,
        }

    def _generate_report(self) -> Dict[str, Any]:
        df = pd.DataFrame(self.results)
        baseline_df = pd.DataFrame(self.baseline_results)

        if df.empty:
            logger.warning("No results to report.")
            summary = self._build_summary(df, baseline_df)
            return summary

        summary = self._build_summary(df, baseline_df)
        hit_counts = df["main_hits"].value_counts().sort_index()

        report_lines = []
        report_lines.append("=== Backtest Report ===")
        report_lines.append(f"Draws Covered: {summary['draws_covered']}")
        report_lines.append(f"Tickets Evaluated: {summary['tickets_evaluated']}")
        report_lines.append(f"Tickets Target: {summary.get('tickets_target', summary['tickets_evaluated'])}")
        report_lines.append(f"Abstained Draws: {summary.get('abstained_draws', 0)}")
        report_lines.append(f"Participation Rate: {summary.get('participation_rate', 1.0):.1%}")
        report_lines.append("Main Number Hits Distribution:")

        total_predictions = max(1, summary["tickets_evaluated"])
        for hits, count in hit_counts.items():
            percentage = (count / total_predictions) * 100
            report_lines.append(f"{hits} Matches: {count} ({percentage:.1f}%)")

        report_lines.append(f"Average Main Hits: {summary['avg_main_hits']:.3f}")
        report_lines.append(f"Average Bonus Hits: {summary['avg_bonus_hits']:.3f}")
        report_lines.append(f"3+ Main Hit Rate: {summary['three_plus_main_rate']:.3%}")

        if "rank" in df.columns:
            rank_hits = df.groupby("rank")["main_hits"].mean().sort_index()
            report_lines.append("Avg Main Hits by Rank:")
            for r, val in rank_hits.items():
                report_lines.append(f"  Rank {int(r)}: {val:.3f}")

        report_lines.append("Hypothetical ROI:")
        report_lines.append(f"Net: {summary['roi_net']:.1f}")
        report_lines.append(f"ROI/Ticket: {summary.get('roi_per_ticket', 0.0):+.4f}")
        report_lines.append(f"Avg Net/Draw: {summary.get('avg_net_per_draw', 0.0):+.4f}")
        if summary.get("consensus_profile_last"):
            report_lines.append(f"Consensus Profile (last): {summary['consensus_profile_last']}")

        if summary["baseline_avg_main_hits"] is not None:
            report_lines.append("Random Baseline:")
            report_lines.append(f"Average Main Hits: {summary['baseline_avg_main_hits']:.3f}")
            report_lines.append(f"Lift vs Baseline: {summary['lift_vs_baseline']:.3f}")

        print("\n".join(report_lines))
        logger.info(" | ".join(report_lines))

        self._plot_report(df, baseline_df)
        self._save_summary(summary)
        return summary

    def _save_summary(self, summary: Dict[str, Any]) -> None:
        try:
            out_name = "backtest_summary.json"
            if self.report_tag:
                out_name = f"backtest_summary_{self.report_tag}.json"
            output_path = PREDICTIONS_DIR / out_name
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            logger.info("Backtest summary saved to %s", output_path)
        except Exception as exc:
            logger.warning("Failed to save backtest summary: %s", exc)

    def _plot_report(self, df: pd.DataFrame, baseline_df: pd.DataFrame) -> None:
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
        except Exception as exc:
            logger.warning("Could not import matplotlib; skipping plot: %s", exc)
            return

        sys_series = df.groupby("draw_idx")["main_hits"].mean().sort_index()
        rolling_window = min(10, max(1, len(sys_series)))
        sys_roll = sys_series.rolling(window=rolling_window, min_periods=1).mean()

        base_series = None
        base_roll = None
        if not baseline_df.empty:
            base_series = baseline_df.groupby("draw_idx")["main_hits"].mean().sort_index()
            base_roll = base_series.rolling(window=rolling_window, min_periods=1).mean()

        fig, axes = plt.subplots(2, 1, figsize=(10, 8))

        axes[0].plot(sys_series.index, sys_roll, label="System (rolling avg)", color="tab:blue")
        if base_roll is not None:
            axes[0].plot(base_series.index, base_roll, label="Random baseline (rolling avg)", color="tab:orange")
        axes[0].set_title("Rolling Average Main Hits")
        axes[0].set_xlabel("Draw index")
        axes[0].set_ylabel("Avg main hits")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        bins = np.arange(0, N_MAIN + 2) - 0.5
        axes[1].hist(df["main_hits"], bins=bins, alpha=0.7, label="System", color="tab:blue")
        if not baseline_df.empty:
            axes[1].hist(
                baseline_df["main_hits"],
                bins=bins,
                alpha=0.7,
                label="Random baseline",
                color="tab:orange",
            )
        axes[1].set_xticks(range(0, N_MAIN + 1))
        axes[1].set_title("Main Hits Distribution")
        axes[1].set_xlabel("Main hits per ticket")
        axes[1].set_ylabel("Count")
        axes[1].legend()

        fig.tight_layout()
        out_name = "backtest_accuracy.png"
        if self.report_tag:
            out_name = f"backtest_accuracy_{self.report_tag}.png"
        output_path = PREDICTIONS_DIR / out_name
        fig.savefig(output_path, dpi=120)
        plt.close(fig)
        logger.info("Backtest plot saved to %s", output_path)


def _safe_ratio(delta: float, baseline: float) -> Optional[float]:
    if baseline == 0:
        return None
    return delta / baseline


def evaluate_readiness_gate(
    window: int = 20,
    top_k: int = 2,
    seed: int = 42,
    use_neural: bool = False,
    neural_mode: str = "frozen",
    neural_retrain_interval: int = 8,
    neural_train_window: int = 400,
    neural_train_epochs: int = 10,
    neural_batch_size: int = 32,
    neural_min_retrains_for_inference: int = 2,
    include_statistical: bool = True,
    include_random_forest: bool = True,
    include_extra_trees: bool = True,
    include_xgboost: bool = True,
    model_speed_profile: str = "ultrafast",
    model_retrain_interval: int = 3,
    use_source_performance_weights: bool = False,
    source_performance_window: int = 20,
    source_performance_min_samples: int = 4,
    min_avg_lift: float = 0.03,
    min_rolling_lift: float = 0.02,
    rolling_window: int = 8,
    confidence: float = 0.90,
    min_ci_lower: float = 0.0,
    min_draws: int = 20,
    min_tickets: int = 10,
    min_roi_per_ticket: float = 0.0,
    min_roi_ci_lower: float = 0.0,
    profit_confidence: float = 0.90,
    enable_abstain: bool = True,
    abstain_min_score: float = 0.12,
    abstain_min_confidence: float = 0.0,
    abstain_min_expected_main_prob: float = 0.35,
    abstain_min_support_count: int = 1,
    auto_tune_consensus: bool = True,
    auto_tune_interval: int = 3,
    auto_tune_window: int = 24,
    auto_tune_exploration: float = 0.15,
    report_tag: str = "readiness",
) -> Dict[str, Any]:
    """
    Run a leak-safe readiness check and return a pass/fail verdict.

    Gate criteria:
    - Average lift vs random baseline meets `min_avg_lift`
    - Latest rolling draw-level lift meets `min_rolling_lift`
    - Confidence interval lower bound of draw-level lift meets `min_ci_lower`
    - Evaluated draw count meets `min_draws`
    - ROI per ticket and ROI confidence bound are non-negative (or stricter custom thresholds)
    """
    bt = Backtester(
        window=window,
        top_k=top_k,
        use_neural=use_neural,
        seed=seed,
        use_ensemble=True,
        fit_filter_thresholds=True,
        neural_mode=neural_mode,
        neural_retrain_interval=neural_retrain_interval,
        neural_train_window=neural_train_window,
        neural_train_epochs=neural_train_epochs,
        neural_batch_size=neural_batch_size,
        neural_min_retrains_for_inference=neural_min_retrains_for_inference,
        include_statistical=include_statistical,
        include_random_forest=include_random_forest,
        include_extra_trees=include_extra_trees,
        include_xgboost=include_xgboost,
        model_speed_profile=model_speed_profile,
        model_retrain_interval=model_retrain_interval,
        use_source_performance_weights=use_source_performance_weights,
        source_performance_window=source_performance_window,
        source_performance_min_samples=source_performance_min_samples,
        enable_abstain=enable_abstain,
        abstain_min_score=abstain_min_score,
        abstain_min_confidence=abstain_min_confidence,
        abstain_min_expected_main_prob=abstain_min_expected_main_prob,
        abstain_min_support_count=abstain_min_support_count,
        auto_tune_consensus=auto_tune_consensus,
        auto_tune_interval=auto_tune_interval,
        auto_tune_window=auto_tune_window,
        auto_tune_exploration=auto_tune_exploration,
        report_tag=report_tag,
    )
    summary = bt.run()

    system_df = pd.DataFrame(bt.results)
    baseline_df = pd.DataFrame(bt.baseline_results)

    draw_lift = pd.Series(dtype=float)
    if not system_df.empty and not baseline_df.empty:
        sys_draw_hits = system_df.groupby("draw_idx")["main_hits"].mean().sort_index()
        base_draw_hits = baseline_df.groupby("draw_idx")["main_hits"].mean().sort_index()
        common_idx = sys_draw_hits.index.intersection(base_draw_hits.index)
        if len(common_idx) > 0:
            draw_lift = (sys_draw_hits.loc[common_idx] - base_draw_hits.loc[common_idx]).astype(float)

    n_draws = int(draw_lift.size)
    avg_lift = float(summary.get("lift_vs_baseline") or 0.0)
    draw_lift_mean = float(draw_lift.mean()) if n_draws > 0 else 0.0
    draw_lift_std = float(draw_lift.std(ddof=1)) if n_draws > 1 else 0.0
    draw_lift_se = float(draw_lift_std / np.sqrt(n_draws)) if n_draws > 1 else 0.0

    confidence_clamped = float(np.clip(confidence, 0.50, 0.999))
    z_value = float(NormalDist().inv_cdf(0.5 + (confidence_clamped / 2.0)))
    ci_lower = float(draw_lift_mean - (z_value * draw_lift_se))
    ci_upper = float(draw_lift_mean + (z_value * draw_lift_se))

    roll_win = max(1, int(rolling_window))
    if n_draws > 0:
        roll_win = min(roll_win, n_draws)
        rolling_series = draw_lift.rolling(window=roll_win, min_periods=roll_win).mean().dropna()
        rolling_lift = float(rolling_series.iloc[-1]) if not rolling_series.empty else draw_lift_mean
    else:
        rolling_lift = 0.0

    ticket_returns = pd.Series(dtype=float)
    if not system_df.empty and "net_return" in system_df.columns:
        ticket_returns = system_df["net_return"].astype(float)
    n_tickets = int(ticket_returns.size)
    roi_per_ticket = float(ticket_returns.mean()) if n_tickets > 0 else float(summary.get("roi_per_ticket", 0.0) or 0.0)
    roi_std = float(ticket_returns.std(ddof=1)) if n_tickets > 1 else 0.0
    roi_se = float(roi_std / np.sqrt(n_tickets)) if n_tickets > 1 else 0.0
    profit_conf_clamped = float(np.clip(profit_confidence, 0.50, 0.999))
    profit_z = float(NormalDist().inv_cdf(0.5 + (profit_conf_clamped / 2.0)))
    roi_ci_lower = float(roi_per_ticket - (profit_z * roi_se))
    roi_ci_upper = float(roi_per_ticket + (profit_z * roi_se))

    checks = {
        "min_draws": n_draws >= int(min_draws),
        "min_tickets": n_tickets >= int(min_tickets),
        "avg_lift": avg_lift >= float(min_avg_lift),
        "rolling_lift": rolling_lift >= float(min_rolling_lift),
        "ci_lower": ci_lower >= float(min_ci_lower),
        "roi_per_ticket": roi_per_ticket >= float(min_roi_per_ticket),
        "roi_ci_lower": roi_ci_lower >= float(min_roi_ci_lower),
    }
    passed = bool(all(checks.values()))

    report = {
        "passed": passed,
        "checks": checks,
        "metrics": {
            "draws_evaluated": n_draws,
            "avg_lift": avg_lift,
            "draw_lift_mean": draw_lift_mean,
            "draw_lift_std": draw_lift_std,
            "draw_lift_se": draw_lift_se,
            "rolling_lift": rolling_lift,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "tickets_evaluated": n_tickets,
            "roi_per_ticket": roi_per_ticket,
            "roi_std": roi_std,
            "roi_se": roi_se,
            "roi_ci_lower": roi_ci_lower,
            "roi_ci_upper": roi_ci_upper,
            "profit_confidence": profit_conf_clamped,
            "profit_z_value": profit_z,
            "confidence": confidence_clamped,
            "z_value": z_value,
        },
        "thresholds": {
            "min_draws": int(min_draws),
            "min_tickets": int(min_tickets),
            "min_avg_lift": float(min_avg_lift),
            "min_rolling_lift": float(min_rolling_lift),
            "rolling_window": int(rolling_window),
            "min_ci_lower": float(min_ci_lower),
            "confidence": confidence_clamped,
            "min_roi_per_ticket": float(min_roi_per_ticket),
            "min_roi_ci_lower": float(min_roi_ci_lower),
            "profit_confidence": profit_conf_clamped,
        },
        "summary": summary,
        "config": {
            "window": int(window),
            "top_k": int(top_k),
            "seed": int(seed),
            "use_neural": bool(use_neural),
            "neural_mode": neural_mode,
            "neural_retrain_interval": int(neural_retrain_interval),
            "neural_train_window": int(neural_train_window),
            "neural_train_epochs": int(neural_train_epochs),
            "neural_batch_size": int(neural_batch_size),
            "neural_min_retrains_for_inference": int(neural_min_retrains_for_inference),
            "include_statistical": bool(include_statistical),
            "include_random_forest": bool(include_random_forest),
            "include_extra_trees": bool(include_extra_trees),
            "include_xgboost": bool(include_xgboost),
            "model_speed_profile": model_speed_profile,
            "model_retrain_interval": int(model_retrain_interval),
            "use_source_performance_weights": bool(use_source_performance_weights),
            "source_performance_window": int(source_performance_window),
            "source_performance_min_samples": int(source_performance_min_samples),
            "enable_abstain": bool(enable_abstain),
            "abstain_min_score": float(abstain_min_score),
            "abstain_min_confidence": float(abstain_min_confidence),
            "abstain_min_expected_main_prob": float(abstain_min_expected_main_prob),
            "abstain_min_support_count": int(abstain_min_support_count),
            "auto_tune_consensus": bool(auto_tune_consensus),
            "auto_tune_interval": int(auto_tune_interval),
            "auto_tune_window": int(auto_tune_window),
            "auto_tune_exploration": float(auto_tune_exploration),
            "report_tag": report_tag,
        },
    }

    output_path = PREDICTIONS_DIR / f"{report_tag}_report.json"
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        logger.info("Readiness report saved to %s", output_path)
    except Exception as exc:
        logger.warning("Failed to save readiness report: %s", exc)

    logger.info(
        "Readiness verdict=%s | draws=%d tickets=%d | avg_lift=%.4f (>= %.4f) | rolling_lift=%.4f (>= %.4f) | ci_lower=%.4f (>= %.4f) | roi/ticket=%.4f (>= %.4f) | roi_ci_lower=%.4f (>= %.4f)",
        "PASS" if passed else "FAIL",
        n_draws,
        n_tickets,
        avg_lift,
        float(min_avg_lift),
        rolling_lift,
        float(min_rolling_lift),
        ci_lower,
        float(min_ci_lower),
        roi_per_ticket,
        float(min_roi_per_ticket),
        roi_ci_lower,
        float(min_roi_ci_lower),
    )
    return report


def run_controlled_comparison(
    window: int = 50,
    top_k: int = 2,
    seed: int = 42,
    use_neural: bool = True,
    neural_retrain_interval: int = 8,
    neural_train_window: int = 400,
    neural_train_epochs: int = 10,
    neural_batch_size: int = 32,
    neural_min_retrains_for_inference: int = 2,
    include_statistical: bool = True,
    include_random_forest: bool = True,
    include_extra_trees: bool = True,
    include_xgboost: bool = True,
    model_speed_profile: str = "default",
    model_retrain_interval: int = 1,
    use_source_performance_weights: bool = True,
    source_performance_window: int = 20,
    source_performance_min_samples: int = 4,
    enable_abstain: bool = False,
    abstain_min_score: float = 0.10,
    abstain_min_confidence: float = 0.0,
    abstain_min_expected_main_prob: float = 0.0,
    abstain_min_support_count: int = 1,
    auto_tune_consensus: bool = False,
    auto_tune_interval: int = 3,
    auto_tune_window: int = 24,
    auto_tune_exploration: float = 0.15,
) -> Dict[str, Any]:
    """Run controlled A/B comparison on the same window and seed."""
    logger.info(
        "Starting controlled comparison (window=%d, top_k=%d, neural=%s)",
        window,
        top_k,
        use_neural,
    )

    baseline = Backtester(
        window=window,
        top_k=top_k,
        use_neural=use_neural,
        seed=seed,
        use_ensemble=False,
        fit_filter_thresholds=False,
        neural_mode="frozen",
        neural_retrain_interval=neural_retrain_interval,
        neural_train_window=neural_train_window,
        neural_train_epochs=neural_train_epochs,
        neural_batch_size=neural_batch_size,
        neural_min_retrains_for_inference=neural_min_retrains_for_inference,
        include_statistical=include_statistical,
        include_random_forest=include_random_forest,
        include_extra_trees=include_extra_trees,
        include_xgboost=include_xgboost,
        model_speed_profile=model_speed_profile,
        model_retrain_interval=model_retrain_interval,
        use_source_performance_weights=False,
        source_performance_window=source_performance_window,
        source_performance_min_samples=source_performance_min_samples,
        enable_abstain=enable_abstain,
        abstain_min_score=abstain_min_score,
        abstain_min_confidence=abstain_min_confidence,
        abstain_min_expected_main_prob=abstain_min_expected_main_prob,
        abstain_min_support_count=abstain_min_support_count,
        auto_tune_consensus=False,
        report_tag="baseline",
    )
    baseline_summary = baseline.run()

    improved = Backtester(
        window=window,
        top_k=top_k,
        use_neural=use_neural,
        seed=seed,
        use_ensemble=True,
        fit_filter_thresholds=True,
        neural_mode="rolling" if use_neural else "frozen",
        neural_retrain_interval=neural_retrain_interval,
        neural_train_window=neural_train_window,
        neural_train_epochs=neural_train_epochs,
        neural_batch_size=neural_batch_size,
        neural_min_retrains_for_inference=neural_min_retrains_for_inference,
        include_statistical=include_statistical,
        include_random_forest=include_random_forest,
        include_extra_trees=include_extra_trees,
        include_xgboost=include_xgboost,
        model_speed_profile=model_speed_profile,
        model_retrain_interval=model_retrain_interval,
        use_source_performance_weights=use_source_performance_weights,
        source_performance_window=source_performance_window,
        source_performance_min_samples=source_performance_min_samples,
        enable_abstain=enable_abstain,
        abstain_min_score=abstain_min_score,
        abstain_min_confidence=abstain_min_confidence,
        abstain_min_expected_main_prob=abstain_min_expected_main_prob,
        abstain_min_support_count=abstain_min_support_count,
        auto_tune_consensus=auto_tune_consensus,
        auto_tune_interval=auto_tune_interval,
        auto_tune_window=auto_tune_window,
        auto_tune_exploration=auto_tune_exploration,
        report_tag="improved",
    )
    improved_summary = improved.run()

    avg_hits_delta = improved_summary["avg_main_hits"] - baseline_summary["avg_main_hits"]
    lift_ratio = _safe_ratio(avg_hits_delta, baseline_summary["avg_main_hits"])

    comparison = {
        "baseline": baseline_summary,
        "improved": improved_summary,
        "delta": {
            "avg_main_hits": avg_hits_delta,
            "avg_bonus_hits": improved_summary["avg_bonus_hits"] - baseline_summary["avg_bonus_hits"],
            "three_plus_main_rate": improved_summary["three_plus_main_rate"] - baseline_summary["three_plus_main_rate"],
            "roi_net": improved_summary["roi_net"] - baseline_summary["roi_net"],
            "lift_ratio_avg_main_hits": lift_ratio,
        },
        "config": {
            "window": window,
            "top_k": top_k,
            "seed": seed,
            "use_neural": use_neural,
            "neural_retrain_interval": neural_retrain_interval,
            "neural_train_window": neural_train_window,
            "neural_train_epochs": neural_train_epochs,
            "neural_batch_size": neural_batch_size,
            "neural_min_retrains_for_inference": neural_min_retrains_for_inference,
            "include_statistical": include_statistical,
            "include_random_forest": include_random_forest,
            "include_extra_trees": include_extra_trees,
            "include_xgboost": include_xgboost,
            "model_speed_profile": model_speed_profile,
            "model_retrain_interval": model_retrain_interval,
            "use_source_performance_weights": use_source_performance_weights,
            "source_performance_window": source_performance_window,
            "source_performance_min_samples": source_performance_min_samples,
            "enable_abstain": enable_abstain,
            "abstain_min_score": abstain_min_score,
            "abstain_min_confidence": abstain_min_confidence,
            "abstain_min_expected_main_prob": abstain_min_expected_main_prob,
            "abstain_min_support_count": abstain_min_support_count,
            "auto_tune_consensus": auto_tune_consensus,
            "auto_tune_interval": auto_tune_interval,
            "auto_tune_window": auto_tune_window,
            "auto_tune_exploration": auto_tune_exploration,
        },
    }

    lines = [
        "=== Controlled Backtest Comparison ===",
        f"Window: {window} draws | Top-k: {top_k} | Seed: {seed}",
        f"Baseline avg main hits: {baseline_summary['avg_main_hits']:.3f}",
        f"Improved avg main hits: {improved_summary['avg_main_hits']:.3f}",
        f"Delta avg main hits: {comparison['delta']['avg_main_hits']:+.3f}",
        f"Delta 3+ hit rate: {comparison['delta']['three_plus_main_rate']:+.3%}",
        f"Delta ROI net: {comparison['delta']['roi_net']:+.1f}",
    ]
    print("\n".join(lines))
    logger.info(" | ".join(lines))

    output_path = PREDICTIONS_DIR / "backtest_comparison.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)
    logger.info("Controlled comparison saved to %s", output_path)

    return comparison
