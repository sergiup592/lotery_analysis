import numpy as np
import logging
from typing import List, Dict, Optional, Tuple, Any
from .config import N_MAIN, N_BONUS, MAIN_NUMBER_RANGE, BONUS_NUMBER_RANGE, SAMPLING_CONFIG
from .filters import StatisticalFilter

logger = logging.getLogger(__name__)

class ConsensusEngine:
    """Consensus Engine for ranking and filtering lottery predictions with adaptive weighting."""

    def __init__(self):
        self.base_weights = {
            'Neural_Transformer': 0.30,  # Adjusted
            'XGBoost': 0.25,
            'RandomForest': 0.20,
            'Statistical': 0.10,
            'ExtraTrees': 0.15,
            'Ensemble': 0.08             # Keep ensemble meaningful but not dominant
        }
        self.filter = StatisticalFilter()
        # Adaptive weighting parameters
        self.confidence_weight_factor = 1.5  # How much to boost weights based on confidence
        self.diversity_penalty_base = 0.15   # Reduced from 0.2 for less aggressive penalty
        self.min_dynamic_source_weight = 0.05
        self.max_dynamic_source_weight = 0.30
        self.max_dynamic_weight_by_source = {
            "Ensemble": 0.22
        }
        self.invalid_filter_penalty = 0.65
        self.performance_weight_factor = 0.40
        self.performance_min_samples = 6
        self.source_repeat_penalty = 0.12
        self.consensus_support_factor = 0.14
        self.coverage_gain_weight = 0.20
        self.selection_support_weight = 0.10
        self.random_main_hit_baseline = (N_MAIN * N_MAIN) / float(MAIN_NUMBER_RANGE)
        self.source_main_hits = {}
        self.source_sample_counts = {}
        self.sampling = SAMPLING_CONFIG
        self._tuning_param_bounds = {
            "confidence_weight_factor": (0.1, 3.0),
            "diversity_penalty_base": (0.02, 0.40),
            "invalid_filter_penalty": (0.30, 1.00),
            "performance_weight_factor": (0.05, 0.80),
            "source_repeat_penalty": (0.02, 0.30),
            "consensus_support_factor": (0.00, 0.40),
            "coverage_gain_weight": (0.00, 0.50),
            "selection_support_weight": (0.00, 0.30),
        }

    @staticmethod
    def tuning_profiles() -> Dict[str, Dict[str, float]]:
        """Named consensus tuning profiles used by prequential auto-tuning."""
        return {
            "balanced": {
                "confidence_weight_factor": 1.50,
                "diversity_penalty_base": 0.15,
                "invalid_filter_penalty": 0.65,
                "performance_weight_factor": 0.40,
                "source_repeat_penalty": 0.12,
                "consensus_support_factor": 0.14,
                "coverage_gain_weight": 0.20,
                "selection_support_weight": 0.10,
            },
            "precision": {
                "confidence_weight_factor": 1.95,
                "diversity_penalty_base": 0.10,
                "invalid_filter_penalty": 0.72,
                "performance_weight_factor": 0.48,
                "source_repeat_penalty": 0.08,
                "consensus_support_factor": 0.18,
                "coverage_gain_weight": 0.12,
                "selection_support_weight": 0.14,
            },
            "diversity": {
                "confidence_weight_factor": 1.10,
                "diversity_penalty_base": 0.24,
                "invalid_filter_penalty": 0.60,
                "performance_weight_factor": 0.35,
                "source_repeat_penalty": 0.18,
                "consensus_support_factor": 0.10,
                "coverage_gain_weight": 0.28,
                "selection_support_weight": 0.06,
            },
            "agreement": {
                "confidence_weight_factor": 1.45,
                "diversity_penalty_base": 0.13,
                "invalid_filter_penalty": 0.68,
                "performance_weight_factor": 0.44,
                "source_repeat_penalty": 0.10,
                "consensus_support_factor": 0.24,
                "coverage_gain_weight": 0.18,
                "selection_support_weight": 0.16,
            },
        }

    def get_tuning_profile(self) -> Dict[str, float]:
        """Return current tunable consensus parameters."""
        return {
            "confidence_weight_factor": float(self.confidence_weight_factor),
            "diversity_penalty_base": float(self.diversity_penalty_base),
            "invalid_filter_penalty": float(self.invalid_filter_penalty),
            "performance_weight_factor": float(self.performance_weight_factor),
            "source_repeat_penalty": float(self.source_repeat_penalty),
            "consensus_support_factor": float(self.consensus_support_factor),
            "coverage_gain_weight": float(self.coverage_gain_weight),
            "selection_support_weight": float(self.selection_support_weight),
        }

    def set_tuning_profile(self, profile: Optional[Dict[str, Any]]) -> None:
        """Apply tunable consensus parameters with bounds clamping."""
        if not profile:
            return
        for key, value in profile.items():
            if key not in self._tuning_param_bounds:
                continue
            lo, hi = self._tuning_param_bounds[key]
            try:
                numeric = float(value)
            except Exception:
                continue
            setattr(self, key, float(np.clip(numeric, lo, hi)))

    def fit_filters(self, historical_data) -> None:
        """Fit hard-filter thresholds using historical draws."""
        try:
            self.filter.fit(historical_data)
        except Exception as exc:
            logger.warning("Failed to fit statistical filters from history: %s", exc)

    def set_source_performance(self, source_main_hits: Dict[str, float], source_sample_counts: Dict[str, int]) -> None:
        """
        Set historical per-source performance for dynamic weighting.
        Values should come from strictly past draws to avoid leakage.
        """
        self.source_main_hits = dict(source_main_hits or {})
        self.source_sample_counts = dict(source_sample_counts or {})

    def rank_predictions(self, predictions: List[Dict]) -> List[Dict]:
        """Rank predictions using weighted global ranking and filtering."""
        
        # 1. Group by Source
        grouped_preds = {}
        for pred in predictions:
            normalized = self._normalize_prediction(pred)
            if normalized is None:
                continue
            s = normalized['source']
            if s not in grouped_preds:
                grouped_preds[s] = []
            grouped_preds[s].append(normalized)
            
        # 2. Filter and Sort each group internally
        valid_grouped_preds = {}
        source_conf = {}
        for source, preds in grouped_preds.items():
            # Keep only the strongest candidate for duplicate combinations within a source.
            deduped = {}
            for p in preds:
                combo_key = tuple(p["main_numbers"] + p["bonus_numbers"])
                prev = deduped.get(combo_key)
                if prev is None or float(p.get("confidence", 0.0)) > float(prev.get("confidence", 0.0)):
                    deduped[combo_key] = p

            valid_preds = []
            for p in deduped.values():
                # Apply statistical filter as soft penalty (avoid over-pruning candidate pool).
                is_valid, _ = self.filter.validate(p['main_numbers'])
                filter_penalty = 1.0 if is_valid else self.invalid_filter_penalty
                
                # IMPORTANT: Create a copy to avoid mutating the original prediction dict
                p_copy = p.copy()
                
                # Apply Soft Validation Rules (Score Adjustment)
                base_score = float(p_copy.get('confidence', 0.0))
                density_boost = self._expected_hit_boost(p_copy)
                final_score = self._apply_validation_rules(p_copy, base_score * density_boost) * filter_penalty
                p_copy['final_score'] = final_score
                if p_copy['final_score'] > 0:
                    valid_preds.append(p_copy)
                    source_conf.setdefault(source, []).append(float(p_copy.get('confidence', 0.0)))
            
            # Sort by score descending
            valid_preds.sort(key=lambda x: x['final_score'], reverse=True)
            valid_grouped_preds[source] = valid_preds

        # Keep only sources that have at least one valid candidate.
        present_sources = [s for s, preds in valid_grouped_preds.items() if preds]
        if not present_sources:
            return []

        # Dynamic weighting: adaptively scale weights based on model confidence and performance
        dynamic_weights = {}
        for src in present_sources:
            base_w = self.base_weights.get(src, 0.1)
            if src in source_conf and source_conf[src]:
                avg_conf = max(1e-3, float(np.mean(source_conf[src])))
                std_conf = float(np.std(source_conf[src])) if len(source_conf[src]) > 1 else 0.0

                # Boost weight for high confidence and consistency (low std)
                confidence_boost = 1.0 + (min(avg_conf, 0.6) * self.confidence_weight_factor)
                consistency_boost = 1.0 + min(0.4, (0.1 / (std_conf + 0.1)))  # Low std = high consistency
                performance_multiplier = self._performance_multiplier(src)

                dynamic_weights[src] = base_w * confidence_boost * consistency_boost * performance_multiplier
            else:
                dynamic_weights[src] = base_w * self._performance_multiplier(src)

        # Normalize weights to sum to 1
        total_w = sum(dynamic_weights.values()) or 1.0
        for k in dynamic_weights:
            dynamic_weights[k] /= total_w

        # Avoid any single source dominating rank order (important when one model is overconfident).
        for k in dynamic_weights:
            max_w = self.max_dynamic_weight_by_source.get(k, self.max_dynamic_source_weight)
            dynamic_weights[k] = float(
                np.clip(dynamic_weights[k], self.min_dynamic_source_weight, max_w)
            )
        clipped_total = sum(dynamic_weights.values()) or 1.0
        for k in dynamic_weights:
            dynamic_weights[k] /= clipped_total

        for source, preds in valid_grouped_preds.items():
            source_weight = dynamic_weights.get(source, self.base_weights.get(source, 0.1))
            for pred in preds:
                pred['final_score'] *= source_weight
            preds.sort(key=lambda x: x['final_score'], reverse=True)

        logger.info(f"Adaptive model weights: {dynamic_weights}")

        # 3. Global ranking across all sources (choose_diverse_top applies final diversity selection)
        all_valid = []
        for preds in valid_grouped_preds.values():
            all_valid.extend(preds)
        # Merge duplicates across sources and reward cross-source agreement.
        combo_aggregate = {}
        for pred in all_valid:
            combo_key = tuple(sorted(pred["main_numbers"]) + sorted(pred["bonus_numbers"]))
            source = pred.get("source", "Unknown")
            score = float(pred.get("final_score", 0.0))

            entry = combo_aggregate.get(combo_key)
            if entry is None:
                combo_aggregate[combo_key] = {
                    "score_sum": score,
                    "best_score": score,
                    "best_pred": pred.copy(),
                    "sources": {source},
                }
                continue

            entry["score_sum"] += score
            entry["sources"].add(source)
            if score > entry["best_score"]:
                entry["best_score"] = score
                entry["best_pred"] = pred.copy()

        final_ranked = []
        for entry in combo_aggregate.values():
            merged_pred = entry["best_pred"].copy()
            support_count = max(1, len(entry["sources"]))
            support_bonus = 1.0 + (self.consensus_support_factor * (support_count - 1))
            merged_pred["support_count"] = support_count
            merged_pred["support_sources"] = sorted(entry["sources"])
            merged_pred["final_score"] = float(entry["score_sum"] * support_bonus)
            final_ranked.append(merged_pred)

        final_ranked.sort(key=lambda x: x["final_score"], reverse=True)

        return final_ranked

    def _normalize_prediction(self, pred: Dict) -> Optional[Dict]:
        """Validate and normalize a candidate prediction before ranking."""
        if not isinstance(pred, dict):
            return None

        main_numbers = pred.get("main_numbers")
        bonus_numbers = pred.get("bonus_numbers")
        source = pred.get("source", "Unknown")

        if not isinstance(main_numbers, (list, tuple)) or not isinstance(bonus_numbers, (list, tuple)):
            return None

        try:
            main_nums = sorted(int(n) for n in main_numbers)
            bonus_nums = sorted(int(n) for n in bonus_numbers)
        except Exception:
            return None

        if len(main_nums) != N_MAIN or len(bonus_nums) != N_BONUS:
            return None
        if len(set(main_nums)) != N_MAIN or len(set(bonus_nums)) != N_BONUS:
            return None
        if any(n < 1 or n > MAIN_NUMBER_RANGE for n in main_nums):
            return None
        if any(n < 1 or n > BONUS_NUMBER_RANGE for n in bonus_nums):
            return None

        out = pred.copy()
        out["source"] = str(source)
        out["main_numbers"] = main_nums
        out["bonus_numbers"] = bonus_nums
        out["confidence"] = float(max(0.0, out.get("confidence", 0.0)))
        return out

    def choose_diverse_top(self, ranked_preds: List[Dict], k: int) -> List[Dict]:
        """
        Select top-k predictions with a greedy diversity-aware score to avoid near-duplicates.
        """
        chosen = []
        seen = set()
        candidates = list(ranked_preds)
        multi_source_mode = len({p.get("source") for p in candidates}) > 1

        while candidates and len(chosen) < k:
            best = None
            best_score = -1.0
            best_idx = -1
            for idx, pred in enumerate(candidates):
                combo_key = tuple(sorted(pred['main_numbers']) + sorted(pred['bonus_numbers']))
                if combo_key in seen:
                    continue

                diversity = self._diversity_factor(pred, chosen)
                source_diversity = self._source_diversity_factor(pred, chosen)
                support_multiplier = 1.0
                coverage_multiplier = 1.0
                if multi_source_mode:
                    support_count = int(pred.get("support_count", 1))
                    support_multiplier = 1.0 + (self.selection_support_weight * max(0, support_count - 1))
                    coverage_gain = self._marginal_coverage_gain(pred, chosen)
                    coverage_multiplier = 1.0 + (self.coverage_gain_weight * coverage_gain)
                score = pred['final_score'] * diversity * source_diversity * support_multiplier * coverage_multiplier
                if score > best_score:
                    best = pred
                    best_score = score
                    best_idx = idx

            if best is None:
                break

            best_copy = best.copy()
            best_copy['final_score'] = best_score
            chosen.append(best_copy)
            seen.add(tuple(sorted(best['main_numbers']) + sorted(best['bonus_numbers'])))
            candidates.pop(best_idx)

        return chosen

    def _performance_multiplier(self, source: str) -> float:
        """Convert source hit history into a bounded multiplicative weight."""
        if source not in self.source_main_hits:
            return 1.0

        avg_hits = float(self.source_main_hits[source])
        sample_count = int(self.source_sample_counts.get(source, 0))
        reliability = min(1.0, sample_count / float(max(1, self.performance_min_samples)))

        # Relative edge vs random expectation (5 hits drawn from 50 -> expected 0.5 overlaps).
        edge = (avg_hits - self.random_main_hit_baseline) / max(0.2, self.random_main_hit_baseline)
        edge = float(np.clip(edge, -0.8, 1.0))

        multiplier = 1.0 + (self.performance_weight_factor * edge * reliability)
        return float(np.clip(multiplier, 0.65, 1.40))

    def _source_diversity_factor(self, pred: Dict, selected: List[Dict]) -> float:
        """
        Mildly penalize reusing the same source in top-k picks.
        This improves model-family coverage when top scores are near-ties.
        """
        if not selected:
            return 1.0
        source = pred.get("source")
        repeats = sum(1 for s in selected if s.get("source") == source)
        if repeats <= 0:
            return 1.0
        return max(0.7, 1.0 - (self.source_repeat_penalty * repeats))

    def _marginal_coverage_gain(self, pred: Dict, selected: List[Dict]) -> float:
        """
        Estimate marginal coverage gain for this candidate relative to already selected picks.
        Uses candidate probability mass so gains prioritize uncovered high-likelihood numbers.
        Returns a bounded [0, 1] value.
        """
        if not selected:
            return 0.0

        covered_main = set()
        covered_bonus = set()
        for existing in selected:
            covered_main.update(existing.get("main_numbers", []))
            covered_bonus.update(existing.get("bonus_numbers", []))

        main_gain = 0.0
        main_vec = pred.get("main_prob_vector")
        if main_vec is not None:
            try:
                probs = np.asarray(main_vec, dtype=float)
                idx = np.asarray(pred["main_numbers"], dtype=int) - 1
                chosen_mass = float(np.sum(probs[idx]))
                if chosen_mass > 0:
                    uncovered = [n for n in pred["main_numbers"] if n not in covered_main]
                    if uncovered:
                        unc_idx = np.asarray(uncovered, dtype=int) - 1
                        main_gain = float(np.sum(probs[unc_idx]) / chosen_mass)
            except Exception:
                main_gain = 0.0

        bonus_gain = 0.0
        bonus_vec = pred.get("bonus_prob_vector")
        if bonus_vec is not None:
            try:
                probs = np.asarray(bonus_vec, dtype=float)
                idx = np.asarray(pred["bonus_numbers"], dtype=int) - 1
                chosen_mass = float(np.sum(probs[idx]))
                if chosen_mass > 0:
                    uncovered = [n for n in pred["bonus_numbers"] if n not in covered_bonus]
                    if uncovered:
                        unc_idx = np.asarray(uncovered, dtype=int) - 1
                        bonus_gain = float(np.sum(probs[unc_idx]) / chosen_mass)
            except Exception:
                bonus_gain = 0.0

        gain = (0.65 * main_gain) + (0.35 * bonus_gain)
        return float(np.clip(gain, 0.0, 1.0))

    def _expected_hit_boost(self, pred: Dict) -> float:
        """
        Boost score using the probability mass of the selected numbers (if provided).
        Encourages combinations that carry more predicted probability weight.

        Uses ONLY one source to avoid double-counting:
        - Prefers precomputed expected_prob values
        - Falls back to calculating from prob_vector if not available
        """
        boost = 1.0
        main_prob_added = False
        bonus_prob_added = False

        # Prefer precomputed expected probability masses if present
        if pred.get('expected_main_prob') is not None:
            boost += float(pred.get('expected_main_prob', 0.0))
            main_prob_added = True
        if pred.get('expected_bonus_prob') is not None:
            boost += float(0.5 * pred.get('expected_bonus_prob', 0.0))
            bonus_prob_added = True

        # Fall back to vectors ONLY if precomputed values not available
        if not main_prob_added:
            main_vec = pred.get('main_prob_vector')
            if main_vec is not None and len(main_vec) > 0:
                try:
                    main_probs = np.array(main_vec, dtype=float)
                    boost += float(np.sum(main_probs[np.array(pred['main_numbers']) - 1]))
                except Exception:
                    pass

        if not bonus_prob_added:
            bonus_vec = pred.get('bonus_prob_vector')
            if bonus_vec is not None and len(bonus_vec) > 0:
                try:
                    bonus_probs = np.array(bonus_vec, dtype=float)
                    boost += float(0.5 * np.sum(bonus_probs[np.array(pred['bonus_numbers']) - 1]))
                except Exception:
                    pass

        return max(boost, 0.0)

    def _diversity_factor(self, pred: Dict, selected: List[Dict]) -> float:
        """
        Penalize candidates that heavily overlap with already selected combos to improve coverage.
        Uses adaptive penalty based on overlap severity.
        """
        if not selected:
            return 1.0
        main_set = set(pred['main_numbers'])
        bonus_set = set(pred['bonus_numbers'])

        max_overlap_main = 0.0
        max_overlap_bonus = 0.0
        avg_overlap_main = 0.0
        avg_overlap_bonus = 0.0

        for s in selected:
            s_main = set(s['main_numbers'])
            s_bonus = set(s['bonus_numbers'])
            overlap_main = len(main_set & s_main) / float(N_MAIN)
            overlap_bonus = len(bonus_set & s_bonus) / float(N_BONUS)
            max_overlap_main = max(max_overlap_main, overlap_main)
            max_overlap_bonus = max(max_overlap_bonus, overlap_bonus)
            avg_overlap_main += overlap_main
            avg_overlap_bonus += overlap_bonus

        avg_overlap_main /= len(selected)
        avg_overlap_bonus /= len(selected)

        # Adaptive penalty: penalize based on both max and average overlap
        # High max overlap = too similar to one prediction
        # High avg overlap = generally not diverse
        main_penalty = (0.5 * max_overlap_main + 0.5 * avg_overlap_main) * self.diversity_penalty_base
        bonus_penalty = (0.5 * max_overlap_bonus + 0.5 * avg_overlap_bonus) * self.diversity_penalty_base * 0.5

        penalty = 1.0 - (main_penalty + bonus_penalty)
        # Keep a reasonable floor to not completely eliminate good candidates
        return max(penalty, 0.6)

    def _apply_validation_rules(self, pred: Dict, current_score: float) -> float:
        """Apply heuristic rules to adjust score."""
        main_nums = pred['main_numbers']
        bonus_nums = pred['bonus_numbers']

        # Rule 1: Penalize consecutive numbers (e.g., 1, 2, 3)
        consecutive_count = 0
        for i in range(len(main_nums) - 1):
            if main_nums[i+1] - main_nums[i] == 1:
                consecutive_count += 1
        if consecutive_count > 2:
            current_score *= 0.5

        # Rule 2: Penalize extreme ranges (all low or all high)
        avg_val = np.mean(main_nums)
        if avg_val < 10 or avg_val > 40:
            current_score *= 0.8

        # Rule 3: Validate unique numbers
        if len(set(main_nums)) != N_MAIN or len(set(bonus_nums)) != N_BONUS:
            return 0.0

        # Rule 4: Probability quality gates - boost high-probability combos
        em = pred.get('expected_main_prob')
        eb = pred.get('expected_bonus_prob')
        # Reward combinations with higher predicted probability mass
        if em is not None:
            if em > 0.55:
                current_score *= 1.20  # Increased reward for high-confidence
            elif em > 0.45:
                current_score *= 1.10  # Mild reward for good confidence
            elif em < 0.30:
                current_score *= 0.70  # Stronger penalty for low confidence
        if eb is not None:
            if eb > 0.45:
                current_score *= 1.10
            elif eb < 0.20:
                current_score *= 0.75

        return current_score

    def ensemble_predictions(self, predictions: List[Dict], num_outputs: int = 5) -> List[Dict]:
        """
        Create ensemble predictions by combining probability distributions from all models.

        Instead of just ranking individual predictions, this combines the probability
        distributions to create a better-calibrated ensemble distribution, then samples
        from that combined distribution.

        Args:
            predictions: List of predictions from all models (must include prob_vectors)
            num_outputs: Number of ensemble predictions to generate

        Returns:
            List of new predictions sampled from ensemble distribution
        """
        # Group predictions by source and extract probability vectors
        source_main_probs = {}
        source_bonus_probs = {}

        for pred in predictions:
            source = pred['source']
            main_vec = pred.get('main_prob_vector')
            bonus_vec = pred.get('bonus_prob_vector')

            if main_vec is not None:
                if source not in source_main_probs:
                    source_main_probs[source] = []
                source_main_probs[source].append(np.array(main_vec))

            if bonus_vec is not None:
                if source not in source_bonus_probs:
                    source_bonus_probs[source] = []
                source_bonus_probs[source].append(np.array(bonus_vec))

        if not source_main_probs:
            logger.warning("No probability vectors found for ensemble; returning empty list")
            return []

        # Average probability vectors within each source
        avg_main_by_source = {}
        avg_bonus_by_source = {}

        for source, probs_list in source_main_probs.items():
            avg_main_by_source[source] = np.mean(probs_list, axis=0)

        for source, probs_list in source_bonus_probs.items():
            avg_bonus_by_source[source] = np.mean(probs_list, axis=0)

        # Combine across sources using weighted average
        ensemble_main = np.zeros(MAIN_NUMBER_RANGE)
        ensemble_bonus = np.zeros(BONUS_NUMBER_RANGE)
        total_main_weight = 0.0
        total_bonus_weight = 0.0

        for source, avg_probs in avg_main_by_source.items():
            weight = self.base_weights.get(source, 0.1) * self._performance_multiplier(source)
            ensemble_main += weight * avg_probs
            total_main_weight += weight

        for source, avg_probs in avg_bonus_by_source.items():
            weight = self.base_weights.get(source, 0.1) * self._performance_multiplier(source)
            ensemble_bonus += weight * avg_probs
            total_bonus_weight += weight

        # Normalize
        if total_main_weight > 0:
            ensemble_main /= total_main_weight
        if total_bonus_weight > 0:
            ensemble_bonus /= total_bonus_weight

        # Apply temperature scaling for sampling diversity
        ensemble_main = self._apply_temperature(ensemble_main, self.sampling.get('temperature_main', 0.65))
        ensemble_bonus = self._apply_temperature(ensemble_bonus, self.sampling.get('temperature_bonus', 0.60))

        # Generate ensemble predictions
        ensemble_preds = []
        seen_combos = set()

        for _ in range(num_outputs * 3):  # Generate extra to filter duplicates
            if len(ensemble_preds) >= num_outputs:
                break

            # Sample main numbers
            main_nums = np.random.choice(
                np.arange(1, MAIN_NUMBER_RANGE + 1),
                size=N_MAIN,
                replace=False,
                p=ensemble_main
            )

            # Sample bonus numbers
            bonus_nums = np.random.choice(
                np.arange(1, BONUS_NUMBER_RANGE + 1),
                size=N_BONUS,
                replace=False,
                p=ensemble_bonus
            )

            combo_key = tuple(sorted(main_nums)) + tuple(sorted(bonus_nums))
            if combo_key in seen_combos:
                continue

            # Validate with filters
            is_valid, _ = self.filter.validate(sorted(main_nums.tolist()))
            if not is_valid:
                continue

            seen_combos.add(combo_key)

            # Calculate confidence
            main_conf = np.mean(ensemble_main[main_nums - 1])
            bonus_conf = np.mean(ensemble_bonus[bonus_nums - 1])

            ensemble_preds.append({
                'main_numbers': sorted(main_nums.tolist()),
                'bonus_numbers': sorted(bonus_nums.tolist()),
                'confidence': float((main_conf + bonus_conf) / 2),
                'main_prob_vector': ensemble_main.tolist(),
                'bonus_prob_vector': ensemble_bonus.tolist(),
                'expected_main_prob': float(np.sum(ensemble_main[main_nums - 1])),
                'expected_bonus_prob': float(np.sum(ensemble_bonus[bonus_nums - 1])),
                'source': 'Ensemble',
                'final_score': float((main_conf + bonus_conf) / 2) * 1.2  # Boost ensemble predictions
            })

        return ensemble_preds

    def _apply_temperature(self, probs: np.ndarray, temperature: float) -> np.ndarray:
        """Apply temperature scaling to probability distribution."""
        probs = np.array(probs, dtype=np.float64)
        probs = np.clip(probs, 1e-10, 1.0 - 1e-10)

        # Convert to logits for proper temperature scaling
        logits = np.log(probs / (1 - probs))

        if temperature and temperature > 0:
            logits = logits / temperature

        # Convert back via softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)

        return probs
