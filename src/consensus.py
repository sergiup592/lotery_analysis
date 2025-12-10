import numpy as np
import logging
from typing import List, Dict, Tuple
from .config import N_MAIN, N_BONUS, MAIN_NUMBER_RANGE, BONUS_NUMBER_RANGE, SAMPLING_CONFIG
from .filters import StatisticalFilter

logger = logging.getLogger(__name__)

class ConsensusEngine:
    """Consensus Engine for ranking and filtering lottery predictions with adaptive weighting."""

    def __init__(self):
        self.base_weights = {
            'Neural_Transformer': 0.35,  # Slightly reduced to balance with other models
            'XGBoost': 0.30,
            'RandomForest': 0.25,        # Increased from 0.2
            'Statistical': 0.10
        }
        self.filter = StatisticalFilter()
        # Adaptive weighting parameters
        self.confidence_weight_factor = 1.5  # How much to boost weights based on confidence
        self.diversity_penalty_base = 0.15   # Reduced from 0.2 for less aggressive penalty
        self.sampling = SAMPLING_CONFIG
        
    def rank_predictions(self, predictions: List[Dict]) -> List[Dict]:
        """Rank predictions using Diversity (Round-Robin) and Filtering."""
        
        # 1. Group by Source
        grouped_preds = {}
        for pred in predictions:
            s = pred['source']
            if s not in grouped_preds:
                grouped_preds[s] = []
            grouped_preds[s].append(pred)
            
        # 2. Filter and Sort each group internally
        valid_grouped_preds = {}
        source_conf = {}
        for source, preds in grouped_preds.items():
            valid_preds = []
            for p in preds:
                # Apply Hard Filters
                is_valid, _ = self.filter.validate(p['main_numbers'])
                if not is_valid:
                    continue
                
                # Apply Soft Validation Rules (Score Adjustment)
                weight = self.base_weights.get(p['source'], 0.1)
                base_score = p['confidence'] * weight
                density_boost = self._expected_hit_boost(p)
                final_score = self._apply_validation_rules(p, base_score * density_boost)
                p['final_score'] = final_score
                valid_preds.append(p)
                source_conf.setdefault(source, []).append(p['confidence'])
            
            # Sort by score descending
            valid_preds.sort(key=lambda x: x['final_score'], reverse=True)
            valid_grouped_preds[source] = valid_preds

        # Dynamic weighting: adaptively scale weights based on model confidence and performance
        dynamic_weights = {}
        for src, base_w in self.base_weights.items():
            if src in source_conf and source_conf[src]:
                avg_conf = max(1e-3, float(np.mean(source_conf[src])))
                std_conf = float(np.std(source_conf[src])) if len(source_conf[src]) > 1 else 0.0

                # Boost weight for high confidence and consistency (low std)
                confidence_boost = 1.0 + (avg_conf * self.confidence_weight_factor)
                consistency_boost = 1.0 + (0.1 / (std_conf + 0.1))  # Low std = high consistency

                dynamic_weights[src] = base_w * confidence_boost * consistency_boost
            else:
                dynamic_weights[src] = base_w

        # Normalize weights to sum to 1
        total_w = sum(dynamic_weights.values()) or 1.0
        for k in dynamic_weights:
            dynamic_weights[k] /= total_w

        logger.info(f"Adaptive model weights: {dynamic_weights}")
        
        # 3. Round-Robin Selection
        # We want to pick: 1st from Source A, 1st from Source B, ..., 2nd from Source A...
        # Order sources by weight? Or just cycle through them?
        # Let's cycle through them in order of weight (Neural -> XGB -> RF -> Stat)
        
        sorted_sources = sorted(dynamic_weights.keys(), key=lambda k: dynamic_weights[k], reverse=True)
        
        final_ranked = []
        seen_combinations = set()
        
        # Find max length to iterate
        max_len = max([len(p) for p in valid_grouped_preds.values()] + [0])
        
        for i in range(max_len):
            for source in sorted_sources:
                if source in valid_grouped_preds and i < len(valid_grouped_preds[source]):
                    pred = valid_grouped_preds[source][i]

                    # Check uniqueness (exact)
                    combo_key = tuple(sorted(pred['main_numbers']) + sorted(pred['bonus_numbers']))
                    if combo_key in seen_combinations:
                        continue

                    # Diversity penalty against already selected combos to avoid near-duplicates
                    diversity_factor = self._diversity_factor(pred, final_ranked)
                    pred['final_score'] *= diversity_factor

                    seen_combinations.add(combo_key)
                    final_ranked.append(pred)
                    
        return final_ranked

    def choose_diverse_top(self, ranked_preds: List[Dict], k: int) -> List[Dict]:
        """
        Select top-k predictions with a greedy diversity-aware score to avoid near-duplicates.
        """
        chosen = []
        seen = set()
        candidates = list(ranked_preds)

        while candidates and len(chosen) < k:
            best = None
            best_score = -1.0
            best_idx = -1
            for idx, pred in enumerate(candidates):
                combo_key = tuple(sorted(pred['main_numbers']) + sorted(pred['bonus_numbers']))
                if combo_key in seen:
                    continue

                diversity = self._diversity_factor(pred, chosen)
                score = pred['final_score'] * diversity
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
            if main_vec:
                try:
                    main_probs = np.array(main_vec, dtype=float)
                    boost += float(np.sum(main_probs[np.array(pred['main_numbers']) - 1]))
                except Exception:
                    pass

        if not bonus_prob_added:
            bonus_vec = pred.get('bonus_prob_vector')
            if bonus_vec:
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
            weight = self.base_weights.get(source, 0.1)
            ensemble_main += weight * avg_probs
            total_main_weight += weight

        for source, avg_probs in avg_bonus_by_source.items():
            weight = self.base_weights.get(source, 0.1)
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
