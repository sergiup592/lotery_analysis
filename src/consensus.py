import numpy as np
import logging
from typing import List, Dict
from .config import N_MAIN, N_BONUS
from .filters import StatisticalFilter

logger = logging.getLogger(__name__)

class ConsensusEngine:
    """Consensus Engine for ranking and filtering lottery predictions."""
    
    def __init__(self):
        self.base_weights = {
            'Neural_Transformer': 0.4,
            'XGBoost': 0.3,
            'RandomForest': 0.2,
            'Statistical': 0.1
        }
        self.filter = StatisticalFilter()
        
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

        # Dynamic weighting: scale base weights by mean confidence per source (if available)
        dynamic_weights = {}
        for src, base_w in self.base_weights.items():
            if src in source_conf and source_conf[src]:
                avg_conf = max(1e-3, float(np.mean(source_conf[src])))
                dynamic_weights[src] = base_w * (1.0 + avg_conf)
            else:
                dynamic_weights[src] = base_w
        total_w = sum(dynamic_weights.values()) or 1.0
        for k in dynamic_weights:
            dynamic_weights[k] /= total_w
        
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
        """
        boost = 1.0

        # Prefer precomputed expected probability masses if present
        if pred.get('expected_main_prob') is not None:
            boost += float(pred.get('expected_main_prob', 0.0))
        if pred.get('expected_bonus_prob') is not None:
            boost += float(0.5 * pred.get('expected_bonus_prob', 0.0))

        # Fall back to vectors for finer-grained sums
        main_vec = pred.get('main_prob_vector')
        bonus_vec = pred.get('bonus_prob_vector')

        if main_vec:
            try:
                main_probs = np.array(main_vec, dtype=float)
                boost += float(np.sum(main_probs[np.array(pred['main_numbers']) - 1]))
            except Exception:
                pass

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
        """
        if not selected:
            return 1.0
        main_set = set(pred['main_numbers'])
        bonus_set = set(pred['bonus_numbers'])

        max_overlap_main = 0.0
        max_overlap_bonus = 0.0
        for s in selected:
            s_main = set(s['main_numbers'])
            s_bonus = set(s['bonus_numbers'])
            overlap_main = len(main_set & s_main) / float(N_MAIN)
            overlap_bonus = len(bonus_set & s_bonus) / float(N_BONUS)
            max_overlap_main = max(max_overlap_main, overlap_main)
            max_overlap_bonus = max(max_overlap_bonus, overlap_bonus)

        # Penalize heavy overlap; keep a floor so good candidates aren't zeroed out
        penalty = 1.0 - (0.4 * max_overlap_main + 0.2 * max_overlap_bonus)
        return max(penalty, 0.4)

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
            
        # Rule 3: Penalize if main numbers are in bonus numbers (unlikely but possible in some lotteries, usually distinct sets)
        if len(set(main_nums)) != N_MAIN or len(set(bonus_nums)) != N_BONUS:
            return 0.0

        # Rule 4: Probability quality gates
        em = pred.get('expected_main_prob')
        eb = pred.get('expected_bonus_prob')
        # Mild boost for high-mass combos, mild penalty for low-mass
        if em is not None:
            if em > 0.55:
                current_score *= 1.15
            elif em < 0.35:
                current_score *= 0.6
        if eb is not None:
            if eb > 0.45:
                current_score *= 1.05
            elif eb < 0.25:
                current_score *= 0.8
            
        return current_score

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
        # (Already handled by Hard Filter, but kept as soft penalty if filter is relaxed)
        avg_val = np.mean(main_nums)
        if avg_val < 10 or avg_val > 40:
            current_score *= 0.8
            
        # Rule 3: Penalize if main numbers are in bonus numbers (unlikely but possible in some lotteries, usually distinct sets)
        if len(set(main_nums)) != N_MAIN or len(set(bonus_nums)) != N_BONUS:
            return 0.0
            
        return current_score
