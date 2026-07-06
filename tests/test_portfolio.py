import unittest

import numpy as np

from src.config import BONUS_NUMBER_RANGE, MAIN_NUMBER_RANGE, N_BONUS, N_MAIN
from src.ev import EVConfig, EVModel, TIER_ALLOCATION, tier_probability
from src.portfolio import (
    _popcount64_lut,
    build_portfolio,
    evaluate_portfolio,
    popcount64,
    single_ticket_prize_probability,
)


def _model() -> EVModel:
    main = np.where(np.arange(1, MAIN_NUMBER_RANGE + 1) <= 25, 1.3, 0.7)
    bonus = np.where(np.arange(1, BONUS_NUMBER_RANGE + 1) <= 6, 1.25, 0.75)
    return EVModel(EVConfig(jackpot=50e6, sales=25e6), main_weights=main, bonus_weights=bonus)


class PopcountTests(unittest.TestCase):
    def test_both_popcount_implementations_match_python(self):
        rng = np.random.default_rng(1)
        values = rng.integers(0, 2**63, size=1000, dtype=np.uint64)
        expected = np.array([int(v).bit_count() for v in values], dtype=np.uint64)
        np.testing.assert_array_equal(popcount64(values).astype(np.uint64), expected)
        np.testing.assert_array_equal(_popcount64_lut(values).astype(np.uint64), expected)


class ExactnessTests(unittest.TestCase):
    def test_single_ticket_prize_probability_matches_tier_sum(self):
        # Any-prize probability must equal the sum over all 13 official tiers.
        tier_sum = sum(tier_probability(k, s) for (k, s) in TIER_ALLOCATION)
        self.assertAlmostEqual(single_ticket_prize_probability(), tier_sum, places=12)

    def test_single_ticket_portfolio_matches_analytic(self):
        stats = evaluate_portfolio([((1, 12, 23, 34, 45), (3, 9))])
        self.assertAlmostEqual(stats.p_any_prize, single_ticket_prize_probability(), places=12)

    def test_disjoint_portfolios_have_identical_main_structure(self):
        # Permutation invariance: any two disjoint portfolios of equal size
        # share the exact same max-match distribution.
        a = evaluate_portfolio([((1, 2, 3, 4, 5), (1, 2)), ((6, 7, 8, 9, 10), (3, 4))])
        b = evaluate_portfolio([((11, 22, 33, 44, 50), (1, 2)), ((13, 26, 39, 41, 47), (3, 4))])
        for k in range(N_MAIN + 1):
            self.assertAlmostEqual(
                a.max_main_match_distribution[k], b.max_main_match_distribution[k], places=12
            )

    def test_expected_winning_tickets_is_linear(self):
        tickets = [((1, 2, 3, 4, 5), (1, 2)), ((1, 2, 3, 4, 6), (1, 2)), ((40, 41, 42, 43, 44), (10, 11))]
        stats = evaluate_portfolio(tickets)
        self.assertAlmostEqual(
            stats.expected_winning_tickets, 3 * single_ticket_prize_probability(), places=12
        )

    def test_exact_probability_matches_monte_carlo(self):
        tickets = [
            ((3, 17, 22, 38, 46), (2, 11)),
            ((5, 9, 28, 33, 49), (2, 7)),
            ((11, 19, 27, 40, 44), (6, 12)),
        ]
        stats = evaluate_portfolio(tickets)

        rng = np.random.default_rng(99)
        n_sims = 200_000
        wins = 0
        ticket_sets = [(frozenset(m), frozenset(s)) for m, s in tickets]
        for _ in range(n_sims):
            draw_main = frozenset(rng.choice(MAIN_NUMBER_RANGE, size=N_MAIN, replace=False) + 1)
            draw_star = frozenset(rng.choice(BONUS_NUMBER_RANGE, size=N_BONUS, replace=False) + 1)
            for mains, stars in ticket_sets:
                m = len(mains & draw_main)
                if m >= 2 or (m == 1 and len(stars & draw_star) == 2):
                    wins += 1
                    break
        mc = wins / n_sims
        se = (stats.p_any_prize * (1 - stats.p_any_prize) / n_sims) ** 0.5
        self.assertAlmostEqual(mc, stats.p_any_prize, delta=4 * se)

    def test_overlapping_portfolio_has_lower_p_any_than_disjoint(self):
        disjoint = evaluate_portfolio([((1, 2, 3, 4, 5), (1, 2)), ((6, 7, 8, 9, 10), (3, 4))])
        overlapping = evaluate_portfolio([((1, 2, 3, 4, 5), (1, 2)), ((1, 2, 3, 4, 6), (3, 4))])
        self.assertGreater(disjoint.p_any_prize, overlapping.p_any_prize)


class BuildPortfolioTests(unittest.TestCase):
    def test_builds_disjoint_tickets_with_distinct_star_pairs(self):
        records = build_portfolio(5, _model(), seed=11)
        self.assertEqual(len(records), 5)
        seen_mains: set = set()
        star_pairs = set()
        for record in records:
            mains = record["main_numbers"]
            stars = record["bonus_numbers"]
            self.assertEqual(len(set(mains)), N_MAIN)
            self.assertFalse(seen_mains & set(mains), "mains must be disjoint")
            seen_mains.update(mains)
            star_pairs.add(tuple(sorted(stars)))
            self.assertIn("ev_eur", record)
        self.assertEqual(len(star_pairs), 5, "star pairs must be distinct")

    def test_polish_prefers_unpopular_numbers(self):
        records = build_portfolio(3, _model(), seed=7)
        chosen = [n for record in records for n in record["main_numbers"]]
        # With weights 1.3 below 26 / 0.7 above, most picks must be high numbers.
        self.assertGreater(sum(n > 25 for n in chosen) / len(chosen), 0.7)

    def test_handles_more_than_ten_tickets(self):
        records = build_portfolio(12, _model(), seed=3)
        self.assertEqual(len(records), 12)
        for record in records:
            self.assertEqual(len(set(record["main_numbers"])), N_MAIN)


if __name__ == "__main__":
    unittest.main()
