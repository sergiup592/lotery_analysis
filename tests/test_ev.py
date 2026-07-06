import unittest
from math import comb

import numpy as np

from src.config import BONUS_NUMBER_RANGE, MAIN_NUMBER_RANGE
from src.coverage import build_candidate_pool, select_diverse_tickets
from src.ev import (
    EVConfig,
    EVModel,
    TIER_ALLOCATION,
    elementary_symmetric,
    pattern_multiplier,
    poisson_share,
    tier_probability,
)


def _test_model(jackpot: float = 17e6, sales: float = 25e6) -> EVModel:
    """Deterministic model: explicit weights and sales, no file dependence."""
    main = np.where(np.arange(1, MAIN_NUMBER_RANGE + 1) <= 25, 1.3, 0.7)
    bonus = np.where(np.arange(1, BONUS_NUMBER_RANGE + 1) <= 6, 1.25, 0.75)
    return EVModel(EVConfig(jackpot=jackpot, sales=sales), main_weights=main, bonus_weights=bonus)


class TierMathTests(unittest.TestCase):
    def test_jackpot_probability_matches_official_odds(self):
        self.assertAlmostEqual(1.0 / tier_probability(5, 2), 139_838_160.0, places=0)

    def test_overall_prize_odds_about_one_in_thirteen(self):
        total = sum(tier_probability(k, s) for (k, s) in TIER_ALLOCATION)
        self.assertAlmostEqual(1.0 / total, 13.0, delta=0.05)

    def test_tier_allocations_sum_to_ninety_percent(self):
        # The remaining 10% is the official Reserve Fund contribution.
        self.assertAlmostEqual(sum(TIER_ALLOCATION.values()), 0.90, places=6)

    def test_poisson_share_limits(self):
        self.assertAlmostEqual(poisson_share(0.0), 1.0)
        self.assertAlmostEqual(poisson_share(1e-9), 1.0, places=6)
        self.assertLess(poisson_share(5.0), poisson_share(1.0))
        self.assertAlmostEqual(poisson_share(2.0), (1 - np.exp(-2.0)) / 2.0)

    def test_elementary_symmetric_matches_binomial_for_uniform_weights(self):
        self.assertAlmostEqual(
            elementary_symmetric(np.ones(50), 5), comb(50, 5), delta=comb(50, 5) * 1e-12
        )

    def test_pattern_multiplier_flags_visual_lines(self):
        self.assertGreater(pattern_multiplier([1, 2, 3, 4, 5]), 50.0)
        self.assertGreater(pattern_multiplier([5, 10, 15, 20, 25]), 5.0)
        self.assertEqual(pattern_multiplier([2, 14, 27, 39, 48]), 1.0)


class EVModelTests(unittest.TestCase):
    def test_unpopular_ticket_beats_neutral_beats_popular(self):
        model = _test_model()
        popular = model.ticket_ev([3, 7, 11, 13, 17], [3, 5])["ev_eur"]
        unpopular = model.ticket_ev([32, 38, 43, 46, 49], [10, 11])["ev_eur"]
        neutral = model.neutral_ev()["ev_eur"]
        self.assertGreater(unpopular, neutral)
        self.assertGreater(neutral, popular)

    def test_visual_line_is_priced_as_heavily_shared(self):
        model = _test_model(jackpot=100e6)
        sequence = model.ticket_ev([1, 2, 3, 4, 5], [1, 2])
        plain = model.ticket_ev([2, 14, 22, 9, 24], [1, 2])
        self.assertGreater(sequence["expected_jackpot_cowinners"], plain["expected_jackpot_cowinners"] * 20)
        self.assertLess(sequence["ev_eur"], plain["ev_eur"])

    def test_ev_increases_with_jackpot(self):
        low = _test_model(jackpot=17e6).neutral_ev()["ev_eur"]
        high = _test_model(jackpot=200e6).neutral_ev()["ev_eur"]
        self.assertGreater(high, low)

    def test_ev_stays_below_ticket_price_at_base_jackpot(self):
        model = _test_model()
        self.assertLess(model.neutral_ev()["ev_eur"], model.config.ticket_price)
        self.assertGreater(model.neutral_ev()["ev_eur"], 0.30)

    def test_tier_breakdown_is_consistent(self):
        model = _test_model()
        detailed = model.ticket_ev([32, 38, 43, 46, 49], [10, 11], detailed=True)
        self.assertEqual(len(detailed["tiers"]), 13)
        self.assertAlmostEqual(
            sum(tier["ev_contribution"] for tier in detailed["tiers"]),
            detailed["ev_eur"],
            places=4,
        )

    def test_jackpot_sweep_is_monotone(self):
        rows = _test_model().jackpot_sweep(jackpots=(17e6, 100e6, 250e6))
        values = [row["neutral_ev"] for row in rows]
        self.assertEqual(values, sorted(values))
        self.assertTrue(rows[-1]["must_be_won"])  # final row prices the must-be-won draw


class MustBeWonTests(unittest.TestCase):
    def _model(self, must_be_won: bool, sales: float = 62e6) -> EVModel:
        main = np.where(np.arange(1, MAIN_NUMBER_RANGE + 1) <= 25, 1.3, 0.7)
        bonus = np.where(np.arange(1, BONUS_NUMBER_RANGE + 1) <= 6, 1.25, 0.75)
        return EVModel(
            EVConfig(jackpot=250e6, sales=sales, must_be_won=must_be_won),
            main_weights=main,
            bonus_weights=bonus,
        )

    def test_must_be_won_beats_ordinary_draw_at_same_parameters(self):
        ordinary = self._model(False).neutral_ev()["ev_eur"]
        must_be_won = self._model(True).neutral_ev()["ev_eur"]
        self.assertGreater(must_be_won, ordinary + 1.0)

    def test_must_be_won_neutral_ev_exceeds_ticket_price_at_observed_cap_sales(self):
        # The headline result: at historically observed cap-draw sales (~62M),
        # the must-be-won draw is EV-positive -- the only such draw type.
        result = self._model(True).neutral_ev()
        self.assertGreater(result["ev_eur"], 2.50)
        self.assertGreater(result["ev_rolldown_component"], 1.0)

    def test_rolldown_premium_vanishes_at_huge_sales(self):
        # With enormous sales someone almost surely hits 5+2, so the rolldown
        # is nearly worthless and must-be-won converges to the ordinary draw.
        ordinary = self._model(False, sales=5e9).neutral_ev()["ev_eur"]
        must_be_won = self._model(True, sales=5e9).neutral_ev()["ev_eur"]
        self.assertLess(must_be_won - ordinary, 0.01 * ordinary)

    def test_unpopular_ticket_amplifies_rolldown_value(self):
        model = self._model(True)
        unpopular = model.ticket_ev([32, 38, 43, 46, 49], [10, 11])
        popular = model.ticket_ev([3, 7, 11, 13, 17], [3, 5])
        self.assertGreater(unpopular["ev_rolldown_component"], popular["ev_rolldown_component"] * 1.5)


class SalesAndRaffleTests(unittest.TestCase):
    def test_sales_elasticity_is_anchored_and_monotone(self):
        from src.ev import BASE_JACKPOT, JACKPOT_CAP, sales_for_jackpot

        self.assertAlmostEqual(sales_for_jackpot(BASE_JACKPOT), 18e6)
        self.assertGreater(sales_for_jackpot(100e6), sales_for_jackpot(50e6))
        self.assertAlmostEqual(sales_for_jackpot(JACKPOT_CAP), 62.27e6, delta=0.5e6)

    def test_explicit_sales_override_elasticity(self):
        model = EVModel(EVConfig(jackpot=250e6, sales=30e6))
        self.assertEqual(model.sales, 30e6)
        self.assertEqual(model.sales_source, "explicit")

    def test_raffle_adds_flat_ev(self):
        base = _test_model()
        with_raffle = EVModel(
            EVConfig(jackpot=17e6, sales=25e6, raffle_prize=1_000_000, raffle_pool=4_000_000),
            main_weights=base.main_weights,
            bonus_weights=base.bonus_weights,
        )
        delta = with_raffle.neutral_ev()["ev_eur"] - base.neutral_ev()["ev_eur"]
        self.assertAlmostEqual(delta, 0.25, places=9)


class EVSelectionTests(unittest.TestCase):
    def test_attach_annotates_candidates(self):
        model = _test_model()
        uniform = np.full(MAIN_NUMBER_RANGE, 1.0 / MAIN_NUMBER_RANGE)
        uniform_bonus = np.full(BONUS_NUMBER_RANGE, 1.0 / BONUS_NUMBER_RANGE)
        candidates = build_candidate_pool(uniform, uniform_bonus, 60, seed=3, source="Test")
        model.attach(candidates)
        for candidate in candidates:
            self.assertIn("ev_eur", candidate)
            self.assertGreater(candidate["ev_eur"], 0.0)

    def test_ev_mode_selects_higher_ev_than_legacy_mode(self):
        model = _test_model()
        uniform = np.full(MAIN_NUMBER_RANGE, 1.0 / MAIN_NUMBER_RANGE)
        uniform_bonus = np.full(BONUS_NUMBER_RANGE, 1.0 / BONUS_NUMBER_RANGE)
        candidates = build_candidate_pool(uniform, uniform_bonus, 400, seed=11, source="Test")

        def mean_ev(tickets):
            model.attach(tickets)
            return float(np.mean([ticket["ev_eur"] for ticket in tickets]))

        common = dict(
            num_tickets=4,
            main_probabilities=uniform,
            bonus_probabilities=uniform_bonus,
            overlap_penalty=0.55,
            coverage_bonus=0.35,
            source="Test",
        )
        legacy = select_diverse_tickets(list(candidates), **common)
        economic = select_diverse_tickets(list(candidates), ev_model=model, **common)
        self.assertGreater(mean_ev(economic), mean_ev(legacy))

    def test_disjoint_main_respected_in_ev_mode(self):
        model = _test_model()
        uniform = np.full(MAIN_NUMBER_RANGE, 1.0 / MAIN_NUMBER_RANGE)
        uniform_bonus = np.full(BONUS_NUMBER_RANGE, 1.0 / BONUS_NUMBER_RANGE)
        candidates = build_candidate_pool(uniform, uniform_bonus, 400, seed=5, source="Test")
        tickets = select_diverse_tickets(
            candidates,
            num_tickets=4,
            main_probabilities=uniform,
            bonus_probabilities=uniform_bonus,
            overlap_penalty=0.55,
            coverage_bonus=0.35,
            source="Test",
            disjoint_main=True,
            ev_model=model,
        )
        seen = set()
        for ticket in tickets:
            self.assertFalse(seen & set(ticket["main_numbers"]))
            seen.update(ticket["main_numbers"])

    def test_anti_popular_candidates_are_valid_and_cheap_to_share(self):
        model = _test_model()
        uniform = np.full(MAIN_NUMBER_RANGE, 1.0 / MAIN_NUMBER_RANGE)
        uniform_bonus = np.full(BONUS_NUMBER_RANGE, 1.0 / BONUS_NUMBER_RANGE)
        records = model.anti_popular_candidates(50, seed=9, main_probabilities=uniform,
                                                bonus_probabilities=uniform_bonus, source="Test")
        self.assertGreaterEqual(len(records), 40)
        keys = set()
        crowdings = []
        for record in records:
            mains = record["main_numbers"]
            stars = record["bonus_numbers"]
            self.assertEqual(len(set(mains)), 5)
            self.assertTrue(all(1 <= n <= MAIN_NUMBER_RANGE for n in mains))
            self.assertEqual(len(set(stars)), 2)
            self.assertTrue(all(1 <= n <= BONUS_NUMBER_RANGE for n in stars))
            keys.add(tuple(mains) + tuple(stars))
            crowdings.append(model.ticket_ev(mains, stars)["jackpot_crowding"])
        self.assertEqual(len(keys), len(records))
        self.assertLess(float(np.median(crowdings)), 1.0)


class CrowdingValidationTests(unittest.TestCase):
    def test_validation_recovers_signal_on_synthetic_winner_counts(self):
        # Build synthetic draws whose winner counts follow the crowding model
        # exactly; validation must then strongly beat the uniform baseline.
        import pandas as pd

        from src.winner_data import RANK_TIERS

        from src.ev import validate_crowding

        rng = np.random.default_rng(7)
        # Generate with the exact weights validate_crowding prices with
        # (the popularity module defaults), so the test is self-consistent
        # regardless of whether a fitted weights file is present.
        model = EVModel(EVConfig(sales=2.5e7))
        rows = []
        for draw_idx in range(120):
            mains = sorted(rng.choice(np.arange(1, 51), size=5, replace=False).tolist())
            stars = sorted(rng.choice(np.arange(1, 13), size=2, replace=False).tolist())
            mean_main = float(np.mean([model.main_weights[n - 1] for n in mains]))
            mean_star = float(np.mean([model.bonus_weights[n - 1] for n in stars]))
            row = {
                "date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=draw_idx),
                "main_numbers": mains,
                "bonus_numbers": stars,
            }
            for rank, (k, s) in RANK_TIERS.items():
                lam = (
                    2.5e7
                    * tier_probability(k, s)
                    * model._main_mean_crowding(mean_main, k)
                    * model._star_mean_crowding(mean_star, s)
                )
                row[f"winners_rank_{rank}"] = int(rng.poisson(lam))
            rows.append(row)

        report = validate_crowding(frame=pd.DataFrame(rows))
        self.assertGreater(report["mean_r2_improvement_over_uniform"], 0.3)


if __name__ == "__main__":
    unittest.main()
