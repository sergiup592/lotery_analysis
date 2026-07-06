import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from src import popularity
from src.popularity_fit import (
    RANK_PROBABILITIES,
    fit_popularity,
    main_feature_matrix,
    star_feature_matrix,
    tier_probability,
)
from src.winner_data import (
    RANK_TIERS,
    load_winner_counts,
    read_winner_counts_cache,
    write_winner_counts_cache,
)


def synthetic_winner_frame(
    main_weights: np.ndarray,
    star_weights: np.ndarray,
    n_draws: int = 400,
    seed: int = 7,
    tickets_per_draw: float = 4e7,
) -> pd.DataFrame:
    """Simulate winner counts from the popularity model with known weights."""
    rng = np.random.default_rng(seed)
    rows = []
    dates = pd.date_range("2017-01-03", periods=n_draws, freq="3D")
    for draw_idx in range(n_draws):
        mains = np.sort(rng.choice(np.arange(1, 51), size=5, replace=False))
        stars = np.sort(rng.choice(np.arange(1, 13), size=2, replace=False))
        sales = tickets_per_draw * rng.uniform(0.7, 1.4)

        mean_main_drawn = main_weights[mains - 1].mean()
        mean_main_undrawn = (50.0 - 5.0 * mean_main_drawn) / 45.0
        mean_star_drawn = star_weights[stars - 1].mean()
        mean_star_undrawn = (12.0 - 2.0 * mean_star_drawn) / 10.0

        row = {"date": dates[draw_idx], "main_numbers": mains.tolist(), "bonus_numbers": stars.tolist()}
        for rank, (main_hits, star_hits) in RANK_TIERS.items():
            expected = (
                sales
                * RANK_PROBABILITIES[rank]
                * mean_main_drawn**main_hits
                * mean_main_undrawn ** (5 - main_hits)
                * mean_star_drawn**star_hits
                * mean_star_undrawn ** (2 - star_hits)
            )
            row[f"winners_rank_{rank}"] = int(rng.poisson(expected))
        rows.append(row)
    return pd.DataFrame(rows)


class TierProbabilityTests(unittest.TestCase):
    def test_probabilities_match_known_euromillions_odds(self):
        self.assertAlmostEqual(1.0 / tier_probability(5, 2), 139_838_160.0, delta=1.0)
        self.assertAlmostEqual(1.0 / tier_probability(2, 0), 22.0, delta=0.5)

    def test_rank_order_matches_decreasing_prize_structure(self):
        # Jackpot is the rarest tier; rank 13 (2+0) the most common.
        probabilities = [RANK_PROBABILITIES[rank] for rank in sorted(RANK_TIERS)]
        self.assertEqual(min(probabilities), probabilities[0])
        self.assertEqual(max(probabilities), probabilities[-1])


class FitRecoveryTests(unittest.TestCase):
    def test_recovers_weights_constructed_inside_feature_span(self):
        main_features, _ = main_feature_matrix()
        star_features, _ = star_feature_matrix()
        true_main_coeff = np.array([1.0, 0.28, 0.10, 0.08, 0.12, -0.10, -0.15])
        true_star_coeff = np.array([1.0, 0.18, 0.05, -0.08])
        true_main = main_features @ true_main_coeff
        true_main = true_main / true_main.mean()
        true_star = star_features @ true_star_coeff
        true_star = true_star / true_star.mean()

        frame = synthetic_winner_frame(true_main, true_star)
        result = fit_popularity(data=frame)

        np.testing.assert_allclose(result.main_weights, true_main, atol=0.02)
        np.testing.assert_allclose(result.bonus_weights, true_star, atol=0.02)

    def test_multiplicative_truth_recovered_approximately(self):
        # The heuristic priors are multiplicative, i.e. outside the additive
        # feature span; the fit should still track them closely.
        true_main = popularity.prior_main_weights()
        true_star = popularity.prior_bonus_weights()
        frame = synthetic_winner_frame(true_main, true_star, seed=11)
        result = fit_popularity(data=frame)

        correlation = float(np.corrcoef(result.main_weights, true_main)[0, 1])
        self.assertGreater(correlation, 0.95)
        self.assertLess(float(np.abs(result.bonus_weights - true_star).max()), 0.05)

    def test_uniform_truth_yields_near_flat_weights(self):
        frame = synthetic_winner_frame(np.ones(50), np.ones(12), seed=3)
        result = fit_popularity(data=frame)
        self.assertLess(float(np.abs(result.main_weights - 1.0).max()), 0.03)
        self.assertLess(float(np.abs(result.bonus_weights - 1.0).max()), 0.03)

    def test_holdout_metrics_reported(self):
        true_main = popularity.prior_main_weights()
        true_star = popularity.prior_bonus_weights()
        frame = synthetic_winner_frame(true_main, true_star, seed=5)
        result = fit_popularity(
            data=frame,
            prior_main_weights=true_main,
            prior_bonus_weights=true_star,
        )
        validation = result.diagnostics["validation"]
        self.assertGreater(validation["main_fitted"]["r_squared"], 0.5)
        self.assertIn("main_prior_heuristic", validation)

    def test_refuses_tiny_datasets(self):
        frame = synthetic_winner_frame(np.ones(50), np.ones(12), n_draws=30)
        with self.assertRaises(ValueError):
            fit_popularity(data=frame)


class ArchiveParsingTests(unittest.TestCase):
    def _fdj_csv(self) -> str:
        winner_headers = ";".join(
            f"nombre_de_gagnant_au_rang{rank}_Euro_Millions" for rank in range(1, 14)
        )
        header = (
            "annee_numero_de_tirage;jour_de_tirage;date_de_tirage;"
            "boule_1;boule_2;boule_3;boule_4;boule_5;etoile_1;etoile_2;"
            f"{winner_headers};devise"
        )
        rows = [
            # pre-era draw: must be filtered out
            "2015123;VENDREDI;06/11/2015;5;12;19;33;41;2;7;"
            "0;2;6;30;800;1500;2500;30000;40000;90000;140000;600000;1400000;EUR",
            "2017005;MARDI;17/01/2017;7;13;21;34;48;3;9;"
            "1;3;5;28;750;1400;2300;28000;38000;85000;130000;550000;1300000;EUR",
            "2017006;VENDREDI;20/01/2017;1;2;14;27;44;7;11;"
            "0;4;7;31;820;1600;2600;31000;42000;95000;145000;620000;1450000;EUR",
        ]
        return header + "\n" + "\n".join(rows)

    def test_parses_fdj_style_zip_and_filters_to_star_era(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            archive_path = Path(tmp_dir) / "euromillions_201702.zip"
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, "w") as archive:
                archive.writestr("euromillions.csv", self._fdj_csv())
            archive_path.write_bytes(buffer.getvalue())

            frame = load_winner_counts(archive_paths=[archive_path], use_cache=False)

        self.assertEqual(len(frame), 2)  # 2015 draw filtered out
        self.assertTrue((frame["date"] >= "2016-09-24").all())
        self.assertEqual(frame.iloc[0]["main_numbers"], [7, 13, 21, 34, 48])
        self.assertEqual(frame.iloc[0]["bonus_numbers"], [3, 9])
        self.assertEqual(int(frame.iloc[0]["winners_rank_13"]), 1_300_000)

    def test_new_archive_merges_with_existing_cache(self):
        # Dropping only the NEWEST FDJ file must extend the cached history,
        # not replace it (the append-new-draws workflow).
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_path = Path(tmp_dir) / "winner_counts.csv"
            old_row = {
                "date": pd.Timestamp("2016-10-04"),
                "main_numbers": [2, 11, 24, 36, 45],
                "bonus_numbers": [4, 8],
                "scope": "europe",
            }
            for rank in RANK_TIERS:
                old_row[f"winners_rank_{rank}"] = 1000 * rank
            write_winner_counts_cache(pd.DataFrame([old_row]), cache_path)

            archive_path = Path(tmp_dir) / "euromillions_latest.zip"
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, "w") as archive:
                archive.writestr("euromillions.csv", self._fdj_csv())
            archive_path.write_bytes(buffer.getvalue())

            frame = load_winner_counts(
                archive_paths=[archive_path], use_cache=True, cache_path=cache_path
            )

            self.assertEqual(len(frame), 3)  # 1 cached + 2 era draws from the new zip
            self.assertEqual(str(frame["date"].min().date()), "2016-10-04")
            self.assertEqual(str(frame["date"].max().date()), "2017-01-20")
            # And the cache on disk was extended too.
            self.assertEqual(len(read_winner_counts_cache(cache_path)), 3)

    def test_plain_csv_with_unlabelled_winner_columns(self):
        winner_headers = ";".join(f"nombre_de_gagnant_au_rang{rank}" for rank in range(1, 14))
        text = (
            "date_de_tirage;boule_1;boule_2;boule_3;boule_4;boule_5;etoile_1;etoile_2;"
            f"{winner_headers}\n"
            "03/01/2020;3;9;22;31;47;4;8;"
            "0;2;4;25;700;1300;2100;26000;36000;80000;120000;500000;1200000\n"
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "euromillions_2020.csv"
            csv_path.write_text(text, encoding="utf-8")
            frame = load_winner_counts(archive_paths=[csv_path], use_cache=False)
        self.assertEqual(len(frame), 1)
        self.assertEqual(int(frame.iloc[0]["winners_rank_1"]), 0)


class FittedWeightsLoadingTests(unittest.TestCase):
    def tearDown(self):
        popularity.reload_weights()  # restore default state for other tests

    def test_reload_uses_fitted_file_and_falls_back_cleanly(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            fitted_path = Path(tmp_dir) / "popularity_fitted.json"
            main = np.linspace(1.4, 0.6, 50)
            payload = {
                "fitted_at_utc": "2026-06-10T00:00:00+00:00",
                "main_weights": (main / main.mean()).tolist(),
                "bonus_weights": np.ones(12).tolist(),
            }
            fitted_path.write_text(json.dumps(payload), encoding="utf-8")

            source = popularity.reload_weights(fitted_path)
            self.assertIn("fitted", source)
            self.assertGreater(popularity.MAIN_POPULARITY[0], popularity.MAIN_POPULARITY[-1])

            missing = popularity.reload_weights(Path(tmp_dir) / "missing.json")
            self.assertEqual(missing, "heuristic priors")

    def test_invalid_fitted_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            fitted_path = Path(tmp_dir) / "popularity_fitted.json"
            fitted_path.write_text(
                json.dumps({"main_weights": [1.0] * 10, "bonus_weights": [1.0] * 12}),
                encoding="utf-8",
            )
            source = popularity.reload_weights(fitted_path)
            self.assertEqual(source, "heuristic priors")


if __name__ == "__main__":
    unittest.main()
