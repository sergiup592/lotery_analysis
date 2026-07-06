import unittest

from src.coverage import CoverageConfig, CoverageOptimizer
from src.popularity import ticket_popularity


class PopularityTests(unittest.TestCase):
    def test_birthday_ticket_more_popular_than_high_numbers(self):
        birthday = ticket_popularity([3, 7, 12, 19, 27], [2, 7])
        unpopular = ticket_popularity([33, 38, 41, 44, 49], [10, 11])
        self.assertGreater(birthday, unpopular)

    def test_sequence_pattern_is_heavily_penalized(self):
        sequence = ticket_popularity([1, 2, 3, 4, 5], [1, 2])
        scattered = ticket_popularity([4, 18, 26, 35, 47], [3, 10])
        self.assertGreater(sequence, scattered + 0.4)

    def test_popularity_weight_steers_selection_away_from_popular_tickets(self):
        plain = CoverageOptimizer(seed=3, config=CoverageConfig(popularity_weight=0.0)).generate(5)
        steered = CoverageOptimizer(seed=3, config=CoverageConfig(popularity_weight=2.5)).generate(5)
        avg_plain = sum(t["popularity"] for t in plain) / len(plain)
        avg_steered = sum(t["popularity"] for t in steered) / len(steered)
        self.assertLessEqual(avg_steered, avg_plain)

    def test_disjoint_main_produces_non_overlapping_tickets(self):
        tickets = CoverageOptimizer(
            seed=11, config=CoverageConfig(disjoint_main=True, candidate_pool_size=400)
        ).generate(5)
        all_mains = [number for ticket in tickets for number in ticket["main_numbers"]]
        self.assertEqual(len(all_mains), len(set(all_mains)))


if __name__ == "__main__":
    unittest.main()
