import unittest

from santorini.PlacementArenaAnalysis import (
    collect_d4_unique_placements,
    summarize_duplicate_aware_records,
)
from santorini.test_batched_arena import MiniPlacementGame, Uniform25NNet
from utils import dotdict


def game(pair, seat, score, exact, d4, full):
    return {
        'pair_index': pair,
        'seat_order': seat,
        'score': float(score),
        'exact_opening_key': exact,
        'd4_opening_key': d4,
        'exact_continuation_key': exact + '-' + seat,
        'full_trajectory_hash': full,
        'continuation_trajectory_hash': exact + '-continuation',
    }


class PlacementArenaAnalysisTests(unittest.TestCase):
    def test_collects_a_d4_unique_learned_opening_suite(self):
        suite, diagnostics = collect_d4_unique_placements(
            game=MiniPlacementGame(),
            controller=Uniform25NNet(),
            search_args=dotdict({'numMCTSSims': 1, 'cpuct': 1.0}),
            target_openings=1,
            batch_size=2,
            seed=29,
            max_occurrences=2,
            sample_batch_size=2,
            quiet=True,
        )

        self.assertEqual(len(suite), 1)
        self.assertEqual(diagnostics['target_d4_unique_openings'], 1)
        self.assertEqual(diagnostics['sampled_occurrences'], 2)
        self.assertGreaterEqual(diagnostics['target_reached_at_occurrence'], 1)
        self.assertGreaterEqual(diagnostics['unused_batch_tail_occurrences'], 0)
        self.assertEqual(len({record['d4_opening_key'] for record in suite}), 1)

    def test_normalized_cap_limits_a_dominant_d4_family(self):
        records = []
        for pair in range(20):
            first_d4 = 'dominant' if pair < 10 else 'first-{}'.format(pair)
            records.extend([
                game(
                    pair, 'contestant1_first', 1, 'a-{}'.format(pair),
                    first_d4, 'fa-{}'.format(pair),
                ),
                game(
                    pair, 'contestant2_first', 0, 'b-{}'.format(pair),
                    'second-{}'.format(pair), 'fb-{}'.format(pair),
                ),
            ])

        summary = summarize_duplicate_aware_records(
            records,
            seed=23,
            bootstrap_samples=1000,
            d4_cap_fraction=0.05,
        )

        self.assertTrue(summary['capped_d4']['cap_achievable'])
        self.assertLessEqual(
            summary['capped_d4']['maximum_weighted_group_share'], 0.05 + 1e-10
        )
        self.assertLess(
            summary['capped_d4']['score'],
            summary['raw_policy_weighted']['score'],
        )

    def test_reports_raw_unique_and_capped_views(self):
        records = [
            game(0, 'contestant1_first', 1, 'a', 'hot', 'f0'),
            game(0, 'contestant2_first', 1, 'b', 'hot', 'f1'),
            game(1, 'contestant1_first', 1, 'a', 'hot', 'f0'),
            game(1, 'contestant2_first', 1, 'c', 'hot', 'f2'),
            game(2, 'contestant1_first', 0, 'd', 'cold-a', 'f3'),
            game(2, 'contestant2_first', 0, 'e', 'cold-b', 'f4'),
        ]

        summary = summarize_duplicate_aware_records(
            records,
            seed=19,
            bootstrap_samples=1000,
            d4_cap_fraction=0.05,
            minimum_d4_ess_ratio=0.75,
        )

        self.assertAlmostEqual(summary['raw_policy_weighted']['score'], 2 / 3)
        self.assertLess(summary['capped_d4']['score'], 2 / 3)
        self.assertFalse(summary['capped_d4']['cap_achievable'])
        self.assertAlmostEqual(
            summary['capped_d4']['maximum_weighted_group_share'], 1 / 3
        )
        self.assertEqual(summary['diversity']['distinct_d4_openings'], 3)
        self.assertEqual(summary['diversity']['maximum_d4_opening_multiplicity'], 4)
        self.assertEqual(summary['diversity']['distinct_full_trajectories'], 5)
        self.assertEqual(summary['diversity']['literal_duplicate_full_trajectories'], 1)
        self.assertEqual(summary['diversity']['maximum_full_trajectory_multiplicity'], 2)
        self.assertEqual(
            summary['pair_outcomes'],
            {
                'candidate_2_0': 2,
                'split_1_1': 0,
                'candidate_0_2': 1,
                'other_with_draws': 0,
            },
        )
        self.assertEqual(summary['unique_exact_blocks']['units'], 3)
        self.assertFalse(summary['decision_checks']['natural_d4_diversity_pass'])


if __name__ == '__main__':
    unittest.main()
