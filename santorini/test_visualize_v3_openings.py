import unittest

import numpy as np

from santorini.SantoriniGame import SantoriniGame
from visualize_santorini_v3_openings import (
    canonical_opening_key,
    rank_symmetry_unique_openings,
    transform_location,
)


class UniformPlacementNNet:
    def __init__(self, game):
        self.game = game

    def predict_batch(self, boards):
        policies = np.ones((len(boards), self.game.getActionSize()), dtype=np.float64)
        return policies, np.zeros(len(boards))


class TestVisualizeV3Openings(unittest.TestCase):
    def test_canonical_key_collapses_d4_variants(self):
        p1 = ((0, 0), (1, 2))
        p2 = ((3, 1), (4, 4))
        transformed_p1 = tuple(transform_location(x, 5, 1, True) for x in p1)
        transformed_p2 = tuple(transform_location(x, 5, 1, True) for x in p2)
        self.assertEqual(
            canonical_opening_key(p1, p2, 5),
            canonical_opening_key(transformed_p1, transformed_p2, 5),
        )

    def test_uniform_three_by_three_policy_produces_normalized_classes(self):
        game = SantoriniGame(3, sequential_placement=True)
        ranked = rank_symmetry_unique_openings(
            UniformPlacementNNet(game), board_size=3, batch_size=32
        )
        self.assertAlmostEqual(sum(row['probability'] for row in ranked), 1.0)
        self.assertTrue(all(row['probability'] > 0 for row in ranked))
        self.assertEqual([row['rank'] for row in ranked], list(range(1, len(ranked) + 1)))


if __name__ == '__main__':
    unittest.main()
