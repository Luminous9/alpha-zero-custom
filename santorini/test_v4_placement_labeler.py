import unittest

import numpy as np

from label_santorini_v4_placement import (
    EXPECTED_ORBITS_BY_WORKER_COUNT,
    _add_observation,
    enumerate_placement_orbits,
)
from santorini.D4Canonical import transform_board
from santorini.SantoriniGame import SantoriniGame


class V4PlacementLabelerTests(unittest.TestCase):
    def setUp(self):
        self.game = SantoriniGame(5, sequential_placement=True)

    def test_exhaustive_prefix_counts(self):
        boards = enumerate_placement_orbits(self.game)
        counts = tuple(
            sum(np.count_nonzero(board[0]) == worker_count for board in boards)
            for worker_count in range(4)
        )
        self.assertEqual(counts, EXPECTED_ORBITS_BY_WORKER_COUNT)
        self.assertEqual(len(boards), 960)
        self.assertTrue(all(self.game.isPlacementPhase(board) for board in boards))

    def test_observation_aggregation_is_d4_invariant(self):
        board = enumerate_placement_orbits(self.game)[1]
        valid = np.flatnonzero(self.game.getValidMoves(board, 1))
        policy = np.zeros(self.game.getActionSize(), dtype=np.float64)
        policy[valid] = 1.0 / len(valid)
        transformed_board = transform_board(board, 1, True)
        transformed_policy = self.game._transform_policy_array(policy, 1, True)
        aggregates = {}
        _add_observation(aggregates, self.game, board, policy, 1.0, "fresh")
        _add_observation(
            aggregates,
            self.game,
            transformed_board,
            transformed_policy,
            -1.0,
            "replay",
        )
        self.assertEqual(len(aggregates), 1)
        record = next(iter(aggregates.values()))
        self.assertEqual(record["observation_count"], 2)
        self.assertEqual(record["fresh_count"], 1)
        self.assertEqual(record["replay_count"], 1)
        self.assertAlmostEqual(record["value_sum"], 0.0)
        np.testing.assert_allclose(record["policy_sum"] / 2, policy)


if __name__ == "__main__":
    unittest.main()
