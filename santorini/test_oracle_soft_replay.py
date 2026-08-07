import unittest

import numpy as np

from santorini.OracleResearch import (
    blend_policies,
    ranked_moves_to_v3_policy,
)
from santorini.SantoriniGame import SantoriniGame


class TestOracleSoftReplay(unittest.TestCase):
    def test_ranked_moves_map_scores_and_winning_aliases_to_policy(self):
        game = SantoriniGame(5, sequential_placement=True)
        board = np.zeros((2, 5, 5), dtype=int)
        board[0, 0, 0] = 1
        board[0, 1, 1] = 2
        board[0, 3, 3] = -1
        board[0, 4, 4] = -2
        board[1, 0, 0] = 2
        board[1, 0, 1] = 3
        moves = [
            {
                "score": 10_000,
                "actions": [
                    {"type": "select_worker", "value": "A1"},
                    {"type": "move_worker", "value": {"dest": "B1", "meta": None}},
                ],
            },
            {
                "score": 0,
                "actions": [
                    {"type": "select_worker", "value": "B2"},
                    {"type": "move_worker", "value": {"dest": "C2", "meta": None}},
                    {"type": "build", "value": "C1"},
                ],
            },
        ]
        policy = ranked_moves_to_v3_policy(game, board, moves, score_temperature=100)
        self.assertAlmostEqual(float(policy.sum()), 1.0, places=6)
        self.assertGreater(np.count_nonzero(policy), 1)
        self.assertGreater(float(policy.max()), 0.1)

    def test_blend_policies_is_normalized(self):
        source = np.asarray([1.0, 0.0], dtype=np.float32)
        oracle = np.asarray([0.0, 1.0], dtype=np.float32)
        np.testing.assert_allclose(blend_policies(source, oracle, 0.5), [0.5, 0.5])


if __name__ == "__main__":
    unittest.main()
