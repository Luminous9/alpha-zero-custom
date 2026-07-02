import unittest

import numpy as np

from santorini.SantoriniGame import SantoriniGame
from santorini.pytorch.LegacyNNet import LegacyNNetWrapper


class TestSantoriniLegacyAdapter(unittest.TestCase):
    def setUp(self):
        self.game = SantoriniGame(5)

    def test_legacy_policy_is_translated_to_v2_action_space(self):
        board = np.zeros((2, 5, 5), dtype=int)
        board[0, 1, 1] = 1
        board[0, 3, 3] = 2
        board[0, 0, 0] = -1
        board[0, 4, 4] = -2

        legacy_policy = np.zeros(128, dtype=np.float32)
        legacy_policy[7] = 0.4
        legacy_policy[64 + 9] = 0.6

        adapter = LegacyNNetWrapper(self.game)
        translated = adapter.translate_policy(board, legacy_policy)

        self.assertEqual(translated.shape, (self.game.getActionSize(),))
        self.assertAlmostEqual(float(translated.sum()), 1.0)
        self.assertEqual(translated[self.game.getActionFromOrigin((1, 1), 0, 7)], 0.4)
        self.assertEqual(translated[self.game.getActionFromOrigin((3, 3), 1, 1)], 0.6)

        nonzero_origins = {
            self.game.decodeAction(action)[0]
            for action in np.flatnonzero(translated)
        }
        self.assertEqual(nonzero_origins, {(1, 1), (3, 3)})


if __name__ == "__main__":
    unittest.main()
