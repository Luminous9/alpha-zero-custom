import unittest

import numpy as np

from diagnose_santorini_v4_canonical_seams import quartile_buckets, seam_profile
from santorini.D4Canonical import canonicalize_board
from santorini.SantoriniGame import SantoriniGame


class TestV4CanonicalSeams(unittest.TestCase):
    def setUp(self):
        self.game = SantoriniGame(5, sequential_placement=True)

    def test_quartile_buckets_are_stable_and_balanced(self):
        exposures = np.asarray([0.4, 0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.6])
        buckets = quartile_buckets(exposures)
        np.testing.assert_array_equal(np.bincount(buckets, minlength=4), [2, 2, 2, 2])
        self.assertTrue(np.all(exposures[buckets == 0] <= exposures[buckets == 1]))
        self.assertTrue(np.all(exposures[buckets == 1] <= exposures[buckets == 2]))
        self.assertTrue(np.all(exposures[buckets == 2] <= exposures[buckets == 3]))

    def test_seam_profile_enumerates_canonical_successors(self):
        board = np.zeros((2, 5, 5), dtype=np.int8)
        board[0, 0, 0] = 1
        board[0, 1, 1] = 2
        board[0, 3, 3] = -1
        board[0, 4, 4] = -2
        board[1, 0, 1] = 1
        board[1, 2, 3] = 2
        canonical, _, _ = canonicalize_board(board)
        profile = seam_profile(self.game, canonical)
        self.assertGreater(profile["legal_actions"], 0)
        self.assertGreater(profile["unique_successors"], 0)
        self.assertLessEqual(
            profile["unique_successors"], profile["legal_actions"]
        )
        self.assertGreaterEqual(profile["frame_switch_exposure"], 0.0)
        self.assertLessEqual(profile["frame_switch_exposure"], 1.0)


if __name__ == "__main__":
    unittest.main()
