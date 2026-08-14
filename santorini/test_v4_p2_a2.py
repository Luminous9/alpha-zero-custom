import tempfile
import unittest
from pathlib import Path

import numpy as np

from run_santorini_v4_p2_a2 import _board_sha256, _freeze_openings


class TestV4P2A2(unittest.TestCase):
    def test_frozen_openings_are_distinct_and_reusable(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            first, manifest = _freeze_openings(output, games=8, seed=20260815)
            second, reused = _freeze_openings(output, games=8, seed=20260815)

            np.testing.assert_array_equal(first, second)
            self.assertEqual(manifest, reused)
            self.assertEqual(len(first), 4)
            self.assertEqual(
                len({_board_sha256(board) for board in first}), len(first)
            )
            self.assertTrue(manifest["fresh_relative_to_longitudinal_suite"])

    def test_changed_contract_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            _freeze_openings(output, games=8, seed=20260815)
            with self.assertRaises(ValueError):
                _freeze_openings(output, games=10, seed=20260815)


if __name__ == "__main__":
    unittest.main()
