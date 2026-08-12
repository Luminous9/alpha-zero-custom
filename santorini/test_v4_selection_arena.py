import unittest
from types import SimpleNamespace

from arena_santorini_v4_selection import _search_args


class V4SelectionArenaTests(unittest.TestCase):
    def test_asymmetric_simulation_override_preserves_search_contract(self):
        args = SimpleNamespace(
            simulations=96,
            search_mode="gumbel",
            gumbel_scale=0.0,
            placement_gumbel_scale=1.5,
            inference_cache_size=4096,
        )
        player1 = _search_args(args, 1, 72, 1)
        player2 = _search_args(args, 8, 128, 8)
        self.assertEqual(player1.numMCTSSims, 72)
        self.assertEqual(player2.numMCTSSims, 128)
        self.assertEqual(player1.searchMode, player2.searchMode)
        self.assertEqual(player1.gumbelPlacementScale, 1.5)
        self.assertFalse(player1.searchSymmetryEvaluation)
        self.assertEqual(player1.placementRootSymmetrySamples, 1)
        self.assertEqual(player2.rootSymmetrySamples, 8)
        self.assertEqual(player2.placementRootSymmetrySamples, 8)


if __name__ == "__main__":
    unittest.main()
