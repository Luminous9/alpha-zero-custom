import unittest

import numpy as np

from diagnose_santorini_v4_replay_signal import (
    _exact_generators,
    _metric_arrays,
    _summarize_metrics,
)


class ThreeActionGame:
    def getValidMoves(self, board, player):
        return np.asarray([1, 1, 0], dtype=np.int8)


class V4ReplaySignalTests(unittest.TestCase):
    def test_metrics_renormalize_prior_over_legal_actions(self):
        boards = np.zeros((1, 2, 5, 5), dtype=np.int8)
        targets = np.asarray([[0.25, 0.75, 0.0]], dtype=np.float64)
        # Half of the network mass is illegal; the legal conditional prior is
        # nevertheless exactly the search target.
        priors = np.asarray([[0.125, 0.375, 0.5]], dtype=np.float64)
        metrics = _metric_arrays(
            ThreeActionGame(), boards, targets, priors,
            np.asarray([0.25]), np.asarray([1.0]),
        )
        summary = _summarize_metrics(metrics)

        self.assertAlmostEqual(summary["kl_target_prior"]["mean"], 0.0)
        self.assertAlmostEqual(summary["total_variation"]["mean"], 0.0)
        self.assertEqual(summary["argmax_agreement"]["mean"], 1.0)
        self.assertAlmostEqual(summary["value_squared_error_z"]["mean"], 0.5625)

    def test_exact_generator_requires_declared_checkpoint(self):
        self.assertEqual(
            _exact_generators(["1=p1c", "2=iter1"], {"p1c", "iter1"}),
            {1: "p1c", 2: "iter1"},
        )
        with self.assertRaises(ValueError):
            _exact_generators(["3=missing"], {"p1c"})


if __name__ == "__main__":
    unittest.main()
