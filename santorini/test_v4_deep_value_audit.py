import unittest

import numpy as np

from audit_santorini_v4_deep_value import (
    _ranks,
    anonymous_d4_hash,
    nominal_score_value,
    paired_bootstrap_delta,
    value_metrics,
)


class TestV4DeepValueAudit(unittest.TestCase):
    def test_anonymous_hash_ignores_worker_labels_and_d4_orientation(self):
        pieces = np.zeros((5, 5), dtype=int)
        pieces[0, 0] = 1
        pieces[1, 1] = 2
        pieces[3, 3] = -1
        pieces[4, 4] = -2
        heights = np.arange(25, dtype=int).reshape(5, 5) % 5
        board = np.asarray([pieces, heights])
        relabeled = board.copy()
        relabeled[0] *= 2
        transformed = np.asarray([
            np.fliplr(np.rot90(relabeled[0], 1)),
            np.fliplr(np.rot90(relabeled[1], 1)),
        ])

        self.assertEqual(anonymous_d4_hash(board), anonymous_d4_hash(transformed))

    def test_nominal_score_value_has_expected_symmetry_and_mate_band(self):
        self.assertEqual(nominal_score_value(0), 0.0)
        self.assertAlmostEqual(nominal_score_value(400), -nominal_score_value(-400))
        self.assertEqual(nominal_score_value(9_000), 1.0)
        self.assertEqual(nominal_score_value(-9_000), -1.0)

    def test_tied_ranks_and_value_metrics(self):
        np.testing.assert_allclose(_ranks([3, 1, 1, 2]), [4, 1.5, 1.5, 3])
        metrics = value_metrics([0.0, 0.5, 1.0], [0.0, 0.5, 1.0])
        self.assertAlmostEqual(metrics["pearson"], 1.0)
        self.assertAlmostEqual(metrics["spearman"], 1.0)
        self.assertEqual(metrics["mse"], 0.0)
        self.assertEqual(metrics["mae"], 0.0)

    def test_paired_bootstrap_reports_candidate_minus_reference(self):
        oracle = np.asarray([-0.8, -0.2, 0.2, 0.8])
        reference = np.asarray([-0.1, 0.1, -0.1, 0.1])
        candidate = oracle.copy()
        strata = np.asarray(["a", "a", "b", "b"])
        result = paired_bootstrap_delta(
            candidate, reference, oracle, strata, samples=100, seed=7
        )
        self.assertLess(result["mse"]["delta"], 0)
        self.assertLess(result["mae"]["delta"], 0)
        self.assertGreater(result["pearson"]["delta"], 0)


if __name__ == "__main__":
    unittest.main()
