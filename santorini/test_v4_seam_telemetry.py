import json
import os
import tempfile
import unittest

import numpy as np

from santorini.V4SeamTelemetry import (
    evaluate_loss_vectors,
    evaluate_seam_telemetry,
    load_seam_telemetry_suite,
    policies_to_csr,
)


class FixedPredictor:
    def __init__(self, policies, values):
        self.policies = np.asarray(policies, dtype=np.float32)
        self.values = np.asarray(values, dtype=np.float32)
        self.offset = 0

    def predict_batch(self, boards):
        start = self.offset
        self.offset += len(boards)
        return (
            self.policies[start:self.offset],
            self.values[start:self.offset],
        )


def make_suite():
    positions = 8
    targets = np.zeros((positions, 3), dtype=np.float32)
    targets[:, 0] = 1.0
    indptr, indices, values = policies_to_csr(targets)
    suite = {
        "boards": np.zeros((positions, 2, 5, 5), dtype=np.int8),
        "policy_indptr": indptr,
        "policy_indices": indices,
        "policy_values": values,
        "value_targets": np.zeros(positions, dtype=np.float32),
        "exposures": np.linspace(0.0, 1.0, positions, dtype=np.float32),
        "exposure_quartiles": np.repeat(np.arange(4, dtype=np.int8), 2),
        "metadata": np.asarray(json.dumps({
            "schema_version": 1,
            "baseline_checkpoint_sha256": "baseline-sha",
        })),
    }
    baseline_policies = np.tile(
        np.asarray([0.8, 0.1, 0.1], dtype=np.float32), (positions, 1)
    )
    baseline = evaluate_loss_vectors(
        FixedPredictor(baseline_policies, np.zeros(positions)), suite, batch_size=3
    )
    for name, metric_values in baseline.items():
        suite["baseline_{}".format(name)] = metric_values.astype(np.float32)
    return suite, baseline_policies


class TestV4SeamTelemetry(unittest.TestCase):
    def test_frozen_baseline_has_zero_contrast_delta(self):
        suite, baseline_policies = make_suite()
        metrics = evaluate_seam_telemetry(
            FixedPredictor(baseline_policies, np.zeros(8)),
            suite,
            batch_size=3,
            bootstrap_samples=100,
        )
        self.assertAlmostEqual(metrics["v4_seam_contrast_delta_from_baseline"], 0.0)
        self.assertAlmostEqual(metrics["v4_seam_objective_delta_from_baseline"], 0.0)
        self.assertFalse(metrics["v4_seam_warning"])
        self.assertFalse(metrics["v4_seam_confirmed_warning"])

    def test_high_exposure_regression_triggers_warning(self):
        suite, policies = make_suite()
        policies = policies.copy()
        policies[6:] = np.asarray([0.4, 0.3, 0.3], dtype=np.float32)
        metrics = evaluate_seam_telemetry(
            FixedPredictor(policies, np.zeros(8)),
            suite,
            batch_size=8,
            bootstrap_samples=100,
            alert_delta=0.02,
        )
        self.assertGreater(metrics["v4_seam_q4_objective_delta_from_baseline"], 0.1)
        self.assertAlmostEqual(metrics["v4_seam_q1_objective_delta_from_baseline"], 0.0)
        self.assertTrue(metrics["v4_seam_warning"])
        self.assertTrue(metrics["v4_seam_confirmed_warning"])

    def test_suite_round_trip_validates_and_fingerprints(self):
        suite, _ = make_suite()
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "suite.npz")
            np.savez_compressed(path, **suite)
            loaded = load_seam_telemetry_suite(path)
        self.assertEqual(loaded["fingerprint"].__len__(), 64)
        self.assertEqual(loaded["metadata_parsed"]["baseline_checkpoint_sha256"], "baseline-sha")
        np.testing.assert_array_equal(loaded["exposure_quartiles"], suite["exposure_quartiles"])


if __name__ == "__main__":
    unittest.main()
