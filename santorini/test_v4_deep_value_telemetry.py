import json
import os
import tempfile
import unittest

import numpy as np

from santorini.V4DeepValueTelemetry import (
    evaluate_deep_value_telemetry,
    load_deep_value_telemetry_suite,
)


class FixedPredictor:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float32)
        self.offset = 0

    def predict_batch(self, boards):
        start = self.offset
        self.offset += len(boards)
        return np.zeros((len(boards), 1)), self.values[start:self.offset]


def make_suite():
    positions = 24
    labels = np.linspace(-0.9, 0.9, positions, dtype=np.float32)
    return {
        "schema_version": np.asarray([1], dtype=np.int16),
        "boards": np.zeros((positions, 2, 5, 5), dtype=np.int8),
        "oracle_values": labels,
        "reference_values": labels * 0.8,
        "stages": np.tile(np.asarray(["early", "middle", "late"]), 8),
        "bands": np.repeat(np.asarray([
            "windows_1_4", "windows_5_8", "windows_9_11", "windows_12_14"
        ]), 6),
        "position_hashes": np.asarray(["hash-{}".format(i) for i in range(positions)]),
        "metadata": np.asarray(json.dumps({
            "schema_version": 1,
            "reference_iteration": 11,
            "reference_checkpoint_sha256": "iter11",
        })),
    }


class DeepValueTelemetryTests(unittest.TestCase):
    def test_reference_is_zero_paired_drift(self):
        suite = make_suite()
        metrics = evaluate_deep_value_telemetry(
            FixedPredictor(suite["reference_values"]), suite,
            batch_size=5, bootstrap_samples=50,
        )
        self.assertAlmostEqual(metrics["v4_deep_value_overall_mse_delta"], 0.0)
        self.assertAlmostEqual(metrics["v4_deep_value_overall_pearson_delta"], 0.0)
        self.assertFalse(metrics["v4_deep_value_warning"])

    def test_recent_regression_warns(self):
        suite = make_suite()
        values = suite["reference_values"].copy()
        recent = suite["bands"] == "windows_9_11"
        values[recent] *= -1
        metrics = evaluate_deep_value_telemetry(
            FixedPredictor(values), suite, bootstrap_samples=50,
        )
        self.assertTrue(metrics["v4_deep_value_recent_pearson_warning"])
        self.assertTrue(metrics["v4_deep_value_warning"])

    def test_round_trip_validates_and_fingerprints(self):
        suite = make_suite()
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "suite.npz")
            np.savez_compressed(path, **suite)
            loaded = load_deep_value_telemetry_suite(path)
        self.assertEqual(len(loaded["fingerprint"]), 64)
        self.assertEqual(loaded["metadata_parsed"]["reference_iteration"], 11)


if __name__ == "__main__":
    unittest.main()
