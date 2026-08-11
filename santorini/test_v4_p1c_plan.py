import unittest

import numpy as np

from build_santorini_v4_p1c_plan import (
    constrained_joint_quotas,
    validate_placement_outcomes,
)


class V4P1cPlanTests(unittest.TestCase):
    def test_constrained_quotas_preserve_coverage_and_marginals(self):
        base = np.asarray([
            [219416, 639985, 2461002],
            [9669, 100843, 735159],
            [1900, 3550, 4550],
        ])
        rows = np.asarray([5476773, 1564792, 71127])
        columns = np.asarray([1422539, 2489442, 3200711])
        result = constrained_joint_quotas(base, rows, columns)
        self.assertTrue(np.all(result >= base))
        np.testing.assert_array_equal(result.sum(axis=1), rows)
        np.testing.assert_array_equal(result.sum(axis=0), columns)

    def test_impossible_coverage_is_rejected(self):
        with self.assertRaises(ValueError):
            constrained_joint_quotas(
                np.eye(3, dtype=np.int64) * 10,
                np.asarray([5, 20, 20]),
                np.asarray([15, 15, 15]),
            )

    def test_oracle_only_component_without_outcomes_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "completed continuation outcomes"):
            validate_placement_outcomes({
                "has_completed_outcomes": np.asarray([False]),
                "winner_means": np.asarray([0.0]),
            })

    def test_declared_completed_outcomes_are_accepted(self):
        validate_placement_outcomes({
            "has_completed_outcomes": np.asarray([True]),
            "winner_means": np.asarray([-1.0, 0.25, 1.0]),
        })


if __name__ == "__main__":
    unittest.main()
