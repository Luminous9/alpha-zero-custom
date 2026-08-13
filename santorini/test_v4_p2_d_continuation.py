import unittest

from prepare_santorini_v4_p2_d_continuation_bundle import CONFIGURATION
from run_santorini_v4_p2_d_continuation_kaggle import (
    _expected_beta,
    _validate_row,
)


class V4P2DContinuationTests(unittest.TestCase):
    @staticmethod
    def manifest():
        return {
            'configuration': CONFIGURATION,
            'protocol': {'games_per_iteration': 240, 'replay_reuse': 2.0},
            'lineage': {'teacher_objective_reference': 0.7517371261657554},
            'inputs': {
                'v4-seam-telemetry-suite.npz': {'sha256': 'seam'},
                'p1c-value-anchor.pth.tar': {'sha256': 'anchor'},
            },
        }

    @staticmethod
    def row(iteration, previous=0.7582732019323439, current=0.76):
        reference = 0.7517371261657554
        return {
            'iteration': iteration,
            'games': 240,
            'oracle_sparring_games': 24,
            'target_replay_reuse': 2.0,
            'replay_reuse_warmup_iters': 0,
            'v4_seam_suite_fingerprint': 'seam',
            'v4_teacher_objective_previous': previous,
            'v4_teacher_objective_current': current,
            'v4_teacher_objective_reference': reference,
            'v4_teacher_objective_step_threshold': 0.05,
            'v4_teacher_objective_cumulative_threshold': 0.10,
            'v4_teacher_objective_cumulative_delta': current - reference,
            'trunk_learning_rate': 1e-4,
            'policy_head_learning_rate': 1e-4,
            'value_head_learning_rate': 1e-4,
            'value_target_mode': 'p1c_anchor_to_outcome_z',
            'value_target_beta': _expected_beta(CONFIGURATION, iteration),
            'value_target_anchor_checkpoint_sha256': 'anchor',
        }

    def test_declared_beta_schedule_reaches_one_at_iteration_11(self):
        expected = {
            5: 0.5,
            6: 7.0 / 12.0,
            7: 2.0 / 3.0,
            8: 0.75,
            9: 5.0 / 6.0,
            10: 11.0 / 12.0,
            11: 1.0,
        }
        for iteration, beta in expected.items():
            self.assertAlmostEqual(
                _expected_beta(CONFIGURATION, iteration), beta
            )

    def test_continuation_row_validates_absolute_lineage(self):
        self.assertEqual(
            _validate_row(
                self.row(5), self.manifest(), 5, 0.7582732019323439
            ),
            0.76,
        )

    def test_changed_teacher_reference_is_rejected(self):
        row = self.row(5)
        row['v4_teacher_objective_reference'] = 0.7582732019323439
        with self.assertRaises(RuntimeError):
            _validate_row(
                row, self.manifest(), 5, 0.7582732019323439
            )


if __name__ == '__main__':
    unittest.main()
