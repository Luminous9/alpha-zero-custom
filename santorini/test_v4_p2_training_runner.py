import json
from pathlib import Path
import tempfile
import unittest

from run_santorini_v4_p2_training_kaggle import (
    _bridge_beta,
    _snapshot,
    _validate_row,
)


class V4P2TrainingRunnerTests(unittest.TestCase):
    @staticmethod
    def manifest():
        return {
            'lineage': {'teacher_objective_reference': 0.75},
            'inputs': {
                'v4-seam-telemetry-suite.npz': {'sha256': 'seam'},
                'v4-deep-value-telemetry-suite.npz': {'sha256': 'deep'},
                'p1c-value-anchor.pth.tar': {'sha256': 'anchor'},
            },
        }

    @staticmethod
    def row(iteration, previous=0.76, current=0.765):
        bridge = iteration <= 11
        return {
            'iteration': iteration,
            'games': 240,
            'oracle_sparring_games': 24,
            'target_replay_reuse': 2.0,
            'replay_reuse_warmup_iters': 0,
            'standard_prior_target_kl_count': 1000,
            'standard_prior_target_kl_mean': 0.45,
            'prior_target_kl_warning_threshold': 0.15,
            'prior_target_kl_warning_iterations': 3,
            'v4_seam_suite_fingerprint': 'seam',
            'v4_deep_value_suite_fingerprint': 'deep',
            'v4_deep_value_suite_positions': 480,
            'v4_deep_value_warning_iterations': 2,
            'v4_teacher_objective_previous': previous,
            'v4_teacher_objective_current': current,
            'v4_teacher_objective_reference': 0.75,
            'v4_teacher_objective_step_threshold': 0.05,
            'v4_teacher_objective_cumulative_threshold': 0.10,
            'v4_teacher_objective_cumulative_delta': current - 0.75,
            'trunk_learning_rate': 1e-4,
            'policy_head_learning_rate': 1e-4,
            'value_head_learning_rate': 1e-4,
            'value_target_mode': (
                'p1c_anchor_to_outcome_z' if bridge else 'outcome_z'
            ),
            'value_target_beta': _bridge_beta(iteration) if bridge else 1.0,
            'value_target_anchor_checkpoint_sha256': (
                'anchor' if bridge else None
            ),
        }

    def test_bridge_boundary_and_post_bridge_rows_validate(self):
        self.assertAlmostEqual(_bridge_beta(11), 1.0)
        self.assertEqual(
            _validate_row(self.row(11), self.manifest(), 11, 0.76),
            0.765,
        )
        self.assertEqual(
            _validate_row(self.row(12), self.manifest(), 12, 0.76),
            0.765,
        )

    def test_post_bridge_row_cannot_silently_keep_bridge_mode(self):
        row = self.row(12)
        row['value_target_mode'] = 'p1c_anchor_to_outcome_z'
        with self.assertRaises(RuntimeError):
            _validate_row(row, self.manifest(), 12, 0.76)

    def test_row_requires_live_prior_target_telemetry(self):
        row = self.row(12)
        del row['standard_prior_target_kl_mean']
        with self.assertRaises(RuntimeError):
            _validate_row(row, self.manifest(), 12, 0.76)

    def test_snapshots_use_flat_iteration_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in (
                'latest-training.pth.tar',
                'latest.pth.tar',
                'latest.examples.npz',
            ):
                (root / name).write_bytes(name.encode())
            destinations = _snapshot(root, 12)
            self.assertEqual(
                sorted(path.name for path in destinations),
                [
                    'checkpoint_12-training.pth.tar',
                    'checkpoint_12.examples.npz',
                    'checkpoint_12.pth.tar',
                ],
            )

    def test_notebook_is_valid_and_exposes_routine_controls(self):
        notebook_path = Path(__file__).with_name('v4_p2_training_kaggle.ipynb')
        notebook = json.loads(notebook_path.read_text())
        self.assertEqual(notebook['nbformat'], 4)
        source = '\n'.join(
            ''.join(cell.get('source', [])) for cell in notebook['cells']
        )
        for control in (
            'NUM_ITERATIONS',
            'RUN_NAME',
            'SNAPSHOT_INTERVAL',
            'RESUME_CHECKPOINT',
            'RESUME_REPLAY',
            'SOURCE_MODE',
            'PACKAGE_OUTPUTS',
            'ENSURE_P100_COMPATIBLE_TORCH',
            'cu126',
            'sm_60',
            'coloredlogs',
            'import main_santorini',
            'prior_target_kl',
        ):
            self.assertIn(control, source)


if __name__ == '__main__':
    unittest.main()
