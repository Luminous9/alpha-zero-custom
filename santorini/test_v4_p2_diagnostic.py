import unittest

from arena_santorini_v4_p2_arm import _paired_current_statistics
from prepare_santorini_v4_p2_diagnostic_bundle import ARMS
from run_santorini_v4_p2_diagnostic_kaggle import _validate_row


class V4P2DiagnosticTests(unittest.TestCase):
    def manifest(self):
        return {
            'protocol': {'games_per_iteration': 240, 'replay_reuse': 2.0},
            'inputs': {
                'v4-seam-telemetry-suite.npz': {'sha256': 'seam'},
                'p1c-value-anchor.pth.tar': {'sha256': 'anchor'},
            },
            'arms': ARMS,
        }

    @staticmethod
    def row(arm, iteration=2):
        config = ARMS[arm]
        beta = 1.0 if arm != 'D' else 0.25
        return {
            'iteration': iteration,
            'games': 240,
            'oracle_sparring_games': 24,
            'target_replay_reuse': 2.0,
            'replay_reuse_warmup_iters': 0,
            'v4_seam_suite_fingerprint': 'seam',
            'v4_teacher_objective_previous': 0.75,
            'v4_teacher_objective_current': 0.76,
            'v4_teacher_objective_step_threshold': 0.05,
            'v4_teacher_objective_cumulative_threshold': 0.10,
            'trunk_learning_rate': config['trunk_learning_rate'],
            'policy_head_learning_rate': config['policy_head_learning_rate'],
            'value_head_learning_rate': config['value_head_learning_rate'],
            'value_target_mode': config['value_target_mode'],
            'value_target_beta': beta,
            'value_target_anchor_checkpoint_sha256': 'anchor' if arm == 'D' else None,
        }

    def test_frozen_arm_rows_validate(self):
        for arm in ARMS:
            self.assertEqual(
                _validate_row(self.row(arm), self.manifest(), arm, 2, 0.75),
                0.76,
            )

    def test_arm_d_beta_advances_by_absolute_iteration(self):
        row = self.row('D', iteration=4)
        row['value_target_beta'] = 0.25 + 2 * (0.75 / 9)
        _validate_row(row, self.manifest(), 'D', 4, 0.75)

    def test_paired_statistics_are_from_current_contestant_perspective(self):
        records = [
            {'pair_index': 0, 'winner': -1},
            {'pair_index': 0, 'winner': -1},
            {'pair_index': 1, 'winner': 1},
            {'pair_index': 1, 'winner': -1},
        ]
        result = _paired_current_statistics(records, seed=7, bootstrap_samples=100)
        self.assertEqual(result['pair_wins'], 1)
        self.assertEqual(result['pair_splits'], 1)
        self.assertEqual(result['pair_losses'], 0)
        self.assertEqual(result['current_game_score'], 0.75)


if __name__ == '__main__':
    unittest.main()
