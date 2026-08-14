import math
import unittest

import numpy as np

from Coach import Coach


class _Game:
    @staticmethod
    def isPlacementPhase(board):
        return bool(board[0])


class _MCTS:
    def __init__(self, prior):
        self.prior = prior

    def getRawPriorFromTree(self, board):
        return self.prior


class PriorTargetTelemetryTests(unittest.TestCase):
    def setUp(self):
        self.coach = Coach.__new__(Coach)
        self.coach.game = _Game()
        self.coach._prior_target_stats = Coach._newPriorTargetStats()

    def test_records_kl_tv_and_stage_separately(self):
        self.coach._recordPriorTargetSignal(
            _MCTS(np.asarray([0.5, 0.5])),
            np.asarray([False]),
            np.asarray([1.0, 0.0]),
        )
        telemetry = self.coach._priorTargetTelemetry()
        self.assertEqual(telemetry['standard_prior_target_kl_count'], 1)
        self.assertAlmostEqual(
            telemetry['standard_prior_target_kl_mean'], math.log(2.0)
        )
        self.assertAlmostEqual(
            telemetry['standard_prior_target_total_variation_mean'], 0.5
        )
        self.assertEqual(
            telemetry['standard_prior_target_argmax_agreement_mean'], 1.0
        )
        self.assertNotIn('placement_prior_target_kl_mean', telemetry)

    def test_exact_tactical_target_is_counted_but_excluded(self):
        self.coach._recordPriorTargetSignal(
            _MCTS(np.asarray([0.5, 0.5])),
            np.asarray([True]),
            np.asarray([1.0, 0.0]),
            exact_tactical=True,
        )
        telemetry = self.coach._priorTargetTelemetry()
        self.assertEqual(
            telemetry['placement_prior_target_exact_tactical_excluded'], 1
        )
        self.assertNotIn('placement_prior_target_kl_mean', telemetry)

    def test_low_signal_watch_requires_three_consecutive_iterations(self):
        self.coach.args = {
            'priorTargetKLWarningThreshold': 0.15,
            'priorTargetKLWarningMinPositions': 256,
            'priorTargetKLWarningIterations': 3,
        }
        self.coach._prior_target_stats['standard']['kl'] = [0.10] * 300
        self.coach._prior_target_kl_warning_streak = 0
        self.coach._prior_target_kl_reference = None
        self.coach._v4_teacher_previous_objective = None
        self.coach._oracle_sparring_stats = {'game_records': []}
        self.coach._oracle_sparring_pair_score_history = []
        first = self.coach._iterationControlMetrics(12, {})
        second = self.coach._iterationControlMetrics(13, {})
        third = self.coach._iterationControlMetrics(14, {})
        self.assertFalse(first['prior_target_kl_warning'])
        self.assertFalse(second['prior_target_kl_warning'])
        self.assertTrue(third['prior_target_kl_warning'])
        self.assertEqual(third['prior_target_kl_warning_streak'], 3)


if __name__ == '__main__':
    unittest.main()
