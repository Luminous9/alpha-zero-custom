import unittest

import numpy as np

from santorini.SantoriniGame import SantoriniGame
from santorini.SantoriniSymmetryDiagnostics import (
    build_diagnostic_suite,
    evaluate_suite,
    transformations,
)


class SyntheticNNet:
    def __init__(self, game, value_function):
        self.game = game
        self.value_function = value_function

    def predict_batch(self, boards):
        policies = np.full(
            (len(boards), self.game.getActionSize()),
            1.0 / self.game.getActionSize(),
            dtype=np.float32,
        )
        values = np.asarray([self.value_function(board) for board in boards], dtype=np.float32)
        return policies, values


class TestSantoriniSymmetryDiagnostics(unittest.TestCase):
    def setUp(self):
        self.game = SantoriniGame(5, sequential_placement=True)

    def _standard_board(self):
        board = self.game.getInitBoard()
        player = 1
        for location in ((0, 1), (4, 3), (1, 3), (3, 1)):
            board, player = self.game.getNextState(
                board,
                player,
                self.game.getPlacementAction(location),
            )
        board = self.game.getCanonicalForm(board, player)
        board = board.copy()
        board[1, 0, 0] = 1
        return board

    def test_invariant_value_head_has_zero_orbit_error(self):
        board = self._standard_board()
        result = evaluate_suite(
            self.game,
            SyntheticNNet(self.game, lambda _: 0.25),
            [board],
            targets=[0.25],
        )
        metrics = result['aggregate']['early']

        self.assertEqual(metrics['positions'], 1)
        self.assertAlmostEqual(metrics['value_mean_orbit_std'], 0.0)
        self.assertAlmostEqual(metrics['value_mean_orbit_range'], 0.0)
        self.assertAlmostEqual(metrics['value_orientation_mse'], 0.0)
        self.assertAlmostEqual(metrics['value_ensemble_mse'], 0.0)
        self.assertAlmostEqual(metrics['value_symmetry_excess_mse'], 0.0)
        self.assertAlmostEqual(metrics['policy_mean_orbit_total_variation'], 0.0)

    def test_orientation_mse_decomposes_into_ensemble_and_symmetry_error(self):
        board = self._standard_board()
        nnet = SyntheticNNet(
            self.game,
            lambda candidate: float(candidate[1, 0, 0]) - 0.5,
        )
        result = evaluate_suite(self.game, nnet, [board], targets=[0.0])
        position = result['positions'][0]

        self.assertGreater(position['value_orbit_range'], 0.0)
        self.assertTrue(position['value_sign_disagreement'])
        self.assertAlmostEqual(
            position['value_orientation_mse'],
            position['value_ensemble_mse'] + position['value_symmetry_excess_mse'],
            places=7,
        )
        self.assertAlmostEqual(
            position['value_symmetry_excess_mse'],
            position['value_orbit_std'] ** 2,
            places=7,
        )

    def test_suite_deduplicates_d4_equivalent_boards_and_averages_targets(self):
        board = self._standard_board()
        rotated = transformations(board)[4][2]
        policy = np.zeros(self.game.getActionSize(), dtype=np.float32)
        boards, targets, buckets = build_diagnostic_suite(
            self.game,
            [(board, policy, 1.0), (rotated, policy, -1.0)],
            sample_size=8,
        )

        self.assertEqual(len(boards), 1)
        self.assertEqual(buckets, ['early'])
        self.assertAlmostEqual(targets[0], 0.0)


if __name__ == '__main__':
    unittest.main()
