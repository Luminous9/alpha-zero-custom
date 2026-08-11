import unittest

import numpy as np

from santorini.SantoriniGame import SantoriniGame
from santorini.V4Supervised import (
    _value_arrays,
    apply_d4_augmentation,
    blended_value_target,
    score_to_value,
    smooth_engine_policy,
)


class V4SupervisedTests(unittest.TestCase):
    def setUp(self):
        self.game = SantoriniGame(5, sequential_placement=True)

    def standard_board(self):
        board = np.zeros((2, 5, 5), dtype=int)
        board[0, 1, 1] = 1
        board[0, 3, 3] = 2
        board[0, 1, 3] = -1
        board[0, 3, 1] = -2
        return board

    def test_score_mapping_clamps_mates_and_is_antisymmetric(self):
        self.assertEqual(score_to_value(9_000), 1.0)
        self.assertEqual(score_to_value(-9_000), -1.0)
        self.assertAlmostEqual(score_to_value(400), -score_to_value(-400))

    def test_stage_aware_blend_downweights_early_scores(self):
        early = blended_value_target(-1.0, 1_000, 0, alpha_boot=0.5)
        late = blended_value_target(-1.0, 1_000, 2, alpha_boot=0.5)
        self.assertLess(early, late)
        self.assertEqual(blended_value_target(-1.0, 1_000, 2, alpha_boot=0.0), -1.0)

    def test_target_construction_rejects_invalid_calibration(self):
        with self.assertRaises(ValueError):
            score_to_value(10, temperature=0)
        with self.assertRaises(ValueError):
            blended_value_target(1.0, 10, 3)
        with self.assertRaises(ValueError):
            blended_value_target(1.0, 10, 0, stage_reliability=(1.0, 1.0))

    def test_placement_uses_completed_outcome_for_every_bootstrap_target(self):
        targets = _value_arrays(
            np.asarray([-0.25], dtype=np.float32),
            np.asarray([9_000], dtype=np.float32),
            np.asarray([-1], dtype=np.int8),
            0.5,
            (0.25, 0.75, 1.0),
            261.8,
        )
        for target in targets:
            self.assertAlmostEqual(float(target[0]), -0.25)

    def test_policy_smoothing_uses_only_legal_alternatives(self):
        board = self.standard_board()
        valids = self.game.getValidMoves(board, 1).astype(bool)
        teacher = np.zeros(self.game.getActionSize(), dtype=np.float32)
        teacher[np.flatnonzero(valids)[0]] = 1.0
        smoothed = smooth_engine_policy(self.game, board, teacher, 0.05)
        self.assertAlmostEqual(float(smoothed.sum()), 1.0, places=6)
        self.assertTrue(np.all(smoothed[~valids] == 0))
        self.assertAlmostEqual(float(smoothed[teacher > 0].sum()), 0.95, places=6)

    def test_d4_augmentation_matches_game_policy_permutation(self):
        board = self.standard_board()
        boards = np.asarray([[board[0] > 0] * 13], dtype=np.float32)
        policy = np.arange(self.game.getActionSize(), dtype=np.float32)
        transformed_boards, transformed_policies = apply_d4_augmentation(
            boards, policy[None, :], np.asarray([3]), self.game
        )
        expected_board = np.flip(np.rot90(boards[0], 1, axes=(-2, -1)), axis=-1)
        expected_policy = self.game._transform_policy(policy, 1, True)
        self.assertTrue(np.array_equal(transformed_boards[0], expected_board))
        self.assertTrue(np.array_equal(transformed_policies[0], expected_policy))

    def test_d4_augmentation_rejects_invalid_symmetry_id(self):
        board = self.standard_board()
        boards = np.asarray([[board[0] > 0] * 13], dtype=np.float32)
        policy = np.zeros((1, self.game.getActionSize()), dtype=np.float32)
        with self.assertRaises(ValueError):
            apply_d4_augmentation(boards, policy, np.asarray([8]), self.game)


if __name__ == "__main__":
    unittest.main()
