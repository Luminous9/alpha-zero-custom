import os
import tempfile
import unittest
from collections import deque

import numpy as np
import torch

from finetune_santorini_oracle import (
    configure_trainable_scope,
    decode_compact_examples,
    replay_stage,
    replace_values_with_source_predictions,
    select_rehearsal_indices,
    split_indices,
    teacher_base_indices,
)
from santorini.ReplayBuffer import save_compact_replay
from santorini.SantoriniGame import SantoriniGame
from santorini.pytorch.NNet import args as nnet_args
from santorini.pytorch.NNet import build_nnet


class TestOracleFinetune(unittest.TestCase):
    def setUp(self):
        self.game = SantoriniGame(5, sequential_placement=True)
        self.policy = np.zeros(self.game.getActionSize(), dtype=np.float32)
        self.policy[0] = 1.0

    def _board(self, stage):
        board = np.zeros((2, 5, 5), dtype=int)
        if stage == "placement":
            board[0, 0, 0] = 1
            return board
        board[0, 0, 0] = 1
        board[0, 0, 4] = 2
        board[0, 4, 0] = -1
        board[0, 4, 4] = -2
        builds = {"early": 2, "middle": 7, "late": 16}[stage]
        for index in range(builds):
            board[1, (index // 4) % 5, index % 4] += 1
        return board

    def test_teacher_base_indices_collapse_consecutive_symmetry_groups(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "teacher.npz")
            examples = [(self._board("early"), self.policy, 1.0) for _ in range(24)]
            save_compact_replay(path, [deque(examples)])
            self.assertEqual(teacher_base_indices(path, 8), [0, 8, 16])
            with self.assertRaises(ValueError):
                teacher_base_indices(path, 7)

    def test_rehearsal_selection_is_balanced_and_deterministic(self):
        examples = []
        for stage in ("placement", "early", "middle", "late"):
            examples.extend((self._board(stage), self.policy, 1.0) for _ in range(5))
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "source.npz")
            save_compact_replay(path, [deque(examples)])
            first = select_rehearsal_indices(path, 12, np.random.RandomState(4))
            second = select_rehearsal_indices(path, 12, np.random.RandomState(4))
            decoded = decode_compact_examples(path, first)

        self.assertEqual(first, second)
        self.assertEqual(
            {
                stage: sum(replay_stage(board) == stage for board, _, _ in decoded)
                for stage in ("placement", "early", "middle", "late")
            },
            {"placement": 3, "early": 3, "middle": 3, "late": 3},
        )

    def test_split_has_no_overlap_and_is_reproducible(self):
        first = split_indices(range(20), 0.2, np.random.RandomState(9))
        second = split_indices(range(20), 0.2, np.random.RandomState(9))
        self.assertEqual(first, second)
        train, validation = first
        self.assertEqual(len(train), 16)
        self.assertEqual(len(validation), 4)
        self.assertFalse(set(train) & set(validation))

    def test_source_predictions_replace_only_teacher_values(self):
        class FakeNNet:
            def predict_batch(self, boards):
                return None, np.asarray(
                    [float(np.sum(board[1])) / 10 for board in boards],
                    dtype=np.float32,
                )

        examples = [
            (self._board("early"), self.policy.copy(), -1.0),
            (self._board("middle"), self.policy.copy(), 1.0),
        ]
        replaced = replace_values_with_source_predictions(FakeNNet(), examples)
        self.assertAlmostEqual(replaced[0][2], 0.2)
        self.assertAlmostEqual(replaced[1][2], 0.7)
        np.testing.assert_array_equal(replaced[0][0], examples[0][0])
        np.testing.assert_array_equal(replaced[0][1], examples[0][1])

    def test_finetune_training_can_freeze_batch_norm_statistics(self):
        previous = {
            key: getattr(nnet_args, key)
            for key in ("cuda", "epochs", "batch_size", "max_train_steps",
                        "on_the_fly_symmetry", "freeze_batch_norm")
        }
        try:
            nnet_args.cuda = False
            nnet_args.epochs = 1
            nnet_args.batch_size = 2
            nnet_args.max_train_steps = 1
            nnet_args.on_the_fly_symmetry = False
            nnet_args.freeze_batch_norm = True
            nnet = build_nnet(self.game, "v3")
            batch_norm = next(
                module for module in nnet.nnet.modules()
                if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
            )
            before = batch_norm.running_mean.clone()
            board = self._board("early")
            valid = self.game.getValidMoves(board, 1)
            policy = valid.astype(np.float32) / valid.sum()
            nnet.train([(board, policy, 1.0), (board, policy, -1.0)])
            torch.testing.assert_close(batch_norm.running_mean, before)
        finally:
            for key, value in previous.items():
                setattr(nnet_args, key, value)

    def test_policy_head_scope_freezes_every_other_parameter(self):
        previous_cuda = nnet_args.cuda
        try:
            nnet_args.cuda = False
            nnet = build_nnet(self.game, "v3")
            trainable = configure_trainable_scope(nnet, "policy-head")
            self.assertTrue(trainable)
            self.assertTrue(all(name.startswith("policy_conv.") for name in trainable))
            self.assertTrue(all(
                parameter.requires_grad == name.startswith("policy_conv.")
                for name, parameter in nnet.nnet.named_parameters()
            ))
        finally:
            nnet_args.cuda = previous_cuda


if __name__ == "__main__":
    unittest.main()
