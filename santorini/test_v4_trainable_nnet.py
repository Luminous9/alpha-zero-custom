import os
import tempfile
import unittest

import numpy as np
import torch

from santorini.SantoriniGame import SantoriniGame
from santorini.pytorch.NNet import args as nnet_args, build_nnet
from santorini.pytorch.V4NNet import (
    V4InferenceWrapper,
    V4_SELECTED_CONFIG,
    build_v4_model,
)


class V4TrainableNNetTests(unittest.TestCase):
    def setUp(self):
        self.saved = {
            key: getattr(nnet_args, key)
            for key in (
                "cuda", "optimizer", "lr", "weight_decay",
                "v4_freeze_torchscript", "v4_autocast_fp16",
            )
        }
        nnet_args.cuda = False
        nnet_args.optimizer = "adamw"
        nnet_args.lr = 3e-4
        nnet_args.weight_decay = 1e-4
        nnet_args.v4_freeze_torchscript = False
        nnet_args.v4_autocast_fp16 = False
        self.game = SantoriniGame(5, sequential_placement=True)

    def tearDown(self):
        for key, value in self.saved.items():
            setattr(nnet_args, key, value)

    @staticmethod
    def board():
        board = np.zeros((2, 5, 5), dtype=np.int8)
        board[0, 0, 1] = 1
        board[0, 2, 1] = 2
        board[0, 3, 4] = -1
        board[0, 4, 2] = -2
        board[1, 1, 1] = 1
        board[1, 2, 3] = 2
        return board

    def _supervised_checkpoint(self, path):
        torch.manual_seed(19)
        model = build_v4_model(self.game, V4_SELECTED_CONFIG)
        torch.save({
            "schema_version": 2,
            "config": dict(V4_SELECTED_CONFIG),
            "state_dict": model.state_dict(),
            "epoch": 4,
        }, path)

    def test_supervised_checkpoint_migration_preserves_player_outputs(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "p1c.pth.tar")
            self._supervised_checkpoint(path)
            reference = V4InferenceWrapper(
                self.game,
                path,
                device="cpu",
                freeze_torchscript=False,
                canonicalize_d4=True,
            )
            trainable = build_nnet(self.game, "v4")
            metadata = trainable.load_checkpoint(folder, "p1c.pth.tar")
            boards = [self.board()]
            boards.extend(
                item[0]
                for item in self.game.getSymmetries(
                    self.board(), np.zeros(self.game.getActionSize())
                )[1:]
            )

            expected_policies, expected_values = reference.predict_batch(boards)
            actual_policies, actual_values = trainable.predict_batch(boards)

        self.assertEqual(metadata, {})
        np.testing.assert_array_equal(actual_policies, expected_policies)
        np.testing.assert_array_equal(actual_values, expected_values)

    def test_training_examples_collapse_to_one_canonical_frame(self):
        trainable = build_nnet(self.game, "v4")
        board = self.board()
        valids = np.flatnonzero(self.game.getValidMoves(board, 1))
        policy = np.zeros(self.game.getActionSize(), dtype=np.float32)
        policy[valids[:3]] = (0.2, 0.3, 0.5)
        transformed_board, transformed_policy = self.game.getSymmetries(
            board, policy
        )[5]

        encoded, policies, values = trainable._encode_training_examples([
            (board, policy, 1.0),
            (transformed_board, transformed_policy, -1.0),
        ])

        np.testing.assert_array_equal(encoded[0], encoded[1])
        np.testing.assert_allclose(policies[0], policies[1])
        np.testing.assert_array_equal(values, np.asarray([1.0, -1.0]))

    def test_v4_checkpoint_round_trip_includes_resume_metadata(self):
        first = build_nnet(self.game, "v4")
        with tempfile.TemporaryDirectory() as folder:
            first.save_checkpoint(
                folder,
                "latest-training.pth.tar",
                include_optimizer=True,
                metadata={"iteration": 3, "training_mode": "latest"},
            )
            second = build_nnet(self.game, "v4")
            metadata = second.load_checkpoint(
                folder, "latest-training.pth.tar", load_optimizer=True
            )
            first_policy, first_value = first.predict(self.board())
            second_policy, second_value = second.predict(self.board())

        self.assertEqual(metadata["iteration"], 3)
        np.testing.assert_array_equal(second_policy, first_policy)
        self.assertEqual(second_value, first_value)

    def test_v4_checkpoint_normalizes_cross_version_rng_state(self):
        first = build_nnet(self.game, "v4")
        with tempfile.TemporaryDirectory() as folder:
            first.save_checkpoint(
                folder,
                "latest-training.pth.tar",
                include_optimizer=True,
                metadata={"iteration": 4},
            )
            path = os.path.join(folder, "latest-training.pth.tar")
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            expected_state = checkpoint["torch_rng_state"].clone()
            checkpoint["torch_rng_state"] = expected_state.numpy().astype(np.int64)
            torch.save(checkpoint, path)

            torch.manual_seed(99)
            second = build_nnet(self.game, "v4")
            metadata = second.load_checkpoint(
                folder, "latest-training.pth.tar", load_optimizer=True
            )

        self.assertEqual(metadata["iteration"], 4)
        self.assertEqual(torch.get_rng_state().dtype, torch.uint8)
        self.assertEqual(torch.get_rng_state().device.type, "cpu")
        torch.testing.assert_close(torch.get_rng_state(), expected_state)


if __name__ == "__main__":
    unittest.main()
