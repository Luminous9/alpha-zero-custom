import os
import tempfile
import unittest

import numpy as np
import torch

from santorini.SantoriniGame import SantoriniGame
from santorini.V4Encoder import encode_v4_board, encode_v4_boards
from santorini.pytorch.V4Prototype import (
    D4RegularNetwork,
    D4SymmetrizedReference,
    transform_spatial,
)


class V4EncoderTests(unittest.TestCase):
    def setUp(self):
        self.game = SantoriniGame(5, sequential_placement=True)

    def board(self):
        board = np.zeros((2, 5, 5), dtype=int)
        board[0, 2, 2] = 1
        board[0, 4, 4] = 2
        board[0, 0, 0] = -1
        board[0, 0, 4] = -2
        board[1, 2, 2] = 2
        board[1, 1, 2] = 3
        board[1, 3, 3] = 4
        board[1, 4, 3] = 1
        return board

    def test_rule_derived_planes(self):
        encoded = encode_v4_board(self.board())
        self.assertEqual(encoded.shape, (13, 5, 5))
        self.assertEqual(encoded[7, 1, 2], 1.0)
        self.assertEqual(encoded[7].sum(), 1.0)
        self.assertEqual(encoded[9, 2, 2], 7.0 / 8.0)
        self.assertEqual(encoded[11, 3, 3], 0.0)
        self.assertTrue(np.all(encoded[12] == 10.0 / 40.0))

    def test_placement_derived_planes_are_zero_and_phase_is_defined(self):
        board = np.zeros((2, 5, 5), dtype=int)
        board[0, 1, 1] = 1
        board[0, 3, 3] = -1
        encoded = encode_v4_board(board)
        self.assertTrue(np.all(encoded[7:12] == 0))
        self.assertTrue(np.all(encoded[12] == 0.5))

    def test_every_plane_is_d4_covariant(self):
        board = self.board()
        dummy_policy = np.zeros(self.game.getActionSize())
        original = encode_v4_board(board)
        for (rotations, flip), (transformed_board, _) in zip(
            ((rotations, flip) for rotations in range(4) for flip in (False, True)),
            self.game.getSymmetries(board, dummy_policy),
        ):
            transformed = encode_v4_board(transformed_board)
            expected = np.asarray([
                np.fliplr(np.rot90(plane, rotations)) if flip
                else np.rot90(plane, rotations)
                for plane in original
            ])
            self.assertTrue(np.array_equal(transformed, expected))

    def test_batch_encoder_exactly_matches_scalar_reference(self):
        placement = np.zeros((2, 5, 5), dtype=np.int8)
        placement[0, 1, 1] = 1
        placement[0, 3, 3] = -1
        boards = np.asarray([
            self.board(),
            np.rot90(self.board(), axes=(-2, -1)),
            placement,
            np.zeros((2, 5, 5), dtype=np.int8),
        ])
        expected = np.asarray([encode_v4_board(board) for board in boards])
        actual = encode_v4_boards(boards)
        self.assertTrue(np.array_equal(actual, expected))


class V4PrototypeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.game = SantoriniGame(5, sequential_placement=True)
        self.model = D4SymmetrizedReference(
            self.game, channels=8, residual_blocks=1, value_hidden_size=16
        ).eval()

    def _transform_policy(self, policy, rotations, flip):
        _, new_indices = self.game.getPolicySymmetryPermutation(rotations, flip)
        transformed = torch.zeros_like(policy)
        transformed[:, torch.as_tensor(new_indices)] = policy
        return transformed

    def test_policy_equivariance_and_value_invariance(self):
        inputs = torch.randn(2, 13, 5, 5)
        with torch.no_grad():
            policy, value = self.model(inputs)
            policy = policy.exp()
            for rotations in range(4):
                for flip in (False, True):
                    transformed_policy, transformed_value = self.model(
                        transform_spatial(inputs, rotations, flip)
                    )
                    expected = self._transform_policy(policy, rotations, flip)
                    self.assertTrue(
                        torch.allclose(transformed_policy.exp(), expected, atol=2e-6, rtol=2e-6)
                    )
                    self.assertTrue(
                        torch.allclose(transformed_value, value, atol=2e-6, rtol=2e-6)
                    )

    def test_checkpoint_reload_and_torchscript_export(self):
        inputs = torch.randn(1, 13, 5, 5)
        with torch.no_grad():
            expected = self.model(inputs)
        restored = D4SymmetrizedReference(
            self.game, channels=8, residual_blocks=1, value_hidden_size=16
        ).eval()
        restored.load_state_dict(self.model.state_dict())
        with torch.no_grad():
            actual = restored(inputs)
        self.assertTrue(torch.equal(expected[0], actual[0]))
        self.assertTrue(torch.equal(expected[1], actual[1]))
        traced = torch.jit.trace(restored, inputs)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "v4-reference.pt")
            traced.save(path)
            loaded = torch.jit.load(path)
            with torch.no_grad():
                exported = loaded(inputs)
        self.assertTrue(torch.allclose(actual[0], exported[0], atol=1e-7, rtol=1e-7))
        self.assertTrue(torch.allclose(actual[1], exported[1], atol=1e-7, rtol=1e-7))


class V4RegularNetworkTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.game = SantoriniGame(5, sequential_placement=True)
        self.model = D4RegularNetwork(
            self.game,
            effective_channels=16,
            residual_blocks=2,
            value_hidden_size=16,
        ).eval()

    def _transform_policy(self, policy, rotations, flip):
        _, new_indices = self.game.getPolicySymmetryPermutation(rotations, flip)
        transformed = torch.zeros_like(policy)
        transformed[:, torch.as_tensor(new_indices)] = policy
        return transformed

    def test_optimized_tower_is_exactly_equivariant(self):
        inputs = torch.randn(2, 13, 5, 5)
        with torch.no_grad():
            policy, value, oracle_value = self.model.forward_with_auxiliary(inputs)
            for rotations in range(4):
                for flip in (False, True):
                    transformed_policy, transformed_value, transformed_oracle_value = (
                        self.model.forward_with_auxiliary(
                            transform_spatial(inputs, rotations, flip)
                        )
                    )
                    self.assertTrue(torch.allclose(
                        transformed_policy.exp(),
                        self._transform_policy(policy.exp(), rotations, flip),
                        atol=3e-6,
                        rtol=3e-6,
                    ))
                    self.assertTrue(torch.allclose(
                        transformed_value, value, atol=3e-6, rtol=3e-6
                    ))
                    self.assertTrue(torch.allclose(
                        transformed_oracle_value, oracle_value, atol=3e-6, rtol=3e-6
                    ))

    def test_expanded_export_matches_and_reloads(self):
        inputs = torch.randn(2, 13, 5, 5)
        exported = self.model.export_inference()
        with torch.no_grad():
            expected = self.model(inputs)
            actual = exported(inputs)
        self.assertTrue(torch.allclose(expected[0], actual[0], atol=1e-7, rtol=1e-7))
        self.assertTrue(torch.allclose(expected[1], actual[1], atol=1e-7, rtol=1e-7))
        traced = torch.jit.trace(exported, inputs)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "v4-regular-export.pt")
            traced.save(path)
            loaded = torch.jit.load(path)
            with torch.no_grad():
                reloaded = loaded(inputs)
        self.assertTrue(torch.allclose(actual[0], reloaded[0], atol=1e-7, rtol=1e-7))
        self.assertTrue(torch.allclose(actual[1], reloaded[1], atol=1e-7, rtol=1e-7))

        restored = D4RegularNetwork(
            self.game,
            effective_channels=16,
            residual_blocks=2,
            value_hidden_size=16,
        ).eval()
        restored.load_state_dict(self.model.state_dict())
        with torch.no_grad():
            checkpoint_output = restored(inputs)
        self.assertTrue(torch.equal(expected[0], checkpoint_output[0]))
        self.assertTrue(torch.equal(expected[1], checkpoint_output[1]))


if __name__ == "__main__":
    unittest.main()
