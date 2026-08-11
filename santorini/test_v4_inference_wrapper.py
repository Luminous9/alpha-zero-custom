import os
import tempfile
import unittest

import numpy as np
import torch

from santorini.D4Canonical import (
    D4_TRANSFORMS,
    canonicalize_board,
    canonicalize_boards,
    restore_canonical_policies,
    restore_canonical_policy,
)
from santorini.SantoriniGame import SantoriniGame
from santorini.pytorch.V4NNet import V4InferenceWrapper, build_v4_model


class V4InferenceWrapperTests(unittest.TestCase):
    def setUp(self):
        self.game = SantoriniGame(5, sequential_placement=True)
        self.config = {
            "name": "test_equivariant",
            "architecture": "equivariant",
            "planes": 13,
            "target": "stage_blend",
            "effective_channels": 16,
            "residual_blocks": 1,
        }

    def board(self):
        board = np.zeros((2, 5, 5), dtype=np.int8)
        board[0, 1, 1] = 1
        board[0, 3, 3] = 2
        board[0, 1, 3] = -1
        board[0, 3, 1] = -2
        return board

    def test_checkpoint_adapter_predicts_single_and_batch(self):
        torch.manual_seed(7)
        model = build_v4_model(self.game, self.config)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "candidate.pth.tar")
            torch.save({
                "schema_version": 2,
                "config": self.config,
                "state_dict": model.state_dict(),
            }, path)
            wrapper = V4InferenceWrapper(self.game, path, device="cpu")
            policy, value = wrapper.predict(self.board())
            policies, values = wrapper.predict_batch([self.board(), self.board()])
        self.assertEqual(policy.shape, (self.game.getActionSize(),))
        self.assertEqual(policies.shape, (2, self.game.getActionSize()))
        self.assertEqual(values.shape, (2,))
        self.assertAlmostEqual(float(policy.sum()), 1.0, places=5)
        self.assertTrue(np.allclose(policies[0], policy))
        self.assertAlmostEqual(float(values[0]), value, places=6)

    def test_empty_batch_and_cpu_fp16_contract(self):
        model = build_v4_model(self.game, self.config)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "candidate.pth.tar")
            torch.save({"config": self.config, "state_dict": model.state_dict()}, path)
            wrapper = V4InferenceWrapper(
                self.game, path, device="cpu", freeze_torchscript=False
            )
            policies, values = wrapper.predict_batch([])
            self.assertEqual(policies.shape, (0, self.game.getActionSize()))
            self.assertEqual(values.shape, (0,))
            with self.assertRaises(ValueError):
                V4InferenceWrapper(self.game, path, device="cpu", autocast_fp16=True)

    def test_ordinary_checkpoint_uses_declared_width_and_depth(self):
        config = {
            "name": "test_ordinary_nondefault_shape",
            "architecture": "ordinary",
            "planes": 13,
            "target": "global_blend",
            "channels": 16,
            "residual_blocks": 1,
        }
        model = build_v4_model(self.game, config)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "ordinary.pth.tar")
            torch.save({"config": config, "state_dict": model.state_dict()}, path)
            wrapper = V4InferenceWrapper(
                self.game, path, device="cpu", freeze_torchscript=False
            )
            policy, value = wrapper.predict(self.board())
        self.assertEqual(len(model.residual_blocks), 1)
        self.assertEqual(model.stem[0].out_channels, 16)
        self.assertEqual(policy.shape, (self.game.getActionSize(),))
        self.assertTrue(np.isfinite(value))

    def test_ordinary_canonical_inference_is_d4_equivariant(self):
        config = {
            "name": "test_ordinary_canonical",
            "architecture": "ordinary",
            "planes": 13,
            "target": "global_blend",
            "channels": 16,
            "residual_blocks": 1,
        }
        torch.manual_seed(19)
        model = build_v4_model(self.game, config)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "ordinary.pth.tar")
            torch.save({"config": config, "state_dict": model.state_dict()}, path)
            wrapper = V4InferenceWrapper(
                self.game,
                path,
                device="cpu",
                freeze_torchscript=False,
                canonicalize_d4=True,
            )
            policy, value = wrapper.predict(self.board())
            dummy = np.zeros(self.game.getActionSize(), dtype=np.float32)
            for rotations in range(4):
                for flip in (False, True):
                    transformed_board, _ = self.game.getSymmetries(
                        self.board(), dummy
                    )[2 * rotations + int(flip)]
                    transformed_policy, transformed_value = wrapper.predict(
                        transformed_board
                    )
                    expected = self.game._transform_policy_array(
                        policy, rotations, flip
                    )
                    self.assertTrue(np.allclose(
                        transformed_policy, expected, atol=2e-7, rtol=2e-7
                    ))
                    self.assertAlmostEqual(transformed_value, value, places=7)

    def test_batched_canonicalization_matches_scalar_reference(self):
        empty = np.zeros((2, 5, 5), dtype=np.int8)
        boards = np.asarray([
            self.board(),
            np.rot90(self.board(), axes=(-2, -1)),
            np.flip(self.board(), axis=-1),
            empty,
        ])
        canonical, matching_masks, keys = canonicalize_boards(boards)
        scalar = [canonicalize_board(board) for board in boards]
        self.assertTrue(np.array_equal(
            canonical, np.asarray([item[0] for item in scalar])
        ))
        self.assertEqual(keys, [item[2] for item in scalar])
        expected_masks = np.asarray([
            [transform in item[1] for transform in D4_TRANSFORMS]
            for item in scalar
        ])
        self.assertTrue(np.array_equal(matching_masks, expected_masks))

        rng = np.random.default_rng(17)
        policies = rng.random(
            (len(boards), self.game.getActionSize()), dtype=np.float32
        )
        restored = restore_canonical_policies(
            self.game, policies, matching_masks
        )
        expected = np.asarray([
            restore_canonical_policy(self.game, policy, item[1])
            for policy, item in zip(policies, scalar)
        ])
        self.assertTrue(np.array_equal(restored, expected))

    def test_canonical_inference_projects_a_position_stabilizer(self):
        config = {
            "name": "test_ordinary_stabilizer",
            "architecture": "ordinary",
            "planes": 13,
            "target": "global_blend",
            "channels": 16,
            "residual_blocks": 1,
        }
        torch.manual_seed(23)
        model = build_v4_model(self.game, config)
        empty = np.zeros((2, 5, 5), dtype=np.int8)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "ordinary.pth.tar")
            torch.save({"config": config, "state_dict": model.state_dict()}, path)
            wrapper = V4InferenceWrapper(
                self.game,
                path,
                device="cpu",
                freeze_torchscript=False,
                canonicalize_d4=True,
            )
            policy, _ = wrapper.predict(empty)
        for rotations in range(4):
            for flip in (False, True):
                transformed = self.game._transform_policy_array(
                    policy, rotations, flip
                )
                self.assertTrue(np.allclose(
                    transformed, policy, atol=2e-7, rtol=2e-7
                ))
        self.assertAlmostEqual(float(policy.sum()), 1.0, places=6)

    def test_equivariant_checkpoint_rejects_redundant_canonicalization(self):
        model = build_v4_model(self.game, self.config)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "candidate.pth.tar")
            torch.save({"config": self.config, "state_dict": model.state_dict()}, path)
            with self.assertRaises(ValueError):
                V4InferenceWrapper(
                    self.game,
                    path,
                    device="cpu",
                    canonicalize_d4=True,
                )

    def test_canonical_frame_cache_is_bounded_and_reports_hits(self):
        config = {
            "name": "test_ordinary_canonical_cache",
            "architecture": "ordinary",
            "planes": 13,
            "target": "global_blend",
            "channels": 16,
            "residual_blocks": 1,
        }
        model = build_v4_model(self.game, config)
        board = self.board()
        rotated = np.rot90(board, axes=(-2, -1)).copy()
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "ordinary.pth.tar")
            torch.save({"config": config, "state_dict": model.state_dict()}, path)
            wrapper = V4InferenceWrapper(
                self.game,
                path,
                device="cpu",
                freeze_torchscript=False,
                canonicalize_d4=True,
                canonical_cache_size=2,
            )
            first_policies, first_values = wrapper.predict_batch([board, rotated])
            second_policies, second_values = wrapper.predict_batch([board, rotated])

        self.assertTrue(np.array_equal(first_policies, second_policies))
        self.assertTrue(np.array_equal(first_values, second_values))
        self.assertEqual(wrapper.canonical_cache_info(), {
            "size": 2,
            "capacity": 2,
            "hits": 2,
            "misses": 2,
        })

    def test_on_device_policy_restoration_matches_numpy_reference(self):
        config = {
            "name": "test_ordinary_device_restore",
            "architecture": "ordinary",
            "planes": 13,
            "target": "global_blend",
            "channels": 16,
            "residual_blocks": 1,
        }
        model = build_v4_model(self.game, config)
        empty = np.zeros((2, 5, 5), dtype=np.int8)
        boards = np.asarray([
            self.board(),
            np.rot90(self.board(), axes=(-2, -1)),
            np.flip(self.board(), axis=-1),
            empty,
        ])
        _, matching_masks, _ = canonicalize_boards(boards)
        rng = np.random.default_rng(31)
        policies = rng.random(
            (len(boards), self.game.getActionSize()), dtype=np.float32
        )
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "ordinary.pth.tar")
            torch.save({"config": config, "state_dict": model.state_dict()}, path)
            wrapper = V4InferenceWrapper(
                self.game,
                path,
                device="cpu",
                freeze_torchscript=False,
                canonicalize_d4=True,
            )
            actual = wrapper._restore_canonical_policies(
                torch.from_numpy(policies), matching_masks
            ).numpy()
        expected = restore_canonical_policies(
            self.game, policies, matching_masks
        )
        self.assertTrue(np.array_equal(actual, expected))


if __name__ == "__main__":
    unittest.main()
