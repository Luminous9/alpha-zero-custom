import unittest
import tempfile
from unittest.mock import patch

import numpy as np
import torch

from MCTS import MCTS
from santorini.SantoriniGame import SantoriniGame
from santorini.pytorch.NNet import NNetWrapper, V3NNetWrapper, args as nnet_args, build_nnet
from utils import dotdict


class TestSantoriniNNet(unittest.TestCase):
    def setUp(self):
        np.random.seed(11)
        self.game = SantoriniGame(5)
        self.nnet = NNetWrapper(self.game)

    def test_encode_board_uses_anonymous_worker_and_height_planes(self):
        board = np.zeros((2, 5, 5), dtype=int)
        board[0, 1, 1] = 1
        board[0, 2, 2] = 2
        board[0, 3, 3] = -1
        board[0, 4, 4] = -2
        board[1, 0, 0] = 1
        board[1, 0, 1] = 2
        board[1, 0, 2] = 3
        board[1, 0, 3] = 4

        encoded = NNetWrapper.encode_board(board)

        self.assertEqual(encoded.shape, (6, 5, 5))
        self.assertEqual(encoded[0, 1, 1], 1)
        self.assertEqual(encoded[0, 2, 2], 1)
        self.assertEqual(encoded[1, 3, 3], 1)
        self.assertEqual(encoded[1, 4, 4], 1)
        self.assertEqual(encoded[2, 0, 0], 1)
        self.assertEqual(encoded[3, 0, 1], 1)
        self.assertEqual(encoded[4, 0, 2], 1)
        self.assertEqual(encoded[5, 0, 3], 1)

    def test_encode_board_ignores_same_color_worker_labels(self):
        board = np.zeros((2, 5, 5), dtype=int)
        board[0, 1, 1] = 1
        board[0, 2, 2] = 2
        board[0, 3, 3] = -1
        board[0, 4, 4] = -2

        swapped = board.copy()
        swapped[0, 1, 1] = 2
        swapped[0, 2, 2] = 1
        swapped[0, 3, 3] = -2
        swapped[0, 4, 4] = -1

        np.testing.assert_array_equal(
            NNetWrapper.encode_board(board),
            NNetWrapper.encode_board(swapped),
        )

    def test_predict_returns_policy_and_scalar_value(self):
        board = self.game.getCanonicalForm(self.game.getInitBoard(), 1)

        pi, v = self.nnet.predict(board)

        self.assertEqual(pi.shape, (self.game.getActionSize(),))
        self.assertAlmostEqual(float(pi.sum()), 1.0, places=5)
        self.assertIsInstance(v, float)
        self.assertGreaterEqual(v, -1.0)
        self.assertLessEqual(v, 1.0)

    def test_predict_batch_returns_policy_and_values(self):
        boards = [
            self.game.getCanonicalForm(self.game.getInitBoard(), 1),
            self.game.getCanonicalForm(self.game.getInitBoard(), 1),
        ]

        pis, vs = self.nnet.predict_batch(boards)

        self.assertEqual(pis.shape, (2, self.game.getActionSize()))
        self.assertEqual(vs.shape, (2,))
        for pi in pis:
            self.assertAlmostEqual(float(pi.sum()), 1.0, places=5)
        self.assertTrue(np.all(vs >= -1.0))
        self.assertTrue(np.all(vs <= 1.0))

    def test_build_nnet_can_select_v2_and_v3_architectures(self):
        v2 = build_nnet(self.game, 'v2')
        v3_game = SantoriniGame(5, sequential_placement=True)
        v3 = build_nnet(v3_game, 'v3')

        self.assertIsInstance(v2, NNetWrapper)
        self.assertIsInstance(v3, V3NNetWrapper)
        self.assertEqual(v2.net_args.num_residual_blocks, 5)
        self.assertEqual(v2.net_args.num_channels, 64)
        self.assertEqual(v3.net_args.num_residual_blocks, 8)
        self.assertEqual(v3.net_args.num_channels, 96)
        self.assertEqual(v3.net_args.policy_channels, 65)
        self.assertEqual(v2.action_size, self.game.getActionSize())
        self.assertEqual(v3.action_size, v3_game.getActionSize())

        pi, _ = v3.predict(v3_game.getInitBoard())
        self.assertEqual(pi.shape, (1625,))

    def test_checkpoint_metadata_rejects_wrong_architecture(self):
        v3 = V3NNetWrapper(SantoriniGame(5, sequential_placement=True))

        with tempfile.TemporaryDirectory() as folder:
            v3.save_checkpoint(folder, 'v3.pth.tar')

            with self.assertRaisesRegex(ValueError, 'Checkpoint architecture "v3"'):
                self.nnet.load_checkpoint(folder, 'v3.pth.tar')

    def test_cuda_rng_restore_tolerates_fewer_runtime_devices(self):
        saved_states = [torch.tensor([1], dtype=torch.uint8), torch.tensor([2], dtype=torch.uint8)]
        with patch.object(torch.cuda, 'device_count', return_value=1), patch.object(
            torch.cuda, 'set_rng_state'
        ) as set_rng_state:
            with self.assertLogs('santorini.pytorch.NNet', level='WARNING'):
                NNetWrapper._restore_cuda_rng_states(saved_states)

        set_rng_state.assert_called_once_with(saved_states[0], device=0)

    def test_v2_policy_adapter_scatter_preserves_square_blocks(self):
        placement_game = SantoriniGame(5, sequential_placement=True)
        v2 = NNetWrapper(placement_game)
        native = np.arange(1600, dtype=np.float32).reshape(1, 1600)

        adapted = v2._adapt_native_policies(native).reshape(25, 65)

        np.testing.assert_array_equal(adapted[:, :64], native.reshape(25, 64))
        np.testing.assert_array_equal(adapted[:, 64], np.zeros(25))

    def test_single_training_step_runs(self):
        old_epochs = nnet_args.epochs
        old_batch_size = nnet_args.batch_size
        old_max_train_steps = nnet_args.max_train_steps
        old_on_the_fly_symmetry = nnet_args.on_the_fly_symmetry
        nnet_args.epochs = 1
        nnet_args.batch_size = 2
        nnet_args.max_train_steps = None
        nnet_args.on_the_fly_symmetry = False
        try:
            examples = []
            for _ in range(2):
                board = self.game.getCanonicalForm(self.game.getInitBoard(), 1)
                valids = self.game.getValidMoves(board, 1).astype(np.float32)
                pi = valids / valids.sum()
                examples.append((board, pi, 1))

            self.nnet.train(examples)
        finally:
            nnet_args.epochs = old_epochs
            nnet_args.batch_size = old_batch_size
            nnet_args.max_train_steps = old_max_train_steps
            nnet_args.on_the_fly_symmetry = old_on_the_fly_symmetry

    def test_training_step_cap_preserves_small_replay_epoch_limit(self):
        old_epochs = nnet_args.epochs
        old_batch_size = nnet_args.batch_size
        old_max_train_steps = nnet_args.max_train_steps
        old_on_the_fly_symmetry = nnet_args.on_the_fly_symmetry
        nnet_args.epochs = 3
        nnet_args.batch_size = 2
        nnet_args.max_train_steps = 4
        nnet_args.on_the_fly_symmetry = False
        try:
            board = self.game.getCanonicalForm(self.game.getInitBoard(), 1)
            valids = self.game.getValidMoves(board, 1).astype(np.float32)
            examples = [(board, valids / valids.sum(), 1.0) for _ in range(6)]

            with patch.object(self.nnet.optimizer, 'step', wraps=self.nnet.optimizer.step) as step:
                metrics = self.nnet.train(examples)

            self.assertEqual(step.call_count, 4)
            self.assertEqual(metrics['training_steps'], 4)
            self.assertEqual(metrics['uncapped_training_steps'], 9)
            self.assertAlmostEqual(metrics['effective_replay_epochs'], 4 * 2 / 6)
            self.assertEqual(metrics['epoch'], 2)
        finally:
            nnet_args.epochs = old_epochs
            nnet_args.batch_size = old_batch_size
            nnet_args.max_train_steps = old_max_train_steps
            nnet_args.on_the_fly_symmetry = old_on_the_fly_symmetry

    def test_fresh_data_reuse_controls_step_budget_independently_of_replay_size(self):
        old_epochs = nnet_args.epochs
        old_batch_size = nnet_args.batch_size
        old_max_train_steps = nnet_args.max_train_steps
        old_replay_reuse = nnet_args.replay_reuse
        old_on_the_fly_symmetry = nnet_args.on_the_fly_symmetry
        nnet_args.epochs = 3
        nnet_args.batch_size = 8
        nnet_args.max_train_steps = None
        nnet_args.replay_reuse = 4.0
        nnet_args.on_the_fly_symmetry = True
        try:
            board = self.game.getCanonicalForm(self.game.getInitBoard(), 1)
            valids = self.game.getValidMoves(board, 1).astype(np.float32)
            examples = [(board, valids / valids.sum(), 1.0) for _ in range(20)]

            with patch.object(self.nnet.optimizer, 'step', wraps=self.nnet.optimizer.step) as step:
                metrics = self.nnet.train(examples, new_example_count=10, iteration=12)

            self.assertEqual(step.call_count, 5)
            self.assertEqual(metrics['uncapped_training_steps'], 5)
            self.assertEqual(metrics['fresh_training_examples'], 10)
            self.assertEqual(metrics['target_replay_reuse'], 4.0)
            self.assertEqual(metrics['actual_replay_reuse'], 4.0)
            self.assertEqual(metrics['base_replay_epochs'], 2.0)
            self.assertEqual(metrics['training_schedule'], 'fresh-data-reuse')
            self.assertEqual(metrics['training_segments_completed'], 3)
            self.assertEqual(metrics['final_segment_policy_loss'], metrics['policy_loss'])
            self.assertEqual(metrics['final_segment_value_loss'], metrics['value_loss'])
            self.assertAlmostEqual(
                metrics['iteration_total_loss'],
                metrics['iteration_policy_loss'] + metrics['iteration_value_loss'],
            )
        finally:
            nnet_args.epochs = old_epochs
            nnet_args.batch_size = old_batch_size
            nnet_args.max_train_steps = old_max_train_steps
            nnet_args.replay_reuse = old_replay_reuse
            nnet_args.on_the_fly_symmetry = old_on_the_fly_symmetry

    def test_validation_metrics_report_placement_and_standard_phases(self):
        game = SantoriniGame(5, sequential_placement=True)
        nnet = V3NNetWrapper(game)
        placement_board = game.getInitBoard()
        standard_board = placement_board
        player = 1
        for location in ((0, 0), (4, 4), (0, 4), (4, 0)):
            standard_board, player = game.getNextState(
                standard_board,
                player,
                game.getPlacementAction(location),
            )
        standard_board = game.getCanonicalForm(standard_board, player)
        examples = []
        for board, value in ((placement_board, 1.0), (standard_board, -1.0)):
            policy = game.getValidMoves(board, 1).astype(np.float32)
            policy /= policy.sum()
            examples.append((board, policy, value))

        metrics = nnet._validation_metrics(examples)

        self.assertEqual(metrics['validation_examples'], 2)
        self.assertEqual(metrics['placement_validation_examples'], 1)
        self.assertEqual(metrics['standard_validation_examples'], 1)
        self.assertIn('placement_validation_policy_kl', metrics)
        self.assertIn('standard_validation_value_loss', metrics)
        self.assertIn('validation_total_loss', metrics)

    def test_iteration_learning_rate_schedule_uses_absolute_iteration(self):
        runtime_args = dotdict({
            'lr': 3e-4,
            'lr_schedule': [(200, 1e-4), (400, 3e-5)],
        })

        self.assertEqual(self.nnet._learning_rate_for_iteration(199, runtime_args), 3e-4)
        self.assertEqual(self.nnet._learning_rate_for_iteration(200, runtime_args), 1e-4)
        self.assertEqual(self.nnet._learning_rate_for_iteration(450, runtime_args), 3e-5)

    def test_adamw_and_weight_decay_are_configurable(self):
        old_optimizer = nnet_args.optimizer
        old_weight_decay = nnet_args.weight_decay
        nnet_args.optimizer = 'adamw'
        nnet_args.weight_decay = 1e-4
        try:
            nnet = NNetWrapper(self.game)
            self.assertIsInstance(nnet.optimizer, torch.optim.AdamW)
            self.assertEqual(nnet.optimizer.param_groups[0]['weight_decay'], 1e-4)
        finally:
            nnet_args.optimizer = old_optimizer
            nnet_args.weight_decay = old_weight_decay

    def test_on_the_fly_symmetry_matches_game_transforms(self):
        game = SantoriniGame(5, sequential_placement=True)
        nnet = V3NNetWrapper(game)
        board = game.getInitBoard()
        player = 1
        for location in ((0, 0), (1, 1), (2, 2), (3, 3)):
            board, player = game.getNextState(board, player, game.getPlacementAction(location))
        board = game.getCanonicalForm(board, player)
        policy = game.getValidMoves(board, 1).astype(np.float32)
        policy /= policy.sum()

        encoded = nnet.encode_board(board)
        encoded_boards = np.repeat(encoded[None, ...], 8, axis=0)
        policies = np.repeat(policy[None, ...], 8, axis=0)
        transformed_boards, transformed_policies = nnet._apply_symmetries(
            encoded_boards,
            policies,
            np.arange(8),
        )

        for symmetry_id, (expected_board, expected_policy) in enumerate(game.getSymmetries(board, policy)):
            np.testing.assert_array_equal(
                transformed_boards[symmetry_id],
                nnet.encode_board(expected_board),
            )
            np.testing.assert_allclose(transformed_policies[symmetry_id], expected_policy)

    def test_symmetry_consistency_losses_map_policies_back_to_source_coordinates(self):
        game = SantoriniGame(5, sequential_placement=True)
        nnet = V3NNetWrapper(game)
        generator = torch.Generator().manual_seed(31)
        primary_logits = torch.randn(
            8,
            game.getActionSize(),
            generator=generator,
        )
        primary_log_policies = torch.log_softmax(primary_logits, dim=1)
        encoded = nnet.encode_board(game.getInitBoard())
        encoded_boards = np.repeat(encoded[None, ...], 8, axis=0)
        transformed_boards, transformed_log_policies = nnet._apply_symmetries(
            encoded_boards,
            primary_log_policies.detach().numpy(),
            np.arange(8),
        )
        del transformed_boards
        primary_values = torch.linspace(-0.5, 0.5, 8)

        policy_js, value_mse = nnet._symmetry_consistency_losses(
            primary_log_policies,
            primary_values,
            torch.from_numpy(transformed_log_policies),
            primary_values.clone(),
            np.arange(8),
        )

        np.testing.assert_allclose(policy_js.detach().numpy(), 0.0, atol=1e-7)
        np.testing.assert_allclose(value_mse.detach().numpy(), 0.0, atol=1e-7)

    def test_v3_training_reports_phase_aware_symmetry_consistency(self):
        old_values = {
            key: getattr(nnet_args, key)
            for key in (
                'epochs',
                'batch_size',
                'max_train_steps',
                'replay_reuse',
                'on_the_fly_symmetry',
                'symmetry_consistency_fraction',
                'symmetry_consistency_policy_weight',
                'symmetry_consistency_value_weight',
            )
        }
        nnet_args.epochs = 1
        nnet_args.batch_size = 4
        nnet_args.max_train_steps = 1
        nnet_args.replay_reuse = None
        nnet_args.on_the_fly_symmetry = True
        nnet_args.symmetry_consistency_fraction = 0.5
        nnet_args.symmetry_consistency_policy_weight = 0.05
        nnet_args.symmetry_consistency_value_weight = 0.05
        try:
            game = SantoriniGame(5, sequential_placement=True)
            nnet = V3NNetWrapper(game)
            board = game.getInitBoard()
            policy = game.getValidMoves(board, 1).astype(np.float32)
            policy /= policy.sum()

            metrics = nnet.train([
                (board, policy, 1.0),
                (board, policy, -1.0),
            ])

            self.assertEqual(metrics['training_steps'], 1)
            self.assertEqual(metrics['symmetry_consistency_examples'], 2)
            self.assertEqual(metrics['placement_symmetry_consistency_examples'], 2)
            self.assertEqual(metrics['standard_symmetry_consistency_examples'], 0)
            self.assertGreaterEqual(metrics['symmetry_consistency_policy_js'], 0.0)
            self.assertGreaterEqual(metrics['symmetry_consistency_value_mse'], 0.0)
            self.assertAlmostEqual(
                metrics['iteration_total_loss'],
                metrics['iteration_policy_loss']
                + metrics['iteration_value_loss']
                + metrics['symmetry_consistency_weighted_loss'],
            )
        finally:
            for key, value in old_values.items():
                setattr(nnet_args, key, value)

    def test_on_the_fly_step_budget_uses_virtual_symmetry_examples(self):
        old_epochs = nnet_args.epochs
        old_batch_size = nnet_args.batch_size
        old_max_train_steps = nnet_args.max_train_steps
        old_on_the_fly_symmetry = nnet_args.on_the_fly_symmetry
        nnet_args.epochs = 3
        nnet_args.batch_size = 2
        nnet_args.max_train_steps = 4
        nnet_args.on_the_fly_symmetry = True
        try:
            game = SantoriniGame(5, sequential_placement=True)
            nnet = V3NNetWrapper(game)
            board = game.getInitBoard()
            policy = game.getValidMoves(board, 1).astype(np.float32)
            policy /= policy.sum()

            metrics = nnet.train([(board, policy, 1.0), (board, policy, -1.0)])

            self.assertEqual(metrics['training_steps'], 4)
            self.assertEqual(metrics['virtual_replay_examples'], 16)
            self.assertEqual(metrics['symmetry_augmentation_multiplier'], 8)
            self.assertEqual(metrics['uncapped_training_steps'], 24)
            self.assertAlmostEqual(metrics['effective_replay_epochs'], 0.5)
            self.assertAlmostEqual(metrics['average_draws_per_stored_position'], 4.0)
        finally:
            nnet_args.epochs = old_epochs
            nnet_args.batch_size = old_batch_size
            nnet_args.max_train_steps = old_max_train_steps
            nnet_args.on_the_fly_symmetry = old_on_the_fly_symmetry

    def test_resumable_checkpoint_restores_adam_state(self):
        old_epochs = nnet_args.epochs
        old_batch_size = nnet_args.batch_size
        old_max_train_steps = nnet_args.max_train_steps
        old_on_the_fly_symmetry = nnet_args.on_the_fly_symmetry
        nnet_args.epochs = 1
        nnet_args.batch_size = 1
        nnet_args.max_train_steps = None
        nnet_args.on_the_fly_symmetry = False
        try:
            board = self.game.getCanonicalForm(self.game.getInitBoard(), 1)
            valids = self.game.getValidMoves(board, 1).astype(np.float32)
            self.nnet.train([(board, valids / valids.sum(), 1.0)])
            self.assertTrue(self.nnet.optimizer.state)

            with tempfile.TemporaryDirectory() as folder:
                self.nnet.save_checkpoint(folder, 'resume.pth.tar', include_optimizer=True)
                restored = NNetWrapper(self.game)
                restored.load_checkpoint(folder, 'resume.pth.tar', load_optimizer=True)

            self.assertEqual(len(restored.optimizer.state), len(self.nnet.optimizer.state))
        finally:
            nnet_args.epochs = old_epochs
            nnet_args.batch_size = old_batch_size
            nnet_args.max_train_steps = old_max_train_steps
            nnet_args.on_the_fly_symmetry = old_on_the_fly_symmetry

    def test_mcts_can_use_santorini_network(self):
        mcts_args = dotdict({'numMCTSSims': 2, 'cpuct': 1.0})
        mcts = MCTS(self.game, self.nnet, mcts_args)
        board = self.game.getCanonicalForm(self.game.getInitBoard(), 1)

        probs = mcts.getActionProb(board, temp=1)

        self.assertEqual(len(probs), self.game.getActionSize())
        self.assertAlmostEqual(sum(probs), 1.0, places=5)

    def test_mcts_accepts_per_move_simulation_cap_and_noise_override(self):
        mcts_args = dotdict({
            'numMCTSSims': 8,
            'cpuct': 1.0,
            'addDirichletNoise': True,
            'dirichletAlpha': 0.3,
            'dirichletEpsilon': 0.25,
        })
        mcts = MCTS(self.game, self.nnet, mcts_args)
        board = self.game.getCanonicalForm(self.game.getInitBoard(), 1)

        with patch.object(mcts, 'search') as search, patch.object(
            mcts,
            'add_root_noise',
        ) as add_root_noise:
            probabilities = mcts.getActionProb(
                board,
                temp=1,
                num_simulations=3,
                add_root_noise=False,
            )

        self.assertEqual(search.call_count, 3)
        add_root_noise.assert_not_called()
        self.assertAlmostEqual(sum(probabilities), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
