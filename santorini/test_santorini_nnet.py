import unittest
import tempfile

import numpy as np

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
        nnet_args.epochs = 1
        nnet_args.batch_size = 2
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

    def test_resumable_checkpoint_restores_adam_state(self):
        old_epochs = nnet_args.epochs
        old_batch_size = nnet_args.batch_size
        nnet_args.epochs = 1
        nnet_args.batch_size = 1
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

    def test_mcts_can_use_santorini_network(self):
        mcts_args = dotdict({'numMCTSSims': 2, 'cpuct': 1.0})
        mcts = MCTS(self.game, self.nnet, mcts_args)
        board = self.game.getCanonicalForm(self.game.getInitBoard(), 1)

        probs = mcts.getActionProb(board, temp=1)

        self.assertEqual(len(probs), self.game.getActionSize())
        self.assertAlmostEqual(sum(probs), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
