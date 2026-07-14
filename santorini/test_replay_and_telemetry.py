from collections import deque
import os
import tempfile
import unittest

import numpy as np

from santorini.ReplayBuffer import load_compact_replay, save_compact_replay
from santorini.SantoriniGame import SantoriniGame
from santorini.SantoriniTelemetry import ReferenceSuite


class FixedNetwork:
    def __init__(self, game):
        self.game = game

    def predict_batch(self, boards):
        policies = []
        for board in boards:
            valids = self.game.getValidMoves(board, 1).astype(np.float32)
            policies.append(valids / valids.sum())
        return np.asarray(policies), np.zeros(len(boards), dtype=np.float32)


class TestReplayAndTelemetry(unittest.TestCase):
    def test_compact_replay_round_trip(self):
        board = np.zeros((2, 5, 5), dtype=int)
        policy = np.zeros(1625, dtype=np.float32)
        policy[[64, 129]] = [0.25, 0.75]
        history = [deque([(board, policy, 1.0)]), deque([(board + 1, policy, -1.0)])]

        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, 'latest.examples.npz')
            save_compact_replay(path, history)
            loaded = load_compact_replay(path)

        self.assertEqual([len(window) for window in loaded], [1, 1])
        for expected_window, actual_window in zip(history, loaded):
            expected = expected_window[0]
            actual = actual_window[0]
            np.testing.assert_array_equal(actual[0], expected[0])
            np.testing.assert_allclose(actual[1], expected[1])
            self.assertEqual(actual[2], expected[2])

    def test_v2_reference_policy_evaluates_against_v3_action_space(self):
        game = SantoriniGame(5, sequential_placement=True)
        board = game.getInitBoard()
        player = 1
        for location in ((0, 0), (1, 1), (2, 2), (3, 3)):
            board, player = game.getNextState(board, player, game.getPlacementAction(location))
        canonical = game.getCanonicalForm(board, player)
        v3_valids = game.getValidMoves(canonical, 1).astype(np.float32)
        v2_policy = v3_valids.reshape(25, 65)[:, :64].reshape(1600)
        v2_policy /= v2_policy.sum()

        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, 'reference.npz')
            with open(path, 'wb') as suite_file:
                np.savez_compressed(
                    suite_file,
                    boards=np.asarray([canonical], dtype=np.int8),
                    policies=np.asarray([v2_policy], dtype=np.float32),
                    values=np.asarray([0.0], dtype=np.float32),
                    stages=np.asarray([0], dtype=np.int8),
                )
            metrics = ReferenceSuite(path).evaluate(game, FixedNetwork(game))

        self.assertAlmostEqual(metrics['reference_legal_policy_mass'], 1.0, places=6)
        self.assertAlmostEqual(metrics['reference_top1_accuracy'], 1.0, places=6)


if __name__ == '__main__':
    unittest.main()
