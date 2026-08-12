from collections import deque
import os
import tempfile
import unittest

import numpy as np

from santorini.ReplayBuffer import (
    collapse_compact_replay_symmetries,
    load_compact_replay,
    save_compact_replay,
    trim_compact_replay,
)
from santorini.SantoriniGame import SantoriniGame
from santorini.SantoriniTelemetry import ReferenceSuite, resolve_reference_suite_path


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
    def test_reference_suite_directory_resolves_single_npz(self):
        with tempfile.TemporaryDirectory() as folder:
            nested = os.path.join(folder, 'dataset')
            os.makedirs(nested)
            path = os.path.join(nested, 'v2_reference_500.npz')
            with open(path, 'wb') as suite_file:
                np.savez_compressed(suite_file, placeholder=np.asarray([1]))

            self.assertEqual(resolve_reference_suite_path(folder), path)

    def test_reference_suite_directory_rejects_ambiguous_npz_files(self):
        with tempfile.TemporaryDirectory() as folder:
            for filename in ('first.npz', 'second.npz'):
                with open(os.path.join(folder, filename), 'wb') as suite_file:
                    np.savez_compressed(suite_file, placeholder=np.asarray([1]))

            with self.assertRaisesRegex(ValueError, 'Multiple .npz files'):
                resolve_reference_suite_path(folder)

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

    def test_compact_replay_preserves_source_metadata(self):
        board = np.zeros((2, 5, 5), dtype=int)
        policy = np.zeros(1625, dtype=np.float32)
        policy[64] = 1.0
        metadata = {
            'source': 'oracle_sparring',
            'oracle_nodes': 100000,
            'oracle_ladder_version': 1,
            'neural_seat': 1,
            'stage': 'middle',
        }
        history = [deque([(board, policy, -1.0, metadata)])]

        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, 'latest.examples.npz')
            save_compact_replay(path, history)
            loaded = load_compact_replay(path)

        self.assertEqual(loaded[0][0][3], metadata)

    def test_compact_replay_still_loads_v1_without_metadata(self):
        board = np.zeros((1, 2, 5, 5), dtype=np.int8)
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, 'legacy.examples.npz')
            np.savez_compressed(
                path,
                format_version=np.asarray([1], dtype=np.int16),
                action_size=np.asarray([1625], dtype=np.int32),
                history_lengths=np.asarray([1], dtype=np.int64),
                boards=board,
                values=np.asarray([1.0], dtype=np.float32),
                policy_offsets=np.asarray([0, 1], dtype=np.int64),
                policy_indices=np.asarray([64], dtype=np.uint16),
                policy_values=np.asarray([1.0], dtype=np.float32),
            )
            loaded = load_compact_replay(path)

        self.assertEqual(len(loaded[0][0]), 3)

    def test_trim_compact_replay_keeps_latest_windows_atomically(self):
        policy = np.zeros(1625, dtype=np.float32)
        policy[[64, 129]] = [0.25, 0.75]
        history = [
            deque([(np.full((2, 5, 5), window), policy, float(window)) for _ in range(window + 1)])
            for window in range(4)
        ]

        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, 'latest.examples.npz')
            save_compact_replay(path, history)

            result = trim_compact_replay(path, keep_last_windows=2)
            loaded = load_compact_replay(path)

        self.assertEqual(result, {
            'before_windows': 4,
            'after_windows': 2,
            'before_examples': 10,
            'after_examples': 7,
            'trimmed': True,
        })
        self.assertEqual([len(window) for window in loaded], [3, 4])
        np.testing.assert_array_equal(loaded[0][0][0], np.full((2, 5, 5), 2))
        np.testing.assert_array_equal(loaded[1][0][0], np.full((2, 5, 5), 3))

    def test_trim_compact_replay_rejects_empty_history_request(self):
        with self.assertRaisesRegex(ValueError, 'at least 1'):
            trim_compact_replay('unused.npz', keep_last_windows=0)

    def test_collapse_compact_replay_symmetries_keeps_one_per_group(self):
        policy = np.zeros(1625, dtype=np.float32)
        policy[64] = 1.0
        history = []
        for window in range(2):
            examples = []
            for position in range(window + 1):
                for symmetry in range(8):
                    board = np.full((2, 5, 5), 100 * window + 10 * position + symmetry)
                    examples.append((board, policy, float(position)))
            history.append(deque(examples))

        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, 'latest.examples.npz')
            save_compact_replay(path, history)

            result = collapse_compact_replay_symmetries(path, group_size=8)
            loaded = load_compact_replay(path)

        self.assertEqual(result, {
            'windows': 2,
            'before_examples': 24,
            'after_examples': 3,
            'symmetry_group_size': 8,
            'collapsed': True,
        })
        self.assertEqual([len(window) for window in loaded], [1, 2])
        self.assertEqual(int(loaded[0][0][0][0, 0, 0]), 0)
        self.assertEqual(int(loaded[1][0][0][0, 0, 0]), 100)
        self.assertEqual(int(loaded[1][1][0][0, 0, 0]), 110)

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
