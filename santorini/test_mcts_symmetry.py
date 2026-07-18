import unittest

import numpy as np

from MCTS import MCTS
from santorini.SantoriniGame import SantoriniGame
from utils import dotdict


class UniformNetwork:
    def __init__(self, game):
        self.policy = np.full(
            game.getActionSize(),
            1.0 / game.getActionSize(),
            dtype=np.float32,
        )

    def predict(self, board):
        return self.policy.copy(), 0.0

    def predict_batch(self, boards):
        return (
            np.repeat(self.policy[None, :], len(boards), axis=0),
            np.zeros(len(boards), dtype=np.float32),
        )


class TestMCTSSymmetryEvaluation(unittest.TestCase):
    def setUp(self):
        self.game = SantoriniGame(5, sequential_placement=True)
        self.args = dotdict({
            'numMCTSSims': 4,
            'cpuct': 1.0,
            'searchSymmetryEvaluation': True,
            'rootSymmetrySamples': 2,
            'placementRootSymmetrySamples': 8,
        })

    def _transformed_policies(self, base_policy, symmetry_ids):
        return np.asarray([
            self.game._transform_policy(
                base_policy,
                int(symmetry_id) // 2,
                bool(int(symmetry_id) % 2),
            )
            for symmetry_id in symmetry_ids
        ])

    def test_placement_root_averages_all_orientations_in_tree_coordinates(self):
        board = self.game.getInitBoard()
        mcts = MCTS(self.game, UniformNetwork(self.game), self.args)
        mcts.prepareSearchRoot(board, 4, rng=np.random.RandomState(7))

        leaf = mcts.select_leaf(board)
        evaluation_boards = mcts.getLeafEvaluationBoards(leaf)
        symmetry_ids = leaf['eval_symmetry_ids']
        self.assertEqual(symmetry_ids, tuple(range(8)))
        self.assertEqual(len(evaluation_boards), 8)

        base_policy = np.arange(1, self.game.getActionSize() + 1, dtype=np.float64)
        base_policy /= base_policy.sum()
        mcts.complete_search(
            leaf,
            self._transformed_policies(base_policy, symmetry_ids),
            np.arange(8, dtype=np.float32),
        )

        state_key = self.game.stringRepresentation(board)
        legal_actions = mcts.As[state_key]
        expected = base_policy[legal_actions]
        expected /= expected.sum()
        np.testing.assert_allclose(mcts.Ps[state_key], expected, rtol=1e-6, atol=1e-7)
        self.assertEqual(mcts.raw_values[state_key], 3.5)
        self.assertIn(state_key, mcts.symmetry_evaluated_roots)
        self.assertEqual(mcts.Ns[state_key], 0)

        stats = mcts.drainSymmetryEvaluationStats()
        self.assertEqual(stats['root_evaluations'], 1)
        self.assertEqual(stats['root_orientations'], 8)
        self.assertEqual(stats['interior_evaluations'], 0)

    def test_root_refresh_preserves_existing_search_statistics(self):
        board = self.game.getInitBoard()
        mcts = MCTS(self.game, UniformNetwork(self.game), self.args)
        mcts.prepareSearchRoot(board, 4)
        leaf = mcts.select_leaf(board)
        policy = np.full(self.game.getActionSize(), 1.0 / self.game.getActionSize())
        mcts.complete_search(
            leaf,
            self._transformed_policies(policy, leaf['eval_symmetry_ids']),
            np.zeros(8),
        )
        state_key = self.game.stringRepresentation(board)
        mcts.Ns[state_key] = 5
        mcts.Nsas[state_key][0] = 3
        mcts.Qs[state_key][0] = 0.75

        mcts.symmetry_evaluated_roots.remove(state_key)
        mcts.prepareSearchRoot(board, 4)
        refresh = mcts.select_leaf(board)
        mcts.complete_search(
            refresh,
            self._transformed_policies(policy, refresh['eval_symmetry_ids']),
            np.zeros(8),
        )

        self.assertEqual(mcts.Ns[state_key], 5)
        self.assertEqual(mcts.Nsas[state_key][0], 3)
        self.assertEqual(mcts.Qs[state_key][0], 0.75)

    def test_standard_root_sampling_is_seeded_and_interior_uses_one_orientation(self):
        standard_board = np.zeros((2, 5, 5), dtype=np.int8)
        standard_board[0, 0, 0] = 1
        standard_board[0, 4, 4] = 1
        standard_board[0, 0, 4] = -1
        standard_board[0, 4, 0] = -1

        first = MCTS(self.game, UniformNetwork(self.game), self.args)
        second = MCTS(self.game, UniformNetwork(self.game), self.args)
        first.prepareSearchRoot(standard_board, 4, rng=np.random.RandomState(19))
        second.prepareSearchRoot(standard_board, 4, rng=np.random.RandomState(19))
        first_leaf = first.select_leaf(standard_board)
        second_leaf = second.select_leaf(standard_board)

        self.assertEqual(first_leaf['eval_symmetry_ids'], second_leaf['eval_symmetry_ids'])
        self.assertEqual(len(first_leaf['eval_symmetry_ids']), 2)
        self.assertEqual(len(set(first_leaf['eval_symmetry_ids'])), 2)

        placement_board = self.game.getInitBoard()
        interior = MCTS(self.game, UniformNetwork(self.game), self.args)
        interior.prepareSearchRoot(placement_board, 4, rng=np.random.RandomState(3))
        root_leaf = interior.select_leaf(placement_board)
        uniform = np.full(
            self.game.getActionSize(),
            1.0 / self.game.getActionSize(),
        )
        interior.complete_search(
            root_leaf,
            self._transformed_policies(uniform, root_leaf['eval_symmetry_ids']),
            np.zeros(8),
        )
        child_leaf = interior.select_leaf(placement_board)
        self.assertTrue(child_leaf['needs_eval'])
        self.assertEqual(len(child_leaf['eval_symmetry_ids']), 1)
        self.assertEqual(len(interior.getLeafEvaluationBoards(child_leaf)), 1)


if __name__ == '__main__':
    unittest.main()
