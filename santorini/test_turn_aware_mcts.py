import unittest

import numpy as np

from Coach import Coach
from MCTS import MCTS
from santorini.SantoriniGame import SantoriniGame
from utils import dotdict


class SamePlayerTerminalGame:
    def getActionSize(self):
        return 1

    def getNextState(self, board, player, action):
        return np.array([1], dtype=np.int8), player

    def getValidMoves(self, board, player):
        return np.array([1], dtype=np.int8)

    def getGameEnded(self, board, player):
        return 1 if int(board[0]) == 1 else 0

    def getCanonicalForm(self, board, player):
        return board

    def stringRepresentation(self, board):
        return board.tobytes()


class ZeroNetwork:
    def predict(self, board):
        return np.array([1.0], dtype=np.float32), 0.0


class UniformSantoriniNetwork:
    def __init__(self, game):
        self.policy = np.full(
            game.getActionSize(),
            1.0 / game.getActionSize(),
            dtype=np.float32,
        )

    def predict(self, board):
        return self.policy.copy(), 0.0


class TestTurnAwareMCTS(unittest.TestCase):
    def test_coach_reads_updated_start_iteration_dictionary_key(self):
        args = dotdict({'startIteration': 0})
        args['startIteration'] = 50
        coach = Coach.__new__(Coach)
        coach.args = args

        self.assertEqual(coach._arg('startIteration', 0), 50)

    def test_same_player_edge_does_not_invert_value(self):
        game = SamePlayerTerminalGame()
        mcts = MCTS(game, ZeroNetwork(), dotdict({'numMCTSSims': 2, 'cpuct': 1.0}))
        root = np.array([0], dtype=np.int8)

        mcts.getActionProb(root, temp=1)

        root_key = game.stringRepresentation(root)
        self.assertEqual(mcts.Qsa[(root_key, 0)], 1.0)

    def test_santorini_tree_stores_only_legal_action_arrays(self):
        game = SantoriniGame(5, sequential_placement=True)
        args = dotdict({
            'numMCTSSims': 1,
            'cpuct': 1.0,
            'addDirichletNoise': True,
            'dirichletAlpha': 0.30,
            'dirichletEpsilon': 0.25,
        })
        mcts = MCTS(game, UniformSantoriniNetwork(game), args)
        root = game.getInitBoard()

        probabilities = np.asarray(mcts.getActionProb(root, temp=1))
        root_key = game.stringRepresentation(root)
        legal_actions = np.flatnonzero(game.getValidMoves(root, 1))

        self.assertEqual(len(mcts.As[root_key]), len(legal_actions))
        self.assertEqual(len(mcts.Ps[root_key]), len(legal_actions))
        self.assertEqual(len(mcts.Qs[root_key]), len(legal_actions))
        self.assertEqual(len(mcts.Nsas[root_key]), len(legal_actions))
        self.assertNotIn(root_key, mcts.Vs)
        self.assertLess(len(mcts.Ps[root_key]), game.getActionSize())
        self.assertAlmostEqual(float(mcts.Ps[root_key].sum()), 1.0, places=6)
        self.assertEqual(probabilities.shape, (game.getActionSize(),))
        self.assertAlmostEqual(float(probabilities.sum()), 1.0, places=6)
        np.testing.assert_array_equal(
            probabilities[np.setdiff1d(np.arange(game.getActionSize()), legal_actions)],
            0.0,
        )

    def test_dense_statistics_scatter_compact_edges_to_global_actions(self):
        game = SantoriniGame(5, sequential_placement=True)
        mcts = MCTS(
            game,
            UniformSantoriniNetwork(game),
            dotdict({'numMCTSSims': 8, 'cpuct': 1.0}),
        )
        root = game.getInitBoard()

        mcts.getActionProb(root, temp=1)
        root_key = game.stringRepresentation(root)
        counts = mcts.getDenseActionCounts(root_key)
        values = mcts.getDenseActionValues(root_key)
        actions = mcts.As[root_key]

        self.assertEqual(counts.shape, (game.getActionSize(),))
        self.assertEqual(values.shape, (game.getActionSize(),))
        self.assertEqual(int(counts.sum()), 7)
        np.testing.assert_array_equal(counts[actions], mcts.Nsas[root_key])
        np.testing.assert_allclose(values[actions], mcts.Qs[root_key])
        for action in actions[counts[actions] > 0]:
            compact_index = int(np.searchsorted(actions, action))
            self.assertEqual(mcts.Nsa[(root_key, int(action))], int(mcts.Nsas[root_key][compact_index]))
            self.assertAlmostEqual(mcts.Qsa[(root_key, int(action))], float(mcts.Qs[root_key][compact_index]))
        illegal = np.setdiff1d(np.arange(game.getActionSize()), actions)
        np.testing.assert_array_equal(counts[illegal], 0)
        np.testing.assert_array_equal(values[illegal], 0.0)


if __name__ == '__main__':
    unittest.main()
