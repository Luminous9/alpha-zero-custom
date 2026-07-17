import unittest

import numpy as np

from MCTS import MCTS
from utils import dotdict


class FourActionGame:
    def getActionSize(self):
        return 4

    def getNextState(self, board, player, action):
        return np.array([int(action) + 1], dtype=np.int8), -player

    def getValidMoves(self, board, player):
        return np.ones(4, dtype=np.int8)

    def getGameEnded(self, board, player):
        return -1 if int(board[0]) > 0 else 0

    def getCanonicalForm(self, board, player):
        return board

    def stringRepresentation(self, board):
        return board.tobytes()


class UniformNetwork:
    def predict(self, board):
        return np.full(4, 0.25, dtype=np.float32), 0.0


class PhaseFourActionGame(FourActionGame):
    def isPlacementPhase(self, board):
        return int(board[0]) == 0


class TestGumbelMCTS(unittest.TestCase):
    def _mcts(self, simulations=9):
        return MCTS(
            FourActionGame(),
            UniformNetwork(),
            dotdict({
                'numMCTSSims': simulations,
                'cpuct': 1.0,
                'searchMode': 'gumbel',
                'gumbelMaxConsideredActions': 4,
                'gumbelScale': 0.0,
                'addDirichletNoise': True,
            }),
        )

    def test_sequential_halving_schedule_matches_reference_algorithm(self):
        self.assertEqual(
            MCTS._gumbel_considered_visits(4, 8),
            (0, 0, 0, 0, 1, 1, 2, 2),
        )

    def test_root_uses_sequential_halving_and_returns_selected_action(self):
        mcts = self._mcts()
        root = np.array([0], dtype=np.int8)

        action_policy = np.asarray(mcts.getActionProb(root, temp=1))
        state_key = mcts.game.stringRepresentation(root)

        np.testing.assert_array_equal(mcts.Nsas[state_key], [3, 3, 1, 1])
        np.testing.assert_array_equal(action_policy, [1.0, 0.0, 0.0, 0.0])
        self.assertNotIn(state_key, mcts.noised_roots)

    def test_training_target_is_improved_policy_not_visit_distribution(self):
        mcts = self._mcts()
        root = np.array([0], dtype=np.int8)
        mcts.getActionProb(root)
        state_key = mcts.game.stringRepresentation(root)
        mcts.Qs[state_key][:] = np.array([1.0, 0.25, -0.5, -1.0])

        training_policy = np.asarray(mcts.getTrainingPolicyFromTree(root))
        visit_policy = mcts.Nsas[state_key] / np.sum(mcts.Nsas[state_key])

        self.assertAlmostEqual(float(training_policy.sum()), 1.0, places=7)
        self.assertGreater(training_policy[0], training_policy[1])
        self.assertFalse(np.allclose(training_policy, visit_policy))

    def test_reused_child_starts_halving_from_root_visit_deltas(self):
        mcts = self._mcts(simulations=8)
        root = np.array([0], dtype=np.int8)
        state_key = mcts.game.stringRepresentation(root)
        mcts.Vs[state_key] = np.ones(4, dtype=np.int8)
        mcts._expand_leaf(state_key, root, np.full(4, 0.25))
        mcts.raw_values[state_key] = 0.0
        mcts.Nsas[state_key][:] = [5, 2, 1, 0]
        mcts.Ns[state_key] = 8

        mcts.prepareSearchRoot(root, 8)
        action, _ = mcts._best_gumbel_root_action(state_key)
        root_data = mcts.gumbel_roots[state_key]

        self.assertEqual(root_data['edge_budget'], 8)
        np.testing.assert_array_equal(root_data['baseline_visits'], [5, 2, 1, 0])
        self.assertEqual(action, 0)

    def test_puct_remains_default(self):
        mcts = MCTS(
            FourActionGame(),
            UniformNetwork(),
            dotdict({'numMCTSSims': 5, 'cpuct': 1.0}),
        )
        root = np.array([0], dtype=np.int8)

        action_policy = np.asarray(mcts.getActionProb(root, temp=1))
        state_key = mcts.game.stringRepresentation(root)

        self.assertFalse(mcts.usesGumbelSearch())
        self.assertAlmostEqual(float(action_policy.sum()), 1.0)
        self.assertNotIn(state_key, mcts.gumbel_roots)
        np.testing.assert_allclose(
            action_policy,
            mcts.getTrainingPolicyFromTree(root, temp=1),
        )

    def test_placement_scale_overrides_standard_scale_by_phase(self):
        game = PhaseFourActionGame()
        mcts = MCTS(
            game,
            UniformNetwork(),
            dotdict({
                'numMCTSSims': 4,
                'cpuct': 1.0,
                'searchMode': 'gumbel',
                'gumbelMaxConsideredActions': 4,
                'gumbelScale': 0.0,
                'gumbelPlacementScale': 1.5,
            }),
        )
        placement = np.array([0], dtype=np.int8)
        standard = np.array([5], dtype=np.int8)

        mcts.prepareSearchRoot(placement, 4)
        mcts.prepareSearchRoot(standard, 4)

        self.assertEqual(
            mcts.gumbel_roots[game.stringRepresentation(placement)]['gumbel_scale'],
            1.5,
        )
        self.assertEqual(
            mcts.gumbel_roots[game.stringRepresentation(standard)]['gumbel_scale'],
            0.0,
        )


if __name__ == '__main__':
    unittest.main()
