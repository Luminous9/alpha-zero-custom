import unittest

import numpy as np

from MCTS import MCTS
from pretrain_santorini_greedy import blocking_targets, greedy_targets
from santorini.SantoriniGame import SantoriniGame
from santorini.SantoriniPlayers import GreedySantoriniPlayer
from utils import dotdict


DIRECTIONS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def action(origin, move_direction, build_direction):
    return (origin[0] * 5 + origin[1]) * 64 + move_direction * 8 + build_direction


def decoded_action(game, board, action_id):
    del board
    worker, move_direction, build_direction = game.decodeAction(action_id)
    move = (
        worker[0] + DIRECTIONS[move_direction][0],
        worker[1] + DIRECTIONS[move_direction][1],
    )
    build = (
        move[0] + DIRECTIONS[build_direction][0],
        move[1] + DIRECTIONS[build_direction][1],
    )
    return worker, move, build


class OnePlyGreedyNNet:
    def __init__(self, game):
        self.game = game

    def predict(self, board):
        valids = self.game.getValidMoves(board, 1)
        policy = np.zeros(self.game.getActionSize(), dtype=np.float32)
        legal_actions = np.flatnonzero(valids)

        if len(legal_actions) == 0:
            return policy, -1

        scores = []
        for action_id in legal_actions:
            next_board, _ = self.game.getNextState(board, 1, int(action_id))
            scores.append(self.game.getScore(next_board, 1))
        scores = np.array(scores, dtype=np.float32)
        scores = scores - np.max(scores)
        weights = np.exp(scores)
        policy[legal_actions] = weights / np.sum(weights)

        best_score = float(np.max(scores))
        value = 1.0 if best_score >= 100 else 0.0
        return policy, value


class CountingUniformNNet:
    def __init__(self, game):
        self.game = game
        self.predict_calls = 0

    def predict(self, board):
        self.predict_calls += 1
        return np.ones(self.game.getActionSize(), dtype=np.float32), 0.0


class TestSantoriniTactics(unittest.TestCase):
    def setUp(self):
        self.game = SantoriniGame(5)

    def empty_board(self):
        board = np.zeros((2, 5, 5), dtype=int)
        board[0, 2, 2] = 1
        board[0, 0, 0] = 2
        board[0, 4, 4] = -1
        board[0, 4, 3] = -2
        return board

    def test_greedy_label_prefers_all_immediate_win_suffixes(self):
        board = self.empty_board()
        board[1, 2, 2] = 2
        board[1, 2, 3] = 3

        policy, value, best_score, tie_count = greedy_targets(self.game, board)
        east_winning_actions = range(action((2, 2), 4, 0), action((2, 2), 4, 7) + 1)

        self.assertEqual(best_score, 100)
        self.assertEqual(value, 1)
        self.assertEqual(tie_count, 8)
        for action_id in east_winning_actions:
            self.assertGreater(policy[action_id], 0)
        self.assertAlmostEqual(float(np.sum(policy)), 1.0)

    def test_greedy_player_takes_immediate_win(self):
        board = self.empty_board()
        board[1, 2, 2] = 2
        board[1, 2, 3] = 3

        chosen_action = GreedySantoriniPlayer(self.game).play(board)

        self.assertIn(chosen_action, range(action((2, 2), 4, 0), action((2, 2), 4, 7) + 1))

    def test_blocking_opponent_level_three_square_is_legal(self):
        board = self.empty_board()
        board[0, 2, 2] = 1
        board[0, 0, 0] = 2
        board[0, 1, 2] = -1
        board[0, 4, 3] = -2
        board[1, 1, 2] = 2
        board[1, 1, 3] = 3

        valids = self.game.getValidMoves(board, 1)
        legal_actions = np.flatnonzero(valids)
        blocking_actions = []
        for action_id in legal_actions:
            _, _, build = decoded_action(self.game, board, int(action_id))
            if build == (1, 3):
                blocking_actions.append(int(action_id))

        self.assertGreater(len(blocking_actions), 0)
        next_board, _ = self.game.getNextState(board, 1, blocking_actions[0])
        self.assertEqual(next_board[1, 1, 3], 4)

    def test_blocking_label_prefers_doming_opponent_win_square(self):
        board = self.empty_board()
        board[0, 2, 2] = 1
        board[0, 0, 0] = 2
        board[0, 1, 2] = -1
        board[0, 4, 3] = -2
        board[1, 1, 2] = 2
        board[1, 1, 3] = 3

        policy, value, threat_count, block_count = blocking_targets(self.game, board)
        support = np.flatnonzero(policy)

        self.assertEqual(value, 0.5)
        self.assertEqual(threat_count, 1)
        self.assertEqual(block_count, len(support))
        self.assertAlmostEqual(float(np.sum(policy)), 1.0)
        for action_id in support:
            _, _, build = decoded_action(self.game, board, int(action_id))
            self.assertEqual(build, (1, 3))

    def test_mcts_with_one_ply_greedy_prior_finds_immediate_win(self):
        board = self.empty_board()
        board[1, 2, 2] = 2
        board[1, 2, 3] = 3

        mcts = MCTS(self.game, OnePlyGreedyNNet(self.game), dotdict({'numMCTSSims': 16, 'cpuct': 1.0}))
        policy = mcts.getActionProb(board, temp=0)
        chosen_action = int(np.argmax(policy))

        self.assertIn(chosen_action, range(action((2, 2), 4, 0), action((2, 2), 4, 7) + 1))

    def test_tactical_shortcut_returns_immediate_win_without_network_search(self):
        board = self.empty_board()
        board[1, 2, 2] = 2
        board[1, 2, 3] = 3
        nnet = CountingUniformNNet(self.game)
        mcts = MCTS(self.game, nnet, dotdict({'numMCTSSims': 16, 'cpuct': 1.0}))

        policy = np.asarray(mcts.getActionProb(board, temp=1))
        support = np.flatnonzero(policy)

        self.assertEqual(nnet.predict_calls, 0)
        self.assertEqual(len(support), 8)
        self.assertTrue(all(action_id in range(
            action((2, 2), 4, 0),
            action((2, 2), 4, 7) + 1,
        ) for action_id in support))
        np.testing.assert_allclose(policy[support], np.full(8, 1.0 / 8.0))

    def test_tactical_shortcut_finds_single_forced_block(self):
        board = self.empty_board()
        board[0, 1, 2] = -1
        board[1, 1, 2] = 2
        board[1, 1, 3] = 3
        nnet = CountingUniformNNet(self.game)
        mcts = MCTS(self.game, nnet, dotdict({'numMCTSSims': 16, 'cpuct': 1.0}))

        tactical = mcts.prepareTacticalRoot(board)
        policy = np.asarray(mcts.getActionProb(board, temp=1))
        chosen_action = int(np.argmax(policy))
        _, _, build = decoded_action(self.game, board, chosen_action)

        self.assertEqual(tactical['kind'], 'single_forced_block')
        self.assertEqual(nnet.predict_calls, 0)
        self.assertEqual(build, (1, 3))

    def test_tactical_shortcut_prunes_root_when_multiple_blocks_exist(self):
        board = self.empty_board()
        board[0, 0, 0] = 0
        board[0, 0, 1] = 2
        board[0, 1, 2] = -1
        board[1, 1, 2] = 2
        board[1, 1, 3] = 3
        nnet = CountingUniformNNet(self.game)
        mcts = MCTS(self.game, nnet, dotdict({'numMCTSSims': 16, 'cpuct': 1.0}))

        tactical = mcts.prepareTacticalRoot(board)
        policy = np.asarray(mcts.getActionProb(board, temp=1))
        support = np.flatnonzero(policy)

        self.assertEqual(tactical['kind'], 'forced_block_pruned')
        self.assertGreater(nnet.predict_calls, 0)
        self.assertGreaterEqual(len(support), 1)
        self.assertTrue(set(support).issubset(set(map(int, tactical['actions']))))
        for action_id in support:
            _, _, build = decoded_action(self.game, board, int(action_id))
            self.assertEqual(build, (1, 3))

    def test_symmetry_refresh_prunes_an_existing_interior_node_to_forced_blocks(self):
        board = self.empty_board()
        board[0, 0, 0] = 0
        board[0, 0, 1] = 2
        board[0, 1, 2] = -1
        board[1, 1, 2] = 2
        board[1, 1, 3] = 3
        nnet = CountingUniformNNet(self.game)
        mcts = MCTS(self.game, nnet, dotdict({
            'numMCTSSims': 4,
            'cpuct': 1.0,
            'searchSymmetryEvaluation': True,
            'rootSymmetrySamples': 2,
        }))

        # Reproduce a node first encountered below another root: it is expanded
        # before root-only tactical pruning has been applied.
        interior_leaf = mcts.select_leaf(board)
        interior_policy, interior_value = nnet.predict(
            mcts.getLeafEvaluationBoards(interior_leaf)[0]
        )
        mcts.complete_search(interior_leaf, interior_policy, interior_value)
        state_key = self.game.stringRepresentation(board)
        original_actions = mcts.As[state_key].copy()

        tactical = mcts.prepareTacticalRoot(board)
        policy = np.asarray(mcts.getActionProb(board, temp=1))

        self.assertEqual(tactical['kind'], 'forced_block_pruned')
        self.assertLess(len(mcts.As[state_key]), len(original_actions))
        self.assertEqual(set(mcts.As[state_key]), set(map(int, tactical['actions'])))
        self.assertAlmostEqual(float(policy.sum()), 1.0)

    def test_tactical_shortcut_proves_loss_when_every_move_allows_a_win(self):
        board = np.zeros((2, 5, 5), dtype=int)
        board[0, 4, 0] = 1
        board[0, 4, 1] = 2
        board[0, 0, 0] = -1
        board[0, 0, 4] = -2
        board[1, 0, 0] = 2
        board[1, 0, 4] = 2
        board[1, 0, 1] = 3
        board[1, 0, 3] = 3
        nnet = CountingUniformNNet(self.game)
        mcts = MCTS(self.game, nnet, dotdict({'numMCTSSims': 16, 'cpuct': 1.0}))

        tactical = mcts.prepareTacticalRoot(board)
        policy = np.asarray(mcts.getActionProb(board, temp=1))

        self.assertEqual(tactical['kind'], 'proven_loss_in_two')
        self.assertEqual(nnet.predict_calls, 0)
        self.assertAlmostEqual(float(policy.sum()), 1.0)
        self.assertEqual(set(np.flatnonzero(policy)), set(np.flatnonzero(
            self.game.getValidMoves(board, 1)
        )))


if __name__ == '__main__':
    unittest.main()
