import unittest

import numpy as np

from BatchedArena import BatchedMCTSArena
from utils import dotdict


class TinyGame:
    def getInitBoard(self):
        return np.array([0], dtype=np.int64)

    def getActionSize(self):
        return 2

    def getNextState(self, board, player, action):
        return np.array([board[0] + 1], dtype=np.int64), -player

    def getValidMoves(self, board, player):
        return np.array([1, 1], dtype=np.int64)

    def getGameEnded(self, board, player):
        if board[0] >= 2:
            return -1
        return 0

    def getCanonicalForm(self, board, player):
        return board

    def stringRepresentation(self, board):
        return board.tobytes()


class OneMovePlacementGame(TinyGame):
    def __init__(self):
        self.actions = []

    def getNextState(self, board, player, action):
        self.actions.append(int(action))
        return np.array([action + 1], dtype=np.int64), -player

    def getGameEnded(self, board, player):
        return 1 if board[0] else 0

    def isPlacementPhase(self, board):
        return board[0] == 0


class BatchCountingNNet:
    def __init__(self):
        self.batch_sizes = []

    def predict(self, board):
        return np.array([0.5, 0.5], dtype=np.float32), 0.0

    def predict_batch(self, boards):
        self.batch_sizes.append(len(boards))
        policies = np.tile(np.array([0.5, 0.5], dtype=np.float32), (len(boards), 1))
        values = np.zeros(len(boards), dtype=np.float32)
        return policies, values


class TestBatchedMCTSArena(unittest.TestCase):
    def test_batched_arena_scores_swapped_starts(self):
        game = TinyGame()
        player1_nnet = BatchCountingNNet()
        player2_nnet = BatchCountingNNet()
        args = dotdict({'numMCTSSims': 2, 'cpuct': 1.0})
        arena = BatchedMCTSArena(
            game,
            player1_nnet,
            player2_nnet,
            args,
            batch_size=2,
            quiet=True,
        )

        one_won, two_won, draws = arena.playGames(4)

        self.assertEqual((one_won, two_won, draws), (2, 2, 0))
        self.assertGreaterEqual(max(player1_nnet.batch_sizes), 2)
        self.assertGreaterEqual(max(player2_nnet.batch_sizes), 2)

    def test_seeded_placement_sampling_is_reproducible_and_varied(self):
        seeds = list(range(20))

        def play():
            game = OneMovePlacementGame()
            arena = BatchedMCTSArena(
                game,
                BatchCountingNNet(),
                BatchCountingNNet(),
                dotdict({'numMCTSSims': 4, 'cpuct': 1.0}),
                batch_size=20,
                quiet=True,
                placement_temperature=1.0,
                game_seeds=seeds,
            )
            result = arena.playGames(40)
            return result, game.actions

        first_result, first_actions = play()
        second_result, second_actions = play()

        self.assertEqual(first_result, second_result)
        self.assertEqual(first_actions, second_actions)
        self.assertEqual(set(first_actions), {0, 1})

    def test_gumbel_search_runs_batched_and_is_seed_reproducible(self):
        seeds = list(range(8))

        def play():
            game = OneMovePlacementGame()
            arena = BatchedMCTSArena(
                game,
                BatchCountingNNet(),
                BatchCountingNNet(),
                dotdict({
                    'numMCTSSims': 4,
                    'cpuct': 1.0,
                    'searchMode': 'gumbel',
                    'gumbelMaxConsideredActions': 2,
                    'gumbelScale': 1.0,
                }),
                batch_size=8,
                quiet=True,
                placement_temperature=1.0,
                game_seeds=seeds,
            )
            arena.playGames(16)
            return game.actions

        first_actions = play()
        second_actions = play()

        self.assertEqual(first_actions, second_actions)
        self.assertEqual(set(first_actions), {0, 1})

    def test_contestants_can_use_independent_search_modes_and_budgets(self):
        shared = dotdict({'numMCTSSims': 2, 'cpuct': 1.0})
        player_args = {
            1: dotdict({
                'numMCTSSims': 4,
                'cpuct': 1.0,
                'searchMode': 'gumbel',
                'gumbelMaxConsideredActions': 2,
                'gumbelScale': 0.0,
            }),
            -1: dotdict({
                'numMCTSSims': 2,
                'cpuct': 1.0,
                'searchMode': 'puct',
            }),
        }
        arena = BatchedMCTSArena(
            TinyGame(),
            BatchCountingNNet(),
            BatchCountingNNet(),
            shared,
            batch_size=2,
            quiet=True,
            player_args=player_args,
        )

        game_state = arena._newGame({1: 1, -1: -1}, game_seed=7)
        result = arena.playGames(4)

        self.assertTrue(game_state['mcts_by_player'][1].usesGumbelSearch())
        self.assertFalse(game_state['mcts_by_player'][-1].usesGumbelSearch())
        self.assertEqual(sum(result), 4)


if __name__ == "__main__":
    unittest.main()
