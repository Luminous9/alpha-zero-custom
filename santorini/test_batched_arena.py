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


class PlacementThenStandardGame(TinyGame):
    def isPlacementPhase(self, board):
        return board[0] < 2


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


class SymmetricOneMoveGame:
    def getInitBoard(self):
        return np.zeros((1, 2, 2), dtype=np.int8)

    def getActionSize(self):
        return 4

    def getNextState(self, board, player, action):
        next_board = board.copy()
        next_board.reshape(-1)[int(action)] = 1
        return next_board, -player

    def getValidMoves(self, board, player):
        return (board.reshape(-1) == 0).astype(np.int8)

    def getGameEnded(self, board, player):
        return -1 if np.any(board) else 0

    def getCanonicalForm(self, board, player):
        return board

    def stringRepresentation(self, board):
        return board.tobytes()

    def getPolicySymmetryPermutation(self, rotations, flip):
        old_indices = np.arange(4, dtype=np.int64)
        squares = old_indices.reshape(2, 2)
        transformed = np.rot90(squares, int(rotations))
        if flip:
            transformed = np.fliplr(transformed)
        new_indices = np.empty(4, dtype=np.int64)
        for new_index, old_index in enumerate(transformed.reshape(-1)):
            new_indices[int(old_index)] = int(new_index)
        return old_indices, new_indices


class FourActionBatchCountingNNet(BatchCountingNNet):
    def predict(self, board):
        return np.full(4, 0.25, dtype=np.float32), 0.0

    def predict_batch(self, boards):
        self.batch_sizes.append(len(boards))
        return (
            np.full((len(boards), 4), 0.25, dtype=np.float32),
            np.zeros(len(boards), dtype=np.float32),
        )


class MiniPlacementGame:
    def getInitBoard(self):
        return np.zeros((2, 5, 5), dtype=np.int8)

    def getActionSize(self):
        return 25

    def getNextState(self, board, player, action):
        board = board.copy()
        count = int(np.count_nonzero(board[0]))
        if count < 4:
            board[0].flat[int(action)] = 1 if count < 2 else -1
            next_count = count + 1
            next_player = 1 if next_count < 2 or next_count == 4 else -1
            return board, next_player
        board[1, 0, 0] = 1
        return board, -player

    def getValidMoves(self, board, player):
        return (board[0].reshape(-1) == 0).astype(np.int8)

    def getGameEnded(self, board, player):
        return -1 if board[1, 0, 0] else 0

    def getCanonicalForm(self, board, player):
        return board

    def stringRepresentation(self, board):
        return board.tobytes()

    def isPlacementPhase(self, board):
        return np.count_nonzero(board[0]) < 4


class Uniform25NNet:
    def predict_batch(self, boards):
        return (
            np.full((len(boards), 25), 1 / 25, dtype=np.float32),
            np.zeros(len(boards), dtype=np.float32),
        )


class TestBatchedMCTSArena(unittest.TestCase):
    def test_generates_unpaired_placement_stream(self):
        arena = BatchedMCTSArena(
            MiniPlacementGame(),
            Uniform25NNet(),
            Uniform25NNet(),
            dotdict({'numMCTSSims': 1, 'cpuct': 1.0}),
            batch_size=4,
            quiet=True,
            placement_temperature=1.0,
            game_seeds=[7, 8, 9],
        )

        records = arena.generatePlacements(3)

        self.assertEqual(len(records), 3)
        self.assertEqual([record['pair_index'] for record in records], [0, 1, 2])
        self.assertEqual({record['seat_order'] for record in records}, {'placement_source'})
        self.assertTrue(all(len(record['placement_actions']) == 4 for record in records))

    def test_generates_placements_without_playing_standard_continuations(self):
        arena = BatchedMCTSArena(
            MiniPlacementGame(),
            Uniform25NNet(),
            Uniform25NNet(),
            dotdict({'numMCTSSims': 1, 'cpuct': 1.0}),
            batch_size=4,
            quiet=True,
            placement_temperature=1.0,
            game_seeds=[7, 8],
        )

        records = arena.generatePlacementGames(4)

        self.assertEqual(len(records), 4)
        self.assertEqual({record['pair_index'] for record in records}, {0, 1})
        self.assertEqual(
            {record['seat_order'] for record in records},
            {'contestant1_first', 'contestant2_first'},
        )
        self.assertTrue(all(len(record['placement_actions']) == 4 for record in records))
        self.assertTrue(all(np.count_nonzero(record['opening_board'][0]) == 4 for record in records))

    def test_plays_arbitrary_fixed_opening_specifications(self):
        game = TinyGame()
        arena = BatchedMCTSArena(
            game,
            BatchCountingNNet(),
            BatchCountingNNet(),
            dotdict({'numMCTSSims': 1, 'cpuct': 1.0}),
            batch_size=2,
            quiet=True,
            record_placement_diagnostics=True,
        )
        result = arena.playGameSpecifications([
            {
                'specification_id': 10,
                'opening_board': np.array([1]),
                'side_to_player': {1: 1, -1: -1},
                'game_seed': 3,
                'game_index': 4,
                'seat_order': 'contestant1_first',
            },
            {
                'specification_id': 11,
                'opening_board': np.array([1]),
                'side_to_player': {1: -1, -1: 1},
                'game_seed': 4,
                'game_index': 4,
                'seat_order': 'contestant2_first',
            },
        ])

        self.assertEqual(result, (1, 1, 0))
        self.assertEqual(
            {record['specification_id'] for record in arena.game_records},
            {10, 11},
        )
        self.assertTrue(all(record['standard_trajectory'] for record in arena.game_records))

    def test_placement_diagnostics_count_symmetries_duplicates_and_trajectories(self):
        arena = BatchedMCTSArena(
            TinyGame(),
            BatchCountingNNet(),
            BatchCountingNNet(),
            dotdict({'numMCTSSims': 2, 'cpuct': 1.0}),
            quiet=True,
        )
        opening = 'p1=0:0,1:1|p2=2:2,3:3'
        arena.placement_records = [
            {
                'opening': opening,
                'labeled_opening_key': 'labeled-a',
                'symmetry_opening': 'symmetry-a',
                'standard_trajectory': (1, 2, 3),
                'winner': 1,
                'winner_side': 1,
                'p1_contestant': 1,
                'p2_contestant': -1,
            },
            {
                'opening': opening,
                'labeled_opening_key': 'labeled-a',
                'symmetry_opening': 'symmetry-a',
                'standard_trajectory': (1, 2, 3),
                'winner': -1,
                'winner_side': 1,
                'p1_contestant': -1,
                'p2_contestant': 1,
            },
            {
                'opening': opening,
                'labeled_opening_key': 'labeled-a',
                'symmetry_opening': 'symmetry-a',
                'standard_trajectory': (1, 4, 3),
                'winner': 1,
                'winner_side': -1,
                'p1_contestant': -1,
                'p2_contestant': 1,
            },
            {
                'opening': 'p1=0:1,1:2|p2=2:3,3:4',
                'labeled_opening_key': 'labeled-b',
                'symmetry_opening': 'symmetry-a',
                'standard_trajectory': (5, 6),
                'winner': 1,
                'winner_side': 1,
                'p1_contestant': 1,
                'p2_contestant': -1,
            },
        ]

        diagnostics = arena.placementDiagnostics()

        self.assertEqual(diagnostics['games_recorded'], 4)
        self.assertEqual(diagnostics['distinct_exact_openings'], 2)
        self.assertEqual(diagnostics['distinct_symmetry_unique_openings'], 1)
        self.assertEqual(diagnostics['duplicate_game_count'], 2)
        self.assertEqual(diagnostics['most_frequent_opening_count'], 3)
        self.assertEqual(diagnostics['repeated_exact_labeled_opening_groups'], 1)
        self.assertEqual(diagnostics['repeated_groups_with_divergent_standard_trajectories'], 1)
        self.assertEqual(
            diagnostics['duplicate_games_matching_an_existing_standard_trajectory'],
            1,
        )
        self.assertEqual(diagnostics['opening_results'][0]['contestant1_wins'], 2)
        self.assertEqual(diagnostics['opening_results'][0]['contestant2_wins'], 1)
        self.assertEqual(diagnostics['opening_results'][0]['player1_wins'], 2)
        self.assertEqual(diagnostics['opening_results'][0]['player2_wins'], 1)
        self.assertEqual(diagnostics['opening_results'][0]['contestant1_as_player1_games'], 1)
        self.assertEqual(diagnostics['opening_results'][0]['contestant1_as_player2_games'], 2)

    def test_symmetry_placement_signature_ignores_orientation_and_worker_labels(self):
        board = np.zeros((2, 5, 5), dtype=np.int64)
        board[0, 0, 1] = 1
        board[0, 2, 2] = 2
        board[0, 3, 0] = -1
        board[0, 4, 4] = -2
        transformed = np.rot90(board, 1, axes=(-2, -1))
        transformed[0][transformed[0] == 1] = 3
        transformed[0][transformed[0] == 2] = 1
        transformed[0][transformed[0] == 3] = 2

        self.assertEqual(
            BatchedMCTSArena._symmetryPlacementSignature(board),
            BatchedMCTSArena._symmetryPlacementSignature(transformed),
        )

    def test_fixed_standard_controller_takes_over_after_placement(self):
        game = PlacementThenStandardGame()
        contestant1 = BatchCountingNNet()
        contestant2 = BatchCountingNNet()
        standard = BatchCountingNNet()
        contestant_args = dotdict({'numMCTSSims': 2, 'cpuct': 1.0})
        standard_args = dotdict({'numMCTSSims': 4, 'cpuct': 1.0})
        arena = BatchedMCTSArena(
            game,
            contestant1,
            contestant2,
            contestant_args,
            quiet=True,
            standard_controller_nnet=standard,
            standard_controller_args=standard_args,
        )
        state = arena._newGame({1: 1, -1: -1}, game_seed=7)

        state['canonicalBoard'] = np.array([0], dtype=np.int64)
        placement_controller = arena._controller(state)
        self.assertIs(placement_controller[1], state['mcts_by_player'][1])
        self.assertIs(placement_controller[2], contestant1)
        self.assertIs(placement_controller[3], contestant_args)

        state['canonicalBoard'] = np.array([2], dtype=np.int64)
        standard_controller = arena._controller(state)
        self.assertEqual(standard_controller[0], 'standard')
        self.assertIs(standard_controller[1], state['standard_mcts_by_side'][1])
        self.assertIs(standard_controller[2], standard)
        self.assertIs(standard_controller[3], standard_args)

        state['curPlayer'] = -1
        other_side_controller = arena._controller(state)
        self.assertIs(other_side_controller[1], state['standard_mcts_by_side'][-1])
        self.assertIsNot(other_side_controller[1], standard_controller[1])

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
        self.assertEqual(len(arena.game_records), 4)
        self.assertEqual({record['pair_index'] for record in arena.game_records}, {0, 1})
        self.assertEqual(
            {record['seat_order'] for record in arena.game_records},
            {'contestant1_first', 'contestant2_first'},
        )

    def test_batched_arena_flattens_and_recombines_root_symmetry_ensembles(self):
        game = SymmetricOneMoveGame()
        player1_nnet = FourActionBatchCountingNNet()
        player2_nnet = FourActionBatchCountingNNet()
        args = dotdict({
            'numMCTSSims': 1,
            'cpuct': 1.0,
            'searchSymmetryEvaluation': True,
            'rootSymmetrySamples': 8,
            'placementRootSymmetrySamples': 8,
        })
        arena = BatchedMCTSArena(
            game,
            player1_nnet,
            player2_nnet,
            args,
            batch_size=2,
            quiet=True,
        )

        self.assertEqual(arena.playGames(4), (2, 2, 0))
        self.assertIn(16, player1_nnet.batch_sizes)
        self.assertIn(16, player2_nnet.batch_sizes)

    def test_batched_arena_deduplicates_identical_symmetry_inputs(self):
        game = SymmetricOneMoveGame()
        player1_nnet = FourActionBatchCountingNNet()
        player2_nnet = FourActionBatchCountingNNet()
        arena = BatchedMCTSArena(
            game,
            player1_nnet,
            player2_nnet,
            dotdict({
                'numMCTSSims': 1,
                'cpuct': 1.0,
                'searchSymmetryEvaluation': True,
                'rootSymmetrySamples': 8,
                'placementRootSymmetrySamples': 8,
                'inferenceDeduplication': True,
                'inferenceCacheSize': 32,
            }),
            batch_size=2,
            quiet=True,
        )

        self.assertEqual(arena.playGames(4), (2, 2, 0))
        self.assertEqual(player1_nnet.batch_sizes, [1])
        self.assertEqual(player2_nnet.batch_sizes, [1])
        diagnostics = arena.inferenceDiagnostics()
        self.assertEqual(diagnostics['requested'], 32)
        self.assertEqual(diagnostics['executed'], 2)
        self.assertEqual(diagnostics['reused'], 30)

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
