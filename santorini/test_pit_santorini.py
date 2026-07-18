import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO

import numpy as np

from pit_santorini import (
    NetworkMCTSPlayer,
    batched_arena_requested,
    build_opening_suite,
    display_name_from_folder,
    format_seat_record,
    load_opening_board,
    opening_book_candidates,
    play_opening_games_by_seat,
    search_args,
    select_legal_action,
    validate_batched_arena_args,
)
from utils import dotdict


class FirstPlayerWinsGame:
    def getInitBoard(self):
        return np.array([0], dtype=int)

    def getCanonicalForm(self, board, player):
        return board

    def getGameEnded(self, board, player):
        if board[0] >= 1:
            return -1
        return 0

    def getNextState(self, board, player, action):
        return np.array([board[0] + 1], dtype=int), -player

    def getValidMoves(self, board, player):
        return np.array([1], dtype=int)


class TinyActionGame:
    def getValidMoves(self, board, player):
        return np.array([1, 1, 0], dtype=int)


class ErrorParser:
    def error(self, message):
        raise ValueError(message)


class TestPitSantoriniOpening(unittest.TestCase):
    def make_opening_payload(self):
        return {
            "metadata": {"board_size": 5},
            "player1_choices": [
                {
                    "player1": ["A1", "B1"],
                    "player1_rank": 1,
                    "minimax_value": 0.1,
                    "responses": [
                        {
                            "id": 7,
                            "player2": ["C1", "D1"],
                            "player2_response_rank": 1,
                            "value_mean": 0.1,
                            "pieces": [
                                [1, 2, -1, -2, 0],
                                [0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0],
                            ],
                        },
                    ],
                },
            ],
        }

    def make_opening_args(self, **overrides):
        args = dotdict({
            'checkpoint_folder': './temp/checkpoints',
            'opponent_checkpoint_folder': None,
            'opening_book_path': None,
            'arena_opening_suite': None,
            'opening_id': None,
            'opening_source': 'book',
            'no_opening_book': False,
            'arena_opening_top_fraction': 0.50,
            'arena_opening_max_abs_value': 0.14,
            'no_opening_random_orientation': True,
            'opening_seed': 20260715,
            'games': 4,
            'arena_batch_size': 1,
            'action_temp': 0.0,
            'opponent_sims': None,
        })
        args.update(overrides)
        return args

    def test_load_opening_board_uses_response_id(self):
        payload = self.make_opening_payload()

        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "opening_book.json")
            with open(path, "w") as book_file:
                json.dump(payload, book_file)

            board, position = load_opening_board(path, 7)

        self.assertEqual(position["id"], 7)
        self.assertEqual(board.shape, (2, 5, 5))
        self.assertTrue(np.array_equal(board[0], np.array(payload["player1_choices"][0]["responses"][0]["pieces"])))
        self.assertEqual(int(np.sum(board[1])), 0)

    def test_opening_book_candidates_include_checkpoint_relative_book(self):
        args = self.make_opening_args(checkpoint_folder="/kaggle/working/Santorini-AZ/checkpoints")

        candidates = opening_book_candidates(args)

        self.assertIn(
            "/kaggle/working/Santorini-AZ/opening_books/checkpoints/opening_book.json",
            candidates,
        )

    def test_build_opening_suite_samples_paired_book_openings(self):
        payload = self.make_opening_payload()

        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "opening_book.json")
            with open(path, "w") as book_file:
                json.dump(payload, book_file)

            args = self.make_opening_args(opening_book_path=path, games=4)
            with redirect_stdout(StringIO()):
                opening_book_path, opening_boards, opening_position, opening_mode = build_opening_suite(args)

        self.assertEqual(opening_book_path, path)
        self.assertEqual(len(opening_boards), 2)
        self.assertIsNone(opening_position)
        self.assertEqual(opening_mode, 'sampled_book')
        self.assertEqual(opening_boards[0].shape, (2, 5, 5))

    def test_build_opening_suite_samples_unique_openings(self):
        args = self.make_opening_args(opening_source='unique', games=4)

        with redirect_stdout(StringIO()):
            opening_book_path, opening_boards, opening_position, opening_mode = build_opening_suite(args)
            _, repeated_boards, _, _ = build_opening_suite(args)

        self.assertIsNone(opening_book_path)
        self.assertEqual(len(opening_boards), 2)
        self.assertIsNone(opening_position)
        self.assertEqual(opening_mode, 'sampled_unique')
        self.assertEqual(opening_boards[0].shape, (2, 5, 5))
        self.assertEqual(int(np.count_nonzero(opening_boards[0][0])), 4)
        self.assertEqual(int(np.sum(opening_boards[0][1])), 0)
        self.assertEqual(len({board.tobytes() for board in opening_boards}), 2)
        for board, repeated in zip(opening_boards, repeated_boards):
            np.testing.assert_array_equal(board, repeated)

    def test_build_opening_suite_can_use_game_random_starts(self):
        args = self.make_opening_args(opening_source='game', games=4)

        with redirect_stdout(StringIO()):
            opening_book_path, opening_boards, opening_position, opening_mode = build_opening_suite(args)

        self.assertIsNone(opening_book_path)
        self.assertIsNone(opening_boards)
        self.assertIsNone(opening_position)
        self.assertEqual(opening_mode, 'random_start')

    def test_build_opening_suite_rejects_opening_id_with_unique_source(self):
        args = self.make_opening_args(opening_source='unique', opening_id=7)

        with self.assertRaisesRegex(ValueError, "opening-source book"):
            build_opening_suite(args)

    def test_build_opening_suite_samples_explicit_arena_suite(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "arena_suite.json")
            with open(path, "w") as suite_file:
                json.dump({
                    "metadata": {"name": "test"},
                    "positions": [
                        {
                            "id": 7,
                            "player1": ["A1", "B1"],
                            "player2": ["C1", "D1"],
                            "player1_rank": 1,
                            "player2_response_rank": 1,
                            "value_mean": 0.1,
                            "pieces": [
                                [1, 2, -1, -2, 0],
                                [0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0],
                            ],
                        },
                    ],
                }, suite_file)

            args = self.make_opening_args(arena_opening_suite=path, games=4)
            with redirect_stdout(StringIO()):
                opening_book_path, opening_boards, opening_position, opening_mode = build_opening_suite(args)

        self.assertEqual(opening_book_path, path)
        self.assertEqual(len(opening_boards), 2)
        self.assertIsNone(opening_position)
        self.assertEqual(opening_mode, 'sampled_suite')
        self.assertEqual(opening_boards[0].shape, (2, 5, 5))

    def test_build_opening_suite_accepts_suite_via_opening_book_path(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "arena_suite.json")
            with open(path, "w") as suite_file:
                json.dump({
                    "positions": [
                        {
                            "id": 7,
                            "player1": ["A1", "B1"],
                            "player2": ["C1", "D1"],
                            "player1_rank": 1,
                            "player2_response_rank": 1,
                            "value_mean": 0.1,
                            "pieces": [
                                [1, 2, -1, -2, 0],
                                [0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0],
                            ],
                        },
                    ],
                }, suite_file)

            args = self.make_opening_args(opening_book_path=path, games=4)
            with redirect_stdout(StringIO()):
                _, opening_boards, _, opening_mode = build_opening_suite(args)

        self.assertEqual(len(opening_boards), 2)
        self.assertEqual(opening_mode, 'sampled_suite')

    def test_load_opening_board_rejects_unknown_id(self):
        payload = {
            "player1_choices": [
                {
                    "player1": ["A1", "B1"],
                    "player1_rank": 1,
                    "minimax_value": 0.2,
                    "responses": [
                        {
                            "id": 7,
                            "player2": ["C1", "D1"],
                            "player2_response_rank": 1,
                            "value_mean": 0.2,
                            "pieces": [[0, 0], [0, 0]],
                        },
                    ],
                },
            ],
        }

        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "opening_book.json")
            with open(path, "w") as book_file:
                json.dump(payload, book_file)

            with self.assertRaises(ValueError):
                load_opening_board(path, 99)

    def test_play_opening_games_by_seat_tracks_each_contestant_and_seat(self):
        game = FirstPlayerWinsGame()
        opening_board = np.array([0], dtype=int)
        player = lambda board: 0

        one_won, two_won, draws, seat_stats = play_opening_games_by_seat(
            player,
            player,
            game,
            opening_board,
            games=4,
            contestant1_name="model-a",
            contestant2_name="model-b",
            show_progress=False,
        )

        self.assertEqual((one_won, two_won, draws), (2, 2, 0))
        self.assertEqual(seat_stats["contestant1"]["name"], "model-a")
        self.assertEqual(seat_stats["contestant2"]["name"], "model-b")
        self.assertEqual(seat_stats["contestant1"]["first_player"], {"wins": 2, "losses": 0, "draws": 0})
        self.assertEqual(seat_stats["contestant1"]["second_player"], {"wins": 0, "losses": 2, "draws": 0})
        self.assertEqual(seat_stats["contestant2"]["first_player"], {"wins": 2, "losses": 0, "draws": 0})
        self.assertEqual(seat_stats["contestant2"]["second_player"], {"wins": 0, "losses": 2, "draws": 0})

    def test_format_seat_record_can_hide_draws(self):
        record = {"wins": 2, "losses": 1, "draws": 0}

        self.assertEqual(format_seat_record(record, include_draws=False), {"wins": 2, "losses": 1})
        self.assertEqual(format_seat_record(record, include_draws=True), record)

    def test_display_name_from_folder_uses_trailing_folder_name(self):
        self.assertEqual(
            display_name_from_folder("./temp/santorini-kaggle-training6/"),
            "santorini-kaggle-training6",
        )

    def test_select_legal_action_can_sample_from_masked_probs(self):
        game = TinyActionGame()
        board = np.array([0], dtype=int)

        self.assertEqual(select_legal_action(game, board, [0.1, 0.9, 1.0]), 1)
        self.assertEqual(select_legal_action(game, board, [0.0, 1.0, 0.0], sample=True), 1)

    def test_loaded_network_player_resets_mcts_for_each_game(self):
        player = NetworkMCTSPlayer(TinyActionGame(), object(), sims=4)

        player.startGame()
        first_tree = player.mcts
        player.startGame()

        self.assertIsNot(player.mcts, first_tree)

    def test_batched_arena_validation_requires_two_checkpoint_pit(self):
        args = self.make_opening_args(arena_batch_size=4)

        with self.assertRaisesRegex(ValueError, "opponent-checkpoint-folder"):
            validate_batched_arena_args(ErrorParser(), args)

    def test_batched_arena_validation_requires_deterministic_play(self):
        parser = ErrorParser()
        args = self.make_opening_args(
            arena_batch_size=4,
            opponent_checkpoint_folder="./temp/opponent",
            action_temp=0.5,
        )
        with self.assertRaisesRegex(ValueError, "action-temp 0"):
            validate_batched_arena_args(parser, args)

        args = self.make_opening_args(
            arena_batch_size=4,
            opponent_checkpoint_folder="./temp/opponent",
            sims=25,
            opponent_sims=50,
        )
        validate_batched_arena_args(parser, args)

    def test_placement_only_comparison_requires_empty_board_and_controller(self):
        base = {
            'placement_only_comparison': True,
            'arena_batch_size': 4,
            'opponent_checkpoint_folder': './temp/opponent',
            'standard_controller_folder': './temp/controller',
            'opening_source': 'game',
            'architecture': 'v3',
            'opponent_architecture': 'v3',
            'action_temp': 0.0,
        }
        validate_batched_arena_args(ErrorParser(), self.make_opening_args(**base))

        with self.assertRaisesRegex(ValueError, "opening-source game"):
            validate_batched_arena_args(
                ErrorParser(),
                self.make_opening_args(**dict(base, opening_source='unique')),
            )

        with self.assertRaisesRegex(ValueError, "standard-controller-folder"):
            validate_batched_arena_args(
                ErrorParser(),
                self.make_opening_args(**dict(base, standard_controller_folder=None)),
            )

    def test_search_args_keep_contestant_modes_independent(self):
        puct = search_args(96, 'puct')
        gumbel = search_args(32, 'gumbel', 8, 0.0, 1.5)

        self.assertEqual((puct.numMCTSSims, puct.searchMode), (96, 'puct'))
        self.assertEqual(
            (
                gumbel.numMCTSSims,
                gumbel.searchMode,
                gumbel.gumbelMaxConsideredActions,
                gumbel.gumbelScale,
                gumbel.gumbelPlacementScale,
            ),
            (32, 'gumbel', 8, 0.0, 1.5),
        )

    def test_batched_arena_requested_accepts_batch_sizes_above_one(self):
        self.assertFalse(batched_arena_requested(self.make_opening_args(arena_batch_size=1)))
        self.assertTrue(batched_arena_requested(self.make_opening_args(arena_batch_size=2)))


if __name__ == "__main__":
    unittest.main()
