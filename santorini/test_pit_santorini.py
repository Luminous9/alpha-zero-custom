import json
import os
import tempfile
import unittest

import numpy as np

from pit_santorini import (
    display_name_from_folder,
    load_opening_board,
    play_opening_games_by_seat,
    select_legal_action,
)


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


class TestPitSantoriniOpening(unittest.TestCase):
    def test_load_opening_board_uses_response_id(self):
        payload = {
            "metadata": {"board_size": 5},
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

        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "opening_book.json")
            with open(path, "w") as book_file:
                json.dump(payload, book_file)

            board, position = load_opening_board(path, 7)

        self.assertEqual(position["id"], 7)
        self.assertEqual(board.shape, (2, 5, 5))
        self.assertTrue(np.array_equal(board[0], np.array(payload["player1_choices"][0]["responses"][0]["pieces"])))
        self.assertEqual(int(np.sum(board[1])), 0)

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


if __name__ == "__main__":
    unittest.main()
