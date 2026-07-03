import unittest

import numpy as np

from play_santorini import (
    OpeningBookPlacementSelector,
    parse_placement,
    place_workers,
)
from santorini.SantoriniOpeningBook import SantoriniOpeningBook


def make_test_book():
    pieces = np.zeros((5, 5), dtype=int)
    pieces[0, 0] = 1
    pieces[0, 1] = 2
    pieces[1, 0] = -1
    pieces[1, 1] = -2
    return SantoriniOpeningBook(
        "test-opening-book.json",
        {"board_size": 5},
        [
            {
                "player1": ["A1", "B1"],
                "player1_locations": [[0, 0], [0, 1]],
                "player1_rank": 1,
                "minimax_value": 0.5,
                "response_count": 1,
                "best_response_id": 1,
                "responses": [
                    {
                        "id": 1,
                        "player2": ["A2", "B2"],
                        "player2_response_rank": 1,
                        "pieces": pieces.tolist(),
                        "value_mean": -0.2,
                    },
                ],
            },
        ],
    )


class TestPlaySantoriniPlacement(unittest.TestCase):
    def test_opening_book_response_maps_back_from_canonical_orientation(self):
        selector = OpeningBookPlacementSelector(make_test_book(), 5)

        response_locations, choice, response = selector.best_response_to_player1([(4, 0), (3, 0)])

        self.assertEqual(choice["player1_rank"], 1)
        self.assertEqual(response["player2_response_rank"], 1)
        self.assertEqual(response_locations, [(4, 1), (3, 1)])

    def test_best_player1_placement_uses_top_ranked_book_choice(self):
        selector = OpeningBookPlacementSelector(make_test_book(), 5)

        locations, choice = selector.best_player1_placement()

        self.assertEqual(choice["player1_rank"], 1)
        self.assertEqual(locations, [(0, 0), (0, 1)])

    def test_parse_placement_rejects_occupied_square(self):
        with self.assertRaisesRegex(ValueError, "already occupied"):
            parse_placement("A1 C3", 5, occupied_locations=[(0, 0)])

    def test_place_workers_uses_player_relative_worker_labels(self):
        board = np.zeros((2, 5, 5), dtype=int)

        board = place_workers(board, -1, [(1, 1), (2, 2)])

        self.assertEqual(board[0, 1, 1], -1)
        self.assertEqual(board[0, 2, 2], -2)


if __name__ == "__main__":
    unittest.main()
