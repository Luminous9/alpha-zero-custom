import json
import os
import tempfile
import unittest

import numpy as np

from santorini.SantoriniOpeningBook import (
    SantoriniOpeningBook,
    SantoriniOpeningSampler,
    random_board_orientation,
)


def sample_book_payload():
    return {
        "metadata": {"board_size": 5},
        "player1_choices": [
            {
                "player1": ["A1", "B1"],
                "player1_locations": [[0, 0], [0, 1]],
                "player1_rank": 1,
                "minimax_value": 0.2,
                "responses": [
                    {
                        "id": 1,
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
                    {
                        "id": 2,
                        "player2": ["E1", "A2"],
                        "player2_response_rank": 2,
                        "value_mean": 0.9,
                        "pieces": [
                            [1, 2, 0, 0, -1],
                            [-2, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                        ],
                    },
                ],
            },
            {
                "player1": ["B2", "C2"],
                "player1_locations": [[1, 1], [1, 2]],
                "player1_rank": 2,
                "minimax_value": 0.1,
                "responses": [
                    {
                        "id": 3,
                        "player2": ["D2", "E2"],
                        "player2_response_rank": 1,
                        "value_mean": 0.1,
                        "pieces": [
                            [0, 0, 0, 0, 0],
                            [0, 1, 2, -1, -2],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                        ],
                    },
                ],
            },
        ],
    }


class TestSantoriniOpeningSampler(unittest.TestCase):
    def write_book(self, folder):
        path = os.path.join(folder, "opening_book.json")
        with open(path, "w") as book_file:
            json.dump(sample_book_payload(), book_file)
        return path

    def test_loads_flattened_positions_and_best_responses(self):
        with tempfile.TemporaryDirectory() as folder:
            book = SantoriniOpeningBook.load(self.write_book(folder))

            self.assertEqual(len(book.positions), 3)
            self.assertEqual(len(book.best_response_positions), 2)

    def test_self_play_sampling_filters_lopsided_positions(self):
        with tempfile.TemporaryDirectory() as folder:
            sampler = SantoriniOpeningSampler.load(
                self.write_book(folder),
                self_play_max_abs_value=0.3,
                self_play_tail_probability=0.0,
                random_orientation=False,
                rng=np.random.RandomState(3),
            )

            for _ in range(20):
                board = sampler.sample_self_play_board()
                self.assertFalse(np.array_equal(board[0], np.array(sample_book_payload()["player1_choices"][0]["responses"][1]["pieces"])))

    def test_arena_suite_prefers_best_responses_from_top_choices(self):
        with tempfile.TemporaryDirectory() as folder:
            sampler = SantoriniOpeningSampler.load(
                self.write_book(folder),
                arena_top_fraction=0.5,
                arena_max_abs_value=0.5,
                random_orientation=False,
                rng=np.random.RandomState(4),
            )

            suite = sampler.sample_arena_suite(3)

            self.assertEqual(len(suite), 3)
            for board in suite:
                self.assertEqual(board.shape, (2, 5, 5))
                self.assertEqual(int(np.count_nonzero(board[0])), 4)

    def test_random_orientation_preserves_piece_multiset(self):
        board = np.zeros((2, 5, 5), dtype=int)
        board[0, 0, 0] = 1
        board[0, 0, 1] = 2
        board[0, 1, 0] = -1
        board[0, 1, 1] = -2

        transformed = random_board_orientation(board, np.random.RandomState(7))

        self.assertEqual(sorted(np.ravel(board[0]).tolist()), sorted(np.ravel(transformed[0]).tolist()))
        self.assertEqual(int(np.sum(transformed[1])), 0)


if __name__ == "__main__":
    unittest.main()
