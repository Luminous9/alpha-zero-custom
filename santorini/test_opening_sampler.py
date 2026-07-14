import json
import os
import tempfile
import unittest

import numpy as np

from santorini.SantoriniOpeningBook import (
    SantoriniOpeningBook,
    SantoriniMixedOpeningSampler,
    SantoriniRandomOpeningSampler,
    SantoriniOpeningSampler,
    SantoriniOpeningSuite,
    passes_old_opening_filter,
    random_board_orientation,
    unique_opening_positions,
)


class FixedSampler:
    def __init__(self, value):
        self.value = value

    def sample_self_play_board(self):
        return np.array([self.value], dtype=int)

    def sample_arena_suite(self, count):
        return [np.array([self.value], dtype=int) for _ in range(count)]


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
                        "value_mean": 0.25,
                        "pieces": [
                            [1, 2, 0, 0, -1],
                            [-2, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                        ],
                    },
                    {
                        "id": 4,
                        "player2": ["A2", "E2"],
                        "player2_response_rank": 3,
                        "value_mean": 0.9,
                        "pieces": [
                            [1, 2, 0, 0, -1],
                            [0, 0, 0, 0, -2],
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

            self.assertEqual(len(book.positions), 4)
            self.assertEqual(len(book.best_response_positions), 2)

    def test_unique_opening_positions_match_symmetry_reduced_count(self):
        self.assertEqual(len(unique_opening_positions(5)), 9664)

    def test_random_opening_sampler_draws_unique_starting_boards(self):
        sampler = SantoriniRandomOpeningSampler(
            board_size=5,
            random_orientation=False,
            rng=np.random.RandomState(1),
        )

        board = sampler.sample_self_play_board()
        suite = sampler.sample_arena_suite(3)

        self.assertEqual(board.shape, (2, 5, 5))
        self.assertEqual(int(np.count_nonzero(board[0])), 4)
        self.assertEqual(int(np.sum(board[1])), 0)
        self.assertEqual(len(suite), 3)
        for opening_board in suite:
            self.assertEqual(opening_board.shape, (2, 5, 5))
            self.assertEqual(int(np.count_nonzero(opening_board[0])), 4)

    def test_mixed_opening_sampler_switches_between_sources(self):
        primary = FixedSampler(1)
        unique = FixedSampler(2)

        old_only = SantoriniMixedOpeningSampler(
            primary,
            unique,
            unique_probability=0.0,
            rng=np.random.RandomState(1),
        )
        unique_only = SantoriniMixedOpeningSampler(
            primary,
            unique,
            unique_probability=1.0,
            rng=np.random.RandomState(1),
        )

        self.assertEqual(int(old_only.sample_self_play_board()[0]), 1)
        self.assertEqual([int(board[0]) for board in old_only.sample_arena_suite(3)], [1, 1, 1])
        self.assertEqual(int(unique_only.sample_self_play_board()[0]), 2)
        self.assertEqual([int(board[0]) for board in unique_only.sample_arena_suite(3)], [2, 2, 2])

    def test_mixed_opening_sampler_proportions_arena_suite_by_count(self):
        sampler = SantoriniMixedOpeningSampler(
            FixedSampler(1),
            FixedSampler(2),
            unique_probability=0.25,
            rng=np.random.RandomState(1),
        )

        suite = sampler.sample_arena_suite(8)
        values = [int(board[0]) for board in suite]

        self.assertEqual(values.count(1), 6)
        self.assertEqual(values.count(2), 2)

    def test_self_play_sampling_filters_lopsided_positions(self):
        with tempfile.TemporaryDirectory() as folder:
            sampler = SantoriniOpeningSampler.load(
                self.write_book(folder),
                self_play_max_abs_value=0.3,
                self_play_old_filter_probability=0.0,
                self_play_value_probability=1.0,
                self_play_tail_probability=0.0,
                random_orientation=False,
                rng=np.random.RandomState(3),
            )

            for _ in range(20):
                board = sampler.sample_self_play_board()
                self.assertFalse(np.array_equal(
                    board[0],
                    np.array(sample_book_payload()["player1_choices"][0]["responses"][2]["pieces"]),
                ))

    def test_self_play_sampling_can_use_old_filter_bucket(self):
        with tempfile.TemporaryDirectory() as folder:
            sampler = SantoriniOpeningSampler.load(
                self.write_book(folder),
                self_play_old_filter_probability=1.0,
                self_play_value_probability=0.0,
                self_play_tail_probability=0.0,
                random_orientation=False,
                rng=np.random.RandomState(5),
            )

            for _ in range(20):
                board = sampler.sample_self_play_board()
                self.assertTrue(passes_old_opening_filter(board[0]))

    def test_arena_suite_allows_non_best_responses_from_top_choices(self):
        with tempfile.TemporaryDirectory() as folder:
            sampler = SantoriniOpeningSampler.load(
                self.write_book(folder),
                arena_top_fraction=0.5,
                arena_max_abs_value=0.5,
                random_orientation=False,
                rng=np.random.RandomState(4),
            )

            candidate_ids = [position["id"] for position in sampler._arena_candidates()]
            self.assertEqual(candidate_ids, [1, 2])

            suite = sampler.sample_arena_suite(3)

            self.assertEqual(len(suite), 3)
            for board in suite:
                self.assertEqual(board.shape, (2, 5, 5))
                self.assertEqual(int(np.count_nonzero(board[0])), 4)

    def test_arena_suite_overrides_book_candidates(self):
        with tempfile.TemporaryDirectory() as folder:
            book_path = self.write_book(folder)
            suite_path = os.path.join(folder, "suite.json")
            suite_payload = {
                "metadata": {"name": "test_suite"},
                "positions": [
                    {
                        "id": 4,
                        "player1": ["A1", "B1"],
                        "player2": ["A2", "E2"],
                        "player1_rank": 1,
                        "player2_response_rank": 3,
                        "value_mean": 0.9,
                        "pieces": sample_book_payload()["player1_choices"][0]["responses"][2]["pieces"],
                    },
                ],
            }
            with open(suite_path, "w") as suite_file:
                json.dump(suite_payload, suite_file)

            sampler = SantoriniOpeningSampler.load(
                book_path,
                arena_suite=SantoriniOpeningSuite.load(suite_path),
                random_orientation=False,
                rng=np.random.RandomState(4),
            )

            suite = sampler.sample_arena_suite(3)

            self.assertEqual(len(suite), 3)
            for board in suite:
                self.assertTrue(np.array_equal(
                    board[0],
                    np.array(sample_book_payload()["player1_choices"][0]["responses"][2]["pieces"]),
                ))

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
