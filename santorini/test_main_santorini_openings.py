import unittest
import json
import os
import tempfile

from main_santorini import build_opening_sampler
from santorini.SantoriniOpeningBook import SantoriniOpeningSampler, SantoriniRandomOpeningSampler
from utils import dotdict


def make_args(**overrides):
    args = dotdict({
        'opening_source': 'unique',
        'no_opening_book': False,
        'no_opening_random_orientation': True,
        'opening_book': None,
        'arena_opening_suite': None,
        'self_play_opening_max_abs_value': 0.30,
        'self_play_old_filter_probability': 0.70,
        'self_play_value_probability': 0.25,
        'self_play_opening_tail_probability': 0.05,
        'arena_opening_top_fraction': 0.50,
        'arena_opening_max_abs_value': 0.14,
    })
    args.update(overrides)
    return args


def make_coach_args():
    return dotdict({
        'checkpoint': './temp/santorini_test',
        'load_folder_file': ('./temp/santorini_test', 'best.pth.tar'),
    })


class TestMainSantoriniOpenings(unittest.TestCase):
    def test_default_opening_sampler_uses_unique_random_positions(self):
        sampler = build_opening_sampler(make_args(), make_coach_args())

        self.assertIsInstance(sampler, SantoriniRandomOpeningSampler)
        self.assertEqual(len(sampler.positions), 9664)

    def test_no_opening_book_alias_uses_game_initial_board(self):
        sampler = build_opening_sampler(
            make_args(no_opening_book=True),
            make_coach_args(),
        )

        self.assertIsNone(sampler)

    def test_opening_source_game_uses_game_initial_board(self):
        sampler = build_opening_sampler(
            make_args(opening_source='game'),
            make_coach_args(),
        )

        self.assertIsNone(sampler)

    def test_explicit_opening_book_uses_book_sampler(self):
        payload = {
            "player1_choices": [
                {
                    "player1": ["A1", "B1"],
                    "player1_rank": 1,
                    "minimax_value": 0.0,
                    "responses": [
                        {
                            "id": 1,
                            "player2": ["C1", "D1"],
                            "player2_response_rank": 1,
                            "value_mean": 0.0,
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

            sampler = build_opening_sampler(
                make_args(opening_book=path),
                make_coach_args(),
            )

        self.assertIsInstance(sampler, SantoriniOpeningSampler)


if __name__ == "__main__":
    unittest.main()
