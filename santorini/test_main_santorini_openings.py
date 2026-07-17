import unittest
import json
import os
import tempfile
from unittest.mock import patch

from main_santorini import (
    build_coach_args,
    build_opening_sampler,
    parse_args,
    parse_lr_schedule,
    resolve_anchor_checkpoint_path,
)
from santorini.SantoriniOpeningBook import (
    SantoriniMixedOpeningSampler,
    SantoriniOpeningSampler,
    SantoriniRandomOpeningSampler,
)
from utils import dotdict


def make_args(**overrides):
    args = dotdict({
        'opening_source': 'mixed',
        'no_opening_book': False,
        'no_opening_random_orientation': True,
        'opening_book': None,
        'arena_opening_suite': None,
        'self_play_opening_max_abs_value': 0.30,
        'self_play_old_filter_probability': 1.00,
        'self_play_value_probability': 0.00,
        'self_play_opening_tail_probability': 0.00,
        'opening_mix_unique_probability': 0.20,
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
    def test_tactical_shortcuts_default_on_and_can_be_disabled(self):
        with patch('sys.argv', ['main_santorini.py', '--architecture', 'v3']):
            enabled = build_coach_args(parse_args())
        with patch(
            'sys.argv',
            ['main_santorini.py', '--architecture', 'v3', '--no-tactical-shortcuts'],
        ):
            disabled = build_coach_args(parse_args())

        self.assertTrue(enabled.tacticalShortcuts)
        self.assertFalse(disabled.tacticalShortcuts)

    def test_playout_cap_randomization_configuration(self):
        with patch(
            'sys.argv',
            [
                'main_santorini.py',
                '--architecture', 'v3',
                '--num-mcts-sims', '96',
                '--playout-cap-randomization',
                '--playout-cap-full-probability', '0.25',
                '--playout-cap-fast-sims', '32',
            ],
        ):
            parsed = parse_args()

        coach_args = build_coach_args(parsed)
        self.assertTrue(coach_args.playoutCapRandomization)
        self.assertEqual(coach_args.playoutCapFullProbability, 0.25)
        self.assertEqual(coach_args.playoutCapFastSims, 32)
        self.assertTrue(coach_args.playoutCapFullPlacement)

    def test_playout_cap_fast_search_must_be_smaller_than_full_search(self):
        with patch(
            'sys.argv',
            [
                'main_santorini.py',
                '--num-mcts-sims', '32',
                '--playout-cap-randomization',
                '--playout-cap-fast-sims', '32',
            ],
        ), self.assertRaises(SystemExit):
            parse_args()

    def test_v3_defaults_to_fresh_replay_reuse_and_validation(self):
        with patch('sys.argv', ['main_santorini.py', '--architecture', 'v3']):
            parsed = parse_args()

        coach_args = build_coach_args(parsed)
        self.assertEqual(coach_args.replayReuse, 16.0)
        self.assertEqual(coach_args.validationFraction, 0.05)

    def test_v2_retains_legacy_epoch_schedule_without_validation(self):
        with patch('sys.argv', ['main_santorini.py', '--architecture', 'v2']):
            parsed = parse_args()

        coach_args = build_coach_args(parsed)
        self.assertIsNone(coach_args.replayReuse)
        self.assertEqual(coach_args.validationFraction, 0.0)

    def test_learning_rate_schedule_parser(self):
        self.assertEqual(parse_lr_schedule('200:0.0001,400:0.00003'), [(200, 1e-4), (400, 3e-5)])
        self.assertEqual(parse_lr_schedule('none'), [])

    def test_v3_defaults_to_temperature_one_policy_targets(self):
        with patch('sys.argv', ['main_santorini.py', '--architecture', 'v3']):
            parsed = parse_args()

        self.assertEqual(build_coach_args(parsed).policyTargetTemperature, 1.0)

    def test_v2_retains_action_temperature_policy_targets(self):
        with patch('sys.argv', ['main_santorini.py', '--architecture', 'v2']):
            parsed = parse_args()

        self.assertIsNone(build_coach_args(parsed).policyTargetTemperature)

    def test_policy_target_temperature_can_be_overridden(self):
        with patch(
            'sys.argv',
            ['main_santorini.py', '--architecture', 'v3', '--policy-target-temperature', '0.5'],
        ):
            parsed = parse_args()

        self.assertEqual(build_coach_args(parsed).policyTargetTemperature, 0.5)

    def test_v3_defaults_to_twenty_iteration_milestones(self):
        with patch('sys.argv', ['main_santorini.py', '--architecture', 'v3']):
            parsed = parse_args()

        self.assertEqual(build_coach_args(parsed).milestoneInterval, 20)

    def test_v2_retains_ten_iteration_milestone_default(self):
        with patch('sys.argv', ['main_santorini.py', '--architecture', 'v2']):
            parsed = parse_args()

        self.assertEqual(build_coach_args(parsed).milestoneInterval, 10)

    def test_anchor_checkpoint_directory_prefers_single_best_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            nested = os.path.join(folder, 'dataset')
            os.makedirs(nested)
            best = os.path.join(nested, 'best.pth.tar')
            latest = os.path.join(nested, 'latest.pth.tar')
            open(best, 'wb').close()
            open(latest, 'wb').close()

            self.assertEqual(resolve_anchor_checkpoint_path(folder), best)

    def test_default_opening_sampler_mixes_book_filter_and_unique_random_positions(self):
        sampler = build_opening_sampler(make_args(), make_coach_args())

        self.assertIsInstance(sampler, SantoriniMixedOpeningSampler)
        self.assertIsInstance(sampler.primary_sampler, SantoriniOpeningSampler)
        self.assertIsInstance(sampler.unique_sampler, SantoriniRandomOpeningSampler)
        self.assertAlmostEqual(sampler.unique_probability, 0.20)
        self.assertEqual(len(sampler.unique_sampler.positions), 9664)

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

    def test_opening_source_book_uses_book_sampler(self):
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
                make_args(opening_source='book', opening_book=path),
                make_coach_args(),
            )

        self.assertIsInstance(sampler, SantoriniOpeningSampler)


if __name__ == "__main__":
    unittest.main()
