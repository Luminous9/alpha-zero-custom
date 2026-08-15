import types
import unittest

import numpy as np
import torch

from Coach import Coach
from santorini.D4Canonical import canonicalize_board_policies
from santorini.SantoriniGame import SantoriniGame
from santorini.V4ReplaySampling import (
    _aggregate_window_placements,
    prepare_v4_replay,
)
from utils import dotdict


class DummyMCTS:
    _placement_scale_bucket = 'base'


class TestV4ReplaySampling(unittest.TestCase):
    def setUp(self):
        self.game = SantoriniGame(5, sequential_placement=True)

    def _position(self, locations):
        board = self.game.getInitBoard()
        player = 1
        for location in locations:
            board, player = self.game.getNextState(
                board, player, self.game.getPlacementAction(location)
            )
        return self.game.getCanonicalForm(board, player)

    def _example(self, board, value=1.0, action=None):
        valids = self.game.getValidMoves(board, 1).astype(np.float32)
        policy = np.zeros(self.game.getActionSize(), dtype=np.float32)
        if action is None:
            policy = valids / valids.sum()
        else:
            self.assertTrue(valids[action])
            policy[action] = 1.0
        return board, policy, float(value)

    def test_balanced_view_has_equal_placement_ply_mass(self):
        placement_boards = [
            self._position([]),
            self._position([(0, 0)]),
            self._position([(0, 0), (4, 4)]),
            self._position([(0, 0), (4, 4), (0, 4)]),
        ]
        standard = self._position([(0, 0), (4, 4), (0, 4), (4, 0)])
        history = []
        for _ in range(2):
            window = []
            for board in placement_boards:
                window.extend(self._example(board) for _ in range(5))
            window.extend(self._example(standard) for _ in range(20))
            history.append(window)

        examples, weights, metrics = prepare_v4_replay(
            self.game, history, placement_fraction=0.15, frequency_exponent=0.5
        )

        self.assertEqual(metrics['replay_raw_examples'], 80)
        self.assertEqual(metrics['replay_raw_placement_examples'], 40)
        self.assertEqual(metrics['replay_aggregated_placement_groups'], 8)
        self.assertAlmostEqual(metrics['replay_placement_sampling_fraction'], 0.15)
        self.assertAlmostEqual(weights.sum(), 1.0)
        for ply in range(4):
            self.assertAlmostEqual(
                metrics['replay_placement_ply_{}_sampling_fraction'.format(ply)],
                0.0375,
            )
        self.assertEqual(len(examples), 48)

    def test_square_root_frequency_softens_duplicate_weight(self):
        common = self._position([(0, 0), (4, 4)])
        rare = self._position([(0, 1), (3, 4)])
        standard = self._position([(0, 0), (4, 4), (0, 4), (4, 0)])
        window = [self._example(common) for _ in range(9)]
        window += [self._example(rare)]
        window += [self._example(standard) for _ in range(20)]

        examples, weights, _ = prepare_v4_replay(
            self.game, [window], placement_fraction=0.15, frequency_exponent=0.5
        )
        placement = [
            (index, example) for index, example in enumerate(examples)
            if len(example) >= 4
            and example[3].get('source') == 'placement_replay_aggregate'
        ]
        self.assertEqual(len(placement), 2)
        by_count = {example[3]['occurrences']: weights[index] for index, example in placement}
        self.assertAlmostEqual(by_count[9] / by_count[1], 3.0)

    def test_count_weighted_aggregate_reproduces_raw_target_gradient(self):
        board = self._position([(0, 0)])
        transformed = self.game.getSymmetries(
            board, np.zeros(self.game.getActionSize(), dtype=np.float32)
        )[3][0]
        valids = np.flatnonzero(self.game.getValidMoves(board, 1))
        policies = []
        values = [-1.0, 1.0, 1.0]
        examples = []
        for index, value in enumerate(values):
            policy = np.zeros(self.game.getActionSize(), dtype=np.float32)
            policy[valids[index]] = 1.0
            if index == 2:
                transformed, policy = self.game.getSymmetries(board, policy)[3]
                examples.append((transformed, policy, value))
            else:
                examples.append((board, policy, value))

        canonical_boards, canonical_policies, _ = canonicalize_board_policies(
            self.game,
            [example[0] for example in examples],
            [example[1] for example in examples],
        )
        _, groups = _aggregate_window_placements(self.game, examples, 0)
        self.assertEqual(len(groups), 1)
        aggregated = groups[0]['example']

        logits_raw = torch.zeros(self.game.getActionSize(), requires_grad=True)
        value_raw = torch.tensor(0.2, requires_grad=True)
        log_policy_raw = torch.log_softmax(logits_raw, dim=0)
        raw_loss = sum(
            -torch.dot(torch.tensor(policy), log_policy_raw)
            + (value_raw - value) ** 2
            for policy, value in zip(canonical_policies, values)
        ) / len(values)
        raw_loss.backward()

        logits_aggregate = torch.zeros(self.game.getActionSize(), requires_grad=True)
        value_aggregate = torch.tensor(0.2, requires_grad=True)
        aggregate_loss = (
            -torch.dot(
                torch.tensor(aggregated[1]),
                torch.log_softmax(logits_aggregate, dim=0),
            )
            + (value_aggregate - aggregated[2]) ** 2
        )
        aggregate_loss.backward()

        torch.testing.assert_close(logits_aggregate.grad, logits_raw.grad)
        torch.testing.assert_close(value_aggregate.grad, value_raw.grad)
        np.testing.assert_array_equal(canonical_boards[0], aggregated[0])

    def test_unique_neural_start_generator_rejects_d4_duplicate(self):
        coach = object.__new__(Coach)
        coach.game = self.game
        coach.args = dotdict({
            'selfPlayBatchSize': 1,
            'uniqueNeuralStartMaxAttemptFactor': 4,
            'placementTemperature': 1.0,
            'playoutCapRandomization': False,
            'numMCTSSims': 1,
        })
        coach._unique_neural_start_stats = coach._newUniqueNeuralStartStats()
        coach._placement_scale_game_counts = {'base': 0, 'exploratory': 0}
        coach._placement_choices = []
        coach._search_symmetry_stats = coach._newSearchSymmetryStats()
        coach._playout_cap_stats = coach._newPlayoutCapStats()
        coach._newSelfPlayMCTS = types.MethodType(
            lambda self, record_placement_scale=True, sample_placement_scale=True: DummyMCTS(),
            coach,
        )
        candidates = [
            [(0, 0), (1, 1), (4, 4), (3, 3)],
            [(0, 4), (1, 3), (4, 0), (3, 1)],
            [(0, 0), (0, 4), (4, 2), (2, 2)],
        ]
        calls = {'count': 0}

        def policies(self, episodes):
            candidate = calls['count'] // 4
            ply = calls['count'] % 4
            calls['count'] += 1
            action = self.game.getPlacementAction(candidates[candidate][ply])
            policy = np.zeros(self.game.getActionSize(), dtype=np.float32)
            policy[action] = 1.0
            return [policy], [policy]

        coach._getBatchedSelfPlayPolicies = types.MethodType(policies, coach)
        openings = coach._generateUniqueNeuralOpenings(2)

        self.assertEqual(len(openings), 2)
        self.assertNotEqual(openings[0]['d4_key'], openings[1]['d4_key'])
        self.assertEqual(coach._unique_neural_start_stats['candidates'], 3)
        self.assertEqual(
            coach._unique_neural_start_stats['d4_duplicates_rejected'], 1
        )
        self.assertEqual(coach._placement_scale_game_counts['base'], 2)
        self.assertEqual(len(coach._placement_choices), 8)


if __name__ == '__main__':
    unittest.main()
