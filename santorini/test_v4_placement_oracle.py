import unittest

import numpy as np

from santorini.SantoriniGame import SantoriniGame
from santorini.V4Placement import (
    aggregate_teacher_observations,
    factor_joint_placement,
    joint_boundary_orbits,
    legal_unordered_pairs,
    pair_softmax,
    symmetrize_joint_pair_scores,
)


class V4PlacementOracleTests(unittest.TestCase):
    def setUp(self):
        self.game = SantoriniGame(5, sequential_placement=True)

    def test_uniform_joint_pair_factors_exactly(self):
        board = self.game.getInitBoard()
        pairs = legal_unordered_pairs(self.game, board)
        observations = factor_joint_placement(
            self.game, board, pairs, np.zeros(len(pairs)), 100.0
        )
        self.assertEqual(len(pairs), 300)
        self.assertEqual(len(observations), 26)
        placement_actions = [self.game.getPlacementAction(divmod(i, 5)) for i in range(25)]
        np.testing.assert_allclose(
            observations[0].policy[placement_actions], np.full(25, 1 / 25)
        )
        for observation in observations[1:]:
            self.assertAlmostEqual(float(observation.policy.sum()), 1.0)
            nonzero = observation.policy[observation.policy > 0]
            np.testing.assert_allclose(nonzero, np.full(24, 1 / 24))

    def test_factored_sequence_reconstructs_joint_distribution(self):
        board = self.game.getInitBoard()
        pairs = legal_unordered_pairs(self.game, board)
        scores = np.linspace(-200.0, 300.0, len(pairs))
        probabilities = pair_softmax(scores, 75.0)
        observations = factor_joint_placement(
            self.game, board, pairs, scores, 75.0, symmetrize_scores=False
        )
        first_location = (0, 0)
        first_action = self.game.getPlacementAction(first_location)
        partial = next(
            item for item in observations[1:]
            if item.board[0][first_location] > 0
        )
        for pair_index, pair in enumerate(pairs):
            if first_location not in pair:
                continue
            second = pair[1] if pair[0] == first_location else pair[0]
            sequence_probability = (
                observations[0].policy[first_action]
                * partial.policy[self.game.getPlacementAction(second)]
            )
            self.assertAlmostEqual(
                float(sequence_probability), float(probabilities[pair_index] / 2)
            )

    def test_all_joint_boundaries_cover_all_960_sequential_orbits(self):
        observations = []
        boundaries = joint_boundary_orbits(self.game)
        self.assertEqual(len(boundaries), 50)
        for board in boundaries:
            pairs = legal_unordered_pairs(self.game, board)
            observations.extend(factor_joint_placement(
                self.game, board, pairs, np.zeros(len(pairs)), 100.0
            ))
        aggregates = aggregate_teacher_observations(self.game, observations)
        counts = tuple(
            sum(np.count_nonzero(item["board"][0]) == worker_count for item in aggregates.values())
            for worker_count in range(4)
        )
        self.assertEqual(counts, (1, 6, 49, 904))

    def test_pair_score_projection_removes_orientation_noise(self):
        board = self.game.getInitBoard()
        pairs = legal_unordered_pairs(self.game, board)
        projected, diagnostics = symmetrize_joint_pair_scores(
            self.game, board, pairs, np.arange(len(pairs), dtype=np.float64)
        )
        self.assertEqual(diagnostics["pair_orbits"], 49)
        groups = {}
        for pair, score in zip(pairs, projected):
            child, player = self.game.getNextState(
                board, 1, self.game.getPlacementAction(pair[0])
            )
            child, player = self.game.getNextState(
                child, player, self.game.getPlacementAction(pair[1])
            )
            from santorini.D4Canonical import canonicalize_board
            key = canonicalize_board(self.game.getCanonicalForm(child, player))[2]
            groups.setdefault(key, set()).add(float(score))
        self.assertTrue(all(len(values) == 1 for values in groups.values()))


if __name__ == "__main__":
    unittest.main()
