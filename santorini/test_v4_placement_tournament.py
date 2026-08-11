import hashlib
import os
import tempfile
import unittest

import numpy as np

from santorini.D4Canonical import canonicalize_board
from santorini.SantoriniGame import SantoriniGame
from santorini.V4PlacementTournament import (
    PlacementChoice,
    PlacementPolicyTeacher,
    build_completed_opening,
    deterministic_block_seed,
    summarize_paired_records,
)


class FirstLegalTeacher:
    def __init__(self, game, name):
        self.game = game
        self.name = name

    def choose(self, canonical_board, mode, rng=None):
        action = int(np.flatnonzero(self.game.getValidMoves(canonical_board, 1))[0])
        return PlacementChoice(action, 1.0, "synthetic")


class V4PlacementTournamentTests(unittest.TestCase):
    def setUp(self):
        self.game = SantoriniGame(5, sequential_placement=True)

    def test_component_policy_projects_empty_board_stabilizer(self):
        board = self.game.getInitBoard()
        representative, _, key = canonicalize_board(board)
        action = self.game.getPlacementAction((0, 0))
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "teacher.npz")
            np.savez_compressed(
                path,
                action_size=np.asarray([self.game.getActionSize()], dtype=np.int32),
                boards=np.asarray([representative], dtype=np.int8),
                worker_counts=np.asarray([0], dtype=np.int8),
                position_hashes=np.asarray([hashlib.sha256(key).hexdigest()]),
                policy_offsets=np.asarray([0, 1], dtype=np.int64),
                policy_indices=np.asarray([action], dtype=np.uint16),
                policy_values=np.asarray([1.0], dtype=np.float32),
            )
            teacher = PlacementPolicyTeacher(
                self.game, "synthetic", path, require_complete=False
            )
            policy, _ = teacher.distribution(board)
        corners = [
            self.game.getPlacementAction(location)
            for location in ((0, 0), (0, 4), (4, 0), (4, 4))
        ]
        self.assertAlmostEqual(float(policy.sum()), 1.0)
        np.testing.assert_allclose(policy[corners], np.full(4, 0.25))
        self.assertEqual(int(np.count_nonzero(policy)), 4)

    def test_two_teachers_build_a_legal_completed_opening(self):
        one = FirstLegalTeacher(self.game, "one")
        two = FirstLegalTeacher(self.game, "two")
        board, trace = build_completed_opening(
            self.game, one, two, "greedy", seed=7
        )
        self.assertEqual(len(trace), 4)
        self.assertEqual([item["player"] for item in trace], [1, 1, -1, -1])
        self.assertEqual(int(np.sum(board[0] > 0)), 2)
        self.assertEqual(int(np.sum(board[0] < 0)), 2)
        self.assertFalse(self.game.isPlacementPhase(board))

    def test_block_seed_is_stable_and_assignment_specific(self):
        first = deterministic_block_seed(3, "a", "b", "sampled", 4, "a_as_p1")
        self.assertEqual(
            first,
            deterministic_block_seed(3, "a", "b", "sampled", 4, "a_as_p1"),
        )
        self.assertNotEqual(
            first,
            deterministic_block_seed(3, "a", "b", "sampled", 4, "b_as_p1"),
        )

    def test_summary_scores_teacher_a_from_both_seats(self):
        records = [
            {
                "a_as_p1": {"result": 1, "plies": 10, "nodes_visited": 100,
                              "opening_hash": "a"},
                "b_as_p1": {"result": -1, "plies": 12, "nodes_visited": 120,
                              "opening_hash": "b"},
            },
            {
                "a_as_p1": {"result": -1, "plies": 8, "nodes_visited": 80,
                              "opening_hash": "c"},
                "b_as_p1": {"result": -1, "plies": 10, "nodes_visited": 100,
                              "opening_hash": "d"},
            },
        ]
        summary = summarize_paired_records(records, bootstrap_samples=0)
        self.assertEqual(summary["teacher_a_wins"], 3)
        self.assertEqual(summary["teacher_b_wins"], 1)
        self.assertAlmostEqual(summary["teacher_a_score"], 0.75)
        self.assertEqual(summary["a_sweeps"], 1)
        self.assertEqual(summary["split_blocks"], 1)
        self.assertEqual(summary["total_nodes_visited"], 400)


if __name__ == "__main__":
    unittest.main()
