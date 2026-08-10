import unittest
import os
import tempfile
import hashlib

import numpy as np

from build_santorini_v4_corpus import (
    UnionFind,
    _retained_key_indices,
    _split_id,
    canonicalize_board_policy,
    validate_converted_corpus,
)
from santorini.SantoriniGame import SantoriniGame


class TestV4CorpusBuilder(unittest.TestCase):
    def setUp(self):
        self.game = SantoriniGame(5, sequential_placement=True)

    @staticmethod
    def board():
        board = np.zeros((2, 5, 5), dtype=int)
        board[0, 0, 0] = 1
        board[0, 1, 1] = 2
        board[0, 3, 3] = -1
        board[0, 4, 4] = -2
        board[1, 0, 1] = 1
        board[1, 2, 3] = 2
        return board

    def test_d4_canonical_board_and_policy_are_orientation_independent(self):
        board = self.board()
        valid = np.flatnonzero(self.game.getValidMoves(board, 1))
        policy = np.zeros(self.game.getActionSize(), dtype=np.float64)
        policy[valid[:2]] = 0.5
        canonical_board, canonical_policy, canonical_key = canonicalize_board_policy(
            self.game, board, policy
        )

        transformed_board, transformed_policy = self.game.getSymmetries(board, policy)[5]
        other_board, other_policy, other_key = canonicalize_board_policy(
            self.game, transformed_board, transformed_policy
        )
        self.assertEqual(canonical_key, other_key)
        np.testing.assert_array_equal(canonical_board, other_board)
        np.testing.assert_allclose(canonical_policy, other_policy)
        self.assertAlmostEqual(float(canonical_policy.sum()), 1.0)

    def test_union_find_keeps_same_game_positions_in_one_split(self):
        union_find = UnionFind(4)
        union_find.union(0, 2)
        union_find.union(2, 3)
        self.assertEqual(union_find.find(0), union_find.find(3))
        self.assertNotEqual(union_find.find(0), union_find.find(1))

    def test_component_split_is_deterministic(self):
        first = _split_id("component", 7, 0.1, 0.1)
        self.assertEqual(first, _split_id("component", 7, 0.1, 0.1))
        self.assertIn(first, (0, 1, 2))

    def test_excluded_corpus_hashes_remove_only_owned_positions(self):
        keys = [b"first-position", b"second-position"]
        excluded = {hashlib.sha256(keys[0]).hexdigest()}
        np.testing.assert_array_equal(
            _retained_key_indices(keys, excluded), np.asarray([1])
        )

    def test_converted_sparse_payload_validates_frequency_and_policy(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "corpus.npz")
            np.savez_compressed(
                path,
                schema_version=np.asarray([1], dtype=np.int16),
                action_size=np.asarray([self.game.getActionSize()], dtype=np.int32),
                boards=np.zeros((1, 2, 5, 5), dtype=np.int8),
                observation_counts=np.asarray([3], dtype=np.int32),
                winner_means=np.asarray([1 / 3], dtype=np.float32),
                score_means=np.asarray([10], dtype=np.float32),
                score_stddevs=np.asarray([2], dtype=np.float32),
                requested_nodes=np.asarray([100_000], dtype=np.int32),
                actual_nodes_means=np.asarray([90_000], dtype=np.float32),
                mate_rates=np.asarray([0], dtype=np.float32),
                stage_ids=np.asarray([0], dtype=np.int8),
                source_counts=np.asarray([[2, 1]], dtype=np.int32),
                split_ids=np.asarray([0], dtype=np.int8),
                policy_offsets=np.asarray([0, 2], dtype=np.int64),
                policy_indices=np.asarray([1, 2], dtype=np.uint16),
                policy_values=np.asarray([0.25, 0.75], dtype=np.float32),
                position_hashes=np.asarray(["a" * 64], dtype="<U64"),
            )
            result = validate_converted_corpus(path, expected_observations=3)
            self.assertEqual(result["positions"], 1)
            self.assertEqual(result["observations"], 3)


if __name__ == "__main__":
    unittest.main()
