import json
import os
import tempfile
import unittest
from collections import deque

import numpy as np

from experiments.santorini_oracle.legacy.generate_santorini_oracle_replay import (
    load_or_initialize_records,
    materialize_teacher_replay,
)
from santorini.OracleResearch import (
    blended_teacher_policy,
    collect_unique_replay_positions,
)
from santorini.ReplayBuffer import load_compact_replay, save_compact_replay
from santorini.SantoriniGame import SantoriniGame


class TestOracleTeacherReplay(unittest.TestCase):
    def setUp(self):
        self.game = SantoriniGame(5, sequential_placement=True)
        board = np.zeros((2, 5, 5), dtype=int)
        board[0, 1, 1] = 1
        board[0, 1, 3] = 2
        board[0, 3, 1] = -1
        board[0, 3, 3] = -2
        self.board = board
        self.valid_actions = np.flatnonzero(self.game.getValidMoves(board, 1))

    def test_blended_policy_distributes_oracle_mass_across_equivalents(self):
        source = np.zeros(self.game.getActionSize(), dtype=np.float32)
        source[self.valid_actions[:3]] = [0.2, 0.3, 0.5]
        oracle_actions = self.valid_actions[3:5]
        blended = blended_teacher_policy(source, oracle_actions, 0.75)

        self.assertAlmostEqual(float(blended.sum()), 1.0, places=6)
        np.testing.assert_allclose(blended[self.valid_actions[:3]], [0.05, 0.075, 0.125])
        np.testing.assert_allclose(blended[oracle_actions], [0.375, 0.375])

    def test_unique_collection_deduplicates_rotated_positions(self):
        policy = np.zeros(self.game.getActionSize(), dtype=np.float32)
        policy[self.valid_actions[0]] = 1.0
        rotated = np.asarray([np.rot90(self.board[0]), np.rot90(self.board[1])])
        with tempfile.TemporaryDirectory() as folder:
            replay = os.path.join(folder, "source.npz")
            save_compact_replay(
                replay,
                [deque([(self.board, policy, 1.0), (rotated, policy, -1.0)])],
            )
            positions = collect_unique_replay_positions(replay)

        self.assertEqual(sum(len(items) for items in positions.values()), 1)
        record = next(items[0] for items in positions.values() if items)
        self.assertEqual(record["replay_index"], 0)
        self.assertEqual(record["replay_observations"], 2)

    def test_materialized_replay_round_trips_augmented_targets(self):
        source = np.zeros(self.game.getActionSize(), dtype=np.float32)
        source[self.valid_actions[:2]] = [0.4, 0.6]
        oracle_action = int(self.valid_actions[2])
        records = [{
            "position_id": 0,
            "replay_index": 0,
            "oracle_action_indices": [oracle_action],
        }]
        with tempfile.TemporaryDirectory() as folder:
            replay = os.path.join(folder, "source.npz")
            output = os.path.join(folder, "teacher.npz")
            save_compact_replay(replay, [deque([(self.board, source, -1.0)])])
            summary = materialize_teacher_replay(
                replay, output, records, oracle_weight=0.75, augment_symmetries=True
            )
            loaded = load_compact_replay(output)

        self.assertEqual(summary["base_positions"], 1)
        self.assertEqual(summary["augmented_examples"], 8)
        self.assertEqual([len(window) for window in loaded], [8])
        self.assertTrue(all(example[2] == -1.0 for example in loaded[0]))
        self.assertTrue(all(abs(float(example[1].sum()) - 1.0) < 1e-6 for example in loaded[0]))
        self.assertAlmostEqual(float(loaded[0][0][1][oracle_action]), 0.75, places=6)

    def test_resume_rejects_changed_teacher_weight(self):
        metadata = {
            "schema_version": 1,
            "replay_path": "/tmp/source",
            "replay_sha256": "abc",
            "output_path": "/tmp/output",
            "positions": 1,
            "oracle_nodes": 10,
            "oracle_weight": 0.75,
            "seed": 4,
            "augment_symmetries": True,
            "selection": [{"position_id": 0}],
            "type": "metadata",
        }
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "records.jsonl")
            self.assertEqual(load_or_initialize_records(path, metadata), [])
            changed = json.loads(json.dumps(metadata))
            changed["oracle_weight"] = 1.0
            with self.assertRaises(ValueError):
                load_or_initialize_records(path, changed)


if __name__ == "__main__":
    unittest.main()
