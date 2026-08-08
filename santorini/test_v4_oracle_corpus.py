import json
import os
import tempfile
import unittest

import numpy as np

from santorini.SantoriniGame import SantoriniGame
from santorini.SantoriniOracle import canonical_board_to_fen
from santorini.V4OracleCorpus import (
    load_v4_shard,
    validate_v4_manifest,
    validate_v4_record,
)


class TestV4OracleCorpus(unittest.TestCase):
    def setUp(self):
        self.game = SantoriniGame(5, sequential_placement=True)
        self.manifest = {
            "type": "manifest",
            "schema_version": 1,
            "engine_digest": "test-engine",
            "shard_id": "test-shard",
            "gods": ["mortal", "mortal"],
            "tt_policy": "reset_per_independent_game",
            "generation": {
                "random_moves_min": 0,
                "random_moves_max": 6,
                "requested_node_limit": 100_000,
                "min_depth_node_limit": 20_000,
                "max_completed_depth": 8,
                "subgame_initial_chance": 0.6,
                "seed": 7,
                "worker_index": 0,
                "target_records": 100,
            },
        }

    @staticmethod
    def standard_board():
        board = np.zeros((2, 5, 5), dtype=int)
        board[0, 0, 0] = 1
        board[0, 1, 1] = 2
        board[0, 3, 3] = -1
        board[0, 4, 4] = -2
        board[1, 0, 1] = 1
        board[1, 4, 3] = 2
        return board

    def make_record(self, board, actions, action):
        next_board, next_player = self.game.getNextState(board, 1, action)
        next_canonical = self.game.getCanonicalForm(next_board, next_player)
        return {
            "type": "position",
            "schema_version": 1,
            "engine_digest": "test-engine",
            "shard_id": "test-shard",
            "game_id": "test-shard:0",
            "record_id": "test-shard:0:0",
            "fen": canonical_board_to_fen(board),
            "side_to_move": 1,
            "best_actions": actions,
            "best_action_string": "test",
            "best_successor_fen": canonical_board_to_fen(next_canonical),
            "winner": 1,
            "score": 100,
            "mate_band": False,
            "completed_depth": 4,
            "requested_nodes": 100_000,
            "actual_nodes": 20_100,
            "ply": 3,
            "build_count": int(np.sum(board[1])),
            "random_prefix_plies": 1,
            "source": "main_line",
            "tt_policy": "reset_per_independent_game",
        }

    def test_standard_action_and_successor_validate_across_engines(self):
        board = self.standard_board()
        actions = [
            {"type": "select_worker", "value": "A1"},
            {"type": "move_worker", "value": {"dest": "B1", "meta": None}},
            {"type": "build", "value": "C1"},
        ]
        action = self.game.getActionFromOrigin((0, 0), 4, 4)
        record = self.make_record(board, actions, action)

        aliases = validate_v4_record(self.game, self.manifest, record)
        self.assertEqual(aliases, [action])

    def test_winning_no_build_action_preserves_all_v4_aliases(self):
        board = self.standard_board()
        board[1, 0, 0] = 2
        board[1, 0, 1] = 3
        actions = [
            {"type": "select_worker", "value": "A1"},
            {"type": "move_worker", "value": {"dest": "B1", "meta": None}},
        ]
        first_alias = self.game.getActionFromOrigin((0, 0), 4, 0)
        record = self.make_record(board, actions, first_alias)
        record["score"] = 10_000
        record["mate_band"] = True

        aliases = validate_v4_record(self.game, self.manifest, record)
        self.assertGreater(len(aliases), 1)

    def test_tampered_successor_is_rejected(self):
        board = self.standard_board()
        actions = [
            {"type": "select_worker", "value": "A1"},
            {"type": "move_worker", "value": {"dest": "B1", "meta": None}},
            {"type": "build", "value": "C1"},
        ]
        action = self.game.getActionFromOrigin((0, 0), 4, 4)
        record = self.make_record(board, actions, action)
        record["best_successor_fen"] = canonical_board_to_fen(board)

        with self.assertRaisesRegex(ValueError, "successor FEN"):
            validate_v4_record(self.game, self.manifest, record)

    def test_terminal_no_moves_record_is_rejected(self):
        board = self.standard_board()
        action = self.game.getActionFromOrigin((0, 0), 4, 4)
        record = self.make_record(board, [{"type": "no_moves"}], action)
        with self.assertRaisesRegex(ValueError, "terminal no-moves"):
            validate_v4_record(self.game, self.manifest, record)

    def test_shard_loader_validates_manifest_and_duplicate_ids(self):
        board = self.standard_board()
        actions = [
            {"type": "select_worker", "value": "A1"},
            {"type": "move_worker", "value": {"dest": "B1", "meta": None}},
            {"type": "build", "value": "C1"},
        ]
        action = self.game.getActionFromOrigin((0, 0), 4, 4)
        record = self.make_record(board, actions, action)

        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "shard.jsonl")
            with open(path, "w") as output:
                output.write(json.dumps(self.manifest) + "\n")
                output.write(json.dumps(record) + "\n")
            manifest, records = load_v4_shard(path, game=self.game)
            self.assertEqual(manifest["shard_id"], "test-shard")
            self.assertEqual(len(records), 1)

            with open(path, "a") as output:
                output.write(json.dumps(record) + "\n")
            with self.assertRaisesRegex(ValueError, "duplicate record ids"):
                load_v4_shard(path, game=self.game)

    def test_manifest_rejects_non_mortal_or_warm_tt_data(self):
        non_mortal = dict(self.manifest, gods=["mortal", "apollo"])
        with self.assertRaisesRegex(ValueError, "Mortal-vs-Mortal"):
            validate_v4_manifest(non_mortal)
        warm = dict(self.manifest, tt_policy="persistent")
        with self.assertRaisesRegex(ValueError, "reset-per-independent-game"):
            validate_v4_manifest(warm)

    def test_shard_rejects_conflicting_winners_within_one_game(self):
        board = self.standard_board()
        actions = [
            {"type": "select_worker", "value": "A1"},
            {"type": "move_worker", "value": {"dest": "B1", "meta": None}},
            {"type": "build", "value": "C1"},
        ]
        action = self.game.getActionFromOrigin((0, 0), 4, 4)
        first = self.make_record(board, actions, action)
        second = dict(first, record_id="test-shard:0:1", winner=2)
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "shard.jsonl")
            with open(path, "w") as output:
                output.write(json.dumps(self.manifest) + "\n")
                output.write(json.dumps(first) + "\n")
                output.write(json.dumps(second) + "\n")
            with self.assertRaisesRegex(ValueError, "disagree about the winner"):
                load_v4_shard(path)


if __name__ == "__main__":
    unittest.main()
