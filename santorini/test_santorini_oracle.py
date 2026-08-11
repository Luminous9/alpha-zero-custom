import os
import unittest

import numpy as np

from santorini.SantoriniGame import SantoriniGame
from santorini.SantoriniOracle import (
    DEFAULT_ORACLE_BINARY,
    SantoriniOracleProcess,
    anonymous_board_key,
    canonical_board_to_fen,
    compare_legal_successors,
    external_actions_to_v3_actions,
    external_joint_placement_locations,
    fen_to_canonical_board,
)


class TestSantoriniOracleAdapter(unittest.TestCase):
    def setUp(self):
        self.game = SantoriniGame(5, sequential_placement=True)

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

    def test_fen_round_trip_flips_height_rows_and_ignores_worker_labels(self):
        board = self.standard_board()
        fen = canonical_board_to_fen(board)
        self.assertTrue(fen.startswith("0002000000000000000001000/1/"))
        decoded = fen_to_canonical_board(fen)
        self.assertEqual(anonymous_board_key(decoded), anonymous_board_key(board))

    def test_joint_placement_boundaries_round_trip(self):
        empty = self.game.getInitBoard()
        empty_fen = canonical_board_to_fen(empty)
        self.assertEqual(empty_fen, "0" * 25 + "/1/mortal:/mortal:")
        self.assertEqual(
            anonymous_board_key(fen_to_canonical_board(empty_fen)),
            anonymous_board_key(empty),
        )

        board, player = self.game.getNextState(
            empty, 1, self.game.getPlacementAction((0, 0))
        )
        with self.assertRaises(ValueError):
            canonical_board_to_fen(board)
        board, player = self.game.getNextState(
            board, player, self.game.getPlacementAction((1, 1))
        )
        boundary = self.game.getCanonicalForm(board, player)
        fen = canonical_board_to_fen(boundary)
        self.assertEqual(fen, "0" * 25 + "/2/mortal:A1,B2/mortal:")
        self.assertEqual(
            anonymous_board_key(fen_to_canonical_board(fen)),
            anonymous_board_key(boundary),
        )

    def test_decodes_unordered_joint_placement(self):
        actions = [
            {"type": "place_worker", "value": "E5"},
            {"type": "place_worker", "value": "A1"},
        ]
        self.assertEqual(
            external_joint_placement_locations(actions),
            ((0, 0), (4, 4)),
        )

    def test_maps_standard_move_build_action(self):
        board = self.standard_board()
        actions = [
            {"type": "select_worker", "value": "A1"},
            {"type": "move_worker", "value": {"dest": "B1", "meta": None}},
            {"type": "build", "value": "C1"},
        ]
        mapped = external_actions_to_v3_actions(self.game, board, actions)
        expected = self.game.getActionFromOrigin((0, 0), 4, 4)
        self.assertEqual(mapped, [expected])

    def test_maps_winning_move_to_all_legal_build_aliases(self):
        board = self.standard_board()
        board[1, 0, 0] = 2
        board[1, 0, 1] = 3
        actions = [
            {"type": "select_worker", "value": "A1"},
            {"type": "move_worker", "value": {"dest": "B1", "meta": None}},
        ]
        mapped = external_actions_to_v3_actions(self.game, board, actions)
        self.assertGreater(len(mapped), 1)
        successor_keys = set()
        for action in mapped:
            next_board, next_player = self.game.getNextState(board, 1, action)
            successor_keys.add(
                anonymous_board_key(self.game.getCanonicalForm(next_board, next_player))
            )
        self.assertEqual(len(successor_keys), 1)


@unittest.skipUnless(os.path.isfile(DEFAULT_ORACLE_BINARY), "native oracle is not built")
class TestSantoriniOracleIntegration(unittest.TestCase):
    def test_fixed_node_analysis_returns_a_legal_v3_action(self):
        game = SantoriniGame(5, sequential_placement=True)
        board = TestSantoriniOracleAdapter.standard_board()
        with SantoriniOracleProcess() as oracle:
            response = oracle.analyze(board, nodes=2_000)
        mapped = external_actions_to_v3_actions(
            game,
            board,
            response["best_move"]["actions"],
        )
        valids = game.getValidMoves(board, 1)
        self.assertTrue(mapped)
        self.assertTrue(all(valids[action] for action in mapped))
        self.assertGreaterEqual(response["nodes_visited"], 2_000)

    def test_ranked_root_moves_map_to_distinct_legal_successors(self):
        game = SantoriniGame(5, sequential_placement=True)
        board = TestSantoriniOracleAdapter.standard_board()
        with SantoriniOracleProcess() as oracle:
            response = oracle.analyze_root_moves(board, nodes_per_move=200, top_k=4)

        self.assertEqual(response["command"], "analyze_root_moves")
        self.assertGreaterEqual(response["legal_move_count"], len(response["moves"]))
        self.assertEqual(len(response["moves"]), 4)
        scores = [move["score"] for move in response["moves"]]
        self.assertEqual(scores, sorted(scores, reverse=True))
        successors = set()
        for move in response["moves"]:
            actions = external_actions_to_v3_actions(game, board, move["actions"])
            self.assertTrue(actions)
            successors.add(move["next_fen"])
        self.assertEqual(len(successors), len(response["moves"]))

    def test_empty_board_ranked_placement_has_300_unordered_pairs(self):
        game = SantoriniGame(5, sequential_placement=True)
        with SantoriniOracleProcess() as oracle:
            response = oracle.analyze_root_moves(
                game.getInitBoard(), nodes_per_move=2, top_k=4
            )
        self.assertEqual(response["legal_move_count"], 300)
        self.assertEqual(response["tt_policy"], "reset_per_root_move")
        self.assertEqual(len(response["moves"]), 4)
        for move in response["moves"]:
            self.assertEqual(len(external_joint_placement_locations(move["actions"])), 2)

    def test_legal_successors_match_on_reachable_positions(self):
        game = SantoriniGame(5, sequential_placement=True)
        board = TestSantoriniOracleAdapter.standard_board()
        rng = np.random.RandomState(19)

        with SantoriniOracleProcess() as oracle:
            compared = 0
            cur_player = 1
            while compared < 12 and game.getGameEnded(board, cur_player) == 0:
                canonical = game.getCanonicalForm(board, cur_player)
                comparison = compare_legal_successors(game, canonical, oracle)
                self.assertTrue(
                    comparison["matches"],
                    "successor mismatch: ours={} theirs={}".format(
                        comparison["ours_count"], comparison["theirs_count"]
                    ),
                )
                valids = np.flatnonzero(game.getValidMoves(canonical, 1))
                action = int(rng.choice(valids))
                board, cur_player = game.getNextState(board, cur_player, action)
                compared += 1
            self.assertGreaterEqual(compared, 4)


if __name__ == "__main__":
    unittest.main()
