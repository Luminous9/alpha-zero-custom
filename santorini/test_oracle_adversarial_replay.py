import unittest

import numpy as np

from generate_santorini_oracle_adversarial_replay import (
    absolute_player_one_result,
    collect_candidates,
    dense_policy,
    select_phase_balanced,
    sparse_policy,
)
from santorini.SantoriniGame import SantoriniGame


class TestOracleAdversarialReplay(unittest.TestCase):
    def test_terminal_result_uses_the_actual_next_player(self):
        class TerminalGame:
            def __init__(self):
                self.player = None

            def getGameEnded(self, board, player):
                self.player = player
                return -1

        game = TerminalGame()
        result = absolute_player_one_result(game, None, current_player=-1)
        self.assertEqual(game.player, -1)
        self.assertEqual(result, 1.0)

    def test_sparse_policy_round_trip(self):
        game = SantoriniGame(5, sequential_placement=True)
        policy = np.zeros(game.getActionSize(), dtype=np.float32)
        policy[[3, 11]] = [0.25, 0.75]
        position = sparse_policy(policy)
        np.testing.assert_allclose(dense_policy(game, position), policy)

    def test_candidate_deduplication_prefers_loss_value(self):
        board = np.zeros((2, 5, 5), dtype=int)
        board[0, 0, 0] = 1
        board[0, 1, 1] = 2
        board[0, 3, 3] = -1
        board[0, 4, 4] = -2
        base = {
            "board": board.tolist(),
            "chosen_action": 0,
            "ply": 1,
            "stage": "early",
            "policy_actions": [0],
            "policy_probabilities": [1.0],
        }
        games = [
            {"game_id": 0, "neural_side": 1, "positions": [dict(base, value=1.0)]},
            {"game_id": 1, "neural_side": -1, "positions": [dict(base, value=-1.0)]},
        ]
        candidates, observations = collect_candidates(games)
        self.assertEqual(observations, 2)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["value"], -1.0)
        self.assertEqual(candidates[0]["observations"], 2)

    def test_selection_balances_phases_before_overflow(self):
        records = []
        candidate_id = 0
        for stage in ("early", "middle", "late"):
            for margin in (300, 200, 100):
                records.append({
                    "stage": stage,
                    "value": -1.0,
                    "candidate_id": candidate_id,
                    "confidence": {"deep_score_margin": margin},
                })
                candidate_id += 1
        selected = select_phase_balanced(records, 6)
        self.assertEqual(len(selected), 6)
        for stage in ("early", "middle", "late"):
            self.assertEqual(sum(record["stage"] == stage for record in selected), 2)


if __name__ == "__main__":
    unittest.main()
