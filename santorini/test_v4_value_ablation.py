import unittest

from summarize_santorini_v4_value_ablation import summarize


class V4ValueAblationDecisionTests(unittest.TestCase):
    @staticmethod
    def comparison(noninferior):
        return {
            "supervised_noninferior": noninferior,
            "decision_contract": {"noninferiority_margin": 0.01},
        }

    @staticmethod
    def arena(gate, scores):
        return {
            "gate": gate,
            "selection_seed": 20260814,
            "games": 40,
            "simulations": 96,
            "fp16": False,
            "final_test_touched": False,
            "final_arena_seeds_touched": False,
            "search_mode": "gumbel",
            "gumbel_scale": 0.0,
            "player1": {
                "name": "global_blend", "canonical_d4": True,
                "root_symmetries": 1,
            },
            "player2": {
                "name": "winner_only", "canonical_d4": True,
                "root_symmetries": 1,
            },
            "paired_statistics": {
                "records": [
                    {"game_seed": 20260814 + index, "contestant1_score": score}
                    for index, score in enumerate(scores)
                ]
            },
        }

    def test_prefers_winner_when_noninferior_and_arena_does_not_veto(self):
        tied = [1.0] * 20
        result = summarize(
            self.comparison(True),
            self.arena("standard", tied),
            self.arena("full", tied),
            bootstrap_samples=100,
            seed=3,
        )
        self.assertEqual(result["selected_target"], "winner")
        self.assertFalse(result["combined_arena"]["global_blend_clear_win"])

    def test_clear_global_arena_win_vetoes_winner(self):
        global_sweep = [2.0] * 20
        result = summarize(
            self.comparison(True),
            self.arena("standard", global_sweep),
            self.arena("full", global_sweep),
            bootstrap_samples=100,
            seed=3,
        )
        self.assertEqual(result["selected_target"], "global_blend")
        self.assertTrue(result["combined_arena"]["global_blend_clear_win"])

    def test_supervised_failure_selects_global_even_if_arena_is_tied(self):
        tied = [1.0] * 20
        result = summarize(
            self.comparison(False),
            self.arena("standard", tied),
            self.arena("full", tied),
            bootstrap_samples=100,
            seed=3,
        )
        self.assertEqual(result["selected_target"], "global_blend")


if __name__ == "__main__":
    unittest.main()
