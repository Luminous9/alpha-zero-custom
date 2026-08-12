import unittest

from summarize_santorini_v4_g1 import build_summary, _standard_classification
from run_santorini_v4_g1 import choose_equal_cost_simulations


class V4G1Tests(unittest.TestCase):
    @staticmethod
    def _arena(gate, p1_sims, p2_sims, score=0.40):
        pair_wins = int(round(score * 20))
        records = [
            {
                "pair_index": index,
                "contestant1_score": 2.0 if index < pair_wins else 0.0,
            }
            for index in range(20)
        ]
        return {
            "gate": gate,
            "final_test_touched": False,
            "final_arena_seeds_touched": False,
            "games": 40,
            "player1_score": score,
            "player1_simulations": p1_sims,
            "player2_simulations": p2_sims,
            "placement_temperature": 1.0 if gate == "full" else 0.0,
            "elapsed_seconds": 1.0,
            "draws": 0,
            "inference": {},
            "player1": {
                "name": "p1c",
                "kind": "v4",
                "canonical_d4": True,
                "root_symmetries": 1,
                "placement_root_symmetries": 1,
                "wins": int(round(score * 40)),
            },
            "player2": {
                "name": "run13",
                "kind": "v3",
                "canonical_d4": False,
                "root_symmetries": 8,
                "placement_root_symmetries": 8,
                "wins": 40 - int(round(score * 40)),
            },
            "paired_statistics": {
                "cluster_bootstrap_95_low": 0.25,
                "cluster_bootstrap_95_high": 0.55,
                "pairs": 20,
                "pair_wins": pair_wins,
                "pair_losses": 20 - pair_wins,
                "pair_ties": 0,
                "records": records,
            },
        }

    def test_standard_gate_green_requires_score_and_interval(self):
        self.assertEqual(
            _standard_classification({
                "candidate_score": 0.40,
                "cluster_bootstrap_95_low": 0.225,
            }),
            "green",
        )
        self.assertEqual(
            _standard_classification({
                "candidate_score": 0.40,
                "cluster_bootstrap_95_low": 0.20,
            }),
            "inconclusive",
        )

    def test_standard_gate_stop_uses_point_score(self):
        self.assertEqual(
            _standard_classification({
                "candidate_score": 0.175,
                "cluster_bootstrap_95_low": 0.05,
            }),
            "stop",
        )

    def test_equal_cost_budget_anchors_faster_model(self):
        self.assertEqual(
            choose_equal_cost_simulations(2.0, 1.0, anchor=128),
            {"p1c": 64, "run13": 128},
        )
        self.assertEqual(
            choose_equal_cost_simulations(1.0, 1.0, anchor=128),
            {"p1c": 128, "run13": 120},
        )

    def test_summary_accepts_frozen_g1_contract(self):
        equal = {
            budget: {
                gate: self._arena(gate, budget, budget)
                for gate in ("standard", "full")
            }
            for budget in (96, 128)
        }
        equal_cost = {
            gate: self._arena(gate, 64, 128)
            for gate in ("standard", "full")
        }
        summary = build_summary(
            {
                "equal_cost_budget_rule": "test",
                "equal_cost_simulations": {"p1c": 64, "run13": 128},
            },
            equal,
            equal_cost,
            seed=20260902,
            bootstrap_samples=100,
        )
        self.assertEqual(summary["decision"], "green_light")
        self.assertFalse(
            summary["decision_reasons"]["material_standard_full_discrepancy"]
        )


if __name__ == "__main__":
    unittest.main()
