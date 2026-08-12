import json
import os
import tempfile
import unittest

from run_santorini_v4_oracle_sweep import (
    _contract_fingerprint,
    _load_existing_budget,
    select_sparring_rung,
)


class TestV4OracleSweep(unittest.TestCase):
    def test_selects_score_closest_to_midpoint_and_breaks_tie_upward(self):
        selected = select_sparring_rung([
            {"oracle_nodes": 5_000, "v4_score": 0.50},
            {"oracle_nodes": 10_000, "v4_score": 0.45},
            {"oracle_nodes": 20_000, "v4_score": 0.40},
        ])
        self.assertEqual(selected["decision"], "selected")
        self.assertEqual(selected["selected_oracle_nodes"], 20_000)
        self.assertEqual(selected["selected_v4_score"], 0.40)

    def test_extends_up_or_down_when_all_scores_miss_same_side(self):
        upward = select_sparring_rung([
            {"oracle_nodes": 10_000, "v4_score": 0.70},
            {"oracle_nodes": 20_000, "v4_score": 0.55},
        ])
        downward = select_sparring_rung([
            {"oracle_nodes": 5_000, "v4_score": 0.25},
            {"oracle_nodes": 10_000, "v4_score": 0.20},
        ])
        self.assertEqual(upward["recommended_next_budget"], 40_000)
        self.assertEqual(downward["recommended_next_budget"], 2_500)

    def test_recommends_log_midpoint_for_a_straddled_band(self):
        result = select_sparring_rung([
            {"oracle_nodes": 10_000, "v4_score": 0.55},
            {"oracle_nodes": 40_000, "v4_score": 0.30},
        ])
        self.assertEqual(result["decision"], "extend_sweep")
        self.assertEqual(result["recommended_next_budget"], 20_000)

    def test_nonmonotonic_miss_is_inconclusive(self):
        result = select_sparring_rung([
            {"oracle_nodes": 5_000, "v4_score": 0.30},
            {"oracle_nodes": 10_000, "v4_score": 0.60},
        ])
        self.assertEqual(result["decision"], "inconclusive")
        self.assertIsNone(result["recommended_next_budget"])

    def test_resume_requires_exact_contract_and_complete_result(self):
        contract = {"games_per_budget": 4}
        fingerprint = _contract_fingerprint(contract)
        payload = {
            "contract_fingerprint": fingerprint,
            "contract": contract,
            "result": {"oracle_nodes": 10_000, "games": 4},
            "final_test_touched": False,
            "final_arena_seeds_touched": False,
        }
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "budget.json")
            with open(path, "w") as output:
                json.dump(payload, output)
            self.assertEqual(
                _load_existing_budget(path, fingerprint, 10_000), payload
            )
            payload["contract_fingerprint"] = "different"
            with open(path, "w") as output:
                json.dump(payload, output)
            with self.assertRaisesRegex(ValueError, "different contract"):
                _load_existing_budget(path, fingerprint, 10_000)


if __name__ == "__main__":
    unittest.main()
