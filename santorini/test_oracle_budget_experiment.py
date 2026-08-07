import os
import tempfile
import unittest

from benchmark_santorini_oracle_budgets import (
    _stage_quotas,
    load_or_initialize_records,
    score_sign,
    select_stratified_positions,
    stage_for_builds,
    summarize_records,
)


class TestOracleBudgetExperiment(unittest.TestCase):
    def test_stage_boundaries_and_quotas(self):
        self.assertEqual(stage_for_builds(0), "early")
        self.assertEqual(stage_for_builds(5), "early")
        self.assertEqual(stage_for_builds(6), "middle")
        self.assertEqual(stage_for_builds(15), "middle")
        self.assertEqual(stage_for_builds(16), "late")
        self.assertEqual(_stage_quotas(8), {"early": 3, "middle": 3, "late": 2})

    def test_stratified_selection_is_deterministic_and_redistributes_shortfall(self):
        pools = {
            "early": [{"fen": "e{}".format(i)} for i in range(1)],
            "middle": [{"fen": "m{}".format(i)} for i in range(6)],
            "late": [{"fen": "l{}".format(i)} for i in range(6)],
        }
        first = select_stratified_positions(pools, 9, seed=7)
        second = select_stratified_positions(pools, 9, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 9)
        self.assertEqual(sum(item["fen"].startswith("e") for item in first), 1)
        self.assertEqual([item["position_id"] for item in first], list(range(9)))

    def test_resume_metadata_rejects_mixed_experiments(self):
        metadata = {
            "schema_version": 1,
            "replay_path": "/tmp/replay",
            "replay_sha256": "abc",
            "budgets": [10, 20],
            "positions": 1,
            "seed": 4,
            "selection": [{"position_id": 0}],
            "type": "metadata",
        }
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "records.jsonl")
            self.assertEqual(load_or_initialize_records(path, metadata), [])
            self.assertEqual(load_or_initialize_records(path, metadata), [])
            changed = dict(metadata)
            changed["budgets"] = [10, 30]
            with self.assertRaises(ValueError):
                load_or_initialize_records(path, changed)

    def test_summary_reports_move_and_sign_stability(self):
        def analysis(next_fen, score, depth=4):
            return {
                "next_fen": next_fen,
                "score": score,
                "completed_depth": depth,
                "nodes_visited": 100,
                "elapsed_seconds": 0.1,
            }

        records = [
            {
                "stage": "early",
                "analyses": {
                    "10": analysis("a", 1),
                    "20": analysis("a", 2),
                    "30": analysis("a", 3),
                },
            },
            {
                "stage": "late",
                "analyses": {
                    "10": analysis("b", -1),
                    "20": analysis("c", 1),
                    "30": analysis("c", 2),
                },
            },
        ]
        summary = summarize_records(records, [10, 20, 30])["all"]
        self.assertEqual(summary["all_budget_move_stability_rate"], 0.5)
        self.assertEqual(summary["agreement_with_deepest"]["10"], 0.5)
        self.assertEqual(summary["agreement_with_deepest"]["20"], 1.0)
        self.assertEqual(summary["score_sign_agreement_with_deepest"]["10"], 0.5)
        self.assertEqual(summary["consecutive_move_agreement"]["10->20"], 0.5)
        self.assertEqual(score_sign(-3), -1)
        self.assertEqual(score_sign(0), 0)
        self.assertEqual(score_sign(2), 1)


if __name__ == "__main__":
    unittest.main()
