import os
import tempfile
import unittest

from calibrate_santorini_oracle_scores import (
    _log_loss,
    adjudicate_from_fen,
    assign_splits,
    fit_temperature,
    load_or_initialize_records,
    nominal_score_value,
    score_probability,
    summarize,
    terminal_winner,
)


class FakeContinuationOracle:
    def __init__(self):
        self.resets = 0
        self.queries = []

    def reset(self):
        self.resets += 1

    def analyze_fen(self, fen, nodes):
        self.queries.append((fen, nodes))
        return {
            "completed_depth": 5,
            "nodes_visited": nodes + 3,
            "best_move": {
                "next_fen": "0" * 25 + "/2/#mortal:A1,A2/mortal:E4,E5",
                "score": 9_999,
            },
        }


class TestOracleScoreCalibration(unittest.TestCase):
    def test_adjudication_resets_and_uses_a_fresh_first_move(self):
        oracle = FakeContinuationOracle()
        start = "0" * 25 + "/1/mortal:A1,A2/mortal:E4,E5"
        result = adjudicate_from_fen(oracle, start, nodes=5_000, max_plies=10)

        self.assertEqual(oracle.resets, 1)
        self.assertEqual(oracle.queries, [(start, 5_000)])
        self.assertEqual(result["winner"], 1)
        self.assertEqual(result["outcome"], 1)
        self.assertEqual(result["plies"], 1)
        self.assertEqual(terminal_winner(result["trajectory"][0]["next_fen"]), 1)

    def test_temperature_fit_is_positive_and_no_worse_than_nominal(self):
        scores = [-800, -400, -200, -100, 100, 200, 400, 800]
        outcomes = [-1, -1, 1, -1, 1, -1, 1, 1]
        fitted = fit_temperature(scores, outcomes)
        self.assertGreater(fitted, 0)
        self.assertLessEqual(
            _log_loss(scores, outcomes, fitted),
            _log_loss(scores, outcomes, 400.0) + 1e-9,
        )
        self.assertEqual(score_probability(10_000, fitted), 1.0)
        self.assertEqual(score_probability(-10_000, fitted), 0.0)
        self.assertAlmostEqual(nominal_score_value(0), 0.0)

    def test_stage_stratified_split_is_deterministic_and_preserves_test_rows(self):
        selection = [
            {"position_id": index, "stage": stage, "fen": "{}{}".format(stage, index)}
            for stage in ("early", "middle", "late")
            for index in range(4)
        ]
        first = assign_splits(selection, 0.75, 9)
        second = assign_splits(selection, 0.75, 9)
        self.assertEqual(first, second)
        for stage in ("early", "middle", "late"):
            stage_rows = [record for record in first if record["stage"] == stage]
            self.assertEqual(sum(record["split"] == "test" for record in stage_rows), 1)

    def test_summary_reports_held_out_stage_and_magnitude_metrics(self):
        def record(split, stage, score, outcome):
            return {
                "split": split,
                "stage": stage,
                "labels": {"100": {"score": score}},
                "adjudication": {"outcome": outcome},
            }

        records = [
            record("fit", "early", -300, -1),
            record("fit", "middle", 200, 1),
            record("fit", "late", 500, 1),
            record("test", "early", -50, -1),
            record("test", "middle", 250, 1),
            record("test", "late", 10_000, 1),
        ]
        result = summarize(records, [100])["budgets"]["100"]
        self.assertEqual(result["nominal_temperature"], 400.0)
        self.assertEqual(result["nominal_test"]["positions"], 3)
        self.assertEqual(result["test"]["positions"], 3)
        self.assertEqual(result["test_by_stage"]["late"]["positions"], 1)
        self.assertEqual(result["test_by_score_magnitude"]["mate"]["positions"], 1)

    def test_resume_metadata_rejects_changed_adjudicator(self):
        metadata = {
            "schema_version": 1,
            "replay_sha256": "abc",
            "positions": 1,
            "label_budgets": [10],
            "adjudicator_nodes": 100,
            "max_adjudication_plies": 20,
            "fit_fraction": 0.7,
            "seed": 1,
            "engine_digest": "engine",
            "selection": [{"position_id": 0}],
        }
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "records.jsonl")
            self.assertEqual(load_or_initialize_records(path, metadata), [])
            changed = dict(metadata, adjudicator_nodes=200)
            with self.assertRaises(ValueError):
                load_or_initialize_records(path, changed)


if __name__ == "__main__":
    unittest.main()
