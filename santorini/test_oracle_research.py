import os
import tempfile
import unittest

from santorini.OracleResearch import OracleLabelCache, ParallelOraclePool
from santorini.SantoriniOracle import DEFAULT_ORACLE_BINARY


class FakeOracle:
    def __init__(self):
        self.events = []
        self.closed = False

    def reset(self):
        self.events.append("reset")
        return {"command": "reset"}

    def analyze_fen(self, fen, nodes):
        self.events.append(("analyze", fen, nodes))
        return {
            "command": "analyze",
            "fen": fen,
            "requested_nodes": nodes,
            "nodes_visited": nodes + 7,
            "completed_depth": 6,
            "best_move": {
                "actions": [],
                "next_fen": "successor",
                "score": 250,
            },
        }

    def close(self):
        self.closed = True


class TestParallelOraclePool(unittest.TestCase):
    def test_analyze_resets_before_invoking_analyzer(self):
        fake = FakeOracle()
        pool = ParallelOraclePool("unused", oracle_factory=lambda _binary: fake)
        try:
            result = pool.analyze(lambda oracle: oracle.events.append("analyze") or 17)
        finally:
            pool.close()

        self.assertEqual(result, 17)
        self.assertEqual(fake.events, ["reset", "analyze"])
        self.assertTrue(fake.closed)

    def test_label_cache_keys_every_contract_dimension_and_survives_reopen(self):
        with tempfile.TemporaryDirectory() as folder:
            binary = os.path.join(folder, "oracle")
            cache_path = os.path.join(folder, "labels.sqlite3")
            with open(binary, "wb") as output:
                output.write(b"oracle build one")

            first_oracle = FakeOracle()
            first_pool = ParallelOraclePool(
                binary,
                cache_path=cache_path,
                oracle_factory=lambda _binary: first_oracle,
            )
            try:
                first = first_pool.label_fen(
                    "d4-fen", 2_000, "temperature-v1", lambda score: score / 1_000
                )
                second = first_pool.label_fen(
                    "d4-fen", 2_000, "temperature-v1", lambda _score: -1.0
                )
                changed_calibration = first_pool.label_fen(
                    "d4-fen", 2_000, "temperature-v2", lambda score: score / 500
                )
            finally:
                first_pool.close()

            self.assertFalse(first["cache_hit"])
            self.assertTrue(second["cache_hit"])
            self.assertFalse(changed_calibration["cache_hit"])
            self.assertEqual(first["mapped_value"], 0.25)
            self.assertEqual(second["mapped_value"], 0.25)
            self.assertEqual(changed_calibration["mapped_value"], 0.5)
            self.assertEqual(first["actual_nodes"], 2_007)
            self.assertEqual(first["completed_depth"], 6)
            self.assertFalse(first["mate_band"])
            self.assertEqual(
                first_oracle.events,
                [
                    "reset",
                    ("analyze", "d4-fen", 2_000),
                    "reset",
                    ("analyze", "d4-fen", 2_000),
                ],
            )

            reopened_oracle = FakeOracle()
            reopened_pool = ParallelOraclePool(
                binary,
                cache_path=cache_path,
                oracle_factory=lambda _binary: reopened_oracle,
            )
            try:
                reopened = reopened_pool.label_fen(
                    "d4-fen", 2_000, "temperature-v1", lambda _score: -1.0
                )
            finally:
                reopened_pool.close()

            self.assertTrue(reopened["cache_hit"])
            self.assertEqual(reopened["mapped_value"], 0.25)
            self.assertEqual(reopened_oracle.events, [])


@unittest.skipUnless(os.path.isfile(DEFAULT_ORACLE_BINARY), "native oracle is not built")
class TestParallelOraclePoolIntegration(unittest.TestCase):
    def test_repeated_queries_are_identical_after_automatic_resets(self):
        fen = "0002000000000000000001000/1/mortal:A1,B2/mortal:D4,E5"
        pool = ParallelOraclePool(DEFAULT_ORACLE_BINARY)
        try:
            first = pool.analyze(lambda oracle: oracle.analyze_fen(fen, nodes=2_000))
            second = pool.analyze(lambda oracle: oracle.analyze_fen(fen, nodes=2_000))
        finally:
            pool.close()

        for response in (first, second):
            response.pop("id", None)
        self.assertEqual(first, second)


class TestOracleLabelCache(unittest.TestCase):
    def test_rejects_empty_version_dimensions(self):
        with tempfile.TemporaryDirectory() as folder:
            with OracleLabelCache(os.path.join(folder, "labels.sqlite3")) as cache:
                with self.assertRaises(ValueError):
                    cache.get("fen", 1, "", "calibration")
                with self.assertRaises(ValueError):
                    cache.get("fen", 1, "engine", "")
                with self.assertRaises(ValueError):
                    cache.get("fen", 0, "engine", "calibration")
                with self.assertRaises(ValueError):
                    cache.put(
                        {
                            "d4_fen": "fen",
                            "requested_nodes": 1,
                            "engine_digest": "engine",
                            "calibration_version": "calibration",
                            "response": {},
                            "score": 0,
                            "mate_band": False,
                            "completed_depth": 0,
                            "actual_nodes": 1,
                            "mapped_value": float("nan"),
                        }
                    )


if __name__ == "__main__":
    unittest.main()
