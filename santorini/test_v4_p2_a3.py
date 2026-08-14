import unittest

from summarize_santorini_v4_p2_a3 import paired_delta


class TestV4P2A3(unittest.TestCase):
    def test_paired_delta_uses_candidate_minus_reference(self):
        candidate = {
            "pair_records": [
                {"opening_sha256": "a", "v4_pair_score": 2.0},
                {"opening_sha256": "b", "v4_pair_score": 1.0},
            ]
        }
        reference = {
            "pair_records": [
                {"opening_sha256": "a", "v4_pair_score": 1.0},
                {"opening_sha256": "b", "v4_pair_score": 0.0},
            ]
        }
        result = paired_delta(candidate, reference, samples=100, seed=3)
        self.assertEqual(result["score_delta"], 0.5)
        self.assertEqual(result["improved_pairs"], 2)
        self.assertEqual(result["unchanged_pairs"], 0)
        self.assertEqual(result["worsened_pairs"], 0)

    def test_paired_delta_rejects_different_openings(self):
        candidate = {"pair_records": [{"opening_sha256": "a", "v4_pair_score": 1.0}]}
        reference = {"pair_records": [{"opening_sha256": "b", "v4_pair_score": 1.0}]}
        with self.assertRaises(ValueError):
            paired_delta(candidate, reference, samples=10, seed=3)


if __name__ == "__main__":
    unittest.main()
