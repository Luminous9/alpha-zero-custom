import unittest

import numpy as np

from benchmark_santorini_oracle_root_moves import (
    confidence_metrics,
    normalized_entropy,
    score_softmax,
    top_overlap,
)


def move(name, score):
    return {"next_fen": name, "score": score}


class TestOracleRootMoves(unittest.TestCase):
    def test_score_softmax_preserves_ties_and_prefers_better_scores(self):
        probabilities = score_softmax(
            [move("a", 100), move("b", 100), move("c", 0)], temperature=100
        )
        self.assertAlmostEqual(float(probabilities.sum()), 1.0)
        self.assertAlmostEqual(probabilities[0], probabilities[1])
        self.assertGreater(probabilities[1], probabilities[2])

    def test_confidence_requires_top_move_and_top_three_stability(self):
        shallow = {"moves": [move("a", 3), move("b", 2), move("c", 1)]}
        deep = {"moves": [move("a", 4), move("c", 2), move("d", 0)]}
        metrics = confidence_metrics(shallow, deep, score_temperature=10)
        self.assertTrue(metrics["top1_agreement"])
        self.assertEqual(metrics["top3_jaccard"], 0.5)
        self.assertTrue(metrics["confident"])

        deep["moves"][0] = move("z", 4)
        self.assertFalse(confidence_metrics(shallow, deep, 10)["confident"])

    def test_overlap_and_entropy_boundaries(self):
        self.assertEqual(top_overlap([move("a", 1)], [move("a", 2)]), 1.0)
        self.assertAlmostEqual(normalized_entropy([0.5, 0.5]), 1.0)
        self.assertEqual(normalized_entropy([1.0]), 0.0)


if __name__ == "__main__":
    unittest.main()
