import unittest

import numpy as np

from santorini.SantoriniInference import predict_batch_deduplicated


class RecordingNetwork:
    def __init__(self):
        self.batch_sizes = []

    def predict_batch(self, boards):
        self.batch_sizes.append(len(boards))
        values = np.asarray([float(np.asarray(board).sum()) for board in boards])
        policies = np.stack([np.asarray([value, value + 1]) for value in values])
        return policies, values


class TestSantoriniInference(unittest.TestCase):
    def test_exact_boards_are_deduplicated_and_restored_in_input_order(self):
        nnet = RecordingNetwork()
        first = np.zeros((2, 5, 5), dtype=np.int8)
        second = first.copy()
        second[1, 2, 2] = 3

        policies, values, stats = predict_batch_deduplicated(
            nnet,
            [first, second, first, second, first],
        )

        self.assertEqual(nnet.batch_sizes, [2])
        self.assertEqual(stats, {'requested': 5, 'executed': 2, 'reused': 3})
        np.testing.assert_array_equal(values, [0, 3, 0, 3, 0])
        np.testing.assert_array_equal(policies[:, 0], values)

    def test_bounded_cache_reuses_predictions_across_calls(self):
        nnet = RecordingNetwork()
        board = np.ones((2, 5, 5), dtype=np.int8)
        cache = {}

        predict_batch_deduplicated(nnet, [board], cache=cache, max_cache_entries=1)
        _, _, stats = predict_batch_deduplicated(
            nnet,
            [board, board],
            cache=cache,
            max_cache_entries=1,
        )

        self.assertEqual(nnet.batch_sizes, [1])
        self.assertEqual(stats, {'requested': 2, 'executed': 0, 'reused': 2})
        self.assertEqual(len(cache), 1)


if __name__ == '__main__':
    unittest.main()
