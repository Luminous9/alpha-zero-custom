import os
import pickle
import tempfile
import unittest

import numpy as np

from bootstrap_santorini_v3 import (
    flatten_examples_history,
    load_examples,
    split_indices,
)


class TestBootstrapSantoriniV3(unittest.TestCase):
    def test_flatten_examples_history_preserves_window_order(self):
        history = [
            [("board-a", "policy-a", 1)],
            [("board-b", "policy-b", -1), ("board-c", "policy-c", 1)],
        ]

        self.assertEqual(
            flatten_examples_history(history),
            [
                ("board-a", "policy-a", 1),
                ("board-b", "policy-b", -1),
                ("board-c", "policy-c", 1),
            ],
        )

    def test_load_examples_can_cap_for_smoke_runs(self):
        history = [
            [(np.zeros((2, 5, 5)), np.zeros(1600), 1) for _ in range(3)],
            [(np.zeros((2, 5, 5)), np.zeros(1600), -1) for _ in range(2)],
        ]

        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "latest.examples")
            with open(path, "wb") as examples_file:
                pickle.dump(history, examples_file)

            examples, history_lengths = load_examples(path, max_examples=4)

        self.assertEqual(len(examples), 4)
        self.assertEqual(history_lengths, [3, 2])

    def test_split_indices_is_deterministic_and_non_empty(self):
        train_a, validation_a = split_indices(10, 0.2, seed=7)
        train_b, validation_b = split_indices(10, 0.2, seed=7)

        np.testing.assert_array_equal(train_a, train_b)
        np.testing.assert_array_equal(validation_a, validation_b)
        self.assertEqual(len(train_a), 8)
        self.assertEqual(len(validation_a), 2)
        self.assertEqual(set(train_a).intersection(set(validation_a)), set())


if __name__ == "__main__":
    unittest.main()
