import unittest

from generate_santorini_opening_book import (
    board_record,
    build_minimax_book_records,
    canonical_key,
    checkpoint_output_dir,
    evaluate_openings,
    format_json,
    iter_player1_choices,
    iter_player2_responses,
    iter_opening_positions,
    make_board,
    transform_location,
)


class TestSantoriniOpeningBook(unittest.TestCase):
    def test_canonical_key_collapses_rotations_and_reflections(self):
        p1 = ((1, 1), (2, 2))
        p2 = ((0, 4), (4, 0))
        rotated_p1 = tuple(transform_location(location, 5, 1, False) for location in p1)
        rotated_p2 = tuple(transform_location(location, 5, 1, False) for location in p2)
        flipped_p1 = tuple(transform_location(location, 5, 0, True) for location in p1)
        flipped_p2 = tuple(transform_location(location, 5, 0, True) for location in p2)

        self.assertEqual(canonical_key(p1, p2, 5), canonical_key(rotated_p1, rotated_p2, 5))
        self.assertEqual(canonical_key(p1, p2, 5), canonical_key(flipped_p1, flipped_p2, 5))

    def test_iter_opening_positions_allows_both_players_on_outer_edge(self):
        positions = list(iter_opening_positions(3))

        self.assertIn((((0, 0), (0, 1)), ((0, 2), (1, 0))), positions)

    def test_player1_choices_are_symmetry_reduced(self):
        self.assertLess(len(list(iter_player1_choices(5))), 300)

    def test_player2_responses_are_reduced_by_player1_stabilizer(self):
        responses = list(iter_player2_responses(((0, 0), (0, 1)), 5))

        self.assertIn(((0, 2), (1, 0)), responses)
        self.assertLess(len(responses), 276)

    def test_make_board_assigns_worker_labels(self):
        board = make_board(((0, 0), (1, 1)), ((2, 2), (3, 3)), board_size=5)

        self.assertEqual(board[0, 0, 0], 1)
        self.assertEqual(board[0, 1, 1], 2)
        self.assertEqual(board[0, 2, 2], -1)
        self.assertEqual(board[0, 3, 3], -2)

    def test_make_board_can_swap_same_color_worker_labels(self):
        board = make_board(((0, 0), (1, 1)), ((2, 2), (3, 3)), p1_swap=True, p2_swap=True, board_size=5)

        self.assertEqual(board[0, 0, 0], 2)
        self.assertEqual(board[0, 1, 1], 1)
        self.assertEqual(board[0, 2, 2], -2)
        self.assertEqual(board[0, 3, 3], -1)

    def test_board_record_scores_mean_abs_plus_std(self):
        record = board_record(
            1,
            ((0, 0), (1, 1)),
            ((2, 2), (3, 3)),
            [-0.2, -0.1, 0.1, 0.2],
            5,
        )

        self.assertAlmostEqual(record["value_mean"], 0.0)
        self.assertAlmostEqual(record["score"], record["value_std"])

    def test_evaluate_openings_returns_minimax_tree_and_flat_positions(self):
        class FakeNNet:
            def __init__(self):
                self.batch_sizes = []

            def predict_batch(self, boards):
                self.batch_sizes.append(len(boards))
                return None, [0.0 for _ in boards]

        fake_nnet = FakeNNet()
        choices = evaluate_openings(fake_nnet, board_size=3, batch_size=8, max_positions=2)

        self.assertGreaterEqual(len(choices), 1)
        self.assertIn("responses", choices[0])
        self.assertEqual(sum(fake_nnet.batch_sizes), 2)
        self.assertFalse(choices[0]["responses"][0]["label_invariance_checked"])

    def test_evaluate_openings_can_check_label_invariance(self):
        class FakeNNet:
            def __init__(self):
                self.batch_sizes = []

            def predict_batch(self, boards):
                self.batch_sizes.append(len(boards))
                return None, [0.0 for _ in boards]

        fake_nnet = FakeNNet()
        choices = evaluate_openings(
            fake_nnet,
            board_size=3,
            batch_size=8,
            max_positions=2,
            check_label_invariance=True,
        )

        self.assertEqual(sum(fake_nnet.batch_sizes), 8)
        self.assertTrue(choices[0]["responses"][0]["label_invariance_checked"])
        self.assertEqual(len(choices[0]["responses"][0]["label_values"]), 4)

    def test_minimax_book_sorts_p2_low_and_p1_high(self):
        p1_a = ((0, 0), (0, 1))
        p1_b = ((1, 0), (1, 1))
        records_by_player1 = {
            p1_a: [
                {"id": 1, "player1": ["A1", "B1"], "player2": ["C1", "D1"], "value_mean": 0.3, "value_std": 0.0},
                {"id": 2, "player1": ["A1", "B1"], "player2": ["C2", "D2"], "value_mean": -0.2, "value_std": 0.0},
            ],
            p1_b: [
                {"id": 3, "player1": ["A2", "B2"], "player2": ["C1", "D1"], "value_mean": 0.1, "value_std": 0.0},
                {"id": 4, "player1": ["A2", "B2"], "player2": ["C2", "D2"], "value_mean": 0.2, "value_std": 0.0},
            ],
        }

        choices = build_minimax_book_records(records_by_player1)

        self.assertEqual(choices[0]["player1"], ["A2", "B2"])
        self.assertEqual(choices[0]["minimax_value"], 0.1)
        self.assertEqual(choices[0]["responses"][0]["id"], 3)

    def test_format_json_keeps_piece_rows_compact(self):
        text = format_json({
            "positions": [
                {
                    "pieces": [
                        [1, 2, 0, 0, 0],
                        [0, 0, 0, 0, -1],
                        [0, -2, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                    ],
                }
            ],
        })

        self.assertIn("[1, 2, 0, 0, 0]", text)
        self.assertNotIn("        1,\n        2,", text)

    def test_checkpoint_output_dir_uses_checkpoint_folder_name(self):
        self.assertEqual(
            checkpoint_output_dir("temp/santorini_opening_books", "./temp/santorini_kaggle_training5"),
            "temp/santorini_opening_books/santorini_kaggle_training5",
        )
        self.assertEqual(
            checkpoint_output_dir("temp/santorini_opening_books", "/kaggle/input/run/checkpoints/"),
            "temp/santorini_opening_books/checkpoints",
        )


if __name__ == "__main__":
    unittest.main()
