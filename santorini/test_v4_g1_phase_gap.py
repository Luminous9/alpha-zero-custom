import unittest

import numpy as np

from diagnose_santorini_v4_g1_phase_gap import (
    _assert_same_placements,
    _balanced_replay_openings,
)


def record(pair_index, seat_order, value):
    pieces = np.zeros((5, 5), dtype=np.int8)
    pieces.flat[value] = 1
    return {
        "pair_index": pair_index,
        "seat_order": seat_order,
        "game_seed": 10 + pair_index,
        "opening": str(value),
        "labeled_opening_key": pieces.tobytes().hex(),
        "symmetry_opening": str(value),
    }


class V4G1PhaseGapTests(unittest.TestCase):
    def test_balanced_replay_alternates_source_seat(self):
        records = [
            record(pair, seat, 2 * pair + offset)
            for pair in range(2)
            for offset, seat in enumerate(
                ("contestant1_first", "contestant2_first")
            )
        ]
        boards, metadata = _balanced_replay_openings(records, 2)
        self.assertEqual(metadata[0]["source_seat_order"], "contestant1_first")
        self.assertEqual(metadata[1]["source_seat_order"], "contestant2_first")
        self.assertEqual(int(np.argmax(boards[0][0])), 0)
        self.assertEqual(int(np.argmax(boards[1][0])), 3)

    def test_controller_change_must_not_change_placements(self):
        records = [record(0, "contestant1_first", 1)]
        _assert_same_placements(
            {"placement_records": records},
            {"placement_records": list(records)},
        )
        changed = [record(0, "contestant1_first", 2)]
        with self.assertRaises(ValueError):
            _assert_same_placements(
                {"placement_records": records},
                {"placement_records": changed},
            )


if __name__ == "__main__":
    unittest.main()
