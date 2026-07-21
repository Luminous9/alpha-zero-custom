import unittest

import numpy as np

from audit_santorini_placements import (
    distinct_transformations,
    effective_action_count,
    parse_opening,
    restore_dense_vector,
    state_stabilizer_action_classes,
    transform_dense_vector,
)
from santorini.SantoriniGame import SantoriniGame


class TestPlacementAudit(unittest.TestCase):
    def setUp(self):
        self.game = SantoriniGame(5, sequential_placement=True)

    def test_dense_policy_transform_round_trip(self):
        policy = np.arange(self.game.getActionSize(), dtype=np.float64)
        for rotations in range(4):
            for flip in (False, True):
                transformed = transform_dense_vector(
                    self.game, policy, rotations, flip
                )
                restored = restore_dense_vector(
                    self.game, transformed, rotations, flip
                )
                np.testing.assert_array_equal(restored, policy)

    def test_distinct_orbit_respects_position_stabilizers(self):
        empty = self.game.getInitBoard()
        self.assertEqual(len(distinct_transformations(empty)), 1)

        board, _ = self.game.getNextState(
            empty, 1, self.game.getPlacementAction((1, 2))
        )
        self.assertEqual(len(distinct_transformations(board)), 4)

    def test_effective_action_count(self):
        self.assertAlmostEqual(effective_action_count([1.0, 0.0]), 1.0)
        self.assertAlmostEqual(effective_action_count([0.25] * 4), 4.0)

    def test_empty_board_groups_symmetric_placement_actions(self):
        classes = state_stabilizer_action_classes(
            self.game, self.game.getInitBoard()
        )
        center = self.game.getPlacementAction((2, 2))
        corners = [self.game.getPlacementAction(location) for location in (
            (0, 0), (0, 4), (4, 0), (4, 4)
        )]
        self.assertEqual(len({classes[action] for action in corners}), 1)
        self.assertNotEqual(classes[corners[0]], classes[center])

    def test_opening_parser(self):
        self.assertEqual(
            parse_opening('p1=1:2,2:2|p2=2:1,2:3'),
            (((1, 2), (2, 2)), ((2, 1), (2, 3))),
        )


if __name__ == '__main__':
    unittest.main()
