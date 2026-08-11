import unittest

import numpy as np

from build_santorini_v4_mixed_placement import _sparse_policies, build_payload
from santorini.SantoriniGame import SantoriniGame
from santorini.V4BootstrapCorpus import decode_sparse_policy


class V4MixedPlacementTests(unittest.TestCase):
    def setUp(self):
        self.game = SantoriniGame(5, sequential_placement=True)
        self.board = self.game.getInitBoard().astype(np.int8)
        self.action_a = self.game.getPlacementAction((0, 0))
        self.action_b = self.game.getPlacementAction((1, 2))

    def component(self, action, winner, oracle):
        policy = np.zeros(self.game.getActionSize(), dtype=np.float32)
        policy[action] = 1.0
        payload = {
            "action_size": np.asarray([self.game.getActionSize()], dtype=np.int32),
            "boards": np.asarray([self.board]),
            "observation_counts": np.asarray([7], dtype=np.int32),
            "winner_means": np.asarray([winner], dtype=np.float32),
            "score_means": np.asarray([123.0], dtype=np.float32),
            "oracle_value_means": np.asarray([0.2], dtype=np.float32),
            "requested_nodes": np.asarray([50_000], dtype=np.int32),
            "worker_counts": np.asarray([0], dtype=np.int8),
            "position_hashes": np.asarray(["a" * 64]),
        }
        payload.update(_sparse_policies([policy]))
        if oracle:
            payload["has_completed_outcomes"] = np.asarray([False])
        else:
            payload["has_completed_outcomes"] = np.asarray([True])
        return payload

    def test_blend_uses_run13_outcome_and_declared_policy_mix(self):
        oracle = self.component(self.action_a, 0.0, True)
        run13 = self.component(self.action_b, -0.75, False)
        payload, diagnostics = build_payload(oracle, run13, "blend", 0.25)
        policy = decode_sparse_policy(payload, 0)
        self.assertAlmostEqual(float(policy[self.action_a]), 0.25)
        self.assertAlmostEqual(float(policy[self.action_b]), 0.75)
        self.assertAlmostEqual(float(payload["winner_means"][0]), -0.75)
        self.assertTrue(bool(payload["has_completed_outcomes"][0]))
        self.assertAlmostEqual(diagnostics["mean_oracle_run13_policy_tv"], 1.0)

    def test_teacher_arms_change_policy_only(self):
        oracle = self.component(self.action_a, 0.0, True)
        run13 = self.component(self.action_b, 0.5, False)
        oracle_payload, _ = build_payload(oracle, run13, "oracle", 0.5)
        run13_payload, _ = build_payload(oracle, run13, "run13", 0.5)
        self.assertEqual(
            int(np.argmax(decode_sparse_policy(oracle_payload, 0))), self.action_a
        )
        self.assertEqual(
            int(np.argmax(decode_sparse_policy(run13_payload, 0))), self.action_b
        )
        np.testing.assert_array_equal(
            oracle_payload["winner_means"], run13_payload["winner_means"]
        )
        np.testing.assert_array_equal(
            oracle_payload["observation_counts"], run13_payload["observation_counts"]
        )


if __name__ == "__main__":
    unittest.main()
