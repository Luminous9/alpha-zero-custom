import unittest

import numpy as np

from santorini.V4BootstrapCorpus import (
    build_sampling_plan,
    joint_marginal_quotas,
    largest_remainder_quotas,
    validate_no_cross_corpus_leakage,
)


class V4BootstrapCorpusTests(unittest.TestCase):
    def payloads(self):
        # Two positions per stage and every engine source in train split.
        engine_stages = np.repeat(np.arange(3, dtype=np.int8), 2)
        engine = {
            "stage_ids": engine_stages,
            "split_ids": np.zeros(6, dtype=np.int8),
            "source_counts": np.asarray([
                [1, 1], [2, 3], [1, 1], [2, 3], [1, 1], [2, 3]
            ], dtype=np.int32),
            "position_hashes": np.asarray(["e{}".format(i) for i in range(6)]),
        }
        run13 = {
            "stage_ids": np.arange(3, dtype=np.int8),
            "split_ids": np.zeros(3, dtype=np.int8),
            "position_hashes": np.asarray(["r0", "r1", "r2"]),
        }
        return engine, run13

    def test_largest_remainder_is_exact(self):
        quotas = largest_remainder_quotas(11, [0.2, 0.35, 0.45])
        self.assertEqual(quotas.sum(), 11)
        self.assertEqual(quotas.tolist(), [2, 4, 5])

    def test_joint_quotas_preserve_explicit_source_and_stage_marginals(self):
        quotas = joint_marginal_quotas(
            300_000,
            (0.20, 0.35, 0.45),
            (225_556 / 300_000, 64_444 / 300_000, 10_000 / 300_000),
            source_counts=(225_556, 64_444, 10_000),
        )
        self.assertEqual(quotas.sum(axis=1).tolist(), [225_556, 64_444, 10_000])
        self.assertEqual(quotas.sum(axis=0).tolist(), [60_000, 105_000, 135_000])

    def test_sampling_plan_hits_declared_marginals_exactly(self):
        engine, run13 = self.payloads()
        plan = build_sampling_plan(
            engine,
            run13,
            draws=10_000,
            split_id=0,
            stage_fractions=(0.20, 0.35, 0.45),
            source_fractions=(0.70, 0.20, 0.10),
            seed=9,
        )
        self.assertEqual(len(plan["position_indices"]), 10_000)
        self.assertEqual(np.bincount(plan["stage_ids"], minlength=3).tolist(), [2000, 3500, 4500])
        self.assertEqual(np.bincount(plan["source_ids"], minlength=3).tolist(), [7000, 2000, 1000])
        self.assertTrue(np.all(plan["corpus_ids"][plan["source_ids"] == 2] == 1))

    def test_unique_sampling_has_no_repeated_corpus_position(self):
        engine, run13 = self.payloads()
        plan = build_sampling_plan(
            engine,
            run13,
            draws=9,
            split_id=0,
            stage_fractions=(1 / 3, 1 / 3, 1 / 3),
            source_fractions=(1 / 3, 1 / 3, 1 / 3),
            seed=9,
            replace=False,
        )
        pairs = np.stack((plan["corpus_ids"], plan["position_indices"]), axis=1)
        self.assertEqual(len(np.unique(pairs, axis=0)), len(pairs))
        self.assertFalse(plan["sampling_with_replacement"])

    def test_unique_sampling_rejects_insufficient_stratum_supply(self):
        engine, run13 = self.payloads()
        with self.assertRaisesRegex(ValueError, "needs .* only"):
            build_sampling_plan(
                engine,
                run13,
                draws=30,
                split_id=0,
                stage_fractions=(1 / 3, 1 / 3, 1 / 3),
                source_fractions=(1 / 3, 1 / 3, 1 / 3),
                seed=9,
                replace=False,
            )

    def test_explicit_joint_counts_preserve_non_independent_mix(self):
        engine, run13 = self.payloads()
        joint = np.ones((3, 3), dtype=np.int64)
        plan = build_sampling_plan(
            engine,
            run13,
            draws=9,
            split_id=0,
            stage_fractions=(1 / 3, 1 / 3, 1 / 3),
            source_fractions=(1 / 3, 1 / 3, 1 / 3),
            seed=11,
            replace=False,
            joint_counts=joint,
        )
        self.assertTrue(np.array_equal(plan["joint_quotas"], joint))

    def test_cross_corpus_split_leakage_is_rejected(self):
        engine, run13 = self.payloads()
        run13["position_hashes"][0] = engine["position_hashes"][0]
        run13["split_ids"][0] = 2
        with self.assertRaisesRegex(ValueError, "crosses"):
            validate_no_cross_corpus_leakage(engine, run13)


if __name__ == "__main__":
    unittest.main()
