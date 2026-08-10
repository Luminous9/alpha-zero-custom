"""Auditable sampling and loading for the mixed V4 bootstrap corpus."""

from collections import Counter
from dataclasses import dataclass

import numpy as np


STAGE_NAMES = ("early", "middle", "late")
SOURCE_NAMES = ("engine_main_line", "engine_randomized_subgame", "run13_replay")


def decode_sparse_policy(payload, index):
    action_size = int(payload["action_size"][0])
    start = int(payload["policy_offsets"][index])
    end = int(payload["policy_offsets"][index + 1])
    policy = np.zeros(action_size, dtype=np.float32)
    policy[payload["policy_indices"][start:end].astype(np.int64)] = (
        payload["policy_values"][start:end]
    )
    return policy


def largest_remainder_quotas(total, probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if int(total) < 1 or np.any(probabilities < 0):
        raise ValueError("Sampling total must be positive and probabilities nonnegative.")
    if not np.isclose(probabilities.sum(), 1.0, atol=1e-9):
        raise ValueError("Sampling probabilities must sum to one.")
    exact = probabilities * int(total)
    quotas = np.floor(exact).astype(np.int64)
    remainder = int(total) - int(quotas.sum())
    order = np.argsort(-(exact - quotas), kind="stable")
    quotas[order[:remainder]] += 1
    return quotas


def joint_marginal_quotas(total, stage_fractions, source_fractions, source_counts=None):
    """Round a 3x3 independent mix while preserving both declared marginals."""
    stage_fractions = np.asarray(stage_fractions, dtype=np.float64)
    source_fractions = np.asarray(source_fractions, dtype=np.float64)
    stage_targets = largest_remainder_quotas(total, stage_fractions)
    source_targets = (
        largest_remainder_quotas(total, source_fractions)
        if source_counts is None
        else np.asarray(source_counts, dtype=np.int64)
    )
    if source_targets.shape != (3,) or np.any(source_targets < 0):
        raise ValueError("Source counts must contain three nonnegative values.")
    if int(source_targets.sum()) != int(total):
        raise ValueError("Source counts must sum to the sampling total.")
    exact = source_targets[:, None] * stage_fractions[None, :]
    quotas = np.floor(exact).astype(np.int64)
    row_deficits = source_targets - quotas.sum(axis=1)
    column_deficits = stage_targets - quotas.sum(axis=0)
    fractions = exact - quotas
    while int(row_deficits.sum()):
        candidates = [
            (fractions[source, stage], -source, -stage, source, stage)
            for source in range(3) for stage in range(3)
            if row_deficits[source] > 0 and column_deficits[stage] > 0
        ]
        if not candidates:
            raise AssertionError("Could not reconcile V4 source/stage quotas.")
        _, _, _, source, stage = max(candidates)
        quotas[source, stage] += 1
        row_deficits[source] -= 1
        column_deficits[stage] -= 1
    if np.any(column_deficits):
        raise AssertionError("V4 stage quotas were not preserved.")
    return quotas


def build_sampling_plan(
    engine,
    run13,
    draws,
    split_id,
    stage_fractions,
    source_fractions,
    seed,
    replace=True,
    source_counts=None,
    joint_counts=None,
):
    stage_fractions = np.asarray(stage_fractions, dtype=np.float64)
    source_fractions = np.asarray(source_fractions, dtype=np.float64)
    if stage_fractions.shape != (3,) or source_fractions.shape != (3,):
        raise ValueError("V4 sampling requires exactly three stage and source fractions.")
    if joint_counts is None:
        quotas = joint_marginal_quotas(
            int(draws), stage_fractions, source_fractions, source_counts
        )
    else:
        quotas = np.asarray(joint_counts, dtype=np.int64)
        if quotas.shape != (3, 3) or np.any(quotas < 0):
            raise ValueError("Joint counts must be a nonnegative 3x3 matrix.")
        if int(quotas.sum()) != int(draws):
            raise ValueError("Joint counts must sum to the sampling total.")
    rng = np.random.RandomState(int(seed))
    corpus_ids = []
    position_indices = []
    source_ids = []
    stage_ids = []

    engine_stages = engine["stage_ids"].astype(np.int64)
    engine_splits = engine["split_ids"].astype(np.int64)
    engine_source_counts = engine["source_counts"].astype(np.float64)
    run13_stages = run13["stage_ids"].astype(np.int64)
    run13_splits = run13["split_ids"].astype(np.int64)
    strata = []
    for source_id in range(3):
        for stage_id in range(3):
            quota = int(quotas[source_id, stage_id])
            if source_id < 2:
                eligible = np.flatnonzero(
                    (engine_stages == stage_id)
                    & (engine_splits == int(split_id))
                    & (engine_source_counts[:, source_id] > 0)
                )
                weights = engine_source_counts[eligible, source_id]
                corpus_id = 0
            else:
                eligible = np.flatnonzero(
                    (run13_stages == stage_id) & (run13_splits == int(split_id))
                )
                weights = np.ones(len(eligible), dtype=np.float64)
                corpus_id = 1
            strata.append({
                "source_id": source_id,
                "stage_id": stage_id,
                "quota": quota,
                "eligible": eligible,
                "weights": weights.astype(np.float64),
                "corpus_id": corpus_id,
            })

    # Preserve the historical deterministic order for replacement sampling.
    # Unique plans process the strata with the least exclusive supply first.
    # A position can carry both engine source labels after aggregation; total
    # eligible/quota ratios alone can incorrectly give those shared positions
    # to a large source whose source-exclusive pool already covers its quota.
    if not replace:
        ownership_counts = {0: Counter(), 1: Counter()}
        for item in strata:
            if item["quota"]:
                ownership_counts[item["corpus_id"]].update(
                    map(int, item["eligible"])
                )
        for item in strata:
            owners = ownership_counts[item["corpus_id"]]
            item["exclusive_eligible"] = sum(
                owners[int(index)] == 1 for index in item["eligible"]
            )
        strata.sort(key=lambda item: (
            item["exclusive_eligible"] / max(item["quota"], 1),
            len(item["eligible"]) / max(item["quota"], 1),
            item["source_id"],
            item["stage_id"],
        ))
    used = {0: set(), 1: set()}
    available_by_stratum = np.zeros((3, 3), dtype=np.int64)
    for item in strata:
        source_id = item["source_id"]
        stage_id = item["stage_id"]
        quota = item["quota"]
        eligible = item["eligible"]
        weights = item["weights"]
        corpus_id = item["corpus_id"]
        available_by_stratum[source_id, stage_id] = len(eligible)
        if not quota:
            continue
        if not len(eligible):
            raise ValueError(
                "Sampling stratum {}/{} is empty in split {}.".format(
                    SOURCE_NAMES[source_id], STAGE_NAMES[stage_id], split_id
                )
            )
        if not replace and used[corpus_id]:
            keep = np.asarray(
                [int(index) not in used[corpus_id] for index in eligible],
                dtype=bool,
            )
            eligible = eligible[keep]
            weights = weights[keep]
        if not replace and len(eligible) < quota:
            raise ValueError(
                "Unique sampling stratum {}/{} needs {} positions but only {} "
                "remain in split {} ({} before cross-source de-duplication).".format(
                    SOURCE_NAMES[source_id], STAGE_NAMES[stage_id], quota,
                    len(eligible), split_id,
                    available_by_stratum[source_id, stage_id],
                )
            )
        probabilities = weights / weights.sum()
        chosen = rng.choice(eligible, size=quota, replace=bool(replace), p=probabilities)
        if not replace:
            used[corpus_id].update(map(int, chosen))
        corpus_ids.extend([corpus_id] * quota)
        position_indices.extend(map(int, chosen))
        source_ids.extend([source_id] * quota)
        stage_ids.extend([stage_id] * quota)

    order = rng.permutation(len(corpus_ids))
    return {
        "corpus_ids": np.asarray(corpus_ids, dtype=np.int8)[order],
        "position_indices": np.asarray(position_indices, dtype=np.int32)[order],
        "source_ids": np.asarray(source_ids, dtype=np.int8)[order],
        "stage_ids": np.asarray(stage_ids, dtype=np.int8)[order],
        "split_ids": np.full(int(draws), int(split_id), dtype=np.int8),
        "joint_quotas": quotas,
        "available_by_stratum": available_by_stratum,
        "sampling_with_replacement": bool(replace),
    }


def validate_no_cross_corpus_leakage(engine, run13):
    engine_hashes = {
        str(key): int(split)
        for key, split in zip(engine["position_hashes"], engine["split_ids"])
    }
    overlaps = 0
    for key, split in zip(run13["position_hashes"], run13["split_ids"]):
        key = str(key)
        if key in engine_hashes:
            overlaps += 1
            if engine_hashes[key] != int(split):
                raise ValueError("A D4 position crosses engine and Run13 data splits.")
    return overlaps


@dataclass
class V4BootstrapExample:
    board: np.ndarray
    policy: np.ndarray
    winner_value: float
    score: float
    requested_nodes: int
    stage_id: int
    source_id: int
    split_id: int


class V4BootstrapDataset:
    def __init__(self, engine_path, run13_path, plan_path):
        self.engine = self._load_arrays(engine_path)
        self.run13 = self._load_arrays(run13_path)
        self.plan = self._load_arrays(plan_path)
        validate_no_cross_corpus_leakage(self.engine, self.run13)
        count = len(self.plan["position_indices"])
        for field in ("corpus_ids", "source_ids", "stage_ids", "split_ids"):
            if len(self.plan[field]) != count:
                raise ValueError("Mixed V4 sampling-plan fields have inconsistent lengths.")

    @staticmethod
    def _load_arrays(path):
        # NpzFile decompresses an array on every __getitem__. Materialize each
        # field once so per-example policy decoding is not I/O bound.
        with np.load(path, allow_pickle=False) as payload:
            return {name: payload[name] for name in payload.files}

    def __len__(self):
        return len(self.plan["position_indices"])

    def __getitem__(self, item):
        corpus_id = int(self.plan["corpus_ids"][item])
        source_id = int(self.plan["source_ids"][item])
        index = int(self.plan["position_indices"][item])
        payload = self.engine if corpus_id == 0 else self.run13
        return V4BootstrapExample(
            board=payload["boards"][index].astype(np.int8),
            policy=decode_sparse_policy(payload, index),
            winner_value=float(payload["winner_means"][index]),
            score=float(payload["score_means"][index]),
            requested_nodes=int(payload["requested_nodes"][index]),
            stage_id=int(payload["stage_ids"][index]),
            source_id=source_id,
            split_id=int(payload["split_ids"][index]),
        )

    def close(self):
        self.engine.clear()
        self.run13.clear()
        self.plan.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
