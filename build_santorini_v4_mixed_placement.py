"""Combine santorini-ai placement policy with Run13 continuation outcomes."""

import argparse
from collections import Counter
import json
import os

import numpy as np

from santorini.OracleResearch import file_sha256
from santorini.SantoriniGame import SantoriniGame
from santorini.V4BootstrapCorpus import decode_sparse_policy
from santorini.V4Placement import EXPECTED_ORBITS_BY_WORKER_COUNT


SCHEMA_VERSION = 1


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-component", required=True)
    parser.add_argument("--run13-component", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report-out")
    parser.add_argument(
        "--policy-mode", choices=("oracle", "run13", "blend"), required=True
    )
    parser.add_argument(
        "--oracle-policy-weight", type=float, default=0.5,
        help="Oracle share of the policy target when --policy-mode=blend.",
    )
    parser.add_argument("--allow-incomplete-coverage", action="store_true")
    return parser.parse_args()


def _load(path):
    with np.load(path, allow_pickle=False) as source:
        return {name: source[name] for name in source.files}


def _sparse_policies(policies):
    offsets = [0]
    indices = []
    values = []
    for policy in policies:
        nonzero = np.flatnonzero(policy > 0)
        indices.extend(map(int, nonzero))
        values.extend(map(float, policy[nonzero]))
        offsets.append(len(indices))
    return {
        "policy_offsets": np.asarray(offsets, dtype=np.int64),
        "policy_indices": np.asarray(indices, dtype=np.uint16),
        "policy_values": np.asarray(values, dtype=np.float32),
    }


def _aligned_indices(payload, ordered_hashes, name):
    hashes = list(map(str, payload["position_hashes"]))
    if len(hashes) != len(set(hashes)):
        raise ValueError("{} placement component has duplicate hashes.".format(name))
    lookup = {key: index for index, key in enumerate(hashes)}
    missing = set(ordered_hashes).difference(lookup)
    extra = set(lookup).difference(ordered_hashes)
    if missing or extra:
        raise ValueError(
            "Placement components differ: {} missing and {} extra {} positions.".format(
                len(missing), len(extra), name
            )
        )
    return np.asarray([lookup[key] for key in ordered_hashes], dtype=np.int64)


def build_payload(oracle, run13, policy_mode, oracle_policy_weight):
    weight = float(oracle_policy_weight)
    if not np.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError("Oracle policy weight must be in [0, 1].")
    if policy_mode != "blend" and weight != 0.5:
        raise ValueError("--oracle-policy-weight applies only to blend mode.")
    if bool(np.asarray(oracle.get("has_completed_outcomes", [False]))[0]):
        raise ValueError("Oracle-only placement input unexpectedly declares outcomes.")
    if not bool(np.asarray(run13.get("has_completed_outcomes", [False]))[0]):
        raise ValueError("Run13 placement input must declare completed outcomes.")
    if "winner_means" not in run13 or not np.all(np.isfinite(run13["winner_means"])):
        raise ValueError("Run13 placement component must contain finite outcomes.")
    action_size = int(run13["action_size"][0])
    if int(oracle["action_size"][0]) != action_size:
        raise ValueError("Placement teacher action sizes differ.")

    ordered_hashes = list(map(str, run13["position_hashes"]))
    run_indices = _aligned_indices(run13, ordered_hashes, "Run13")
    oracle_indices = _aligned_indices(oracle, ordered_hashes, "oracle")
    game = SantoriniGame(5, sequential_placement=True)
    policies = []
    total_variations = []
    for run_index, oracle_index in zip(run_indices, oracle_indices):
        run_policy = decode_sparse_policy(run13, int(run_index)).astype(np.float64)
        oracle_policy = decode_sparse_policy(oracle, int(oracle_index)).astype(np.float64)
        for name, policy in (("Run13", run_policy), ("oracle", oracle_policy)):
            if np.any(~np.isfinite(policy)) or np.any(policy < 0):
                raise ValueError("{} placement policy is invalid.".format(name))
            if not np.isclose(policy.sum(), 1.0, atol=1e-5):
                raise ValueError("{} placement policy does not sum to one.".format(name))
        board = run13["boards"][run_index]
        if not np.array_equal(board, oracle["boards"][oracle_index]):
            raise ValueError("Placement boards disagree for a shared canonical hash.")
        valids = game.getValidMoves(board, 1).astype(bool)
        if np.any(run_policy[~valids] > 1e-7) or np.any(oracle_policy[~valids] > 1e-7):
            raise ValueError("A placement teacher assigns mass to an illegal action.")
        total_variations.append(0.5 * float(np.abs(run_policy - oracle_policy).sum()))
        if policy_mode == "oracle":
            policy = oracle_policy
        elif policy_mode == "run13":
            policy = run_policy
        else:
            policy = weight * oracle_policy + (1.0 - weight) * run_policy
        policies.append((policy / policy.sum()).astype(np.float32))

    oracle_indices = np.asarray(oracle_indices, dtype=np.int64)
    run_indices = np.asarray(run_indices, dtype=np.int64)
    count = len(run_indices)
    payload = {
        "schema_version": np.asarray([SCHEMA_VERSION], dtype=np.int16),
        "action_size": np.asarray([action_size], dtype=np.int32),
        "boards": run13["boards"][run_indices].astype(np.int8),
        # Keep sampling identical between policy bake-off arms.
        "observation_counts": run13["observation_counts"][run_indices].astype(np.int32),
        "winner_means": run13["winner_means"][run_indices].astype(np.float32),
        "has_completed_outcomes": np.asarray([True]),
        # Oracle score/value remain diagnostic; placement training consumes z.
        "score_means": oracle["score_means"][oracle_indices].astype(np.float32),
        "oracle_value_means": oracle["oracle_value_means"][oracle_indices].astype(np.float32),
        "score_stddevs": np.zeros(count, dtype=np.float32),
        "requested_nodes": oracle["requested_nodes"][oracle_indices].astype(np.int32),
        "actual_nodes_means": np.zeros(count, dtype=np.float32),
        "mate_rates": np.zeros(count, dtype=np.float32),
        "completed_depths": np.zeros(count, dtype=np.int16),
        "stage_ids": np.full(count, -1, dtype=np.int8),
        "split_ids": np.zeros(count, dtype=np.int8),
        "replay_indices": np.full(count, -1, dtype=np.int32),
        "worker_counts": run13["worker_counts"][run_indices].astype(np.int8),
        "position_hashes": np.asarray(ordered_hashes, dtype="<U64"),
        "policy_mode": np.asarray([policy_mode]),
        "oracle_policy_weight": np.asarray([
            weight if policy_mode == "blend" else float(policy_mode == "oracle")
        ], dtype=np.float32),
    }
    payload.update(_sparse_policies(policies))
    diagnostics = {
        "mean_oracle_run13_policy_tv": float(np.mean(total_variations)),
        "max_oracle_run13_policy_tv": float(np.max(total_variations)),
    }
    return payload, diagnostics


def build_component(args):
    oracle = _load(args.oracle_component)
    run13 = _load(args.run13_component)
    payload, diagnostics = build_payload(
        oracle, run13, args.policy_mode, args.oracle_policy_weight
    )
    counts = Counter(map(int, payload["worker_counts"]))
    coverage = tuple(counts[index] for index in range(4))
    if (
        not args.allow_incomplete_coverage
        and coverage != EXPECTED_ORBITS_BY_WORKER_COUNT
    ):
        raise ValueError(
            "Placement coverage is {} instead of {}.".format(
                coverage, EXPECTED_ORBITS_BY_WORKER_COUNT
            )
        )
    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    temporary = output_path + ".tmp.npz"
    np.savez_compressed(temporary, **payload)
    os.replace(temporary, output_path)
    report = {
        "schema_version": SCHEMA_VERSION,
        "contract": "p1c_mixed_placement_teacher",
        "policy_mode": args.policy_mode,
        "oracle_policy_weight": float(payload["oracle_policy_weight"][0]),
        "value_target": "run13_completed_continuation_outcome",
        "oracle_component": os.path.abspath(args.oracle_component),
        "oracle_component_sha256": file_sha256(args.oracle_component),
        "run13_component": os.path.abspath(args.run13_component),
        "run13_component_sha256": file_sha256(args.run13_component),
        "unique_positions": len(payload["boards"]),
        "unique_positions_by_worker_count": {
            str(index): int(counts[index]) for index in range(4)
        },
        **diagnostics,
        "output": output_path,
        "output_sha256": file_sha256(output_path),
        "output_bytes": os.path.getsize(output_path),
    }
    report_path = os.path.abspath(args.report_out or output_path + ".report.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    temporary_report = report_path + ".tmp"
    with open(temporary_report, "w") as output:
        json.dump(report, output, indent=2, sort_keys=True)
    os.replace(temporary_report, report_path)
    return report


def main():
    print(json.dumps(build_component(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
