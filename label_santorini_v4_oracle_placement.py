"""Label all joint placement boundaries with santorini-ai and factor them for V4."""

import argparse
import hashlib
import json
import os
import time

import numpy as np

from santorini.D4Canonical import canonicalize_board
from santorini.OracleResearch import file_sha256
from santorini.SantoriniGame import SantoriniGame
from santorini.SantoriniOracle import (
    SantoriniOracleProcess,
    canonical_board_to_fen,
    external_joint_placement_locations,
)
from santorini.V4Placement import (
    aggregate_teacher_observations,
    factor_joint_placement,
    joint_boundary_orbits,
    legal_unordered_pairs,
    symmetrize_joint_pair_scores,
)


SCHEMA_VERSION = 1


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-binary")
    parser.add_argument("--records-out", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report-out")
    parser.add_argument("--nodes-per-move", type=int, default=2_000)
    parser.add_argument("--policy-temperature", type=float, default=100.0)
    parser.add_argument("--value-temperature", type=float, default=261.8)
    parser.add_argument("--max-boundaries", type=int)
    return parser.parse_args()


def _suite(game):
    items = []
    for boundary_id, board in enumerate(joint_boundary_orbits(game)):
        _, _, key = canonicalize_board(board)
        items.append({
            "boundary_id": boundary_id,
            "board": board,
            "position_hash": hashlib.sha256(key).hexdigest(),
            "fen": canonical_board_to_fen(board),
            "worker_count": int(np.count_nonzero(board[0])),
            "legal_pair_count": len(legal_unordered_pairs(game, board)),
        })
    return items


def _suite_fingerprint(items):
    digest = hashlib.sha256()
    for item in items:
        digest.update(item["position_hash"].encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _metadata(args, oracle, suite):
    return {
        "type": "metadata",
        "schema_version": SCHEMA_VERSION,
        "contract": "p1c_santorini_ai_joint_placement_scores",
        "engine_binary": str(oracle.binary_path.resolve()),
        "engine_sha256": file_sha256(oracle.binary_path),
        "engine_info": oracle.info,
        "nodes_per_move": int(args.nodes_per_move),
        "tt_policy": "reset_per_root_move",
        "boundary_count": len(suite),
        "suite_fingerprint": _suite_fingerprint(suite),
    }


def _load_or_initialize_records(path, expected_metadata):
    path = os.path.abspath(path)
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as output:
            output.write(json.dumps(expected_metadata, sort_keys=True) + "\n")
        return {}
    with open(path) as source:
        lines = [json.loads(line) for line in source if line.strip()]
    if not lines or lines[0].get("type") != "metadata":
        raise ValueError("Oracle placement records are missing metadata.")
    for field in (
        "schema_version", "contract", "engine_sha256", "nodes_per_move",
        "tt_policy", "boundary_count", "suite_fingerprint",
    ):
        if lines[0].get(field) != expected_metadata.get(field):
            raise ValueError("Oracle placement metadata changed at {}.".format(field))
    records = {}
    for record in lines[1:]:
        boundary_id = int(record["boundary_id"])
        if boundary_id in records:
            raise ValueError("Duplicate oracle placement boundary record.")
        records[boundary_id] = record
    return records


def _label_boundaries(args, game, oracle, suite):
    metadata = _metadata(args, oracle, suite)
    records = _load_or_initialize_records(args.records_out, metadata)
    selected = suite
    if args.max_boundaries is not None:
        if args.max_boundaries < 1:
            raise ValueError("--max-boundaries must be positive.")
        selected = selected[: int(args.max_boundaries)]
    started = time.perf_counter()
    with open(args.records_out, "a") as output:
        for item in selected:
            if item["boundary_id"] in records:
                continue
            oracle.reset()
            response = oracle.analyze_root_moves_fen(
                item["fen"], nodes_per_move=args.nodes_per_move, top_k=1_000
            )
            if response.get("tt_policy") != "reset_per_root_move":
                raise ValueError("Oracle did not declare independent root-move searches.")
            if int(response["legal_move_count"]) != item["legal_pair_count"]:
                raise ValueError("Oracle placement branching factor disagrees with V4.")
            if len(response["moves"]) != item["legal_pair_count"]:
                raise ValueError("Oracle placement response was truncated.")
            pairs = [
                external_joint_placement_locations(move["actions"])
                for move in response["moves"]
            ]
            if set(pairs) != set(legal_unordered_pairs(game, item["board"])):
                raise ValueError("Oracle and V4 legal placement pairs disagree.")
            record = {
                "type": "boundary",
                "schema_version": SCHEMA_VERSION,
                "boundary_id": item["boundary_id"],
                "position_hash": item["position_hash"],
                "worker_count": item["worker_count"],
                "fen": item["fen"],
                "legal_pair_count": item["legal_pair_count"],
                "total_nodes_visited": int(response["total_nodes_visited"]),
                "moves": response["moves"],
            }
            output.write(json.dumps(record, sort_keys=True) + "\n")
            output.flush()
            records[item["boundary_id"]] = record
            print(
                "boundary {}/{}: workers={} pairs={} nodes={}".format(
                    item["boundary_id"] + 1, len(selected), item["worker_count"],
                    item["legal_pair_count"], record["total_nodes_visited"],
                ),
                flush=True,
            )
    return metadata, records, time.perf_counter() - started


def _materialize(args, game, suite, metadata, records, labeling_seconds):
    observations = []
    processed = []
    symmetry_diagnostics = []
    for item in suite:
        record = records.get(item["boundary_id"])
        if record is None:
            continue
        pairs = [
            external_joint_placement_locations(move["actions"])
            for move in record["moves"]
        ]
        scores = [float(move["score"]) for move in record["moves"]]
        _, symmetry = symmetrize_joint_pair_scores(
            game, item["board"], pairs, scores
        )
        symmetry_diagnostics.append(symmetry)
        observations.extend(factor_joint_placement(
            game,
            item["board"],
            pairs,
            scores,
            args.policy_temperature,
            args.value_temperature,
        ))
        processed.append(item)
    if not processed:
        raise ValueError("No oracle placement boundaries were available to materialize.")
    aggregates = aggregate_teacher_observations(game, observations)
    keys = sorted(aggregates)
    boards = []
    observation_counts = []
    reach_weights = []
    score_means = []
    oracle_value_means = []
    pair_support_means = []
    worker_counts = []
    position_hashes = []
    policy_offsets = [0]
    policy_indices = []
    policy_values = []
    for key in keys:
        record = aggregates[key]
        weight = record["reach_weight"]
        policy = record["policy_sum"] / weight
        policy /= policy.sum()
        nonzero = np.flatnonzero(policy > 0)
        boards.append(record["board"])
        observation_counts.append(record["observation_count"])
        reach_weights.append(weight)
        score_means.append(record["score_sum"] / weight)
        oracle_value_means.append(record["value_sum"] / weight)
        pair_support_means.append(record["pair_support_sum"] / weight)
        worker_counts.append(int(np.count_nonzero(record["board"][0])))
        position_hashes.append(hashlib.sha256(key).hexdigest())
        policy_indices.extend(map(int, nonzero))
        policy_values.extend(map(float, policy[nonzero]))
        policy_offsets.append(len(policy_indices))

    count = len(boards)
    payload = {
        "schema_version": np.asarray([SCHEMA_VERSION], dtype=np.int16),
        "action_size": np.asarray([game.getActionSize()], dtype=np.int32),
        "boards": np.asarray(boards, dtype=np.int8),
        "observation_counts": np.asarray(observation_counts, dtype=np.int32),
        "teacher_reach_weights": np.asarray(reach_weights, dtype=np.float32),
        # There is deliberately no fabricated completed-game outcome. A mixed
        # component must source z from Run13 continuations.
        "winner_means": np.zeros(count, dtype=np.float32),
        "has_completed_outcomes": np.asarray([False]),
        "score_means": np.asarray(score_means, dtype=np.float32),
        "oracle_value_means": np.asarray(oracle_value_means, dtype=np.float32),
        "pair_support_means": np.asarray(pair_support_means, dtype=np.float32),
        "score_stddevs": np.zeros(count, dtype=np.float32),
        "requested_nodes": np.full(count, int(args.nodes_per_move), dtype=np.int32),
        "actual_nodes_means": np.zeros(count, dtype=np.float32),
        "mate_rates": np.zeros(count, dtype=np.float32),
        "completed_depths": np.zeros(count, dtype=np.int16),
        "stage_ids": np.full(count, -1, dtype=np.int8),
        "split_ids": np.zeros(count, dtype=np.int8),
        "replay_indices": np.full(count, -1, dtype=np.int32),
        "worker_counts": np.asarray(worker_counts, dtype=np.int8),
        "policy_offsets": np.asarray(policy_offsets, dtype=np.int64),
        "policy_indices": np.asarray(policy_indices, dtype=np.uint16),
        "policy_values": np.asarray(policy_values, dtype=np.float32),
        "position_hashes": np.asarray(position_hashes, dtype="<U64"),
    }
    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    temporary = output_path + ".tmp.npz"
    np.savez_compressed(temporary, **payload)
    os.replace(temporary, output_path)
    counts = {
        str(worker_count): int(np.sum(np.asarray(worker_counts) == worker_count))
        for worker_count in range(4)
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "contract": "p1c_factored_santorini_ai_placement",
        "output": output_path,
        "output_sha256": file_sha256(output_path),
        "records": os.path.abspath(args.records_out),
        "records_sha256": file_sha256(args.records_out),
        "engine_sha256": metadata["engine_sha256"],
        "nodes_per_move": int(args.nodes_per_move),
        "policy_temperature": float(args.policy_temperature),
        "value_temperature": float(args.value_temperature),
        "boundaries_processed": len(processed),
        "boundaries_total": len(suite),
        "unique_positions": count,
        "unique_positions_by_worker_count": counts,
        "complete_960_orbit_coverage": count == 960,
        "labeling_seconds_this_run": float(labeling_seconds),
        "total_root_move_search_nodes": int(sum(
            records[item["boundary_id"]]["total_nodes_visited"] for item in processed
        )),
        "tt_policy": "reset_per_root_move",
        "completed_outcomes_present": False,
        "score_projection": "mean_over_parent_d4_stabilizer_orbit",
        "raw_score_symmetry": {
            "mean_orbit_range": float(np.mean([
                item["mean_raw_orbit_score_range"] for item in symmetry_diagnostics
            ])),
            "max_orbit_range": float(max(
                item["max_raw_orbit_score_range"] for item in symmetry_diagnostics
            )),
            "boundaries_with_nonzero_ranges": int(sum(
                item["nonzero_raw_orbit_score_ranges"] > 0
                for item in symmetry_diagnostics
            )),
        },
        "output_bytes": os.path.getsize(output_path),
    }
    report_path = os.path.abspath(args.report_out or output_path + ".report.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    temporary_report = report_path + ".tmp"
    with open(temporary_report, "w") as output:
        json.dump(report, output, indent=2, sort_keys=True)
    os.replace(temporary_report, report_path)
    return report


def build_component(args):
    if args.nodes_per_move < 1:
        raise ValueError("--nodes-per-move must be positive.")
    if args.policy_temperature <= 0 or args.value_temperature <= 0:
        raise ValueError("Policy and value temperatures must be positive.")
    game = SantoriniGame(5, sequential_placement=True)
    suite = _suite(game)
    with SantoriniOracleProcess(args.oracle_binary) as oracle:
        metadata, records, seconds = _label_boundaries(args, game, oracle, suite)
    return _materialize(args, game, suite, metadata, records, seconds)


def main():
    print(json.dumps(build_component(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
