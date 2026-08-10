"""Materialize oracle-labeled Run13 replay positions as a V4 bootstrap component."""

import argparse
import hashlib
import json
import os

import numpy as np

from build_santorini_v4_corpus import STAGE_IDS, _split_id
from santorini.D4Canonical import canonicalize_board_policy
from santorini.OracleResearch import canonical_d4_fen, file_sha256, stage_for_builds
from santorini.SantoriniGame import SantoriniGame
from santorini.SantoriniOracle import anonymous_board_key


SCHEMA_VERSION = 1


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", required=True)
    parser.add_argument("--records", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report-out")
    parser.add_argument("--selection-fraction", type=float, default=0.10)
    parser.add_argument("--test-fraction", type=float, default=0.10)
    parser.add_argument("--split-seed", type=int, default=20260808)
    return parser.parse_args()


def _decode_policy(payload, index):
    action_size = int(payload["action_size"][0])
    start = int(payload["policy_offsets"][index])
    end = int(payload["policy_offsets"][index + 1])
    policy = np.zeros(action_size, dtype=np.float64)
    policy[payload["policy_indices"][start:end].astype(np.int64)] = (
        payload["policy_values"][start:end]
    )
    if not np.isclose(policy.sum(), 1.0, atol=1e-5):
        raise ValueError("Run13 replay policy does not sum to one.")
    return policy / policy.sum()


def _load_records(path):
    with open(path) as source:
        lines = [json.loads(line) for line in source if line.strip()]
    if not lines or lines[0].get("type") != "metadata":
        raise ValueError("Run13 records are missing their metadata header.")
    records = lines[1:]
    if len(records) != int(lines[0]["positions"]):
        raise ValueError("Run13 records are incomplete.")
    ids = [int(record["position_id"]) for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Run13 records contain duplicate position IDs.")
    return lines[0], sorted(records, key=lambda record: int(record["position_id"]))


def build_component(args):
    if args.selection_fraction < 0 or args.test_fraction < 0:
        raise ValueError("Split fractions cannot be negative.")
    if args.selection_fraction + args.test_fraction >= 1:
        raise ValueError("Selection and test fractions must sum to less than one.")
    metadata, records = _load_records(args.records)
    if file_sha256(args.replay) != metadata["replay_sha256"]:
        raise ValueError("Run13 replay digest does not match label metadata.")
    game = SantoriniGame(5, sequential_placement=True)
    boards = []
    winner_means = []
    score_means = []
    requested_nodes = []
    actual_nodes_means = []
    mate_rates = []
    completed_depths = []
    stage_ids = []
    split_ids = []
    replay_indices = []
    position_hashes = []
    policy_offsets = [0]
    policy_indices = []
    policy_values = []

    with np.load(args.replay, allow_pickle=False) as replay:
        if int(replay["action_size"][0]) != game.getActionSize():
            raise ValueError("Run13 replay action size is incompatible with V4.")
        for record in records:
            replay_index = int(record["replay_index"])
            board = replay["boards"][replay_index].astype(int)
            if canonical_d4_fen(board) != record["fen"]:
                raise ValueError("Run13 record FEN no longer matches its replay board.")
            if stage_for_builds(int(np.sum(board[1]))) != record["stage"]:
                raise ValueError("Run13 record stage disagrees with its replay board.")
            policy = _decode_policy(replay, replay_index)
            valids = game.getValidMoves(board, 1).astype(bool)
            if np.any(policy[~valids] > 1e-7):
                raise ValueError("Run13 replay policy assigns mass to an illegal action.")
            canonical_board, canonical_policy, canonical_key = canonicalize_board_policy(
                game, board, policy
            )
            position_hash = hashlib.sha256(canonical_key).hexdigest()
            if position_hash != record["position_hash"]:
                raise ValueError("Run13 canonical position hash changed during materialization.")
            indices = np.flatnonzero(canonical_policy)
            values = canonical_policy[indices]
            if not np.isclose(values.sum(), 1.0, atol=1e-6):
                raise ValueError("Canonical Run13 policy does not sum to one.")

            label = record["label"]
            boards.append(canonical_board.astype(np.int8))
            winner_means.append(float(record["replay_value_mean"]))
            score_means.append(float(label["score"]))
            requested_nodes.append(int(label["requested_nodes"]))
            actual_nodes_means.append(float(label["actual_nodes"]))
            mate_rates.append(float(bool(label["mate_band"])))
            completed_depths.append(int(label["completed_depth"]))
            stage_ids.append(STAGE_IDS[record["stage"]])
            split_ids.append(_split_id(
                "run13-window:{}".format(record["history_window"]),
                args.split_seed,
                args.selection_fraction,
                args.test_fraction,
            ))
            replay_indices.append(replay_index)
            position_hashes.append(position_hash)
            policy_indices.extend(map(int, indices))
            policy_values.extend(map(float, values))
            policy_offsets.append(len(policy_indices))

    payload = {
        "schema_version": np.asarray([SCHEMA_VERSION], dtype=np.int16),
        "action_size": np.asarray([game.getActionSize()], dtype=np.int32),
        "boards": np.asarray(boards, dtype=np.int8),
        "observation_counts": np.ones(len(boards), dtype=np.int32),
        "winner_means": np.asarray(winner_means, dtype=np.float32),
        "score_means": np.asarray(score_means, dtype=np.float32),
        "score_stddevs": np.zeros(len(boards), dtype=np.float32),
        "requested_nodes": np.asarray(requested_nodes, dtype=np.int32),
        "actual_nodes_means": np.asarray(actual_nodes_means, dtype=np.float32),
        "mate_rates": np.asarray(mate_rates, dtype=np.float32),
        "completed_depths": np.asarray(completed_depths, dtype=np.int16),
        "stage_ids": np.asarray(stage_ids, dtype=np.int8),
        "split_ids": np.asarray(split_ids, dtype=np.int8),
        "replay_indices": np.asarray(replay_indices, dtype=np.int32),
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
    report = {
        "schema_version": SCHEMA_VERSION,
        "output": output_path,
        "records": len(records),
        "replay": os.path.abspath(args.replay),
        "replay_sha256": metadata["replay_sha256"],
        "engine_digest": metadata["engine_digest"],
        "stage_node_budgets": metadata["stage_node_budgets"],
        "positions_by_stage": {
            stage: int(np.sum(np.asarray(stage_ids) == stage_id))
            for stage, stage_id in STAGE_IDS.items()
        },
        "positions_by_split": {
            name: int(np.sum(np.asarray(split_ids) == split_id))
            for name, split_id in (("train", 0), ("selection", 1), ("test", 2))
        },
        "policy_entries": len(policy_indices),
        "output_bytes": os.path.getsize(output_path),
    }
    report_path = args.report_out or output_path + ".report.json"
    temporary_report = report_path + ".tmp"
    with open(temporary_report, "w") as output:
        json.dump(report, output, indent=2, sort_keys=True)
    os.replace(temporary_report, report_path)
    return report


def main():
    print(json.dumps(build_component(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
