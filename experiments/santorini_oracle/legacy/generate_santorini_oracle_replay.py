"""LEGACY: generate hard best-move oracle targets (negative-result experiment)."""

import argparse
from collections import deque
import json
import os
from pathlib import Path
import time

import numpy as np
from tqdm import tqdm

from benchmark_santorini_oracle_budgets import (
    file_sha256,
    select_stratified_positions,
)
from santorini.OracleResearch import (
    blended_teacher_policy,
    collect_unique_replay_positions,
    decode_policy,
)
from santorini.ReplayBuffer import load_compact_replay, save_compact_replay
from santorini.SantoriniGame import SantoriniGame
from santorini.SantoriniOracle import (
    SantoriniOracleProcess,
    external_actions_to_v3_actions,
)


SCHEMA_VERSION = 1
DEFAULT_REPLAY = "./temp/santorini_v3_run13_gumbel/latest.examples.npz"
DEFAULT_OUTPUT = "./temp/run13_oracle_teacher_5k.examples.npz"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Sample D4-unique standard Santorini replay positions and blend their "
            "MCTS policies with fixed-node oracle best moves."
        )
    )
    parser.add_argument("--replay", default=DEFAULT_REPLAY)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--positions", type=int, default=5_000)
    parser.add_argument("--oracle-nodes", type=int, default=500_000)
    parser.add_argument("--oracle-weight", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--oracle-binary")
    parser.add_argument(
        "--records-out",
        help="Append-only resume file; defaults beside --output with .records.jsonl.",
    )
    parser.add_argument(
        "--metadata-out",
        help="Final JSON summary; defaults beside --output with .metadata.json.",
    )
    parser.add_argument(
        "--no-symmetries",
        action="store_true",
        help="Write one example per sampled position instead of all eight D4 transforms.",
    )
    return parser.parse_args()


def companion_path(output_path, suffix):
    output = Path(output_path)
    if output.name.endswith(".examples.npz"):
        stem = output.name[:-len(".examples.npz")]
        return str(output.with_name(stem + suffix))
    return str(output) + suffix


def experiment_metadata(args, replay_path, replay_digest, selection, records_path, metadata_path):
    return {
        "type": "metadata",
        "schema_version": SCHEMA_VERSION,
        "replay_path": os.path.abspath(replay_path),
        "replay_sha256": replay_digest,
        "output_path": os.path.abspath(args.output),
        "records_path": os.path.abspath(records_path),
        "metadata_path": os.path.abspath(metadata_path),
        "positions": len(selection),
        "oracle_nodes": int(args.oracle_nodes),
        "oracle_weight": float(args.oracle_weight),
        "seed": int(args.seed),
        "augment_symmetries": not args.no_symmetries,
        "selection": selection,
    }


def _metadata_identity(metadata):
    return {
        key: metadata[key]
        for key in (
            "schema_version",
            "replay_path",
            "replay_sha256",
            "output_path",
            "positions",
            "oracle_nodes",
            "oracle_weight",
            "seed",
            "augment_symmetries",
            "selection",
        )
    }


def load_or_initialize_records(path, metadata):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as output:
            output.write(json.dumps(metadata, sort_keys=True) + "\n")
        return []

    with open(path) as source:
        lines = [line for line in source if line.strip()]
    if not lines:
        raise ValueError("Existing records file is empty: {}".format(path))
    stored_metadata = json.loads(lines[0])
    if _metadata_identity(stored_metadata) != _metadata_identity(metadata):
        raise ValueError(
            "Existing records metadata does not match this teacher run. "
            "Choose a different --records-out/--output path."
        )
    records = [json.loads(line) for line in lines[1:]]
    completed_ids = [int(record["position_id"]) for record in records]
    if len(completed_ids) != len(set(completed_ids)):
        raise ValueError("Records file contains duplicate position ids.")
    return records


def append_record(path, record):
    with open(path, "a") as output:
        output.write(json.dumps(record, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())


def analyze_position(game, oracle, position, board, nodes):
    started = time.perf_counter()
    oracle.reset()
    search_started = time.perf_counter()
    response = oracle.analyze(board, nodes=nodes)
    finished = time.perf_counter()
    best = response["best_move"]
    action_indices = external_actions_to_v3_actions(game, board, best["actions"])
    record = dict(position)
    record.update({
        "type": "position",
        "oracle_action": best["action"],
        "oracle_actions": best["actions"],
        "oracle_action_indices": action_indices,
        "oracle_next_fen": best["next_fen"],
        "oracle_score": int(best["score"]),
        "best_move_depth": int(best["depth"]),
        "completed_depth": int(response["completed_depth"]),
        "nodes_visited": int(response["nodes_visited"]),
        "requested_nodes": int(response["requested_nodes"]),
        "elapsed_seconds": float(finished - started),
        "search_elapsed_seconds": float(finished - search_started),
        "reset_elapsed_seconds": float(search_started - started),
    })
    return record


def materialize_teacher_replay(
    replay_path,
    output_path,
    records,
    oracle_weight,
    augment_symmetries=True,
):
    game = SantoriniGame(5, true_random_placement=True, sequential_placement=True)
    examples = []
    source_top1_matches = []
    source_oracle_mass = []
    records = sorted(records, key=lambda record: int(record["position_id"]))

    with np.load(replay_path, allow_pickle=False) as payload:
        action_size = int(payload["action_size"][0])
        if action_size != game.getActionSize():
            raise ValueError(
                "Replay action size {} does not match V3 action size {}.".format(
                    action_size, game.getActionSize()
                )
            )
        for record in records:
            replay_index = int(record["replay_index"])
            board = payload["boards"][replay_index].astype(int)
            value = float(payload["values"][replay_index])
            source_policy = decode_policy(payload, replay_index)
            oracle_actions = [int(action) for action in record["oracle_action_indices"]]
            valids = game.getValidMoves(board, 1).astype(bool)
            if not all(valids[action] for action in oracle_actions):
                raise ValueError("Oracle target contains an illegal V3 action.")
            blended = blended_teacher_policy(source_policy, oracle_actions, oracle_weight)
            if np.any(blended[~valids] > 1e-7):
                raise ValueError("Blended policy assigns probability to an illegal action.")

            source_top1_matches.append(int(np.argmax(source_policy)) in oracle_actions)
            source_oracle_mass.append(float(source_policy[oracle_actions].sum()))
            symmetries = game.getSymmetries(board, blended) if augment_symmetries else [(board, blended)]
            examples.extend((sym_board, sym_policy, value) for sym_board, sym_policy in symmetries)

    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    temporary_path = output_path + ".tmp"
    try:
        save_compact_replay(temporary_path, [deque(examples)])
        loaded = load_compact_replay(temporary_path)
        if len(loaded) != 1 or len(loaded[0]) != len(examples):
            raise ValueError("Written teacher replay failed its round-trip count check.")
        os.replace(temporary_path, output_path)
    finally:
        try:
            os.remove(temporary_path)
        except FileNotFoundError:
            pass

    return {
        "base_positions": len(records),
        "augmented_examples": len(examples),
        "history_windows": 1,
        "source_top1_oracle_agreement": float(np.mean(source_top1_matches)),
        "mean_source_policy_mass_on_oracle_actions": float(np.mean(source_oracle_mass)),
    }


def write_json_atomic(path, payload):
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary_path = path + ".tmp"
    with open(temporary_path, "w") as output:
        json.dump(payload, output, indent=2, sort_keys=True)
    os.replace(temporary_path, path)


def main():
    args = parse_args()
    if args.positions < 1:
        raise ValueError("--positions must be positive.")
    if args.oracle_nodes < 1:
        raise ValueError("--oracle-nodes must be positive.")
    if not 0 <= args.oracle_weight <= 1:
        raise ValueError("--oracle-weight must be between zero and one.")
    if not os.path.isfile(args.replay):
        raise FileNotFoundError("Replay file not found: {}".format(args.replay))

    records_path = args.records_out or companion_path(args.output, ".records.jsonl")
    metadata_path = args.metadata_out or companion_path(args.output, ".metadata.json")
    print("Hashing replay...")
    replay_digest = file_sha256(args.replay)
    print("Collecting D4-unique standard positions...")
    unique_by_stage = collect_unique_replay_positions(args.replay)
    selection = select_stratified_positions(unique_by_stage, args.positions, args.seed)
    metadata = experiment_metadata(
        args, args.replay, replay_digest, selection, records_path, metadata_path
    )
    records = load_or_initialize_records(records_path, metadata)
    completed = {int(record["position_id"]) for record in records}
    pending = [position for position in selection if int(position["position_id"]) not in completed]
    stage_counts = {
        stage: sum(position["stage"] == stage for position in selection) for stage in STAGES
    }
    print(
        "Selected {} positions (early={}, middle={}, late={}); resuming with {} complete.".format(
            len(selection), stage_counts["early"], stage_counts["middle"],
            stage_counts["late"], len(records)
        )
    )

    game = SantoriniGame(5, true_random_placement=True, sequential_placement=True)
    oracle_info = None
    with np.load(args.replay, allow_pickle=False) as payload:
        with SantoriniOracleProcess(args.oracle_binary) as oracle:
            oracle_info = dict(oracle.info)
            for position in tqdm(pending, desc="Oracle teacher labels"):
                board = payload["boards"][int(position["replay_index"])].astype(int)
                record = analyze_position(game, oracle, position, board, args.oracle_nodes)
                append_record(records_path, record)
                records.append(record)

    records.sort(key=lambda record: int(record["position_id"]))
    if len(records) != len(selection):
        raise RuntimeError("Teacher generation ended without all selected positions.")
    dataset = materialize_teacher_replay(
        args.replay,
        args.output,
        records,
        args.oracle_weight,
        augment_symmetries=not args.no_symmetries,
    )
    elapsed = [float(record["elapsed_seconds"]) for record in records]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_unix": time.time(),
        "replay_path": os.path.abspath(args.replay),
        "replay_sha256": replay_digest,
        "output_path": os.path.abspath(args.output),
        "records_path": os.path.abspath(records_path),
        "positions": len(records),
        "stage_counts": stage_counts,
        "oracle_nodes": int(args.oracle_nodes),
        "oracle_weight": float(args.oracle_weight),
        "seed": int(args.seed),
        "augment_symmetries": not args.no_symmetries,
        "oracle": oracle_info,
        "independent_searches": True,
        "transposition_table_reset_before_each_query": True,
        "oracle_elapsed_seconds": float(sum(elapsed)),
        "oracle_mean_seconds_per_position": float(np.mean(elapsed)),
        "dataset": dataset,
    }
    write_json_atomic(metadata_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("Teacher replay: {}".format(os.path.abspath(args.output)))
    print("Metadata: {}".format(os.path.abspath(metadata_path)))
    print("Resume records: {}".format(os.path.abspath(records_path)))


if __name__ == "__main__":
    main()
