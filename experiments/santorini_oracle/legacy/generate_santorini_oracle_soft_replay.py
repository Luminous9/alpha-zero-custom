"""LEGACY: generate broad soft oracle targets (negative-result experiment)."""

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import time

import numpy as np
from tqdm import tqdm

from benchmark_santorini_oracle_budgets import (
    STAGES,
    default_records_path,
    file_sha256,
    select_stratified_positions,
)
from santorini.OracleResearch import (
    ParallelOraclePool,
    blend_policies,
    collect_unique_replay_positions,
    confidence_metrics,
    decode_policy,
    ranked_moves_to_v3_policy,
)
from santorini.ReplayBuffer import load_compact_replay, save_compact_replay
from santorini.SantoriniGame import SantoriniGame
from santorini.SantoriniOracle import SantoriniOracleProcess


SCHEMA_VERSION = 1
DEFAULT_REPLAY = "./temp/santorini_v3_run13_gumbel/latest.examples.npz"
DEFAULT_OUTPUT = "./temp/run13_oracle_soft_teacher_5k_candidates.examples.npz"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate confidence-filtered soft oracle policy targets."
    )
    parser.add_argument("--replay", default=DEFAULT_REPLAY)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--candidates", type=int, default=5_000)
    parser.add_argument("--shallow-nodes-per-move", type=int, default=2_000)
    parser.add_argument("--deep-nodes-per-move", type=int, default=10_000)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--score-temperature", type=float, default=100.0)
    parser.add_argument("--oracle-weight", type=float, default=0.50)
    parser.add_argument("--min-top3-jaccard", type=float, default=0.50)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Independent oracle processes to run in parallel (default: 1).",
    )
    parser.add_argument("--oracle-binary")
    parser.add_argument("--records-out")
    parser.add_argument("--metadata-out")
    parser.add_argument("--no-symmetries", action="store_true")
    return parser.parse_args()


def companion_path(output_path, suffix):
    if output_path.endswith(".examples.npz"):
        return output_path[:-len(".examples.npz")] + suffix
    return output_path + suffix


def experiment_metadata(args, replay_digest, selection, records_path, metadata_path):
    return {
        "type": "metadata",
        "schema_version": SCHEMA_VERSION,
        "replay_path": os.path.abspath(args.replay),
        "replay_sha256": replay_digest,
        "output_path": os.path.abspath(args.output),
        "records_path": os.path.abspath(records_path),
        "metadata_path": os.path.abspath(metadata_path),
        "candidates": len(selection),
        "shallow_nodes_per_move": int(args.shallow_nodes_per_move),
        "deep_nodes_per_move": int(args.deep_nodes_per_move),
        "top_k": int(args.top_k),
        "score_temperature": float(args.score_temperature),
        "oracle_weight": float(args.oracle_weight),
        "min_top3_jaccard": float(args.min_top3_jaccard),
        "seed": int(args.seed),
        "augment_symmetries": not args.no_symmetries,
        "selection": selection,
    }


def _metadata_identity(metadata):
    keys = (
        "schema_version", "replay_path", "replay_sha256", "output_path",
        "candidates", "shallow_nodes_per_move", "deep_nodes_per_move", "top_k",
        "score_temperature", "oracle_weight", "min_top3_jaccard", "seed",
        "augment_symmetries", "selection",
    )
    return {key: metadata[key] for key in keys}


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
    if _metadata_identity(json.loads(lines[0])) != _metadata_identity(metadata):
        raise ValueError("Existing soft-teacher records do not match this run.")
    records = [json.loads(line) for line in lines[1:]]
    ids = [int(record["position_id"]) for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Records file contains duplicate position ids.")
    return records


def append_record(path, record):
    with open(path, "a") as output:
        output.write(json.dumps(record, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())


def analyze_candidate(oracle, board, position, args):
    analyses = {}
    for label, nodes in (
        ("shallow", args.shallow_nodes_per_move),
        ("deep", args.deep_nodes_per_move),
    ):
        started = time.perf_counter()
        oracle.reset()
        response = oracle.analyze_root_moves(
            board, nodes_per_move=nodes, top_k=args.top_k
        )
        response["elapsed_seconds"] = float(time.perf_counter() - started)
        analyses[label] = response
    confidence = confidence_metrics(
        analyses["shallow"], analyses["deep"], args.score_temperature
    )
    confidence["confident"] = bool(
        confidence["top1_agreement"]
        and confidence["top3_jaccard"] >= float(args.min_top3_jaccard)
    )
    record = dict(position)
    record.update({"type": "position", "analyses": analyses, "confidence": confidence})
    return record


def materialize(replay_path, output_path, records, args):
    game = SantoriniGame(5, sequential_placement=True)
    accepted = [record for record in records if record["confidence"]["confident"]]
    if not accepted:
        raise ValueError("No candidates passed the confidence filter.")
    examples = []
    source_mass = []
    source_top1 = []
    with np.load(replay_path, allow_pickle=False) as payload:
        for record in accepted:
            replay_index = int(record["replay_index"])
            board = payload["boards"][replay_index].astype(int)
            value = float(payload["values"][replay_index])
            source = decode_policy(payload, replay_index)
            moves = record["analyses"]["deep"]["moves"]
            oracle = ranked_moves_to_v3_policy(
                game, board, moves, args.score_temperature
            )
            blended = blend_policies(source, oracle, args.oracle_weight)
            valids = game.getValidMoves(board, 1).astype(bool)
            if np.any(blended[~valids] > 1e-7):
                raise ValueError("Soft teacher assigns probability to an illegal action.")
            oracle_support = np.flatnonzero(oracle)
            source_mass.append(float(source[oracle_support].sum()))
            source_top1.append(int(np.argmax(source)) in set(oracle_support.tolist()))
            symmetries = (
                game.getSymmetries(board, blended)
                if not args.no_symmetries else [(board, blended)]
            )
            examples.extend((sym_board, sym_policy, value) for sym_board, sym_policy in symmetries)

    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    temporary_path = output_path + ".tmp"
    try:
        save_compact_replay(temporary_path, [deque(examples)])
        loaded = load_compact_replay(temporary_path)
        if len(loaded) != 1 or len(loaded[0]) != len(examples):
            raise ValueError("Soft teacher replay failed round-trip validation.")
        os.replace(temporary_path, output_path)
    finally:
        try:
            os.remove(temporary_path)
        except FileNotFoundError:
            pass
    return {
        "accepted_positions": len(accepted),
        "rejected_positions": len(records) - len(accepted),
        "acceptance_rate": float(len(accepted) / len(records)),
        "accepted_by_stage": {
            stage: sum(record["stage"] == stage for record in accepted) for stage in STAGES
        },
        "augmented_examples": len(examples),
        "source_top1_in_oracle_support_rate": float(np.mean(source_top1)),
        "mean_source_mass_on_oracle_support": float(np.mean(source_mass)),
    }


def write_json_atomic(path, payload):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary_path = path + ".tmp"
    with open(temporary_path, "w") as output:
        json.dump(payload, output, indent=2, sort_keys=True)
    os.replace(temporary_path, path)


def main():
    args = parse_args()
    if (
        args.candidates < 1 or args.shallow_nodes_per_move < 1
        or args.top_k < 1 or args.workers < 1
    ):
        raise ValueError("Candidates, node budgets, top-k, and workers must be positive.")
    if args.deep_nodes_per_move <= args.shallow_nodes_per_move:
        raise ValueError("Deep nodes per move must exceed shallow nodes per move.")
    if args.score_temperature <= 0:
        raise ValueError("Score temperature must be positive.")
    if not 0 <= args.oracle_weight <= 1 or not 0 <= args.min_top3_jaccard <= 1:
        raise ValueError("Oracle weight and minimum Jaccard must be between zero and one.")

    records_path = args.records_out or default_records_path(args.output)
    metadata_path = args.metadata_out or companion_path(args.output, ".metadata.json")
    replay_digest = file_sha256(args.replay)
    selection = select_stratified_positions(
        collect_unique_replay_positions(args.replay), args.candidates, args.seed
    )
    metadata = experiment_metadata(
        args, replay_digest, selection, records_path, metadata_path
    )
    records = load_or_initialize_records(records_path, metadata)
    completed = {int(record["position_id"]) for record in records}
    pending = [position for position in selection if int(position["position_id"]) not in completed]
    print("Selected {} candidates; resuming with {} complete.".format(len(selection), len(records)))

    generation_started = time.perf_counter()
    with SantoriniOracleProcess(args.oracle_binary) as oracle:
        oracle_info = dict(oracle.info)

    with np.load(args.replay, allow_pickle=False) as payload:
        tasks = [
            (
                payload["boards"][int(position["replay_index"])].astype(int),
                position,
            )
            for position in pending
        ]

    if args.workers == 1:
        with SantoriniOracleProcess(args.oracle_binary) as oracle:
            for board, position in tqdm(tasks, desc="Soft oracle candidates"):
                record = analyze_candidate(oracle, board, position, args)
                append_record(records_path, record)
                records.append(record)
    else:
        pool = ParallelOraclePool(args.oracle_binary)
        try:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = [
                    executor.submit(pool.analyze, analyze_candidate, board, position, args)
                    for board, position in tasks
                ]
                for future in tqdm(
                    as_completed(futures), total=len(futures),
                    desc="Soft oracle candidates",
                ):
                    record = future.result()
                    append_record(records_path, record)
                    records.append(record)
        finally:
            pool.close()
    generation_wall_seconds = time.perf_counter() - generation_started

    records.sort(key=lambda record: int(record["position_id"]))
    if len(records) != len(selection):
        raise RuntimeError("Soft teacher generation did not finish all candidates.")
    dataset = materialize(args.replay, args.output, records, args)
    result = {
        "schema_version": SCHEMA_VERSION,
        "replay_path": os.path.abspath(args.replay),
        "replay_sha256": replay_digest,
        "output_path": os.path.abspath(args.output),
        "records_path": os.path.abspath(records_path),
        "oracle": oracle_info,
        "candidates": len(records),
        "shallow_nodes_per_move": int(args.shallow_nodes_per_move),
        "deep_nodes_per_move": int(args.deep_nodes_per_move),
        "top_k": int(args.top_k),
        "score_temperature": float(args.score_temperature),
        "oracle_weight": float(args.oracle_weight),
        "min_top3_jaccard": float(args.min_top3_jaccard),
        "seed": int(args.seed),
        "workers": int(args.workers),
        "augment_symmetries": not args.no_symmetries,
        "dataset": dataset,
        "generation_wall_seconds": float(generation_wall_seconds),
        "oracle_elapsed_seconds": float(sum(
            record["analyses"][label]["elapsed_seconds"]
            for record in records for label in ("shallow", "deep")
        )),
    }
    write_json_atomic(metadata_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    print("Soft teacher replay: {}".format(os.path.abspath(args.output)))
    print("Metadata: {}".format(os.path.abspath(metadata_path)))
    print("Resume records: {}".format(os.path.abspath(records_path)))


if __name__ == "__main__":
    main()
