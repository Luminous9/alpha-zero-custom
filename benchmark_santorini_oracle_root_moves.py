"""Calibrate ranked-root Santorini oracle stability and soft policy targets."""

import argparse
import json
import os
import time

import numpy as np
from tqdm import tqdm

from benchmark_santorini_oracle_budgets import (
    default_records_path,
    file_sha256,
    select_stratified_positions,
)
from santorini.OracleResearch import (
    STAGES,
    collect_unique_replay_positions,
    confidence_metrics,
    normalized_entropy,
    score_softmax,
    top_overlap,
)
from santorini.SantoriniOracle import SantoriniOracleProcess


SCHEMA_VERSION = 1
DEFAULT_REPLAY = "./temp/santorini_v3_run13_gumbel/latest.examples.npz"
DEFAULT_OUTPUT = "./temp/run13_oracle_root_stability.json"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare ranked oracle root moves at two per-move node budgets."
    )
    parser.add_argument("--replay", default=DEFAULT_REPLAY)
    parser.add_argument("--positions", type=int, default=200)
    parser.add_argument("--shallow-nodes-per-move", type=int, default=2_000)
    parser.add_argument("--deep-nodes-per-move", type=int, default=10_000)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--score-temperature", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--oracle-binary")
    parser.add_argument("--json-out", default=DEFAULT_OUTPUT)
    parser.add_argument("--records-out")
    return parser.parse_args()


def experiment_metadata(args, replay_digest, selection):
    return {
        "type": "metadata",
        "schema_version": SCHEMA_VERSION,
        "replay_path": os.path.abspath(args.replay),
        "replay_sha256": replay_digest,
        "positions": len(selection),
        "shallow_nodes_per_move": int(args.shallow_nodes_per_move),
        "deep_nodes_per_move": int(args.deep_nodes_per_move),
        "top_k": int(args.top_k),
        "score_temperature": float(args.score_temperature),
        "seed": int(args.seed),
        "selection": selection,
    }


def _metadata_identity(metadata):
    keys = (
        "schema_version", "replay_path", "replay_sha256", "positions",
        "shallow_nodes_per_move", "deep_nodes_per_move", "top_k",
        "score_temperature", "seed", "selection",
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
        raise ValueError("Existing ranked-root records do not match this experiment.")
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


def analyze_position(oracle, board, position, args):
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
    record = dict(position)
    record.update({
        "type": "position",
        "analyses": analyses,
        "confidence": confidence_metrics(
            analyses["shallow"], analyses["deep"], args.score_temperature
        ),
    })
    return record


def _mean(values):
    return float(np.mean(values)) if values else None


def summarize_group(records):
    if not records:
        return {"positions": 0}
    shallow_seconds = sum(r["analyses"]["shallow"]["elapsed_seconds"] for r in records)
    deep_seconds = sum(r["analyses"]["deep"]["elapsed_seconds"] for r in records)
    return {
        "positions": len(records),
        "top1_agreement": _mean([r["confidence"]["top1_agreement"] for r in records]),
        "top3_jaccard": _mean([r["confidence"]["top3_jaccard"] for r in records]),
        "confident_rate": _mean([r["confidence"]["confident"] for r in records]),
        "deep_score_margin_mean": _mean([
            r["confidence"]["deep_score_margin"] for r in records
            if r["confidence"]["deep_score_margin"] is not None
        ]),
        "deep_soft_target_entropy_mean": _mean([
            r["confidence"]["deep_soft_target_entropy"] for r in records
        ]),
        "legal_move_count_mean": _mean([
            r["analyses"]["deep"]["legal_move_count"] for r in records
        ]),
        "shallow_total_seconds": float(shallow_seconds),
        "deep_total_seconds": float(deep_seconds),
        "total_seconds": float(shallow_seconds + deep_seconds),
        "total_nodes_visited": int(sum(
            r["analyses"][label]["total_nodes_visited"]
            for r in records for label in ("shallow", "deep")
        )),
    }


def summarize(records):
    return {
        "all": summarize_group(records),
        "by_stage": {
            stage: summarize_group([record for record in records if record["stage"] == stage])
            for stage in STAGES
        },
    }


def write_json_atomic(path, payload):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary_path = path + ".tmp"
    with open(temporary_path, "w") as output:
        json.dump(payload, output, indent=2, sort_keys=True)
    os.replace(temporary_path, path)


def main():
    args = parse_args()
    if args.positions < 1 or args.shallow_nodes_per_move < 1 or args.top_k < 1:
        raise ValueError("Positions, node budgets, and top-k must be positive.")
    if args.deep_nodes_per_move <= args.shallow_nodes_per_move:
        raise ValueError("Deep nodes per move must exceed shallow nodes per move.")
    if args.score_temperature <= 0:
        raise ValueError("Score temperature must be positive.")

    replay_digest = file_sha256(args.replay)
    unique = collect_unique_replay_positions(args.replay)
    selection = select_stratified_positions(unique, args.positions, args.seed)
    metadata = experiment_metadata(args, replay_digest, selection)
    records_path = args.records_out or default_records_path(args.json_out)
    records = load_or_initialize_records(records_path, metadata)
    completed = {int(record["position_id"]) for record in records}
    pending = [position for position in selection if int(position["position_id"]) not in completed]
    print("Selected {} positions; resuming with {} complete.".format(len(selection), len(records)))

    with np.load(args.replay, allow_pickle=False) as payload:
        with SantoriniOracleProcess(args.oracle_binary) as oracle:
            oracle_info = dict(oracle.info)
            for position in tqdm(pending, desc="Ranked root calibration"):
                board = payload["boards"][int(position["replay_index"])].astype(int)
                record = analyze_position(oracle, board, position, args)
                append_record(records_path, record)
                records.append(record)

    records.sort(key=lambda record: int(record["position_id"]))
    if len(records) != len(selection):
        raise RuntimeError("Ranked-root calibration did not finish all positions.")
    result = {
        "schema_version": SCHEMA_VERSION,
        "replay_path": os.path.abspath(args.replay),
        "replay_sha256": replay_digest,
        "records_path": os.path.abspath(records_path),
        "oracle": oracle_info,
        "positions": len(records),
        "shallow_nodes_per_move": int(args.shallow_nodes_per_move),
        "deep_nodes_per_move": int(args.deep_nodes_per_move),
        "top_k": int(args.top_k),
        "score_temperature": float(args.score_temperature),
        "seed": int(args.seed),
        "summary": summarize(records),
    }
    write_json_atomic(args.json_out, result)
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    print("Summary: {}".format(os.path.abspath(args.json_out)))
    print("Records: {}".format(os.path.abspath(records_path)))


if __name__ == "__main__":
    main()
