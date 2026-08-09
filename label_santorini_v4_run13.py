"""Create a fresh, stage-stratified oracle-label component from Run13 replay."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm

from benchmark_santorini_oracle_budgets import collect_unique_positions, file_sha256
from santorini.OracleResearch import ParallelOraclePool, STAGES
from santorini.SantoriniOracle import anonymous_board_key, fen_to_canonical_board


SCHEMA_VERSION = 1
DEFAULT_REPLAY = "./temp/santorini_v3_run13_gumbel/latest.examples.npz"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cold-TT oracle labels for a fresh Run13-replay bootstrap component."
    )
    parser.add_argument("--replay", default=DEFAULT_REPLAY)
    parser.add_argument("--positions", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--early-nodes", type=int, default=250_000)
    parser.add_argument("--middle-nodes", type=int, default=100_000)
    parser.add_argument("--late-nodes", type=int, default=20_000)
    parser.add_argument("--oracle-binary")
    parser.add_argument("--records-out", required=True)
    parser.add_argument("--label-cache", required=True)
    parser.add_argument("--exclude-records", action="append", default=[])
    parser.add_argument("--exclude-corpus", action="append", default=[])
    return parser.parse_args()


def _score_value(score):
    score = int(score)
    if abs(score) >= 9_000:
        return float(np.sign(score))
    scaled = float(np.clip(score / 400.0, -50.0, 50.0))
    return float(2.0 / (1.0 + math.exp(-scaled)) - 1.0)


def _position_hash(fen):
    board = fen_to_canonical_board(fen)
    keys = []
    for rotations in range(4):
        rotated = np.asarray([
            np.rot90(board[0], rotations),
            np.rot90(board[1], rotations),
        ])
        keys.append(anonymous_board_key(rotated))
        keys.append(anonymous_board_key(np.asarray([
            np.fliplr(rotated[0]),
            np.fliplr(rotated[1]),
        ])))
    return hashlib.sha256(min(keys)).hexdigest()


def _excluded_fens(paths):
    excluded = set()
    for path in paths:
        if not os.path.exists(path):
            raise FileNotFoundError("Excluded records file not found: {}".format(path))
        with open(path) as source:
            for line in source:
                record = json.loads(line)
                if record.get("type") == "position":
                    excluded.add(str(record["fen"]))
    return excluded


def _excluded_hashes(paths):
    excluded = set()
    for path in paths:
        with np.load(path, allow_pickle=False) as payload:
            excluded.update(map(str, payload["position_hashes"]))
    return excluded


def _history_windows(replay_path):
    with np.load(replay_path, allow_pickle=False) as payload:
        lengths = payload["history_lengths"].astype(np.int64)
    boundaries = np.cumsum(lengths)
    return boundaries


def select_positions(replay_path, count, seed, excluded_fens, excluded_hashes):
    by_stage = collect_unique_positions(replay_path)
    boundaries = _history_windows(replay_path)
    quota, remainder = divmod(int(count), len(STAGES))
    rng = np.random.RandomState(int(seed))
    selected = []
    for stage_index, stage in enumerate(STAGES):
        candidates = [
            record
            for record in by_stage[stage]
            if record["fen"] not in excluded_fens
            and _position_hash(record["fen"]) not in excluded_hashes
        ]
        stage_quota = quota + (1 if stage_index < remainder else 0)
        if len(candidates) < stage_quota:
            raise ValueError(
                "Stage {} has only {} eligible positions for quota {}.".format(
                    stage, len(candidates), stage_quota
                )
            )
        indices = rng.choice(len(candidates), size=stage_quota, replace=False)
        for candidate_index in indices:
            record = dict(candidates[int(candidate_index)])
            record["history_window"] = int(
                np.searchsorted(boundaries, int(record["replay_index"]), side="right")
            )
            record["position_hash"] = _position_hash(record["fen"])
            selected.append(record)
    rng.shuffle(selected)
    for position_id, record in enumerate(selected):
        record["position_id"] = int(position_id)
    return selected


def _metadata_identity(metadata):
    return {
        key: metadata[key]
        for key in (
            "schema_version", "replay_sha256", "positions", "seed",
            "stage_node_budgets", "engine_digest", "selection",
        )
    }


def _load_or_initialize(path, metadata):
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w") as output:
            output.write(json.dumps(metadata, sort_keys=True) + "\n")
        return []
    with open(path) as source:
        records = [json.loads(line) for line in source if line.strip()]
    if not records or _metadata_identity(records[0]) != _metadata_identity(metadata):
        raise ValueError("Existing Run13 label records do not match this experiment.")
    positions = records[1:]
    ids = [int(record["position_id"]) for record in positions]
    if len(ids) != len(set(ids)):
        raise ValueError("Run13 label records contain duplicate position IDs.")
    return positions


def _append_record(path, record):
    with open(path, "a") as output:
        output.write(json.dumps(record, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())


def main():
    args = parse_args()
    if args.positions < 3 or args.workers < 1:
        raise ValueError("Positions must be at least three and workers must be positive.")
    stage_budgets = {
        "early": int(args.early_nodes),
        "middle": int(args.middle_nodes),
        "late": int(args.late_nodes),
    }
    if any(value < 1 for value in stage_budgets.values()):
        raise ValueError("Every stage node budget must be positive.")
    replay_digest = file_sha256(args.replay)
    selection = select_positions(
        args.replay,
        args.positions,
        args.seed,
        _excluded_fens(args.exclude_records),
        _excluded_hashes(args.exclude_corpus),
    )
    pool = ParallelOraclePool(args.oracle_binary, cache_path=args.label_cache)
    try:
        metadata = {
            "type": "metadata",
            "schema_version": SCHEMA_VERSION,
            "replay_path": os.path.abspath(args.replay),
            "replay_sha256": replay_digest,
            "positions": len(selection),
            "seed": int(args.seed),
            "stage_node_budgets": stage_budgets,
            "engine_digest": pool.engine_digest,
            "selection": selection,
        }
        completed_records = _load_or_initialize(args.records_out, metadata)
        completed = {int(record["position_id"]) for record in completed_records}
        pending = [
            record for record in selection if int(record["position_id"]) not in completed
        ]

        def label(position):
            budget = stage_budgets[position["stage"]]
            result = pool.label_fen(
                position["fen"], budget, "v4-run13-component-score-v1", _score_value
            )
            record = dict(position)
            record.update({
                "type": "position",
                "label": {
                    "requested_nodes": budget,
                    "actual_nodes": int(result["actual_nodes"]),
                    "score": int(result["score"]),
                    "mapped_value": float(result["mapped_value"]),
                    "mate_band": bool(result["mate_band"]),
                    "completed_depth": int(result["completed_depth"]),
                    "cache_hit": bool(result["cache_hit"]),
                },
            })
            return record

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(label, position): position for position in pending}
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Run13 oracle labels"
            ):
                record = future.result()
                _append_record(args.records_out, record)
                completed_records.append(record)
    finally:
        pool.close()

    if len(completed_records) != len(selection):
        raise RuntimeError("Run13 component labeling stopped before completion.")
    print(json.dumps({
        "records": len(completed_records),
        "by_stage": {
            stage: sum(record["stage"] == stage for record in completed_records)
            for stage in STAGES
        },
        "records_out": os.path.abspath(args.records_out),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
