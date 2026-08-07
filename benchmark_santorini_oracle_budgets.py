import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from tqdm import tqdm

from santorini.OracleResearch import STAGES, canonical_d4_fen, stage_for_builds
from santorini.SantoriniOracle import SantoriniOracleProcess


SCHEMA_VERSION = 1
DEFAULT_REPLAY = "./temp/santorini_v3_run13_gumbel/latest.examples.npz"
DEFAULT_OUTPUT = "./temp/run13_oracle_budget_stability.json"
DEFAULT_BUDGETS = [20_000, 50_000, 100_000, 250_000]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Measure santorini-ai best-move stability across independent fixed-node "
            "searches on stratified replay positions."
        )
    )
    parser.add_argument("--replay", default=DEFAULT_REPLAY)
    parser.add_argument("--positions", type=int, default=500)
    parser.add_argument("--budgets", type=int, nargs="+", default=DEFAULT_BUDGETS)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--oracle-binary")
    parser.add_argument("--json-out", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--records-out",
        help="Append-only resume file; defaults beside --json-out with .records.jsonl.",
    )
    return parser.parse_args()


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        while True:
            chunk = source.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def collect_unique_positions(replay_path):
    by_stage = {stage: {} for stage in STAGES}
    with np.load(replay_path, allow_pickle=False) as payload:
        boards = payload["boards"]
        values = payload["values"]
        for replay_index, (board, value) in enumerate(zip(boards, values)):
            if int(np.count_nonzero(board[0])) != 4:
                continue
            build_count = int(np.sum(board[1]))
            stage = stage_for_builds(build_count)
            fen = canonical_d4_fen(board.astype(int))
            existing = by_stage[stage].get(fen)
            if existing is None:
                by_stage[stage][fen] = {
                    "fen": fen,
                    "stage": stage,
                    "build_count": build_count,
                    "replay_index": int(replay_index),
                    "replay_observations": 1,
                    "replay_value_sum": float(value),
                }
            else:
                existing["replay_observations"] += 1
                existing["replay_value_sum"] += float(value)

    result = {}
    for stage in STAGES:
        records = []
        for record in by_stage[stage].values():
            record["replay_value_mean"] = (
                record.pop("replay_value_sum") / record["replay_observations"]
            )
            records.append(record)
        result[stage] = sorted(records, key=lambda item: item["fen"])
    return result


def _stage_quotas(total):
    base, remainder = divmod(int(total), len(STAGES))
    return {
        stage: base + (1 if index < remainder else 0)
        for index, stage in enumerate(STAGES)
    }


def select_stratified_positions(unique_by_stage, count, seed):
    count = int(count)
    available_total = sum(len(unique_by_stage[stage]) for stage in STAGES)
    if count < 1:
        raise ValueError("--positions must be positive.")
    if count > available_total:
        raise ValueError(
            "Requested {} positions, but only {} D4-unique standard positions exist.".format(
                count, available_total
            )
        )

    quotas = _stage_quotas(count)
    selected = []
    shortfall = 0
    for stage in STAGES:
        quota = min(quotas[stage], len(unique_by_stage[stage]))
        quotas[stage] = quota
        shortfall += _stage_quotas(count)[stage] - quota
    while shortfall:
        progressed = False
        for stage in STAGES:
            if quotas[stage] < len(unique_by_stage[stage]):
                quotas[stage] += 1
                shortfall -= 1
                progressed = True
                if not shortfall:
                    break
        if not progressed:
            raise RuntimeError("Could not redistribute the stage sampling shortfall.")

    rng = np.random.RandomState(seed)
    for stage in STAGES:
        candidates = unique_by_stage[stage]
        indices = rng.choice(len(candidates), size=quotas[stage], replace=False)
        selected.extend(dict(candidates[int(index)]) for index in indices)
    rng.shuffle(selected)
    for position_id, record in enumerate(selected):
        record["position_id"] = int(position_id)
    return selected


def default_records_path(json_out):
    output = Path(json_out)
    if output.suffix:
        return str(output.with_suffix(".records.jsonl"))
    return str(output) + ".records.jsonl"


def experiment_metadata(args, replay_path, replay_digest, selection):
    return {
        "type": "metadata",
        "schema_version": SCHEMA_VERSION,
        "replay_path": os.path.abspath(replay_path),
        "replay_sha256": replay_digest,
        "budgets": [int(budget) for budget in args.budgets],
        "positions": len(selection),
        "seed": int(args.seed),
        "selection": selection,
    }


def _metadata_identity(metadata):
    return {
        key: metadata[key]
        for key in (
            "schema_version",
            "replay_path",
            "replay_sha256",
            "budgets",
            "positions",
            "seed",
            "selection",
        )
    }


def load_or_initialize_records(path, metadata):
    if not os.path.exists(path):
        output_dir = os.path.dirname(os.path.abspath(path))
        os.makedirs(output_dir, exist_ok=True)
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
            "Existing records metadata does not match this experiment. "
            "Choose a different --records-out/--json-out path."
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


def score_sign(score):
    return int(score > 0) - int(score < 0)


def _mean(values):
    return float(np.mean(values)) if values else None


def _distribution(values):
    if not values:
        return {"mean": None, "median": None, "p10": None, "p90": None}
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p10": float(np.percentile(values, 10)),
        "p90": float(np.percentile(values, 90)),
    }


def summarize_records(records, budgets):
    budget_keys = [str(int(budget)) for budget in budgets]
    deepest = budget_keys[-1]

    def summarize_group(group):
        result = {
            "positions": len(group),
            "all_budget_move_stability_rate": _mean([
                len({record["analyses"][budget]["next_fen"] for budget in budget_keys}) == 1
                for record in group
            ]),
            "mean_unique_best_moves": _mean([
                len({record["analyses"][budget]["next_fen"] for budget in budget_keys})
                for record in group
            ]),
            "agreement_with_deepest": {},
            "score_sign_agreement_with_deepest": {},
            "consecutive_move_agreement": {},
            "budgets": {},
        }
        for budget in budget_keys:
            result["agreement_with_deepest"][budget] = _mean([
                record["analyses"][budget]["next_fen"]
                == record["analyses"][deepest]["next_fen"]
                for record in group
            ])
            result["score_sign_agreement_with_deepest"][budget] = _mean([
                score_sign(record["analyses"][budget]["score"])
                == score_sign(record["analyses"][deepest]["score"])
                for record in group
            ])
            elapsed = [record["analyses"][budget]["elapsed_seconds"] for record in group]
            actual_nodes = [record["analyses"][budget]["nodes_visited"] for record in group]
            total_seconds = float(sum(elapsed))
            result["budgets"][budget] = {
                "completed_depth": _distribution([
                    record["analyses"][budget]["completed_depth"] for record in group
                ]),
                "nodes_visited": _distribution(actual_nodes),
                "elapsed_seconds": _distribution(elapsed),
                "total_elapsed_seconds": total_seconds,
                "nodes_per_second": (
                    float(sum(actual_nodes)) / total_seconds if total_seconds > 0 else None
                ),
                "forced_score_rate": _mean([
                    abs(record["analyses"][budget]["score"]) >= 9000 for record in group
                ]),
            }
        for previous, current in zip(budget_keys, budget_keys[1:]):
            result["consecutive_move_agreement"]["{}->{}".format(previous, current)] = _mean([
                record["analyses"][previous]["next_fen"]
                == record["analyses"][current]["next_fen"]
                for record in group
            ])
        return result

    summary = {"all": summarize_group(records), "by_stage": {}}
    for stage in STAGES:
        summary["by_stage"][stage] = summarize_group([
            record for record in records if record["stage"] == stage
        ])
    return summary


def analyze_position(oracle, position, budgets):
    analyses = {}
    for budget in budgets:
        # Independent searches are essential for a fair budget comparison.
        query_started = time.perf_counter()
        oracle.reset()
        search_started = time.perf_counter()
        response = oracle.analyze_fen(position["fen"], nodes=budget)
        finished = time.perf_counter()
        best = response["best_move"]
        analyses[str(int(budget))] = {
            "action": best["action"],
            "actions": best["actions"],
            "next_fen": best["next_fen"],
            "score": int(best["score"]),
            "best_move_depth": int(best["depth"]),
            "completed_depth": int(response["completed_depth"]),
            "nodes_visited": int(response["nodes_visited"]),
            "requested_nodes": int(response["requested_nodes"]),
            "elapsed_seconds": float(finished - query_started),
            "search_elapsed_seconds": float(finished - search_started),
            "reset_elapsed_seconds": float(search_started - query_started),
        }
    record = dict(position)
    record["type"] = "position"
    record["analyses"] = analyses
    return record


def main():
    args = parse_args()
    budgets = [int(budget) for budget in args.budgets]
    if any(budget < 1 for budget in budgets):
        raise ValueError("Every --budgets value must be positive.")
    if budgets != sorted(set(budgets)):
        raise ValueError("--budgets must be unique and listed in increasing order.")
    if not os.path.isfile(args.replay):
        raise FileNotFoundError("Replay file not found: {}".format(args.replay))

    print("Hashing replay...")
    replay_digest = file_sha256(args.replay)
    print("Collecting D4-unique standard positions...")
    unique_by_stage = collect_unique_positions(args.replay)
    selection = select_stratified_positions(unique_by_stage, args.positions, args.seed)
    metadata = experiment_metadata(args, args.replay, replay_digest, selection)
    records_path = args.records_out or default_records_path(args.json_out)
    records = load_or_initialize_records(records_path, metadata)
    completed = {int(record["position_id"]) for record in records}
    pending = [position for position in selection if position["position_id"] not in completed]

    print(
        "Selected {} positions (early={}, middle={}, late={}); resuming with {} complete.".format(
            len(selection),
            sum(position["stage"] == "early" for position in selection),
            sum(position["stage"] == "middle" for position in selection),
            sum(position["stage"] == "late" for position in selection),
            len(records),
        )
    )

    with SantoriniOracleProcess(args.oracle_binary) as oracle:
        oracle_info = dict(oracle.info)
        if pending:
            for position in tqdm(pending, desc="Oracle budget sweep"):
                record = analyze_position(oracle, position, budgets)
                append_record(records_path, record)
                records.append(record)

    records.sort(key=lambda record: int(record["position_id"]))
    if len(records) != len(selection):
        raise RuntimeError("Experiment ended without all selected positions.")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_unix": time.time(),
        "replay_path": os.path.abspath(args.replay),
        "replay_sha256": replay_digest,
        "records_path": os.path.abspath(records_path),
        "budgets": budgets,
        "positions": len(records),
        "seed": int(args.seed),
        "oracle": oracle_info,
        "independent_searches": True,
        "transposition_table_reset_before_each_query": True,
        "summary": summarize_records(records, budgets),
    }
    output_dir = os.path.dirname(os.path.abspath(args.json_out))
    os.makedirs(output_dir, exist_ok=True)
    temp_output = args.json_out + ".tmp"
    with open(temp_output, "w") as output:
        json.dump(summary, output, indent=2, sort_keys=True)
    os.replace(temp_output, args.json_out)
    print(json.dumps(summary["summary"]["all"], indent=2, sort_keys=True))
    print("Summary: {}".format(os.path.abspath(args.json_out)))
    print("Detailed records: {}".format(os.path.abspath(records_path)))


if __name__ == "__main__":
    main()
