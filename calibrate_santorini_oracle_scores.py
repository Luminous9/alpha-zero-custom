"""Calibrate fixed-node oracle scores against deeper oracle continuations."""

import argparse
import json
import math
import os
from pathlib import Path
import time

import numpy as np
from tqdm import tqdm

from benchmark_santorini_oracle_budgets import (
    collect_unique_positions,
    file_sha256,
    select_stratified_positions,
)
from santorini.OracleResearch import ParallelOraclePool, STAGES


SCHEMA_VERSION = 1
NOMINAL_TEMPERATURE = 400.0
DEFAULT_REPLAY = "./temp/santorini_v3_run13_gumbel/latest.examples.npz"
DEFAULT_OUTPUT = "./temp/run13_oracle_score_calibration.json"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Fit score temperatures on independent fixed-node labels using deeper "
            "oracle-vs-oracle continuation outcomes."
        )
    )
    parser.add_argument("--replay", default=DEFAULT_REPLAY)
    parser.add_argument("--positions", type=int, default=300)
    parser.add_argument("--label-budgets", type=int, nargs="+", default=[20_000, 50_000])
    parser.add_argument("--adjudicator-nodes", type=int, default=250_000)
    parser.add_argument("--max-adjudication-plies", type=int, default=200)
    parser.add_argument("--fit-fraction", type=float, default=0.70)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--oracle-binary")
    parser.add_argument("--json-out", default=DEFAULT_OUTPUT)
    parser.add_argument("--records-out")
    parser.add_argument("--label-cache")
    return parser.parse_args()


def companion_path(path, suffix):
    path = Path(path)
    return str(path.with_suffix(suffix)) if path.suffix else str(path) + suffix


def nominal_score_value(score):
    score = int(score)
    if abs(score) >= 9_000:
        return float(np.sign(score))
    return float(2.0 / (1.0 + math.exp(-score / 400.0)) - 1.0)


def terminal_winner(fen):
    sections = str(fen).split("/")
    if len(sections) != 4:
        raise ValueError("Oracle continuation returned malformed FEN.")
    winners = [
        index
        for index, section in enumerate(sections[2:], start=1)
        if section.startswith("#")
    ]
    if len(winners) > 1:
        raise ValueError("Oracle continuation FEN declares multiple winners.")
    return winners[0] if winners else None


def adjudicate_from_fen(oracle, fen, nodes, max_plies):
    """Play a fresh deeper engine continuation and return the starter outcome."""
    starting_player = int(str(fen).split("/")[1])
    oracle.reset()
    current_fen = str(fen)
    trajectory = []
    for ply in range(int(max_plies)):
        winner = terminal_winner(current_fen)
        if winner is not None:
            return {
                "outcome": 1 if winner == starting_player else -1,
                "winner": winner,
                "plies": ply,
                "trajectory": trajectory,
            }
        response = oracle.analyze_fen(current_fen, nodes=int(nodes))
        best = response["best_move"]
        trajectory.append({
            "fen": current_fen,
            "next_fen": best["next_fen"],
            "score": int(best["score"]),
            "completed_depth": int(response["completed_depth"]),
            "nodes_visited": int(response["nodes_visited"]),
        })
        current_fen = best["next_fen"]
    raise RuntimeError(
        "Deeper adjudication exceeded {} plies from {}.".format(max_plies, fen)
    )


def assign_splits(selection, fit_fraction, seed):
    """Create deterministic stage-stratified fit/test assignments."""
    assigned = [dict(record) for record in selection]
    for stage_index, stage in enumerate(STAGES):
        indices = [index for index, record in enumerate(assigned) if record["stage"] == stage]
        rng = np.random.RandomState(int(seed) + stage_index)
        rng.shuffle(indices)
        if len(indices) <= 1:
            fit_count = len(indices)
        else:
            fit_count = min(len(indices) - 1, max(1, int(round(len(indices) * fit_fraction))))
        fit_indices = set(indices[:fit_count])
        for index in indices:
            assigned[index]["split"] = "fit" if index in fit_indices else "test"
    return assigned


def experiment_metadata(args, replay_digest, selection, engine_digest):
    return {
        "type": "metadata",
        "schema_version": SCHEMA_VERSION,
        "replay_path": os.path.abspath(args.replay),
        "replay_sha256": replay_digest,
        "positions": len(selection),
        "label_budgets": [int(value) for value in args.label_budgets],
        "adjudicator_nodes": int(args.adjudicator_nodes),
        "max_adjudication_plies": int(args.max_adjudication_plies),
        "fit_fraction": float(args.fit_fraction),
        "seed": int(args.seed),
        "engine_digest": engine_digest,
        "selection": selection,
    }


def _metadata_identity(metadata):
    return {
        key: metadata[key]
        for key in (
            "schema_version",
            "replay_sha256",
            "positions",
            "label_budgets",
            "adjudicator_nodes",
            "max_adjudication_plies",
            "fit_fraction",
            "seed",
            "engine_digest",
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
        lines = [json.loads(line) for line in source if line.strip()]
    if not lines or _metadata_identity(lines[0]) != _metadata_identity(metadata):
        raise ValueError("Existing calibration records do not match this experiment.")
    records = lines[1:]
    ids = [int(record["position_id"]) for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Calibration records contain duplicate position ids.")
    return records


def append_record(path, record):
    with open(path, "a") as output:
        output.write(json.dumps(record, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())


def score_probability(score, temperature):
    score = int(score)
    if score >= 9_000:
        return 1.0
    if score <= -9_000:
        return 0.0
    scaled = float(np.clip(score / float(temperature), -50.0, 50.0))
    return float(1.0 / (1.0 + math.exp(-scaled)))


def _log_loss(scores, outcomes, temperature):
    probabilities = np.asarray(
        [score_probability(score, temperature) for score in scores], dtype=np.float64
    )
    targets = (np.asarray(outcomes, dtype=np.float64) + 1.0) / 2.0
    probabilities = np.clip(probabilities, 1e-12, 1.0 - 1e-12)
    return float(
        -np.mean(
            targets * np.log(probabilities)
            + (1 - targets) * np.log(1 - probabilities)
        )
    )


def fit_temperature(scores, outcomes):
    if not scores:
        raise ValueError("Cannot fit a temperature without fit records.")
    # Golden-section search in log-temperature space keeps T positive and avoids
    # adding a scipy dependency to the measurement path.
    left, right = math.log(10.0), math.log(10_000.0)
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    x1 = right - ratio * (right - left)
    x2 = left + ratio * (right - left)
    f1 = _log_loss(scores, outcomes, math.exp(x1))
    f2 = _log_loss(scores, outcomes, math.exp(x2))
    for _ in range(80):
        if f1 <= f2:
            right, x2, f2 = x2, x1, f1
            x1 = right - ratio * (right - left)
            f1 = _log_loss(scores, outcomes, math.exp(x1))
        else:
            left, x1, f1 = x1, x2, f2
            x2 = left + ratio * (right - left)
            f2 = _log_loss(scores, outcomes, math.exp(x2))
    return float(math.exp((left + right) / 2.0))


def score_magnitude_bucket(score):
    magnitude = abs(int(score))
    if magnitude >= 9_000:
        return "mate"
    if magnitude < 100:
        return "0_99"
    if magnitude < 400:
        return "100_399"
    if magnitude < 1_000:
        return "400_999"
    return "1000_8999"


def calibration_metrics(records, budget, temperature):
    if not records:
        return {
            "positions": 0,
            "brier_score": None,
            "log_loss": None,
            "expected_calibration_error": None,
            "score_sign_accuracy": None,
        }
    scores = np.asarray([record["labels"][str(budget)]["score"] for record in records])
    outcomes = np.asarray([record["adjudication"]["outcome"] for record in records])
    probabilities = np.asarray(
        [score_probability(score, temperature) for score in scores], dtype=np.float64
    )
    targets = (outcomes.astype(np.float64) + 1.0) / 2.0
    bins = np.minimum((probabilities * 10).astype(int), 9)
    calibration_error = 0.0
    for bin_index in range(10):
        mask = bins == bin_index
        if np.any(mask):
            calibration_error += float(np.mean(mask)) * abs(
                float(np.mean(probabilities[mask])) - float(np.mean(targets[mask]))
            )
    return {
        "positions": len(records),
        "brier_score": float(np.mean((probabilities - targets) ** 2)),
        "log_loss": _log_loss(scores, outcomes, temperature),
        "expected_calibration_error": float(calibration_error),
        "score_sign_accuracy": float(np.mean(np.sign(scores) == outcomes)),
        "mean_predicted_win_probability": float(np.mean(probabilities)),
        "observed_win_rate": float(np.mean(targets)),
    }


def summarize(records, budgets):
    result = {"budgets": {}}
    fit_records = [record for record in records if record["split"] == "fit"]
    test_records = [record for record in records if record["split"] == "test"]
    for budget in budgets:
        budget = int(budget)
        fit_scores = [record["labels"][str(budget)]["score"] for record in fit_records]
        fit_outcomes = [record["adjudication"]["outcome"] for record in fit_records]
        temperature = fit_temperature(fit_scores, fit_outcomes)
        budget_summary = {
            "temperature": temperature,
            "nominal_temperature": NOMINAL_TEMPERATURE,
            "fit": calibration_metrics(fit_records, budget, temperature),
            "test": calibration_metrics(test_records, budget, temperature),
            "nominal_fit": calibration_metrics(
                fit_records, budget, NOMINAL_TEMPERATURE
            ),
            "nominal_test": calibration_metrics(
                test_records, budget, NOMINAL_TEMPERATURE
            ),
            "test_by_stage": {},
            "test_by_score_magnitude": {},
        }
        for stage in STAGES:
            budget_summary["test_by_stage"][stage] = calibration_metrics(
                [record for record in test_records if record["stage"] == stage],
                budget,
                temperature,
            )
        for bucket in ("0_99", "100_399", "400_999", "1000_8999", "mate"):
            budget_summary["test_by_score_magnitude"][bucket] = calibration_metrics(
                [
                    record
                    for record in test_records
                    if score_magnitude_bucket(record["labels"][str(budget)]["score"])
                    == bucket
                ],
                budget,
                temperature,
            )
        result["budgets"][str(budget)] = budget_summary
    result["fit_positions"] = len(fit_records)
    result["test_positions"] = len(test_records)
    return result


def write_json_atomic(path, payload):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w") as output:
        json.dump(payload, output, indent=2, sort_keys=True)
    os.replace(temporary, path)


def validate_args(args):
    budgets = [int(value) for value in args.label_budgets]
    if args.positions < 2 or any(value < 1 for value in budgets):
        raise ValueError("Positions and node budgets must be positive.")
    if budgets != sorted(set(budgets)):
        raise ValueError("Label budgets must be unique and increasing.")
    if args.adjudicator_nodes < 4 * budgets[-1]:
        raise ValueError(
            "Adjudicator budget must be at least four times the largest label budget."
        )
    if args.max_adjudication_plies < 1:
        raise ValueError("Maximum adjudication plies must be positive.")
    if not 0 < args.fit_fraction < 1:
        raise ValueError("Fit fraction must be strictly between zero and one.")


def main():
    args = parse_args()
    validate_args(args)
    if not os.path.isfile(args.replay):
        raise FileNotFoundError("Replay file not found: {}".format(args.replay))
    records_path = args.records_out or companion_path(args.json_out, ".records.jsonl")
    cache_path = args.label_cache or companion_path(args.json_out, ".labels.sqlite3")

    replay_digest = file_sha256(args.replay)
    selection = select_stratified_positions(
        collect_unique_positions(args.replay), args.positions, args.seed
    )
    selection = assign_splits(selection, args.fit_fraction, args.seed)
    pool = ParallelOraclePool(args.oracle_binary, cache_path=cache_path)
    try:
        engine_digest = pool.engine_digest
        metadata = experiment_metadata(args, replay_digest, selection, engine_digest)
        records = load_or_initialize_records(records_path, metadata)
        completed = {int(record["position_id"]) for record in records}
        pending = [record for record in selection if int(record["position_id"]) not in completed]
        oracle_info = dict(pool.oracle().info)
        for position in tqdm(pending, desc="Oracle score calibration"):
            labels = {}
            for budget in args.label_budgets:
                label = pool.label_fen(
                    position["fen"],
                    int(budget),
                    "raw-score-v1",
                    nominal_score_value,
                )
                labels[str(int(budget))] = {
                    "score": int(label["score"]),
                    "mate_band": bool(label["mate_band"]),
                    "completed_depth": int(label["completed_depth"]),
                    "actual_nodes": int(label["actual_nodes"]),
                    "cache_hit": bool(label["cache_hit"]),
                }
            # This reset occurs after all label queries, so the adjudication's
            # first move cannot reuse any label search or its TT state.
            adjudication = adjudicate_from_fen(
                pool.oracle(),
                position["fen"],
                args.adjudicator_nodes,
                args.max_adjudication_plies,
            )
            record = dict(position)
            record.update({"type": "position", "labels": labels, "adjudication": adjudication})
            append_record(records_path, record)
            records.append(record)
    finally:
        pool.close()

    records.sort(key=lambda record: int(record["position_id"]))
    if len(records) != len(selection):
        raise RuntimeError("Calibration stopped before every selected position completed.")
    output = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_unix": time.time(),
        "replay_path": os.path.abspath(args.replay),
        "replay_sha256": replay_digest,
        "records_path": os.path.abspath(records_path),
        "label_cache": os.path.abspath(cache_path),
        "positions": len(records),
        "label_budgets": [int(value) for value in args.label_budgets],
        "adjudicator_nodes": int(args.adjudicator_nodes),
        "adjudicator_is_materially_deeper": (
            args.adjudicator_nodes >= 4 * max(args.label_budgets)
        ),
        "independent_label_searches": True,
        "adjudication_reset_after_labels": True,
        "oracle": oracle_info,
        "engine_digest": engine_digest,
        "summary": summarize(records, args.label_budgets),
    }
    write_json_atomic(args.json_out, output)
    print(json.dumps(output["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
