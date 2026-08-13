"""Measure how much P2 replay targets teach selected V4 checkpoints.

The compact replay stores search targets but not a checkpoint identity per
window.  Callers can therefore declare the few windows whose generating
checkpoint is actually available; all other checkpoint/window comparisons are
reported as fixed-reference diagnostics rather than reconstructed online KL.
"""

import argparse
from collections import defaultdict
import json
import os
import sqlite3
import time

import numpy as np
import torch

from santorini.OracleResearch import canonical_d4_fen, stage_for_builds
from santorini.SantoriniGame import SantoriniGame
from santorini.pytorch.V4NNet import V4InferenceWrapper


EPSILON = 1e-30


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", required=True, help="Compact replay NPZ/ZIP path.")
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="Checkpoint reference; repeat for multiple fixed references.",
    )
    parser.add_argument(
        "--exact-generator",
        action="append",
        default=[],
        metavar="WINDOW=NAME",
        help="Declare an exact generating checkpoint for a one-based replay window.",
    )
    parser.add_argument(
        "--oracle-cache",
        action="append",
        default=[],
        help="Existing OracleLabelCache SQLite file; repeat to merge caches.",
    )
    parser.add_argument(
        "--oracle-component",
        action="append",
        default=[],
        help="Companion corpus NPZ carrying split_ids; required with oracle caches.",
    )
    parser.add_argument(
        "--oracle-allowed-split",
        action="append",
        type=int,
        choices=(0, 1),
        help="Allowed oracle corpus split ID; defaults to train (0) and selection (1).",
    )
    parser.add_argument(
        "--oracle-min-nodes",
        type=int,
        default=250_000,
        help="Minimum node budget for the deep-oracle comparison.",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--torch-threads", type=int, default=0)
    parser.add_argument(
        "--max-positions-per-window",
        type=int,
        default=0,
        help="Deterministically subsample each window; zero analyzes every position.",
    )
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--no-canonical-d4",
        action="store_false",
        dest="canonical_d4",
        help="Disable the production canonical wrapper (normally incorrect for V4 ordinary).",
    )
    parser.set_defaults(canonical_d4=True)
    return parser.parse_args()


def _named_values(entries, value_name):
    parsed = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError("Expected NAME={} syntax, received {!r}.".format(value_name, entry))
        name, value = entry.split("=", 1)
        name = name.strip()
        value = value.strip()
        if not name or not value or name in parsed:
            raise ValueError("Invalid or duplicate named argument {!r}.".format(entry))
        parsed[name] = value
    return parsed


def _exact_generators(entries, checkpoint_names):
    parsed = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError("Expected WINDOW=NAME syntax, received {!r}.".format(entry))
        window_text, name = entry.split("=", 1)
        window = int(window_text)
        name = name.strip()
        if window < 1 or window in parsed:
            raise ValueError("Invalid or duplicate exact-generator window {!r}.".format(entry))
        if name not in checkpoint_names:
            raise ValueError("Exact generator {!r} is not a declared checkpoint.".format(name))
        parsed[window] = name
    return parsed


def _atomic_json(path, payload):
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w") as output:
        json.dump(payload, output, indent=2, sort_keys=True)
    os.replace(temporary, path)


def _load_replay(path):
    with np.load(path, allow_pickle=False) as payload:
        required = {
            "format_version", "action_size", "history_lengths", "boards", "values",
            "policy_offsets", "policy_indices", "policy_values",
        }
        missing = required.difference(payload.files)
        if missing:
            raise ValueError("Compact replay is missing fields: {}".format(sorted(missing)))
        lengths = payload["history_lengths"].astype(np.int64)
        if int(lengths.sum()) != len(payload["boards"]):
            raise ValueError("Replay history lengths do not match its board count.")
        metadata = (
            payload["example_metadata"].astype(str)
            if "example_metadata" in payload.files
            else np.asarray(['{"source":"unknown"}'] * len(payload["boards"]))
        )
        return {
            "format_version": int(payload["format_version"][0]),
            "action_size": int(payload["action_size"][0]),
            "history_lengths": lengths,
            "boards": payload["boards"].astype(np.int8),
            "values": payload["values"].astype(np.float64),
            "policy_offsets": payload["policy_offsets"].astype(np.int64),
            "policy_indices": payload["policy_indices"].astype(np.int64),
            "policy_values": payload["policy_values"].astype(np.float64),
            "metadata": metadata,
        }


def _selected_indices(lengths, maximum, seed):
    rng = np.random.RandomState(seed)
    selections = []
    start = 0
    for window_index, length in enumerate(lengths, start=1):
        local = np.arange(int(length), dtype=np.int64)
        if maximum and len(local) > maximum:
            local = np.sort(rng.choice(local, size=maximum, replace=False))
        selections.append(start + local)
        start += int(length)
    return selections


def _dense_targets(replay, indices):
    targets = np.zeros((len(indices), replay["action_size"]), dtype=np.float64)
    for row, example_index in enumerate(indices):
        start = int(replay["policy_offsets"][example_index])
        end = int(replay["policy_offsets"][example_index + 1])
        targets[row, replay["policy_indices"][start:end]] = replay["policy_values"][start:end]
    totals = targets.sum(axis=1)
    if np.any(~np.isfinite(totals)) or np.any(totals <= 0):
        raise ValueError("Replay contains an invalid search target.")
    return targets / totals[:, None]


def _stages(boards):
    labels = []
    for board in boards:
        if int(np.count_nonzero(board[0])) < 4:
            labels.append("placement")
        else:
            labels.append(stage_for_builds(int(np.sum(board[1]))))
    return np.asarray(labels)


def _sources(metadata):
    result = []
    for encoded in metadata:
        try:
            result.append(str(json.loads(str(encoded)).get("source", "unknown")))
        except (TypeError, ValueError, json.JSONDecodeError):
            result.append("invalid_metadata")
    return np.asarray(result)


def _safe_correlation(first, second):
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if len(first) < 2 or np.std(first) == 0 or np.std(second) == 0:
        return None
    return float(np.corrcoef(first, second)[0, 1])


def _distribution_summary(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return None
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p10": float(np.quantile(values, 0.10)),
        "p25": float(np.quantile(values, 0.25)),
        "p75": float(np.quantile(values, 0.75)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
    }


def _metric_arrays(game, boards, targets, priors, predictions, outcomes):
    valids = np.stack([game.getValidMoves(board.astype(int), 1) for board in boards])
    illegal_target_mass = np.sum(targets * (1 - valids), axis=1)
    legal_priors = np.asarray(priors, dtype=np.float64) * valids
    prior_mass = legal_priors.sum(axis=1)
    if np.any(~np.isfinite(prior_mass)) or np.any(prior_mass <= 0):
        raise ValueError("Checkpoint assigned no finite mass to a legal action.")
    legal_priors /= prior_mass[:, None]

    target_entropy = -np.sum(targets * np.log(np.maximum(targets, EPSILON)), axis=1)
    kl = np.sum(
        targets * (
            np.log(np.maximum(targets, EPSILON))
            - np.log(np.maximum(legal_priors, EPSILON))
        ),
        axis=1,
    )
    total_variation = 0.5 * np.sum(np.abs(targets - legal_priors), axis=1)
    target_max = np.max(targets, axis=1, keepdims=True)
    prior_argmax = np.argmax(legal_priors, axis=1)
    target_argmax = np.argmax(targets, axis=1)
    row_indices = np.arange(len(boards))
    target_top_set = targets >= (target_max - 1e-10)
    target_support = targets > 0
    return {
        "kl_target_prior": kl,
        "total_variation": total_variation,
        "target_entropy": target_entropy,
        "target_support_size": np.sum(target_support, axis=1).astype(np.float64),
        "target_support_prior_mass": np.sum(legal_priors * target_support, axis=1),
        "argmax_agreement": (prior_argmax == target_argmax).astype(np.float64),
        "prior_argmax_in_target_top_set": target_top_set[row_indices, prior_argmax].astype(np.float64),
        "illegal_target_mass": illegal_target_mass,
        "value_squared_error_z": (np.asarray(predictions) - outcomes) ** 2,
        "value_absolute_error_z": np.abs(np.asarray(predictions) - outcomes),
        "value_sign_accuracy_z": (np.sign(predictions) == np.sign(outcomes)).astype(np.float64),
        "value_prediction": np.asarray(predictions, dtype=np.float64),
        "outcome_z": np.asarray(outcomes, dtype=np.float64),
    }


def _summarize_metrics(metrics, mask=None):
    if mask is None:
        mask = np.ones(len(metrics["kl_target_prior"]), dtype=bool)
    if not np.any(mask):
        return None
    result = {
        key: _distribution_summary(np.asarray(values)[mask])
        for key, values in metrics.items()
        if key not in ("value_prediction", "outcome_z")
    }
    predictions = metrics["value_prediction"][mask]
    outcomes = metrics["outcome_z"][mask]
    result["value_prediction"] = _distribution_summary(predictions)
    result["outcome_z"] = _distribution_summary(outcomes)
    result["value_outcome_correlation"] = _safe_correlation(predictions, outcomes)
    return result


def _load_oracle_splits(component_paths):
    splits = {}
    records = []
    for path in component_paths:
        path = os.path.abspath(path)
        if not os.path.isfile(path):
            raise FileNotFoundError("Oracle component not found: {}".format(path))
        with np.load(path, allow_pickle=False) as payload:
            if "boards" not in payload.files or "split_ids" not in payload.files:
                raise ValueError("Oracle component lacks boards/split_ids: {}".format(path))
            if len(payload["boards"]) != len(payload["split_ids"]):
                raise ValueError("Oracle component boards/splits have different lengths.")
            for board, split_id in zip(payload["boards"], payload["split_ids"]):
                fen = canonical_d4_fen(board.astype(int))
                split_id = int(split_id)
                previous = splits.get(fen)
                if previous is not None and previous != split_id:
                    raise ValueError("A D4 position has conflicting corpus split IDs.")
                splits[fen] = split_id
            records.append({"path": path, "positions": len(payload["boards"])})
    return splits, records


def _load_oracle_labels(paths, minimum_nodes, split_by_fen, allowed_splits):
    labels = {}
    cache_records = []
    excluded_by_split = defaultdict(int)
    for path in paths:
        path = os.path.abspath(path)
        if not os.path.isfile(path):
            raise FileNotFoundError("Oracle cache not found: {}".format(path))
        connection = sqlite3.connect(path)
        rows = connection.execute(
            """
            SELECT d4_fen, node_budget, engine_digest, calibration_version, mapped_value
            FROM oracle_labels WHERE node_budget >= ?
            """,
            (int(minimum_nodes),),
        ).fetchall()
        connection.close()
        cache_records.append({"path": path, "eligible_rows": len(rows)})
        for fen, nodes, digest, calibration, value in rows:
            split_id = split_by_fen.get(str(fen))
            if split_id not in allowed_splits:
                excluded_by_split[str(split_id)] += 1
                continue
            candidate = {
                "nodes": int(nodes),
                "engine_digest": str(digest),
                "calibration_version": str(calibration),
                "value": float(value),
                "split_id": int(split_id),
            }
            current = labels.get(str(fen))
            if current is None or candidate["nodes"] > current["nodes"]:
                labels[str(fen)] = candidate
    return labels, cache_records, dict(sorted(excluded_by_split.items()))


def _oracle_comparison(boards, outcomes, predictions, labels):
    oracle = []
    selected_outcomes = []
    selected_predictions = []
    nodes = []
    keys = []
    for board, outcome, prediction in zip(boards, outcomes, predictions):
        if int(np.count_nonzero(board[0])) != 4:
            continue
        key = canonical_d4_fen(board.astype(int))
        label = labels.get(key)
        if label is None:
            continue
        keys.append(key)
        oracle.append(label["value"])
        nodes.append(label["nodes"])
        selected_outcomes.append(float(outcome))
        selected_predictions.append(float(prediction))
    if not oracle:
        return {
            "matched_observations": 0,
            "unique_matched_positions": 0,
        }
    oracle = np.asarray(oracle, dtype=np.float64)
    selected_outcomes = np.asarray(selected_outcomes, dtype=np.float64)
    selected_predictions = np.asarray(selected_predictions, dtype=np.float64)
    seen_keys = set()
    unique_indices = []
    for index, key in enumerate(keys):
        if key not in seen_keys:
            unique_indices.append(index)
            seen_keys.add(key)
    unique_indices = np.asarray(unique_indices, dtype=np.int64)

    def comparison(mask):
        reference = oracle[mask]
        net = selected_predictions[mask]
        z = selected_outcomes[mask]
        net_error = np.abs(net - reference)
        z_error = np.abs(z - reference)
        return {
            "count": int(len(reference)),
            "net_mse_to_oracle": float(np.mean((net - reference) ** 2)),
            "z_mse_to_oracle": float(np.mean((z - reference) ** 2)),
            "net_mae_to_oracle": float(np.mean(net_error)),
            "z_mae_to_oracle": float(np.mean(z_error)),
            "net_closer_than_z_fraction": float(np.mean(net_error < z_error)),
            "net_oracle_correlation": _safe_correlation(net, reference),
            "z_oracle_correlation": _safe_correlation(z, reference),
        }

    return {
        "matched_observations": int(len(oracle)),
        "unique_matched_positions": int(len(unique_indices)),
        "node_budgets": sorted(set(int(value) for value in nodes)),
        "observation_weighted": comparison(np.arange(len(oracle))),
        "unique_position_weighted": comparison(unique_indices),
    }


def _repeated_outcome_diagnostic(boards, outcomes, restricted_keys=None):
    by_position = defaultdict(list)
    eligible_observations = 0
    for board, outcome in zip(boards, outcomes):
        if int(np.count_nonzero(board[0])) != 4:
            continue
        key = canonical_d4_fen(board.astype(int))
        if restricted_keys is not None and key not in restricted_keys:
            continue
        eligible_observations += 1
        by_position[key].append(float(outcome))
    repeated = [values for values in by_position.values() if len(values) >= 2]
    if not repeated:
        return {
            "eligible_observations": eligible_observations,
            "unique_positions": len(by_position),
            "repeated_unique_positions": 0,
        }
    variances = np.asarray([
        np.mean((np.asarray(values) - np.mean(values)) ** 2)
        for values in repeated
    ])
    weights = np.asarray([len(values) for values in repeated], dtype=np.float64)
    conflicting = sum(len(set(values)) > 1 for values in repeated)
    return {
        "eligible_observations": eligible_observations,
        "unique_positions": len(by_position),
        "repeated_unique_positions": len(repeated),
        "repeated_observations": int(weights.sum()),
        "conflicting_repeated_positions": int(conflicting),
        "conflicting_repeated_position_fraction": float(conflicting / len(repeated)),
        "unique_position_weighted_within_position_z_variance": float(np.mean(variances)),
        "observation_weighted_within_position_z_variance": float(
            np.average(variances, weights=weights)
        ),
        "interpretation": (
            "This is an in-archive repeat estimate, not an unbiased irreducible-noise "
            "estimate; each exact standard state can only occur in different games."
        ),
    }


def _merge_metric_arrays(collection):
    return {
        key: np.concatenate([item[key] for item in collection])
        for key in collection[0]
    }


def main():
    args = parse_args()
    if args.batch_size < 1 or args.max_positions_per_window < 0:
        raise ValueError("Batch size must be positive and sampling limit nonnegative.")
    if args.torch_threads < 0:
        raise ValueError("Torch thread count cannot be negative.")
    if args.torch_threads:
        torch.set_num_threads(args.torch_threads)

    checkpoints = _named_values(args.checkpoint, "PATH")
    exact_generators = _exact_generators(args.exact_generator, checkpoints)
    replay = _load_replay(args.replay)
    selected_by_window = _selected_indices(
        replay["history_lengths"], args.max_positions_per_window, args.seed
    )
    if exact_generators and max(exact_generators) > len(selected_by_window):
        raise ValueError("An exact-generator declaration exceeds the replay window count.")

    game = SantoriniGame(5, sequential_placement=True)
    if replay["action_size"] != game.getActionSize():
        raise ValueError("Replay action size does not match sequential-placement Santorini.")
    if bool(args.oracle_cache) != bool(args.oracle_component):
        raise ValueError(
            "Oracle caches and companion components must be supplied together so "
            "final-test and unclassified labels can be excluded."
        )
    allowed_oracle_splits = set(
        args.oracle_allowed_split
        if args.oracle_allowed_split is not None else (0, 1)
    )
    oracle_splits, oracle_component_records = _load_oracle_splits(
        args.oracle_component
    )
    (
        oracle_labels,
        oracle_cache_records,
        oracle_excluded_by_split,
    ) = _load_oracle_labels(
        args.oracle_cache,
        args.oracle_min_nodes,
        oracle_splits,
        allowed_oracle_splits,
    )
    started = time.perf_counter()
    all_selected_indices = np.concatenate(selected_by_window)
    selected_boards = replay["boards"][all_selected_indices]
    selected_outcomes = replay["values"][all_selected_indices]
    report = {
        "schema_version": 1,
        "type": "santorini_v4_replay_teaching_signal",
        "replay": os.path.abspath(args.replay),
        "replay_format_version": replay["format_version"],
        "history_lengths": replay["history_lengths"].tolist(),
        "selected_positions_per_window": [len(value) for value in selected_by_window],
        "canonical_d4": bool(args.canonical_d4),
        "exact_generators": {str(key): value for key, value in exact_generators.items()},
        "oracle": {
            "minimum_nodes": int(args.oracle_min_nodes),
            "allowed_split_ids": sorted(allowed_oracle_splits),
            "final_test_touched": False,
            "cache_records": oracle_cache_records,
            "component_records": oracle_component_records,
            "excluded_label_rows_by_split": oracle_excluded_by_split,
            "eligible_unique_labels": len(oracle_labels),
        },
        "interpretation": [
            "KL(target||prior) uses the checkpoint prior renormalized over legal actions.",
            "Only declared generating-checkpoint windows are exact online teaching-signal measurements; all others are fixed-reference retrospectives.",
            "A single outcome z per observation cannot by itself identify irreducible outcome noise.",
            "Deep-oracle mapped values are an independent calibrated engine proxy, not game-theoretic ground truth.",
        ],
        "repeated_outcome_diagnostic": {
            "all_standard_positions": _repeated_outcome_diagnostic(
                selected_boards, selected_outcomes
            ),
            "deep_oracle_matched_positions": _repeated_outcome_diagnostic(
                selected_boards, selected_outcomes, set(oracle_labels)
            ) if oracle_labels else None,
        },
        "checkpoints": {},
    }

    for checkpoint_name, checkpoint_path in checkpoints.items():
        checkpoint_path = os.path.abspath(checkpoint_path)
        wrapper = V4InferenceWrapper(
            game,
            checkpoint_path,
            device="cpu",
            autocast_fp16=False,
            freeze_torchscript=True,
            canonicalize_d4=args.canonical_d4,
        )
        windows = []
        all_metrics = []
        all_boards = []
        all_outcomes = []
        all_predictions = []
        for window_number, indices in enumerate(selected_by_window, start=1):
            boards = replay["boards"][indices]
            outcomes = replay["values"][indices]
            targets = _dense_targets(replay, indices)
            priors = []
            predictions = []
            for batch_start in range(0, len(boards), args.batch_size):
                batch_policies, batch_values = wrapper.predict_batch(
                    boards[batch_start:batch_start + args.batch_size]
                )
                priors.append(batch_policies)
                predictions.append(batch_values)
            priors = np.concatenate(priors)
            predictions = np.concatenate(predictions).astype(np.float64)
            metrics = _metric_arrays(game, boards, targets, priors, predictions, outcomes)
            stages = _stages(boards)
            sources = _sources(replay["metadata"][indices])
            generator = exact_generators.get(window_number)
            window_report = {
                "window": window_number,
                "positions": len(indices),
                "reference_relation": (
                    "exact_generating_prior" if generator == checkpoint_name
                    else "different_checkpoint" if generator is not None
                    else "generating_checkpoint_unavailable"
                ),
                "declared_generator": generator,
                "overall": _summarize_metrics(metrics),
                "by_stage": {
                    str(stage): _summarize_metrics(metrics, stages == stage)
                    for stage in sorted(set(stages))
                },
                "by_source": {
                    str(source): _summarize_metrics(metrics, sources == source)
                    for source in sorted(set(sources))
                },
            }
            if oracle_labels:
                window_report["deep_oracle"] = _oracle_comparison(
                    boards, outcomes, predictions, oracle_labels
                )
            windows.append(window_report)
            all_metrics.append(metrics)
            all_boards.append(boards)
            all_outcomes.append(outcomes)
            all_predictions.append(predictions)

        merged = _merge_metric_arrays(all_metrics)
        checkpoint_report = {
            "path": checkpoint_path,
            "windows": windows,
            "all_selected_positions": _summarize_metrics(merged),
        }
        if oracle_labels:
            checkpoint_report["deep_oracle_all_selected_positions"] = _oracle_comparison(
                np.concatenate(all_boards),
                np.concatenate(all_outcomes),
                np.concatenate(all_predictions),
                oracle_labels,
            )
        report["checkpoints"][checkpoint_name] = checkpoint_report

    report["elapsed_seconds"] = time.perf_counter() - started
    _atomic_json(args.output, report)
    print(json.dumps({
        "output": os.path.abspath(args.output),
        "elapsed_seconds": report["elapsed_seconds"],
        "windows": len(selected_by_window),
        "positions": int(sum(len(value) for value in selected_by_window)),
        "checkpoints": list(checkpoints),
        "oracle_labels": len(oracle_labels),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
