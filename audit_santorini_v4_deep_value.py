"""Build and evaluate a frozen deep-oracle value audit for Santorini V4.

The audit has three deliberately resumable phases:

1. freeze a D4-unique, stage/window-stratified suite before seeing labels;
2. label that suite with cold-TT oracle searches cached in SQLite; and
3. evaluate every declared checkpoint on the identical frozen positions.

The resulting oracle values are calibrated engine proxies, not solved-game
truth.  Checkpoint comparisons are paired because every model sees the same
positions and labels.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch

from santorini.OracleResearch import (
    MATE_SCORE_THRESHOLD,
    ParallelOraclePool,
    STAGES,
    canonical_d4_fen,
    file_sha256,
    stage_for_builds,
)
from santorini.SantoriniGame import SantoriniGame
from santorini.SantoriniOracle import anonymous_board_key
from santorini.pytorch.V4NNet import V4InferenceWrapper


SCHEMA_VERSION = 1
CALIBRATION_VERSION = "v4-p2-deep-value-score400-v1"
DEFAULT_SEED = 20260814
DEFAULT_BANDS = (
    ("windows_1_4", 1, 4),
    ("windows_5_8", 5, 8),
    ("windows_9_11", 9, 11),
    ("windows_12_14", 12, 14),
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", required=True)
    parser.add_argument(
        "--checkpoint", action="append", default=[], metavar="NAME=PATH",
        help="Checkpoint to evaluate; repeat for multiple checkpoints.",
    )
    parser.add_argument(
        "--exclude-final-corpus", action="append", default=[],
        help="Corpus NPZ whose split_id=2 position hashes must be excluded.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--positions-per-stratum", type=int, default=40)
    parser.add_argument("--nodes", type=int, default=250_000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--torch-threads", type=int, default=0)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--oracle-binary")
    parser.add_argument(
        "--suite-only", action="store_true",
        help="Freeze/validate the suite without searching or evaluating checkpoints.",
    )
    return parser.parse_args()


def _atomic_json(path, payload):
    path = os.path.abspath(os.fspath(path))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w") as output:
        json.dump(payload, output, indent=2, sort_keys=True)
    os.replace(temporary, path)


def _atomic_npz(path, **payload):
    path = os.path.abspath(os.fspath(path))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + ".tmp"
    try:
        with open(temporary, "wb") as output:
            np.savez_compressed(output, **payload)
        os.replace(temporary, path)
    finally:
        try:
            os.remove(temporary)
        except FileNotFoundError:
            pass


def _named_paths(entries):
    parsed = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError("Expected NAME=PATH, received {!r}.".format(entry))
        name, path = (part.strip() for part in entry.split("=", 1))
        if not name or not path or name in parsed:
            raise ValueError("Invalid or duplicate checkpoint {!r}.".format(entry))
        if not os.path.isfile(path):
            raise FileNotFoundError("Checkpoint not found: {}".format(path))
        parsed[name] = os.path.abspath(path)
    return parsed


def nominal_score_value(score):
    """Map the established nominal T=400 win probability onto [-1, 1]."""
    score = int(score)
    if abs(score) >= MATE_SCORE_THRESHOLD:
        return float(np.sign(score))
    scaled = float(np.clip(score / 400.0, -50.0, 50.0))
    return float(2.0 / (1.0 + math.exp(-scaled)) - 1.0)


def anonymous_d4_hash(board):
    """Match the corpus' worker-label-independent, D4-invariant position hash."""
    board = np.asarray(board, dtype=int)
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


def _source(encoded):
    try:
        return str(json.loads(str(encoded)).get("source", "unknown"))
    except (TypeError, ValueError, json.JSONDecodeError):
        return "invalid_metadata"


def _band_for_window(window):
    matches = [name for name, start, end in DEFAULT_BANDS if start <= window <= end]
    if len(matches) != 1:
        raise ValueError("Replay window {} belongs to no declared audit band.".format(window))
    return matches[0]


def _load_final_hashes(paths):
    excluded = set()
    records = []
    for path in paths:
        path = os.path.abspath(path)
        if not os.path.isfile(path):
            raise FileNotFoundError("Final corpus not found: {}".format(path))
        with np.load(path, allow_pickle=False) as payload:
            if "position_hashes" not in payload.files or "split_ids" not in payload.files:
                raise ValueError("Final corpus lacks position_hashes/split_ids: {}".format(path))
            hashes = payload["position_hashes"]
            splits = payload["split_ids"]
            if len(hashes) != len(splits):
                raise ValueError("Final corpus hash and split arrays differ in length.")
            final_hashes = set(map(str, hashes[splits == 2]))
        excluded.update(final_hashes)
        records.append({
            "path": path,
            "sha256": file_sha256(path),
            "final_split_hashes": len(final_hashes),
        })
    return excluded, records


def _selection_contract(replay_path, final_records, positions_per_stratum, seed):
    return {
        "schema_version": SCHEMA_VERSION,
        "type": "santorini_v4_p2_frozen_deep_value_suite",
        "replay": {
            "path": os.path.abspath(replay_path),
            "sha256": file_sha256(replay_path),
        },
        "excluded_final_corpora": final_records,
        "standard_play_only": True,
        "d4_unique": True,
        "bands": [
            {"name": name, "first_window": start, "last_window": end}
            for name, start, end in DEFAULT_BANDS
        ],
        "stages": list(STAGES),
        "positions_per_band_stage_stratum": int(positions_per_stratum),
        "positions": int(positions_per_stratum * len(DEFAULT_BANDS) * len(STAGES)),
        "seed": int(seed),
        "selection_before_labels": True,
        "final_test_touched": False,
    }


def _stable_occurrence_score(seed, fen, replay_index):
    value = "{}:{}:{}".format(seed, fen, replay_index).encode("utf-8")
    return hashlib.sha256(value).digest()


def freeze_suite(
    replay_path, suite_path, manifest_path, final_hashes, final_records,
    positions_per_stratum, seed,
):
    contract = _selection_contract(
        replay_path, final_records, positions_per_stratum, seed
    )
    if os.path.exists(suite_path) or os.path.exists(manifest_path):
        if not os.path.isfile(suite_path) or not os.path.isfile(manifest_path):
            raise ValueError("Frozen suite and manifest must either both exist or both be absent.")
        with open(manifest_path) as source:
            existing = json.load(source)
        existing_contract = dict(existing)
        for key in ("suite_sha256", "selected_by_stratum", "selected_by_source", "excluded_matches"):
            existing_contract.pop(key, None)
        if existing_contract != contract:
            raise ValueError("Existing frozen suite belongs to a different selection contract.")
        if file_sha256(suite_path) != existing["suite_sha256"]:
            raise ValueError("Frozen suite digest does not match its manifest.")
        return existing

    with np.load(replay_path, allow_pickle=False) as replay:
        required = {"history_lengths", "boards", "values", "example_metadata"}
        missing = required.difference(replay.files)
        if missing:
            raise ValueError("Replay is missing required fields: {}".format(sorted(missing)))
        lengths = replay["history_lengths"].astype(np.int64)
        if len(lengths) != DEFAULT_BANDS[-1][2]:
            raise ValueError(
                "The default frozen-suite contract expects exactly {} replay windows; got {}."
                .format(DEFAULT_BANDS[-1][2], len(lengths))
            )
        boundaries = np.cumsum(lengths)
        boards = replay["boards"]
        outcomes = replay["values"]
        metadata = replay["example_metadata"]

        # Pick one deterministic observation for every D4-equivalence class
        # before stratifying. This prevents duplicated openings from being
        # represented in multiple history bands.
        unique = {}
        excluded_matches = 0
        for replay_index, board in enumerate(boards):
            board = np.asarray(board, dtype=np.int8)
            if int(np.count_nonzero(board[0])) != 4:
                continue
            position_hash = anonymous_d4_hash(board)
            if position_hash in final_hashes:
                excluded_matches += 1
                continue
            fen = canonical_d4_fen(board.astype(int))
            candidate = {
                "board": board.copy(),
                "outcome_z": float(outcomes[replay_index]),
                "fen": fen,
                "position_hash": position_hash,
                "replay_index": int(replay_index),
                "window": int(np.searchsorted(boundaries, replay_index, side="right") + 1),
                "stage": stage_for_builds(int(np.sum(board[1]))),
                "build_count": int(np.sum(board[1])),
                "source": _source(metadata[replay_index]),
            }
            candidate["band"] = _band_for_window(candidate["window"])
            score = _stable_occurrence_score(seed, fen, replay_index)
            current = unique.get(fen)
            if current is None or score < current[0]:
                unique[fen] = (score, candidate)

    strata = {
        (band, stage): []
        for band, _, _ in DEFAULT_BANDS
        for stage in STAGES
    }
    for _, candidate in unique.values():
        strata[(candidate["band"], candidate["stage"])].append(candidate)

    selected = []
    selected_by_stratum = {}
    for band, _, _ in DEFAULT_BANDS:
        for stage in STAGES:
            candidates = strata[(band, stage)]
            quota = int(positions_per_stratum)
            if len(candidates) < quota:
                raise ValueError(
                    "Audit stratum {}/{} has {} eligible positions for quota {}."
                    .format(band, stage, len(candidates), quota)
                )
            candidates.sort(key=lambda record: hashlib.sha256(
                "{}:{}".format(seed, record["fen"]).encode("utf-8")
            ).digest())
            selected.extend(candidates[:quota])
            selected_by_stratum["{}/{}".format(band, stage)] = quota

    selected.sort(key=lambda record: hashlib.sha256(
        "{}:final:{}".format(seed, record["fen"]).encode("utf-8")
    ).digest())
    _atomic_npz(
        suite_path,
        schema_version=np.asarray([SCHEMA_VERSION], dtype=np.int16),
        boards=np.stack([record["board"] for record in selected]).astype(np.int8),
        oracle_fens=np.asarray([record["fen"] for record in selected]),
        position_hashes=np.asarray([record["position_hash"] for record in selected]),
        stages=np.asarray([record["stage"] for record in selected]),
        build_counts=np.asarray([record["build_count"] for record in selected], dtype=np.int16),
        windows=np.asarray([record["window"] for record in selected], dtype=np.int16),
        bands=np.asarray([record["band"] for record in selected]),
        replay_indices=np.asarray([record["replay_index"] for record in selected], dtype=np.int64),
        sources=np.asarray([record["source"] for record in selected]),
        outcomes_z=np.asarray([record["outcome_z"] for record in selected], dtype=np.float32),
    )
    manifest = dict(contract)
    manifest.update({
        "suite_sha256": file_sha256(suite_path),
        "selected_by_stratum": selected_by_stratum,
        "selected_by_source": {
            source: sum(record["source"] == source for record in selected)
            for source in sorted({record["source"] for record in selected})
        },
        "excluded_matches": int(excluded_matches),
    })
    _atomic_json(manifest_path, manifest)
    return manifest


def label_suite(suite_path, cache_path, oracle_binary, nodes, workers):
    with np.load(suite_path, allow_pickle=False) as suite:
        fens = list(map(str, suite["oracle_fens"]))
    pool = ParallelOraclePool(oracle_binary, cache_path=cache_path)
    started = time.perf_counter()
    try:
        labels = [None] * len(fens)

        def label(index):
            return index, pool.label_fen(
                fens[index], nodes, CALIBRATION_VERSION, nominal_score_value
            )

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(label, index) for index in range(len(fens))]
            completed = 0
            for future in as_completed(futures):
                index, result = future.result()
                labels[index] = result
                completed += 1
                if completed == 1 or completed % 25 == 0 or completed == len(fens):
                    print(
                        "Deep-value labels: {}/{} complete.".format(completed, len(fens)),
                        flush=True,
                    )
        return labels, {
            "binary": os.path.abspath(os.fspath(oracle_binary)),
            "engine_digest": pool.engine_digest,
            "requested_nodes": int(nodes),
            "calibration_version": CALIBRATION_VERSION,
            "nominal_score_temperature": 400.0,
            "cold_tt_per_position": True,
            "cache_hits": sum(bool(label["cache_hit"]) for label in labels),
            "cache_misses": sum(not bool(label["cache_hit"]) for label in labels),
            "elapsed_seconds": time.perf_counter() - started,
        }
    finally:
        pool.close()


def _pearson(first, second):
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if len(first) < 2 or np.std(first) == 0 or np.std(second) == 0:
        return None
    return float(np.corrcoef(first, second)[0, 1])


def _ranks(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def _spearman(first, second):
    return _pearson(_ranks(first), _ranks(second))


def value_metrics(predictions, oracle_values):
    predictions = np.asarray(predictions, dtype=np.float64)
    oracle_values = np.asarray(oracle_values, dtype=np.float64)
    errors = predictions - oracle_values
    return {
        "count": int(len(predictions)),
        "pearson": _pearson(predictions, oracle_values),
        "spearman": _spearman(predictions, oracle_values),
        "mse": float(np.mean(errors ** 2)),
        "mae": float(np.mean(np.abs(errors))),
        "bias": float(np.mean(errors)),
        "prediction_mean": float(np.mean(predictions)),
        "prediction_stddev": float(np.std(predictions)),
        "oracle_mean": float(np.mean(oracle_values)),
        "oracle_stddev": float(np.std(oracle_values)),
        "sign_agreement": float(np.mean(np.sign(predictions) == np.sign(oracle_values))),
    }


def _stratified_bootstrap_indices(strata, samples, seed):
    rng = np.random.RandomState(seed)
    groups = [np.flatnonzero(strata == value) for value in sorted(set(map(str, strata)))]
    for _ in range(samples):
        yield np.concatenate([
            rng.choice(group, size=len(group), replace=True) for group in groups
        ])


def paired_bootstrap_delta(candidate, reference, oracle_values, strata, samples, seed):
    candidate = np.asarray(candidate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    oracle_values = np.asarray(oracle_values, dtype=np.float64)
    values = {"pearson": [], "mse": [], "mae": []}
    rng = np.random.RandomState(seed)
    groups = [np.flatnonzero(strata == value) for value in sorted(set(map(str, strata)))]
    chunk_size = 250
    completed = 0
    while completed < samples:
        count = min(chunk_size, samples - completed)
        indices = np.concatenate([
            rng.choice(group, size=(count, len(group)), replace=True)
            for group in groups
        ], axis=1)
        candidate_draws = candidate[indices]
        reference_draws = reference[indices]
        oracle_draws = oracle_values[indices]
        values["mse"].extend(np.mean((candidate_draws - oracle_draws) ** 2, axis=1)
                             - np.mean((reference_draws - oracle_draws) ** 2, axis=1))
        values["mae"].extend(np.mean(np.abs(candidate_draws - oracle_draws), axis=1)
                             - np.mean(np.abs(reference_draws - oracle_draws), axis=1))

        oracle_centered = oracle_draws - np.mean(oracle_draws, axis=1, keepdims=True)

        def row_correlation(draws):
            centered = draws - np.mean(draws, axis=1, keepdims=True)
            denominator = np.sqrt(
                np.sum(centered ** 2, axis=1) * np.sum(oracle_centered ** 2, axis=1)
            )
            numerator = np.sum(centered * oracle_centered, axis=1)
            return np.divide(
                numerator, denominator,
                out=np.full(count, np.nan, dtype=np.float64),
                where=denominator > 0,
            )

        correlation_deltas = row_correlation(candidate_draws) - row_correlation(reference_draws)
        values["pearson"].extend(correlation_deltas[np.isfinite(correlation_deltas)])
        completed += count
    result = {}
    candidate_metrics = value_metrics(candidate, oracle_values)
    reference_metrics = value_metrics(reference, oracle_values)
    for metric, draws in values.items():
        point_delta = (
            None
            if candidate_metrics[metric] is None or reference_metrics[metric] is None
            else float(candidate_metrics[metric] - reference_metrics[metric])
        )
        result[metric] = {
            "delta": point_delta,
            "bootstrap_95": (
                [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]
                if draws else None
            ),
        }
    return result


def _metric_slices(predictions, oracle_values, stages, bands, sources):
    return {
        "overall": value_metrics(predictions, oracle_values),
        "by_stage": {
            stage: value_metrics(predictions[stages == stage], oracle_values[stages == stage])
            for stage in STAGES
        },
        "by_window_band": {
            band: value_metrics(predictions[bands == band], oracle_values[bands == band])
            for band, _, _ in DEFAULT_BANDS
        },
        "by_source": {
            source: value_metrics(predictions[sources == source], oracle_values[sources == source])
            for source in sorted(set(map(str, sources)))
        },
    }


def evaluate_checkpoints(
    suite_path, labels, checkpoints, batch_size, bootstrap_samples, seed,
):
    with np.load(suite_path, allow_pickle=False) as suite:
        boards = suite["boards"].astype(np.int8)
        stages = suite["stages"].astype(str)
        bands = suite["bands"].astype(str)
        sources = suite["sources"].astype(str)
        outcomes = suite["outcomes_z"].astype(np.float64)
        replay_indices = suite["replay_indices"].astype(np.int64)
        position_hashes = suite["position_hashes"].astype(str)
    oracle_values = np.asarray([label["mapped_value"] for label in labels], dtype=np.float64)
    game = SantoriniGame(5, sequential_placement=True)
    predictions = {}
    checkpoint_reports = {}
    for name, path in checkpoints.items():
        print("Evaluating checkpoint {}...".format(name), flush=True)
        wrapper = V4InferenceWrapper(
            game, path, device="cpu", autocast_fp16=False,
            freeze_torchscript=True, canonicalize_d4=True,
        )
        batches = []
        for start in range(0, len(boards), batch_size):
            _, values = wrapper.predict_batch(boards[start:start + batch_size])
            batches.append(np.asarray(values, dtype=np.float64))
        checkpoint_predictions = np.concatenate(batches)
        predictions[name] = checkpoint_predictions
        checkpoint_reports[name] = {
            "path": path,
            "sha256": file_sha256(path),
            **_metric_slices(
                checkpoint_predictions, oracle_values, stages, bands, sources
            ),
        }
        del wrapper

    strata = np.asarray([
        "{}/{}".format(band, stage) for band, stage in zip(bands, stages)
    ])
    comparisons = {}
    comparison_offset = 0
    # P1c is the pretraining anchor; iteration 4 is the end of the diagnostic
    # arm and a second predeclared healthy reference. Compare the full lineage
    # with both whenever available.
    for anchor in ("p1c", "iter4"):
        if anchor not in predictions:
            continue
        for name in checkpoints:
            if name == anchor:
                continue
            key = "{}_vs_{}".format(name, anchor)
            comparisons[key] = paired_bootstrap_delta(
                predictions[name], predictions[anchor], oracle_values, strata,
                bootstrap_samples, seed + comparison_offset,
            )
            comparison_offset += 1
    ordered_iteration_names = [
        name for name in checkpoints
        if name.startswith("iter") and name[4:].isdigit()
    ]
    ordered_iteration_names.sort(key=lambda name: int(name[4:]))
    for reference, candidate in zip(
        ordered_iteration_names, ordered_iteration_names[1:]
    ):
        key = "{}_vs_{}".format(candidate, reference)
        if key not in comparisons:
            comparisons[key] = paired_bootstrap_delta(
                predictions[candidate], predictions[reference], oracle_values, strata,
                bootstrap_samples, seed + comparison_offset,
            )
            comparison_offset += 1
    if "iter14" in predictions and "iter11" in predictions:
        comparisons["iter14_vs_iter11"] = paired_bootstrap_delta(
            predictions["iter14"], predictions["iter11"], oracle_values, strata,
            bootstrap_samples, seed + comparison_offset,
        )

    # Preserve uncertainty for the predeclared lineage landmarks within each
    # stage and replay band. These slices are diagnostic, not independent
    # promotion gates, so keep the set small and explicit.
    landmark_pairs = (
        ("iter1", "p1c"),
        ("iter4", "p1c"),
        ("iter8", "iter4"),
        ("iter11", "p1c"),
        ("iter11", "iter4"),
        ("iter14", "p1c"),
        ("iter14", "iter4"),
        ("iter14", "iter11"),
    )
    comparison_slices = {}
    for candidate, reference in landmark_pairs:
        if candidate not in predictions or reference not in predictions:
            continue
        key = "{}_vs_{}".format(candidate, reference)
        slice_report = {"by_stage": {}, "by_window_band": {}}
        for stage in STAGES:
            mask = stages == stage
            slice_report["by_stage"][stage] = paired_bootstrap_delta(
                predictions[candidate][mask], predictions[reference][mask],
                oracle_values[mask], bands[mask], bootstrap_samples,
                seed + comparison_offset,
            )
            comparison_offset += 1
        for band, _, _ in DEFAULT_BANDS:
            mask = bands == band
            slice_report["by_window_band"][band] = paired_bootstrap_delta(
                predictions[candidate][mask], predictions[reference][mask],
                oracle_values[mask], stages[mask], bootstrap_samples,
                seed + comparison_offset,
            )
            comparison_offset += 1
        comparison_slices[key] = slice_report

    rows = {
        "schema_version": np.asarray([SCHEMA_VERSION], dtype=np.int16),
        "position_hashes": position_hashes,
        "replay_indices": replay_indices,
        "stages": stages,
        "bands": bands,
        "sources": sources,
        "oracle_values": oracle_values.astype(np.float32),
        "outcomes_z": outcomes.astype(np.float32),
    }
    for name, values in predictions.items():
        rows["prediction_{}".format(name)] = values.astype(np.float32)
    return checkpoint_reports, comparisons, comparison_slices, rows, {
        "outcome_z": _metric_slices(outcomes, oracle_values, stages, bands, sources),
        "score": {
            "mean": float(np.mean([label["score"] for label in labels])),
            "stddev": float(np.std([label["score"] for label in labels])),
            "mate_band_fraction": float(np.mean([label["mate_band"] for label in labels])),
        },
    }


def main():
    args = parse_args()
    if args.positions_per_stratum < 1 or args.nodes < 1 or args.workers < 1:
        raise ValueError("Position quota, node budget, and worker count must be positive.")
    if args.batch_size < 1 or args.bootstrap_samples < 1 or args.torch_threads < 0:
        raise ValueError("Batch/bootstrap counts must be positive and torch threads nonnegative.")
    if args.torch_threads:
        torch.set_num_threads(args.torch_threads)

    replay_path = os.path.abspath(args.replay)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    suite_path = output_dir / "frozen-value-suite.npz"
    manifest_path = output_dir / "frozen-value-suite-manifest.json"
    cache_path = output_dir / "deep-oracle-labels.sqlite3"
    rows_path = output_dir / "deep-value-audit-rows.npz"
    summary_path = output_dir / "deep-value-audit-summary.json"

    final_hashes, final_records = _load_final_hashes(args.exclude_final_corpus)
    manifest = freeze_suite(
        replay_path, suite_path, manifest_path, final_hashes, final_records,
        args.positions_per_stratum, args.seed,
    )
    print(json.dumps({
        "frozen_suite": str(suite_path),
        "suite_sha256": manifest["suite_sha256"],
        "positions": manifest["positions"],
        "selected_by_source": manifest["selected_by_source"],
        "excluded_final_matches": manifest["excluded_matches"],
    }, indent=2, sort_keys=True), flush=True)
    if args.suite_only:
        return
    checkpoints = _named_paths(args.checkpoint)
    if not checkpoints:
        raise ValueError("At least one --checkpoint is required unless --suite-only is used.")
    oracle_binary = args.oracle_binary
    if oracle_binary is None:
        raise ValueError("--oracle-binary is required for the frozen audit.")
    if not os.path.isfile(oracle_binary):
        raise FileNotFoundError("Oracle binary not found: {}".format(oracle_binary))

    labels, oracle_contract = label_suite(
        suite_path, cache_path, oracle_binary, args.nodes, args.workers
    )
    checkpoint_reports, comparisons, comparison_slices, rows, references = evaluate_checkpoints(
        suite_path, labels, checkpoints, args.batch_size,
        args.bootstrap_samples, args.seed,
    )
    _atomic_npz(rows_path, **rows)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "type": "santorini_v4_p2_deep_value_audit",
        "suite": {
            "path": str(suite_path),
            "sha256": manifest["suite_sha256"],
            "manifest": str(manifest_path),
            "positions": manifest["positions"],
        },
        "oracle": oracle_contract,
        "references": references,
        "checkpoints": checkpoint_reports,
        "paired_checkpoint_deltas": comparisons,
        "paired_checkpoint_slice_deltas": comparison_slices,
        "rows": {
            "path": str(rows_path),
            "sha256": file_sha256(rows_path),
        },
        "bootstrap": {
            "samples": int(args.bootstrap_samples),
            "seed": int(args.seed),
            "stratified_by": ["window_band", "stage"],
        },
        "interpretation": [
            "The 250k oracle value is an independently searched calibrated proxy, not solved-game truth.",
            "All checkpoint deltas are paired on the same frozen positions and labels.",
            "An erosion conclusion should require materially worse iteration-11/14 metrics than the P1c/iteration-4 references with paired intervals supporting the change; exact monotonicity is not required.",
            "The frozen suite excludes split_id=2 hashes from every declared final corpus.",
        ],
        "final_test_touched": False,
    }
    _atomic_json(summary_path, summary)
    print(json.dumps({
        "summary": str(summary_path),
        "rows": str(rows_path),
        "checkpoints": list(checkpoints),
        "oracle": oracle_contract,
    }, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
