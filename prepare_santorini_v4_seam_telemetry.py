"""Freeze the P1c canonical-seam suite consumed by P2 telemetry."""

import argparse
import json
import os
import time

import numpy as np

from santorini.SantoriniGame import SantoriniGame
from santorini.V4SeamTelemetry import (
    POLICY_WEIGHT,
    SCHEMA_VERSION,
    evaluate_loss_vectors,
    file_sha256,
    policies_to_csr,
    quartile_buckets,
    raw_selection_boards,
    seam_profile,
    summarize_loss_vectors,
)
from santorini.V4Supervised import (
    DEFAULT_ALPHA_BOOT,
    DEFAULT_STAGE_RELIABILITY,
    GLOBAL_SCORE_TEMPERATURE,
    StreamingPreparedV4Corpus,
)
from santorini.pytorch.V4NNet import V4InferenceWrapper


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-corpus", required=True)
    parser.add_argument("--run13-component", required=True)
    parser.add_argument("--selection-plan", required=True)
    parser.add_argument("--baseline-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--policy-epsilon", type=float, default=0.05)
    parser.add_argument("--alpha-boot", type=float, default=DEFAULT_ALPHA_BOOT)
    parser.add_argument("--score-temperature", type=float, default=GLOBAL_SCORE_TEMPERATURE)
    parser.add_argument(
        "--stage-reliability",
        type=float,
        nargs=3,
        default=DEFAULT_STAGE_RELIABILITY,
        metavar=("EARLY", "MIDDLE", "LATE"),
    )
    return parser.parse_args()


def _save_npz_atomic(path, suite):
    path = os.path.abspath(path)
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    temporary = path + ".tmp.npz"
    np.savez_compressed(temporary, **suite)
    os.replace(temporary, path)
    return path


def _save_json_atomic(path, payload):
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w") as output:
        json.dump(payload, output, indent=2, sort_keys=True)
    os.replace(temporary, path)


def prepare(args):
    if args.batch_size < 1:
        raise ValueError("Batch size must be positive.")
    if len(args.stage_reliability) != 3:
        raise ValueError("Stage reliability requires exactly three values.")
    started = time.perf_counter()
    game = SantoriniGame(5, sequential_placement=True)
    boards = np.asarray(raw_selection_boards(
        args.engine_corpus, args.run13_component, args.selection_plan
    ), dtype=np.int8)
    profiles = []
    for index, board in enumerate(boards):
        profiles.append(seam_profile(game, board))
        if (index + 1) % 250 == 0:
            print("profiled {}/{} positions".format(index + 1, len(boards)), flush=True)
    exposures = np.asarray([
        profile["frame_switch_exposure"] for profile in profiles
    ], dtype=np.float32)
    buckets = quartile_buckets(exposures)

    corpus = StreamingPreparedV4Corpus(
        engine_path=args.engine_corpus,
        run13_path=args.run13_component,
        plan_path=args.selection_plan,
        expected_split=1,
        policy_epsilon=args.policy_epsilon,
        alpha_boot=args.alpha_boot,
        stage_reliability=args.stage_reliability,
        temperature=args.score_temperature,
    )
    policy_batches = []
    value_batches = []
    stage_batches = []
    source_batches = []
    for start in range(0, len(corpus), args.batch_size):
        indices = np.arange(start, min(start + args.batch_size, len(corpus)))
        batch = corpus.batch(indices, 13)
        policy_batches.append(batch.policies.astype(np.float32))
        value_batches.append(batch.global_blended_values.astype(np.float32))
        stage_batches.append(batch.stage_ids.astype(np.int8))
        source_batches.append(batch.source_ids.astype(np.int8))
    policies = np.concatenate(policy_batches)
    indptr, policy_indices, policy_values = policies_to_csr(policies)
    del policies
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "type": "santorini_v4_frozen_seam_telemetry_suite",
        "definition": (
            "fraction of unique legal one-ply successors whose D4 canonical "
            "representative does not admit the identity spatial transform"
        ),
        "baseline_name": "p1c_handoff",
        "baseline_checkpoint": os.path.abspath(args.baseline_checkpoint),
        "baseline_checkpoint_sha256": file_sha256(args.baseline_checkpoint),
        "engine_corpus_sha256": file_sha256(args.engine_corpus),
        "run13_component_sha256": file_sha256(args.run13_component),
        "selection_plan_sha256": file_sha256(args.selection_plan),
        "policy_target": "frozen_bootstrap_teacher_policy",
        "value_target": "frozen_global_score_winner_blend",
        "policy_weight": POLICY_WEIGHT,
        "policy_epsilon": float(args.policy_epsilon),
        "alpha_boot": float(args.alpha_boot),
        "score_temperature": float(args.score_temperature),
        "stage_reliability": list(map(float, args.stage_reliability)),
        "positions": int(len(boards)),
    }
    suite = {
        "boards": boards,
        "policy_indptr": indptr,
        "policy_indices": policy_indices,
        "policy_values": policy_values,
        "value_targets": np.concatenate(value_batches),
        "exposures": exposures,
        "exposure_quartiles": buckets,
        "stage_ids": np.concatenate(stage_batches),
        "source_ids": np.concatenate(source_batches),
        "metadata": np.asarray(json.dumps(metadata, sort_keys=True)),
    }
    baseline = V4InferenceWrapper(
        game,
        args.baseline_checkpoint,
        device=args.device,
        autocast_fp16=False,
        freeze_torchscript=True,
        canonicalize_d4=True,
        canonical_cache_size=0,
    )
    # Validation needs baseline vectors to exist; placeholders are never written.
    for name in ("policy_loss", "value_loss", "objective", "top1"):
        suite["baseline_{}".format(name)] = np.zeros(len(boards), dtype=np.float32)
    suite["metadata_parsed"] = metadata
    baseline_metrics = evaluate_loss_vectors(baseline, suite, batch_size=args.batch_size)
    for name, values in baseline_metrics.items():
        suite["baseline_{}".format(name)] = np.asarray(values, dtype=np.float32)
    suite.pop("metadata_parsed")
    output_path = _save_npz_atomic(args.output, suite)
    summary = summarize_loss_vectors(baseline_metrics, buckets)
    report = {
        "schema_version": SCHEMA_VERSION,
        "type": "santorini_v4_frozen_seam_telemetry_preparation",
        "suite": output_path,
        "suite_sha256": file_sha256(output_path),
        "metadata": metadata,
        "exposure": {
            "min": float(np.min(exposures)),
            "mean": float(np.mean(exposures)),
            "median": float(np.median(exposures)),
            "max": float(np.max(exposures)),
            "positions_per_quartile": [
                int(np.sum(buckets == bucket)) for bucket in range(4)
            ],
        },
        "baseline": summary,
        "baseline_high_minus_low_objective_contrast": (
            summary["by_quartile"]["4"]["objective"]
            - summary["by_quartile"]["1"]["objective"]
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "final_test_touched": False,
        "final_arena_seeds_touched": False,
    }
    corpus.close()
    report_path = args.report or os.path.splitext(output_path)[0] + ".json"
    _save_json_atomic(report_path, report)
    report["report"] = os.path.abspath(report_path)
    return report


def main():
    report = prepare(parse_args())
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
