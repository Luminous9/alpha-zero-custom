"""Measure whether hard D4 canonicalization seams hurt V4 candidates.

The converted selection boards are already D4 canonical.  For each board, this
script enumerates its unique legal one-ply successors and measures the fraction
whose D4 canonical representative cannot be reached by the identity spatial
transform.  That fraction is the board's canonical-frame-switch exposure.

Models are evaluated on the unchanged frozen selection targets.  The primary
diagnostic is the paired objective difference in deterministic exposure
quartiles, especially the high-minus-low contrast.  This is an association,
not a causal strength gate: seam exposure can correlate with game stage and
tactical complexity, so stage/source composition is retained in the report.
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from santorini.SantoriniGame import SantoriniGame
from santorini.V4BootstrapCorpus import SOURCE_NAMES, STAGE_NAMES
from santorini.V4SeamTelemetry import (
    file_sha256,
    quartile_buckets,
    raw_selection_boards,
    seam_profile,
)
from santorini.V4Supervised import (
    DEFAULT_ALPHA_BOOT,
    DEFAULT_STAGE_RELIABILITY,
    GLOBAL_SCORE_TEMPERATURE,
    StreamingPreparedV4Corpus,
)
from santorini.pytorch.V4NNet import load_v4_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        action="append",
        dest="checkpoints",
        required=True,
        metavar="NAME=PATH",
    )
    parser.add_argument("--engine-corpus", required=True)
    parser.add_argument("--run13-component", required=True)
    parser.add_argument("--selection-plan", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260818)
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


def _device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return torch.device(name)


def _checkpoint_specs(values):
    specs = []
    for value in values:
        if "=" not in value:
            raise ValueError("Checkpoint arguments must use NAME=PATH.")
        name, path = value.split("=", 1)
        if not name or not path or any(existing[0] == name for existing in specs):
            raise ValueError("Checkpoint names and paths must be nonempty and unique.")
        if not os.path.isfile(path):
            raise FileNotFoundError("Missing checkpoint: {}".format(path))
        specs.append((name, os.path.abspath(path)))
    if len(specs) < 2:
        raise ValueError("The seam diagnostic requires at least two checkpoints.")
    return specs


def _evaluate_model(model, corpus, planes, batch_size, device):
    model = model.to(device).eval()
    policy_losses = []
    value_losses = []
    top1 = []
    with torch.inference_mode():
        for start in range(0, len(corpus), batch_size):
            indices = np.arange(start, min(start + batch_size, len(corpus)))
            batch = corpus.batch(indices, planes)
            inputs = torch.from_numpy(
                np.ascontiguousarray(batch.encoded_boards)
            ).to(device)
            log_policy, prediction = model(inputs)
            log_policy = log_policy.cpu().numpy()
            prediction = prediction[:, 0].cpu().numpy()
            policy_losses.append(-np.sum(batch.policies * log_policy, axis=1))
            value_losses.append((prediction - batch.global_blended_values) ** 2)
            top1.append(
                np.argmax(log_policy, axis=1) == np.argmax(batch.policies, axis=1)
            )
    policy_losses = np.concatenate(policy_losses)
    value_losses = np.concatenate(value_losses)
    return {
        "policy_loss": policy_losses,
        "value_loss": value_losses,
        "objective": 0.25 * policy_losses + value_losses,
        "top1": np.concatenate(top1).astype(np.float64),
    }


def _bootstrap_mean_interval(values, samples, rng):
    values = np.asarray(values, dtype=np.float64)
    means = []
    remaining = int(samples)
    while remaining:
        count = min(500, remaining)
        indices = rng.randint(len(values), size=(count, len(values)))
        means.append(values[indices].mean(axis=1))
        remaining -= count
    return list(map(float, np.quantile(np.concatenate(means), (0.025, 0.975))))


def _high_minus_low_interval(values, buckets, samples, rng):
    low = np.asarray(values, dtype=np.float64)[buckets == 0]
    high = np.asarray(values, dtype=np.float64)[buckets == 3]
    contrasts = []
    remaining = int(samples)
    while remaining:
        count = min(500, remaining)
        low_means = low[rng.randint(len(low), size=(count, len(low)))].mean(axis=1)
        high_means = high[rng.randint(len(high), size=(count, len(high)))].mean(axis=1)
        contrasts.append(high_means - low_means)
        remaining -= count
    return list(map(float, np.quantile(
        np.concatenate(contrasts), (0.025, 0.975)
    )))


def _mean_metrics(metrics, mask):
    return {
        name: float(np.mean(values[mask]))
        for name, values in metrics.items()
    }


def diagnose(args):
    if args.batch_size < 1 or args.bootstrap_samples < 1:
        raise ValueError("Batch size and bootstrap sample count must be positive.")
    if len(args.stage_reliability) != 3:
        raise ValueError("Stage reliability requires exactly three values.")
    started = time.perf_counter()
    device = _device(args.device)
    specs = _checkpoint_specs(args.checkpoints)
    game = SantoriniGame(5, sequential_placement=True)
    boards = raw_selection_boards(
        args.engine_corpus, args.run13_component, args.selection_plan
    )
    profiles = []
    for index, board in enumerate(boards):
        profiles.append(seam_profile(game, board))
        if (index + 1) % 250 == 0:
            print("profiled {}/{} positions".format(index + 1, len(boards)), flush=True)
    exposures = np.asarray([
        profile["frame_switch_exposure"] for profile in profiles
    ], dtype=np.float64)
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
    all_indices = np.arange(len(corpus))
    metadata = corpus.batch(all_indices, 13)
    stage_ids = metadata.stage_ids.astype(np.int64)
    source_ids = metadata.source_ids.astype(np.int64)

    model_metrics = {}
    model_reports = {}
    for name, path in specs:
        print("evaluating {}".format(name), flush=True)
        model, config, checkpoint = load_v4_checkpoint(path, game)
        metrics = _evaluate_model(
            model, corpus, int(config["planes"]), args.batch_size, device
        )
        model_metrics[name] = metrics
        model_reports[name] = {
            "checkpoint": path,
            "checkpoint_sha256": file_sha256(path),
            "checkpoint_epoch": checkpoint.get("epoch"),
            "config": config,
            "overall": _mean_metrics(metrics, np.ones(len(corpus), dtype=bool)),
            "by_exposure_quartile": {
                str(bucket + 1): _mean_metrics(metrics, buckets == bucket)
                for bucket in range(4)
            },
        }

    rng = np.random.RandomState(int(args.seed))
    comparisons = {}
    for first_index in range(len(specs)):
        for second_index in range(first_index + 1, len(specs)):
            first = specs[first_index][0]
            second = specs[second_index][0]
            differences = {
                metric: model_metrics[first][metric] - model_metrics[second][metric]
                for metric in ("objective", "policy_loss", "value_loss", "top1")
            }
            by_quartile = {}
            for bucket in range(4):
                mask = buckets == bucket
                by_quartile[str(bucket + 1)] = {
                    metric: float(np.mean(values[mask]))
                    for metric, values in differences.items()
                }
                by_quartile[str(bucket + 1)]["objective_bootstrap_95"] = (
                    _bootstrap_mean_interval(
                        differences["objective"][mask], args.bootstrap_samples, rng
                    )
                )
            objective_contrast = (
                float(np.mean(differences["objective"][buckets == 3]))
                - float(np.mean(differences["objective"][buckets == 0]))
            )
            comparisons["{}_minus_{}".format(first, second)] = {
                "overall": {
                    metric: float(np.mean(values))
                    for metric, values in differences.items()
                },
                "overall_objective_bootstrap_95": _bootstrap_mean_interval(
                    differences["objective"], args.bootstrap_samples, rng
                ),
                "by_exposure_quartile": by_quartile,
                "high_minus_low_objective_contrast": objective_contrast,
                "high_minus_low_objective_bootstrap_95": (
                    _high_minus_low_interval(
                        differences["objective"], buckets, args.bootstrap_samples, rng
                    )
                ),
            }

    bucket_reports = {}
    for bucket in range(4):
        mask = buckets == bucket
        bucket_reports[str(bucket + 1)] = {
            "positions": int(np.sum(mask)),
            "exposure_min": float(np.min(exposures[mask])),
            "exposure_mean": float(np.mean(exposures[mask])),
            "exposure_max": float(np.max(exposures[mask])),
            "mean_unique_successors": float(np.mean([
                profiles[index]["unique_successors"]
                for index in np.flatnonzero(mask)
            ])),
            "stabilizer_positions": int(np.sum([
                profiles[index]["current_stabilizer_size"] > 1
                for index in np.flatnonzero(mask)
            ])),
            "positions_by_stage": {
                name: int(np.sum(mask & (stage_ids == stage)))
                for stage, name in enumerate(STAGE_NAMES)
            },
            "positions_by_source": {
                name: int(np.sum(mask & (source_ids == source)))
                for source, name in enumerate(SOURCE_NAMES)
            },
        }

    report = {
        "schema_version": 1,
        "type": "santorini_v4_canonical_seam_diagnostic",
        "definition": (
            "fraction of unique legal one-ply successors whose D4 canonical "
            "representative does not admit the identity spatial transform"
        ),
        "interpretation": (
            "diagnostic association only; exposure may correlate with stage, "
            "source, branching factor, and tactical complexity"
        ),
        "device": str(device),
        "selection_positions": len(boards),
        "selection_plan": os.path.abspath(args.selection_plan),
        "engine_corpus": os.path.abspath(args.engine_corpus),
        "run13_component": os.path.abspath(args.run13_component),
        "policy_weight": 0.25,
        "bootstrap_samples": int(args.bootstrap_samples),
        "seed": int(args.seed),
        "exposure": {
            "min": float(np.min(exposures)),
            "mean": float(np.mean(exposures)),
            "median": float(np.median(exposures)),
            "max": float(np.max(exposures)),
            "zero_exposure_positions": int(np.sum(exposures == 0)),
            "quartiles": bucket_reports,
        },
        "models": model_reports,
        "paired_comparisons": comparisons,
        "elapsed_seconds": time.perf_counter() - started,
        "final_test_touched": False,
        "final_arena_seeds_touched": False,
    }
    corpus.close()
    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    temporary = output_path + ".tmp"
    with open(temporary, "w") as output:
        json.dump(report, output, indent=2, sort_keys=True)
    os.replace(temporary, output_path)
    return report


def main():
    args = parse_args()
    report = diagnose(args)
    summary = {
        "output": os.path.abspath(args.output),
        "selection_positions": report["selection_positions"],
        "exposure": report["exposure"],
        "paired_comparisons": report["paired_comparisons"],
        "elapsed_seconds": report["elapsed_seconds"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
