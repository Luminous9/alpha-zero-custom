"""Build a deterministic stage/source-balanced V4 bootstrap sampling epoch."""

import argparse
import json
import os
import time

import numpy as np

from santorini.V4BootstrapCorpus import (
    SOURCE_NAMES,
    STAGE_NAMES,
    build_sampling_plan,
    validate_no_cross_corpus_leakage,
)


SCHEMA_VERSION = 1
SPLIT_IDS = {"train": 0, "selection": 1, "test": 2}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-corpus", required=True)
    parser.add_argument("--run13-component", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report-out")
    parser.add_argument("--draws", type=int, default=10_000)
    parser.add_argument("--split", choices=tuple(SPLIT_IDS), default="train")
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument(
        "--sampling-mode",
        choices=("with-replacement", "without-replacement"),
        default="with-replacement",
        help=(
            "Pilot plans may sample with replacement. Scaled architecture "
            "plans should use without-replacement so nominal size equals "
            "unique corpus coverage."
        ),
    )
    parser.add_argument(
        "--stage-fractions", type=float, nargs=3, default=(0.20, 0.35, 0.45),
        metavar=("EARLY", "MIDDLE", "LATE"),
    )
    parser.add_argument(
        "--source-fractions", type=float, nargs=3, default=(0.70, 0.20, 0.10),
        metavar=("MAIN", "SUBGAME", "RUN13"),
    )
    parser.add_argument(
        "--source-counts", type=int, nargs=3,
        metavar=("MAIN", "SUBGAME", "RUN13"),
        help=(
            "Exact source draw counts; overrides --source-fractions and must "
            "sum to --draws. This supports a fixed unique Run13 anchor while "
            "the engine corpus grows."
        ),
    )
    parser.add_argument(
        "--joint-counts", type=int, nargs=9,
        metavar=(
            "MAIN_E", "MAIN_M", "MAIN_L",
            "SUB_E", "SUB_M", "SUB_L",
            "RUN13_E", "RUN13_M", "RUN13_L",
        ),
        help=(
            "Exact row-major source/stage counts. Overrides both marginal "
            "options; row and column sums are reported as the contract."
        ),
    )
    return parser.parse_args()


def build(args):
    started = time.perf_counter()
    joint_counts = None
    if args.joint_counts is not None:
        joint_counts = np.asarray(args.joint_counts, dtype=np.int64).reshape(3, 3)
        if np.any(joint_counts < 0) or int(joint_counts.sum()) != int(args.draws):
            raise ValueError("Joint counts must be nonnegative and sum to --draws.")
        source_counts = tuple(map(int, joint_counts.sum(axis=1)))
        stage_counts = tuple(map(int, joint_counts.sum(axis=0)))
        source_fractions = tuple(value / float(args.draws) for value in source_counts)
        stage_fractions = tuple(value / float(args.draws) for value in stage_counts)
    elif args.source_counts is not None:
        if any(value < 0 for value in args.source_counts):
            raise ValueError("Source counts must be nonnegative.")
        if sum(args.source_counts) != int(args.draws):
            raise ValueError("Source counts must sum exactly to --draws.")
        source_fractions = tuple(
            value / float(args.draws) for value in args.source_counts
        )
        source_counts = args.source_counts
        stage_fractions = args.stage_fractions
    else:
        source_fractions = args.source_fractions
        source_counts = None
        stage_fractions = args.stage_fractions
    with np.load(args.engine_corpus, allow_pickle=False) as engine, np.load(
        args.run13_component, allow_pickle=False
    ) as run13:
        overlaps = validate_no_cross_corpus_leakage(engine, run13)
        plan = build_sampling_plan(
            engine,
            run13,
            args.draws,
            SPLIT_IDS[args.split],
            stage_fractions,
            source_fractions,
            args.seed,
            replace=args.sampling_mode == "with-replacement",
            source_counts=source_counts,
            joint_counts=joint_counts,
        )
    payload = {
        "schema_version": np.asarray([SCHEMA_VERSION], dtype=np.int16),
        "seed": np.asarray([args.seed], dtype=np.int64),
        "corpus_ids": plan["corpus_ids"],
        "position_indices": plan["position_indices"],
        "source_ids": plan["source_ids"],
        "stage_ids": plan["stage_ids"],
        "split_ids": plan["split_ids"],
    }
    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    temporary = output_path + ".tmp.npz"
    np.savez_compressed(temporary, **payload)
    os.replace(temporary, output_path)
    pairs = np.stack((plan["corpus_ids"], plan["position_indices"]), axis=1)
    _, repeat_counts = np.unique(pairs, axis=0, return_counts=True)
    effective_sample_size = float(
        repeat_counts.sum() ** 2 / np.sum(repeat_counts.astype(np.float64) ** 2)
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "output": output_path,
        "engine_corpus": os.path.abspath(args.engine_corpus),
        "run13_component": os.path.abspath(args.run13_component),
        "draws": int(args.draws),
        "split": args.split,
        "seed": int(args.seed),
        "sampling_mode": args.sampling_mode,
        "target_stage_fractions": dict(zip(STAGE_NAMES, stage_fractions)),
        "target_source_fractions": dict(zip(SOURCE_NAMES, source_fractions)),
        "target_source_counts": {
            name: int(np.sum(plan["source_ids"] == index))
            for index, name in enumerate(SOURCE_NAMES)
        },
        "draws_by_stage": {
            name: int(np.sum(plan["stage_ids"] == index))
            for index, name in enumerate(STAGE_NAMES)
        },
        "draws_by_source": {
            name: int(np.sum(plan["source_ids"] == index))
            for index, name in enumerate(SOURCE_NAMES)
        },
        "joint_quotas": {
            SOURCE_NAMES[source]: {
                STAGE_NAMES[stage]: int(plan["joint_quotas"][source, stage])
                for stage in range(3)
            }
            for source in range(3)
        },
        "available_by_stratum": {
            SOURCE_NAMES[source]: {
                STAGE_NAMES[stage]: int(plan["available_by_stratum"][source, stage])
                for stage in range(3)
            }
            for source in range(3)
        },
        "unique_corpus_positions_drawn": int(len(np.unique(pairs, axis=0))),
        "unique_fraction": float(len(repeat_counts) / len(pairs)),
        "maximum_position_repetitions": int(repeat_counts.max()),
        "repeat_effective_sample_size": effective_sample_size,
        "cross_corpus_d4_overlaps": int(overlaps),
        "elapsed_seconds": time.perf_counter() - started,
        "output_bytes": os.path.getsize(output_path),
    }
    report_path = args.report_out or output_path + ".report.json"
    temporary_report = report_path + ".tmp"
    with open(temporary_report, "w") as output:
        json.dump(report, output, indent=2, sort_keys=True)
    os.replace(temporary_report, report_path)
    return report


def main():
    print(json.dumps(build(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
