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
        "--stage-fractions", type=float, nargs=3, default=(0.20, 0.35, 0.45),
        metavar=("EARLY", "MIDDLE", "LATE"),
    )
    parser.add_argument(
        "--source-fractions", type=float, nargs=3, default=(0.70, 0.20, 0.10),
        metavar=("MAIN", "SUBGAME", "RUN13"),
    )
    return parser.parse_args()


def build(args):
    started = time.perf_counter()
    with np.load(args.engine_corpus, allow_pickle=False) as engine, np.load(
        args.run13_component, allow_pickle=False
    ) as run13:
        overlaps = validate_no_cross_corpus_leakage(engine, run13)
        plan = build_sampling_plan(
            engine,
            run13,
            args.draws,
            SPLIT_IDS[args.split],
            args.stage_fractions,
            args.source_fractions,
            args.seed,
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
    report = {
        "schema_version": SCHEMA_VERSION,
        "output": output_path,
        "engine_corpus": os.path.abspath(args.engine_corpus),
        "run13_component": os.path.abspath(args.run13_component),
        "draws": int(args.draws),
        "split": args.split,
        "seed": int(args.seed),
        "target_stage_fractions": dict(zip(STAGE_NAMES, args.stage_fractions)),
        "target_source_fractions": dict(zip(SOURCE_NAMES, args.source_fractions)),
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
        "unique_corpus_positions_drawn": int(len(np.unique(pairs, axis=0))),
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
