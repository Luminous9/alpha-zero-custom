"""Build the frozen coverage-balanced P1c pretraining epoch.

Every unique engine-train and Run13-standard-train position appears at least
once.  Repeats restore the frozen 20/35/45 standard-stage mix and 77/22/1
standard-source mix.  A separate placement bucket matches Run13's observed
self-play phase fraction and balances the four sequential placement decisions.
"""

import argparse
import json
import math
import os

import numpy as np

from santorini.OracleResearch import file_sha256
from santorini.V4BootstrapCorpus import (
    SOURCE_NAMES,
    STAGE_NAMES,
    largest_remainder_quotas,
    validate_no_cross_corpus_leakage,
)


SCHEMA_VERSION = 1
DEFAULT_STAGE_FRACTIONS = (0.20, 0.35, 0.45)
DEFAULT_SOURCE_FRACTIONS = (0.77, 0.22, 0.01)
DEFAULT_PLACEMENT_FRACTION = 19_200 / 57_909


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-corpus", required=True)
    parser.add_argument("--run13-component", required=True)
    parser.add_argument("--placement-component", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report-out")
    parser.add_argument(
        "--stage-fractions", type=float, nargs=3, default=DEFAULT_STAGE_FRACTIONS
    )
    parser.add_argument(
        "--source-fractions", type=float, nargs=3, default=DEFAULT_SOURCE_FRACTIONS
    )
    parser.add_argument(
        "--placement-fraction", type=float, default=DEFAULT_PLACEMENT_FRACTION
    )
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument(
        "--allow-incomplete-placement-coverage",
        action="store_true",
        help="Testing only: permit fewer than the expected 960 placement orbits.",
    )
    return parser.parse_args()


def _load(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def constrained_joint_quotas(base, row_targets, column_targets):
    """Complete a base 3x3 coverage table to fixed row/column marginals."""
    base = np.asarray(base, dtype=np.int64)
    row_targets = np.asarray(row_targets, dtype=np.int64)
    column_targets = np.asarray(column_targets, dtype=np.int64)
    if base.shape != (3, 3):
        raise ValueError("Base coverage counts must be 3x3.")
    if int(row_targets.sum()) != int(column_targets.sum()):
        raise ValueError("Source and stage targets must have the same total.")
    row_deficits = row_targets - base.sum(axis=1)
    column_deficits = column_targets - base.sum(axis=0)
    if np.any(row_deficits < 0) or np.any(column_deficits < 0):
        raise ValueError("Requested marginals cannot contain the unique coverage base.")
    additions = np.zeros((3, 3), dtype=np.int64)
    remaining_columns = column_deficits.copy()
    for source_id in range(2):
        quota = int(row_deficits[source_id])
        if quota == 0:
            continue
        probabilities = remaining_columns / remaining_columns.sum()
        row = largest_remainder_quotas(quota, probabilities)
        # A row quota can exceed a small column's capacity after rounding.
        overflow = np.maximum(row - remaining_columns, 0)
        if np.any(overflow):
            row = np.minimum(row, remaining_columns)
            missing = quota - int(row.sum())
            capacity = remaining_columns - row
            while missing:
                eligible = np.flatnonzero(capacity > 0)
                take = min(missing, int(capacity[eligible].sum()))
                extra = largest_remainder_quotas(
                    take, capacity[eligible] / capacity[eligible].sum()
                )
                extra = np.minimum(extra, capacity[eligible])
                row[eligible] += extra
                capacity[eligible] -= extra
                missing = quota - int(row.sum())
        additions[source_id] = row
        remaining_columns -= row
    additions[2] = remaining_columns
    if int(additions[2].sum()) != int(row_deficits[2]):
        raise AssertionError("Could not reconcile constrained P1c marginals.")
    result = base + additions
    if not np.array_equal(result.sum(axis=1), row_targets):
        raise AssertionError("P1c source marginals were not preserved.")
    if not np.array_equal(result.sum(axis=0), column_targets):
        raise AssertionError("P1c stage marginals were not preserved.")
    return result


def _minimum_coverage_total(stage_counts, stage_fractions):
    total = max(
        int(math.ceil(int(count) / float(fraction)))
        for count, fraction in zip(stage_counts, stage_fractions)
    )
    while np.any(largest_remainder_quotas(total, stage_fractions) < stage_counts):
        total += 1
    return total


def _sample_repeats(rng, eligible, weights, count):
    if count == 0:
        return np.empty(0, dtype=np.int32)
    if not len(eligible):
        raise ValueError("A P1c repeat stratum has no eligible positions.")
    weights = np.asarray(weights, dtype=np.float64)
    return rng.choice(
        eligible,
        size=int(count),
        replace=True,
        p=weights / weights.sum(),
    ).astype(np.int32)


def validate_placement_outcomes(placement):
    if "has_completed_outcomes" not in placement or not bool(
        np.asarray(placement["has_completed_outcomes"])[0]
    ):
        raise ValueError(
            "Placement component must declare real completed continuation outcomes."
        )
    winners = np.asarray(placement["winner_means"], dtype=np.float64)
    if np.any(~np.isfinite(winners)) or np.any(np.abs(winners) > 1.0 + 1e-6):
        raise ValueError("Placement completed outcomes must be finite values in [-1, 1].")


def build_plan(args):
    stage_fractions = np.asarray(args.stage_fractions, dtype=np.float64)
    source_fractions = np.asarray(args.source_fractions, dtype=np.float64)
    if np.any(stage_fractions <= 0) or not np.isclose(stage_fractions.sum(), 1.0):
        raise ValueError("Stage fractions must be positive and sum to one.")
    if np.any(source_fractions <= 0) or not np.isclose(source_fractions.sum(), 1.0):
        raise ValueError("Source fractions must be positive and sum to one.")
    if not 0.0 < args.placement_fraction < 1.0:
        raise ValueError("Placement fraction must be in (0, 1).")

    engine = _load(args.engine_corpus)
    run13 = _load(args.run13_component)
    placement = _load(args.placement_component)
    validate_placement_outcomes(placement)
    cross_corpus_overlaps = validate_no_cross_corpus_leakage(engine, run13)
    engine_indices = np.flatnonzero(engine["split_ids"] == 0).astype(np.int32)
    run13_indices = np.flatnonzero(run13["split_ids"] == 0).astype(np.int32)
    if len(engine_indices) != len(engine["boards"]):
        raise ValueError("The anchored P1c engine component must be train-only.")
    if np.any(placement["stage_ids"] != -1) or np.any(placement["split_ids"] != 0):
        raise ValueError("Placement component must use stage -1 and train split 0.")
    worker_counts = placement["worker_counts"].astype(np.int64)
    placement_unique_counts = np.asarray([
        np.sum(worker_counts == worker_count) for worker_count in range(4)
    ], dtype=np.int64)
    expected_placement_counts = np.asarray((1, 6, 49, 904), dtype=np.int64)
    if (
        not args.allow_incomplete_placement_coverage
        and not np.array_equal(placement_unique_counts, expected_placement_counts)
    ):
        raise ValueError(
            "Placement component coverage is {} instead of {}.".format(
                placement_unique_counts.tolist(), expected_placement_counts.tolist()
            )
        )

    primary_engine_sources = np.argmax(engine["source_counts"], axis=1).astype(np.int8)
    base = np.zeros((3, 3), dtype=np.int64)
    for source_id in range(2):
        for stage_id in range(3):
            base[source_id, stage_id] = int(np.sum(
                (primary_engine_sources == source_id)
                & (engine["stage_ids"] == stage_id)
            ))
    for stage_id in range(3):
        base[2, stage_id] = int(np.sum(
            (run13["stage_ids"] == stage_id) & (run13["split_ids"] == 0)
        ))

    standard_draws = _minimum_coverage_total(base.sum(axis=0), stage_fractions)
    stage_targets = largest_remainder_quotas(standard_draws, stage_fractions)
    source_targets = largest_remainder_quotas(standard_draws, source_fractions)
    joint_targets = constrained_joint_quotas(base, source_targets, stage_targets)
    placement_draws = int(round(
        standard_draws * float(args.placement_fraction) / (1.0 - args.placement_fraction)
    ))
    placement_worker_quotas = largest_remainder_quotas(
        placement_draws, np.full(4, 0.25, dtype=np.float64)
    )
    total_draws = standard_draws + placement_draws
    corpus_ids = np.empty(total_draws, dtype=np.int8)
    position_indices = np.empty(total_draws, dtype=np.int32)
    source_ids = np.empty(total_draws, dtype=np.int8)
    stage_ids = np.empty(total_draws, dtype=np.int8)
    split_ids = np.zeros(total_draws, dtype=np.int8)
    rng = np.random.RandomState(int(args.seed))
    offset = 0

    # Unique coverage base.
    count = len(engine_indices)
    corpus_ids[offset:offset + count] = 0
    position_indices[offset:offset + count] = engine_indices
    source_ids[offset:offset + count] = primary_engine_sources[engine_indices]
    stage_ids[offset:offset + count] = engine["stage_ids"][engine_indices]
    offset += count
    count = len(run13_indices)
    corpus_ids[offset:offset + count] = 1
    position_indices[offset:offset + count] = run13_indices
    source_ids[offset:offset + count] = 2
    stage_ids[offset:offset + count] = run13["stage_ids"][run13_indices]
    offset += count

    # Standard repeats needed to reach both frozen marginals.
    additions = joint_targets - base
    for source_id in range(3):
        for stage_id in range(3):
            count = int(additions[source_id, stage_id])
            if source_id < 2:
                eligible = np.flatnonzero(
                    (primary_engine_sources == source_id)
                    & (engine["stage_ids"] == stage_id)
                ).astype(np.int32)
                weights = engine["source_counts"][eligible, source_id]
                corpus_id = 0
            else:
                eligible = np.flatnonzero(
                    (run13["stage_ids"] == stage_id)
                    & (run13["split_ids"] == 0)
                ).astype(np.int32)
                weights = run13["observation_counts"][eligible]
                corpus_id = 1
            chosen = _sample_repeats(rng, eligible, weights, count)
            corpus_ids[offset:offset + count] = corpus_id
            position_indices[offset:offset + count] = chosen
            source_ids[offset:offset + count] = source_id
            stage_ids[offset:offset + count] = stage_id
            offset += count

    # Placement repeats are independently balanced over decisions 1-4.
    for worker_count, count in enumerate(placement_worker_quotas):
        count = int(count)
        eligible = np.flatnonzero(worker_counts == worker_count).astype(np.int32)
        chosen = _sample_repeats(
            rng, eligible, placement["observation_counts"][eligible], count
        )
        corpus_ids[offset:offset + count] = 2
        position_indices[offset:offset + count] = chosen
        source_ids[offset:offset + count] = 2
        stage_ids[offset:offset + count] = -1
        offset += count
    if offset != total_draws:
        raise AssertionError("P1c plan materialization produced the wrong length.")

    order = rng.permutation(total_draws)
    payload = {
        "schema_version": np.asarray([SCHEMA_VERSION], dtype=np.int16),
        "corpus_ids": corpus_ids[order],
        "position_indices": position_indices[order],
        "source_ids": source_ids[order],
        "stage_ids": stage_ids[order],
        "split_ids": split_ids,
        "standard_joint_quotas": joint_targets,
        "standard_unique_base": base,
        "placement_worker_quotas": placement_worker_quotas,
        "sampling_with_replacement": np.asarray([True]),
    }
    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    temporary = output_path + ".tmp.npz"
    np.savez_compressed(temporary, **payload)
    os.replace(temporary, output_path)
    report = {
        "schema_version": SCHEMA_VERSION,
        "contract": "p1c_coverage_balanced_epoch",
        "output": output_path,
        "output_sha256": file_sha256(output_path),
        "inputs": {
            "engine_corpus": os.path.abspath(args.engine_corpus),
            "engine_corpus_sha256": file_sha256(args.engine_corpus),
            "run13_component": os.path.abspath(args.run13_component),
            "run13_component_sha256": file_sha256(args.run13_component),
            "placement_component": os.path.abspath(args.placement_component),
            "placement_component_sha256": file_sha256(args.placement_component),
        },
        "seed": int(args.seed),
        "standard_draws": standard_draws,
        "placement_draws": placement_draws,
        "total_draws": total_draws,
        "placement_fraction": placement_draws / total_draws,
        "declared_placement_fraction": float(args.placement_fraction),
        "standard_stage_fractions": stage_fractions.tolist(),
        "standard_source_fractions": source_fractions.tolist(),
        "standard_unique_base": base.tolist(),
        "standard_joint_quotas": joint_targets.tolist(),
        "placement_unique_by_worker_count": placement_unique_counts.tolist(),
        "placement_draws_by_worker_count": placement_worker_quotas.tolist(),
        "all_engine_train_positions_covered": True,
        "all_run13_standard_train_positions_covered": True,
        "cross_corpus_d4_overlaps": int(cross_corpus_overlaps),
        "selection_basis": (
            "frozen standard holdout only; placement roots are shared by all games, "
            "so placement is validated by coverage diagnostics and the full-game G1 arena"
        ),
        "final_test_touched": False,
        "output_bytes": os.path.getsize(output_path),
    }
    report_path = os.path.abspath(args.report_out or output_path + ".report.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    temporary_report = report_path + ".tmp"
    with open(temporary_report, "w") as output:
        json.dump(report, output, indent=2, sort_keys=True)
    os.replace(temporary_report, report_path)
    return report


def main():
    print(json.dumps(build_plan(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
