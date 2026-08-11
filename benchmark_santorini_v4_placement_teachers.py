"""Compare placement teachers with paired seats and common oracle continuation."""

import argparse
from collections import defaultdict
from itertools import combinations
import json
import os
import time

from santorini.OracleResearch import file_sha256
from santorini.SantoriniGame import SantoriniGame
from santorini.SantoriniOracle import SantoriniOracleProcess
from santorini.V4PlacementTournament import (
    PlacementPolicyTeacher,
    paired_placement_block,
    summarize_paired_records,
)


SCHEMA_VERSION = 1


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--teacher", action="append", required=True, metavar="NAME=COMPONENT"
    )
    parser.add_argument("--blocks", type=int, default=64)
    parser.add_argument(
        "--modes", nargs="+", choices=("sampled", "greedy"),
        default=("sampled", "greedy"),
    )
    parser.add_argument("--oracle-nodes", type=int, default=20_000)
    parser.add_argument("--oracle-binary")
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--max-pairings", type=int, help="Testing only.")
    parser.add_argument("--records-out", required=True)
    parser.add_argument("--json-out", required=True)
    return parser.parse_args()


def _teacher_specs(items):
    specs = []
    for item in items:
        if "=" not in item:
            raise ValueError("--teacher must use NAME=COMPONENT syntax.")
        name, path = item.split("=", 1)
        name = name.strip()
        path = os.path.abspath(path)
        if not name or not os.path.isfile(path):
            raise ValueError("Invalid placement teacher: {}".format(item))
        specs.append((name, path))
    names = [name for name, _ in specs]
    if len(names) < 2 or len(names) != len(set(names)):
        raise ValueError("Placement tournament needs at least two unique teachers.")
    return specs


def _metadata(args, teachers, oracle, pairings):
    return {
        "type": "metadata",
        "schema_version": SCHEMA_VERSION,
        "contract": "santorini_v4_placement_only_teacher_tournament",
        "teachers": [
            {
                "name": teacher.name,
                "path": os.path.abspath(teacher.path),
                "sha256": teacher.sha256,
                "coverage": list(teacher.coverage),
            }
            for teacher in teachers
        ],
        "pairings": [list(pairing) for pairing in pairings],
        "modes": list(args.modes),
        "sampled_blocks": int(args.blocks),
        "greedy_blocks": 1,
        "oracle_nodes": int(args.oracle_nodes),
        "oracle_binary": str(oracle.binary_path.resolve()),
        "oracle_sha256": file_sha256(oracle.binary_path),
        "tt_policy": "reset_per_completed_opening",
        "seed": int(args.seed),
    }


def _load_or_initialize(path, metadata):
    path = os.path.abspath(path)
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as output:
            output.write(json.dumps(metadata, sort_keys=True) + "\n")
        return {}
    with open(path) as source:
        lines = [json.loads(line) for line in source if line.strip()]
    if not lines or lines[0] != metadata:
        raise ValueError("Placement tournament metadata changed; use new records.")
    records = {}
    for record in lines[1:]:
        key = (
            record["teacher_a"], record["teacher_b"],
            record["mode"], int(record["block_id"]),
        )
        if key in records:
            raise ValueError("Duplicate placement tournament block record.")
        records[key] = record
    return records


def _tasks(args, pairings):
    tasks = []
    for teacher_a, teacher_b in pairings:
        for mode in args.modes:
            blocks = int(args.blocks) if mode == "sampled" else 1
            tasks.extend(
                (teacher_a, teacher_b, mode, block_id)
                for block_id in range(blocks)
            )
    return tasks


def _summaries(args, records, pairings):
    matchups = []
    ranking = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))
    for teacher_a, teacher_b in pairings:
        for mode in args.modes:
            selected = [
                record for key, record in records.items()
                if key[0] == teacher_a and key[1] == teacher_b and key[2] == mode
            ]
            expected = int(args.blocks) if mode == "sampled" else 1
            if len(selected) != expected:
                continue
            selected.sort(key=lambda item: int(item["block_id"]))
            summary = summarize_paired_records(
                selected,
                bootstrap_samples=args.bootstrap_samples,
                seed=args.seed + len(matchups),
            )
            summary.update({
                "teacher_a": teacher_a,
                "teacher_b": teacher_b,
                "mode": mode,
            })
            matchups.append(summary)
            ranking[mode][teacher_a][0] += summary["teacher_a_score"] * summary["games"]
            ranking[mode][teacher_a][1] += summary["games"]
            ranking[mode][teacher_b][0] += (1.0 - summary["teacher_a_score"]) * summary["games"]
            ranking[mode][teacher_b][1] += summary["games"]
    rankings = {}
    for mode, teachers in ranking.items():
        rankings[mode] = sorted(
            [
                {
                    "teacher": name,
                    "score": points / games,
                    "games": games,
                }
                for name, (points, games) in teachers.items()
            ],
            key=lambda item: (-item["score"], item["teacher"]),
        )
    return matchups, rankings


def run_tournament(args):
    if args.blocks < 1 or args.oracle_nodes < 1 or args.bootstrap_samples < 0:
        raise ValueError("Blocks/nodes must be positive and bootstrap samples nonnegative.")
    specs = _teacher_specs(args.teacher)
    game = SantoriniGame(5, sequential_placement=True)
    teachers = [
        PlacementPolicyTeacher(game, name, path) for name, path in specs
    ]
    by_name = {teacher.name: teacher for teacher in teachers}
    pairings = list(combinations([teacher.name for teacher in teachers], 2))
    if args.max_pairings is not None:
        if args.max_pairings < 1:
            raise ValueError("--max-pairings must be positive.")
        pairings = pairings[: int(args.max_pairings)]
    started = time.perf_counter()
    with SantoriniOracleProcess(args.oracle_binary) as oracle:
        metadata = _metadata(args, teachers, oracle, pairings)
        records = _load_or_initialize(args.records_out, metadata)
        tasks = _tasks(args, pairings)
        with open(args.records_out, "a") as output:
            for task_index, (name_a, name_b, mode, block_id) in enumerate(tasks):
                key = (name_a, name_b, mode, int(block_id))
                if key in records:
                    continue
                block = paired_placement_block(
                    game,
                    by_name[name_a],
                    by_name[name_b],
                    mode,
                    block_id,
                    args.seed,
                    oracle,
                    args.oracle_nodes,
                )
                block["type"] = "paired_block"
                block["schema_version"] = SCHEMA_VERSION
                output.write(json.dumps(block, sort_keys=True) + "\n")
                output.flush()
                records[key] = block
                print(
                    "block {}/{}: {} vs {} {} #{} -> {}/{}".format(
                        task_index + 1, len(tasks), name_a, name_b, mode,
                        block_id,
                        block["a_as_p1"]["result"],
                        block["b_as_p1"]["result"],
                    ),
                    flush=True,
                )
    matchups, rankings = _summaries(args, records, pairings)
    report = {
        **metadata,
        "type": "summary",
        "records": os.path.abspath(args.records_out),
        "records_sha256": file_sha256(args.records_out),
        "completed_blocks": len(records),
        "expected_blocks": len(_tasks(args, pairings)),
        "complete": len(records) == len(_tasks(args, pairings)),
        "elapsed_seconds_this_run": time.perf_counter() - started,
        "matchups": matchups,
        "rankings": rankings,
        "selection_scope": (
            "placement actions only; both sides use the same deterministic "
            "santorini-ai continuation after setup"
        ),
        "final_test_touched": False,
    }
    output_path = os.path.abspath(args.json_out)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    temporary = output_path + ".tmp"
    with open(temporary, "w") as output:
        json.dump(report, output, indent=2, sort_keys=True)
    os.replace(temporary, output_path)
    return report


def main():
    print(json.dumps(run_tournament(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
