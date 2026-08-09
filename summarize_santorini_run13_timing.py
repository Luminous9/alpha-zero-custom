"""Combine ordinary and milestone Run13 profiles into an amortized baseline."""

import argparse
import json
import os

from benchmark_santorini_run13_timing import TIMING_FIELDS, write_json_atomic


def _normalized_workload_command(command):
    """Remove the two profile-specific arguments before workload comparison."""
    normalized = []
    index = 0
    while index < len(command):
        if command[index] in ("--checkpoint", "--milestone-interval"):
            index += 2
            continue
        normalized.append(command[index])
        index += 1
    return normalized


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ordinary", required=True)
    parser.add_argument("--milestone", required=True)
    parser.add_argument("--milestone-interval", type=int, default=20)
    parser.add_argument("--json-out", required=True)
    return parser.parse_args()


def combine_profiles(ordinary, milestone, milestone_interval):
    milestone_interval = int(milestone_interval)
    if milestone_interval < 1:
        raise ValueError("Milestone interval must be positive.")
    if ordinary.get("profile") != "ordinary" or milestone.get("profile") != "milestone":
        raise ValueError("Timing summaries have the wrong profile types.")
    if ordinary.get("smoke") or milestone.get("smoke"):
        raise ValueError("Smoke profiles cannot establish the representative baseline.")
    for key in ("games", "num_mcts_sims"):
        if ordinary[key] != milestone[key]:
            raise ValueError("Timing profiles disagree on {}.".format(key))
    if ordinary["hardware"] != milestone["hardware"]:
        raise ValueError("Timing profiles were not measured on identical hardware.")
    for key in ("iteration", "source", "training_steps"):
        if key in ordinary and key in milestone and ordinary[key] != milestone[key]:
            raise ValueError("Timing profiles disagree on {}.".format(key))
    if "command" in ordinary and "command" in milestone:
        if _normalized_workload_command(ordinary["command"]) != _normalized_workload_command(
            milestone["command"]
        ):
            raise ValueError("Timing profiles were measured with different workloads.")

    milestone_weight = 1.0 / milestone_interval
    ordinary_weight = 1.0 - milestone_weight
    phases = {}
    total = 0.0
    for phase in TIMING_FIELDS:
        seconds = (
            ordinary_weight * ordinary["phases"][phase]["seconds"]
            + milestone_weight * milestone["phases"][phase]["seconds"]
        )
        phases[phase] = {"seconds": seconds}
        total += seconds
    for phase in TIMING_FIELDS:
        phases[phase]["fraction"] = phases[phase]["seconds"] / total
    return {
        "schema_version": 1,
        "type": "amortized_run13_baseline",
        "milestone_interval": milestone_interval,
        "ordinary_weight": ordinary_weight,
        "milestone_weight": milestone_weight,
        "games": ordinary["games"],
        "num_mcts_sims": ordinary["num_mcts_sims"],
        "hardware": ordinary["hardware"],
        "wall_total_seconds": total,
        "phases": phases,
        "sources": {
            "ordinary": os.path.abspath(ordinary["output"]),
            "milestone": os.path.abspath(milestone["output"]),
        },
    }


def main():
    args = parse_args()
    with open(args.ordinary) as source:
        ordinary = json.load(source)
    with open(args.milestone) as source:
        milestone = json.load(source)
    result = combine_profiles(ordinary, milestone, args.milestone_interval)
    write_json_atomic(args.json_out, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
