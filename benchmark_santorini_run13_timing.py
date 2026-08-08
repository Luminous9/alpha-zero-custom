"""Run and summarize a representative instrumented Run13 iteration."""

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import torch


DEFAULT_SOURCE = "./temp/santorini_v3_run13_gumbel"
TIMING_FIELDS = (
    "self_play",
    "training",
    "arena_telemetry",
    "serialization",
    "other",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Resume Run13 into an isolated folder and capture phase wall time."
    )
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--output", required=True)
    parser.add_argument("--profile", choices=("ordinary", "milestone"), default="ordinary")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--json-out")
    return parser.parse_args()


def run13_command(args):
    source = os.path.abspath(args.source)
    output = os.path.abspath(args.output)
    games = 2 if args.smoke else 240
    simulations = 4 if args.smoke else 96
    train_steps = 1 if args.smoke else 1_500
    fast_simulations = 1 if args.smoke else 32
    command = [
        sys.executable,
        "main_santorini.py",
        "--preset", "local",
        "--architecture", "v3",
        "--training-mode", "latest",
        "--num-iters", "1",
        "--num-eps", str(games),
        "--num-mcts-sims", str(simulations),
        "--search-mode", "gumbel",
        "--gumbel-max-considered-actions", "16",
        "--gumbel-scale", "1.0",
        "--gumbel-placement-scale", "1.5",
        "--placement-scale-exploration-probability", "0.1",
        "--placement-exploration-gumbel-scale", "2.25",
        "--evaluation-gumbel-scale", "0.0",
        "--evaluation-gumbel-placement-scale", "1.0",
        "--playout-cap-randomization",
        "--playout-cap-full-probability", "0.25",
        "--playout-cap-fast-sims", str(fast_simulations),
        "--history-iters", "20",
        "--replay-reuse", "16.0",
        "--validation-fraction", "0.05",
        "--epochs", "3",
        "--batch-size", "512",
        "--optimizer", "adamw",
        "--learning-rate", "0.0003",
        "--weight-decay", "0.0001",
        "--lr-schedule", "200:0.0001,400:0.00003",
        "--symmetry-consistency-fraction", "0.25",
        "--symmetry-consistency-policy-weight", "0.1",
        "--symmetry-consistency-value-weight", "0.1",
        "--symmetry-augmentation", "on-the-fly",
        "--root-symmetry-samples", "2",
        "--placement-root-symmetry-samples", "8",
        "--evaluation-root-symmetry-samples", "8",
        "--evaluation-placement-root-symmetry-samples", "8",
        "--inference-cache-size", "4096",
        "--self-play-batch-size", "128",
        "--compact-replay",
        "--policy-target-temperature", "1.0",
        "--placement-temperature", "1.0",
        "--dirichlet-alpha", "0.30",
        "--dirichlet-epsilon", "0.25",
        "--telemetry-match-games", "40",
        "--telemetry-placement-games", "40",
        "--telemetry-placement-temperature", "1.0",
        "--telemetry-opening-seed", "20260715",
        "--max-train-steps", str(train_steps),
        "--checkpoint", output,
        "--load-folder", source,
        "--load-file", "latest-training.pth.tar",
        "--load-model",
        "--load-examples",
        "--examples-file", "latest.examples.npz",
        "--keep-loaded-examples",
        "--checkpoint-examples-to-keep", "0",
        "--seed", str(args.seed),
        "--quiet",
    ]
    if args.profile == "milestone":
        command.extend(["--milestone-interval", "1"])
    elif args.smoke:
        command.append("--no-telemetry-matches")
        command.extend(["--telemetry-sample-size", "0"])
        command.extend(["--symmetry-telemetry-sample-size", "0"])
    return command


def read_single_telemetry(output):
    path = Path(output) / "telemetry" / "telemetry.jsonl"
    with path.open() as source:
        rows = [json.loads(line) for line in source if line.strip()]
    if len(rows) != 1:
        raise ValueError("Expected exactly one telemetry row, found {}.".format(len(rows)))
    row = rows[0]
    if int(row.get("wall_timing_schema_version", 0)) != 1:
        raise ValueError("Run13 telemetry is missing the phase timing schema.")
    return row


def timing_summary(args, row, command, elapsed):
    phases = {
        phase: {
            "seconds": float(row["wall_{}_seconds".format(phase)]),
            "fraction": float(row["wall_{}_fraction".format(phase)]),
        }
        for phase in TIMING_FIELDS
    }
    return {
        "schema_version": 1,
        "profile": args.profile,
        "smoke": bool(args.smoke),
        "source": os.path.abspath(args.source),
        "output": os.path.abspath(args.output),
        "command": command,
        "hardware": {
            "platform": platform.platform(),
            "torch_version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
        },
        "process_elapsed_seconds": float(elapsed),
        "iteration": int(row["iteration"]),
        "games": int(row["games"]),
        "num_mcts_sims": int(row["num_mcts_sims"]),
        "training_steps": int(row.get("training_steps", 0)),
        "wall_total_seconds": float(row["wall_total_seconds"]),
        "phases": phases,
    }


def write_json_atomic(path, payload):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w") as output:
        json.dump(payload, output, indent=2, sort_keys=True)
    os.replace(temporary, path)


def main():
    args = parse_args()
    if not args.smoke and not args.allow_cpu and not torch.cuda.is_available():
        raise RuntimeError(
            "The representative profile requires CUDA; use --allow-cpu only for diagnostics."
        )
    source = Path(args.source)
    for filename in ("latest-training.pth.tar", "latest.examples.npz"):
        if not (source / filename).is_file():
            raise FileNotFoundError("Missing Run13 source artifact: {}".format(source / filename))
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Timing output directory must be new or empty: {}".format(output))

    command = run13_command(args)
    started = time.perf_counter()
    subprocess.run(command, cwd=Path(__file__).resolve().parent, check=True)
    elapsed = time.perf_counter() - started
    row = read_single_telemetry(args.output)
    summary = timing_summary(args, row, command, elapsed)
    json_out = args.json_out or str(Path(args.output) / "timing-summary.json")
    write_json_atomic(json_out, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
