"""Run one extracted P100 P2 smoke arm with a bundled Linux oracle."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time


INPUT_ROOT = Path("/kaggle/input")
WORKING_ROOT = Path("/kaggle/working/santorini_v4_p2_smoke")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm", choices=("ordinary", "mixed", "transition"), required=True
    )
    return parser.parse_args()


def _exactly_one(filename):
    matches = sorted(INPUT_ROOT.rglob(filename))
    if len(matches) != 1:
        raise RuntimeError(
            "Expected exactly one {} under {}, found: {}".format(
                filename, INPUT_ROOT, [str(path) for path in matches]
            )
        )
    return matches[0]


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    args = parse_args()
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("The P2 P100 smoke requires CUDA.")
    gpu_name = torch.cuda.get_device_name(0)
    if "P100" not in gpu_name.upper():
        raise RuntimeError(
            "The reference smoke requires a P100, but Kaggle provided {}.".format(
                gpu_name
            )
        )
    entry_point = _exactly_one("main_santorini.py")
    project_root = entry_point.parent
    manifest_path = _exactly_one("p2-smoke-manifest.json")
    manifest = json.loads(manifest_path.read_text())
    checkpoint = _exactly_one("p2-start.pth.tar")
    seam_suite = _exactly_one("v4-seam-telemetry-suite.npz")
    for path in (checkpoint, seam_suite):
        expected = manifest["inputs"][path.name]["sha256"]
        if _sha256(path) != expected:
            raise RuntimeError("Input digest changed: {}".format(path))

    output_dir = WORKING_ROOT / args.arm
    output_dir.mkdir(parents=True, exist_ok=True)
    build_seconds = 0.0
    oracle_binary = None
    if args.arm != "ordinary":
        bundled_binary = manifest_path.parent / "oracle-build" / "santorini-oracle-linux-x86_64"
        expected_binary = manifest.get("oracle_build", {}).get("linux_binary_sha256")
        if not bundled_binary.is_file():
            raise FileNotFoundError("Bundled Linux oracle is missing: {}".format(bundled_binary))
        if not expected_binary or _sha256(bundled_binary) != expected_binary:
            raise RuntimeError("Bundled Linux oracle digest changed.")
        oracle_binary = Path("/kaggle/working/santorini-oracle-linux-x86_64")
        shutil.copy2(bundled_binary, oracle_binary)
        oracle_binary.chmod(0o755)

    sparring_probability = "0" if args.arm == "ordinary" else "0.10"
    oracle_nodes = 5_000 if args.arm == "transition" else 100_000
    oracle_ladder_version = 2 if args.arm == "transition" else 1
    replay_reuse_warmup_iters = 8 if args.arm == "transition" else 0
    command = [
        sys.executable, str(entry_point),
        "--architecture", "v4",
        "--training-mode", "latest",
        "--num-iters", "1",
        "--num-eps", "240",
        "--num-mcts-sims", "96",
        "--search-mode", "gumbel",
        "--gumbel-max-considered-actions", "16",
        "--gumbel-scale", "1.0",
        "--gumbel-placement-scale", "1.5",
        "--placement-scale-exploration-probability", "0.10",
        "--placement-exploration-gumbel-scale", "2.25",
        "--playout-cap-randomization",
        "--playout-cap-full-probability", "0.25",
        "--playout-cap-fast-sims", "32",
        "--self-play-batch-size", "128",
        "--batch-size", "256",
        "--replay-reuse", "16",
        "--replay-reuse-warmup-iters", str(replay_reuse_warmup_iters),
        "--validation-fraction", "0.05",
        "--optimizer", "adamw",
        "--learning-rate", "0.0003",
        "--weight-decay", "0.0001",
        "--lr-schedule", "200:0.0001,400:0.00003",
        "--history-iters", "20",
        "--maxlen-of-queue", "200000",
        "--load-model",
        "--load-folder", str(checkpoint.parent),
        "--load-file", checkpoint.name,
        "--checkpoint", str(output_dir),
        "--oracle-sparring-probability", sparring_probability,
        "--oracle-sparring-nodes", str(oracle_nodes),
        "--oracle-sparring-workers", "4",
        "--oracle-sparring-opening-seed", "20260921",
        "--oracle-sparring-ladder-version", str(oracle_ladder_version),
        "--oracle-sparring-ratchet-games", "80",
        "--oracle-sparring-ratchet-score", "0.55",
        "--v4-seam-telemetry-suite", str(seam_suite),
        "--v4-seam-telemetry-interval", "1",
        "--v4-seam-telemetry-batch-size", "256",
        "--v4-teacher-objective-step-threshold", "0.05",
        "--milestone-interval", "20",
        "--no-telemetry-matches",
        "--seed", "20260930",
    ]
    if oracle_binary is not None:
        command.extend(("--oracle-binary", str(oracle_binary)))
    print("P2 smoke arm:", args.arm, flush=True)
    print("Running:", " ".join(command), flush=True)
    started = time.perf_counter()
    subprocess.run(command, cwd=project_root, check=True)
    elapsed_seconds = time.perf_counter() - started

    telemetry_path = output_dir / "telemetry" / "telemetry.jsonl"
    rows = [json.loads(line) for line in telemetry_path.read_text().splitlines() if line]
    if len(rows) != 1:
        raise RuntimeError("Expected one telemetry row, found {}.".format(len(rows)))
    telemetry = rows[0]
    expected_sparring_games = 0 if args.arm == "ordinary" else 24
    if telemetry.get("games") != 240:
        raise RuntimeError("P2 smoke did not complete 240 games.")
    if telemetry.get("oracle_sparring_games") != expected_sparring_games:
        raise RuntimeError("Unexpected completed sparring-game count.")
    if args.arm != "ordinary" and telemetry.get(
        "oracle_sparring_complete_pairs"
    ) != 12:
        raise RuntimeError("Sparring telemetry is missing complete opening pairs.")
    if args.arm == "transition":
        if abs(float(telemetry.get("target_replay_reuse", -1)) - 2.0) > 1e-9:
            raise RuntimeError("Transition smoke did not apply 2x iteration-one reuse.")
        if telemetry.get("replay_reuse_warmup_iters") != 8:
            raise RuntimeError("Transition smoke did not retain the warm-up contract.")
    if telemetry.get("v4_seam_telemetry_due") is not True:
        raise RuntimeError("Frozen V4 seam telemetry did not run.")
    if telemetry.get("v4_teacher_objective_gate_triggered") is not False:
        raise RuntimeError("Frozen teacher-objective step gate did not pass.")
    if telemetry.get("oracle_sparring_ratchet_games") != 80:
        raise RuntimeError("Oracle ratchet window changed.")
    if abs(float(telemetry.get("oracle_sparring_ratchet_score_threshold", -1)) - 0.55) > 1e-9:
        raise RuntimeError("Oracle ratchet threshold changed.")
    required_outputs = (
        output_dir / "latest-training.pth.tar",
        output_dir / "latest.pth.tar",
        output_dir / "latest.examples.npz",
    )
    for path in required_outputs:
        if not path.is_file():
            raise FileNotFoundError(path)
    from santorini.ReplayBuffer import load_compact_replay, replay_metadata
    replay = [
        example
        for window in load_compact_replay(output_dir / "latest.examples.npz")
        for example in window
    ]
    sparring_metadata = [
        replay_metadata(example)
        for example in replay
        if replay_metadata(example).get("source") == "oracle_sparring"
    ]
    if args.arm != "ordinary":
        if not sparring_metadata:
            raise RuntimeError("Sparring replay contains no oracle-sparring records.")
        if {item.get("neural_seat") for item in sparring_metadata} != {-1, 1}:
            raise RuntimeError("Sparring replay does not contain both neural seats.")
        if {item.get("oracle_nodes") for item in sparring_metadata} != {oracle_nodes}:
            raise RuntimeError("Sparring replay contains the wrong oracle rung.")
        if {item.get("oracle_ladder_version") for item in sparring_metadata} != {
            oracle_ladder_version
        }:
            raise RuntimeError("Sparring replay contains the wrong oracle ladder version.")
    elif sparring_metadata:
        raise RuntimeError("Ordinary replay unexpectedly contains sparring records.")
    contract = {
        "schema_version": 1,
        "contract": "santorini_v4_p2_p100_end_to_end_smoke",
        "arm": args.arm,
        "gpu": gpu_name,
        "elapsed_seconds": elapsed_seconds,
        "oracle_build_seconds": build_seconds,
        "checkpoint_sha256": _sha256(checkpoint),
        "seam_suite_sha256": _sha256(seam_suite),
        "oracle_binary_sha256": _sha256(oracle_binary) if oracle_binary else None,
        "telemetry": telemetry,
        "replay_source_counts": {
            source: sum(
                replay_metadata(example).get("source") == source
                for example in replay
            )
            for source in ("self_play", "oracle_sparring")
        },
        "outputs": {
            path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in required_outputs
        },
        "disagreement_starts_enabled": False,
        "auxiliary_oracle_head_enabled": False,
        "final_test_touched": False,
        "final_arena_seeds_touched": False,
    }
    (output_dir / "p2-smoke-contract.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n"
    )
    print("Outputs:", sorted(path.name for path in output_dir.iterdir()), flush=True)


if __name__ == "__main__":
    main()
