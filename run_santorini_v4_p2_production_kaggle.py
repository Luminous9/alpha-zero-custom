"""Resume the accepted Santorini V4 P2 lineage on an extracted Kaggle bundle."""

import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time


INPUT_ROOT = Path("/kaggle/input")
WORKING_ROOT = Path("/kaggle/working/santorini_v4_p2")
CHECKPOINT_NAME = "p2-resume-training.pth.tar"
REPLAY_NAME = "p2-resume.examples.npz"
SEAM_SUITE_NAME = "v4-seam-telemetry-suite.npz"
MILESTONE_NAME = "p2-milestone-anchor.pth.tar"


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


def _close(left, right, tolerance=1e-8):
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)


def _validate_telemetry(rows, manifest, returncode):
    start_iteration = int(manifest["lineage"]["start_iteration"])
    end_iteration = int(manifest["lineage"]["end_iteration"])
    if not rows:
        raise RuntimeError("Production continuation wrote no telemetry rows.")
    expected_iterations = list(range(start_iteration + 1, int(rows[-1]["iteration"]) + 1))
    actual_iterations = [int(row["iteration"]) for row in rows]
    if actual_iterations != expected_iterations:
        raise RuntimeError("Telemetry iterations are not contiguous: {}".format(actual_iterations))
    if actual_iterations[-1] > end_iteration:
        raise RuntimeError("Continuation exceeded its declared end iteration.")

    previous_objective = float(manifest["lineage"]["resume_teacher_objective"])
    resume_pairs = len(manifest["lineage"]["resume_ratchet_pair_scores"])
    configured_reuse = float(manifest["protocol"]["replay_reuse"])
    warmup_iters = int(manifest["protocol"]["replay_reuse_warmup_iters"])
    for row_index, row in enumerate(rows, start=1):
        iteration = int(row["iteration"])
        if int(row.get("games", -1)) != 240:
            raise RuntimeError("Iteration {} did not complete 240 games.".format(iteration))
        if int(row.get("oracle_sparring_games", -1)) != 24:
            raise RuntimeError("Iteration {} has the wrong sparring-game count.".format(iteration))
        if int(row.get("oracle_sparring_complete_pairs", -1)) != 12:
            raise RuntimeError("Iteration {} lacks 12 complete sparring pairs.".format(iteration))
        if int(row.get("oracle_sparring_nodes", -1)) != 5_000:
            raise RuntimeError("Iteration {} changed the oracle rung.".format(iteration))
        if int(row.get("oracle_sparring_ladder_version", -1)) != 2:
            raise RuntimeError("Iteration {} changed the oracle ladder version.".format(iteration))
        expected_reuse = configured_reuse * min(1.0, iteration / warmup_iters)
        if not _close(row.get("target_replay_reuse", -1), expected_reuse):
            raise RuntimeError(
                "Iteration {} expected {}x target reuse, found {}.".format(
                    iteration, expected_reuse, row.get("target_replay_reuse")
                )
            )
        if int(row.get("replay_reuse_warmup_iters", -1)) != warmup_iters:
            raise RuntimeError("Iteration {} changed the replay warm-up.".format(iteration))
        if row.get("v4_seam_telemetry_due") is not True:
            raise RuntimeError("Iteration {} skipped frozen seam telemetry.".format(iteration))
        if row.get("v4_seam_suite_fingerprint") != manifest["inputs"][SEAM_SUITE_NAME]["sha256"]:
            raise RuntimeError("Iteration {} used the wrong seam suite.".format(iteration))
        if not _close(row.get("v4_teacher_objective_previous"), previous_objective):
            raise RuntimeError("Iteration {} broke the teacher-objective chain.".format(iteration))
        previous_objective = float(row["v4_teacher_objective_current"])
        if not _close(row.get("v4_teacher_objective_step_threshold", -1), 0.05):
            raise RuntimeError("Iteration {} changed the teacher gate.".format(iteration))
        expected_history = min(40, resume_pairs + 12 * row_index)
        if int(row.get("oracle_sparring_ratchet_history_pairs", -1)) != expected_history:
            raise RuntimeError("Iteration {} broke the ratchet history.".format(iteration))
        if int(row.get("oracle_sparring_ratchet_games", -1)) != 80:
            raise RuntimeError("Iteration {} changed the ratchet window.".format(iteration))
        if not _close(row.get("oracle_sparring_ratchet_score_threshold", -1), 0.55):
            raise RuntimeError("Iteration {} changed the ratchet threshold.".format(iteration))
        if row_index < len(rows) and (
            row.get("v4_teacher_objective_gate_triggered")
            or row.get("oracle_sparring_ratchet_triggered")
        ):
            raise RuntimeError("A safety trigger occurred before the final telemetry row.")

    last = rows[-1]
    paused = bool(
        last.get("v4_teacher_objective_gate_triggered")
        or last.get("oracle_sparring_ratchet_triggered")
    )
    if returncode == 0:
        if paused:
            raise RuntimeError("A safety control triggered but the trainer returned success.")
        if actual_iterations[-1] != end_iteration:
            raise RuntimeError("Trainer returned success before the requested end iteration.")
    elif not paused:
        raise RuntimeError(
            "Trainer exited with code {} without a declared safety pause.".format(returncode)
        )
    return "paused" if paused else "completed"


def main():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("The P2 production continuation requires CUDA.")
    gpu_name = torch.cuda.get_device_name(0)
    if "P100" not in gpu_name.upper():
        raise RuntimeError(
            "The reference P2 job requires a P100, but Kaggle provided {}.".format(
                gpu_name
            )
        )

    entry_point = _exactly_one("main_santorini.py")
    project_root = entry_point.parent
    manifest_path = _exactly_one("p2-production-manifest.json")
    manifest = json.loads(manifest_path.read_text())
    checkpoint = _exactly_one(CHECKPOINT_NAME)
    replay = _exactly_one(REPLAY_NAME)
    seam_suite = _exactly_one(SEAM_SUITE_NAME)
    for path in (checkpoint, replay, seam_suite):
        expected = manifest["inputs"][path.name]["sha256"]
        if _sha256(path) != expected:
            raise RuntimeError("Input digest changed: {}".format(path))

    bundled_oracle = manifest_path.parent / "oracle-build" / "santorini-oracle-linux-x86_64"
    expected_oracle = manifest.get("oracle_build", {}).get("linux_binary_sha256")
    if not bundled_oracle.is_file() or _sha256(bundled_oracle) != expected_oracle:
        raise RuntimeError("Bundled Linux oracle is missing or changed.")
    oracle_binary = WORKING_ROOT / "santorini-oracle-linux-x86_64"
    WORKING_ROOT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(bundled_oracle, oracle_binary)
    oracle_binary.chmod(0o755)

    start_iteration = int(manifest["lineage"]["start_iteration"])
    end_iteration = int(manifest["lineage"]["end_iteration"])
    replay_reuse = float(manifest["protocol"]["replay_reuse"])
    replay_reuse_warmup_iters = int(
        manifest["protocol"]["replay_reuse_warmup_iters"]
    )
    output_dir = WORKING_ROOT / "iterations_{}_{}".format(
        start_iteration + 1, end_iteration
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(
            "Output directory is not empty; refusing to overwrite a partial lineage: {}".format(
                output_dir
            )
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    if MILESTONE_NAME in manifest.get("inputs", {}):
        milestone = _exactly_one(MILESTONE_NAME)
        if _sha256(milestone) != manifest["inputs"][MILESTONE_NAME]["sha256"]:
            raise RuntimeError("Milestone anchor digest changed.")
        shutil.copy2(
            milestone,
            output_dir / "checkpoint_{}.pth.tar".format(start_iteration),
        )

    command = [
        sys.executable, str(entry_point),
        "--architecture", "v4",
        "--training-mode", "latest",
        "--num-iters", str(end_iteration - start_iteration),
        "--num-eps", "240",
        "--num-mcts-sims", "96",
        "--search-mode", "gumbel",
        "--gumbel-max-considered-actions", "16",
        "--gumbel-scale", "1.0",
        "--gumbel-placement-scale", "1.5",
        "--evaluation-gumbel-scale", "0.0",
        "--evaluation-gumbel-placement-scale", "1.5",
        "--placement-scale-exploration-probability", "0.10",
        "--placement-exploration-gumbel-scale", "2.25",
        "--playout-cap-randomization",
        "--playout-cap-full-probability", "0.25",
        "--playout-cap-fast-sims", "32",
        "--self-play-batch-size", "128",
        "--batch-size", "256",
        "--replay-reuse", str(replay_reuse),
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
        "--load-examples",
        "--examples-file", str(replay),
        "--keep-loaded-examples",
        "--checkpoint", str(output_dir),
        "--checkpoint-examples-to-keep", "0",
        "--oracle-sparring-probability", "0.10",
        "--oracle-sparring-nodes", "5000",
        "--oracle-sparring-workers", "4",
        "--oracle-sparring-opening-seed", "20260921",
        "--oracle-sparring-ladder-version", "2",
        "--oracle-sparring-ratchet-games", "80",
        "--oracle-sparring-ratchet-score", "0.55",
        "--oracle-binary", str(oracle_binary),
        "--v4-seam-telemetry-suite", str(seam_suite),
        "--v4-seam-telemetry-interval", "1",
        "--v4-seam-telemetry-batch-size", "256",
        "--v4-teacher-objective-step-threshold", "0.05",
        "--milestone-interval", "20",
        "--telemetry-match-games", "40",
        "--telemetry-match-batch-size", "128",
        "--telemetry-placement-games", "40",
        "--telemetry-placement-temperature", "1.0",
        "--telemetry-opening-seed", "20260715",
        "--seed", "20260930",
        "--quiet",
    ]
    print("P2 production lineage: iterations {}-{}".format(start_iteration + 1, end_iteration), flush=True)
    print("Running:", " ".join(command), flush=True)
    started = time.perf_counter()
    result = subprocess.run(command, cwd=project_root, check=False)
    elapsed_seconds = time.perf_counter() - started

    telemetry_path = output_dir / "telemetry" / "telemetry.jsonl"
    rows = (
        [json.loads(line) for line in telemetry_path.read_text().splitlines() if line]
        if telemetry_path.is_file() else []
    )
    status = _validate_telemetry(rows, manifest, result.returncode)
    last_iteration = int(rows[-1]["iteration"])
    required_outputs = (
        output_dir / "latest-training.pth.tar",
        output_dir / "latest.pth.tar",
        output_dir / "latest.examples.npz",
    )
    for path in required_outputs:
        if not path.is_file():
            raise FileNotFoundError(path)

    checkpoint_payload = torch.load(
        required_outputs[0], map_location="cpu", weights_only=False
    )
    if int(checkpoint_payload.get("training_metadata", {}).get("iteration", -1)) != last_iteration:
        raise RuntimeError("Resumable checkpoint iteration does not match telemetry.")
    from santorini.ReplayBuffer import load_compact_replay, replay_metadata
    replay_windows = load_compact_replay(required_outputs[2])
    replay_examples = [example for window in replay_windows for example in window]
    source_counts = {
        source: sum(
            replay_metadata(example).get("source") == source
            for example in replay_examples
        )
        for source in ("self_play", "oracle_sparring")
    }
    if source_counts["oracle_sparring"] <= 0:
        raise RuntimeError("Production replay contains no oracle-sparring records.")

    contract = {
        "schema_version": 1,
        "contract": "santorini_v4_p2_production_continuation",
        "status": status,
        "gpu": gpu_name,
        "elapsed_seconds": elapsed_seconds,
        "process_returncode": int(result.returncode),
        "start_iteration": start_iteration,
        "requested_end_iteration": end_iteration,
        "last_completed_iteration": last_iteration,
        "completed_iterations": len(rows),
        "input_checkpoint_sha256": _sha256(checkpoint),
        "input_replay_sha256": _sha256(replay),
        "seam_suite_sha256": _sha256(seam_suite),
        "oracle_binary_sha256": _sha256(oracle_binary),
        "last_telemetry": rows[-1],
        "replay_windows": len(replay_windows),
        "replay_examples": len(replay_examples),
        "replay_source_counts": source_counts,
        "outputs": {
            path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in required_outputs
        },
        "disagreement_starts_enabled": False,
        "auxiliary_oracle_head_enabled": False,
        "final_test_touched": False,
        "final_arena_seeds_touched": False,
    }
    contract_path = output_dir / "p2-production-contract.json"
    contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    print("P2 continuation status:", status, flush=True)
    print("Last completed iteration:", last_iteration, flush=True)
    print("Outputs:", sorted(path.name for path in output_dir.iterdir()), flush=True)


if __name__ == "__main__":
    main()
