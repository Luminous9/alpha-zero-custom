"""Build a self-contained Kaggle bundle that resumes the accepted V4 P2 lineage."""

import argparse
import json
import os
from pathlib import Path
import zipfile

import torch

from santorini.OracleResearch import file_sha256
from santorini.ReplayBuffer import load_compact_replay


CHECKPOINT_NAME = "p2-resume-training.pth.tar"
REPLAY_NAME = "p2-resume.examples.npz"
SEAM_SUITE_NAME = "v4-seam-telemetry-suite.npz"
MILESTONE_NAME = "p2-milestone-anchor.pth.tar"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default=(
            "temp/santorini_v4_p2_smoke_results/transition/"
            "latest-training.pth.zip"
        ),
        help="Resumable latest-training checkpoint (a browser .zip rename is accepted).",
    )
    parser.add_argument(
        "--replay",
        default=(
            "temp/santorini_v4_p2_smoke_results/transition/"
            "latest.examples.zip"
        ),
        help="Compact replay NPZ (a browser .zip rename is accepted).",
    )
    parser.add_argument(
        "--seam-suite",
        default="temp/santorini_v4_p2_preparation/v4-seam-telemetry-suite.npz",
    )
    parser.add_argument(
        "--linux-oracle-binary",
        default=(
            "temp/santorini_v4_p2_linux_build/target/release/"
            "santorini-oracle"
        ),
    )
    parser.add_argument("--santorini-ai-license", default="../santorini-ai/LICENSE")
    parser.add_argument(
        "--milestone-anchor",
        help=(
            "Optional inference checkpoint for the preceding 20-iteration milestone. "
            "Required when resuming at an iteration divisible by 20."
        ),
    )
    parser.add_argument("--end-iteration", type=int, default=20)
    parser.add_argument(
        "--replay-reuse-warmup-iters",
        type=int,
        default=8,
        help="Absolute iteration at which the configured 16x reuse is reached.",
    )
    parser.add_argument(
        "--output", default="temp/santorini_v4_p2_iterations_2_20_bundle.zip"
    )
    return parser.parse_args()


def _load_checkpoint_metadata(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    metadata = dict(payload.get("training_metadata", {}))
    if "iteration" not in metadata:
        raise ValueError("Resume checkpoint has no iteration metadata.")
    return metadata


def _validate_resume_state(checkpoint, replay, seam_suite, end_iteration):
    metadata = _load_checkpoint_metadata(checkpoint)
    start_iteration = int(metadata["iteration"])
    if end_iteration <= start_iteration:
        raise ValueError(
            "--end-iteration must be later than checkpoint iteration {}.".format(
                start_iteration
            )
        )
    if metadata.get("training_mode") != "latest":
        raise ValueError("Resume checkpoint is not a latest-mode training checkpoint.")
    if int(metadata.get("oracle_sparring_nodes", -1)) != 5_000:
        raise ValueError("Resume checkpoint does not use the frozen 5k oracle rung.")
    if int(metadata.get("oracle_sparring_ladder_version", -1)) != 2:
        raise ValueError("Resume checkpoint does not use oracle ladder version 2.")
    if int(metadata.get("replay_reuse_warmup_iters", -1)) < 1:
        raise ValueError("Resume checkpoint does not declare its replay-reuse warm-up.")
    if metadata.get("v4_seam_suite_fingerprint") != file_sha256(seam_suite):
        raise ValueError("Resume checkpoint and frozen seam suite do not match.")
    if metadata.get("v4_teacher_objective_current") is None:
        raise ValueError("Resume checkpoint lacks the standing teacher-objective state.")
    pair_history = metadata.get("oracle_sparring_pair_score_history")
    if not isinstance(pair_history, list):
        raise ValueError("Resume checkpoint lacks the oracle ratchet history.")

    windows = load_compact_replay(replay)
    if not windows or not any(windows):
        raise ValueError("Resume replay is empty.")
    if len(windows) > 20:
        raise ValueError("Resume replay exceeds the frozen 20-iteration window.")
    return metadata, windows


def build_bundle(args):
    root = Path(__file__).resolve().parent
    checkpoint = Path(args.checkpoint).resolve()
    replay = Path(args.replay).resolve()
    seam_suite = Path(args.seam_suite).resolve()
    linux_oracle = Path(args.linux_oracle_binary).resolve()
    license_path = Path(args.santorini_ai_license).resolve()
    milestone_anchor = (
        Path(args.milestone_anchor).resolve() if args.milestone_anchor else None
    )
    required = [checkpoint, replay, seam_suite, linux_oracle, license_path]
    if milestone_anchor is not None:
        required.append(milestone_anchor)
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    if int(args.replay_reuse_warmup_iters) < 1:
        raise ValueError("--replay-reuse-warmup-iters must be positive.")

    metadata, replay_windows = _validate_resume_state(
        checkpoint, replay, seam_suite, args.end_iteration
    )
    start_iteration = int(metadata["iteration"])
    if start_iteration > 0 and start_iteration % 20 == 0 and milestone_anchor is None:
        raise ValueError(
            "A --milestone-anchor is required when resuming from iteration {} so "
            "the next 20-iteration self-match has its fixed opponent.".format(
                start_iteration
            )
        )

    sources = sorted(root.glob("*.py")) + sorted((root / "santorini").rglob("*.py"))
    input_paths = {
        CHECKPOINT_NAME: checkpoint,
        REPLAY_NAME: replay,
        SEAM_SUITE_NAME: seam_suite,
    }
    if milestone_anchor is not None:
        input_paths[MILESTONE_NAME] = milestone_anchor
    manifest = {
        "schema_version": 1,
        "purpose": "santorini_v4_p2_production_continuation",
        "lineage": {
            "start_iteration": start_iteration,
            "end_iteration": int(args.end_iteration),
            "iterations_requested": int(args.end_iteration) - start_iteration,
            "resume_teacher_objective": float(
                metadata["v4_teacher_objective_current"]
            ),
            "resume_ratchet_pair_scores": [
                float(score)
                for score in metadata["oracle_sparring_pair_score_history"]
            ],
            "resume_replay_windows": len(replay_windows),
            "resume_replay_examples": sum(len(window) for window in replay_windows),
            "resume_replay_reuse_warmup_iters": int(
                metadata["replay_reuse_warmup_iters"]
            ),
        },
        "inputs": {
            name: {
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            for name, path in input_paths.items()
        },
        "protocol": {
            "games_per_iteration": 240,
            "self_play_concurrency": 128,
            "full_simulations": 96,
            "fast_simulations": 32,
            "full_search_probability": 0.25,
            "placement_full_search": True,
            "search_mode": "gumbel",
            "self_play_gumbel_scale": 1.0,
            "evaluation_gumbel_scale": 0.0,
            "placement_gumbel_scale": 1.5,
            "placement_exploration_probability": 0.10,
            "placement_exploration_gumbel_scale": 2.25,
            "oracle_sparring_probability": 0.10,
            "oracle_nodes": 5_000,
            "oracle_workers": 4,
            "oracle_ladder_version": 2,
            "oracle_ratchet_games": 80,
            "oracle_ratchet_score": 0.55,
            "replay_reuse": 16.0,
            "replay_reuse_warmup_iters": int(args.replay_reuse_warmup_iters),
            "teacher_objective_step_threshold": 0.05,
            "history_iterations": 20,
            "milestone_interval": 20,
            "milestone_standard_games": 40,
            "milestone_placement_games": 40,
            "opening_seed": 20260921,
            "telemetry_opening_seed": 20260715,
            "inference_precision": "fp32",
            "console_log_mode": "compact_no_progress_bars",
            "disagreement_starts": False,
            "auxiliary_oracle_head": False,
        },
        "oracle_build": {
            "oracle_version": "0.2.0",
            "platform": "linux-x86_64",
            "linux_binary_bytes": linux_oracle.stat().st_size,
            "linux_binary_sha256": file_sha256(linux_oracle),
            "runtime_cargo_required": False,
        },
        "python_source_files": len(sources),
    }

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with zipfile.ZipFile(temporary, "w", allowZip64=True) as archive:
        for path in sources:
            archive.write(
                path,
                path.relative_to(root).as_posix(),
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=6,
            )
        for name, path in input_paths.items():
            archive.write(
                path,
                "inputs/{}".format(name),
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=6,
            )
        archive.write(
            linux_oracle,
            "oracle-build/santorini-oracle-linux-x86_64",
            compress_type=zipfile.ZIP_DEFLATED,
            compresslevel=6,
        )
        archive.write(license_path, "oracle-build/SANTORINI_AI_LICENSE")
        archive.writestr(
            "p2-production-manifest.json",
            json.dumps(manifest, indent=2, sort_keys=True),
            compress_type=zipfile.ZIP_DEFLATED,
            compresslevel=6,
        )
    os.replace(temporary, output)
    report = {
        **manifest,
        "output": str(output),
        "output_bytes": output.stat().st_size,
        "output_sha256": file_sha256(output),
    }
    output.with_suffix(output.suffix + ".report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def main():
    print(json.dumps(build_bundle(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
