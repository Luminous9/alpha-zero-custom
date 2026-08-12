"""Build the self-contained, auto-extracted Kaggle P2 smoke upload."""

import argparse
import json
import os
from pathlib import Path
import zipfile

from santorini.OracleResearch import file_sha256


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default="temp/santorini_v4_p2_handoff/p2-start.pth.tar",
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
        "--output", default="temp/santorini_v4_p2_smoke_bundle.zip"
    )
    return parser.parse_args()


def build_bundle(args):
    root = Path(__file__).resolve().parent
    checkpoint = Path(args.checkpoint).resolve()
    seam_suite = Path(args.seam_suite).resolve()
    linux_oracle = Path(args.linux_oracle_binary).resolve()
    license_path = Path(args.santorini_ai_license).resolve()
    required = (checkpoint, seam_suite, linux_oracle, license_path)
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)

    sources = sorted(root.glob("*.py")) + sorted((root / "santorini").rglob("*.py"))
    manifest = {
        "schema_version": 1,
        "purpose": "santorini_v4_p2_p100_end_to_end_smoke",
        "inputs": {
            "p2-start.pth.tar": {
                "bytes": checkpoint.stat().st_size,
                "sha256": file_sha256(checkpoint),
            },
            "v4-seam-telemetry-suite.npz": {
                "bytes": seam_suite.stat().st_size,
                "sha256": file_sha256(seam_suite),
            },
        },
        "protocol": {
            "games": 240,
            "self_play_concurrency": 128,
            "full_simulations": 96,
            "fast_simulations": 32,
            "full_search_probability": 0.25,
            "placement_full_search": True,
            "search_mode": "gumbel",
            "gumbel_scale": 1.0,
            "placement_gumbel_scale": 1.5,
            "placement_exploration_probability": 0.10,
            "placement_exploration_gumbel_scale": 2.25,
            "oracle_sparring_probability": {
                "ordinary": 0.0,
                "mixed": 0.10,
                "transition": 0.10,
            },
            "oracle_nodes": {
                "ordinary": 100_000,
                "mixed": 100_000,
                "transition": 5_000,
            },
            "oracle_workers": 4,
            "oracle_ratchet_games": 80,
            "oracle_ratchet_score": 0.55,
            "oracle_ladder_version": {
                "ordinary": 1,
                "mixed": 1,
                "transition": 2,
            },
            "replay_reuse": 16.0,
            "replay_reuse_warmup_iters": {
                "ordinary": 0,
                "mixed": 0,
                "transition": 8,
            },
            "teacher_objective_step_threshold": 0.05,
            "opening_seed": 20260921,
            "inference_precision": "fp32",
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
        archive.write(checkpoint, "inputs/p2-start.pth.tar", zipfile.ZIP_DEFLATED)
        archive.write(
            seam_suite,
            "inputs/v4-seam-telemetry-suite.npz",
            zipfile.ZIP_DEFLATED,
        )
        archive.write(
            linux_oracle,
            "oracle-build/santorini-oracle-linux-x86_64",
            compress_type=zipfile.ZIP_DEFLATED,
            compresslevel=6,
        )
        archive.write(license_path, "oracle-build/SANTORINI_AI_LICENSE")
        archive.writestr(
            "p2-smoke-manifest.json",
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
