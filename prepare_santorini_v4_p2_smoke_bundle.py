"""Build the self-contained, auto-extracted Kaggle P2 smoke upload."""

import argparse
import json
import os
from pathlib import Path
import zipfile

from prepare_santorini_v4_oracle_sweep_bundle import (
    CARGO_CONFIG,
    ORACLE_CARGO_TOML,
    ROOT_CARGO_TOML,
    _tree_files,
    _tree_sha256,
    _vendor_sources,
    _write_tree,
)
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
    parser.add_argument("--santorini-ai-root", default="../santorini-ai")
    parser.add_argument(
        "--output", default="temp/santorini_v4_p2_smoke_bundle.zip"
    )
    return parser.parse_args()


def build_bundle(args):
    root = Path(__file__).resolve().parent
    checkpoint = Path(args.checkpoint).resolve()
    seam_suite = Path(args.seam_suite).resolve()
    santorini_ai = Path(args.santorini_ai_root).resolve()
    oracle_source = root / "tools" / "santorini_oracle"
    core_source = santorini_ai / "santorini_core"
    model = santorini_ai / "models" / "batch5_final.bin"
    license_path = santorini_ai / "LICENSE"
    lockfile = oracle_source / "Cargo.lock"
    main_rs = oracle_source / "src" / "main.rs"
    required = (
        checkpoint,
        seam_suite,
        core_source / "Cargo.toml",
        model,
        license_path,
        lockfile,
        main_rs,
    )
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)

    sources = sorted(root.glob("*.py")) + sorted((root / "santorini").rglob("*.py"))
    vendor = _vendor_sources()
    core_files = _tree_files(core_source)
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
            "oracle_sparring_probability": {"ordinary": 0.0, "mixed": 0.10},
            "oracle_nodes": 100_000,
            "oracle_workers": 4,
            "oracle_ladder_version": 1,
            "opening_seed": 20260921,
            "inference_precision": "fp32",
            "disagreement_starts": False,
            "auxiliary_oracle_head": False,
        },
        "oracle_build": {
            "oracle_version": "0.2.0",
            "oracle_main_sha256": file_sha256(main_rs),
            "core_tree_sha256": _tree_sha256(core_source, core_files),
            "model_sha256": file_sha256(model),
            "cargo_lock_sha256": file_sha256(lockfile),
            "offline": True,
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
        archive.writestr("oracle-build/Cargo.toml", ROOT_CARGO_TOML)
        archive.writestr("oracle-build/oracle/Cargo.toml", ORACLE_CARGO_TOML)
        archive.writestr("oracle-build/.cargo/config.toml", CARGO_CONFIG)
        archive.write(lockfile, "oracle-build/Cargo.lock")
        archive.write(main_rs, "oracle-build/oracle/src/main.rs")
        archive.write(model, "oracle-build/models/batch5_final.bin")
        archive.write(license_path, "oracle-build/SANTORINI_AI_LICENSE")
        _write_tree(archive, core_source, "oracle-build/santorini_core")
        for name, path in sorted(vendor.items()):
            _write_tree(archive, path, "oracle-build/vendor/{}".format(name))
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
