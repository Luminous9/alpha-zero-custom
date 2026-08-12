"""Build the self-contained, auto-extracted Kaggle P2 oracle-sweep upload."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import zipfile

from santorini.OracleResearch import file_sha256


ROOT_CARGO_TOML = """[workspace]
resolver = "3"
members = ["oracle", "santorini_core"]

[workspace.dependencies]
colored = "2.0.4"
rand = { version = "0.9" }
serde = { version = "1.0.219", features = ["derive"] }
serde_json = "1.0.140"
strum = { version = "0.27.1", features = ["derive"] }
clap = { version = "4.5.40", features = ["derive"] }
serde_yaml = "0.9.34"
num_cpus = "1.17.0"

[profile.release]
opt-level = 3
rpath = false
debug-assertions = false
codegen-units = 1
lto = true
panic = "abort"
debug = false
"""

ORACLE_CARGO_TOML = """[package]
name = "santorini-oracle"
version = "0.2.0"
edition = "2024"
publish = false

[dependencies]
santorini_core = { path = "../santorini_core" }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
"""

CARGO_CONFIG = """[source.crates-io]
replace-with = "vendored-sources"

[source.vendored-sources]
directory = "vendor"

[net]
offline = true
"""


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default=(
            "temp/santorini_v4_p1c_pretraining_results/"
            "ordinary_6x192_13_global_blend.pth.tar"
        ),
    )
    parser.add_argument("--santorini-ai-root", default="../santorini-ai")
    parser.add_argument(
        "--output", default="temp/santorini_v4_p2_oracle_sweep_bundle.zip"
    )
    return parser.parse_args()


def _tree_files(root):
    root = Path(root)
    return sorted(
        path for path in root.rglob("*")
        if path.is_file() and ".git" not in path.parts and "target" not in path.parts
    )


def _tree_sha256(root, files):
    root = Path(root)
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _vendor_sources():
    roots = sorted((Path.home() / ".cargo" / "registry" / "src").glob("*"))
    sources = {}
    for registry in roots:
        if not registry.is_dir():
            continue
        for crate in sorted(registry.iterdir()):
            if not crate.is_dir():
                continue
            existing = sources.get(crate.name)
            if existing is not None and _tree_sha256(existing, _tree_files(existing)) != _tree_sha256(crate, _tree_files(crate)):
                raise ValueError("Conflicting cached Cargo source: {}".format(crate.name))
            sources[crate.name] = crate
    if not sources:
        raise FileNotFoundError("No cached Cargo registry sources were found.")
    return sources


def _write_tree(archive, source_root, archive_root):
    files = _tree_files(source_root)
    for path in files:
        archive.write(
            path,
            "{}/{}".format(archive_root, path.relative_to(source_root).as_posix()),
            compress_type=zipfile.ZIP_DEFLATED,
            compresslevel=6,
        )
    return files


def build_bundle(args):
    root = Path(__file__).resolve().parent
    checkpoint = Path(args.checkpoint).resolve()
    santorini_ai = Path(args.santorini_ai_root).resolve()
    oracle_source = root / "tools" / "santorini_oracle"
    core_source = santorini_ai / "santorini_core"
    model = santorini_ai / "models" / "batch5_final.bin"
    license_path = santorini_ai / "LICENSE"
    lockfile = oracle_source / "Cargo.lock"
    main_rs = oracle_source / "src" / "main.rs"
    required = (checkpoint, core_source / "Cargo.toml", model, license_path, lockfile, main_rs)
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)

    sources = sorted(root.glob("*.py")) + sorted((root / "santorini").rglob("*.py"))
    vendor = _vendor_sources()
    core_files = _tree_files(core_source)
    vendor_file_count = sum(len(_tree_files(path)) for path in vendor.values())
    manifest = {
        "schema_version": 1,
        "purpose": "santorini_v4_p2_oracle_sparring_budget_sweep",
        "checkpoint": {
            "filename": "p1c-checkpoint.pth.tar",
            "bytes": checkpoint.stat().st_size,
            "sha256": file_sha256(checkpoint),
        },
        "protocol": {
            "oracle_budgets": [5_000, 10_000, 20_000, 50_000, 100_000, 250_000],
            "games_per_budget": 40,
            "paired_openings": 20,
            "simulations": 96,
            "search_mode": "gumbel",
            "gumbel_scale": 0.0,
            "canonical_d4": True,
            "root_symmetry_samples": 1,
            "fp16": True,
            "opening_seed": 20260921,
            "bootstrap_seed": 20260922,
            "target_v4_score": [0.35, 0.50],
            "target_midpoint": 0.425,
            "final_test_touched": False,
            "final_arena_seeds_touched": False,
        },
        "oracle_build": {
            "oracle_version": "0.2.0",
            "oracle_main_sha256": file_sha256(main_rs),
            "core_tree_sha256": _tree_sha256(core_source, core_files),
            "model_sha256": file_sha256(model),
            "cargo_lock_sha256": file_sha256(lockfile),
            "vendor_crates": len(vendor),
            "vendor_files": vendor_file_count,
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
        archive.write(
            checkpoint,
            "inputs/p1c-checkpoint.pth.tar",
            compress_type=zipfile.ZIP_DEFLATED,
            compresslevel=6,
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
            "p2-oracle-sweep-manifest.json",
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
    report_path = output.with_suffix(output.suffix + ".report.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main():
    print(json.dumps(build_bundle(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
