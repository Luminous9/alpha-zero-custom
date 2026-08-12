"""Build the bundled Linux oracle offline and run the extracted P2 sweep."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


INPUT_ROOT = Path("/kaggle/input")
OUTPUT_ROOT = Path("/kaggle/working/santorini_v4_p2_oracle_sweep")
BUILD_ROOT = Path("/kaggle/tmp/santorini_v4_p2_oracle_build")


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
    entry_point = _exactly_one("run_santorini_v4_oracle_sweep.py")
    project_root = entry_point.parent
    manifest_path = _exactly_one("p2-oracle-sweep-manifest.json")
    manifest = json.loads(manifest_path.read_text())
    checkpoint = _exactly_one("p1c-checkpoint.pth.tar")
    if _sha256(checkpoint) != manifest["checkpoint"]["sha256"]:
        raise RuntimeError("P1c checkpoint digest does not match the bundle manifest.")

    oracle_root = manifest_path.parent / "oracle-build"
    oracle_manifest = oracle_root / "Cargo.toml"
    oracle_main = oracle_root / "oracle" / "src" / "main.rs"
    oracle_model = oracle_root / "models" / "batch5_final.bin"
    oracle_lock = oracle_root / "Cargo.lock"
    for path in (oracle_manifest, oracle_main, oracle_model, oracle_lock):
        if not path.is_file():
            raise FileNotFoundError(path)
    expected_build = manifest["oracle_build"]
    for path, expected in (
        (oracle_main, expected_build["oracle_main_sha256"]),
        (oracle_model, expected_build["model_sha256"]),
        (oracle_lock, expected_build["cargo_lock_sha256"]),
    ):
        if _sha256(path) != expected:
            raise RuntimeError("Bundled oracle source digest changed: {}".format(path))
    if shutil.which("cargo") is None:
        raise RuntimeError("Kaggle image does not provide cargo for the offline oracle build.")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    BUILD_ROOT.mkdir(parents=True, exist_ok=True)
    target_root = BUILD_ROOT / "target"
    build_env = dict(os.environ)
    build_env["CARGO_TARGET_DIR"] = str(target_root)
    build_command = [
        "cargo", "build", "--release", "--offline", "--locked",
        "--manifest-path", str(oracle_manifest),
        "-p", "santorini-oracle",
    ]
    print("Building bundled oracle offline:", " ".join(build_command), flush=True)
    build_started = time.perf_counter()
    subprocess.run(build_command, cwd=oracle_root, env=build_env, check=True)
    build_seconds = time.perf_counter() - build_started
    oracle_binary = target_root / "release" / "santorini-oracle"
    if not oracle_binary.is_file():
        raise RuntimeError("Offline build completed without the oracle binary.")

    command = [
        sys.executable,
        str(entry_point),
        "--checkpoint", str(checkpoint),
        "--oracle-binary", str(oracle_binary),
        "--output-dir", str(OUTPUT_ROOT),
        "--budgets", "5000", "10000", "20000", "50000", "100000", "250000",
        "--games", "40",
        "--simulations", "96",
        "--opening-seed", "20260921",
        "--bootstrap-seed", "20260922",
        "--bootstrap-samples", "10000",
        "--inference-cache-size", "4096",
        "--device", "cuda",
        "--fp16",
    ]
    print("Running:", " ".join(command), flush=True)
    sweep_started = time.perf_counter()
    subprocess.run(command, cwd=project_root, check=True)
    sweep_seconds = time.perf_counter() - sweep_started
    summary_path = OUTPUT_ROOT / "oracle-sweep-summary.json"
    if not summary_path.is_file():
        raise RuntimeError("Oracle sweep completed without its summary.")
    summary = json.loads(summary_path.read_text())
    if summary.get("contract", {}).get("checkpoint_sha256") != manifest["checkpoint"]["sha256"]:
        raise RuntimeError("Sweep summary belongs to a different checkpoint.")
    contract = {
        "schema_version": 1,
        "contract": "santorini_v4_p2_oracle_sweep_kaggle_job",
        "bundle_manifest": str(manifest_path),
        "checkpoint_sha256": _sha256(checkpoint),
        "oracle_binary_sha256": _sha256(oracle_binary),
        "oracle_version": summary.get("oracle_version"),
        "oracle_build_seconds": build_seconds,
        "sweep_seconds": sweep_seconds,
        "selection": summary["selection"],
        "final_test_touched": False,
        "final_arena_seeds_touched": False,
    }
    (OUTPUT_ROOT / "kaggle-job-contract.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n"
    )
    print("Selection:", json.dumps(summary["selection"], sort_keys=True), flush=True)
    print(
        "Outputs:",
        sorted(path.name for path in OUTPUT_ROOT.iterdir()),
        flush=True,
    )


if __name__ == "__main__":
    main()
