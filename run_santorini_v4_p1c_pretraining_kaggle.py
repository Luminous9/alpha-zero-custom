"""Run selected oracle-T25 P1c pretraining from an extracted Kaggle bundle."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


INPUT_ROOT = Path("/kaggle/input")
WORKING_ROOT = Path("/kaggle/working/santorini_v4_p1c_pretraining")
ARM_FILENAMES = {
    "oracle_t25": "placement-oracle-t25.npz",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=tuple(ARM_FILENAMES), required=True)
    parser.add_argument("--epochs", type=int, default=4)
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
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("--epochs must be positive.")
    entry_point = _exactly_one("screen_santorini_v4_bootstrap.py")
    project_root = entry_point.parent
    manifest_path = _exactly_one("p1c-pretraining-manifest.json")
    manifest = json.loads(manifest_path.read_text())
    inputs = {
        name: _exactly_one(filename)
        for name, filename in {
            "engine_train": "engine-corpus-train.npz",
            "selection_engine": "selection-engine-corpus.npz",
            "run13_standard": "run13-standard-component.npz",
            "train_plan": "train-plan.npz",
            "selection_plan": "selection-plan.npz",
            "placement": ARM_FILENAMES[args.arm],
        }.items()
    }
    for name, path in inputs.items():
        expected = manifest["inputs"][path.name]["sha256"]
        actual = _sha256(path)
        if actual != expected:
            raise RuntimeError(
                "{} digest changed: {} != {}".format(name, actual, expected)
            )

    output_dir = WORKING_ROOT / args.arm
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(entry_point),
        "--engine-corpus", str(inputs["engine_train"]),
        "--selection-engine-corpus", str(inputs["selection_engine"]),
        "--run13-component", str(inputs["run13_standard"]),
        "--placement-component", str(inputs["placement"]),
        "--train-plan", str(inputs["train_plan"]),
        "--selection-plan", str(inputs["selection_plan"]),
        "--output-dir", str(output_dir),
        "--epochs", str(args.epochs),
        "--batch-size", "256",
        "--learning-rate", "0.0003",
        "--weight-decay", "0.0001",
        "--policy-weight", "0.25",
        "--policy-epsilon", "0.05",
        "--alpha-boot", "0.5",
        "--score-temperature", "261.8",
        "--stage-reliability", "0.25", "0.75", "1.0",
        "--seed", "20260812",
        "--device", "cuda",
        "--data-loading", "streaming",
        "--configs", "ordinary_6x192_13_global_blend",
    ]
    print("P1c arm:", args.arm, flush=True)
    print("Project root:", project_root, flush=True)
    print("Running:", " ".join(command), flush=True)
    started = time.perf_counter()
    subprocess.run(command, cwd=project_root, check=True)
    job = {
        "schema_version": 1,
        "contract": "santorini_v4_p1c_selected_oracle_t25",
        "arm": args.arm,
        "epochs": args.epochs,
        "seed": 20260812,
        "elapsed_seconds": time.perf_counter() - started,
        "inputs": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in inputs.items()
        },
        "final_test_touched": False,
    }
    (output_dir / "job-contract.json").write_text(
        json.dumps(job, indent=2, sort_keys=True) + "\n"
    )
    print("Outputs:", sorted(str(path) for path in output_dir.iterdir()), flush=True)


if __name__ == "__main__":
    main()
