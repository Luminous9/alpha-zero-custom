"""Locate an extracted G1 Kaggle dataset, verify it, and run the gate."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


INPUT_ROOT = Path("/kaggle/input")
OUTPUT_ROOT = Path("/kaggle/working/santorini_v4_g1")


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
    entry_point = _exactly_one("run_santorini_v4_g1.py")
    project_root = entry_point.parent
    manifest_path = _exactly_one("g1-manifest.json")
    manifest = json.loads(manifest_path.read_text())
    inputs = {
        name: _exactly_one(filename)
        for name, filename in {
            "candidate": "p1c-checkpoint.pth.tar",
            "run13": "run13-checkpoint.pth.tar",
            "engine_corpus": "selection-engine-corpus.npz",
            "run13_component": "run13-component.npz",
            "selection_plan": "selection-plan.npz",
        }.items()
    }
    for name, path in inputs.items():
        expected = manifest["inputs"][path.name]["sha256"]
        actual = _sha256(path)
        if actual != expected:
            raise RuntimeError(
                "{} digest changed: {} != {}".format(name, actual, expected)
            )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(entry_point),
        "--candidate", str(inputs["candidate"]),
        "--run13", str(inputs["run13"]),
        "--engine-corpus", str(inputs["engine_corpus"]),
        "--run13-component", str(inputs["run13_component"]),
        "--selection-plan", str(inputs["selection_plan"]),
        "--output-dir", str(OUTPUT_ROOT),
        "--device", "cuda",
    ]
    print("Running:", " ".join(command), flush=True)
    started = time.perf_counter()
    subprocess.run(command, cwd=project_root, check=True)
    summary_path = OUTPUT_ROOT / "g1-summary.json"
    if not summary_path.is_file():
        raise RuntimeError("G1 completed without its summary.")
    summary = json.loads(summary_path.read_text())
    contract = {
        "schema_version": 1,
        "contract": "santorini_v4_p1c_gate_g1_kaggle_job",
        "elapsed_seconds": time.perf_counter() - started,
        "decision": summary["decision"],
        "inputs": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in inputs.items()
        },
        "final_test_touched": False,
        "final_arena_seeds_touched": False,
    }
    (OUTPUT_ROOT / "kaggle-job-contract.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n"
    )
    print("Outputs:", sorted(path.name for path in OUTPUT_ROOT.iterdir()), flush=True)


if __name__ == "__main__":
    main()
