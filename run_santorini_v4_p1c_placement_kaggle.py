"""Kaggle entry point for an auto-extracted P1c placement bundle.

Paste this file's body into a Kaggle cell, or execute it directly.  It searches
the already-extracted dataset tree; it never attempts to open a tar archive.
"""

import subprocess
import sys
from pathlib import Path


INPUT_ROOT = Path("/kaggle/input")
WORKING_ROOT = Path("/kaggle/working/santorini_v4_p1c_placement")


def _exactly_one(pattern):
    matches = sorted(INPUT_ROOT.rglob(pattern))
    if len(matches) != 1:
        raise RuntimeError(
            "Expected exactly one {} under {}, found: {}".format(
                pattern, INPUT_ROOT, [str(path) for path in matches]
            )
        )
    return matches[0]


entry_point = _exactly_one("label_santorini_v4_placement.py")
project_root = entry_point.parent
checkpoint = _exactly_one("run13-latest.pth.tar")
replay = _exactly_one("run13-latest.examples.npz")
WORKING_ROOT.mkdir(parents=True, exist_ok=True)

command = [
    sys.executable,
    str(entry_point),
    "--checkpoint", str(checkpoint),
    "--replay", str(replay),
    "--output", str(WORKING_ROOT / "placement-component.npz"),
    "--report-out", str(WORKING_ROOT / "placement-report.json"),
    "--device", "cuda",
    "--simulations", "96",
    "--fast-simulations", "32",
    "--full-search-probability", "0.25",
    "--continuations-per-state", "4",
    "--batch-size", "128",
    "--seed", "20260822",
]
print("Project root:", project_root)
print("Running:", " ".join(command))
subprocess.run(command, cwd=project_root, check=True)
print("Outputs:", sorted(str(path) for path in WORKING_ROOT.iterdir()))
