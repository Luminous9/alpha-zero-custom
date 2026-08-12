"""Create the self-contained auto-extracted Kaggle upload for Gate G1."""

import argparse
import json
import os
from pathlib import Path
import zipfile

from santorini.OracleResearch import file_sha256


INPUTS = {
    "p1c-checkpoint.pth.tar": (
        "temp/santorini_v4_p1c_pretraining_results/"
        "ordinary_6x192_13_global_blend.pth.tar"
    ),
    "run13-checkpoint.pth.tar": "temp/santorini_v3_run13_gumbel/latest.pth.tar",
    "selection-engine-corpus.npz": "temp/santorini_v4_scaled/engine-corpus.npz",
    "run13-component.npz": "temp/santorini_v4_scaled/run13-component.npz",
    "selection-plan.npz": "temp/santorini_v4_scaled/selection-3k.npz",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="temp/santorini_v4_g1_bundle.zip")
    return parser.parse_args()


def build_bundle(args):
    root = Path(__file__).resolve().parent
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    sources = sorted(root.glob("*.py")) + sorted((root / "santorini").rglob("*.py"))
    inputs = {
        name: (root / relative).resolve() for name, relative in INPUTS.items()
    }
    for path in inputs.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = {
        "schema_version": 1,
        "purpose": "santorini_v4_p1c_gate_g1",
        "source_files": len(sources),
        "inputs": {
            name: {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
            for name, path in sorted(inputs.items())
        },
        "protocol": {
            "selection_seed": 20260901,
            "games_per_gate": 40,
            "equal_simulation_budgets": [96, 128],
            "full_game_placement_temperature": 1.0,
            "p1c_root_symmetries": {"standard": 1, "placement": 1},
            "run13_root_symmetries": {"standard": 8, "placement": 8},
            "precision": "fp32",
            "final_test_touched": False,
            "final_arena_seeds_touched": False,
        },
    }
    temporary = output.with_suffix(output.suffix + ".tmp")
    with zipfile.ZipFile(temporary, "w", allowZip64=True) as archive:
        for path in sources:
            archive.write(
                path,
                path.relative_to(root).as_posix(),
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=6,
            )
        for name, path in sorted(inputs.items()):
            compression = (
                zipfile.ZIP_STORED if path.suffix == ".npz"
                else zipfile.ZIP_DEFLATED
            )
            archive.write(
                path,
                "inputs/" + name,
                compress_type=compression,
                compresslevel=6 if compression == zipfile.ZIP_DEFLATED else None,
            )
        archive.writestr(
            "g1-manifest.json",
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
