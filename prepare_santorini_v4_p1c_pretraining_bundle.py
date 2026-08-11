"""Create the self-contained Kaggle upload for selected P1c pretraining."""

import argparse
import json
import os
from pathlib import Path
import zipfile

from santorini.OracleResearch import file_sha256


INPUTS = {
    "engine-corpus-train.npz": "temp/santorini_v4_scaled/engine-corpus-1m-train.npz",
    "selection-engine-corpus.npz": "temp/santorini_v4_scaled/engine-corpus.npz",
    "run13-standard-component.npz": "temp/santorini_v4_scaled/run13-component.npz",
    "selection-plan.npz": "temp/santorini_v4_scaled/selection-3k.npz",
    "train-plan.npz": "temp/santorini_v4_p1c/train-plan.npz",
    "placement-oracle-t25.npz": "temp/santorini_v4_placement/oracle-t25-policy.npz",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", default="temp/santorini_v4_p1c_pretraining_bundle.zip"
    )
    return parser.parse_args()


def build_bundle(args):
    root = Path(__file__).resolve().parent
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    sources = sorted(root.glob("*.py")) + sorted((root / "santorini").rglob("*.py"))
    inputs = {
        archive_name: (root / relative_path).resolve()
        for archive_name, relative_path in INPUTS.items()
    }
    for path in inputs.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = {
        "schema_version": 1,
        "purpose": "santorini_v4_p1c_selected_oracle_t25_pretraining",
        "source_files": len(sources),
        "inputs": {
            name: {
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            for name, path in sorted(inputs.items())
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
            # NPZ inputs are already compressed; storing avoids a slow no-op pass.
            archive.write(path, "inputs/" + name, compress_type=zipfile.ZIP_STORED)
        archive.writestr(
            "p1c-pretraining-manifest.json",
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
