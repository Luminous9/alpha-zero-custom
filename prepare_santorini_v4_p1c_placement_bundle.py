"""Create the self-contained zip uploaded for the P1c placement Kaggle job."""

import argparse
import json
import os
import zipfile
from pathlib import Path

from santorini.OracleResearch import file_sha256


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default="temp/santorini_v3_run13_gumbel/latest.pth.tar",
    )
    parser.add_argument(
        "--replay",
        default="temp/santorini_v3_run13_gumbel/latest.examples.npz",
    )
    parser.add_argument(
        "--output",
        default="temp/santorini_v4_p1c_placement_bundle.zip",
    )
    return parser.parse_args()


def build_bundle(args):
    root = Path(__file__).resolve().parent
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    sources = sorted(root.glob("*.py")) + sorted((root / "santorini").rglob("*.py"))
    required = [Path(args.checkpoint).resolve(), Path(args.replay).resolve()]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with zipfile.ZipFile(
        temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for path in sources:
            archive.write(path, path.relative_to(root).as_posix())
        archive.write(required[0], "inputs/run13-latest.pth.tar")
        archive.write(required[1], "inputs/run13-latest.examples.npz")
        manifest = {
            "schema_version": 1,
            "purpose": "santorini_v4_p1c_placement_distillation",
            "checkpoint_sha256": file_sha256(required[0]),
            "replay_sha256": file_sha256(required[1]),
            "source_files": len(sources),
        }
        archive.writestr("bundle-manifest.json", json.dumps(manifest, indent=2, sort_keys=True))
    os.replace(temporary, output)
    report = {
        "output": str(output),
        "output_bytes": output.stat().st_size,
        "output_sha256": file_sha256(output),
        "checkpoint_sha256": file_sha256(required[0]),
        "replay_sha256": file_sha256(required[1]),
        "source_files": len(sources),
    }
    report_path = output.with_suffix(output.suffix + ".report.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main():
    print(json.dumps(build_bundle(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
