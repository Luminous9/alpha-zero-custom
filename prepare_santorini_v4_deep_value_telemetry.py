"""Combine the frozen A1 suite, deep labels, and iteration-11 reference."""

import argparse
import json
from pathlib import Path

import numpy as np

from santorini.OracleResearch import file_sha256


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        default="temp/santorini_v4_p2_deep_value_audit/frozen-value-suite.npz",
    )
    parser.add_argument(
        "--audit-rows",
        default="temp/santorini_v4_p2_deep_value_audit/deep-value-audit-rows.npz",
    )
    parser.add_argument(
        "--iteration11-checkpoint",
        default="temp/santorini_v4_p2_iter_5-11/checkpoint_11.pth.zip",
    )
    parser.add_argument(
        "--output",
        default="temp/santorini_v4_p2_deep_value_telemetry.npz",
    )
    return parser.parse_args()


def build(args):
    suite_path = Path(args.suite).resolve()
    rows_path = Path(args.audit_rows).resolve()
    checkpoint_path = Path(args.iteration11_checkpoint).resolve()
    for path in (suite_path, rows_path, checkpoint_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    with np.load(suite_path, allow_pickle=False) as suite, np.load(
        rows_path, allow_pickle=False
    ) as rows:
        hashes = suite["position_hashes"]
        if not np.array_equal(hashes, rows["position_hashes"]):
            raise ValueError("Frozen suite and audit rows are not aligned.")
        for name in ("stages", "bands", "replay_indices"):
            if not np.array_equal(suite[name], rows[name]):
                raise ValueError("Frozen suite and audit rows differ at {}.".format(name))
        metadata = {
            "schema_version": 1,
            "contract": "santorini_v4_p2_deep_value_telemetry",
            "reference_iteration": 11,
            "reference_checkpoint_sha256": file_sha256(checkpoint_path),
            "source_suite_sha256": file_sha256(suite_path),
            "source_audit_rows_sha256": file_sha256(rows_path),
            "oracle_label_kind": "santorini-ai-250k-cold-tt-score400",
            "paired_reference": True,
            "final_test_touched": False,
        }
        payload = {
            "schema_version": np.asarray([1], dtype=np.int16),
            "boards": suite["boards"].astype(np.int8),
            "oracle_values": rows["oracle_values"].astype(np.float32),
            "reference_values": rows["prediction_iter11"].astype(np.float32),
            "stages": suite["stages"],
            "bands": suite["bands"],
            "position_hashes": hashes,
            "metadata": np.asarray(json.dumps(metadata, sort_keys=True)),
        }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("wb") as destination:
        np.savez_compressed(destination, **payload)
    temporary.replace(output)
    report = {
        **metadata,
        "positions": int(len(payload["boards"])),
        "output": str(output),
        "output_bytes": output.stat().st_size,
        "output_sha256": file_sha256(output),
    }
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def main():
    print(json.dumps(build(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
