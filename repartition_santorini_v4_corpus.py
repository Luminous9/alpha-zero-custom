"""Anchor a scaled V4 corpus to an existing frozen selection/test split.

The original converter groups positions by connected components of root games.
That is exact for a small corpus, but repeated positions make the graph develop a
giant component at scale.  This tool keeps the already-frozen holdouts instead:

* every root game that directly visits a frozen holdout position is blocked;
* every position visited by a blocked root game is removed from the scaled corpus;
* all remaining positions are assigned to the training split.

The input corpus must already have been fully validated by
``build_santorini_v4_corpus.py``.  Its immutable raw shards are scanned only to
recover root-game provenance, so the expensive differential validation and
aggregation do not need to be repeated.
"""

import argparse
from collections import Counter
import hashlib
import json
import os
import sys
import time

import numpy as np

from build_santorini_v4_corpus import validate_converted_corpus
from santorini.D4Canonical import canonicalize_board
from santorini.SantoriniOracle import fen_to_canonical_board
from santorini.V4OracleCorpus import validate_v4_manifest


POSITION_FIELDS = (
    "boards",
    "observation_counts",
    "winner_means",
    "score_means",
    "score_stddevs",
    "requested_nodes",
    "actual_nodes_means",
    "mate_rates",
    "stage_ids",
    "source_counts",
    "position_hashes",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shards", nargs="+")
    parser.add_argument("--input-corpus", required=True)
    parser.add_argument("--frozen-holdout-corpus", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report-out")
    return parser.parse_args()


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _position_hash(fen):
    board = fen_to_canonical_board(fen)
    _, _, key = canonicalize_board(board)
    return hashlib.sha256(key).hexdigest()


def _validate_record_identity(manifest, record):
    if record.get("type") != "position":
        raise ValueError("V4 corpus data lines must have type=position.")
    for field in ("schema_version", "engine_digest", "shard_id", "tt_policy"):
        if record.get(field) != manifest[field]:
            raise ValueError("V4 record {} does not match its manifest.".format(field))
    game_id = str(record.get("game_id"))
    if not game_id.startswith(str(manifest["shard_id"]) + ":"):
        raise ValueError("V4 record game_id does not belong to its shard.")
    return game_id.rsplit(":t", 1)[0]


def scan_blocked_provenance(shards, heldout_hashes):
    """Return hashes visited by root games that directly touch a holdout."""
    blocked_hashes = set()
    blocked_games = 0
    blocked_records = 0
    raw_records = 0
    root_games = 0
    expected_identity = None
    shard_summaries = []

    for path in shards:
        shard_started = time.perf_counter()
        with open(path) as source:
            first_line = source.readline()
            if not first_line:
                raise ValueError("V4 corpus shard is empty: {}".format(path))
            manifest = validate_v4_manifest(json.loads(first_line))
            identity = {
                field: manifest[field]
                for field in ("schema_version", "engine_digest", "gods", "tt_policy")
            }
            if expected_identity is None:
                expected_identity = identity
            elif identity != expected_identity:
                raise ValueError("V4 shards have incompatible generation identities.")

            current_game = None
            current_hashes = set()
            current_records = 0
            completed_games = set()
            shard_records = 0

            def flush_game():
                nonlocal blocked_games, blocked_records, root_games
                if current_game is None:
                    return
                root_games += 1
                if current_hashes & heldout_hashes:
                    blocked_games += 1
                    blocked_records += current_records
                    blocked_hashes.update(current_hashes)

            for line in source:
                if not line.strip():
                    continue
                record = json.loads(line)
                root_game_id = _validate_record_identity(manifest, record)
                if root_game_id != current_game:
                    flush_game()
                    if root_game_id in completed_games:
                        raise ValueError(
                            "Root-game records are not contiguous in {}: {}".format(
                                path, root_game_id
                            )
                        )
                    if current_game is not None:
                        completed_games.add(current_game)
                    current_game = root_game_id
                    current_hashes = set()
                    current_records = 0
                current_hashes.add(_position_hash(record["fen"]))
                current_records += 1
                shard_records += 1
            flush_game()

        expected_records = int(manifest["generation"]["target_records"])
        if shard_records != expected_records:
            raise ValueError(
                "Shard {} contains {} records; its manifest declares {}.".format(
                    manifest["shard_id"], shard_records, expected_records
                )
            )
        raw_records += shard_records
        shard_summaries.append({
            "path": os.path.abspath(path),
            "shard_id": manifest["shard_id"],
            "records": shard_records,
            "elapsed_seconds": time.perf_counter() - shard_started,
        })
        print(
            "scanned {}/{} shards: {:,} records, {:,} blocked root games".format(
                len(shard_summaries), len(shards), raw_records, blocked_games
            ),
            file=sys.stderr,
            flush=True,
        )

    return {
        "blocked_hashes": blocked_hashes,
        "blocked_root_games": blocked_games,
        "blocked_raw_records": blocked_records,
        "raw_records": raw_records,
        "root_games": root_games,
        "shards": shard_summaries,
        "engine_digest": expected_identity["engine_digest"],
    }


def _subset_payload(source, retained_indices):
    payload = {
        "schema_version": source["schema_version"],
        "action_size": source["action_size"],
    }
    for field in POSITION_FIELDS:
        payload[field] = source[field][retained_indices]
    payload["split_ids"] = np.zeros(len(retained_indices), dtype=np.int8)

    old_offsets = source["policy_offsets"]
    all_lengths = np.diff(old_offsets)
    lengths = all_lengths[retained_indices]
    position_mask = np.zeros(len(all_lengths), dtype=bool)
    position_mask[retained_indices] = True
    policy_mask = np.repeat(position_mask, all_lengths)
    payload["policy_offsets"] = np.concatenate((
        np.asarray([0], dtype=np.int64),
        np.cumsum(lengths, dtype=np.int64),
    ))
    payload["policy_indices"] = source["policy_indices"][policy_mask]
    payload["policy_values"] = source["policy_values"][policy_mask]
    return payload


def repartition(args):
    started = time.perf_counter()
    input_path = os.path.abspath(args.input_corpus)
    frozen_path = os.path.abspath(args.frozen_holdout_corpus)
    with np.load(frozen_path, allow_pickle=False) as frozen:
        frozen_splits = frozen["split_ids"]
        frozen_hashes = frozen["position_hashes"]
        heldout_hashes = set(map(str, frozen_hashes[frozen_splits != 0]))
        frozen_counts = {
            name: int(np.sum(frozen_splits == split))
            for name, split in (("train", 0), ("selection", 1), ("test", 2))
        }
    if not heldout_hashes:
        raise ValueError("The frozen corpus contains no selection/test positions.")

    provenance = scan_blocked_provenance(args.shards, heldout_hashes)
    forbidden_hashes = heldout_hashes | provenance["blocked_hashes"]

    with np.load(input_path, allow_pickle=False) as source:
        input_hashes = source["position_hashes"]
        retained_mask = np.asarray([
            str(position_hash) not in forbidden_hashes
            for position_hash in input_hashes
        ], dtype=bool)
        retained_indices = np.flatnonzero(retained_mask)
        payload = _subset_payload(source, retained_indices)
        input_positions = len(input_hashes)
        input_observations = int(source["observation_counts"].sum())
        input_split_counts = {
            name: int(np.sum(source["split_ids"] == split))
            for name, split in (("train", 0), ("selection", 1), ("test", 2))
        }
        exact_heldout_positions = int(np.sum([
            str(position_hash) in heldout_hashes for position_hash in input_hashes
        ]))

    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    temporary = output_path + ".tmp.npz"
    np.savez_compressed(temporary, **payload)
    os.replace(temporary, output_path)
    retained_observations = int(payload["observation_counts"].sum())
    validation = validate_converted_corpus(output_path, retained_observations)

    stage_counts = Counter(map(int, payload["stage_ids"]))
    source_counts = np.asarray(payload["source_counts"])
    report = {
        "schema_version": 1,
        "mode": "frozen_holdout_anchored_training_only",
        "input_corpus": input_path,
        "input_corpus_sha256": _file_sha256(input_path),
        "frozen_holdout_corpus": frozen_path,
        "frozen_holdout_corpus_sha256": _file_sha256(frozen_path),
        "output": output_path,
        "engine_digest": provenance["engine_digest"],
        "frozen_positions_by_split": frozen_counts,
        "frozen_holdout_hashes": len(heldout_hashes),
        "raw_records_scanned": provenance["raw_records"],
        "root_games_scanned": provenance["root_games"],
        "blocked_root_games": provenance["blocked_root_games"],
        "blocked_raw_records": provenance["blocked_raw_records"],
        "blocked_game_position_hashes": len(provenance["blocked_hashes"]),
        "input_positions": input_positions,
        "input_observations": input_observations,
        "input_positions_by_split": input_split_counts,
        "exact_frozen_holdout_positions_in_input": exact_heldout_positions,
        "removed_positions": input_positions - len(retained_indices),
        "removed_observations": input_observations - retained_observations,
        "retained_positions": len(retained_indices),
        "retained_observations": retained_observations,
        "positions_by_split": {
            "train": len(retained_indices), "selection": 0, "test": 0,
        },
        "positions_by_stage": {
            name: int(stage_counts[stage])
            for name, stage in (("early", 0), ("middle", 1), ("late", 2))
        },
        "observations_by_source": {
            "main_line": int(source_counts[:, 0].sum()),
            "randomized_subgame": int(source_counts[:, 1].sum()),
        },
        "shards": provenance["shards"],
        "load_validation": validation,
        "elapsed_seconds": time.perf_counter() - started,
        "output_bytes": os.path.getsize(output_path),
    }
    report_path = args.report_out or output_path + ".report.json"
    temporary_report = report_path + ".tmp"
    with open(temporary_report, "w") as output:
        json.dump(report, output, indent=2, sort_keys=True)
    os.replace(temporary_report, report_path)
    return report


def main():
    print(json.dumps(repartition(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
