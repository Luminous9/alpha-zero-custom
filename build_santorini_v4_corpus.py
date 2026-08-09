"""Validate V4 oracle shards and convert them into a canonical pilot dataset."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
import time

import numpy as np

from santorini.OracleResearch import stage_for_builds
from santorini.SantoriniGame import SantoriniGame
from santorini.SantoriniOracle import anonymous_board_key
from santorini.V4OracleCorpus import (
    load_v4_shard,
    validate_v4_record,
    validate_v4_record_metadata,
)


OUTPUT_SCHEMA_VERSION = 1
STAGE_IDS = {"early": 0, "middle": 1, "late": 2}
SOURCE_IDS = {"main_line": 0, "randomized_subgame": 1}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Differentially validate and canonicalize V4 datagen JSONL shards."
    )
    parser.add_argument("shards", nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--report-out")
    parser.add_argument("--selection-fraction", type=float, default=0.10)
    parser.add_argument("--test-fraction", type=float, default=0.10)
    parser.add_argument("--split-seed", type=int, default=20260808)
    return parser.parse_args()


def _normalize_worker_labels(board):
    board = np.asarray(board).copy()
    for sign in (1, -1):
        for label, location in enumerate(
            sorted(map(tuple, np.argwhere(board[0] * sign > 0))), start=1
        ):
            board[0][location] = sign * label
    return board


def canonicalize_board_policy(game, board, policy):
    """Choose the minimal D4 board and average policy over its stabilizer."""
    symmetries = game.getSymmetries(board, policy)
    keys = [anonymous_board_key(symmetry_board) for symmetry_board, _ in symmetries]
    canonical_key = min(keys)
    matching = [
        (symmetry_board, symmetry_policy)
        for key, (symmetry_board, symmetry_policy) in zip(keys, symmetries)
        if key == canonical_key
    ]
    canonical_board = _normalize_worker_labels(matching[0][0])
    canonical_policy = np.mean(
        np.asarray([item[1] for item in matching], dtype=np.float64), axis=0
    )
    return canonical_board, canonical_policy, canonical_key


class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))

    def find(self, item):
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, first, second):
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root != second_root:
            self.parent[second_root] = first_root


def _split_id(component_key, seed, selection_fraction, test_fraction):
    digest = hashlib.sha256(
        "{}:{}".format(seed, component_key).encode("utf-8")
    ).digest()
    value = int.from_bytes(digest[:8], "big") / float(2**64)
    if value < test_fraction:
        return 2
    if value < test_fraction + selection_fraction:
        return 1
    return 0


def _compatible_manifest_identity(manifest):
    generation = manifest["generation"]
    return {
        "schema_version": int(manifest["schema_version"]),
        "engine_digest": manifest["engine_digest"],
        "gods": manifest["gods"],
        "tt_policy": manifest["tt_policy"],
        "requested_node_limit": int(generation["requested_node_limit"]),
        "min_depth_node_limit": int(generation["min_depth_node_limit"]),
        "max_completed_depth": int(generation["max_completed_depth"]),
    }


def _generation_distribution(manifest):
    generation = manifest["generation"]
    return {
        "random_moves_min": int(generation["random_moves_min"]),
        "random_moves_max": int(generation["random_moves_max"]),
        "subgame_initial_chance": float(generation["subgame_initial_chance"]),
    }


def _new_aggregate(board, record):
    return {
        "board": board.astype(np.int8),
        "observations": 0,
        "policy": defaultdict(float),
        "winner_sum": 0.0,
        "score_sum": 0.0,
        "score_square_sum": 0.0,
        "actual_nodes_sum": 0.0,
        "mate_count": 0,
        "source_counts": Counter(),
        "root_game_ids": set(),
        "build_count": int(record["build_count"]),
    }


def validate_converted_corpus(path, expected_observations=None):
    started = time.perf_counter()
    with np.load(path, allow_pickle=False) as payload:
        if int(payload["schema_version"][0]) != OUTPUT_SCHEMA_VERSION:
            raise ValueError("Unsupported converted V4 corpus schema.")
        boards = payload["boards"]
        count = len(boards)
        if boards.shape[1:] != (2, 5, 5):
            raise ValueError("Converted V4 boards have the wrong shape.")
        for name in (
            "observation_counts", "winner_means", "score_means", "score_stddevs",
            "requested_nodes", "actual_nodes_means", "mate_rates", "stage_ids",
            "source_counts", "split_ids", "position_hashes",
        ):
            if len(payload[name]) != count:
                raise ValueError("Converted field {} has the wrong length.".format(name))
        offsets = payload["policy_offsets"]
        indices = payload["policy_indices"]
        values = payload["policy_values"]
        if len(offsets) != count + 1 or int(offsets[0]) != 0:
            raise ValueError("Converted policy offsets are malformed.")
        if int(offsets[-1]) != len(indices) or len(indices) != len(values):
            raise ValueError("Converted sparse policy arrays disagree.")
        if np.any(indices >= int(payload["action_size"][0])):
            raise ValueError("Converted policy contains an out-of-range action.")
        for index in range(count):
            start, end = int(offsets[index]), int(offsets[index + 1])
            if start == end or not np.isclose(values[start:end].sum(), 1.0, atol=1e-6):
                raise ValueError("Converted policy mass does not sum to one.")
        observations = int(payload["observation_counts"].sum())
        if expected_observations is not None and observations != int(expected_observations):
            raise ValueError("Converted observation counts do not preserve raw frequency.")
        if int(payload["source_counts"].sum()) != observations:
            raise ValueError("Converted source counts do not preserve raw frequency.")
        if np.any(np.abs(payload["winner_means"]) > 1.0):
            raise ValueError("Converted winner means are outside [-1, 1].")
    return {
        "positions": count,
        "observations": observations,
        "elapsed_seconds": time.perf_counter() - started,
    }


def convert_shards(args):
    if not 0 <= args.selection_fraction < 1 or not 0 <= args.test_fraction < 1:
        raise ValueError("Split fractions must be in [0, 1).")
    if args.selection_fraction + args.test_fraction >= 1:
        raise ValueError("Selection and test fractions must sum to less than one.")
    game = SantoriniGame(5, sequential_placement=True)
    started = time.perf_counter()
    aggregates = {}
    shard_summaries = []
    expected_manifest = None
    generation_distributions = Counter()
    raw_records = 0
    seen_record_ids = set()
    game_winners = {}
    trajectory_ids = set()
    raw_stage_counts = Counter()
    source_stage_counts = Counter()
    random_prefix_counts = Counter()
    completed_depths = []
    raw_scores = []
    actual_nodes = []
    alias_counts = Counter()
    outcomes = Counter()
    stage_score_outcomes = defaultdict(list)

    for path in args.shards:
        manifest, records = load_v4_shard(path, game=None)
        identity = _compatible_manifest_identity(manifest)
        if expected_manifest is None:
            expected_manifest = identity
        elif identity != expected_manifest:
            raise ValueError("V4 shards have incompatible generation identities.")
        distribution = _generation_distribution(manifest)
        generation_distributions[json.dumps(distribution, sort_keys=True)] += len(records)
        expected_records = int(manifest["generation"]["target_records"])
        if len(records) != expected_records:
            raise ValueError(
                "Shard {} contains {} records; its manifest declares {}.".format(
                    manifest["shard_id"], len(records), expected_records
                )
            )

        for record in records:
            record_id = str(record["record_id"])
            if record_id in seen_record_ids:
                raise ValueError("Duplicate record id across V4 shards: {}".format(record_id))
            seen_record_ids.add(record_id)
            game_id = str(record["game_id"])
            trajectory_ids.add(game_id)
            root_game_id = game_id.rsplit(":t", 1)[0]
            winner = int(record["winner"])
            if game_id in game_winners and game_winners[game_id] != winner:
                raise ValueError("Records from one game disagree about the winner.")
            game_winners[game_id] = winner
            board, _ = validate_v4_record_metadata(manifest, record)
            aliases = validate_v4_record(game, manifest, record)
            alias_counts[len(aliases)] += 1
            policy = np.zeros(game.getActionSize(), dtype=np.float64)
            policy[aliases] = 1.0 / len(aliases)
            canonical_board, canonical_policy, canonical_key = canonicalize_board_policy(
                game, board, policy
            )
            aggregate = aggregates.get(canonical_key)
            if aggregate is None:
                aggregate = _new_aggregate(canonical_board, record)
                aggregates[canonical_key] = aggregate
            elif not np.array_equal(aggregate["board"], canonical_board):
                raise AssertionError("Anonymous D4 key collision changed the canonical board.")

            aggregate["observations"] += 1
            for action in np.flatnonzero(canonical_policy):
                aggregate["policy"][int(action)] += float(canonical_policy[action])
            outcome = 1.0 if int(record["winner"]) == int(record["side_to_move"]) else -1.0
            outcomes[int(outcome)] += 1
            score = float(record["score"])
            aggregate["winner_sum"] += outcome
            aggregate["score_sum"] += score
            aggregate["score_square_sum"] += score * score
            aggregate["actual_nodes_sum"] += float(record["actual_nodes"])
            aggregate["mate_count"] += int(bool(record["mate_band"]))
            aggregate["source_counts"][record["source"]] += 1
            aggregate["root_game_ids"].add(root_game_id)
            record_stage = stage_for_builds(int(record["build_count"]))
            raw_stage_counts[record_stage] += 1
            source_stage_counts[(str(record["source"]), record_stage)] += 1
            stage_score_outcomes[record_stage].append((score, int(outcome)))
            random_prefix_counts[int(record["random_prefix_plies"])] += 1
            completed_depths.append(int(record["completed_depth"]))
            raw_scores.append(score)
            actual_nodes.append(int(record["actual_nodes"]))
            raw_records += 1

        shard_summaries.append({
            "path": os.path.abspath(path),
            "shard_id": manifest["shard_id"],
            "records": len(records),
            "differentially_validated": len(records),
            "seed": int(manifest["generation"]["seed"]),
            "worker_index": int(manifest["generation"]["worker_index"]),
        })

    keys = sorted(aggregates)
    key_to_index = {key: index for index, key in enumerate(keys)}
    game_positions = defaultdict(list)
    for key, aggregate in aggregates.items():
        for game_id in aggregate["root_game_ids"]:
            game_positions[game_id].append(key_to_index[key])
    union_find = UnionFind(len(keys))
    for positions in game_positions.values():
        for position in positions[1:]:
            union_find.union(positions[0], position)

    component_members = defaultdict(list)
    for index in range(len(keys)):
        component_members[union_find.find(index)].append(index)
    split_ids = np.zeros(len(keys), dtype=np.int8)
    for members in component_members.values():
        component_key = ":".join(hashlib.sha256(keys[index]).hexdigest() for index in members)
        split = _split_id(
            component_key,
            args.split_seed,
            args.selection_fraction,
            args.test_fraction,
        )
        split_ids[members] = split

    boards = []
    observation_counts = []
    winner_means = []
    score_means = []
    score_stddevs = []
    actual_nodes_means = []
    mate_rates = []
    stages = []
    source_counts = []
    policy_offsets = [0]
    policy_indices = []
    policy_values = []
    position_hashes = []
    for key in keys:
        aggregate = aggregates[key]
        count = aggregate["observations"]
        boards.append(aggregate["board"])
        observation_counts.append(count)
        winner_means.append(aggregate["winner_sum"] / count)
        score_mean = aggregate["score_sum"] / count
        score_means.append(score_mean)
        score_stddevs.append(
            max(0.0, aggregate["score_square_sum"] / count - score_mean**2) ** 0.5
        )
        actual_nodes_means.append(aggregate["actual_nodes_sum"] / count)
        mate_rates.append(aggregate["mate_count"] / count)
        stages.append(STAGE_IDS[stage_for_builds(aggregate["build_count"])])
        source_counts.append([
            aggregate["source_counts"]["main_line"],
            aggregate["source_counts"]["randomized_subgame"],
        ])
        indices = sorted(aggregate["policy"])
        values = np.asarray([aggregate["policy"][index] / count for index in indices])
        if not np.isclose(values.sum(), 1.0, atol=1e-6):
            raise AssertionError("Aggregated policy mass does not sum to one.")
        policy_indices.extend(indices)
        policy_values.extend(values)
        policy_offsets.append(len(policy_indices))
        position_hashes.append(hashlib.sha256(key).hexdigest())

    payload = {
        "schema_version": np.asarray([OUTPUT_SCHEMA_VERSION], dtype=np.int16),
        "action_size": np.asarray([game.getActionSize()], dtype=np.int32),
        "boards": np.asarray(boards, dtype=np.int8),
        "observation_counts": np.asarray(observation_counts, dtype=np.int32),
        "winner_means": np.asarray(winner_means, dtype=np.float32),
        "score_means": np.asarray(score_means, dtype=np.float32),
        "score_stddevs": np.asarray(score_stddevs, dtype=np.float32),
        "requested_nodes": np.full(
            len(keys), expected_manifest["requested_node_limit"], dtype=np.int32
        ),
        "actual_nodes_means": np.asarray(actual_nodes_means, dtype=np.float32),
        "mate_rates": np.asarray(mate_rates, dtype=np.float32),
        "stage_ids": np.asarray(stages, dtype=np.int8),
        "source_counts": np.asarray(source_counts, dtype=np.int32),
        "split_ids": split_ids,
        "policy_offsets": np.asarray(policy_offsets, dtype=np.int64),
        "policy_indices": np.asarray(policy_indices, dtype=np.uint16),
        "policy_values": np.asarray(policy_values, dtype=np.float32),
        "position_hashes": np.asarray(position_hashes, dtype="<U64"),
    }
    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    temporary = output_path + ".tmp.npz"
    np.savez_compressed(temporary, **payload)
    os.replace(temporary, output_path)
    load_validation = validate_converted_corpus(output_path, raw_records)

    elapsed = time.perf_counter() - started
    report = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "output": output_path,
        "engine_digest": expected_manifest["engine_digest"],
        "generation": expected_manifest,
        "generation_distributions": [
            {
                "configuration": json.loads(configuration),
                "records": int(records),
            }
            for configuration, records in sorted(generation_distributions.items())
        ],
        "shards": shard_summaries,
        "raw_records": raw_records,
        "unique_d4_positions": len(keys),
        "duplicate_observations": raw_records - len(keys),
        "duplicate_rate": (raw_records - len(keys)) / raw_records,
        "root_games": len(game_positions),
        "independent_trajectories": len(trajectory_ids),
        "split_components": len(component_members),
        "positions_by_split": {
            name: int(np.sum(split_ids == split))
            for name, split in (("train", 0), ("selection", 1), ("test", 2))
        },
        "positions_by_stage": {
            name: int(np.sum(np.asarray(stages) == stage_id))
            for name, stage_id in STAGE_IDS.items()
        },
        "raw_records_by_stage": dict(sorted(raw_stage_counts.items())),
        "raw_records_by_source_and_stage": {
            source: {
                stage: int(source_stage_counts[(source, stage)])
                for stage in STAGE_IDS
            }
            for source in ("main_line", "randomized_subgame")
        },
        "random_prefix_distribution": {
            str(key): value for key, value in sorted(random_prefix_counts.items())
        },
        "completed_depth": {
            "min": int(min(completed_depths)),
            "median": float(np.median(completed_depths)),
            "max": int(max(completed_depths)),
        },
        "score": {
            "mean": float(np.mean(raw_scores)),
            "median": float(np.median(raw_scores)),
            "mean_absolute": float(np.mean(np.abs(raw_scores))),
        },
        "actual_nodes": {
            "mean": float(np.mean(actual_nodes)),
            "median": float(np.median(actual_nodes)),
        },
        "side_to_move_outcomes": {
            "win": int(outcomes[1]),
            "loss": int(outcomes[-1]),
        },
        "score_sign_outcome_agreement": {
            "all": float(np.mean([
                np.sign(score) == outcome
                for values in stage_score_outcomes.values()
                for score, outcome in values
            ])),
            "by_stage": {
                stage: float(np.mean([
                    np.sign(score) == outcome for score, outcome in values
                ]))
                for stage, values in sorted(stage_score_outcomes.items())
            },
        },
        "policy_alias_count_distribution": {
            str(key): value for key, value in sorted(alias_counts.items())
        },
        "observations_by_source": {
            name: int(np.asarray(source_counts)[:, source_id].sum())
            for name, source_id in SOURCE_IDS.items()
        },
        "max_observations_per_position": int(max(observation_counts)),
        "elapsed_seconds": elapsed,
        "raw_records_per_second": raw_records / elapsed,
        "output_bytes": os.path.getsize(output_path),
        "load_validation": load_validation,
    }
    report_path = args.report_out or output_path + ".report.json"
    temporary_report = report_path + ".tmp"
    with open(temporary_report, "w") as output:
        json.dump(report, output, indent=2, sort_keys=True)
    os.replace(temporary_report, report_path)
    return report


def main():
    args = parse_args()
    report = convert_shards(args)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
