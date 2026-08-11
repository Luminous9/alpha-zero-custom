"""Build an exhaustive Run13 placement-distillation component for V4 P1c.

The compact Run13 replay is deliberately included as the on-distribution part
of the component.  Fresh continuations start from every reachable D4-unique
placement prefix, which closes the replay's severe prefix-coverage gap while
retaining Run13's native search policy and completed-game value semantics.
"""

import argparse
import hashlib
import json
import os
import random
import time
from collections import Counter

import numpy as np
import torch

from Coach import Coach
from santorini.D4Canonical import canonicalize_board, canonicalize_board_policy
from santorini.OracleResearch import file_sha256
from santorini.SantoriniGame import SantoriniGame
from santorini.V4Placement import (
    EXPECTED_ORBITS_BY_WORKER_COUNT,
    enumerate_placement_orbits,
)
from santorini.pytorch import NNet as legacy_nnet
from santorini.pytorch.NNet import V3NNetWrapper
from utils import dotdict


SCHEMA_VERSION = 1
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--replay")
    parser.add_argument("--output", required=True)
    parser.add_argument("--report-out")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--simulations", type=int, default=96)
    parser.add_argument("--fast-simulations", type=int, default=32)
    parser.add_argument("--full-search-probability", type=float, default=0.25)
    parser.add_argument("--continuations-per-state", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument(
        "--max-states",
        type=int,
        help="Debug-only cap over the ordered 960-state coverage suite.",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


class OrderedPlacementSampler:
    def __init__(self, boards):
        self.boards = [np.asarray(board).copy() for board in boards]
        self.offset = 0

    def sample_self_play_board(self):
        if self.offset >= len(self.boards):
            raise RuntimeError("Placement coverage sampler was exhausted.")
        board = self.boards[self.offset]
        self.offset += 1
        return board.copy()


def _decode_replay_policy(payload, index):
    action_size = int(payload["action_size"][0])
    start = int(payload["policy_offsets"][index])
    end = int(payload["policy_offsets"][index + 1])
    policy = np.zeros(action_size, dtype=np.float64)
    policy[payload["policy_indices"][start:end].astype(np.int64)] = (
        payload["policy_values"][start:end]
    )
    mass = float(policy.sum())
    if not np.isclose(mass, 1.0, atol=1e-5):
        raise ValueError("Run13 replay policy {} sums to {}.".format(index, mass))
    return policy / mass


def _add_observation(aggregates, game, board, policy, value, source):
    canonical_board, canonical_policy, canonical_key = canonicalize_board_policy(
        game, board, policy
    )
    valids = game.getValidMoves(canonical_board, 1).astype(bool)
    if np.any(canonical_policy[~valids] > 1e-7):
        raise ValueError("Placement teacher policy assigns mass to an illegal action.")
    canonical_policy = np.where(valids, canonical_policy, 0.0)
    canonical_policy /= canonical_policy.sum()
    record = aggregates.get(canonical_key)
    if record is None:
        record = {
            "board": canonical_board.astype(np.int8),
            "policy_sum": np.zeros(game.getActionSize(), dtype=np.float64),
            "value_sum": 0.0,
            "observation_count": 0,
            "fresh_count": 0,
            "replay_count": 0,
        }
        aggregates[canonical_key] = record
    record["policy_sum"] += canonical_policy
    record["value_sum"] += float(value)
    record["observation_count"] += 1
    record["{}_count".format(source)] += 1


def _add_replay_observations(aggregates, game, replay_path):
    if replay_path is None:
        return 0
    added = 0
    # Accessing an NpzFile field decompresses it every time.  Materialize once
    # because sparse-policy decoding consults the offset/value arrays per row.
    with np.load(replay_path, allow_pickle=False) as archive:
        replay = {name: archive[name] for name in archive.files}
        if int(replay["action_size"][0]) != game.getActionSize():
            raise ValueError("Run13 replay action size is incompatible with V4.")
        for index, board in enumerate(replay["boards"]):
            board = board.astype(int)
            if not game.isPlacementPhase(board):
                continue
            _add_observation(
                aggregates,
                game,
                board,
                _decode_replay_policy(replay, index),
                float(replay["values"][index]),
                "replay",
            )
            added += 1
    return added


def _coach_args(args):
    return dotdict({
        "numMCTSSims": int(args.simulations),
        "tempThreshold": 25,
        "searchMode": "gumbel",
        "gumbelMaxConsideredActions": 16,
        "gumbelScale": 1.0,
        "gumbelPlacementScale": 1.5,
        "placementScaleExplorationProbability": 0.0,
        "cpuct": 1.0,
        "selfPlayBatchSize": int(args.batch_size),
        "quiet": bool(args.quiet),
        "trainingMode": "latest",
        "placementTemperature": 1.0,
        "policyTargetTemperature": 1.0,
        "playoutCapRandomization": True,
        "playoutCapFullProbability": float(args.full_search_probability),
        "playoutCapFastSims": int(args.fast_simulations),
        "playoutCapFullPlacement": True,
        "addDirichletNoise": True,
        "dirichletAlpha": 0.30,
        "dirichletEpsilon": 0.25,
        "tacticalShortcuts": True,
        "symmetryAugmentation": "on-the-fly",
        "searchSymmetryEvaluation": True,
        "inferenceDeduplication": True,
        "inferenceCacheSize": 4096,
        "rootSymmetrySamples": 2,
        "placementRootSymmetrySamples": 8,
        "telemetryMatchGames": 0,
        "telemetryPlacementGames": 0,
    })


def _load_teacher(game, checkpoint, use_cuda):
    legacy_nnet.args.cuda = bool(use_cuda)
    legacy_nnet.args.quiet = True
    teacher = V3NNetWrapper(game)
    checkpoint = os.path.abspath(checkpoint)
    teacher.load_checkpoint(os.path.dirname(checkpoint), os.path.basename(checkpoint))
    return teacher


def _fresh_observations(args, game, coverage_boards):
    repeated = [
        board
        for _ in range(int(args.continuations_per_state))
        for board in coverage_boards
    ]
    rng = np.random.RandomState(int(args.seed))
    order = rng.permutation(len(repeated))
    repeated = [repeated[int(index)] for index in order]
    sampler = OrderedPlacementSampler(repeated)
    use_cuda = args.device == "cuda" or (
        args.device == "auto" and torch.cuda.is_available()
    )
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable.")
    teacher = _load_teacher(game, args.checkpoint, use_cuda)
    coach = Coach(game, teacher, _coach_args(args), opening_sampler=sampler)
    return coach.executeEpisodesBatched(len(repeated)), use_cuda


def _materialize(args, game, coverage_boards, fresh_examples, use_cuda, started):
    aggregates = {}
    replay_observations = _add_replay_observations(aggregates, game, args.replay)
    fresh_observations = 0
    for board, policy, value in fresh_examples:
        if not game.isPlacementPhase(board):
            continue
        _add_observation(aggregates, game, board, policy, value, "fresh")
        fresh_observations += 1

    expected_keys = {canonicalize_board(board)[2] for board in coverage_boards}
    missing = expected_keys.difference(aggregates)
    if missing:
        raise ValueError("Fresh placement search missed {} coverage states.".format(len(missing)))

    keys = sorted(aggregates)
    boards = []
    observation_counts = []
    fresh_counts = []
    replay_counts = []
    winner_means = []
    worker_counts = []
    position_hashes = []
    policy_offsets = [0]
    policy_indices = []
    policy_values = []
    for key in keys:
        record = aggregates[key]
        count = int(record["observation_count"])
        policy = record["policy_sum"] / count
        policy /= policy.sum()
        nonzero = np.flatnonzero(policy > 0)
        boards.append(record["board"])
        observation_counts.append(count)
        fresh_counts.append(record["fresh_count"])
        replay_counts.append(record["replay_count"])
        winner_means.append(record["value_sum"] / count)
        worker_counts.append(int(np.count_nonzero(record["board"][0])))
        position_hashes.append(hashlib.sha256(key).hexdigest())
        policy_indices.extend(map(int, nonzero))
        policy_values.extend(map(float, policy[nonzero]))
        policy_offsets.append(len(policy_indices))

    count = len(boards)
    payload = {
        "schema_version": np.asarray([SCHEMA_VERSION], dtype=np.int16),
        "action_size": np.asarray([game.getActionSize()], dtype=np.int32),
        "boards": np.asarray(boards, dtype=np.int8),
        "observation_counts": np.asarray(observation_counts, dtype=np.int32),
        "fresh_observation_counts": np.asarray(fresh_counts, dtype=np.int32),
        "replay_observation_counts": np.asarray(replay_counts, dtype=np.int32),
        "winner_means": np.asarray(winner_means, dtype=np.float32),
        "has_completed_outcomes": np.asarray([True]),
        "score_means": np.zeros(count, dtype=np.float32),
        "score_stddevs": np.zeros(count, dtype=np.float32),
        "requested_nodes": np.zeros(count, dtype=np.int32),
        "actual_nodes_means": np.zeros(count, dtype=np.float32),
        "mate_rates": np.zeros(count, dtype=np.float32),
        "completed_depths": np.zeros(count, dtype=np.int16),
        "stage_ids": np.full(count, -1, dtype=np.int8),
        "split_ids": np.zeros(count, dtype=np.int8),
        "replay_indices": np.full(count, -1, dtype=np.int32),
        "worker_counts": np.asarray(worker_counts, dtype=np.int8),
        "teacher_simulations": np.full(count, int(args.simulations), dtype=np.int16),
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

    worker_counter = Counter(worker_counts)
    report = {
        "schema_version": SCHEMA_VERSION,
        "contract": "p1c_run13_placement_distillation",
        "output": output_path,
        "checkpoint": os.path.abspath(args.checkpoint),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "replay": os.path.abspath(args.replay) if args.replay else None,
        "replay_sha256": file_sha256(args.replay) if args.replay else None,
        "device": "cuda" if use_cuda else "cpu",
        "seed": int(args.seed),
        "coverage_states_requested": len(coverage_boards),
        "continuations_per_state": int(args.continuations_per_state),
        "continuations": len(coverage_boards) * int(args.continuations_per_state),
        "unique_positions": count,
        "unique_positions_by_worker_count": {
            str(worker_count): int(worker_counter[worker_count])
            for worker_count in range(4)
        },
        "fresh_placement_observations": fresh_observations,
        "replay_placement_observations": replay_observations,
        "aggregated_placement_observations": int(sum(observation_counts)),
        "search": dict(_coach_args(args)),
        "elapsed_seconds": float(time.perf_counter() - started),
        "output_bytes": os.path.getsize(output_path),
        "output_sha256": file_sha256(output_path),
    }
    report_path = os.path.abspath(args.report_out or output_path + ".report.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    temporary_report = report_path + ".tmp"
    with open(temporary_report, "w") as output:
        json.dump(report, output, indent=2, sort_keys=True)
    os.replace(temporary_report, report_path)
    return report


def build_component(args):
    if args.simulations < 1 or args.fast_simulations < 1:
        raise ValueError("Search simulation counts must be positive.")
    if args.fast_simulations >= args.simulations and args.simulations > 1:
        raise ValueError("Fast simulations must be below full simulations.")
    if args.continuations_per_state < 1 or args.batch_size < 1:
        raise ValueError("Continuation and batch counts must be positive.")
    if not 0.0 < args.full_search_probability <= 1.0:
        raise ValueError("Full-search probability must be in (0, 1].")
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    game = SantoriniGame(5, sequential_placement=True)
    coverage_boards = enumerate_placement_orbits(game)
    if args.max_states is not None:
        if args.max_states < 1:
            raise ValueError("--max-states must be positive.")
        coverage_boards = coverage_boards[: int(args.max_states)]
    started = time.perf_counter()
    fresh_examples, use_cuda = _fresh_observations(args, game, coverage_boards)
    return _materialize(
        args,
        game,
        coverage_boards,
        fresh_examples,
        use_cuda,
        started,
    )


def main():
    print(json.dumps(build_component(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
