"""Run fixed paired P2 milestone arenas against a declared anchor."""

import argparse
import json
import math
import os
import time

import numpy as np
import torch

from BatchedArena import BatchedMCTSArena
from santorini.SantoriniGame import SantoriniGame
from santorini.SantoriniOpeningBook import SantoriniRandomOpeningSampler
from santorini.pytorch.V4NNet import V4InferenceWrapper
from utils import dotdict


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--anchor', required=True)
    parser.add_argument('--anchor-iteration', type=int, default=1)
    parser.add_argument('--current', required=True)
    parser.add_argument('--current-iteration', type=int, required=True)
    parser.add_argument('--games', type=int, default=40)
    parser.add_argument('--simulations', type=int, default=96)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=20260715)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--output', required=True)
    return parser.parse_args()


def _device(name):
    if name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if name == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable.')
    return torch.device(name)


def _search_args(simulations):
    return dotdict({
        'numMCTSSims': int(simulations),
        'cpuct': 1.0,
        'searchMode': 'gumbel',
        'gumbelMaxConsideredActions': 16,
        'gumbelScale': 0.0,
        'gumbelPlacementScale': 1.5,
        'tacticalShortcuts': True,
        'searchSymmetryEvaluation': False,
        'rootSymmetrySamples': 1,
        'placementRootSymmetrySamples': 1,
        'inferenceDeduplication': True,
        'inferenceCacheSize': 4096,
    })


def _interval(wins, games):
    if not games:
        return None, None
    rate = wins / games
    z = 1.959963984540054
    denominator = 1.0 + z * z / games
    center = (rate + z * z / (2.0 * games)) / denominator
    half = z * math.sqrt(
        rate * (1.0 - rate) / games + z * z / (4.0 * games * games)
    ) / denominator
    return float(center - half), float(center + half)


def _paired_current_statistics(records, seed, bootstrap_samples=10_000):
    grouped = {}
    for record in records:
        grouped.setdefault(int(record['pair_index']), []).append(record)
    scores = []
    pair_records = []
    for pair_index, group in sorted(grouped.items()):
        if len(group) != 2:
            raise ValueError('Arena did not produce exactly two games per pair.')
        current_score = sum(
            1.0 if record['winner'] == -1 else 0.5 if record['winner'] == 0 else 0.0
            for record in group
        )
        scores.append(current_score)
        pair_records.append({
            'pair_index': pair_index,
            'current_score': current_score,
            'games': group,
        })
    scores = np.asarray(scores, dtype=np.float64)
    rng = np.random.RandomState(int(seed) ^ 0x5A17)
    bootstrap = scores[
        rng.randint(len(scores), size=(bootstrap_samples, len(scores)))
    ].mean(axis=1) / 2.0
    return {
        'pairs': len(scores),
        'pair_wins': int(np.sum(scores > 1.0)),
        'pair_splits': int(np.sum(scores == 1.0)),
        'pair_losses': int(np.sum(scores < 1.0)),
        'current_game_score': float(np.mean(scores) / 2.0),
        'cluster_bootstrap_95_low': float(np.quantile(bootstrap, 0.025)),
        'cluster_bootstrap_95_high': float(np.quantile(bootstrap, 0.975)),
        'records': pair_records,
    }


def _arena_payload(arena, games, seed, started):
    anchor_wins, current_wins, draws = arena.playGames(games)
    low, high = _interval(current_wins + 0.5 * draws, games)
    return {
        'games': int(games),
        'anchor_wins': int(anchor_wins),
        'current_wins': int(current_wins),
        'draws': int(draws),
        'current_score': float((current_wins + 0.5 * draws) / games),
        'current_score_wilson_95_low': low,
        'current_score_wilson_95_high': high,
        'paired': _paired_current_statistics(arena.game_records, seed),
        'inference': arena.inferenceDiagnostics(),
        'elapsed_seconds': float(time.perf_counter() - started),
    }


def _atomic_json(path, payload):
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + '.tmp'
    with open(temporary, 'w') as output:
        json.dump(payload, output, indent=2, sort_keys=True)
    os.replace(temporary, path)


def main():
    args = parse_args()
    if args.games < 2 or args.games % 2:
        raise ValueError('--games must be a positive even number.')
    if args.simulations < 1 or args.batch_size < 1:
        raise ValueError('Simulation and batch sizes must be positive.')
    device = _device(args.device)
    game = SantoriniGame(5, sequential_placement=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
    anchor = V4InferenceWrapper(
        game, args.anchor, device=device, freeze_torchscript=True,
        canonicalize_d4=True,
    )
    current = V4InferenceWrapper(
        game, args.current, device=device, freeze_torchscript=True,
        canonicalize_d4=True,
    )
    search_args = _search_args(args.simulations)
    opening_count = args.games // 2
    openings = SantoriniRandomOpeningSampler(
        board_size=5,
        random_orientation=True,
        rng=np.random.RandomState(args.seed),
    ).sample_distinct_arena_suite(opening_count)
    standard = BatchedMCTSArena(
        game, anchor, current, search_args,
        batch_size=args.batch_size,
        quiet=False,
        opening_boards=openings,
        game_seeds=[args.seed + index for index in range(opening_count)],
    )
    started = time.perf_counter()
    standard_payload = _arena_payload(standard, args.games, args.seed, started)

    placement_seed_rng = np.random.RandomState(args.seed + 1)
    placement_seeds = []
    while len(placement_seeds) < opening_count:
        candidate = int(placement_seed_rng.randint(0, 2 ** 31 - 1))
        if candidate not in placement_seeds:
            placement_seeds.append(candidate)
    placement = BatchedMCTSArena(
        game, anchor, current, search_args,
        batch_size=args.batch_size,
        quiet=False,
        placement_temperature=1.0,
        game_seeds=placement_seeds,
        record_placement_diagnostics=True,
    )
    started = time.perf_counter()
    placement_payload = _arena_payload(
        placement, args.games, args.seed + 1, started
    )
    placement_payload['placement_diagnostics'] = placement.placementDiagnostics()

    payload = {
        'schema_version': 1,
        'type': 'santorini_v4_p2_arm_arena',
        'anchor_iteration': int(args.anchor_iteration),
        'current_iteration': int(args.current_iteration),
        'anchor_checkpoint': os.path.abspath(args.anchor),
        'current_checkpoint': os.path.abspath(args.current),
        'simulations': int(args.simulations),
        'search_mode': 'gumbel',
        'gumbel_scale': 0.0,
        'placement_gumbel_scale': 1.5,
        'canonical_d4': True,
        'standard_opening_seed': int(args.seed),
        'placement_seed': int(args.seed + 1),
        'standard': standard_payload,
        'placement_inclusive': placement_payload,
        'final_test_touched': False,
        'final_arena_seeds_touched': False,
    }
    _atomic_json(args.output, payload)
    console = {
        'output': os.path.abspath(args.output),
        'current_iteration': args.current_iteration,
        'standard': {
            key: standard_payload[key]
            for key in ('anchor_wins', 'current_wins', 'draws', 'current_score')
        },
        'placement_inclusive': {
            key: placement_payload[key]
            for key in ('anchor_wins', 'current_wins', 'draws', 'current_score')
        },
    }
    print(json.dumps(console, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
