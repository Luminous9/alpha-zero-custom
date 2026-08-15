"""Run reusable duplicate-aware V4 placement checkpoint arenas."""

import argparse
import hashlib
import json
import os
import time

import numpy as np
import torch

from BatchedArena import BatchedMCTSArena
from arena_santorini_v4_p2_arm import _arena_payload, _device, _search_args
from santorini.PlacementArenaAnalysis import (
    collect_d4_unique_placements,
    run_duplicate_aware_matchup,
)
from santorini.SantoriniGame import SantoriniGame
from santorini.pytorch.V4NNet import V4InferenceWrapper


DEFAULT_SEED = 20260821


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--mode',
        choices=('natural', 'unique-learned'),
        required=True,
        help=(
            'natural samples both contestants\' placement policies and reports '
            'duplicate-aware views; unique-learned freezes a D4-unique suite '
            'from one declared placement source.'
        ),
    )
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--baseline-label', default='baseline')
    parser.add_argument('--candidate-label', default='candidate')
    parser.add_argument('--games', type=int, default=120)
    parser.add_argument('--openings', type=int, default=60)
    parser.add_argument(
        '--placement-source',
        choices=('baseline', 'candidate'),
        default='candidate',
    )
    parser.add_argument('--simulations', type=int, default=96)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--sample-batch-size', type=int, default=120)
    parser.add_argument('--max-placement-occurrences', type=int, default=1200)
    parser.add_argument('--bootstrap-samples', type=int, default=10_000)
    parser.add_argument('--d4-cap-fraction', type=float, default=0.05)
    parser.add_argument('--minimum-d4-ess-ratio', type=float, default=0.75)
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--output', required=True)
    return parser.parse_args()


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _suite_sha256(records):
    digest = hashlib.sha256()
    for record in records:
        digest.update(record['opening_board'].tobytes())
    return digest.hexdigest()


def _atomic_json(path, payload):
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + '.tmp'
    with open(temporary, 'w') as output:
        json.dump(payload, output, indent=2, sort_keys=True)
    os.replace(temporary, path)


def _checkpoint_payload(path, label):
    return {
        'label': str(label),
        'path': os.path.abspath(path),
        'sha256': _file_sha256(path),
    }


def _placement_seeds(seed, pairs):
    rng = np.random.RandomState(int(seed))
    result = []
    seen = set()
    while len(result) < int(pairs):
        candidate = int(rng.randint(0, 2 ** 31 - 1))
        if candidate not in seen:
            seen.add(candidate)
            result.append(candidate)
    return result


def _common_payload(args):
    return {
        'schema_version': 1,
        'checkpoints': {
            'baseline': _checkpoint_payload(args.baseline, args.baseline_label),
            'candidate': _checkpoint_payload(args.candidate, args.candidate_label),
        },
        'score_owner': 'candidate',
        'simulations': int(args.simulations),
        'search_mode': 'gumbel',
        'gumbel_scale': 0.0,
        'placement_gumbel_scale': 1.5,
        'placement_temperature': 1.0,
        'canonical_d4': True,
        'seed': int(args.seed),
        'final_test_touched': False,
        'final_arena_seeds_touched': False,
    }


def _run_natural(game, baseline, candidate, search_args, args):
    if args.games < 2 or args.games % 2:
        raise ValueError('--games must be a positive even number in natural mode.')
    game_seeds = _placement_seeds(args.seed, args.games // 2)
    result = run_duplicate_aware_matchup(
        game=game,
        contestant1=baseline,
        contestant2=candidate,
        search_args=search_args,
        games=args.games,
        batch_size=args.batch_size,
        game_seeds=game_seeds,
        seed=args.seed,
        quiet=False,
        bootstrap_samples=args.bootstrap_samples,
        d4_cap_fraction=args.d4_cap_fraction,
        minimum_d4_ess_ratio=args.minimum_d4_ess_ratio,
    )
    payload = _common_payload(args)
    payload.update({
        'type': 'santorini_v4_duplicate_aware_natural_placement_arena',
        'games': int(args.games),
        'pairs': int(args.games // 2),
        'game_seeds': game_seeds,
        'duplicate_contract': {
            'deterministic_continuations_executed_once': True,
            'raw_policy_score_preserves_sampled_multiplicity': True,
            'd4_cap_fraction': float(args.d4_cap_fraction),
            'minimum_d4_ess_ratio': float(args.minimum_d4_ess_ratio),
            'bootstrap_samples': int(args.bootstrap_samples),
        },
        'result': result,
    })
    return payload, {
        'raw_score': result['raw_policy_weighted']['score'],
        'capped_d4_score': result['capped_d4']['score'],
        'd4_opening_ess': result['diversity']['d4_opening_ess'],
        'unique_continuations_executed': result['execution'][
            'unique_continuations_executed'
        ],
    }


def _run_unique_learned(game, baseline, candidate, search_args, args):
    if args.openings < 1:
        raise ValueError('--openings must be positive in unique-learned mode.')
    if args.max_placement_occurrences < args.openings:
        raise ValueError('Placement occurrence budget cannot be below opening count.')
    source = baseline if args.placement_source == 'baseline' else candidate
    source_label = (
        args.baseline_label
        if args.placement_source == 'baseline'
        else args.candidate_label
    )
    suite, collection = collect_d4_unique_placements(
        game=game,
        controller=source,
        search_args=search_args,
        target_openings=args.openings,
        batch_size=args.batch_size,
        seed=args.seed,
        max_occurrences=args.max_placement_occurrences,
        sample_batch_size=args.sample_batch_size,
        quiet=False,
    )
    if len({record['d4_opening_key'] for record in suite}) != args.openings:
        raise AssertionError('Frozen learned-opening suite is not D4-unique.')
    arena = BatchedMCTSArena(
        game,
        baseline,
        candidate,
        search_args,
        batch_size=args.batch_size,
        quiet=False,
        opening_boards=[record['opening_board'] for record in suite],
        game_seeds=[args.seed ^ 0x7110 ^ index for index in range(args.openings)],
        record_placement_diagnostics=True,
    )
    started = time.perf_counter()
    result = _arena_payload(arena, 2 * args.openings, args.seed ^ 0x7111, started)
    payload = _common_payload(args)
    payload.update({
        'type': 'santorini_v4_d4_unique_learned_opening_arena',
        'placement_source': args.placement_source,
        'placement_source_label': str(source_label),
        'opening_selection': 'first_discovered_d4_unique_families',
        'openings': int(args.openings),
        'games': int(2 * args.openings),
        'suite_sha256': _suite_sha256(suite),
        'collection': collection,
        'suite': [
            {
                **{key: value for key, value in record.items() if key != 'opening_board'},
                'opening_board': record['opening_board'].tolist(),
                'placement_actions': list(record['placement_actions']),
            }
            for record in suite
        ],
        'result': result,
    })
    return payload, {
        'suite_sha256': payload['suite_sha256'],
        'sampled_placement_occurrences': collection['sampled_occurrences'],
        'candidate_score': result['current_score'],
        'paired': {
            key: result['paired'][key]
            for key in (
                'pair_wins', 'pair_splits', 'pair_losses',
                'cluster_bootstrap_95_low', 'cluster_bootstrap_95_high',
            )
        },
    }


def main():
    args = parse_args()
    if args.simulations < 1 or args.batch_size < 1 or args.sample_batch_size < 1:
        raise ValueError('Simulation and batch sizes must be positive.')
    if not 0 < args.d4_cap_fraction <= 1:
        raise ValueError('--d4-cap-fraction must be in (0, 1].')
    if not 0 < args.minimum_d4_ess_ratio <= 2:
        raise ValueError('--minimum-d4-ess-ratio must be in (0, 2].')

    device = _device(args.device)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
    game = SantoriniGame(5, sequential_placement=True)
    baseline = V4InferenceWrapper(
        game, args.baseline, device=device, freeze_torchscript=True,
        canonicalize_d4=True,
    )
    candidate = V4InferenceWrapper(
        game, args.candidate, device=device, freeze_torchscript=True,
        canonicalize_d4=True,
    )
    search_args = _search_args(args.simulations)
    if args.mode == 'natural':
        payload, console = _run_natural(
            game, baseline, candidate, search_args, args
        )
    else:
        payload, console = _run_unique_learned(
            game, baseline, candidate, search_args, args
        )
    _atomic_json(args.output, payload)
    print(json.dumps({
        'mode': args.mode,
        'output': os.path.abspath(args.output),
        **console,
    }, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
