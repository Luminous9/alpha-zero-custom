#!/usr/bin/env python3
"""Measure raw-network and deterministic-search D4 consistency in standard play."""

import argparse
import json
import os

import numpy as np

from MCTS import MCTS
from pit_santorini import search_args
from santorini.SantoriniGame import SantoriniGame
from santorini.pytorch.NNet import build_nnet


STAGES = (
    ('early', 0, 5),
    ('middle', 6, 15),
    ('late', 16, None),
)


def parse_model(value):
    if '=' not in value:
        raise argparse.ArgumentTypeError('Model must be NAME=CHECKPOINT_PATH.')
    name, path = value.split('=', 1)
    return name, path


def transform_board(board, rotations, flip):
    transformed = np.rot90(board, rotations, axes=(-2, -1))
    return np.flip(transformed, axis=-1).copy() if flip else transformed.copy()


def transformations(board):
    return [
        (rotations, flip, transform_board(board, rotations, flip))
        for rotations in range(4)
        for flip in (False, True)
    ]


def transform_policy(game, policy, rotations, flip):
    old_indices, new_indices = game.getPolicySymmetryPermutation(rotations, flip)
    transformed = np.zeros_like(policy)
    transformed[new_indices] = policy[old_indices]
    return transformed


def transform_action(game, action, rotations, flip):
    old_indices, new_indices = game.getPolicySymmetryPermutation(rotations, flip)
    positions = np.flatnonzero(old_indices == int(action))
    return int(new_indices[int(positions[0])])


def normalized_legal_policy(game, board, policy):
    valids = game.getValidMoves(board, 1).astype(bool)
    legal = np.where(valids, np.asarray(policy, dtype=np.float64), 0.0)
    if legal.sum():
        legal /= legal.sum()
    return legal


def has_full_symmetry_orbit(board):
    return len({candidate.tobytes() for _, _, candidate in transformations(board)}) == 8


def sample_stage_boards(replay_path, count, seed):
    with np.load(replay_path, allow_pickle=False) as payload:
        boards = payload['boards'].astype(int)
    build_counts = boards[:, 1].sum(axis=(1, 2))
    complete = np.count_nonzero(boards[:, 0], axis=(1, 2)) == 4
    rng = np.random.RandomState(seed)
    sampled = {}
    for name, low, high in STAGES:
        mask = complete & (build_counts >= low)
        if high is not None:
            mask &= build_counts <= high
        candidates = np.flatnonzero(mask)
        rng.shuffle(candidates)
        selected = []
        for index in candidates:
            board = boards[int(index)]
            if has_full_symmetry_orbit(board):
                selected.append(board)
            if len(selected) == count:
                break
        if len(selected) < count:
            raise ValueError('Could only find {} {} positions.'.format(len(selected), name))
        sampled[name] = selected
    return sampled


def raw_position_metrics(game, nnet, board):
    variants = transformations(board)
    policies, values = nnet.predict_batch([variant[2] for variant in variants])
    legal = [
        normalized_legal_policy(game, variant[2], policy)
        for variant, policy in zip(variants, policies)
    ]
    base = legal[0]
    base_action = int(np.argmax(base))
    comparisons = []
    for index, (rotations, flip, _) in enumerate(variants[1:], start=1):
        expected = transform_policy(game, base, rotations, flip)
        difference = np.abs(legal[index] - expected)
        observed_action = int(np.argmax(legal[index]))
        expected_action = transform_action(game, base_action, rotations, flip)
        comparisons.append({
            'total_variation': float(0.5 * difference.sum()),
            'max_action_probability_error': float(difference.max()),
            'top_action_consistent': observed_action == expected_action,
            'value_absolute_error': float(abs(float(values[index]) - float(values[0]))),
        })
    return comparisons


def searched_actions(game, nnet, boards, simulations):
    args = search_args(
        simulations,
        search_mode='gumbel',
        gumbel_max_considered_actions=16,
        gumbel_scale=0.0,
        gumbel_placement_scale=0.0,
    )
    searches = [MCTS(game, nnet, args) for _ in boards]
    actions = [None] * len(boards)
    active = []
    for index, (mcts, board) in enumerate(zip(searches, boards)):
        tactical = mcts.prepareTacticalRoot(board)
        if tactical is not None and tactical['policy'] is not None:
            actions[index] = int(np.argmax(tactical['policy']))
        else:
            mcts.prepareSearchRoot(board, simulations, rng=np.random.RandomState(0))
            active.append(index)

    for _ in range(simulations):
        pending = []
        for index in active:
            leaf = searches[index].select_leaf(boards[index])
            if leaf['needs_eval']:
                pending.append((index, leaf))
            else:
                searches[index].complete_search(leaf)
        if pending:
            policies, values = nnet.predict_batch([leaf['board'] for _, leaf in pending])
            for (index, leaf), policy, value in zip(pending, policies, values):
                searches[index].complete_search(leaf, policy, float(value))

    for index in active:
        probabilities = searches[index].getActionProbFromTree(boards[index], temp=0)
        actions[index] = int(np.argmax(probabilities))
    return actions


def search_position_metrics(game, nnet, board, simulations):
    variants = transformations(board)
    actions = searched_actions(game, nnet, [variant[2] for variant in variants], simulations)
    return [
        actions[index] == transform_action(game, actions[0], rotations, flip)
        for index, (rotations, flip, _) in enumerate(variants[1:], start=1)
    ]


def aggregate_raw(position_results):
    comparisons = [item for position in position_results for item in position]
    return {
        'positions': len(position_results),
        'transform_comparisons': len(comparisons),
        'mean_policy_total_variation': float(np.mean([
            item['total_variation'] for item in comparisons
        ])),
        'max_policy_total_variation': float(np.max([
            item['total_variation'] for item in comparisons
        ])),
        'raw_top_action_consistency_rate': float(np.mean([
            item['top_action_consistent'] for item in comparisons
        ])),
        'positions_with_any_raw_top_action_change': int(sum(
            not all(item['top_action_consistent'] for item in position)
            for position in position_results
        )),
        'mean_value_absolute_error': float(np.mean([
            item['value_absolute_error'] for item in comparisons
        ])),
        'max_value_absolute_error': float(np.max([
            item['value_absolute_error'] for item in comparisons
        ])),
    }


def diagnose_model(game, model_spec, sampled, raw_count, search_count, simulations):
    name, checkpoint = model_spec
    folder, filename = os.path.split(checkpoint)
    nnet = build_nnet(game, 'v3')
    nnet.load_checkpoint(folder or '.', filename)
    stages = {}
    for stage, boards in sampled.items():
        raw_results = [raw_position_metrics(game, nnet, board) for board in boards[:raw_count]]
        search_results = [
            search_position_metrics(game, nnet, board, simulations)
            for board in boards[:search_count]
        ]
        metrics = aggregate_raw(raw_results)
        search_comparisons = [value for position in search_results for value in position]
        metrics.update({
            'search_positions': len(search_results),
            'search_transform_comparisons': len(search_comparisons),
            'search_top_action_consistency_rate': float(np.mean(search_comparisons)),
            'positions_with_any_search_top_action_change': int(sum(
                not all(position) for position in search_results
            )),
        })
        stages[stage] = metrics
        print(
            '{} {}: raw top {:.1f}% consistent (changed on {}/{} positions), '
            'policy TV mean/max {:.3f}/{:.3f}, value error mean/max {:.3f}/{:.3f}; '
            'search top {:.1f}% consistent (changed on {}/{})'.format(
                name,
                stage,
                100.0 * metrics['raw_top_action_consistency_rate'],
                metrics['positions_with_any_raw_top_action_change'],
                metrics['positions'],
                metrics['mean_policy_total_variation'],
                metrics['max_policy_total_variation'],
                metrics['mean_value_absolute_error'],
                metrics['max_value_absolute_error'],
                100.0 * metrics['search_top_action_consistency_rate'],
                metrics['positions_with_any_search_top_action_change'],
                metrics['search_positions'],
            )
        )
    return {'name': name, 'checkpoint': checkpoint, 'stages': stages}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', action='append', type=parse_model, required=True)
    parser.add_argument('--replay', required=True)
    parser.add_argument('--raw-positions-per-stage', type=int, default=32)
    parser.add_argument('--search-positions-per-stage', type=int, default=4)
    parser.add_argument('--search-sims', type=int, default=128)
    parser.add_argument('--seed', type=int, default=20260718)
    parser.add_argument('--json-out')
    args = parser.parse_args()
    sample_count = max(args.raw_positions_per_stage, args.search_positions_per_stage)
    game = SantoriniGame(5, sequential_placement=True)
    sampled = sample_stage_boards(args.replay, sample_count, args.seed)
    payload = {
        'replay': args.replay,
        'seed': args.seed,
        'raw_positions_per_stage': args.raw_positions_per_stage,
        'search_positions_per_stage': args.search_positions_per_stage,
        'search_sims': args.search_sims,
        'models': [
            diagnose_model(
                game,
                model,
                sampled,
                args.raw_positions_per_stage,
                args.search_positions_per_stage,
                args.search_sims,
            )
            for model in args.model
        ],
    }
    if args.json_out:
        output_dir = os.path.dirname(args.json_out)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.json_out, 'w') as output_file:
            json.dump(payload, output_file, indent=2, sort_keys=True)
        print('Wrote standard-play symmetry diagnostics: {}'.format(args.json_out))


if __name__ == '__main__':
    main()
