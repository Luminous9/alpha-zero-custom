#!/usr/bin/env python3
"""Audit whether a V3 placement policy is sharply and stably supported by search."""

import argparse
import json
import math
import os
from collections import Counter
from datetime import datetime, timezone
from itertools import permutations

import numpy as np

from MCTS import MCTS
from pit_santorini import search_args
from santorini.SantoriniGame import SantoriniGame
from santorini.SantoriniInference import predict_batch_deduplicated
from santorini.pytorch.NNet import build_nnet


EPS = 1e-12


def parse_int_list(value):
    try:
        values = tuple(int(item.strip()) for item in value.split(',') if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError('Expected a comma-separated list of integers.') from error
    if not values or any(item < 1 for item in values):
        raise argparse.ArgumentTypeError('Values must be positive integers.')
    return values


def parse_opening(signature):
    sides = {}
    try:
        for part in signature.split('|'):
            name, encoded = part.split('=', 1)
            sides[name] = tuple(
                tuple(map(int, location.split(':')))
                for location in encoded.split(',')
            )
        p1, p2 = sides['p1'], sides['p2']
    except (KeyError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            'Opening must look like p1=1:2,2:2|p2=2:1,2:3.'
        ) from error
    if len(p1) != 2 or len(p2) != 2 or len(set(p1 + p2)) != 4:
        raise argparse.ArgumentTypeError('Opening must contain four distinct worker locations.')
    return p1, p2


def transform_board(board, rotations, flip):
    transformed = np.rot90(np.asarray(board), rotations, axes=(-2, -1))
    if flip:
        transformed = np.flip(transformed, axis=-1)
    return np.ascontiguousarray(transformed)


def distinct_transformations(board):
    results = []
    seen = set()
    for rotations in range(4):
        for flip in (False, True):
            transformed = transform_board(board, rotations, flip)
            key = transformed.tobytes()
            if key in seen:
                continue
            seen.add(key)
            results.append((rotations, flip, transformed))
    return results


def transform_dense_vector(game, vector, rotations, flip):
    old_indices, new_indices = game.getPolicySymmetryPermutation(rotations, flip)
    transformed = np.zeros_like(vector)
    transformed[new_indices] = np.asarray(vector)[old_indices]
    return transformed


def restore_dense_vector(game, vector, rotations, flip):
    old_indices, new_indices = game.getPolicySymmetryPermutation(rotations, flip)
    restored = np.zeros_like(vector)
    restored[old_indices] = np.asarray(vector)[new_indices]
    return restored


def normalized_legal_policy(game, board, policy):
    valids = game.getValidMoves(board, 1).astype(bool)
    result = np.where(valids, np.asarray(policy, dtype=np.float64), 0.0)
    total = float(result.sum())
    if total <= 0:
        raise ValueError('Policy assigned no probability to legal actions.')
    return result / total


def policy_entropy(policy):
    probabilities = np.asarray(policy, dtype=np.float64)
    probabilities = probabilities[probabilities > 0]
    return float(-np.sum(probabilities * np.log(probabilities)))


def effective_action_count(policy):
    return float(math.exp(policy_entropy(policy)))


def policy_kl(target, reference):
    target = np.asarray(target, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    mask = target > 0
    return float(np.sum(target[mask] * (
        np.log(np.maximum(target[mask], EPS))
        - np.log(np.maximum(reference[mask], EPS))
    )))


def total_variation(first, second):
    return float(0.5 * np.abs(np.asarray(first) - np.asarray(second)).sum())


def placement_location(game, action):
    action = int(action)
    if not game.isPlacementAction(action):
        return None
    origin = action // game.local_action_size
    return [int(origin // game.n), int(origin % game.n)]


def action_label(game, action):
    location = placement_location(game, action)
    return '{}:{}'.format(*location) if location is not None else str(int(action))


def state_stabilizer_action_classes(game, board):
    """Map legal actions to D4 classes that preserve the current labeled board."""
    valids = np.flatnonzero(game.getValidMoves(board, 1))
    preserving = []
    for rotations in range(4):
        for flip in (False, True):
            if np.array_equal(transform_board(board, rotations, flip), board):
                _, new_indices = game.getPolicySymmetryPermutation(rotations, flip)
                preserving.append(new_indices)
    classes = {}
    for action in valids:
        classes[int(action)] = min(
            int(permutation[int(action)]) for permutation in preserving
        )
    return classes


def collapse_policy_to_action_classes(policy, action_classes):
    collapsed = Counter()
    for action, representative in action_classes.items():
        collapsed[representative] += float(policy[action])
    return dict(collapsed)


def ensemble_prediction(game, nnet, board):
    variants = distinct_transformations(board)
    policies, values = nnet.predict_batch([item[2] for item in variants])
    restored = [
        normalized_legal_policy(
            game,
            board,
            restore_dense_vector(game, policy, rotations, flip),
        )
        for policy, (rotations, flip, _) in zip(policies, variants)
    ]
    mean_policy = np.mean(restored, axis=0)
    mean_policy /= mean_policy.sum()
    return {
        'policy': mean_policy,
        'values': np.asarray(values, dtype=np.float64),
        'orientation_policies': restored,
        'orbit_size': len(variants),
    }


def most_likely_opening_sequence(game, nnet, opening):
    p1, p2 = opening
    candidates = []
    for p1_order in permutations(p1):
        for p2_order in permutations(p2):
            sequence = p1_order + p2_order
            board = game.getInitBoard()
            player = 1
            probability = 1.0
            for location in sequence:
                canonical = game.getCanonicalForm(board, player)
                prediction = ensemble_prediction(game, nnet, canonical)
                probability *= float(
                    prediction['policy'][game.getPlacementAction(location)]
                )
                board, player = game.getNextState(
                    board,
                    player,
                    game.getPlacementAction(location),
                )
            candidates.append((probability, sequence))
    return max(candidates, key=lambda item: (item[0], item[1]))


def build_prefix_states(game, sequence):
    board = game.getInitBoard()
    player = 1
    states = []
    for index, location in enumerate(sequence):
        states.append({
            'name': 'before_placement_{}'.format(index + 1),
            'placed_workers': index,
            'player_to_place': int(player),
            'next_location_on_selected_path': list(location),
            'board': game.getCanonicalForm(board, player),
        })
        board, player = game.getNextState(
            board,
            player,
            game.getPlacementAction(location),
        )
    return states


def raw_state_metrics(game, nnet, state):
    prediction = ensemble_prediction(game, nnet, state['board'])
    policy = prediction['policy']
    orientation_policies = prediction['orientation_policies']
    top_action = int(np.argmax(policy))
    return {
        'orbit_size': prediction['orbit_size'],
        'policy': policy,
        'policy_entropy': policy_entropy(policy),
        'effective_actions': effective_action_count(policy),
        'top_action': top_action,
        'top_location': placement_location(game, top_action),
        'top_probability': float(policy[top_action]),
        'mean_orbit_total_variation': float(np.mean([
            total_variation(item, policy) for item in orientation_policies
        ])),
        'top_action_consistency_rate': float(np.mean([
            int(np.argmax(item)) == top_action for item in orientation_policies
        ])),
        'value_mean': float(np.mean(prediction['values'])),
        'value_std': float(np.std(prediction['values'])),
        'value_range': float(np.ptp(prediction['values'])),
    }


def predict_leaf_batches(nnet, boards, cache, batch_size, cache_size):
    policies = []
    values = []
    stats = {'requested': 0, 'executed': 0, 'reused': 0}
    for start in range(0, len(boards), batch_size):
        batch_policies, batch_values, batch_stats = predict_batch_deduplicated(
            nnet,
            boards[start:start + batch_size],
            cache=cache,
            max_cache_entries=cache_size,
        )
        policies.extend(batch_policies)
        values.extend(batch_values)
        for key in stats:
            stats[key] += int(batch_stats[key])
    return np.asarray(policies), np.asarray(values), stats


def run_batched_searches(
    game,
    nnet,
    specs,
    simulations,
    gumbel_scale,
    batch_size,
    cache_size,
    progress_interval,
):
    controller_args = search_args(
        simulations,
        search_mode='gumbel',
        gumbel_max_considered_actions=16,
        gumbel_scale=0.0,
        gumbel_placement_scale=gumbel_scale,
        search_symmetry_evaluation=True,
        root_symmetry_samples=8,
        placement_root_symmetry_samples=8,
        inference_deduplication=True,
        inference_cache_size=cache_size,
    )
    searches = []
    for spec in specs:
        mcts = MCTS(game, nnet, controller_args)
        mcts.prepareSearchRoot(
            spec['board'],
            simulations,
            rng=np.random.RandomState(spec['seed']),
        )
        searches.append(mcts)

    cache = {}
    inference = {'requested': 0, 'executed': 0, 'reused': 0}
    for simulation_index in range(simulations):
        pending = []
        boards = []
        ranges = []
        for index, (spec, mcts) in enumerate(zip(specs, searches)):
            leaf = mcts.select_leaf(spec['board'])
            if not leaf['needs_eval']:
                mcts.complete_search(leaf)
                continue
            leaf_boards = mcts.getLeafEvaluationBoards(leaf)
            start = len(boards)
            boards.extend(leaf_boards)
            ranges.append((start, len(boards)))
            pending.append((index, leaf))
        if boards:
            policies, values, batch_stats = predict_leaf_batches(
                nnet, boards, cache, batch_size, cache_size
            )
            for key in inference:
                inference[key] += int(batch_stats[key])
            for (index, leaf), (start, end) in zip(pending, ranges):
                searches[index].complete_search(
                    leaf,
                    policies[start:end],
                    values[start:end],
                )
        if progress_interval and (
            simulation_index + 1 == simulations
            or (simulation_index + 1) % progress_interval == 0
        ):
            print(
                '  budget {}: completed {}/{} simulations'.format(
                    simulations, simulation_index + 1, simulations
                ),
                flush=True,
            )

    results = []
    for spec, mcts in zip(specs, searches):
        state_key = game.stringRepresentation(spec['board'])
        prior = np.zeros(game.getActionSize(), dtype=np.float64)
        prior[mcts.As[state_key]] = mcts.Ps[state_key]
        improved = np.asarray(
            mcts.getTrainingPolicyFromTree(spec['board']), dtype=np.float64
        )
        selected_policy = np.asarray(
            mcts.getActionProbFromTree(spec['board'], temp=1), dtype=np.float64
        )
        counts = mcts.getDenseActionCounts(state_key).astype(np.float64)
        qvalues = mcts.getDenseActionValues(state_key).astype(np.float64)
        completed = np.zeros(game.getActionSize(), dtype=np.float64)
        completed[mcts.As[state_key]] = mcts._completed_qvalues(state_key)

        rotations, flip = spec['rotations'], spec['flip']
        restored = {}
        for name, vector in (
            ('prior', prior),
            ('improved', improved),
            ('selected_policy', selected_policy),
            ('counts', counts),
            ('qvalues', qvalues),
            ('completed_qvalues', completed),
        ):
            restored[name] = restore_dense_vector(
                game, vector, rotations, flip
            ).astype(np.float64)
        restored['prior'] = normalized_legal_policy(
            game, spec['base_board'], restored['prior']
        )
        restored['improved'] = normalized_legal_policy(
            game, spec['base_board'], restored['improved']
        )
        restored['selected_action'] = int(np.argmax(restored['selected_policy']))
        total_visits = float(np.sum(restored['counts']))
        restored.update({
            'state_name': spec['state_name'],
            'seed': spec['seed'],
            'rotations': rotations,
            'flip': flip,
            'network_root_value': float(mcts.raw_values.get(state_key, 0.0)),
            'search_root_value': (
                float(np.sum(restored['counts'] * restored['qvalues']) / total_visits)
                if total_visits else float(mcts.raw_values.get(state_key, 0.0))
            ),
        })
        results.append(restored)
    inference['reuse_rate'] = (
        float(inference['reused'] / inference['requested'])
        if inference['requested'] else 0.0
    )
    return results, inference


def summarize_search_runs(game, board, raw_policy, runs, top_n):
    mean_policy = np.mean([run['improved'] for run in runs], axis=0)
    mean_policy /= mean_policy.sum()
    selections = Counter(run['selected_action'] for run in runs)
    dominant_action, dominant_count = selections.most_common(1)[0]
    action_classes = state_stabilizer_action_classes(game, board)
    class_selections = Counter(
        action_classes[action] for action in selections.elements()
    )
    dominant_class, dominant_class_count = class_selections.most_common(1)[0]
    raw_classes = collapse_policy_to_action_classes(raw_policy, action_classes)
    mean_classes = collapse_policy_to_action_classes(mean_policy, action_classes)
    entropies = [policy_entropy(run['improved']) for run in runs]
    class_entropies = [
        policy_entropy(list(collapse_policy_to_action_classes(
            run['improved'], action_classes
        ).values()))
        for run in runs
    ]
    raw_kls = [policy_kl(run['improved'], raw_policy) for run in runs]
    tvs = [total_variation(run['improved'], mean_policy) for run in runs]
    visited_q_gaps = []
    for run in runs:
        class_qvalues = {}
        for action in np.flatnonzero(run['counts'] > 0):
            representative = action_classes[int(action)]
            class_qvalues.setdefault(representative, []).append(
                float(run['qvalues'][action])
            )
        if len(class_qvalues) >= 2:
            ordered = np.sort([
                max(values) for values in class_qvalues.values()
            ])[::-1]
            visited_q_gaps.append(float(ordered[0] - ordered[1]))

    class_order = sorted(mean_classes, key=mean_classes.get, reverse=True)
    action_rows = []
    for representative in class_order:
        members = [
            action for action, action_class in action_classes.items()
            if action_class == representative
        ]
        visited_values = [
            max(
                run['qvalues'][action]
                for action in members
                if run['counts'][action] > 0
            )
            for run in runs
            if any(run['counts'][action] > 0 for action in members)
        ]
        action_rows.append({
            'representative_action': int(representative),
            'representative_location': placement_location(game, representative),
            'member_locations': [placement_location(game, action) for action in members],
            'mean_improved_probability': float(mean_classes[representative]),
            'raw_probability': float(raw_classes[representative]),
            'selection_rate': float(class_selections[representative] / len(runs)),
            'mean_visits': float(np.mean([
                sum(run['counts'][action] for action in members) for run in runs
            ])),
            'visited_run_count': len(visited_values),
            'mean_q_when_visited': (
                float(np.mean(visited_values)) if visited_values else None
            ),
            'q_std_when_visited': (
                float(np.std(visited_values)) if visited_values else None
            ),
        })
        if len(action_rows) == top_n:
            break
    return {
        'runs': len(runs),
        'distinct_selected_actions': len(selections),
        'dominant_action': int(dominant_action),
        'dominant_location': placement_location(game, dominant_action),
        'dominant_selection_rate': float(dominant_count / len(runs)),
        'action_class_count': len(set(action_classes.values())),
        'distinct_selected_action_classes': len(class_selections),
        'dominant_action_class': int(dominant_class),
        'dominant_action_class_location': placement_location(game, dominant_class),
        'dominant_action_class_members': [
            placement_location(game, action)
            for action, representative in action_classes.items()
            if representative == dominant_class
        ],
        'dominant_action_class_selection_rate': float(
            dominant_class_count / len(runs)
        ),
        'mean_improved_policy_entropy': float(np.mean(entropies)),
        'mean_improved_effective_actions': float(np.mean(np.exp(entropies))),
        'mean_improved_action_class_entropy': float(np.mean(class_entropies)),
        'mean_improved_effective_action_classes': float(
            np.mean(np.exp(class_entropies))
        ),
        'mean_target_to_raw_policy_kl': float(np.mean(raw_kls)),
        'mean_target_total_variation_from_run_mean': float(np.mean(tvs)),
        'mean_network_root_value': float(np.mean([
            run['network_root_value'] for run in runs
        ])),
        'mean_search_root_value': float(np.mean([
            run['search_root_value'] for run in runs
        ])),
        'search_root_value_std': float(np.std([
            run['search_root_value'] for run in runs
        ])),
        'mean_search_value_shift': float(np.mean([
            run['search_root_value'] - run['network_root_value'] for run in runs
        ])),
        'mean_best_vs_second_visited_q_gap': (
            float(np.mean(visited_q_gaps)) if visited_q_gaps else None
        ),
        'selection_histogram': {
            action_label(game, action): count
            for action, count in selections.most_common()
        },
        'action_class_selection_histogram': {
            action_label(game, action): count
            for action, count in class_selections.most_common()
        },
        'top_actions': action_rows,
        '_mean_policy': mean_policy,
    }


def serializable_summary(summary):
    return {key: value for key, value in summary.items() if not key.startswith('_')}


def print_state_report(game, state_result):
    raw = state_result['raw']
    print(
        '\n{} ({} workers placed, player {}): raw top {} at {:.1f}%, '
        'effective actions {:.2f}, value {:.3f} ± {:.3f}'.format(
            state_result['name'],
            state_result['placed_workers'],
            state_result['player_to_place'],
            action_label(game, raw['top_action']),
            100.0 * raw['top_probability'],
            raw['effective_actions'],
            raw['value_mean'],
            raw['value_std'],
        )
    )
    for budget, result in state_result['budgets'].items():
        print(
            '  {:>4} sims: top {} selected {:5.1f}% across {:2d} runs; '
            '{} selected classes; effective classes {:.2f}; KL(search||raw) {:.3f}; '
            'Q gap {:.3f}; value net/search {:+.3f}/{:+.3f}'.format(
                budget,
                action_label(game, result['dominant_action_class']),
                100.0 * result['dominant_action_class_selection_rate'],
                result['runs'],
                result['distinct_selected_action_classes'],
                result['mean_improved_effective_action_classes'],
                result['mean_target_to_raw_policy_kl'],
                result['mean_best_vs_second_visited_q_gap'] or 0.0,
                result['mean_network_root_value'],
                result['mean_search_root_value'],
            )
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-folder', required=True)
    parser.add_argument('--checkpoint-file', default='latest.pth.tar')
    parser.add_argument(
        '--opening',
        type=parse_opening,
        default=parse_opening('p1=1:2,2:2|p2=2:1,2:3'),
        help='Completed opening whose most likely ordered prefix path is audited.',
    )
    parser.add_argument('--budgets', type=parse_int_list, default=(96, 256, 512))
    parser.add_argument('--runs-per-budget', type=int, default=4)
    parser.add_argument('--seed', type=int, default=20260721)
    parser.add_argument('--gumbel-placement-scale', type=float, default=1.5)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--cache-size', type=int, default=4096)
    parser.add_argument('--top-actions', type=int, default=8)
    parser.add_argument('--progress-interval', type=int, default=64)
    parser.add_argument('--json-out', required=True)
    args = parser.parse_args()
    if args.runs_per_budget < 1:
        parser.error('--runs-per-budget must be positive.')
    if args.gumbel_placement_scale < 0:
        parser.error('--gumbel-placement-scale cannot be negative.')

    checkpoint = os.path.join(args.checkpoint_folder, args.checkpoint_file)
    if not os.path.isfile(checkpoint):
        raise SystemExit('error: checkpoint not found: {}'.format(checkpoint))

    game = SantoriniGame(5, sequential_placement=True)
    nnet = build_nnet(game, 'v3')
    nnet.load_checkpoint(args.checkpoint_folder, args.checkpoint_file)
    path_probability, sequence = most_likely_opening_sequence(game, nnet, args.opening)
    states = build_prefix_states(game, sequence)
    seeds = [args.seed + index * 1009 for index in range(args.runs_per_budget)]
    print('Checkpoint: {}'.format(checkpoint))
    print('Auditing most likely order {} (raw path probability {:.6f}).'.format(
        ['{}:{}'.format(*location) for location in sequence], path_probability
    ))
    print('Budgets: {}; seeds per budget: {}; placement Gumbel scale: {}'.format(
        ', '.join(map(str, args.budgets)), args.runs_per_budget, args.gumbel_placement_scale
    ))

    state_results = []
    raw_by_state = {}
    for state in states:
        raw = raw_state_metrics(game, nnet, state)
        raw_by_state[state['name']] = raw
        state_results.append({
            key: value for key, value in state.items() if key != 'board'
        })
        state_results[-1]['board'] = np.asarray(state['board']).tolist()
        state_results[-1]['raw'] = {
            key: value for key, value in raw.items() if key != 'policy'
        }
        state_results[-1]['budgets'] = {}

    total_inference = {'requested': 0, 'executed': 0, 'reused': 0}
    for budget in args.budgets:
        specs = []
        for state in states:
            for rotations, flip, board in distinct_transformations(state['board']):
                for seed in seeds:
                    specs.append({
                        'state_name': state['name'],
                        'base_board': state['board'],
                        'board': board,
                        'rotations': rotations,
                        'flip': flip,
                        'seed': seed,
                    })
        print('\nRunning {} searches at {} simulations...'.format(len(specs), budget))
        runs, inference = run_batched_searches(
            game,
            nnet,
            specs,
            budget,
            args.gumbel_placement_scale,
            args.batch_size,
            args.cache_size,
            args.progress_interval,
        )
        for key in total_inference:
            total_inference[key] += int(inference[key])
        for state_result in state_results:
            state_runs = [
                run for run in runs if run['state_name'] == state_result['name']
            ]
            summary = summarize_search_runs(
                game,
                next(
                    state['board'] for state in states
                    if state['name'] == state_result['name']
                ),
                raw_by_state[state_result['name']]['policy'],
                state_runs,
                args.top_actions,
            )
            state_result['budgets'][str(budget)] = serializable_summary(summary)

    total_inference['reuse_rate'] = (
        float(total_inference['reused'] / total_inference['requested'])
        if total_inference['requested'] else 0.0
    )
    payload = {
        'checkpoint': checkpoint,
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'opening': {
            'p1': [list(item) for item in args.opening[0]],
            'p2': [list(item) for item in args.opening[1]],
            'selected_order': [list(item) for item in sequence],
            'raw_selected_path_probability': float(path_probability),
        },
        'configuration': {
            'budgets': list(args.budgets),
            'runs_per_budget': args.runs_per_budget,
            'seeds': seeds,
            'gumbel_placement_scale': args.gumbel_placement_scale,
            'root_symmetry_samples': 8,
            'interior_symmetry_samples': 1,
        },
        'inference': total_inference,
        'states': state_results,
    }
    output_dir = os.path.dirname(args.json_out)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.json_out, 'w') as output_file:
        json.dump(payload, output_file, indent=2, sort_keys=True)

    for state_result in state_results:
        print_state_report(game, state_result)
    print(
        '\nInference reuse: {}/{} ({:.1f}%).'.format(
            total_inference['reused'],
            total_inference['requested'],
            100.0 * total_inference['reuse_rate'],
        )
    )
    print('Wrote placement audit: {}'.format(args.json_out))


if __name__ == '__main__':
    main()
