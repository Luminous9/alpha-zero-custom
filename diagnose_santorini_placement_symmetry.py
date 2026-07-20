#!/usr/bin/env python3
"""Measure D4 symmetry error in raw V3 placement policies."""

import argparse
import json
import os

import numpy as np

from santorini.SantoriniGame import SantoriniGame
from santorini.SantoriniSymmetryDiagnostics import (
    position_metrics,
    transformations,
)
from santorini.pytorch.NNet import build_nnet


def parse_model(value):
    if '=' not in value:
        raise argparse.ArgumentTypeError('Model must be NAME=CHECKPOINT_PATH.')
    name, path = value.split('=', 1)
    if not name or not path:
        raise argparse.ArgumentTypeError('Model must be NAME=CHECKPOINT_PATH.')
    return name, path


def parse_opening(signature):
    sides = {}
    for part in signature.split('|'):
        name, encoded = part.split('=', 1)
        sides[name] = [tuple(map(int, location.split(':'))) for location in encoded.split(',')]
    return sides['p1'], sides['p2']


def transform_policy(game, policy, rotations, flip):
    old_indices, new_indices = game.getPolicySymmetryPermutation(rotations, flip)
    transformed = np.zeros_like(policy)
    transformed[new_indices] = policy[old_indices]
    return transformed


def normalized_legal_policy(game, board, policy):
    valids = game.getValidMoves(board, 1).astype(bool)
    legal = np.where(valids, np.asarray(policy, dtype=np.float64), 0.0)
    mass = float(legal.sum())
    if mass:
        legal /= mass
    return legal, mass


def state_symmetry_metrics(game, nnet, board):
    variants = transformations(board)
    policies, values = nnet.predict_batch([variant[2] for variant in variants])
    base_policy = policies[0]
    base_legal, legal_mass = normalized_legal_policy(game, board, base_policy)
    transforms = []
    for index, (rotations, flip, transformed_board) in enumerate(variants[1:], start=1):
        transformed_legal, transformed_mass = normalized_legal_policy(
            game,
            transformed_board,
            policies[index],
        )
        expected = transform_policy(game, base_legal, rotations, flip)
        difference = np.abs(transformed_legal - expected)
        transforms.append({
            'rotations': rotations,
            'flip': flip,
            'total_variation': float(0.5 * difference.sum()),
            'max_action_probability_error': float(difference.max()),
            'legal_policy_mass': transformed_mass,
            'value': float(values[index]),
            'value_absolute_error_from_base': float(abs(values[index] - values[0])),
        })
    result = {
        'legal_policy_mass': legal_mass,
        'mean_total_variation': float(np.mean([item['total_variation'] for item in transforms])),
        'max_total_variation': float(np.max([item['total_variation'] for item in transforms])),
        'max_action_probability_error': float(np.max([
            item['max_action_probability_error'] for item in transforms
        ])),
        'transforms': transforms,
        '_base_legal_policy': base_legal,
    }
    result.update(position_metrics(game, variants, policies, values))
    return result


def placement_grid(game, policy):
    return [
        [float(policy[game.getPlacementAction((row, col))]) for col in range(game.n)]
        for row in range(game.n)
    ]


def build_prefix_states(game, opening_signature):
    p1, p2 = parse_opening(opening_signature)
    sequence = p1 + p2
    board = game.getInitBoard()
    player = 1
    states = []
    for step, location in enumerate(sequence):
        states.append({
            'name': 'before_placement_{}'.format(step + 1),
            'placed_workers': step,
            'player_to_place': int(player),
            'board': game.getCanonicalForm(board, player),
        })
        board, player = game.getNextState(board, player, game.getPlacementAction(location))
    return states


def diagnose_model(game, name, checkpoint_path, states):
    folder, filename = os.path.split(checkpoint_path)
    nnet = build_nnet(game, 'v3')
    nnet.load_checkpoint(folder or '.', filename)
    state_results = []
    for state in states:
        metrics = state_symmetry_metrics(game, nnet, state['board'])
        base_policy = metrics.pop('_base_legal_policy')
        result = {
            key: value for key, value in state.items() if key != 'board'
        }
        result.update(metrics)
        result['placement_probability_grid'] = placement_grid(game, base_policy)
        state_results.append(result)
    return {
        'name': name,
        'checkpoint': checkpoint_path,
        'states': state_results,
    }


def print_report(result):
    print('\n{} ({})'.format(result['name'], result['checkpoint']))
    for state in result['states']:
        print(
            '  {}: mean TV {:.4f}, max TV {:.4f}, max action error {:.4f}, legal mass {:.4f}; '
            'value mean/std/range {:.4f}/{:.4f}/{:.4f}, sign disagreement {}'.format(
                state['name'],
                state['mean_total_variation'],
                state['max_total_variation'],
                state['max_action_probability_error'],
                state['legal_policy_mass'],
                state['value_orbit_mean'],
                state['value_orbit_std'],
                state['value_orbit_range'],
                state['value_sign_disagreement'],
            )
        )
        if state['placed_workers'] == 0:
            print('  Empty-board normalized placement probabilities:')
            for row in state['placement_probability_grid']:
                print('    ' + ' '.join('{:6.3f}'.format(value) for value in row))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', action='append', type=parse_model, required=True)
    parser.add_argument(
        '--opening',
        default='p1=1:2,2:2|p2=2:1,2:3',
        help='Exact opening whose four placement prefixes should be tested.',
    )
    parser.add_argument('--json-out')
    args = parser.parse_args()

    game = SantoriniGame(5, sequential_placement=True)
    states = build_prefix_states(game, args.opening)
    payload = {
        'opening': args.opening,
        'models': [
            diagnose_model(game, name, path, states)
            for name, path in args.model
        ],
    }
    for result in payload['models']:
        print_report(result)

    if args.json_out:
        output_dir = os.path.dirname(args.json_out)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.json_out, 'w') as output_file:
            json.dump(payload, output_file, indent=2, sort_keys=True)
        print('\nWrote symmetry diagnostics: {}'.format(args.json_out))


if __name__ == '__main__':
    main()
