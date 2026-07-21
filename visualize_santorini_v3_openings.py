#!/usr/bin/env python3
"""Rank and visualize symmetry-unique openings from a raw V3 placement policy."""

import argparse
import html
import json
import os
from datetime import datetime, timezone
from itertools import combinations

import numpy as np

from santorini.SantoriniGame import SantoriniGame
from santorini.pytorch.NNet import build_nnet


def transform_location(location, board_size, rotations, flip):
    row, col = location
    for _ in range(rotations):
        row, col = col, board_size - 1 - row
    if flip:
        col = board_size - 1 - col
    return row, col


def canonical_opening_key(p1, p2, board_size):
    variants = []
    for rotations in range(4):
        for flip in (False, True):
            variants.append((
                tuple(sorted(transform_location(x, board_size, rotations, flip) for x in p1)),
                tuple(sorted(transform_location(x, board_size, rotations, flip) for x in p2)),
            ))
    return min(variants)


def normalized_placement_probabilities(game, board, policy):
    valid = game.getValidMoves(board, 1).astype(bool)
    probabilities = np.where(valid, np.asarray(policy, dtype=np.float64), 0.0)
    total = float(probabilities.sum())
    if total <= 0.0:
        raise ValueError('Network assigned no probability mass to legal placement actions.')
    return probabilities / total


def predict_placement_batches(game, nnet, boards, batch_size):
    results = []
    for start in range(0, len(boards), batch_size):
        batch = boards[start:start + batch_size]
        policies, _ = nnet.predict_batch(batch)
        results.extend(
            normalized_placement_probabilities(game, board, policy)
            for board, policy in zip(batch, policies)
        )
    return results


def location_probability(game, policy, location):
    return float(policy[game.getPlacementAction(location)])


def rank_symmetry_unique_openings(nnet, board_size=5, batch_size=1024):
    """Return completed-opening probability mass, summing orders and D4 variants."""
    game = SantoriniGame(board_size, sequential_placement=True)
    squares = [(row, col) for row in range(board_size) for col in range(board_size)]
    initial = game.getInitBoard()
    first_policy = predict_placement_batches(game, nnet, [initial], batch_size)[0]

    first_states = []
    for first in squares:
        board, player = game.getNextState(initial, 1, game.getPlacementAction(first))
        first_states.append((first, board, player))
    second_policies = predict_placement_batches(
        game, nnet, [game.getCanonicalForm(board, player) for _, board, player in first_states], batch_size
    )

    p1_states = []
    for first_index, second_index in combinations(range(len(squares)), 2):
        first, second = squares[first_index], squares[second_index]
        probability = (
            location_probability(game, first_policy, first)
            * location_probability(game, second_policies[first_index], second)
            + location_probability(game, first_policy, second)
            * location_probability(game, second_policies[second_index], first)
        )
        board, player = game.getNextState(
            first_states[first_index][1], 1, game.getPlacementAction(second)
        )
        p1_states.append((tuple(sorted((first, second))), board, player, probability))

    third_policies = predict_placement_batches(
        game, nnet, [game.getCanonicalForm(board, player) for _, board, player, _ in p1_states], batch_size
    )
    p2_prefixes = []
    for index, (p1, board, player, prefix_probability) in enumerate(p1_states):
        for first_p2 in squares:
            if first_p2 in p1:
                continue
            probability = prefix_probability * location_probability(
                game, third_policies[index], first_p2
            )
            next_board, next_player = game.getNextState(
                board, player, game.getPlacementAction(first_p2)
            )
            p2_prefixes.append((p1, first_p2, next_board, next_player, probability))

    fourth_policies = predict_placement_batches(
        game,
        nnet,
        [game.getCanonicalForm(board, player) for _, _, board, player, _ in p2_prefixes],
        batch_size,
    )
    probability_by_key = {}
    path_count_by_key = {}
    for index, (p1, first_p2, board, _, prefix_probability) in enumerate(p2_prefixes):
        occupied = set(p1 + (first_p2,))
        for second_p2 in squares:
            if second_p2 in occupied:
                continue
            probability = prefix_probability * location_probability(
                game, fourth_policies[index], second_p2
            )
            key = canonical_opening_key(
                p1, tuple(sorted((first_p2, second_p2))), board_size
            )
            probability_by_key[key] = probability_by_key.get(key, 0.0) + probability
            path_count_by_key[key] = path_count_by_key.get(key, 0) + 1

    ranked = sorted(probability_by_key.items(), key=lambda item: (-item[1], item[0]))
    return [
        {
            'rank': rank,
            'p1': [list(location) for location in key[0]],
            'p2': [list(location) for location in key[1]],
            'probability': float(probability),
            'path_count': path_count_by_key[key],
        }
        for rank, (key, probability) in enumerate(ranked, 1)
    ]


def board_svg(record, board_size):
    size, margin = 210, 10
    cell = (size - 2 * margin) / board_size
    elements = ['<svg class="board" viewBox="0 0 210 210" role="img">',
                '<rect width="210" height="210" rx="16" fill="#f7f1e3"/>']
    for row in range(board_size):
        for col in range(board_size):
            fill = '#f4ead7' if (row + col) % 2 == 0 else '#e9dfca'
            elements.append('<rect x="{:.1f}" y="{:.1f}" width="{:.1f}" height="{:.1f}" fill="{}" stroke="#c7b99d"/>'.format(
                margin + col * cell, margin + row * cell, cell, cell, fill))
    for locations, color, label in ((record['p1'], '#2563eb', 'P1'), (record['p2'], '#e34b4b', 'P2')):
        for row, col in locations:
            cx, cy = margin + (col + .5) * cell, margin + (row + .5) * cell
            elements.append('<circle cx="{:.1f}" cy="{:.1f}" r="14" fill="{}" stroke="white" stroke-width="3"/>'.format(cx, cy, color))
            elements.append('<text x="{:.1f}" y="{:.1f}" text-anchor="middle" dominant-baseline="central" fill="white" font-size="9" font-weight="800">{}</text>'.format(cx, cy, label))
    elements.append('</svg>')
    return ''.join(elements)


def render_html(payload, top_n):
    openings = payload['openings'][:top_n]
    maximum = openings[0]['probability'] if openings else 1.0
    cards = []
    for record in openings:
        probability = record['probability']
        cards.append('''<article class="card"><div class="rank">#{rank}</div>{board}
<div class="details"><div class="prob">{percent:.5f}%</div>
<div class="relative">{relative:.1f}% of the top opening</div>
<div class="bar"><span style="width:{relative:.2f}%"></span></div>
<code>P1 {p1} · P2 {p2}</code></div></article>'''.format(
            rank=record['rank'], board=board_svg(record, payload['board_size']),
            percent=100.0 * probability, relative=100.0 * probability / maximum,
            p1=html.escape(str(record['p1'])), p2=html.escape(str(record['p2']))))
    top_mass = sum(item['probability'] for item in openings)
    return '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>V3 opening policy — {model}</title><style>
:root{{--ink:#172033;--muted:#68738a;--paper:#f4f6fb;--card:#fff;--line:#dfe4ee;--blue:#2563eb}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:15px/1.45 Inter,system-ui,sans-serif}}
main{{width:min(1180px,calc(100% - 32px));margin:42px auto 80px}}h1{{margin:0;font-size:40px;letter-spacing:-.04em}}.subtitle{{color:var(--muted);margin:7px 0 24px}}
.summary{{display:flex;gap:12px;margin-bottom:24px}}.metric{{background:white;border:1px solid var(--line);padding:15px 20px;border-radius:14px}}.metric strong{{display:block;font-size:25px}}.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}}
.card{{position:relative;display:grid;grid-template-columns:150px 1fr;align-items:center;gap:15px;background:white;border:1px solid var(--line);border-radius:16px;padding:14px}}.board{{width:150px}}.rank{{position:absolute;top:9px;left:9px;background:#172033dd;color:white;border-radius:20px;padding:3px 8px;font-weight:800;font-size:12px}}.prob{{font-size:22px;font-weight:800}}.relative,code{{color:var(--muted);font-size:11px}}.bar{{height:7px;background:#edf0f5;border-radius:8px;margin:12px 0}}.bar span{{display:block;height:100%;background:var(--blue);border-radius:8px}}code{{display:block;word-break:break-word}}
@media(max-width:950px){{.grid{{grid-template-columns:repeat(2,1fr)}}}}@media(max-width:650px){{.grid{{grid-template-columns:1fr}}h1{{font-size:30px}}}}
</style></head><body><main><h1>Top {count} V3 openings</h1><p class="subtitle">Raw sequential placement policy · worker orders and D4 symmetries combined · {model}</p>
<section class="summary"><div class="metric"><strong>{unique:,}</strong>symmetry-unique openings</div><div class="metric"><strong>{mass:.3f}%</strong>top-{count} probability mass</div><div class="metric"><strong>{total:.9f}</strong>total probability check</div></section>
<section class="grid">{cards}</section></main></body></html>'''.format(
        model=html.escape(payload['checkpoint']), count=len(openings), unique=len(payload['openings']),
        mass=100.0 * top_mass, total=payload['total_probability'], cards=''.join(cards))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-folder', required=True)
    parser.add_argument('--checkpoint-file', default='latest.pth.tar')
    parser.add_argument('--output-html', required=True)
    parser.add_argument('--output-json')
    parser.add_argument('--top-n', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=1024)
    args = parser.parse_args()
    checkpoint = os.path.join(args.checkpoint_folder, args.checkpoint_file)
    if not os.path.isfile(checkpoint):
        raise SystemExit('error: no model in path {}'.format(checkpoint))
    game = SantoriniGame(5, sequential_placement=True)
    nnet = build_nnet(game, 'v3')
    nnet.load_checkpoint(args.checkpoint_folder, args.checkpoint_file)
    openings = rank_symmetry_unique_openings(nnet, board_size=5, batch_size=args.batch_size)
    payload = {
        'checkpoint': checkpoint, 'generated_at': datetime.now(timezone.utc).isoformat(),
        'board_size': 5, 'method': 'raw_policy_probability; placement orders and D4 variants summed',
        'total_probability': float(sum(item['probability'] for item in openings)),
        'openings': openings,
    }
    output_json = args.output_json or os.path.splitext(args.output_html)[0] + '.json'
    for path in (args.output_html, output_json):
        folder = os.path.dirname(path)
        if folder:
            os.makedirs(folder, exist_ok=True)
    with open(output_json, 'w') as output_file:
        json.dump(payload, output_file, indent=2)
    with open(args.output_html, 'w') as output_file:
        output_file.write(render_html(payload, args.top_n))
    print('Ranked {} symmetry-unique openings; probability sum {:.9f}.'.format(len(openings), payload['total_probability']))
    print('Wrote JSON: {}'.format(output_json))
    print('Wrote HTML: {}'.format(args.output_html))


if __name__ == '__main__':
    main()
