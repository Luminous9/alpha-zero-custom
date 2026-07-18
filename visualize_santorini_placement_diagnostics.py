#!/usr/bin/env python3
"""Render learned-placement diagnostics from pit_santorini JSON as an HTML report."""

import argparse
import html
import json
import os


BOARD_SIZE = 5


def parse_opening(signature):
    sides = {}
    for part in signature.split('|'):
        name, encoded = part.split('=', 1)
        locations = []
        if encoded:
            for location in encoded.split(','):
                row, col = location.split(':')
                locations.append((int(row), int(col)))
        sides[name] = tuple(sorted(locations))
    return sides['p1'], sides['p2']


def transform_location(location, rotations, flip):
    row, col = location
    for _ in range(rotations):
        row, col = BOARD_SIZE - 1 - col, row
    if flip:
        col = BOARD_SIZE - 1 - col
    return row, col


def symmetry_key(signature):
    p1, p2 = parse_opening(signature)
    variants = []
    for rotations in range(4):
        for flip in (False, True):
            variants.append((
                tuple(sorted(transform_location(location, rotations, flip) for location in p1)),
                tuple(sorted(transform_location(location, rotations, flip) for location in p2)),
            ))
    return min(variants)


def board_svg(signature):
    p1, p2 = parse_opening(signature)
    size = 210
    margin = 10
    cell = (size - 2 * margin) / BOARD_SIZE
    elements = [
        '<svg class="board" viewBox="0 0 {0} {0}" role="img" '
        'aria-label="Santorini placement {1}">'.format(size, html.escape(signature)),
        '<rect x="0" y="0" width="210" height="210" rx="16" fill="#f7f1e3"/>',
    ]
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            x = margin + col * cell
            y = margin + row * cell
            fill = '#e9dfca' if (row + col) % 2 else '#f4ead7'
            elements.append(
                '<rect x="{:.1f}" y="{:.1f}" width="{:.1f}" height="{:.1f}" '
                'fill="{}" stroke="#c7b99d" stroke-width="1"/>'.format(x, y, cell, cell, fill)
            )
    for player, locations, color, label in (
        ('p1', p1, '#2563eb', 'P1'),
        ('p2', p2, '#e34b4b', 'P2'),
    ):
        for row, col in locations:
            cx = margin + (col + 0.5) * cell
            cy = margin + (row + 0.5) * cell
            elements.extend([
                '<circle cx="{:.1f}" cy="{:.1f}" r="13.5" fill="{}" '
                'stroke="#ffffff" stroke-width="3"/>'.format(cx, cy, color),
                '<text x="{:.1f}" y="{:.1f}" text-anchor="middle" dominant-baseline="central" '
                'fill="white" font-size="9" font-weight="800">{}</text>'.format(
                    cx, cy + 0.5, label
                ),
            ])
    elements.append('</svg>')
    return ''.join(elements)


def metric(label, value, detail=''):
    return (
        '<div class="metric"><div class="metric-value">{}</div>'
        '<div class="metric-label">{}</div><div class="metric-detail">{}</div></div>'
    ).format(html.escape(str(value)), html.escape(label), html.escape(detail))


def render_report(payload, source_path):
    diagnostics = payload.get('learned_placement_diagnostics')
    if not diagnostics:
        raise ValueError('Input JSON does not contain learned_placement_diagnostics.')

    rows = sorted(
        diagnostics['opening_results'],
        key=lambda row: (-int(row['games']), row['opening']),
    )
    symmetry_keys = sorted({symmetry_key(row['opening']) for row in rows})
    symmetry_ids = {key: index + 1 for index, key in enumerate(symmetry_keys)}
    max_games = max(int(row['games']) for row in rows) if rows else 1

    cards = []
    for rank, row in enumerate(rows, 1):
        games = int(row['games'])
        run10_wins = int(row['contestant1_wins'])
        run9_wins = int(row['contestant2_wins'])
        win_rate = 100.0 * run10_wins / games if games else 0.0
        group = symmetry_ids[symmetry_key(row['opening'])]
        bar_width = 100.0 * games / max_games
        physical_available = 'player1_wins' in row and 'player2_wins' in row
        if physical_available:
            physical_record = '{}–{}'.format(row['player1_wins'], row['player2_wins'])
            seat_detail = (
                'Run 10 seats: P1 {} / P2 {} · Run 9 seats: P1 {} / P2 {}'.format(
                    row['contestant1_as_player1_games'],
                    row['contestant1_as_player2_games'],
                    row['contestant2_as_player1_games'],
                    row['contestant2_as_player2_games'],
                )
            )
        else:
            physical_record = 'Not recorded'
            seat_detail = 'Physical winner and contestant seat counts were unavailable in this prior run.'
        cards.append('''
        <article class="opening-card">
          <div class="rank">#{rank}</div>
          {board}
          <div class="opening-content">
            <div class="opening-head">
              <div><strong>{games} game{plural}</strong><span>Symmetry group {group}</span></div>
              <div class="records">
                <div class="record"><strong>{run10}–{run9}</strong><span>Contestant attribution<br>Run 10 – Run 9</span></div>
                <div class="record physical"><strong>{physical_record}</strong><span>Physical outcome<br>P1 – P2</span></div>
              </div>
            </div>
            <div class="frequency"><span style="width:{bar:.2f}%"></span></div>
            <div class="stats">
              <span>Run 10 win rate <strong>{win_rate:.1f}%</strong></span>
              <span>Trajectories <strong>{trajectories}</strong></span>
              <span>Labeled variants <strong>{variants}</strong></span>
            </div>
            <div class="seat-detail">{seat_detail}</div>
            <code>{signature}</code>
          </div>
        </article>'''.format(
            rank=rank,
            board=board_svg(row['opening']),
            games=games,
            plural='' if games == 1 else 's',
            group=group,
            run10=run10_wins,
            run9=run9_wins,
            physical_record=html.escape(physical_record),
            seat_detail=html.escape(seat_detail),
            bar=bar_width,
            win_rate=win_rate,
            trajectories=int(row['distinct_standard_trajectories']),
            variants=int(row['labeled_variants']),
            signature=html.escape(row['opening']),
        ))

    title = '{} vs {} — learned placements'.format(
        payload.get('contestant1_name', 'Contestant 1'),
        payload.get('contestant2_name', 'Contestant 2'),
    )
    duplicate_rate = (
        100.0 * diagnostics['duplicate_game_count'] / diagnostics['games_recorded']
        if diagnostics['games_recorded'] else 0.0
    )
    return '''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<style>
:root {{ color-scheme: light; --ink:#172033; --muted:#68738a; --paper:#f4f6fb;
  --card:#fff; --line:#dfe4ee; --blue:#2563eb; --red:#e34b4b; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--paper); color:var(--ink); font:15px/1.45 Inter,ui-sans-serif,system-ui,sans-serif; }}
main {{ width:min(1180px,calc(100% - 32px)); margin:42px auto 80px; }}
h1 {{ margin:0 0 7px; font-size:clamp(27px,4vw,43px); letter-spacing:-.04em; }}
.subtitle {{ color:var(--muted); margin:0 0 28px; }}
.summary {{ display:grid; grid-template-columns:repeat(5,1fr); gap:12px; margin-bottom:28px; }}
.metric {{ padding:18px; background:var(--card); border:1px solid var(--line); border-radius:14px; }}
.metric-value {{ font-size:28px; font-weight:800; letter-spacing:-.03em; }}
.metric-label {{ font-weight:700; }} .metric-detail {{ color:var(--muted); font-size:12px; margin-top:3px; }}
.legend {{ display:flex; gap:20px; align-items:center; padding:14px 18px; background:#e9eef9; border-radius:12px; margin-bottom:18px; }}
.dot {{ display:inline-block; width:11px; height:11px; border-radius:50%; margin-right:6px; }}
.opening-list {{ display:grid; gap:13px; }}
.opening-card {{ position:relative; display:grid; grid-template-columns:150px 1fr; gap:18px; align-items:center;
  padding:15px 18px 15px 15px; background:var(--card); border:1px solid var(--line); border-radius:16px; }}
.board {{ width:150px; height:150px; display:block; }}
.rank {{ position:absolute; top:10px; left:10px; z-index:2; padding:3px 8px; border-radius:20px;
  color:#fff; background:rgba(23,32,51,.82); font-size:12px; font-weight:800; }}
.opening-head {{ display:flex; justify-content:space-between; gap:20px; font-size:20px; }}
.opening-head span {{ display:block; color:var(--muted); font-size:12px; font-weight:500; }}
.records {{ display:flex; gap:22px; justify-content:flex-end; }}
.record {{ min-width:105px; text-align:right; }} .record strong {{ color:var(--blue); }}
.record.physical strong {{ color:var(--ink); }}
.frequency {{ height:8px; margin:16px 0 12px; overflow:hidden; background:#edf0f5; border-radius:10px; }}
.frequency span {{ display:block; height:100%; background:linear-gradient(90deg,var(--blue),#7aa7ff); border-radius:10px; }}
.stats {{ display:flex; flex-wrap:wrap; gap:8px 22px; color:var(--muted); font-size:13px; }}
.stats strong {{ color:var(--ink); }} code {{ display:block; color:#7a8498; font-size:11px; margin-top:11px; }}
.seat-detail {{ color:#7a8498; font-size:11px; margin-top:8px; }}
footer {{ margin-top:24px; color:var(--muted); font-size:12px; }}
@media (max-width:850px) {{ .summary {{ grid-template-columns:repeat(2,1fr); }} }}
@media (max-width:620px) {{ main {{ width:min(100% - 20px,1180px); margin-top:22px; }}
  .opening-card {{ grid-template-columns:1fr; }} .board {{ width:100%; height:auto; max-width:240px; margin:auto; }}
  .opening-head {{ font-size:17px; flex-direction:column; }} .records {{ justify-content:flex-start; }}
  .record {{ text-align:left; }} .summary {{ grid-template-columns:1fr 1fr; }} }}
</style>
</head>
<body><main>
  <h1>{title}</h1>
  <p class="subtitle">Placement-only matchup · {games} games · sorted by exact-opening frequency</p>
  <section class="summary">
    {games_metric}
    {exact_metric}
    {symmetry_metric}
    {duplicate_metric}
    {frequent_metric}
  </section>
  <div class="legend"><span><i class="dot" style="background:var(--blue)"></i>Player 1</span>
    <span><i class="dot" style="background:var(--red)"></i>Player 2</span>
    <span>Contestant attribution changes when networks swap seats; physical outcome shows which board side won.</span></div>
  <section class="opening-list">{cards}</section>
  <footer>Generated from {source}. Exact openings ignore interchangeable worker labels; symmetry groups merge rotations and reflections.</footer>
</main></body></html>'''.format(
        title=html.escape(title),
        games=int(diagnostics['games_recorded']),
        games_metric=metric('Games recorded', diagnostics['games_recorded'], 'paired placement trials'),
        exact_metric=metric('Exact openings', diagnostics['distinct_exact_openings'], 'worker labels ignored'),
        symmetry_metric=metric('Symmetry groups', diagnostics['distinct_symmetry_unique_openings'], 'rotations/reflections merged'),
        duplicate_metric=metric('Duplicate rate', '{:.1f}%'.format(duplicate_rate), '{} repeated games'.format(diagnostics['duplicate_game_count'])),
        frequent_metric=metric('Top frequency', diagnostics['most_frequent_opening_count'], 'games sharing one opening'),
        cards=''.join(cards),
        source=html.escape(os.path.basename(source_path)),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_json')
    parser.add_argument('output_html')
    args = parser.parse_args()
    with open(args.input_json) as input_file:
        payload = json.load(input_file)
    report = render_report(payload, args.input_json)
    output_dir = os.path.dirname(args.output_html)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output_html, 'w') as output_file:
        output_file.write(report)
    print('Wrote placement visualization: {}'.format(args.output_html))


if __name__ == '__main__':
    main()
