# P2 A2 fresh-suite checkpoint arenas

## Result

A2 is complete locally. On 120 paired-seat standard games per matchup,
iteration 14 scored **43.3% against iteration 11** and **53.3% against iteration
1**. Neither paired interval excludes 50%, but the fresh head-to-head direction
supports retaining iteration 11 and does not support promoting iteration 14.

Combined with the deep-value audit, the most coherent reading is a plateau with
a possible modest family-relative regression after iteration 11, not a broad
value-head collapse. The earlier 22.5-point common-anchor change was too large
to treat as a general strength estimate; A2 does not reproduce a regression of
that magnitude.

## Frozen contract

- 120 games per matchup: 60 openings with both seat orders.
- Iteration 14 versus iteration 11 and iteration 14 versus iteration 1.
- Both matchups use the exact same opening boards and game seeds.
- Fresh opening seed: `20260815`; the previous longitudinal/selection seed was
  `20260715`.
- The 60 new openings have zero D4 overlap with the 20-opening longitudinal
  suite regenerated from `20260715`.
- Deterministic 96-simulation Gumbel evaluation, scale zero.
- Canonical D4 inference, batch size 128, FP32.
- Opening-suite SHA-256:
  `419901810a313d00650724b9d2c55bf4a924e871f2463d41f433ab943967c34b`.
- No final-test positions or reserved final-arena seeds were used.

## Matchups

| Matchup | Iteration-14 score | Games | Pair 2-0 / 1-1 / 0-2 | Paired bootstrap 95% | Paired sign-flip p |
|---|---:|---:|---:|---:|---:|
| Iteration 14 vs iteration 11 | 52-68 (43.3%) | 120 | 9 / 34 / 17 | 35.0%-51.7% | 0.1686 |
| Iteration 14 vs iteration 1 | 64-56 (53.3%) | 120 | 15 / 34 / 11 | 45.0%-61.7% | 0.5572 |

The iteration-14-versus-11 point estimate is a 6.7-point deficit. Its interval
narrowly includes parity, so call this directional evidence of a modest
regression, not a statistically resolved regression. Iteration 14 remains
roughly even with iteration 1.

## Local CPU versus P100

The P100 is faster, but not enough to be required for this arena:

- This local run took 454.8 seconds for iteration 14 versus iteration 11 and
  438.2 seconds versus iteration 1, 893.1 seconds total.
- The prior P100 40-game runs took 111.1 and 110.2 seconds respectively.
- Normalizing for game count gives approximately 151.6/146.1 seconds locally
  versus 111.1/110.2 seconds on P100 per 40 games.
- Inference execution rates tell the same story: the P100 was about 1.3-1.4x
  faster, saving an estimated four minutes over this complete A2 run.

The arena spends substantial time in Python/MCTS coordination and uses many
irregular inference batches, so raw GPU throughput does not translate into a
large wall-clock multiple. Local execution is reasonable for one-off milestone
arenas. P100 becomes worthwhile for repeated arenas, larger search budgets, or
when it is already attached for training.

## Decision

Keep iteration 11 as the tentative production head. Do not restart or change
value-target semantics based on A2: the aggregate deep-value audit was healthy,
and this arena identifies a strength/style issue rather than its mechanism.

A3 is now complete; see `P2_A3_EXTERNAL_ANCHORS.md`. It finds no demonstrated
transferable gain over P1c at the 100k external anchor.

## Artifacts

- `temp/santorini_v4_p2_a2/a2-openings.npz`
- `temp/santorini_v4_p2_a2/a2-openings-manifest.json`
- `temp/santorini_v4_p2_a2/iteration14-vs-iteration11.json`
- `temp/santorini_v4_p2_a2/iteration14-vs-iteration1.json`
- `temp/santorini_v4_p2_a2/a2-summary.json`

The reusable runner is `run_santorini_v4_p2_a2.py`.
