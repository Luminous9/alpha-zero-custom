# P1b Supervised Target and Architecture Screen

Status: the original P1b pilot is complete, but its terminal stop conclusion has
been withdrawn after review. The supervised pipeline, Candidate A-D sizing,
matched target screen, matched ordinary control, and two paired selection-arena
gates are valid. Global score+winner blend wins this pilot target screen, and no
tested equivariant candidate clears the ordinary 8x96 control. The result rejects
this candidate set at this data scale; it does not select the final architecture
or reject the fresh-network/bootstrap program. P1b continues with scaled learning
curves and broader ordinary/equivariant controls. The final test split and final
arena seeds remain untouched.

## Implemented contract

`santorini/V4Supervised.py` and `screen_santorini_v4_bootstrap.py` implement the
P1b pilot contract:

- 10,000 train draws and 3,000 independently generated selection draws from
  the already assigned train/selection splits;
- exact 70/20/10 main-line/randomized-subgame/Run13 source balance and
  20/35/45 early/middle/late stage balance;
- 5% engine-policy smoothing over other legal actions only, retaining all
  illegal logits at zero; Run13 retains its native replay distribution;
- separate winner, mapped-score, global-blend, and stage-aware-blend arrays, so
  checkpoint metadata and the self-play target transition cannot conflate the
  target meanings;
- the provisional declared blend `alpha_boot=0.5`, P0b global temperature
  `T=261.8`, and early/middle/late score reliability multipliers
  `0.25/0.75/1.0`;
- matched AdamW optimization (`lr=3e-4`, weight decay `1e-4`, batch 256), with
  policy loss weighted 0.25 relative to value MSE;
- random executable D4 augmentation for ordinary controls and construction-time
  equivariance for the regular-representation model;
- selection metrics after every epoch and retention of both the best checkpoint
  under `0.25 * policy CE + selected-target MSE` and the final checkpoint.

The train plan contains 5,848 unique corpus positions; the weighted selection
plan contains 1,176 unique corpus positions. Sampling with replacement is
intentional: it realizes the declared source/stage distribution while retaining
the converter's observation-frequency weights. There are no cross-corpus D4
overlaps.

## Matched four-epoch screen

All runs below use seed 20260812 on the local CPU. Timing is useful for checking
gross regressions only; it is not the required P100 FP32/FP16 or end-to-end
self-play measurement.

| Configuration | Parameters | Train sec | Selection policy CE | Top-1 | Winner MSE | Stage-blend MSE | Global-blend MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Ordinary 8x96, 6 planes, stage blend | 1,345,413 | 104.8 | 4.1727 | 6.53% | 0.8351 | 0.5885 | 0.4982 |
| Ordinary 8x96, 13 planes, stage blend | 1,351,461 | 95.1 | 4.1099 | 7.43% | 0.9112 | 0.6605 | 0.5696 |
| Equivariant A 8x96, stage blend | 175,489 | 110.9 | 5.4546 | 2.20% | 0.9783 | 0.7284 | 0.6341 |
| Equivariant A 8x96, global blend | 175,489 | 106.7 | 5.4601 | 1.77% | 0.9413 | 0.6895 | 0.5934 |
| Equivariant A 8x96, winner only | 175,489 | 110.8 | 5.6254 | 1.07% | 1.0155 | 0.7706 | 0.6736 |

The target-specific MSE is not an oracle-truth metric: a model trained on a
target is partly being measured against that target. The cross-target columns,
policy metrics, later arenas, and P0b calibration evidence must therefore be
considered together.

## Interpretation

### 6 versus 13 planes

The 13-plane ordinary control improves policy cross-entropy by 0.0628 and top-1
accuracy by 0.90 percentage point, but worsens every reported value MSE. This is
a mixed result, not evidence to remove the tactical planes. The cheap planes
retain a plausible policy benefit, but this pilot is too small to lock the input
set.

### Ordinary versus equivariant Candidate A

Candidate A is substantially behind both ordinary controls after four epochs.
Because its learning curve was still descending, the stage-blend run was
extended to 12 epochs. Its training policy loss improved from 5.3869 at epoch 4
to 3.4779 at epoch 12 and its training value MSE from 0.4299 to 0.0296, but
selection policy loss was still 4.4664 and selection stage-blend MSE worsened
from 0.7284 to 0.8769. This separates slow convergence from the more important
problem: Candidate A is fitting the repeated 10k plan without matching the
ordinary control's held-out generalization.

Candidate A therefore fails this pilot screen. That does **not** select the
ordinary network for V4: the plan's fresh-start condition still requires an
architecture upgrade.

The 175k-parameter model also was not faster during local training. Its dynamically expanded
block-circulant training path processed roughly 361 examples/sec versus 421 for
the 13-plane ordinary control. The already tested frozen ordinary-Conv2d export
path must be used for target-GPU and end-to-end inference conclusions.

### Candidate A-D sizing

Per-epoch selection was added before testing the larger candidates. The table
uses each run's retained epoch and compares the same 13-plane stage-blend target.
Candidate A's objective is reconstructed from its legacy four-epoch endpoint.

| Model | Shape | Parameters | Epoch | Policy CE | Top-1 | Stage MSE | Selection objective |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Ordinary control | 8x96 | 1,351,461 | 4 | 4.1099 | 7.43% | 0.6605 | 1.6880 |
| Candidate A | 8x96 | 175,489 | 4 | 5.4546 | 2.20% | 0.7284 | 2.0920 |
| Candidate B | 10x128 | 379,241 | 4 | 5.3884 | 2.13% | 0.8527 | 2.1998 |
| Candidate C | 6x192 | 509,689 | 8 | 4.1331 | 11.13% | 0.7335 | 1.7668 |
| Candidate D | 12x160 | 702,865 | 3 | 4.8585 | 4.07% | 0.6500 | 1.8646 |

Candidate B does not improve the combined held-out result. Candidate C is the
first equivariant model to close the ordinary control's policy gap and has the
best equivariant objective. Candidate D is effectively tied with C's four-epoch
objective but is worse on policy and materially slower locally; further D
training also regresses the selection objective. The upper probe has therefore
flattened, and D is rejected.

Candidate C is the equivariant architecture knee, not the overall winner. At its
best epoch it remains 0.0730 worse than the ordinary control on stage-value MSE
and 0.0788 worse on the combined objective, despite the much better policy
top-1. This made it the only equivariant candidate worth carrying into arenas.

### Winner-only versus score/result blend

The target variants were rerun for eight epochs on otherwise identical Candidate
C models. The table uses each variant's retained checkpoint and reports every
value MSE against the same predictions.

| Candidate C target | Epoch | Policy CE | Top-1 | Winner MSE | Stage MSE | Global MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Stage blend | 8 | 4.1331 | 11.13% | 0.9836 | 0.7335 | 0.6426 |
| Global blend | 8 | 4.0926 | 9.60% | 0.9467 | 0.7051 | 0.6057 |
| Winner only | 7 | 4.3421 | 8.23% | 1.1309 | 0.8932 | 0.7881 |

Winner-only does not meet the stated preference condition: it is worse on
policy and even on held-out winner MSE. It is rejected at this screen rather
than carried into an arena merely because its target semantics would make the
self-play handoff simpler.

Global blend has slightly better policy CE and every value MSE than stage blend;
stage blend retains better top-1. This made both blend variants worth a small
selection arena. The supervised result alone did not overturn the calibration evidence:
supervised MSE against these constructed labels cannot establish which label is
truer, and the stage rule was introduced because P0b's early global calibration
was biased.

### Target selection arenas

The Candidate-C variants then played paired 40-game gates at 96 simulations,
using selection seed 20260814. The standard gate uses 20 distinct positions
from the selection split, paired across seats. The full-game gate starts from
the empty board with paired placement-search seeds. Both disable runtime root
symmetry evaluation because the network is equivariant by construction.

| Gate | Global-stage games | Global pair W-L-T | Global score bootstrap 95% | Paired sign p |
| --- | ---: | ---: | ---: | ---: |
| Standard play | 26-14 | 8-2-10 | 50.0%-77.5% | 0.1094 |
| Full game | 27-13 | 9-2-9 | 52.5%-82.5% | 0.0654 |

The gates independently point the same way, 53-27 in aggregate. Neither
20-pair sign test alone clears 0.05, but the full-game cluster-bootstrap interval
excludes 50%, and the arena direction agrees with every cross-target MSE and
policy CE result. Global blend is therefore selected as the P1b bootstrap
target. The selected contract is
`alpha_boot=0.5`, `T=261.8`, with no phase reliability multiplier. This is still
the explicitly declared bootstrap exception and does not change the self-play
value target after handoff.

### Fair architecture control and arenas

After global won, the ordinary 13-plane control was retrained with the same
global target, eight-epoch ceiling, optimizer, sampling plans, and per-epoch
selection. It peaks at epoch 6:

| Model | Policy CE | Top-1 | Winner MSE | Stage MSE | Global MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Candidate C global | 4.0926 | 9.60% | 0.9467 | 0.7051 | 0.6057 |
| Ordinary 13-plane global | 3.7046 | 13.60% | 0.7987 | 0.5601 | 0.4779 |

The control dominates every supervised metric. The same paired 40-game,
96-simulation gates then compare architecture while holding input and target
fixed:

| Gate | C-ordinary games | C pair W-L-T | C score bootstrap 95% | Paired sign p |
| --- | ---: | ---: | ---: | ---: |
| Standard play | 17-23 | 3-6-11 | 27.5%-57.5% | 0.5078 |
| Full game | 10-30 | 1-11-8 | 12.5%-37.5% | 0.0063 |

Candidate C loses 27-53 across the separately reported gates. The standard gate
alone is inconclusive, but the full-game result is decisive at the paired-block
level and the direction agrees with every supervised metric. Per-game Wilson
intervals and exact binomial p-values are retained in JSON only as secondary
summaries. The ordinary control is also faster in local frozen-export smoke
tests. P100 timing can quantify the cost ratio but cannot repair this strength
failure.

## Corrected decision after review

Do not proceed directly to P1c, but continue P1b. The original decision promoted
a deliberately small screening result into a final-strength architecture claim.
Only 5,848 unique training positions and one ordinary shape were tested. The
supported conclusion is therefore limited to the tested checkpoints: Candidate C
loses to the ordinary 8x96 control under this pilot contract.

The next screen adds ordinary 10x128 and 6x192 towers, retains Candidate C as a
continuity control, and adds one roughly learned-parameter-matched equivariant
candidate. It uses stage/source-correct data with learning-curve observations at
approximately 100k, 300k, and 1M training examples. Sampling reports unique
coverage, maximum repetition, and per-stratum supply; a large nominal draw count
is not treated as a large corpus when it repeatedly draws a scarce stratum.

The current equivariant policy readout is not classified as an implementation
bug. It applies a shared local map to every group component, maps all eight
outputs into canonical policy coordinates, and averages them. The invariant
value head likewise performs a valid group projection. The open question is
learned capacity: effective-channel matching gave Candidate C substantially
fewer learned parameters than the ordinary control, so the scaled screen must
include a capacity-aware equivariant comparison rather than an unconstrained
concatenation that would break equivariance.

Global score+winner remains the provisional bootstrap default and winner-only
remains the declared bake-off ablation. Neither target nor the 13-plane input is
permanently locked by this pilot. A small candidate-versus-Run13 arena may be run
as a transfer/pipeline diagnostic during the learning curves, but Gate G1 remains
the 96/128-simulation comparison of the selected, scaled checkpoint. P1c begins
only after that repeated P1b selects an architecture. The P100 inference benchmark
remains part of cost-aware selection.

## Reproduction and validation

Primary outputs:

- `temp/santorini_v4_p1b_screen/results.json`
- `temp/santorini_v4_p1b_screen_equiv12/results.json`
- `temp/santorini_v4_p1b_screen_bc/results.json`
- `temp/santorini_v4_p1b_screen_c8/results.json`
- `temp/santorini_v4_p1b_screen_c_targets/results.json`
- `temp/santorini_v4_p1b_screen_d/results.json`
- `temp/santorini_v4_p1b_screen_ordinary_global8/results.json`
- `temp/santorini_v4_p1b_arena_stage_vs_global/standard-40x96.json`
- `temp/santorini_v4_p1b_arena_stage_vs_global/full-40x96.json`
- `temp/santorini_v4_p1b_arena_c_vs_ordinary/standard-40x96.json`
- `temp/santorini_v4_p1b_arena_c_vs_ordinary/full-40x96.json`

These pilot checkpoints were written before checkpoint schema 2 was added; their
companion result JSON records the complete contract. New runs embed the target,
optimizer, corpus/plan paths, and untouched-final-test declaration directly in
the checkpoint as well.

The first matched suite was produced with:

```bash
.venv/bin/python screen_santorini_v4_bootstrap.py \
  --train-plan temp/santorini_v4_mixed_pilot/train-plan-10k.npz \
  --selection-plan temp/santorini_v4_mixed_pilot/selection-plan-3k.npz \
  --output-dir temp/santorini_v4_p1b_screen
```

The 12-epoch Candidate-A diagnostic and later candidate/target screens use
separate output directories. `benchmark_santorini_v4_inference.py` exports and
freezes every checkpoint, verifies eager/frozen agreement, measures configured
batch sizes and precision modes, and records FP32/FP16 policy/value agreement.
Its all-model local CPU smoke passes; its CPU rates are not target measurements.
`arena_santorini_v4_selection.py` loads frozen V4 checkpoints through the normal
batched MCTS path, records exact selection openings/seeds, per-game and paired
block outcomes, cluster-bootstrap intervals, and inference counts, and
explicitly declares that final evaluation data was untouched.

The Santorini suite passes 268 tests. Repository-wide
automatic discovery additionally requires TensorFlow for legacy Othello tests;
that optional dependency is absent from the current virtual environment.
