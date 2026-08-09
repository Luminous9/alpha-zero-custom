# Santorini V4 P1a Mixed Corpus and Feasibility Results

## Mixed-corpus contract

The raw engine corpus remains immutable. A deterministic sampling epoch declares
the desired training distribution rather than deleting or physically duplicating
raw positions:

- stage mix: 20% early, 35% middle, 45% late;
- source mix: 70% engine main line, 20% engine randomized subgame, 10% Run13;
- train/selection/test plans are generated independently from already assigned
  split IDs;
- sampling inside an engine stratum is proportional to its preserved source
  observation count;
- any D4 position occurring in different splits or sources is rejected.

The stage proportions approximate a 6-ply early, 10-ply middle, and 14-ply late
standard game without allowing long late games to dominate. The 10% Run13 weight
is deliberately modest: it hedges distribution transfer without turning the
bootstrap into Run13 imitation. Both are declared pilot defaults, not frozen
hyperparameters.

## Fresh Run13 component

The pilot selected 300 new D4-unique Run13 replay positions, 100 per stage. It
excluded all P0b calibration positions and every position in the 20k engine
pilot. Each oracle query started with a cold TT and used the P0b stage budgets:

| stage | positions | nodes/position |
|---|---:|---:|
| early | 100 | 250,000 |
| middle | 100 | 100,000 |
| late | 100 | 20,000 |

The component preserves Run13's native replay policy and completed-game outcome
while storing the oracle score and search metadata separately. This keeps policy,
self-play outcome, and oracle telemetry distinguishable for the winner-only and
score-plus-winner bake-off. History windows are split conservatively as units.

The resulting component contains 236 train, 27 selection, and 37 test positions.
The 10k train plan hit every requested marginal exactly, drew 5,848 unique corpus
positions, and had zero cross-source D4 overlaps. Eager one-time NPZ loading plus
sparse policy reconstruction sustained about 127k examples/second locally.

Artifacts are under `temp/santorini_v4_mixed_pilot/`. The repeatable commands are:

```bash
.venv/bin/python label_santorini_v4_run13.py \
  --positions 300 --workers 8 \
  --records-out temp/santorini_v4_mixed_pilot/run13-labels-v2.jsonl \
  --label-cache temp/santorini_v4_mixed_pilot/run13-labels.sqlite3 \
  --exclude-records temp/v4_p0b_score_calibration.records.jsonl \
  --exclude-corpus temp/santorini_v4_pilot_branch_010/corpus.npz

.venv/bin/python build_santorini_v4_run13_component.py \
  --replay temp/santorini_v3_run13_gumbel/latest.examples.npz \
  --records temp/santorini_v4_mixed_pilot/run13-labels-v2.jsonl \
  --output temp/santorini_v4_mixed_pilot/run13-component.npz

.venv/bin/python build_santorini_v4_sampling_plan.py \
  --engine-corpus temp/santorini_v4_pilot_branch_010/corpus.npz \
  --run13-component temp/santorini_v4_mixed_pilot/run13-component.npz \
  --output temp/santorini_v4_mixed_pilot/train-plan-10k.npz \
  --draws 10000
```

## Encoder and equivariance feasibility

The 13-plane encoder is implemented with explicit placement behavior and D4
covariance tests. The standard-play phase plane is
`min(sum(heights), 40) / 40`; during placement it is `workers_placed / 4`.
Climb access treats each non-dome origin as a hypothetical worker square,
ignores occupancy at that origin, and respects real occupancy/domes at each
destination. Tactical derived planes remain zero until four workers exist.

Because `escnn` is not installed in the current environment, the first
hand-rolled spike established an exact reference architecture: eight tied orientation
branches, inverse-permuted policy averaging, and invariant value averaging. It
passes all eight policy/value transformations, checkpoint reload, and TorchScript
save/load. At the full 8x96 shape it has 1,351,461 learned parameters and processes
about 333 examples/second for batch 32 on this CPU, versus about 1,456 examples/
second for the ordinary tower. This 4.4x local slowdown is expected and makes it
a correctness oracle, **not** the production candidate.

The optimized hand-rolled tower then replaces full orientation branches with
12-channel regular-representation multiplicities (96 effective channels) and
block-circulant D4 convolutions. Its auxiliary oracle-value head is independently
invariant. At Candidate A's 8x96 shape:

| model | learned parameters | local batch-32 examples/s |
|---|---:|---:|
| ordinary 13-plane control | ~1.35M | 1,456 |
| eight-branch correctness reference | ~1.35M | 333 |
| tied regular tower | 175,489 | 1,254 |
| frozen expanded export | 1,336,705 | 1,498 |

The expanded inference model is bit-identical to the tied model in local FP32
tests (zero maximum policy/value delta), uses ordinary `Conv2d` layers, and
survives TorchScript save/load. The larger exported parameter count consists of
frozen copies of tied weights; the trainable model retains the roughly 1/8
parameterization expected from the regular representation. All eight transforms,
the main value, auxiliary value, checkpoint reload, and exported model pass.

This closes the local P1a architecture-feasibility gate through the hand-rolled
path. P1b must still measure FP32/FP16 and end-to-end behavior on the target GPU;
the local CPU result is not substituted for that decision.

The completed Run13 P100 baseline is 348.45 seconds per amortized iteration,
with 322.97 seconds (92.69%) in self-play and only 8.24 seconds (2.36%) in
training. Run13 executed about 46.8 evaluations per configured inference batch
of 128, so P1b compares full self-play iterations as well as isolated dense
batches. The Candidate A target is measured against this baseline rather than
against the local CPU numbers above.
