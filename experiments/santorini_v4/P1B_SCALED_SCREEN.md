# P1b.2 Scaled Architecture Screen

Status: the 100k unique-data gate and matched GPU screen are complete. The
screen selects ordinary 6x192 provisionally. The strict 300k data gate now also
passes and its matched GPU screen is next; architecture selection remains open
until the scaled curves and selection arenas are available. The frozen 100k
engine corpus contains 281,186 unique positions; the fresh Run13
component contains exactly 10,000 train, 300 selection, and 300 reserved-test
positions. The strict plans contain 100,000 train and 3,000 selection positions,
with no repetition or cross-corpus D4 overlap and exact declared marginals. This
continuation corrects the first P1b pilot's over-broad stop conclusion. It does
not consume the final test split or final arena seeds.

## Decision being tested

The first pilot established only that its ordinary 8x96 checkpoint beat the
tested equivariant checkpoints after training on 5,848 unique positions. P1b.2
tests whether that ordering persists with materially more data and whether a
larger ordinary tower provides the architecture upgrade needed for a fresh run.

The matched candidate set is:

| Configuration | Shape | Learned parameters | Role |
|---|---:|---:|---|
| `ordinary_13_global_blend` | ordinary 8x96 | 1,351,461 | pilot baseline |
| `ordinary_10x128_13_global_blend` | ordinary 10x128 | 2,981,445 | deep/wide ordinary probe |
| `ordinary_6x192_13_global_blend` | ordinary 6x192 | 4,024,965 | wide/shallow ordinary probe |
| `equivariant_c_13_global_blend` | regular 6x192 effective | 509,689 | equivariant continuity control |
| `equivariant_e_13_global_blend` | regular 6x320 effective | 1,397,785 | learned-parameter control vs ordinary 8x96 |

Candidate E increases regular-representation multiplicity from 24 to 40. It
does not replace the valid equivariant group projection with an unconstrained
concatenation. All models use 13 planes and the provisional global score+winner
target. Ordinary models retain executable D4 augmentation.

## Data-scale contract

The architecture curves use unique sampling plans at 100k, 300k, and 1M draws.
The Run13 replay anchor is capped at 10k unique train positions rather than
remaining 10% at every scale. Run13 contains only 2,612 D4-unique early replay
positions in total (2,166 under the declared train-window split), so demanding
100k unique Run13 examples in the 1M plan is impossible without generating new
Run13 self-play. Repeating the current 236-position train component hundreds of
times would recreate the original pilot's problem.

The source counts are therefore:

| plan | engine main | engine subgame | Run13 | total |
|---|---:|---:|---:|---:|
| 100k | 70,000 | 20,000 | 10,000 | 100,000 |
| 300k | 225,556 | 64,444 | 10,000 | 300,000 |
| 1M | 770,000 | 220,000 | 10,000 | 1,000,000 |

After the fixed anchor, the engine sources retain the declared 7:2 ratio. Every
plan retains the 20/35/45 early/middle/late marginal. The fixed selection plan
uses 3,000 unique examples with the original 2,100/600/300 source counts. The
final test split is never loaded by the trainer.

The joint source/stage distribution is explicit rather than assuming source and
stage are independent. Randomized subgames are a distribution-diversity source,
not a reason to manufacture an identical early/middle/late mix inside every
source. The declared train matrices (rows = main/subgame/Run13; columns =
early/middle/late) are:

```text
100k:  17100  23450  29450     1000   8000  11000     1900  3550  4550
300k:  54878  75672  95006     3222  25778  35444     1900  3550  4550
1M:   187100 258450 324450    11000  88000 121000     1900  3550  4550
```

All three preserve their exact source and stage marginals. The 3k selection
matrix is `490 745 865 / 50 200 350 / 60 105 135`.

`build_santorini_v4_sampling_plan.py --sampling-mode without-replacement`
enforces the contract and fails if any stratum is short. Its report includes
available supply, unique fraction, maximum repetition, and repeat effective
sample size. With-replacement remains available only for reproducing the first
pilot.

The original corpora do not pass this gate. The targeted 300k raw run converted
successfully into 281,186 unique positions. Its engine train split now clears
the 100k matrix: main-line supply is 17,227/36,891/129,220 and randomized supply
is 1,152/8,844/61,872. The current Run13 component still has only 83/82/71 train
positions by stage, versus 1,900/3,550/4,550, and must be replaced before the
plan can pass. The old component also overlaps the expanded engine corpus across
split assignments, so it is explicitly rejected rather than reused.

The replacement Run13 selector found only 1,934 eligible early train-window
positions after exclusions. Its split-aware contract therefore uses 1,900 early
train positions, leaving a small safety margin, and moves the corresponding 100
early examples to engine main-line without changing either marginal. Labeling
all 10,600 positions took 142 seconds locally. Component materialization then
verified 457,245 sparse policy entries and the exact 10,000/300/300 split.

## Bulk data commands

Use multiple deterministic generation distributions. Zero random-prefix plies
and no branching supply early/main-line records; a separate zero-prefix branch
run supplies randomized subgames. The converter accepts multiple declared
generation distributions only when schema, engine digest, gods, cold-TT policy,
and search limits match, and reports every distribution separately.

Illustrative first 100k-scale bake-off (increase targets based on the converted
stratum report; raw-record counts are not assumed to equal unique train supply):

```bash
cd ../santorini-ai

cargo run -p datagen --release -- \
  --v4 --threads 8 --target-records 150000 --seed 20260830 \
  --output-dir ../alpha-zero-general/temp/santorini_v4_scaled/raw-main-early \
  --random-moves-min 0 --random-moves-max 0 \
  --subgame-initial-chance 0

cargo run -p datagen --release -- \
  --v4 --threads 8 --target-records 100000 --seed 20260831 \
  --output-dir ../alpha-zero-general/temp/santorini_v4_scaled/raw-branch \
  --random-moves-min 0 --random-moves-max 0 \
  --subgame-initial-chance 0.25

cargo run -p datagen --release -- \
  --v4 --threads 8 --target-records 50000 --seed 20260903 \
  --output-dir ../alpha-zero-general/temp/santorini_v4_scaled/raw-main-extension-50k \
  --random-moves-min 0 --random-moves-max 0 \
  --subgame-initial-chance 0

cd ../alpha-zero-general

.venv/bin/python build_santorini_v4_corpus.py \
  temp/santorini_v4_scaled/raw-main-early/*.jsonl \
  temp/santorini_v4_scaled/raw-main-extension-50k/*.jsonl \
  temp/santorini_v4_scaled/raw-branch/*.jsonl \
  --output temp/santorini_v4_scaled/engine-corpus.npz
```

Expand the Run13 component before building the 100k plan. Split-aware stage
quotas avoid the previous equal-stage-only interface and respect the limited
early replay supply after calibration/engine exclusions. Rows below are
train/selection/test; test rows are materialized but never loaded during
selection or training.

```bash
.venv/bin/python label_santorini_v4_run13.py \
  --split-stage-positions \
    1900 3550 4550 60 105 135 60 105 135 \
  --workers 8 \
  --records-out temp/santorini_v4_scaled/run13-labels.jsonl \
  --label-cache temp/santorini_v4_scaled/run13-labels.sqlite3 \
  --exclude-records temp/v4_p0b_score_calibration.records.jsonl \
  --exclude-corpus temp/santorini_v4_scaled/engine-corpus.npz

.venv/bin/python build_santorini_v4_run13_component.py \
  --replay temp/santorini_v3_run13_gumbel/latest.examples.npz \
  --records temp/santorini_v4_scaled/run13-labels.jsonl \
  --output temp/santorini_v4_scaled/run13-component.npz
```

Build the plans only after the converted reports show sufficient supply:

```bash
.venv/bin/python build_santorini_v4_sampling_plan.py \
  --engine-corpus temp/santorini_v4_scaled/engine-corpus.npz \
  --run13-component temp/santorini_v4_scaled/run13-component.npz \
  --output temp/santorini_v4_scaled/train-100k.npz \
  --draws 100000 --joint-counts \
    17100 23450 29450 1000 8000 11000 1900 3550 4550 \
  --sampling-mode without-replacement --split train --seed 20260901

.venv/bin/python build_santorini_v4_sampling_plan.py \
  --engine-corpus temp/santorini_v4_scaled/engine-corpus.npz \
  --run13-component temp/santorini_v4_scaled/run13-component.npz \
  --output temp/santorini_v4_scaled/selection-3k.npz \
  --draws 3000 --joint-counts \
    490 745 865 50 200 350 60 105 135 \
  --sampling-mode without-replacement --split selection --seed 20260902
```

For 300k and 1M, use the matrices declared above. Generate additional immutable
shards and reconvert them; do not obtain scale by switching the plan back to
replacement sampling.

### 300k data-gate result

The final expansion contains 1.65M differentially validated raw observations
from 1.25M zero-branch and 400k branch-distribution records. After D4
aggregation and removal of 243 positions owned by the frozen Run13 component,
`engine-corpus-300k.npz` contains 1,283,400 unique engine positions and has zero
cross-corpus hash overlap. Exclusion occurs after connected-component split
assignment, so removing a Run13-owned position cannot split the remaining
positions from one root game across data splits.

The strict train availability and quota are:

| Source/stage | Available | Drawn | Margin |
|---|---:|---:|---:|
| main / early | 56,680 | 54,878 | 1,802 |
| main / middle | 181,528 | 75,672 | 105,856 |
| main / late | 725,950 | 95,006 | 630,944 |
| subgame / early | 4,868 | 3,222 | 1,646 |
| subgame / middle | 38,369 | 25,778 | 12,591 |
| subgame / late | 269,472 | 35,444 | 234,028 |

The resulting `train-300k.npz` has exactly 300,000 unique corpus positions,
maximum repetition one, repeat effective sample size 300,000, exact declared
source/stage marginals, and zero engine/Run13 D4 overlap. Its seed is
`20260908`.

Corpus expansion changes both sorted position indices and connected-component
split assignments. The trainer therefore accepts a separate
`--selection-engine-corpus`: 300k training reads the expanded corpus while the
fixed 3k holdout continues to read the original 100k corpus. Checkpoint metadata
records both paths. Reinterpreting the old selection plan against the expanded
corpus is prohibited.

## GPU learning curves

The scaled trainer uses streaming preparation so it never materializes a full
`N x 1625` dense policy matrix. A smoke comparison produced identical losses and
metrics to eager preparation; elapsed-rate fields differ as expected.

The compact Kaggle input archive is
`temp/santorini_v4_scaled_kaggle_input.tar.gz` (about 19 MB), SHA-256
`702de5ac485e54be18ae8c237f2459616aa2278c890f781ba25bece71ae6f532`.
It contains only the converted engine corpus, Run13 component, strict train and
selection plans, and their reports; raw JSONL and reserved evaluation artifacts
are excluded. Extract it into `temp/santorini_v4_scaled/` in the Kaggle working
copy before running the command below.

For 300k, use `temp/santorini_v4_300k_kaggle_bundle.tar.gz` (about 96 MB),
SHA-256 `f130d4a1ab12caa3262cdbde085ceaff61bfdae1be3f34aee308e3691af2883c`.
It contains the expanded and frozen engine corpora, fixed Run13 component,
strict 300k train and 3k selection plans, reports, and the two patched runtime
files. Extract it at the repository root, not inside `temp/`, because its paths
already include `temp/santorini_v4_scaled/`.

Run four epochs at each scale, from the same initialization seed and optimizer
contract. Retain per-epoch selection curves and both the best and final
checkpoints. The five models are explicitly selected so the historical target
ablations are not accidentally rerun:

```bash
.venv/bin/python screen_santorini_v4_bootstrap.py \
  --engine-corpus temp/santorini_v4_scaled/engine-corpus.npz \
  --run13-component temp/santorini_v4_scaled/run13-component.npz \
  --train-plan temp/santorini_v4_scaled/train-100k.npz \
  --selection-plan temp/santorini_v4_scaled/selection-3k.npz \
  --output-dir temp/santorini_v4_scaled/screen-100k \
  --epochs 4 --batch-size 256 --device cuda --data-loading streaming \
  --configs \
    ordinary_13_global_blend \
    ordinary_10x128_13_global_blend \
    ordinary_6x192_13_global_blend \
    equivariant_c_13_global_blend \
    equivariant_e_13_global_blend
```

The 300k command keeps the holdout frozen explicitly:

```bash
.venv/bin/python screen_santorini_v4_bootstrap.py \
  --engine-corpus temp/santorini_v4_scaled/engine-corpus-300k.npz \
  --selection-engine-corpus temp/santorini_v4_scaled/engine-corpus.npz \
  --run13-component temp/santorini_v4_scaled/run13-component.npz \
  --train-plan temp/santorini_v4_scaled/train-300k.npz \
  --selection-plan temp/santorini_v4_scaled/selection-3k.npz \
  --output-dir temp/santorini_v4_scaled/screen-300k \
  --epochs 4 --batch-size 256 --device cuda --data-loading streaming \
  --configs \
    ordinary_13_global_blend \
    ordinary_10x128_13_global_blend \
    ordinary_6x192_13_global_blend \
    equivariant_c_13_global_blend \
    equivariant_e_13_global_blend
```

Repeat with the 300k and 1M plans. Continue from 100k to 300k only if at least
one candidate's selection policy/value curve is still materially improving;
continue to 1M on the same rule. Architecture selection uses curve separation,
paired standard/full arenas between survivors, and measured P100 exported cost.

### 100k result and continuation decision

The matched Kaggle/P100 run used the declared seed and optimizer contract, four
epochs, streaming data preparation, and did not load the reserved final test.
Reported metrics below are from each model's best selection-objective checkpoint;
the objective is `0.25 * policy cross-entropy + global-blend value MSE`.

| Configuration | Best epoch | Objective | Policy CE | Policy top-1 | Value MSE | Train examples/s |
|---|---:|---:|---:|---:|---:|---:|
| ordinary 8x96 | 3 | 1.0607 | 2.8814 | 23.60% | 0.3403 | 1,920 |
| ordinary 10x128 | 3 | 1.0471 | 2.7686 | 26.07% | 0.3549 | 1,949 |
| **ordinary 6x192** | **3** | **1.0154** | **2.7124** | **26.43%** | **0.3373** | **1,857** |
| equivariant C | 4 | 1.0798 | 2.9350 | 23.07% | 0.3460 | 1,519 |
| equivariant E | 4 | 1.0449 | 2.8226 | 24.57% | 0.3392 | 1,330 |

Ordinary 6x192 is the provisional winner. Relative to ordinary 8x96 it lowers
the selection objective by 4.3%, lowers policy CE by 5.9%, raises policy top-1
by 2.83 percentage points, and slightly improves value MSE while reducing
training throughput by only 3.3%. Its policy CE is also lower in every reported
stage and source stratum. A paired position-level bootstrap on the fixed
selection set puts its objective advantage over ordinary 10x128 at 0.023-0.041
(95% interval); this is diagnostic rather than a game-cluster-aware strength
claim.

Capacity helps the equivariant family: E improves substantially over C and
slightly beats the ordinary 8x96 objective. It does not win the architecture
comparison, however: ordinary 6x192 has a 2.8% lower objective, better policy,
nearly identical value MSE, and 40% higher training throughput. Candidate C is
also both slower and worse than the ordinary baseline at this scale.

The continuation gate passes. Policy CE and top-1 improve through epoch four
for all five candidates, while E and C obtain their best combined objective at
epoch four. The ordinary value heads become noisy at epoch four, so the next
experiment uses more unique positions from a fresh initialization rather than
adding epochs over the 100k set. Retain all five candidates at 300k to preserve
the predeclared curve comparison; use ordinary 6x192's epoch-three checkpoint
for the optional Run13 diagnostic preview. Do not select the final architecture
or run the winner-only target bake-off yet.

### 100k Run13 diagnostic preview

The ordinary 6x192 best checkpoint scored 10-30 (25%) in the 40-game,
96-simulation standard-play preview and 8-32 (20%) in the corresponding
full-game preview. Paired-opening bootstrap intervals were 15-35% and 10-30%,
respectively. Both used selection seed `20260814`; reserved final arena seeds
and final-test positions remained untouched.

This is a marginal transfer result, not a terminal G1 verdict. The five-point
standard/full difference is not material at 20 paired blocks, and the 100k
screen intentionally contains no phase-balanced placement distillation. The
result rules out a gross action/encoding mapping failure, supports continuing
to 300k, and keeps placement distillation as an explicit P1c dependency.

The initial preview smoke also exposed an inference-only integration defect:
the ordinary V4 checkpoint loader reconstructed every checkpoint as 8x96 even
when its config declared another width/depth. The loader now honors saved
`channels` and `residual_blocks`, with a non-default-shape regression test. The
screen metrics and checkpoint weights were unaffected because training already
constructed the declared shapes correctly.

After selecting the best architecture at scale, rerun winner-only against global
blend on that architecture. The current global result remains the default unless
winner-only matches it under the predeclared preference rule.

## Run13 preview and G1

A small standard/full arena against Run13 may be run for the best 100k checkpoint
as a mapping, placement, and gross-transfer diagnostic. It is not a terminal
strength verdict. Gate G1 remains the selected scaled checkpoint versus Run13 at
96 and 128 simulations, with paired blocks and equal-cost reporting as declared
in `v4-planning.md`.
