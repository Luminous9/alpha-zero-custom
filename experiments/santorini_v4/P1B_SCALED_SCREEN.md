# P1b.2 Scaled Architecture Screen

Status: the 100k and 300k unique-data gates and matched GPU screens are
complete. Ordinary 6x192 retains the best supervised objective, but equivariant
E closes to within 0.5 policy-top-1 percentage points and slightly improves
value MSE. Architecture selection therefore remains open through the 1M curve,
canonical-inference benchmark, and selection arenas. The frozen 100k
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
continue to 1M on the same rule. After the 300k screen, the 1M run retains only
equivariant E, ordinary 6x192, and ordinary 10x128; equivariant C and ordinary
8x96 are dominated at both measured scales. Architecture selection uses curve separation,
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

### 300k result and 1M continuation

The matched 300k P100 run obeyed the frozen optimizer, seed, selection, and
no-final-test contracts. Every candidate reached its best objective at epoch
four, so the predeclared 1M continuation gate passes.

| Configuration | Objective | Policy CE | Policy top-1 | Value MSE | Train examples/s |
|---|---:|---:|---:|---:|---:|
| ordinary 8x96 | 0.9146 | 2.4316 | 32.67% | 0.3067 | 1,778 |
| ordinary 10x128 | 0.9024 | 2.3719 | 32.83% | 0.3095 | 1,763 |
| **ordinary 6x192** | **0.8923** | **2.3311** | **33.90%** | 0.3096 | 1,713 |
| equivariant C | 0.9624 | 2.6109 | 29.47% | 0.3096 | 1,405 |
| equivariant E | 0.9149 | 2.4319 | 33.40% | **0.3069** | 1,267 |

Equivariant E trails ordinary 6x192 by 0.0226 objective units (2.5%). A paired
position bootstrap on the frozen 3k selection set gives an E-minus-ordinary
95% interval of 0.0088-0.0366. This is a real supervised-fit gap, not an
architecture verdict: E is only 0.5 points behind on policy top-1, wins early
policy CE (2.3887 versus 2.4465), and has slightly better value MSE. Its deficit
is concentrated in middle/late policy CE. Both candidates continue to improve
at the measured endpoint.

Exact D4 neural behavior is now a selection gate. Bare ordinary inference is not a
candidate. `V4InferenceWrapper(canonicalize_d4=True)` maps every position to the
same anonymous D4 representative used by corpus conversion, evaluates it once,
averages the canonical policy over every transform reaching that representative
when it has a non-trivial stabilizer, and maps the result back. On 128 actual
300k selection positions, canonical inference changed canonical-frame policy
and value by exactly zero and produced zero discrepancy across all eight input
orientations. The targeted unit suite includes an empty-board stabilizer test.
Root orientation averaging must be set to one when this wrapper is active.
This does not by itself make every finite MCTS trace equivariant: action-index
tie breaking and independently indexed Gumbel/Dirichlet noise can break a
stabilizer orbit. A deterministic 32-simulation audit on four asymmetric
selection positions had zero transformed-root policy discrepancy; a symmetric
diagnostic exposed the expected one-visit tie discrepancy. Stabilizer projection
of the returned root policy and canonical-coordinate root noise are therefore
separate pre-self-play integration requirements for either architecture.

The pre-expansion corpus was insufficient for the 1M plan in four binding train
strata:

| Source/stage | Available | 1M quota | Shortfall |
|---|---:|---:|---:|
| main / early | 56,680 | 187,100 | 130,420 |
| main / middle | 181,528 | 258,450 | 76,922 |
| subgame / early | 4,868 | 11,000 | 6,132 |
| subgame / middle | 38,369 | 88,000 | 49,631 |

Early main-line diversity is the binding constraint. Before extrapolating the
zero-prefix duplicate rate into a multi-million-record run, generate and
convert a deterministic randomized-prefix probe:

```bash
cd ../santorini-ai
cargo run -p datagen --release -- \
  --v4 --threads 8 --target-records 200000 --seed 20260910 \
  --output-dir ../alpha-zero-general/temp/santorini_v4_scaled/raw-main-prefix-probe-200k \
  --random-moves-min 1 --random-moves-max 6 \
  --subgame-initial-chance 0

cd ../alpha-zero-general
.venv/bin/python build_santorini_v4_corpus.py \
  temp/santorini_v4_scaled/raw-main-prefix-probe-200k/*.jsonl \
  --output temp/santorini_v4_scaled/main-prefix-probe-200k.corpus.npz
```

The 1-6 probe produced 199,723 unique positions from 200,000 observations, with
9,431 train-early and 37,890 train-middle positions. A follow-up 100k prefix-1
probe produced 99,127 unique positions, including 7,656 train-early and 15,600
train-middle positions. Prefix one is therefore the more efficient binding-
stratum supply despite its slightly higher internal duplicate rate (0.87%
versus 0.14%). The frozen expansion sizes and seeds are:

```bash
cd ../santorini-ai
cargo run -p datagen --release -- \
  --v4 --threads 8 --target-records 1500000 --seed 20260914 \
  --output-dir ../alpha-zero-general/temp/santorini_v4_scaled/raw-main-prefix1-extension-1500k \
  --random-moves-min 1 --random-moves-max 1 \
  --subgame-initial-chance 0

cargo run -p datagen --release -- \
  --v4 --threads 8 --target-records 800000 --seed 20260915 \
  --output-dir ../alpha-zero-general/temp/santorini_v4_scaled/raw-branch-extension-800k \
  --random-moves-min 0 --random-moves-max 0 \
  --subgame-initial-chance 0.25
```

The sizes include projected safety margins of roughly 15-17k main/early and
1.8k subgame/early positions before final combined de-duplication. The two probe
shards remain immutable inputs rather than being discarded. Randomized-subgame
supply remains separate so source attribution and yield remain auditable.

The 5.05M-record conversion exposed a scaling limit in the original split
contract. Exact connected-component assignment over both root games and repeated
canonical positions produced a giant component: 3,307,974 of 4,380,914 unique
positions hashed to selection and only 945,037 to train. This is graph
percolation, not insufficient generated supply, and changing the component seed
would merely move the giant component between splits.

The scaled corpus therefore uses an explicit frozen-holdout contract. The
original `engine-corpus.npz` continues to own selection and test. The raw shards
are scanned for direct provenance; every root game that visits any frozen
selection/test position is blocked, every canonical position visited by such a
game is removed globally from the expanded corpus, and all remaining expanded
positions are train-only. `repartition_santorini_v4_corpus.py` performs this
anchoring without repeating differential validation or aggregation. It scanned
all 5,050,000 records and 86,859 root games, blocked 3,350 games, removed 214,840
positions, and produced `engine-corpus-1m-train.npz`: 4,166,074 validated unique
training positions from 4,752,217 observations. The frozen holdout labels and
boards are not consumed by training, and the Run13 cross-corpus overlap remains
zero.

The first strict-plan attempt also found a deterministic allocator edge case:
3,330 early positions carried both main-line and subgame observations. Ordering
strata only by total supply allowed main-line to consume shared examples even
though its exclusive pool covered its quota. Unique allocation now prioritizes
the least source-exclusive supply. The frozen plan passes with exactly 1,000,000
distinct examples, maximum repetition one, and the declared joint matrix.

The frozen 1M train-plan seed is `20260911`:

```bash
.venv/bin/python build_santorini_v4_sampling_plan.py \
  --engine-corpus temp/santorini_v4_scaled/engine-corpus-1m-train.npz \
  --run13-component temp/santorini_v4_scaled/run13-component.npz \
  --output temp/santorini_v4_scaled/train-1m.npz \
  --draws 1000000 --joint-counts \
    187100 258450 324450 11000 88000 121000 1900 3550 4550 \
  --sampling-mode without-replacement --split train --seed 20260911
```

The Kaggle archive is `temp/santorini_v4_1m_kaggle_bundle.tar.gz` (about
266 MiB), SHA-256
`fd3332ab691ea0878685ddc21097cbacae484cc195494d6a32ba7659737ef734`.
It contains the anchored training corpus, frozen selection corpus, Run13
component, both fixed plans and reports, plus the four runtime files needed by
the current network/data path. Extract it at the repository root because the
archive already contains `temp/santorini_v4_scaled/`.

The narrowed 1M GPU screen is:

```bash
.venv/bin/python screen_santorini_v4_bootstrap.py \
  --engine-corpus temp/santorini_v4_scaled/engine-corpus-1m-train.npz \
  --selection-engine-corpus temp/santorini_v4_scaled/engine-corpus.npz \
  --run13-component temp/santorini_v4_scaled/run13-component.npz \
  --train-plan temp/santorini_v4_scaled/train-1m.npz \
  --selection-plan temp/santorini_v4_scaled/selection-3k.npz \
  --output-dir temp/santorini_v4_scaled/screen-1m \
  --epochs 4 --batch-size 256 --device cuda --data-loading streaming \
  --configs \
    ordinary_10x128_13_global_blend \
    ordinary_6x192_13_global_blend \
    equivariant_e_13_global_blend
```

After selecting the best architecture at scale, rerun winner-only against global
blend on that architecture. The current global result remains the default unless
winner-only matches it under the predeclared preference rule.

### 1M result and architecture-selection continuation

The matched 1M P100 run used the frozen plan, initialization seed, optimizer,
four-epoch contract, and 3k selection holdout. All candidates reached their best
objective at epoch four, and the reserved final test remained untouched.

| Configuration | Objective | Policy CE | Policy top-1 | Value MSE | Train examples/s | Parameters |
|---|---:|---:|---:|---:|---:|---:|
| ordinary 10x128 | 0.8138 | 2.1497 | **38.83%** | 0.2763 | **1,895** | 2.98M |
| **ordinary 6x192** | **0.8115** | **2.1446** | 38.53% | **0.2754** | 1,853 | 4.02M |
| equivariant E | 0.8436 | 2.2557 | 36.97% | 0.2797 | 1,322 | 1.40M |

The two ordinary shapes are tied for selection purposes. Their paired
per-position 6x192-minus-10x128 objective difference is -0.00225 with a 95%
bootstrap interval of -0.00951 to +0.00475. The 10x128 model has 25.9% fewer
parameters, 2.3% higher measured training throughput, and 0.3 percentage points
higher top-1; 6x192 has slightly lower policy CE and value MSE. Neither aggregate
objective nor the paired diagnostic resolves the choice.

Equivariant E's supervised gap is now clear: E-minus-6x192 is +0.0321 objective
units, with a paired interval of +0.0201 to +0.0439. Its policy CE is 0.1112
higher, top-1 is 1.57 points lower, and training throughput is 28.7% lower. E
still has 0.0223 lower early-stage policy CE and 0.0017 lower winner MSE. Exact
symmetry remains mandatory, but it is not unique to E: the ordinary candidates'
canonical wrapper supplies exact neural D4 behavior. Finite-search stabilizer
projection and canonical root noise remain common integration work regardless
of which architecture wins.

This is enough to prefer the ordinary family on supervised fit, but not to name
6x192 or 10x128 as the final architecture or to claim a game-strength rejection
of E. Complete the frozen selection protocol: benchmark raw exported inference
and full wrapper inference on P100, including canonical preprocessing/restoration
for both ordinary models, then run 40-game paired standard and full arenas at 96
simulations for each candidate pairing. If the ordinary arena is inconclusive,
prefer the faster end-to-end model; only then run the winner-only target ablation
on the selected shape.

#### Canonical-seam diagnostic

Hard canonicalization has a genuine representation seam even though it provides
exact output symmetry. A one-ply move or build can change the lexicographically
minimal D4 transform, making adjacent states appear globally rotated or
reflected to the ordinary network. This can defeat convolutional locality; an
equivariant network does not need to choose such a frame. The canonical 3k
selection loss alone does not test transitions across that seam.

Before observing the selection arenas, we froze a diagnostic definition. For
each canonical selection position, enumerate unique legal one-ply successors
and define frame-switch exposure as the fraction whose canonical representative
does not admit the identity spatial transform. Stable rank quartiles contain 750
positions each. Their mean exposures are 0.095, 0.342, 0.514, and 0.751; the
overall mean is 0.426, with 298 zero-exposure positions. Evaluate the unchanged
per-position supervised objective and use paired bootstraps within each quartile.
The primary diagnostic is the high-minus-low contrast in each model-pair loss
difference. This is associative rather than causal because exposure can still
correlate with stage, source, branching factor, and tactics; the report retains
those compositions.

| Paired objective difference | Lowest-exposure Q | Highest-exposure Q | High minus low | 95% interval |
|---|---:|---:|---:|---:|
| E minus ordinary 6x192 | +0.03257 | +0.03256 | -0.00001 | -0.03424 to +0.03448 |
| E minus ordinary 10x128 | +0.03296 | +0.02559 | -0.00738 | -0.04356 to +0.02828 |
| ordinary 6x192 minus 10x128 | +0.00039 | -0.00697 | -0.00736 | -0.02794 to +0.01279 |

No comparison shows a detectable seam-specific loss effect. In particular, E's
gap to 6x192 is essentially unchanged between the lowest and highest exposure
quartiles. This does not prove that hard canonicalization is harmless in MCTS:
the audit tests teacher-target error at states categorized by their local
neighborhood, not temporal search stability or playing strength. It does rule
out the proposed seam effect as an explanation already visible in the frozen
supervised losses. The paired arenas remain necessary and were not inspected
before this definition and result were recorded.

The selection handoff is
`temp/santorini_v4_1m_selection_bundle.tar.gz` (about 49 MiB), SHA-256
`7602f8b1b5198d9f32110bebf0a0b03dd44dd1d9b5df4c1eafce8c2f1e1f95be`.
It contains the three best checkpoints, frozen selection inputs, canonical
inference/runtime files, seam diagnostic, benchmark driver, and
`run_santorini_v4_1m_selection.sh`.
The checkpoint payloads use a neutral `.pt` suffix; naming PyTorch's internal
ZIP container `.zip` caused Kaggle input processing to expand it into a
directory, which is not loadable by `torch.load`.
Extract it at the repository root and run:

```bash
./run_santorini_v4_1m_selection.sh
```

The script first reproduces the CPU seam diagnostic, then records both raw
frozen-model throughput and complete wrapper throughput at FP32/FP16 and batches
1/8/32/64/128/192. End-to-end ordinary cases
include exact D4 canonicalization and policy restoration; E uses its native
equivariant wrapper. It then runs the complete three-pair round robin at both
standard and full gates: 40 games per pairing, 96 simulations, paired seats,
selection seed `20260814`, deterministic standard-play Gumbel scale zero, and
no root symmetry averaging. This is selection evidence only; final arena seeds
and final-test positions remain untouched.

## Run13 preview and G1

A small standard/full arena against Run13 may be run for the best 100k checkpoint
as a mapping, placement, and gross-transfer diagnostic. It is not a terminal
strength verdict. Gate G1 remains the selected scaled checkpoint versus Run13 at
96 and 128 simulations, with paired blocks and equal-cost reporting as declared
in `v4-planning.md`.
