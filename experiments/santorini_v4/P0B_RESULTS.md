# Santorini V4 P0b Measurement Results

P0b establishes the oracle-label calibration, label-budget stability, rules
agreement, and Run13 timing measurements needed before the V4 pilot corpus and
architecture work. The source distribution is Run13's final replay
(`latest.examples.npz`, SHA-256
`40df820594b630bc9e87f5f6a06b31d84d7179945e78aa25b3b94feb9727c2a9`).
Oracle results below use `santorini-oracle` 0.2.0, binary SHA-256
`bb0eff84d4d928afdf6f4baccafbb6d1319b007137d7a91f4b17223f278dde89`.

## Rules cross-validation

`validate_santorini_oracle.py` matched the Python and Rust legal-successor sets
on all 500 sampled D4-unique positions from 12 random games. The unbalanced
natural-game sample contained 72 early, 120 middle, and 308 late positions.
This is a clean prerequisite pass, not evidence about playing strength.

## Score stability by stage and node budget

The completed 500-position Run13 stability corpus was re-summarized with two
declared label-stability thresholds: at least 90% score-sign agreement with the
250k search and at most 100 points median absolute score difference. Searches
are independent and reset the transposition table before every query.

| Stage | Positions | Cheapest passing budget | Sign agreement at 20k / 50k / 100k | Median absolute delta at 20k / 50k / 100k |
| --- | ---: | ---: | --- | --- |
| Early | 167 | 250k | 77.8% / 75.4% / 76.0% | 71 / 56 / 51 |
| Middle | 167 | 100k | 86.8% / 86.2% / 91.0% | 126 / 75 / 58 |
| Late | 166 | 20k | 97.0% / 97.0% / 96.4% | 21 / 4 / 0 |
| All | 500 | 250k | 87.2% / 86.2% / 87.8% | 72 / 60 / 46 |

The important result is that a single 20k-50k budget is not stable under the
declared value-label criterion. Late positions are cheap, middle positions need
100k, and no tested sub-250k budget passes for early positions. The 250k row
passes by construction because it is the reference search, not because this
study established its convergence. For the first V4 pilot, use the per-stage
ladder above if labeling throughput matters; use 250k everywhere when uniform
semantics are more important than cost. Treat the thresholds as an explicit
operating choice rather than a proof that 250k is ground truth.

## Score calibration against deeper continuations

The production study samples 300 D4-unique Run13 positions, balanced across
early/middle/late stages. It measures independent 20k and 50k root scores, then
resets again and plays an oracle-vs-oracle continuation at 250k nodes per move.
No label search or transposition-table state is reused as the continuation's
first move. Positions are deterministically split 70/30 by D4 key within each
stage before fitting `sigmoid(score/T)`; mate-band scores are clamped separately.

The run completed all 300 positions: 210 fit and 90 untouched test positions,
with 30 test positions per stage.

| Label budget | Fitted T | Test Brier (T=400 → fitted) | Test log loss (T=400 → fitted) | Test ECE (T=400 → fitted) | Sign accuracy |
| ---: | ---: | --- | --- | --- | ---: |
| 20k | 261.8 | 0.1275 → 0.1229 | 0.3814 → 0.3626 | 0.0830 → 0.0725 | 82.2% |
| 50k | 239.8 | 0.1324 → 0.1291 | 0.3924 → 0.3726 | 0.0962 → 0.0775 | 77.8% |

Both fitted temperatures improve every held-out probability metric over the
engine's nominal T=400 convention. The 20k labels are also slightly better than
50k on this particular held-out set, but the 90-position comparison is not
strong enough to claim that extra search is intrinsically harmful.

The phase breakdown is more consequential:

| Budget | Early Brier / ECE / sign | Middle Brier / ECE / sign | Late Brier / ECE / sign |
| ---: | --- | --- | --- |
| 20k | 0.222 / 0.241 / 70.0% | 0.128 / 0.168 / 76.7% | 0.018 / 0.070 / 100.0% |
| 50k | 0.223 / 0.272 / 66.7% | 0.142 / 0.175 / 70.0% | 0.022 / 0.074 / 96.7% |

Early-stage calibration is poor: the deeper continuation win rate was 26.7%,
while the calibrated mean prediction remained near 50% for both budgets. A
single global temperature cannot remove that phase-dependent bias. Therefore
T=261.8 at 20k is the preferred cheap pilot baseline, but early-stage score
targets should not be treated as equally trustworthy. Before freezing the
score-plus-winner bootstrap blend, the P1 pilot should compare a stage-aware
calibration/downweighting rule against the simple global mapping and retain the
planned winner-only ablation. Late-stage scores are already highly reliable.

The continuation is a deeper-engine outcome, not perfect-play truth. Therefore
the fitted temperature calibrates labels to that operational target only.

## Run13 wall-clock instrumentation

`Coach.py` now records self-play, training, arena/telemetry, serialization, and
uncategorized wall time independently with a monotonic clock. A Run13 checkpoint
resume smoke (two games, four simulations, one training step, telemetry matches
disabled) reconciled the timing fields to total time:

| Phase | Seconds | Fraction |
| --- | ---: | ---: |
| Self-play | 0.958 | 75.2% |
| Training | 0.236 | 18.5% |
| Arena/telemetry | 0.008 | 0.6% |
| Serialization | 0.067 | 5.3% |
| Other | 0.005 | 0.4% |
| Total | 1.274 | 100.0% |

This smoke validates attribution and the actual V3/Run13 code path, but its
fractions are not a representative Run13 throughput baseline. The authoritative
baseline must be collected on the target GPU with the Run13 workload (240 games,
96 simulations, playout-cap randomization, and normal telemetry). Historical
iteration totals cannot recover this split retroactively.

The target-GPU workflow is pinned and refuses to run the full profile without
CUDA unless `--allow-cpu` is explicitly supplied:

```bash
.venv/bin/python benchmark_santorini_run13_timing.py \
  --output temp/run13_timing_ordinary --profile ordinary
.venv/bin/python benchmark_santorini_run13_timing.py \
  --output temp/run13_timing_milestone --profile milestone
.venv/bin/python summarize_santorini_run13_timing.py \
  --ordinary temp/run13_timing_ordinary/timing-summary.json \
  --milestone temp/run13_timing_milestone/timing-summary.json \
  --milestone-interval 10 \
  --json-out temp/run13_timing_baseline.json
```

The ordinary/milestone combination amortizes the once-per-ten-iterations arena
cost instead of reporting either an atypically arena-free iteration or a
milestone on every iteration.

## Reproduction artifacts

Local raw outputs are intentionally kept under `temp/`:

- `v4_p0b_rules_validation.json`
- `v4_p0b_budget_stability.json`
- `v4_p0b_score_calibration.json` and its resumable JSONL/SQLite companions
- `v4_p0b_run13_timing_smoke/telemetry/telemetry.jsonl`
