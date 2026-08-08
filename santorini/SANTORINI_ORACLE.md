# Santorini Native Oracle Bridge

This bridge exposes the sibling `santorini-ai` engine as a deterministic,
fixed-node search opponent and analysis oracle for Santorini V3.

The bridge, rules validation, paired arena, and search diagnostics are the
supported workflows. The completed Run13 distillation experiments did not
produce a checkpoint worth promoting; Run13 remains the active network. Their
scripts and results are preserved under `experiments/santorini_oracle/`.

## Build

The Cargo manifest expects `santorini-ai` beside this repository:

```text
parent/
├── alpha-zero-general/
└── santorini-ai/
```

Install the Rust nightly toolchain, then build the persistent JSONL service:

```bash
cargo build --release --manifest-path tools/santorini_oracle/Cargo.toml
```

Override the binary at runtime with `--oracle-binary` or the
`SANTORINI_ORACLE_BINARY` environment variable. The service supports `ping`,
`reset`, `legal_moves`, fixed-node `analyze`, and independently searched
`analyze_root_moves` requests.

Node limits are preferred to wall-clock limits so repeated experiments are
comparable across runs.

## Validate the adapter

Run both the unit/integration suite and a randomized successor-state check:

```bash
.venv/bin/python -m unittest santorini.test_santorini_oracle
.venv/bin/python validate_santorini_oracle.py --games 20 --max-positions 500
```

The comparison deduplicates successor states. A level-three win is represented
as a move without a build by `santorini-ai`, whereas V3 exposes multiple legal
build-direction aliases that reach the same terminal state.

Do not trust arena or analysis results after a rules change until this check
passes.

## Play a paired arena

```bash
.venv/bin/python pit_santorini_oracle.py \
  --checkpoint-folder ./temp/santorini_v3_run13_gumbel \
  --checkpoint-file latest.pth.tar \
  --games 40 \
  --sims 128 \
  --oracle-nodes 20000 \
  --json-out ./temp/run13_vs_santorini_ai_n20000.json
```

Each symmetry-distinct completed opening is played with the neural network in
both seat assignments. `--games` is the total number of games, and therefore
must be even. The arena displays two progress bars of `--games / 2`: one for
each seat assignment. The oracle transposition table is reset before every
game.

Use the network's normal evaluation search budget, typically 96 or 128
simulations, when measuring playing strength. A 1024-simulation run answers a
different question—how the same network behaves with substantially more MCTS
compute—and is useful as a separate stress test rather than the primary
benchmark.

## Calibrate oracle node budgets

Measure whether deeper independent searches change the oracle answer on
D4-unique early, middle, and late replay positions:

```bash
.venv/bin/python benchmark_santorini_oracle_budgets.py \
  --replay ./temp/santorini_v3_run13_gumbel/latest.examples.npz \
  --positions 500 \
  --budgets 20000 50000 100000 250000 \
  --min-sign-agreement 0.90 \
  --max-median-score-delta 100 \
  --json-out ./temp/run13_oracle_budget_stability.json
```

Detailed records are appended beside the summary, so the same experiment can
resume after interruption. A changed replay, budget list, seed, or selection is
rejected instead of being silently mixed with prior results. Use the shallowest
budget that is sufficiently stable for the intended measurement; a larger
budget is not automatically a better cost/quality tradeoff.

Calibrate candidate label budgets against fresh, materially deeper
oracle-vs-oracle continuation outcomes:

```bash
.venv/bin/python calibrate_santorini_oracle_scores.py \
  --replay ./temp/santorini_v3_run13_gumbel/latest.examples.npz \
  --positions 300 \
  --label-budgets 20000 50000 \
  --adjudicator-nodes 250000 \
  --fit-fraction 0.70 \
  --json-out ./temp/run13_oracle_score_calibration.json
```

The tool splits D4-unique positions into fit and held-out test sets within each
game stage before fitting the score temperature. It reports calibrated and
nominal-temperature Brier score, log loss, expected calibration error, and
score-sign accuracy, with held-out stage and score-magnitude breakdowns. Label
queries reset independently; adjudication resets again and does not reuse a
label search as its first move. The JSONL records and SQLite label cache make
the long study resumable.

## Diagnose ranked root moves

Use equal per-move searches to inspect whether the oracle's root preference is
stable and how concentrated a potential soft target would be:

```bash
.venv/bin/python benchmark_santorini_oracle_root_moves.py \
  --replay ./temp/santorini_v3_run13_gumbel/latest.examples.npz \
  --positions 200 \
  --shallow-nodes-per-move 2000 \
  --deep-nodes-per-move 10000 \
  --top-k 8 \
  --score-temperature 100 \
  --json-out ./temp/run13_oracle_root_stability.json
```

The report includes top-one agreement, top-three overlap, score margins,
target entropy, branching factor, and measured cost overall and by phase.

## Research tooling retained

`generate_santorini_oracle_adversarial_replay.py` remains useful for mining
positions where Run13 and the oracle disagree. `finetune_santorini_oracle.py`
retains the safety mechanisms developed during the experiments: frozen
BatchNorm statistics, phase-balanced rehearsal, held-out selection, source
checkpoint value targets, and independent rehearsal policy/value gates.

These are research components, not a recommended checkpoint-promotion recipe.
The tested adversarial fine-tunes were neutral at direct play at best and
regressed with deeper MCTS. See
`experiments/santorini_oracle/RESULTS.md` before starting another training run.

## Known boundary

The adapter deliberately starts after all four workers have been placed. The
external engine chooses both workers in one joint placement action, while V3
learns four sequential placement actions. Standard-play agreement is therefore
validated, but placement-policy distillation is not supported.
