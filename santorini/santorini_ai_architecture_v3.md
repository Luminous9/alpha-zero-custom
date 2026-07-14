# Santorini AI V3 Architecture and Training Design

This document is the authoritative specification for Santorini AI V3. It replaces the original V3 draft and the subsequent architecture revision.

V3 is a clean training line. It is not compatible with checkpoints or replay buffers produced by the earlier, 1,600-action V3 experiment. V2 remains supported for evaluation and reference generation, but V3 training begins from random weights and learns worker placement and standard play together.

## Goals

V3 is designed to:

1. Increase evaluator capacity without making self-play prohibitively expensive.
2. Learn all four opening placements organically from the empty board.
3. Train continuously without promotion-gating every update through an arena.
4. Preserve enough state to resume a long Kaggle run exactly, including Adam optimizer state.
5. Replace promotion gating with useful, non-gating telemetry and periodic strength measurements.
6. Keep replay files and committed Kaggle outputs within practical disk limits.

## Network Architecture

V3 uses a shared residual tower with separate policy and value heads.

| Component | V3 configuration |
| --- | --- |
| Board size | `5 x 5` |
| Input planes | 6 anonymous, canonical planes |
| Residual blocks | 8 |
| Tower channels | 96 |
| Policy channels | 65 |
| Policy actions | 1,625 |
| Value hidden size | 128 |

The six input planes encode:

1. Current player's workers
2. Opponent's workers
3. Height 1 squares
4. Height 2 squares
5. Height 3 squares
6. Domes, represented by height 4 or greater

Boards are canonicalized to the player who is about to act. The representation therefore does not require separate identities for player one and player two.

The 8-block, 96-channel tower has approximately 1.35 million parameters. It is materially larger than V2's 5-block, 64-channel tower while remaining small enough for batched P100 self-play.

## Unified Action Encoding

V3 extends the spatial V2 policy from 64 to 65 local actions per square:

```text
action = (x * 5 + y) * 65 + local_action
```

The local action has the following meaning:

- `0..63`: one of the 8 move directions combined with one of the 8 build directions
- `64`: place a worker on `(x, y)`

This produces:

```text
25 squares * 65 local actions = 1,625 actions
```

The neural network always emits the same 1,625-value policy. The game-provided legal-action mask determines which portion of that policy is meaningful in the current phase.

### Placement phase

A full game starts from an empty board and decomposes setup into four micro-actions:

1. Player 1 places their first worker: 25 legal squares.
2. Player 1 places their second worker: 24 legal squares.
3. Player 2 places their first worker: 23 legal squares.
4. Player 2 places their second worker: 22 legal squares.

During these plies, all move/build actions are illegal. Only channel 64 on an empty square is legal. Control remains with the same player between that player's first and second placements; MCTS value backup changes sign only when control actually changes.

### Standard phase

After four workers have been placed, channel 64 is permanently illegal. Legal actions are the normal move/build combinations originating from one of the current player's workers.

### Symmetry

Board transformations also transform the spatial action origin. This applies to both move/build actions and placement actions, allowing placement examples to use the same geometric augmentation as standard positions.

## Training From Scratch

The production V3 run starts from random weights. V2 weights, V2 replay data, and the earlier V3 bootstrap are not training inputs.

Every self-play game begins at the empty board. The resulting replay therefore contains examples from:

- all four placement plies;
- early standard play;
- midgame positions; and
- late tactical positions.

The policy target is the MCTS visit distribution. The value target is the final game outcome from the acting player's perspective.

### Exploration

Placement and standard play use related but distinct temperature rules:

- All four placement plies use `placementTemperature`, which defaults to `1.0`.
- Standard moves use temperature `1` until `tempThreshold`, counted from the first standard move, and temperature `0` afterward.
- Root Dirichlet noise is enabled by default in V3 latest-training mode.
- The default Dirichlet parameters are alpha `0.30` and epsilon `0.25`.

Temperature samples from the MCTS visit distribution; it does not enforce a fixed percentage of deliberately bad moves. Dirichlet noise supplies additional root exploration, while the legal-action mask prevents invalid placement or move/build actions.

## Continuous Latest Training

V3 defaults to `latest` training mode instead of arena-gated promotion:

1. Generate self-play with the current network.
2. Append the new examples to the replay history.
3. Train on the retained replay window.
4. Immediately make the trained network the latest network.
5. Save resumable state and telemetry.
6. Begin the next iteration.

There is no accept/reject match in this loop. A temporarily weaker update is not automatically rolled back. Telemetry and milestones reveal regressions, but they do not prove that every gradient update improves playing strength.

V2 and legacy workflows can still select `arena` mode. V3 uses `latest` by default.

## Replay Buffer

Latest mode stores replay history in `latest.examples.npz` using a compact sparse-policy format:

- boards: `int8` arrays;
- values: `float32`;
- policy: nonzero indices and probabilities;
- history-window lengths: retained so iteration boundaries can be reconstructed;
- action size and format version: stored for validation.

The default full preset retains 20 iterations and caps each iteration queue at 200,000 examples. Compact replay avoids storing a dense 1,625-value vector for every example on disk.

Replay writes are non-atomic by default to avoid temporarily requiring space for both the old and new archive. Atomic saving remains available when additional disk space is acceptable.

## Checkpoints and Exact Resume

Latest mode writes two current checkpoints:

- `latest-training.pth.tar`: model weights, optimizer state, architecture metadata, iteration metadata, and random-number-generator state needed for training resume.
- `latest.pth.tar`: weight-only checkpoint for inference or evaluation.

Milestone weight checkpoints use `checkpoint_<iteration>.pth.tar`.

Preserving the optimizer means restoring Adam's accumulated first- and second-moment estimates along with the weights. Loading only model weights would reset those estimates and change the effective optimization trajectory after every Kaggle session. A proper resume therefore loads both `latest-training.pth.tar` and `latest.examples.npz` with optimizer loading enabled.

The saved iteration number is restored so iteration numbering, milestone scheduling, and telemetry continue rather than restarting at one.

## Telemetry

Telemetry is written after every completed training iteration to:

- TensorBoard event files under the configured telemetry directory; and
- `telemetry.jsonl`, with one machine-readable record per iteration.

The training notebook launches TensorBoard so metrics can be viewed while the run is active.

### Training and game metrics

The run records available training losses and operational measurements, including:

- policy loss, value loss, and total loss;
- iteration duration and replay example count;
- games completed;
- average total game length;
- average, median, and percentile standard-play length, excluding placement plies; and
- first-player win rate.

### Placement metrics

For each of the four placement plies, telemetry records:

- normalized selection entropy;
- maximum single-square frequency; and
- a TensorBoard `5 x 5` placement heatmap.

It also records the number of unique completed openings seen during the iteration. These metrics help detect premature opening collapse and visualize learned setup preferences.

### Raw-policy metrics

On a sample of replay boards, V3 records metrics separately for placement and standard positions:

- probability mass assigned to legal actions; and
- normalized entropy over legal actions.

Entropy should be interpreted diagnostically rather than as a score that must always decrease. Some positions are genuinely ambiguous, and an abrupt change matters more than a universally monotonic trend.

### Milestone matches

Every 10 iterations by default, V3 saves a milestone checkpoint. Once both endpoints exist, it plays a non-gating match between the current checkpoint and the checkpoint 10 iterations earlier.

The result includes wins, draws, decisive-game win rate, and a 95% Wilson confidence interval. The match never accepts, rejects, or rolls back a model. With a fresh run, iteration 10 creates the first milestone and iteration 20 produces the first 10-iteration comparison.

The completed record and confidence interval are logged immediately in the training output and are also retained in TensorBoard and `telemetry.jsonl`.

## V2 Reference-Search Suite

The optional reference suite measures whether V3 is approaching or diverging from stable, high-budget V2 search targets. It is evaluation-only and does not compromise training from scratch.

The default suite contains 500 distinct V2 replay positions:

- 150 early positions;
- 200 midgame positions; and
- 150 late positions.

Each position is labeled by the best V2 checkpoint using 1,600 MCTS simulations. The saved target includes the complete visit policy and the search-derived root value. These are strong, reproducible reference targets, not mathematically proven optimal moves or exact game-theoretic values.

The builder is CPU-parallel and reports loading, selection, per-position progress, throughput, elapsed time, and ETA:

```bash
.venv/bin/python -u build_santorini_reference_suite.py \
  --checkpoint-folder ./temp/santorini_bootstrap_result \
  --checkpoint-file best.pth.tar \
  --examples-file ./temp/santorini_kaggle_training6/merged_20.examples \
  --output ./santorini/reference_suites/v2_reference_500.npz \
  --mcts-sims 1600 \
  --workers 4 \
  --threads-per-worker 1
```

Four workers is the conservative default for an M1 MacBook Pro. Worker count can be increased after measuring throughput and memory use.

When attached with `--reference-suite`, iteration telemetry records:

- policy cross-entropy against the V2 visit distribution;
- policy KL divergence;
- top-1 agreement;
- value MSE;
- legal policy mass; and
- normalized legal-action entropy.

For V3 comparison, its 65th placement channel is removed from standard-position policies before comparing them with the V2 1,600-action target.

Five hundred positions are sufficient for routine trend monitoring. A larger suite can reduce sampling noise but increases both labeling cost and per-iteration evaluation time.

## V2 Compatibility

V2 retains its 1,600-action policy, 5-block/64-channel network, and existing post-placement workflows. Checkpoints contain architecture metadata, and incompatible architectures fail explicitly rather than partially loading.

Direct V2/V3 evaluation is naturally performed from a completed placement position because V2 cannot select the four V3 placement micro-actions. Full-game V3 self-play does not use an opening book or opening sampler.

The 1,625-action V3 format intentionally supersedes the earlier 1,600-action V3 experiment. Old V3 checkpoints and replay buffers must not be loaded into this architecture.

## Kaggle Training Workflow

The maintained entry point is `santorini/training_kaggle.ipynb`.

The baseline long-run configuration is:

| Setting | Value |
| --- | ---: |
| Architecture | V3 |
| Training mode | latest |
| Iterations per notebook chunk | 10 |
| Self-play games per iteration | 80 |
| MCTS simulations | 64 |
| Self-play batch size | 128 |
| Training epochs | 3 |
| Training batch size | 512 |
| Replay history | 20 iterations |
| Milestone interval | 10 iterations |
| Placement temperature | 1.0 |
| Dirichlet alpha / epsilon | 0.30 / 0.25 |

These values are an initial operating point, not immutable architecture constants. Throughput, strength telemetry, and run stability should guide later tuning.

### Fresh run inputs

A fresh run requires only the repository and a Kaggle GPU. Leave the resume source empty. The reference suite is optional; attach it as a Kaggle Dataset and configure its path if reference metrics are desired.

### Resume inputs

A resumed run needs the previous export containing at least:

- `latest-training.pth.tar`;
- `latest.examples.npz`; and
- retained telemetry and milestone files.

The notebook copies attached read-only inputs into writable scratch space before training.

### Disk layout

Kaggle storage is split deliberately:

- `/kaggle/tmp/`: repository checkout, active training state, caches, transient checkpoints, and other large scratch files. This space is not retained with the notebook version.
- `/kaggle/working/`: only the compact export that must survive a notebook commit. Notebook outputs saved here count against Kaggle's 5 GB saved-output limit.

The export step must be run after each training chunk. A later Kaggle session attaches that export as an input Dataset and resumes from it.

## Performance Baseline

Existing benchmark artifacts provide the initial hardware expectations:

- On a Kaggle P100, V3 batched MCTS was approximately 1.08 times slower than V2 at batch size 192 in the measured sweep.
- Training throughput was best at batch size 512 among the tested values.
- On the tested M1 CPU, V3 MCTS was approximately 1.7 to 2.0 times slower than V2, but remained fast enough for interactive 64-simulation play.

The benchmark JSON files are retained as evidence rather than treating theoretical convolution counts as wall-clock predictions.

## Implemented Validation

Before the initial V3 training run, the implementation passed:

- the complete Santorini test suite;
- fresh empty-board V3 self-play and training smoke tests;
- compact replay round trips;
- model, optimizer, replay, and iteration resume tests;
- turn-aware MCTS tests for placement micro-actions;
- V2 and V3 benchmark smoke tests;
- a complete V3 game against the greedy player; and
- Kaggle notebook JSON validation.

## Deferred Work

The following ideas are intentionally not part of the current implementation:

- Apple MPS acceleration for the reference-suite builder;
- promotion gating or automatic rollback based on telemetry;
- an exact solved-position puzzle suite;
- a permanent old-data anchor buffer;
- automatic learning-rate reduction after a detected regression; and
- claiming that lower loss or lower entropy alone proves greater playing strength.

These can be added when telemetry from the first clean V3 run provides evidence that they are needed.
