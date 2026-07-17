# Santorini AI V3 Architecture and Training Design

This document is the authoritative specification for Santorini AI V3. It replaces the original V3 draft and the subsequent architecture revision.

V3 is a clean training line. It is not compatible with checkpoints or replay buffers produced by the earlier, 1,600-action V3 experiment. V2 remains supported for evaluation and reference generation, but V3 training begins from random weights and learns worker placement and standard play together.

## Goals

V3 is designed to:

1. Increase evaluator capacity without making self-play prohibitively expensive.
2. Learn all four opening placements organically from the empty board.
3. Train continuously without promotion-gating every update through an arena.
4. Preserve enough state to resume a long Kaggle run exactly, including optimizer state.
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

V3 stores one canonical board/policy example per played position. Each time an example is sampled for training, one of the eight rotation/reflection symmetries is selected uniformly and applied to both the encoded board and the spatial policy. This preserves geometric augmentation without materializing eight replay entries per position. Older replay files that already contain expanded symmetries remain valid: their examples receive another uniformly sampled symmetry during training and age out normally as new single-position windows enter the retained history.

## Training From Scratch

The production V3 run starts from random weights. V2 weights, V2 replay data, and the earlier V3 bootstrap are not training inputs.

Every self-play game begins at the empty board. The resulting replay therefore contains examples from:

- all four placement plies;
- early standard play;
- midgame positions; and
- late tactical positions.

The policy target is the MCTS visit distribution at temperature `1.0`. The action played in self-play is sampled from a separately temperature-adjusted copy of those same root visits. The value target is the final game outcome from the acting player's perspective.

### Fresh-data replay reuse

V3 budgets optimization from the amount of new self-play data rather than the total replay size. The default target is 16 training draws for every newly generated, non-validation position. With a batch size of 512, the requested step count is:

```text
requested_steps = ceil(fresh_training_examples * replay_reuse / batch_size)
actual_steps = min(requested_steps, max_train_steps)
```

The 1,500-step setting remains a safety ceiling; under the expected 80-game workload, the fresh-data calculation normally requests far fewer steps. `epochs=3` divides those steps into three reporting segments and does not multiply the budget. Every draw independently receives a random D4 symmetry, so augmentation variety is retained without treating eight possible transforms as eight additional stored examples.

Telemetry records requested and completed steps, fresh training positions, target and actual replay reuse, base replay passes, stored and virtual replay sizes, and average draws per stored position. V2 retains its legacy epoch-based schedule.

### Held-out replay validation

Five percent of V3 replay boards are held out by a stable hash of the network-visible position. The hash normalizes anonymous same-color worker labels and groups all eight rotations/reflections, preventing on-the-fly augmentation or an equivalent replay copy from leaking a held-out input into training. The split is reconstructed from the replay file and is therefore deterministic across Kaggle sessions. Held-out positions are never sampled by the optimizer.

After every iteration, validation reports policy cross-entropy, policy KL relative to the MCTS target, value mean-squared error, policy top-1 agreement, and value-sign accuracy. Every metric is separated into placement and standard-play positions, with aggregate policy and value losses also recorded. These values measure generalization to replay-like positions; milestone, anchor, and greedy matches remain the evidence for playing strength.

### Optimizer and learning-rate schedule

V3 uses AdamW with an initial learning rate of `3e-4` and weight decay of `1e-4`. The default absolute-iteration schedule changes the rate to `1e-4` at iteration 200 and `3e-5` at iteration 400. All values and milestones are command-line configurable, and `--lr-schedule none` disables the milestones. Absolute iteration numbers ensure that a resumed Kaggle chunk does not restart the schedule.

An older V3 Adam checkpoint can be resumed into AdamW: its compatible moment estimates are loaded, and the configured learning rate and weight decay are applied on the next training iteration. V2 retains Adam at `1e-3`, no weight decay, and no default schedule.

### Exploration

V3 separates the policy saved for training from the policy used to choose the self-play action:

- The replay policy always uses `policyTargetTemperature=1.0`, preserving the relative MCTS visit counts even when action selection is greedy.
- All four placement actions use `placementTemperature`, which defaults to `1.0`.
- Standard actions use temperature `1` until `tempThreshold`, counted from the first standard action, and temperature `0` afterward.
- Both policies come from the same MCTS search; this does not run extra simulations.
- Root Dirichlet noise is enabled on full-search turns in V3 latest-training mode.
- The default Dirichlet parameters are alpha `0.30` and epsilon `0.25`.

Action temperature controls how self-play samples from the MCTS visit distribution; it does not enforce a fixed percentage of deliberately bad moves. Target temperature controls only the replay label. Dirichlet noise supplies additional root exploration, while the legal-action mask prevents invalid placement or move/build actions.

### Exact tactical shortcuts

Before standard MCTS, Santorini checks exact level-three tactics. If the current player can win immediately, search is skipped and the policy is distributed over the equivalent winning actions. If the opponent threatens an immediate level-three win, actions that leave such a win available are removed from the root. A single safe defense is played directly; multiple safe defenses are searched normally; if every legal action permits an immediate reply win, the position is recorded as a proven loss in two plies.

The shortcut never declares a longer heuristic combination solved. It is enabled by default for self-play and evaluation so both contestants use the same rules, and `--no-tactical-shortcuts` provides an ablation. Exact policies are stored only when playout-cap randomization selected a full-search turn, preserving the existing replay-volume schedule. Telemetry records each tactical case and the nominal MCTS simulations skipped.

### Playout-cap randomization

V3 optionally separates the search budget needed for strong policy targets from the independent games needed for value learning. The production notebook enables playout-cap randomization with:

- full 96-simulation searches on all four placement turns to preserve opening quality and policy coverage;
- 96-simulation full searches on 25% of standard turns;
- 32-simulation fast searches on the other 75% of standard turns;
- 240 games per iteration, approximately doubling independent outcomes for only about 10% more nominal root simulations than Run 6 at its observed game length; and
- replay storage only for full-search turns.

Fast standard turns disable root Dirichlet noise and choose the highest-visit action without temperature sampling. They still advance the game and determine its final outcome, so each game contributes value information through its full-search positions without adding a low-quality policy target. Full turns retain the normal temperature-1 replay policy, configured action temperature, and root noise. The choice is sampled independently at every standard turn and is covered by the resumable RNG state. Placement randomization is available as an explicit option but is not enabled in the production notebook.

This follows the core playout-cap randomization method from KataGo, adapted conservatively to Santorini's much smaller full-search budget. It requires no replay-format change: older examples are implicitly full-search examples, and new fast-search positions are simply not stored. Telemetry records full/fast move counts and rates, average and total simulations, and phase-specific full-search rates.

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

The default full preset retains 20 iterations and caps each iteration queue at 200,000 examples. Compact replay avoids storing a dense 1,625-value vector for every example on disk. With on-the-fly symmetry, a steady 20-iteration V3 replay contains one entry per played position instead of eight pre-expanded entries.

Replay writes are non-atomic by default to avoid temporarily requiring space for both the old and new archive. Atomic saving remains available when additional disk space is acceptable.

The replay-maintenance utility can retain only the newest history windows and, when transitioning a legacy expanded replay to on-the-fly augmentation, collapse each consecutive eight-symmetry group to one representative:

```bash
.venv/bin/python trim_santorini_replay.py latest.examples.npz \
  --keep-last-windows 5 \
  --collapse-symmetry-group-size 8
```

The utility validates history lengths, sparse-policy offsets, and array counts before atomically replacing the archive.

## Checkpoints and Exact Resume

Latest mode writes two current checkpoints:

- `latest-training.pth.tar`: model weights, optimizer state, architecture metadata, iteration and self-play configuration metadata (including MCTS simulations and policy-target temperature), and random-number-generator state needed for training resume.
- `latest.pth.tar`: weight-only checkpoint for inference or evaluation.

Milestone weight checkpoints use `checkpoint_<iteration>.pth.tar`.

Preserving the optimizer means restoring its accumulated first- and second-moment estimates along with the weights. Loading only model weights would reset those estimates and change the effective optimization trajectory after every Kaggle session. A proper resume therefore loads both `latest-training.pth.tar` and `latest.examples.npz` with optimizer loading enabled.

The saved iteration number is restored so iteration numbering, milestone scheduling, and telemetry continue rather than restarting at one.

When a run resumes exactly on a milestone boundary and the numbered checkpoint was not included in the downloaded export, startup recreates that weight-only milestone from the loaded current model. This lets the next milestone compare against the correct endpoint without requiring an otherwise redundant upload.

Evaluation-only milestone and anchor matches preserve and restore Python, NumPy, Torch, and CUDA random-number-generator state. Loading an opponent or sampling a telemetry game therefore does not change the subsequent self-play or optimization trajectory recorded by `latest-training.pth.tar`.

## Telemetry

Telemetry is written after every completed training iteration to:

- TensorBoard event files under the configured telemetry directory; and
- `telemetry.jsonl`, with one machine-readable record per iteration.

The training notebook launches TensorBoard so metrics can be viewed while the run is active.

### Training and game metrics

The run records available training losses and operational measurements, including:

- whole-iteration policy, value, and total loss;
- final-segment policy, value, and total loss, with the historical unqualified loss fields retained as aliases;
- held-out placement and standard policy/value metrics;
- optimizer steps, replay reuse, learning rate, and weight decay;
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

Each player's two selected worker locations are also paired per completed self-play game. Geometry uses Chebyshev distance, matching the workers' eight-direction movement, and records separately for Player 1 and Player 2:

- mean worker distance from the center (`0` center, `1` inner ring, `2` outer ring);
- the rate at which both workers occupy the central `3 x 3`;
- mean worker-to-worker separation and the rates of adjacent (`1`), moderate (`2`), and far (`3-4`) separation; and
- center distance and separation separately for games that player eventually won and lost.

For uniformly random distinct placements on a `5 x 5` board, the comparison baselines are `1.60` mean center distance, `12%` both-central placements, and `2.36` mean worker separation. Centrality has a natural direction, but separation is descriptive: neither maximum nor minimum distance is assumed to be optimal.

The full-search visit targets for both players additionally report expected center distance, center/inner/outer-ring probability mass, and expected second-worker separation. Player 2 telemetry includes whether the center was available before its first placement and its center probability mass only on turns where the square was legal. These target-weighted measurements are less sensitive than selected actions to temperature sampling, although root Dirichlet exploration is still present and averages out across the iteration.

After all four workers are placed, interaction telemetry records center ownership, minimum opposing-worker distance, Player 2's mean distance to its nearest Player 1 worker, and the rate at which Player 2 workers are adjacent to Player 1. These measurements distinguish a deliberate reply or contesting strategy from placement changes forced by Player 1 occupying preferred squares.

### Raw-policy metrics

On a sample of replay boards, V3 records metrics separately for placement and standard positions:

- probability mass assigned to legal actions; and
- normalized entropy over legal actions.

Entropy should be interpreted diagnostically rather than as a score that must always decrease. Some positions are genuinely ambiguous, and an abrupt change matters more than a universally monotonic trend.

### Self-play target metrics

For the replay labels produced during each iteration, V3 records the mean policy-target entropy, mean number of visited actions, and one-hot target rate separately for placement and standard positions. These measurements verify that the temperature-1 training targets remain informative after standard action selection switches to temperature zero.

### Milestone matches

Every 20 iterations by default, V3 saves a milestone checkpoint. Once both endpoints exist, it plays a non-gating match between the current checkpoint and the checkpoint 20 iterations earlier. Each milestone retains 40 standard and 40 placement-inclusive games; reducing frequency preserves the existing per-match confidence while halving evaluation overhead.

Each milestone now contains two paired evaluations:

- The standard-play match uses 20 fixed, symmetry-distinct completed openings for 40 games. Every opening is played once with each network in each seat. The suite is sampled once from a fixed telemetry seed and reconstructed identically after a Kaggle resume.
- The placement-inclusive match starts from the empty board. Placement actions are sampled from each network's MCTS visit distribution using 20 fixed per-game seeds and placement temperature `1.0`; standard actions remain deterministic. Reusing the paired seeds makes the measurement reproducible while avoiding 20 identical copies of the same empty-board game.

The two results are kept separate because a completed-opening match measures standard play cleanly, while the placement-inclusive result measures the combined effect of learned setup and play. Both include wins, draws, decisive-game win rate, and an approximate 95% Wilson confidence interval. The match never accepts, rejects, or rolls back a model. With a fresh run, iteration 20 creates the first milestone and iteration 40 produces the first 20-iteration comparison.

The completed record and confidence interval are logged immediately in the training output and are also retained in TensorBoard and `telemetry.jsonl`.

### Fixed strength anchor

Training can optionally load a fixed V1, V2, or V3 checkpoint as an evaluation-only anchor. The recommended V3 training configuration uses the early V1 `santorini_kaggle_training2` model every 10 iterations because it provides a stable absolute target that does not move with the self-play population.

The anchor uses the same fixed symmetry-distinct completed-opening suite and equal MCTS simulation budgets for both contestants. It therefore measures standard-play strength and is compatible with V1's legacy policy, which cannot select V3 placement actions. Anchor results are logged live and written under `anchor_*` fields in TensorBoard and `telemetry.jsonl`. They never gate, roll back, or otherwise affect training.

### Greedy benchmark

The notebook's final greedy benchmark is also a standard-play evaluation. It uses 20 fixed, distinct symmetry-reduced completed openings, plays both seat assignments, and resets V3's MCTS tree before every game. The fixed `--opening-seed` makes the suite identical across runs, and the JSON result records both the requested and distinct opening counts.

The equivalent terminal command is:

```bash
.venv/bin/python pit_santorini.py \
  --architecture v3 \
  --checkpoint-folder ./temp/santorini_v3_run5 \
  --checkpoint-file latest.pth.tar \
  --baseline greedy \
  --opening-source unique \
  --opening-seed 20260715 \
  --games 40 \
  --sims 64
```

Using `--opening-source game` with V3 intentionally starts every game from the same empty board. Although MCTS is now reset correctly, deterministic temperature-zero play can still repeat the same trajectory, so that mode is a single-start diagnostic rather than a diverse strength benchmark. Placement-inclusive strength is measured by the seeded milestone match instead.

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
| Iterations per notebook chunk | Configurable; 10 by default |
| Self-play games per iteration | 240 with playout-cap randomization |
| Full / fast MCTS simulations | 96 / 32 |
| Full-search probability | 100% placement; 25% standard |
| Self-play batch size | 128 |
| Optimizer | AdamW |
| Initial learning rate | 0.0003 |
| Weight decay | 0.0001 |
| Learning-rate schedule | 0.0001 at iteration 200; 0.00003 at iteration 400 |
| Training reporting segments (`epochs`) | 3 |
| Replay reuse target | 16 draws per new training position |
| Validation replay fraction | 5% deterministic holdout |
| Maximum optimizer steps | 1,500 per iteration |
| Training batch size | 512 |
| Symmetry augmentation | Random on-the-fly rotation/reflection |
| Replay history | 20 iterations |
| Milestone interval | 20 iterations |
| Standard milestone games | 40 on 20 fixed completed openings |
| Placement-inclusive milestone games | 40 using 20 fixed seed pairs |
| Fixed V1 anchor | Optional, 40 games every 10 iterations |
| Replay policy-target temperature | 1.0 |
| Placement temperature | 1.0 |
| Dirichlet alpha / epsilon | 0.30 / 0.25 |

These values are an initial operating point, not immutable architecture constants. Throughput, strength telemetry, and run stability should guide later tuning.

### Fresh run inputs

A fresh run requires only the repository and a Kaggle GPU. Leave the resume source empty. The reference suite and fixed strength anchor are optional evaluation inputs. Attach either as a Kaggle Dataset and configure `REFERENCE_SUITE` and `ANCHOR_CHECKPOINT` in the notebook. `ANCHOR_CHECKPOINT` may be the exact `best.pth.tar` file or a dataset directory containing a single preferred `best.pth.tar`.

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

### Compact MCTS edge storage

MCTS stores policy priors, Q values, and visit counts only for a state's legal actions. A compact slot maps each legal edge back to its global 1,625-action index. Dense visit policies and Q/count arrays are reconstructed only for public outputs and reference-suite export, so replay targets and external artifacts retain their established action format. Dense valid-action masks are discarded as soon as a leaf is expanded.

In the isolated M1 hot-path benchmark used for this refactor (80 standard roots, 64 simulations per root, fixed uniform batched predictions), compact storage improved throughput from approximately 12,338 to 13,547 simulations per second, or 9.8%. For a representative root with 56 legal actions, NumPy edge-array storage fell from approximately 21.6 KB to 896 bytes. End-to-end Kaggle improvement can differ because GPU inference and training occupy additional wall time.

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
