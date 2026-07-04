# Engineering Plan: Santorini AI Architecture V3

Our current V2 model is intentionally small: 5 residual blocks and 64 filters. That has been a good fit for fast MCTS, but the V2 representation may now be strong enough that the network itself is becoming a limiting factor. V3 is an experiment to test a middleweight model: larger tactical capacity, but still small enough to keep self-play practical.

The first V3 candidate will be:

- **8 residual blocks**
- **96 filters**
- **Same V2 game representation:** anonymous 6-plane input and 1,600-action physical-origin policy
- **Pure supervised bootstrap:** no network surgery for the first attempt

This should be treated as an empirical strength-per-time experiment, not an automatic migration. V2 remains the production architecture unless V3 earns promotion in controlled evaluation.

## Architectural Math

The rough scaling estimate is:

- **Depth penalty:** 8 blocks vs 5 blocks = **$1.6\times$**
- **Width penalty:** $(96 / 64)^2 = 1.5^2 =$ **$2.25\times$**

Combined, the residual tower is expected to cost about **$3.6\times$** more convolution work than the current 5x64 model. Actual wall-clock MCTS cost must be measured, because the full turn cost also includes CPU move generation, tree traversal, valid-move masking, batching efficiency, and GPU launch overhead.

Current parameter counts:

- **5x64 V2:** about **381k** parameters
- **8x96 V3 candidate:** about **1.35M** parameters, about **3.5x** V2
- **10x128 larger alternative:** about **2.97M** parameters, about **7.8x** V2

So 8x96 is large enough to be a meaningful test while staying far below the 10x128 jump.

---

## Phase 0: Throughput And Batch-Size Calibration

Before relying on theoretical speed ratios, benchmark the actual hardware target, especially Kaggle/Colab GPU behavior.

Measure:

1. **Raw neural inference throughput**
   - Compare 5x64 and 8x96 using `predict_batch`.
   - Test batch sizes such as `1, 8, 16, 32, 64, 128, 256`.
   - Record latency per batch, positions/sec, and approximate GPU memory use.

2. **Full MCTS throughput**
   - Compare completed simulations/sec from representative midgame positions.
   - Test both unbatched MCTS and batched self-play style leaf evaluation.
   - This is the number that matters for time-odds arena matches.

3. **Training throughput**
   - Test supervised training epoch time at batch sizes such as `64, 128, 256, 512`.
   - Pick the largest batch size that is stable, keeps GPU utilization healthy, and does not degrade validation behavior.

Practical batch-size tuning rule:

- Use **training batch size** to maximize stable supervised learning throughput.
- Use **self-play batch size** to improve GPU occupancy during MCTS leaf evaluation.
- Use **arena batch size** to speed deterministic checkpoint-vs-checkpoint evaluation.
- Prefer powers of two as the first sweep values, but choose based on measured throughput rather than the power-of-two rule alone.
- If GPU utilization is low, increase self-play or arena batch size.
- If game throughput stops improving, CPU move generation/tree work has likely become the bottleneck.
- If memory pressure appears, step down one batch level.

Initial tooling:

```bash
.venv/bin/python benchmark_santorini_phase0.py \
  --device auto \
  --architectures v2,v3 \
  --modes inference,mcts,training \
  --json-out temp/santorini_phase0_benchmark.json
```

For a faster smoke run:

```bash
.venv/bin/python benchmark_santorini_phase0.py \
  --device auto \
  --architectures v2,v3 \
  --inference-batch-sizes 1,8,32 \
  --mcts-batch-sizes 1,8 \
  --training-batch-sizes 64,128 \
  --timed-batches 10 \
  --mcts-sims 16 \
  --mcts-repeats 1 \
  --timed-steps 5
```

P100 benchmark result:

- Hardware: **Tesla P100-PCIE-16GB**
- Raw inference speed penalty for V3 vs V2: about **1.2x to 1.7x**, depending on batch size.
- Batched MCTS speed penalty for V3 vs V2: about **1.1x to 1.2x** over tested batch sizes.
- Training speed penalty for V3 vs V2: about **1.4x to 3.0x**, with the penalty increasing at larger batch sizes.
- GPU memory is not a concern for either architecture in these microbenchmarks.

Best measured settings from `santorini/benchmark.json`:

- **Inference:** throughput keeps improving through batch `256` for both V2 and V3.
- **Training:** throughput is best at batch `512` for both models, though V3 gains only modestly from `256` to `512`.

Follow-up MCTS sweep from `santorini/benchmark2.json`:

- Tested MCTS batch sizes: `64, 96, 128, 192`.
- Best V2 result: batch `192`, about **5,371 sims/sec**.
- Best V3 result: batch `192`, about **4,967 sims/sec**.
- V3 was about **1.08x** slower than V2 at batch `192`.
- There is visible run-to-run noise, especially at batch `64`, so use the broad trend rather than a single row.

Local M1 CPU benchmark from `santorini/benchmark_m1_cpu.json`:

- Raw inference speed penalty for V3 vs V2: about **1.8x to 2.3x**.
- MCTS speed penalty for V3 vs V2: about **1.7x to 2.0x**.
- Best tested local MCTS throughput:
  - V2: batch `32`, about **2,727 sims/sec**.
  - V3: batch `32`, about **1,383 sims/sec**.
- Single-game interactive speed is still comfortable:
  - V2 batch `1`: about **647 sims/sec**, so 64 sims is about **0.10s/turn**.
  - V3 batch `1`: about **377 sims/sec**, so 64 sims is about **0.17s/turn**.

Implications:

- The old **3.6x equal-time penalty** is too pessimistic for batched MCTS on the P100.
- For evaluation, start with **equal sims**, then try a measured time-odds approximation around **V2 64 sims vs V3 59 sims** when using large-batch play, because V3 is about **1.08x** slower than V2 at MCTS batch `192`.
- For self-play and arena batching, use at least batch `64` on this hardware. Prefer batch `192` when enough parallel games are available to keep it fed; otherwise use the largest batch the run can naturally fill.
- For local CPU-only play, use smaller expectations: V3 is closer to **2x** slower than V2, but still fast enough for interactive 64-sim games.
- For supervised V3 bootstrap, start with training batch `512`; a follow-up sweep at `768, 1024, 1536` may be worthwhile before the long run.

---

## Phase 1: Codebase Updates

V3 should be implemented as a parallel architecture, not as a hard replacement for V2.

1. **Add a selectable 8x96 architecture**
   - Keep the existing V2 `NNetWrapper` behavior at 5x64.
   - Add a V3 wrapper/config path that uses `num_residual_blocks=8` and `num_channels=96`.
   - Make `pit_santorini.py` able to load both V2 and V3 checkpoints in the same process for direct comparison.

2. **Keep checkpoint compatibility explicit**
   - A V2 checkpoint should load only into the V2 wrapper.
   - A V3 checkpoint should load only into the V3 wrapper.
   - Evaluation commands should make architecture choice visible in logs and JSON output.

3. **Add benchmark tooling**
   - Create a small benchmark script for inference throughput and MCTS simulations/sec.
   - Use the measured speed ratio later when choosing fixed-sim and time-odds evaluation settings.

Implementation status:

- `santorini.pytorch.NNet` now exposes V2 and V3 wrappers through `build_nnet(game, architecture)`.
- V2 remains the default `NNetWrapper`; V3 uses `V3NNetWrapper` with 8 residual blocks and 96 channels.
- New checkpoints save architecture metadata so V2/V3 mismatches fail clearly.
- `main_santorini.py` accepts `--architecture v2|v3`.
- `pit_santorini.py` accepts `--architecture v1|v2|v3` and `--opponent-architecture v1|v2|v3`.

---

## Phase 2: Offline Bootstrapping (Pure Distillation)

We will bypass the slow, erratic random-weights phase by training the fresh 8x96 model on the best available V2 replay buffer.

Dataset choice should be explicit. At the time this draft was written, `./temp/santorini_kaggle_training6_v2/latest.examples` was the safer known-good dataset. If a newer run produces stronger examples, use that instead. Do not blindly use the newest buffer if the run went in a bad strategic direction.

For the first V3 bootstrap run, use the known-good `training6_v2/latest.examples` dataset rather than `training7`.

Create a new supervised bootstrap script with:

1. **Master dataset input**
   - Load a selected `.examples` replay buffer.
   - Accept the dataset path as a CLI argument.

2. **Fresh V3 initialization**
   - Instantiate the 8x96 model from random weights.
   - Do not use V2 network surgery for the first attempt.

3. **Train/validation split**
   - Use a deterministic 90% / 10% split.
   - Track policy cross-entropy, value MSE, and total validation loss.

4. **Checkpointing**
   - Save the best checkpoint by validation loss.
   - Also save final training metadata: dataset path, seed, architecture, epochs, batch size, validation losses, and timestamp.

5. **Stop condition**
   - Train up to roughly **10 to 15 epochs** initially.
   - Stop early if validation loss flatlines or worsens for a configured patience window.
   - If validation loss improves but arena strength does not, treat that as evidence that imitation quality is not enough and self-play continuation is required.

Implementation status:

- `bootstrap_santorini_v3.py` performs supervised V3 bootstrapping from a `.examples` replay buffer.
- It defaults to `temp/santorini_kaggle_training6_v2/latest.examples`.
- It creates a deterministic train/validation split, saves the best checkpoint by validation loss, saves a final checkpoint, and writes `bootstrap_metadata.json`.
- It supports `--max-examples` for smoke runs and `--cpu` for local testing.
- `santorini/bootstrap_v3_kaggle.ipynb` provides the Kaggle workflow for the first V3 bootstrap run.

Recommended Kaggle command:

```bash
.venv/bin/python bootstrap_santorini_v3.py \
  --examples-file /kaggle/input/<dataset-containing-training6-v2>/latest.examples \
  --output-folder /kaggle/working/Santorini-AZ/v3_bootstrap \
  --architecture v3 \
  --epochs 15 \
  --batch-size 512 \
  --validation-fraction 0.10 \
  --patience 3 \
  --seed 7 \
  --quiet
```

First Kaggle bootstrap result:

- Dataset: `training6_v2/latest.examples`
- Train examples: **359,431**
- Validation examples: **39,937**
- Architecture: V3, 8 residual blocks, 96 channels
- Hardware: CUDA on Kaggle P100
- Batch size: `512`
- Early stopping: stopped after epoch 9 because validation loss did not improve for 3 epochs
- Best checkpoint: **epoch 6**
- Best validation total loss: **2.8370**
- Best validation policy loss: **2.0447**
- Best validation value loss: **0.7924**
- Output checkpoint copied locally to `temp/santorini_v3_bootstrap_result/best.pth.tar`

Training/validation curve summary:

- Validation total loss improved quickly from **3.7115** at epoch 1 to **2.8370** at epoch 6.
- Training loss kept falling through epoch 9, while validation loss drifted upward after epoch 6.
- This is a clean early-stopping shape: the model learned the replay distribution, then began to overfit.
- Local load smoke passed: the checkpoint reports architecture `v3`, predicts a `(1600,)` policy, and plays a tiny pit through `pit_santorini.py`.

Local smoke command:

```bash
.venv/bin/python bootstrap_santorini_v3.py \
  --examples-file temp/santorini_kaggle_training6_v2/latest.examples \
  --output-folder temp/santorini_v3_bootstrap_smoke \
  --architecture v3 \
  --epochs 1 \
  --batch-size 32 \
  --max-examples 256 \
  --cpu \
  --quiet
```

---

## Phase 3: Fixed-Sim Evaluation

Before using clock-based evaluation, run same-search matches to isolate evaluator strength.

1. **Matchup**
   - Current best 5x64 V2 checkpoint vs bootstrapped 8x96 V3 checkpoint.

2. **Search settings**
   - Use the same number of MCTS simulations for both sides, such as 64 and 128.
   - Use paired openings from the opening book where possible.
   - Keep deterministic play with action temperature 0.

3. **Interpretation**
   - If V3 loses badly at equal sims, the larger network has not learned a better evaluator yet.
   - If V3 wins at equal sims, it is a promising candidate for time-odds testing.
   - If results are close, continue with self-play or a better distillation dataset before promotion.

First fixed-sim evaluation results:

- Assumption: records are listed as **V3 wins - V2 wins**.
- 100 games at 128 sims using the `santorini_bootstrap_result` opening book: **37-63**.
- 200 games at 128 sims using `bootstrap_arena_suite`: **82-118**.
- Combined fixed-sim result: **119-181** across 300 games, or **39.7%** for V3.

Interpretation:

- The bootstrapped V3 checkpoint is clearly weaker than the V2 opponent at equal 128-sim search.
- Do not advance this checkpoint to time-odds promotion testing as-is.
- This result does not disprove the 8x96 architecture; it says pure supervised bootstrap from `training6_v2/latest.examples` did not produce a stronger evaluator.
- Next useful tests are either self-play continuation from the V3 bootstrap or a stronger/distinct distillation target.

---

## Phase 4: Time-Odds Arena Validation

The key production question is not just whether V3 is smarter per node. It is whether V3 is stronger per unit wall-clock time.

1. **Time-control support**
   - Add clock-limited play to `pit_santorini.py` or a dedicated evaluation script.
   - Example target: **1 second of compute per turn**.
   - Record actual simulations completed per turn for each model.

2. **Measured odds**
   - Do not assume V2 gets exactly 3.6x more simulations.
   - Use the benchmarked or observed simulations/sec ratio.
   - Optionally run a fixed-sim approximation first, such as V2 at `N` sims vs V3 at `N / measured_ratio` sims.

3. **Match design**
   - Use paired openings and balanced seats.
   - Start with 100 games as a smoke test.
   - Use 200 to 400 games before calling a small edge real.

4. **Promotion bar**
   - A 55-45 result over 100 games is encouraging but not definitive.
   - V3 should beat V2 under equal-time conditions before becoming the default production architecture.
   - If V3 wins equal sims but loses equal time, it may still be useful for analysis or high-time-control play, but not normal self-play.

---

## Phase 5: Self-Play Continuation

If the bootstrapped V3 model passes the evaluation gates, resume AlphaZero training from the V3 checkpoint.

Recommended initial continuation:

- Use the best V3 bootstrapped checkpoint as `best.pth.tar`.
- Load the selected replay examples only if intentionally continuing from that distribution.
- Prefer fresh V3 self-play soon after bootstrap so the larger network can move beyond imitation.
- Use the default `main_santorini.py` opening sampler, which now draws self-play and paired arena starts uniformly from the **9,664 symmetry-unique worker placements** instead of from the value-filtered opening book.
- Re-benchmark self-play batch size after switching to V3.
- Keep V2 checkpoints and opening books available for regression matches.

Promotion should be based on repeated arena evidence, not on validation loss alone.

Opening-sampling update:

- Training now defaults to `--opening-source unique`.
- `unique` samples all 9,664 symmetry-unique initial worker placements, then optionally applies random orientation.
- Arena comparisons still use paired openings: the same sampled opening board is used once with each model as first player.
- The old book-based training sampler remains available with `--opening-source book`, or by passing `--opening-book` / `--arena-opening-suite` explicitly.
- `--opening-source game` or the deprecated `--no-opening-book` flag uses `SantoriniGame.getInitBoard()`.
