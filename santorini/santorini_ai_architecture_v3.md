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

---

## Phase 2: Offline Bootstrapping (Pure Distillation)

We will bypass the slow, erratic random-weights phase by training the fresh 8x96 model on the best available V2 replay buffer.

Dataset choice should be explicit. At the time this draft was written, `./temp/santorini_kaggle_training6_v2/latest.examples` was the safer known-good dataset. If a newer run produces stronger examples, use that instead. Do not blindly use the newest buffer if the run went in a bad strategic direction.

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
- Re-benchmark self-play batch size after switching to V3.
- Keep V2 checkpoints and opening books available for regression matches.

Promotion should be based on repeated arena evidence, not on validation loss alone.
