# P2-start oracle sparring calibration

The original deterministic ladder-v1 sweep below is retained as an evaluation
diagnostic. It was superseded for live P2 sparring by the exact-policy
calibration at the end of this document.

This preflight originally calibrated a deterministic proxy against the exact
P1c handoff checkpoint. It did not reuse Run13's old 20k rung.

## Frozen contract

- P1c checkpoint SHA-256:
  `374f0b72adbdf009d19abaed87addbdfe89364ecc1e6a7246b423233be51b42e`
- Oracle 0.2.0 binary SHA-256:
  `aeca25af0b00a1e8e834223990f90c93a5dc8eb8bf1d39fb60e387e2eee8614c`
- Oracle budgets: 5k, 10k, 20k, 50k, 100k, and 250k nodes per move.
- V4 search: 96 simulations, Gumbel mode at evaluation scale zero, exact
  canonical D4 inference, one root frame, deterministic action temperature
  zero, and a 4,096-entry inference cache.
- Openings: the same 20 fixed symmetry-distinct completed placements at every
  budget, with both seats played for 40 games per budget.
- Oracle transposition table and V4 MCTS are reset at every game boundary.
- Selection band: 35-50% V4 score, targeting 42.5%. Select the qualifying rung
  closest to 42.5%; an exact tie goes to the higher node budget.
- Local execution uses Torch FP32 on CPU. This changes throughput, not the
  declared player/search semantics. A different live simulation budget requires
  recalibration; a later CUDA/FP16 check of the selected rung is confirmatory,
  not required to begin implementing P2.

The production command was:

```bash
.venv/bin/python run_santorini_v4_oracle_sweep.py \
  --checkpoint temp/santorini_v4_p1c_pretraining_results/ordinary_6x192_13_global_blend.pth.tar \
  --oracle-binary tools/santorini_oracle/target/release/santorini-oracle \
  --output-dir temp/santorini_v4_p2_oracle_sweep \
  --budgets 5000 10000 20000 50000 100000 250000 \
  --games 40 --simulations 96 \
  --opening-seed 20260921 --bootstrap-seed 20260922 \
  --bootstrap-samples 10000 --inference-cache-size 4096 --device cpu
```

Each completed budget is written atomically and is reused only when its full
contract fingerprint matches, so the sweep is safe to resume.

## Results

| Oracle nodes | V4 score | Paired bootstrap 95% | V4 2-0 | Split 1-1 | V4 0-2 | Seconds |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5k | 85.0% | 75.0%-95.0% | 14 | 6 | 0 | 157 |
| 10k | 65.0% | 55.0%-75.0% | 6 | 14 | 0 | 166 |
| 20k | 55.0% | 42.5%-67.5% | 5 | 12 | 3 | 177 |
| 50k | 42.5% | 27.5%-57.5% | 3 | 11 | 6 | 173 |
| 100k | 42.5% | 27.5%-57.5% | 3 | 11 | 6 | 189 |
| 250k | 22.5% | 10.0%-35.0% | 1 | 7 | 12 | 232 |

For this deterministic proxy, the curve brackets the intended difficulty.
Although 50k and
100k have identical aggregate results, eight of their 20 pair scores differ;
this is not duplicated output. Both land exactly on the target midpoint, so the
frozen upward tie break selected **100k nodes per move** as ladder version 1.
The live-policy results below supersede that selection for replay generation.

The six arenas took 1,094 seconds (18.2 minutes) locally. The authoritative
summary is `temp/santorini_v4_p2_oracle_sweep/oracle-sweep-summary.json`,
SHA-256 `ac691569f05e6530ccacb0938258b412c61d8ef812e19c225477f59a1c5ac131`.
Neither final-test data nor final arena seeds were touched.

## 250k oracle at 1,024 V4 simulations

A separate deep-search diagnostic reran only the 250k matchup on the identical
20 openings and paired seats, raising V4 from 96 to 1,024 simulations. V4
improved from 9-31 (22.5%) to 18-22 (45.0%), with a paired-opening bootstrap
interval of 32.5%-57.5%. The paired score increase is +22.5 percentage points
(95% interval +10.0 to +35.0): ten opening pairs improve, nine are unchanged,
and one worsens. Pair outcomes move from 1/7/12 to 3/12/5 for V4
2-0/1-1/0-2.

This is strong evidence against the deep-MCTS collapse seen in the old
distillation experiments: the P1c network benefits substantially from more
search and approaches parity with the 250k oracle on this suite. It does not
select the live sparring rung, which must match P2's stochastic 96/32 player
and is recalibrated below. The diagnostic
summary is `temp/santorini_v4_p2_oracle_250k_1024/oracle-sweep-summary.json`,
SHA-256 `6597056cfdfd3b72b9f787306225305149a7ba892555747a90ec51e0c56e6a5e`.

## Exact live-policy recalibration (ladder version 2)

The P100 mixed smoke showed that ladder v1 did not transfer to the actual
sparring player: V4 scored only 2-22 at 100k. The proxy had used 96 simulations
on every neural move and Gumbel scale zero, while live P2 uses 96/32
playout-cap randomization (25% full) and Gumbel scale one. We therefore reran
the ladder through `Coach.executeOracleSparringEpisodes`, restoring the P2-start
RNG state and using the exact live search, paired openings/seats, four workers,
and reset-per-game oracle contract.

The 24-game coarse sweep was:

| Oracle nodes | V4 score | Paired bootstrap 95% | V4 2-0 | Split 1-1 | V4 0-2 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2.5k | 66.7% | 54.2%-79.2% | 4 | 8 | 0 |
| 5k | 41.7% | 25.0%-62.5% | 2 | 6 | 4 |
| 7.5k | 41.7% | 20.8%-62.5% | 2 | 6 | 4 |
| 10k | 25.0% | 8.3%-45.8% | 1 | 4 | 7 |
| 20k | 20.8% | 8.3%-33.3% | 0 | 5 | 7 |

Because the two midpoint candidates were broad, a fresh 40-game confirmation
resolved them:

| Oracle nodes | V4 score | Paired bootstrap 95% | V4 2-0 | Split 1-1 | V4 0-2 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 5k | **42.5%** | 27.5%-57.5% | 3 | 11 | 6 |
| 7.5k | 27.5% | 12.5%-42.5% | 2 | 7 | 11 |
| 10k | 30.0% | 17.5%-42.5% | 1 | 10 | 9 |

Freeze **5k nodes per move as live sparring ladder version 2**. The coarse and
confirmation summaries have SHA-256
`dc5bcf0daec63c4281f15b4a4eaedbe2ad7e3dce97e9291c390182edc91b7097`
and `cdc7ac0662f6873f3e4c59b1d7166e80c8b81c2b32750ba6900e1364db71b81a`.
No final-test data or final arena seeds were touched.
