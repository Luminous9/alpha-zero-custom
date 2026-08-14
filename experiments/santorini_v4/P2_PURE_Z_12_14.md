# P2 iterations 12-14 pure-`z` review

## Decision

The first sustained post-bridge block completed iterations 12-14 from the
accepted iteration-11 state. The run is operationally healthy, but iteration 14
is **not promoted**. Iteration 11 remains the production head; iteration 14 and
its replay are retained as diagnostic branch state. Do not continue routine
training from iteration 14 until a larger fresh-suite P100 arena resolves the
mixed strength evidence.

This is not evidence of an abrupt value-target discontinuity at iteration 12.
Iteration 11 already had bridge beta 1.0, so its mathematical target was pure
game outcome `z`; iteration 12 merely stopped evaluating the now-zero-weight
P1c anchor.

## Run contract and integrity

The reusable notebook resumed the exact iteration-11 training checkpoint
(SHA-256 `7789cdba51ba79726596332b520f10bee4e2a1148432d3d6de359da0afaedac6`)
and completed all three requested iterations on a Tesla P100. It retained the
selected contract: 240 games/iteration, 96 full / 32 fast Gumbel search, fixed
2x fresh-data reuse, global AdamW LR 1e-4, 20-window replay, 10% 5k-node
ladder-v2 sparring, and the frozen teacher/seam controls.

All flat snapshots are present. The iteration-14 snapshot and `latest` files
are byte-identical. Replay contains 12/13/14 nonempty windows and
38,163/41,402/44,531 examples respectively. The packaged run therefore passes
the resume and serialization checks independently of its promotion decision.

## Safety and optimization telemetry

| Metric | Iteration 12 | Iteration 13 | Iteration 14 |
| --- | ---: | ---: | ---: |
| Frozen teacher objective | 0.77506 | 0.77837 | 0.78230 |
| Objective step delta | +0.00099 | +0.00330 | +0.00393 |
| Delta from iteration-1 reference | +0.02332 | +0.02663 | +0.03056 |
| Validation policy loss | 1.3643 | 1.3571 | 1.3591 |
| Validation value loss | 0.9101 | 0.9008 | 0.8946 |
| Seam contrast delta from baseline | -0.00576 | -0.00883 | -0.01670 |
| Oracle game score | 6/24 | 13/24 | 9/24 |
| Rolling 80-game oracle score | 38.75% | 38.75% | 38.75% |

No teacher step, cumulative teacher, seam, oracle-ratchet, or search-signal
control fired. The teacher objective rose only +0.00822 from iteration 11 to
14, and the exposure contrast moved in the safe direction. Replay validation
losses are stable or improving. The optimizer/data pipeline is therefore
healthy; this is a playing-strength review, not a corrupt-run diagnosis.

## Live search-signal telemetry

| Metric | Iteration 12 | Iteration 13 | Iteration 14 |
| --- | ---: | ---: | ---: |
| Standard measured positions | 2,104 | 2,188 | 2,084 |
| Standard mean KL(target \| prior) | 0.5247 | 0.5258 | 0.5493 |
| Standard median KL | 0.2930 | 0.2758 | 0.3114 |
| Standard mean total variation | 0.3344 | 0.3303 | 0.3441 |
| Standard prior/target top-1 agreement | 64.0% | 64.5% | 62.9% |
| Placement mean KL | 0.1324 | 0.1409 | 0.1374 |
| Raw prior unavailable | 0 | 0 | 0 |
| Low-signal warning streak | 0 | 0 | 0 |

The 96-simulation stored teacher is not becoming thin. In standard play it
still changes the raw-prior top move roughly 35%-37% of the time, and KL is
stable to slightly higher. Increasing search solely to recover policy teaching
signal is not justified by this block. The metric does not separately isolate
the 32-simulation fast trajectory policy.

## Playing-strength evidence

| Arena | Iteration-14 result | Paired W/S/L | Paired bootstrap 95% interval |
| --- | ---: | ---: | ---: |
| Standard vs iteration 11 | 19-21 (47.5%) | 4/11/5 | 32.5%-62.5% |
| Placement-inclusive vs iteration 11 | 22-18 (55.0%) | 4/14/2 | 42.5%-67.5% |
| Standard vs iteration 1 | 20-20 (50.0%) | 4/12/4 | 37.5%-65.0% |
| Placement-inclusive vs iteration 1 | 19-21 (47.5%) | 4/11/5 | 32.5%-62.5% |

The direct primary arena shows no demonstrated improvement over iteration 11,
but also does not establish a catastrophic head-to-head regression. The fixed
longitudinal suite is more concerning: iteration 11 previously scored 29-11
(72.5%) standard against iteration 1 on the exact same opening/seat blocks,
whereas iteration 14 scored 20-20. Treating each opening pair as a block, the
iteration-14-minus-iteration-11 score change against the common anchor is
-22.5 points with a bootstrap interval of approximately -42.5 to -5.0 points.
Placement-inclusive changes from 25-15 to 19-21 (-15 points), with an interval
of approximately -35 to +5 points.

The direct and longitudinal results can coexist through sampling variance,
style sensitivity, or non-transitivity. They are sufficient to deny promotion,
but not sufficient to declare iteration 14 universally weaker.

## Required next check

Keep iteration 11 as production and pause the iteration-14 branch. Run a
compact P100 confirmation on fresh fixed standard openings:

1. iteration 14 versus iteration 11, 80-120 paired-seat games;
2. iteration 14 versus iteration 1, 80-120 paired-seat games; and
3. placement-inclusive confirmation only as a secondary check, since the
   direct placement result did not regress.

Do not spend a local CPU hour sweeping the intermediate snapshots. If the fresh
confirmation establishes a real regression, then use the saved iteration-12
and iteration-13 checkpoints only to localize its onset and branch again from
iteration 11. Candidate follow-ups include a lower long-run global LR and the
previously retained differential-LR Arm C fallback. If the larger arenas are
even, classify iterations 12-14 as a plateau and decide the next learning-rate
experiment before resuming rather than accumulating more unverified updates.

## Artifacts

- `temp/santorini_v4_p2_iter_12-14/p2-training-contract.json`
  (`144bcafee08be2b8140a5ff665450d57a26b948edcdc8adb46a559f4572c1cf4`)
- `temp/santorini_v4_p2_iter_12-14/telemetry/telemetry.jsonl`
  (`e34ad7ce489c64cbdb11473cc4aba8ec2fc45e9f35eb0eee40d5a52432ffbd65`)
- `temp/santorini_v4_p2_iter_12-14/vs-iteration11.json`
  (`064fed342d11ad6973cfaf245ddc3de5dcb488112d8320ba3167956699378498`)
- `temp/santorini_v4_p2_iter_12-14/vs-iteration1.json`
  (`eda523bca5d748abe761f2c6b1307f310dde8e789e76abc841a819c3c5df4957`)
- iteration-14 resumable checkpoint
  (`f2c3a26ec0a8a7b0a8a0abe6cd18d730a0a97bfae9f532ac9937835b635be14c`)
- iteration-14 inference checkpoint
  (`e91006a0919769d83b0e8097f6e94ba0732540239eab2bbc149bf2465ed4148f`)
- iteration-14 replay
  (`4500335b1cc8f534160b6ea0ac55e7cebb7876c83a8bf710be818c2fbf6e0747`)

No final-test positions or final arena seeds were touched.
