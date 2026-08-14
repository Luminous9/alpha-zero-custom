# P2 entry, P100 smoke, and transition recheck

The implementation gates are complete: P1c migration preserves outputs,
deterministic complete-search D4 checks pass, compact replay schema 2 preserves
source metadata, paired oracle processes reset at every game boundary, and the
frozen seam diagnostic runs inside the training loop.

## First P100 smoke: informative, not a production checkpoint

Two 240-game arms started from the identical P1c handoff:

- `ordinary`: 240 ordinary games.
- `mixed`: 216 ordinary games and 24 paired 100k ladder-v1 sparring games.

Both completed valid checkpoint/replay round trips. Ordinary took 354.0 seconds
and mixed took 364.4 seconds, so replacing 10% of games added only 2.9% total
wall time. The seam exposure contrast remained clean in both arms. The mixed
arm did create 50% more inference batches because its four-game sparring worker
batch was much smaller than the 128-game ordinary batch; this is an optimization
opportunity, not an entry blocker.

The smoke also found two blockers before a production lineage was created:

1. The live 100k player scored only 2-22. Ladder v1 had been calibrated against
   deterministic 96-simulation search, not the actual stochastic 96/32
   playout-cap player. Exact live-policy calibration now freezes **5k nodes as
   ladder version 2**; its 40-game confirmation score was 17-23 (42.5%).
2. Applying 16x fresh-data reuse to the first small replay window caused a
   cold-start optimization shock. The ordinary and mixed frozen teacher
   objectives moved by +0.346/+0.349, and their equal-96 standard scores against
   P1c were 35.0% and 22.5%. An isolated retrain of the exact ordinary replay at
   2x reuse scored 22-18 (55.0%, paired interval 45-65%) and moved the frozen
   objective only +0.0106. The initial continuation plan therefore ramped reuse
   over absolute iterations 1-8 as 2x, 4x, 6x, 8x, 10x, 12x, 14x, 16x; the
   continuation evidence below supersedes that schedule.

Do not continue from either first-smoke checkpoint. They are measurements only;
the corrected lineage starts again from `p2-start.pth.tar`.

Full measurements and hashes are in
`experiments/santorini_v4/P2_SMOKE_RESULTS.md`; live ladder records are in
`experiments/santorini_v4/P2_ORACLE_SWEEP.md`.

## Revised transition smoke: complete and promoted to iteration 1

The corrected 240-game P100 transition completed in 341.30 seconds of measured
loop wall time. It contains 216 ordinary games and 24 paired 5k ladder-v2
sparring games. V4 scored 9-15 against the oracle, or 37.5%, with pair outcomes
3/3/6 for 2-0/1-1/0-2. This is inside the intended curriculum band and supplies
the first 12 of 40 pairs to the resumable ratchet window.

Iteration-one training correctly used 2x target reuse (25 optimizer steps;
2.072x actual). The frozen teacher objective moved only +0.01414 against its
+0.05 gate, and the seam contrast moved -0.00409 with paired interval
-0.01721 to +0.00922. Every checkpoint/replay digest matches the Kaggle
contract, and replay contains 3,094 ordinary plus 111 oracle-sparring records
with the correct rung, both neural seats, 12 openings, and all stage strata.

The local equal-96 standard check scores 19-21 against the untouched P1c
handoff: 47.5%, paired bootstrap 37.5%-57.5%, with pair outcomes 2/15/3. This is
consistent with parity and unlike the failed 16x mixed arm's 9-31 regression.

Promote this exact transition output to **P2 iteration 1**. Its resumable
checkpoint SHA-256 is
`42b61409ec5ac6a3fd15d93ec6a700b87623e840e468fb8f80e857c1c8df1f78`;
compact replay is
`ffb947a7af216f1c77cc4a1369e407e97dbe2d6b2a51587e3df6b58d3f834f10`.
Keep it as the fixed restart ancestor for the next diagnostic below.
Neither final-test data nor final arena seeds were used.

## First production continuation: superseded

The original iterations 2-20 bundle used the eight-iteration warm-up. It is
retained only as diagnostic history; **do not launch it again**. It requested
4x at iteration 2 and 6x at iteration 3, where the standing gate paused it.

## Iteration-three safety pause and discarded branch

The first production continuation stopped intentionally after iteration 3.
Iteration 2 passed at 4x reuse with a +0.02522 teacher-objective step. Iteration
3 used 6x reuse and raised the objective by +0.05651, just beyond the standing
+0.05 gate. The saved contract reports `status: paused`; all checkpoint and
replay hashes match, and the checkpoint retains iteration 3, its optimizer/RNG
state, objective baseline, and 36 oracle pair scores.

This is not a seam failure. Iteration 3's frozen Q4-minus-Q1 contrast delta is
+0.01233, below the +0.02 warning threshold, with paired interval -0.02253 to
+0.04498. The frozen-objective step consists of about +0.01973 weighted policy
loss and +0.03677 value loss. Main replay validation is essentially flat from
iteration 1 through 3. Strength checks do not show a resolved regression:

- iteration 3 versus iteration 1, equal-96 standard: 19-21 (47.5%), paired
  interval 35%-60%, pairs 3/13/4;
- iteration 3 versus P1c, equal-96 standard: 17-23 (42.5%), paired interval
  30%-55%, pairs 2/13/5; and
- iteration 3 versus iteration 1, equal-96 placement-inclusive: 20-20, paired
  interval 35%-65%, pairs 5/10/5.

The initial diagnostic recommendation was to retain iteration 3 and test one
iteration at lower dose. That recovery bundle is now **superseded and must not
be run**. The production decision is more conservative: discard iterations 2-3
from the lineage and restart from the accepted iteration-one checkpoint and
replay with the slower schedule below. The stopped artifacts and strength
checks remain useful diagnostic evidence, not production ancestors.

## Second continuation: 16-iteration warm-up, discarded

The second continuation used
`temp/santorini_v4_p2_iterations_2_20_warmup16_bundle.zip` (SHA-256
`93bff6de4217b5be3e7be2b680d2e0ca1bc169a4e4c9cdbf495a727fe95cede4`)
and the exact accepted iteration-one checkpoint and replay. It used 2x at
iteration 2, 3x at iteration 3, then added one reuse unit per iteration. The
standing teacher gate saved and paused the run after iteration **8**, not
iteration 7: iteration 7 passed at +0.04365, while iteration 8 rose +0.06012
above the +0.05 threshold.

The slower schedule delayed the first gate from iteration 3 to iteration 8 but
did not remove cumulative degradation. Frozen teacher objective rose from
0.75174 at iteration 1 to 0.97853 at iteration 8. Main replay validation was
best around iterations 2-3 and then worsened: total loss moved from 2.48112 at
iteration 3 to 2.72688 at iteration 8, while value loss moved from 0.98089 to
1.12254. The final teacher step was almost entirely a value-head event:
+0.00696 from weighted policy loss and +0.05316 from value loss.

This was not a canonical-seam pathology. Iteration 8's Q4-minus-Q1 contrast
delta was -0.00277, with paired interval -0.06168 to +0.05478, so neither seam
warning fired. Nor was it an oracle-ratchet event: the rolling 40-pair V4 score
was 31.25%, below the 50% watch band and 55% ratchet threshold.

The equal-96 comparisons resolve the teacher warning as a real standard-play
collapse:

- iteration 8 versus iteration 1, standard: 9-31 (22.5%), paired interval
  12.5%-32.5%, pairs 0/9/11; and
- iteration 8 versus iteration 1, placement-inclusive: 17-23 (42.5%), paired
  interval 27.5%-57.5%, pairs 4/9/7.

All downloaded artifacts match the paused contract. The iteration-eight
resumable checkpoint, replay, and inference hashes are respectively
`721d6c0df41a64b7f8e291b5bce29df9b9bcec7a35358789c3f96adaff3f50c9`,
`9e31b150ee99cd78a1caed8ded94d66a980f321ff0f548150fc001fc91e0a3ca`,
and `8c8617926a5efcdd177441dc741e1e2befd1199ee98aa0869999f8e844f63e8d`.
The checkpoint is technically resumable, but **must not continue the production
lineage**.

## Search-signal/value-semantics diagnostic

The local replay and arena diagnostic is complete. Exact generating-prior KL is
substantial for both retained windows, and P1c search96 beats its own raw legal
prior 38-2. The policy teacher is therefore not thin. Conversely, iteration 8
fits realized `z` much better while drifting sharply away from the cached
250K-node value proxy, and repeated replay positions expose high outcome-label
variance. Full methods, limitations, results, and the corrected multi-arm
design are in `P2_SIGNAL_DIAGNOSTIC.md`.

## Superseded proposed diagnostic

Restart again from the accepted iteration-one checkpoint and replay. Hold reuse
at **2x** instead of ramping, and initially run only iterations 2-4. Save a
resumable checkpoint and replay for every completed iteration. Then compare
iteration 4 with iteration 1 in the fixed equal-96 standard and
placement-inclusive arenas.

If iteration 4 remains healthy, run a second fixed-2x block through iteration 7
and inspect the same teacher, seam, validation, oracle, and strength signals
before considering either 3x reuse or a lower learning rate. Add a cumulative
teacher-review threshold near +0.10 from iteration 1: crossing it should save
state and require a paired strength check, but should not by itself reject a
checkpoint. The existing +0.05 one-step automatic pause remains in force.

This fixed-2x-at-3e-4 sequence is superseded as an expected fix. It remained
only the high-LR control in the completed multi-arm diagnostic. Arm D won the
diagnostic: at iteration 4 it scored 26-14 standard and 31-9
placement-inclusive against iteration 1 while moving the frozen objective only
+0.00654. The B-versus-D comparison at identical global 1e-4 LR isolates a
large benefit from the temporary P1c-value bridge.

Arm D subsequently completed its bridge through beta 1.0 at iteration 11.
Iteration 11 is the current P2 production head; iterations 4 and 1 remain the
rollback and longitudinal anchors. The first sustained pure-`z` block through
iteration 14 also completed cleanly, but did not earn promotion: its direct
standard arena against iteration 11 was 19-21, and its common-suite standard
score against iteration 1 fell from iteration 11's 29-11 to 20-20. Routine
continuation is paused pending larger fresh-suite confirmations against
iterations 11 and 1. See `P2_D_CONTINUATION.md` and
`P2_PURE_Z_12_14.md`. No diagnostic or continuation job touches final-test data
or final arena seeds.

Production runners use compact console logging because Kaggle's captured
subprocess output commits terminal carriage-return redraws as separate records.
Progress bars are disabled, while full JSONL and TensorBoard telemetry remain
unchanged.
