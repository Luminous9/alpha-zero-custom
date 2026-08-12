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
   objective only +0.0106. P2 now ramps reuse over absolute iterations 1-8 as
   2x, 4x, 6x, 8x, 10x, 12x, 14x, 16x.

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
Keep it as the fixed restart ancestor for the slower continuation below.
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

## Production restart from iteration 1

Upload `temp/santorini_v4_p2_iterations_2_20_warmup16_bundle.zip` (SHA-256
`93bff6de4217b5be3e7be2b680d2e0ca1bc169a4e4c9cdbf495a727fe95cede4`)
as a fresh Kaggle dataset and select a P100. Run:

```python
from pathlib import Path
import subprocess
import sys

matches = list(Path("/kaggle/input").rglob("run_santorini_v4_p2_production_kaggle.py"))
assert len(matches) == 1, matches
subprocess.run([sys.executable, str(matches[0])], check=True)
```

This package uses the exact accepted iteration-one checkpoint and replay, not
the paused iteration-three artifacts. Its input checkpoint/replay SHA-256 values
are respectively
`42b61409ec5ac6a3fd15d93ec6a700b87623e840e468fb8f80e857c1c8df1f78`
and `ffb947a7af216f1c77cc4a1369e407e97dbe2d6b2a51587e3df6b58d3f834f10`.
Iteration 1 remains the accepted 2x bootstrap transition. The restarted branch
uses 2x at iteration 2, 3x at iteration 3, then one additional reuse unit per
absolute iteration until reaching 16x at iteration 16. All standing controls
remain active and can save and pause the job early.

The runner uses compact console logging because Kaggle's captured subprocess
output commits terminal carriage-return redraws as separate records. Self-play,
oracle, optimizer, and milestone progress bars are therefore disabled. Each
iteration instead logs its phase plan/completion, optimizer schedule/final
losses, held-out validation, and one summary containing reuse, teacher step,
seam delta, sparring result, ratchet state, and wall time. Full telemetry remains
unchanged in `telemetry/telemetry.jsonl` and TensorBoard events.

Download the complete
`/kaggle/working/santorini_v4_p2/iterations_2_20/` directory. Only outputs from
this warmup-16 branch may continue the production lineage.
