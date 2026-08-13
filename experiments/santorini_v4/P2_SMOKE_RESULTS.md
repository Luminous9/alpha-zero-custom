# P2 P100 smoke results

## End-to-end throughput and replay

| Metric | Ordinary | Mixed 100k ladder v1 |
| --- | ---: | ---: |
| Games | 240 | 240 (216 ordinary + 24 sparring) |
| Total wall time | 354.03 s | 364.42 s |
| Self-play wall time | 344.33 s | 355.06 s |
| Games/hour, total wall | 2,440 | 2,371 |
| Fresh stored positions | 3,496 | 3,184 |
| Training positions | 3,371 | 3,070 |
| Optimizer steps | 211 | 192 |
| MCTS simulations | 530,816 | 485,632 |
| Inference batches | 11,681 | 17,538 |
| Requested/executed evaluations | 506,366 / 467,273 | 464,872 / 427,107 |
| Inference reuse | 7.72% | 8.12% |

The mixed arm costs 2.94% more total wall time and 3.12% more self-play time.
Its smaller sparring concurrency makes each inference batch much smaller, so it
issues 50.1% more batches despite requesting 8.2% fewer evaluations.

The 24 sparring games score 2-22 for V4. At the paired-opening level, two pairs
split 1-1 and ten score 0-2; no pair scores 2-0. The 90 stored sparring records
are 2.83% of the fresh replay and carry both neural seats, 12 opening hashes,
early/middle/late stage labels, and the correct engine/rung provenance. One
losing game legitimately stores no full-search neural decision, leaving 23
game keys in replay.

## Seam and transition behavior

The primary frozen seam contrast remains healthy:

| Arm | Q4-Q1 contrast delta | Paired bootstrap 95% | Warning |
| --- | ---: | ---: | --- |
| Ordinary | +0.0090 | -0.0586 to +0.0819 | false |
| Mixed | -0.0065 | -0.0806 to +0.0609 | false |

However, the overall frozen teacher objective rises by +0.3455 ordinary and
+0.3486 mixed. Equal-96 deterministic standard matches against the untouched
P1c handoff confirm a playing-strength shock:

| Candidate after one iteration | Score vs P1c | Paired bootstrap 95% | Pair 2-0 / 1-1 / 0-2 |
| --- | ---: | ---: | ---: |
| Ordinary, 16x reuse | 14-26 (35.0%) | 22.5%-47.5% | 1 / 12 / 7 |
| Mixed 100k, 16x reuse | 9-31 (22.5%) | 12.5%-32.5% | 0 / 9 / 11 |
| Exact ordinary replay, 2x reuse | 22-18 (55.0%) | 45.0%-65.0% | 3 / 16 / 1 |
| Mixed 5k ladder v2, 2x reuse | 19-21 (47.5%) | 37.5%-57.5% | 2 / 15 / 3 |

The 2x isolate uses the same P1c checkpoint, ordinary replay, AdamW state,
3e-4 learning rate, and held-out split. It changes only optimizer dose: 27
steps rather than 211. Its frozen objective delta is +0.01063 and seam-contrast
delta is +0.00157 (95% -0.01292 to +0.01627). This isolates first-window 16x
reuse as the ordinary-arm regression mechanism.

The resulting production control is a standing one-step gate, not a comparison
only to P1c: a frozen-objective increase greater than +0.05 versus the preceding
checkpoint saves all resumable artifacts and pauses the run. The preceding
objective is checkpointed. Separately, the live oracle rung ratchets when the
latest 40 complete paired openings (80 games) reach 55% V4 score; 50%-55% is a
watch band. Its rolling scores and ladder identity are checkpointed as well.

The corrected mixed transition passes all entry controls. It stores 3,205
positions (3,094 ordinary and 111 sparring), takes 25 optimizer steps at 2.072x
actual reuse, moves the teacher objective by +0.01414, and moves the seam
contrast by -0.00409 (95% -0.01721 to +0.00922). Its live 5k sparring score is
9-15 (37.5%; pairs 3/3/6), initializing 12 of the ratchet's 40-pair window.
It is promoted to production P2 iteration 1.
On the identical paired standard suite, it improves by +25.0 percentage points
over the failed 100k/16x mixed checkpoint (paired 95% +10.0 to +40.0; ten pairs
improve, nine tie, one worsens). Its -7.5-point difference from the ordinary 2x
isolate is unresolved (paired 95% -22.5 to +10.0), so there is no evidence that
the corrected 5k mixture itself creates a regression.

## Production continuation attempts

The accepted 2x transition remains the fixed iteration-one ancestor. Two
continuation schedules have since been tested and discarded.

| Attempt | Schedule after iteration 1 | Pause | Teacher evidence | Equal-96 standard vs iteration 1 | Decision |
| --- | --- | --- | --- | --- | --- |
| 1 | 4x at iteration 2; 6x at iteration 3 | iteration 3 | +0.02522, then +0.05651 | 19-21 (47.5%), 35%-60%, pairs 3/13/4 | discard iterations 2-3 conservatively |
| 2 | 2x at iteration 2; then +1x/iteration | iteration 8 | objective 0.75174 to 0.97853; final step +0.06012 | 9-31 (22.5%), 12.5%-32.5%, pairs 0/9/11 | confirmed standard collapse; discard iterations 2-8 |

Attempt 1 did not show a resolved strength regression: iteration 3 also scored
17-23 against P1c and 20-20 against iteration 1 in the placement-inclusive
arena. It was discarded to avoid carrying an uncertain checkpoint forward.

Attempt 2 provides the stronger diagnosis. Its per-iteration frozen-objective
steps for iterations 2-8 were +0.01184, +0.01532, +0.02602, +0.02257,
+0.04726, +0.04365, and +0.06012. Iteration 7 therefore passed; iteration 8
triggered the gate. The slower ramp delayed rather than eliminated the failure.
Replay validation total loss bottomed at 2.48112 on iteration 3 and rose to
2.72688 by iteration 8. The final +0.06012 teacher step decomposes into
+0.00696 weighted policy and +0.05316 value loss, pointing primarily to the
value head.

The independent sentinels correctly separate this regression from other
failure modes. Iteration 8's seam contrast delta is -0.00277 (paired interval
-0.06168 to +0.05478), so no seam warning is present. Its rolling 5k oracle
score is 31.25% over 40 pairs, below both the 50% watch band and 55% ratchet.
Placement-inclusive iteration-8 strength is only mildly lower at 17-23 (42.5%,
paired interval 27.5%-57.5%, pairs 4/9/7), while the standard result collapses
to 9-31. The one-step teacher gate therefore caught a real, predominantly
standard/value regression that neither the seam nor ladder controls should be
expected to catch.

The next proposed diagnostic restarts from iteration 1, fixes reuse at 2x, and
runs only iterations 2-4 with per-iteration resumable snapshots. Promotion
requires equal-96 standard and placement-inclusive comparisons of iteration 4
against iteration 1. If healthy, repeat fixed 2x through iteration 7 before
testing 3x or a lower learning rate. Add a cumulative +0.10 teacher-review
threshold from iteration 1 to request an early strength check while retaining
the existing +0.05 one-step automatic pause.

## Artifact integrity

The downloaded `.zip` suffixes are browser renames of native PyTorch/NPZ zip
containers, not outer archives. Their payload hashes match the Kaggle contracts.

| Arm | Resumable checkpoint | Compact replay | Inference checkpoint |
| --- | --- | --- | --- |
| Ordinary | `452955e401d404c90a08e5d36cee4f61f2c74a59122cd07dda30a96cc6c9bc4b` | `ac67674749c7facc55348f4f151e7593bfb7f09bcec675b6f8396825fc71337c` | `199dbabd250d6bed081c0cf5dc09556522b022e3ec3b26c12a1c94930fe7a338` |
| Mixed | `bdcab6937ac457677f04dda8041b8e554f00613e0a1a14908174a86ddf5024fe` | `d05ec359531411213092f18d2675e08207cc81b4fc9b8fa59b12f32dd1c9a781` | `78410d8cab0a382921d853099ea9c0ebe878a25766705fc1f60d0a536ab681bc` |
| Transition | `42b61409ec5ac6a3fd15d93ec6a700b87623e840e468fb8f80e857c1c8df1f78` | `ffb947a7af216f1c77cc4a1369e407e97dbe2d6b2a51587e3df6b58d3f834f10` | `d18f76903a7d17b1fdd35537cbaf424f62c688bb191e32ff75fe6da6c378b2c3` |
| Attempt 1, iteration 3 (discarded) | `5b8a3490df4d2593aeb6bc3ea087ac22deb3f191c7dbdb76c6562d03d979b197` | `c21c20f8b906947b85bf22a29b94ce4b0cb8f4d18d940e7d26c2b6c72df86e12` | `85bd067438636f646538c338bbb9f05d5f76691e2727a42c83eedb700682380e` |
| Attempt 2, iteration 8 (discarded) | `721d6c0df41a64b7f8e291b5bce29df9b9bcec7a35358789c3f96adaff3f50c9` | `9e31b150ee99cd78a1caed8ded94d66a980f321ff0f548150fc001fc91e0a3ca` | `8c8617926a5efcdd177441dc741e1e2befd1199ee98aa0869999f8e844f63e8d` |

Neither first-smoke checkpoint nor either discarded continuation is part of the
production P2 lineage. The corrected transition remains the sole accepted
iteration-one ancestor. Final-test data and final arena seeds remain untouched.
