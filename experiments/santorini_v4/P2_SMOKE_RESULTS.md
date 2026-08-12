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

## Artifact integrity

The downloaded `.zip` suffixes are browser renames of native PyTorch/NPZ zip
containers, not outer archives. Their payload hashes match the Kaggle contracts.

| Arm | Resumable checkpoint | Compact replay | Inference checkpoint |
| --- | --- | --- | --- |
| Ordinary | `452955e401d404c90a08e5d36cee4f61f2c74a59122cd07dda30a96cc6c9bc4b` | `ac67674749c7facc55348f4f151e7593bfb7f09bcec675b6f8396825fc71337c` | `199dbabd250d6bed081c0cf5dc09556522b022e3ec3b26c12a1c94930fe7a338` |
| Mixed | `bdcab6937ac457677f04dda8041b8e554f00613e0a1a14908174a86ddf5024fe` | `d05ec359531411213092f18d2675e08207cc81b4fc9b8fa59b12f32dd1c9a781` | `78410d8cab0a382921d853099ea9c0ebe878a25766705fc1f60d0a536ab681bc` |

Neither first-smoke checkpoint is part of the production P2 lineage. Final-test
data and final arena seeds remain untouched.
