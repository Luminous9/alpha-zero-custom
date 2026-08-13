# P2 search-signal and value-semantics diagnostic

This diagnostic was run after the second discarded continuation and before any
third P2 attempt. It tests whether the 96/32 search contract still supplies a
useful policy-improvement signal to the strong P1c checkpoint, and whether the
pure-outcome value handoff is a plausible cause of the observed regression.
No final-test positions or final arena seeds were used.

## Replay scope and exactness

The failed branch's compact replay contains eight windows with 25,505 stored
full-search decisions. The archive does not store a generating-checkpoint ID
per window, and intermediate checkpoints 2-7 were not retained. Consequently:

- window 1 versus P1c is an exact generating-prior comparison;
- window 2 versus accepted iteration 1 is exact; and
- later comparisons against P1c, iteration 1, or iteration 8 are explicitly
  fixed-reference retrospectives, not reconstructed online KL curves.

`diagnose_santorini_v4_replay_signal.py` computes the checkpoint prior in the
production canonical D4 frame, masks illegal moves, renormalizes over legal
moves, and reports `KL(search_target || legal_prior)`, total variation, top-move
agreement, value residuals against `z`, and optional cached deep-oracle
comparisons. Cached oracle labels are joined through companion corpus NPZs so
only split 0/1 labels are eligible; split 2 and unclassified labels are
excluded by construction.

## The 96-search policy target is not thin

| Exact window | Positions | Mean / median KL | Mean TV | Prior/search top-1 agreement |
| --- | ---: | ---: | ---: | ---: |
| P1c generating window 1 | 3,205 | 0.599 / 0.420 | 0.387 | 52.9% |
| Iteration-1 generating window 2 | 3,101 | 0.506 / 0.276 | 0.336 | 53.1% |

The signal remains substantial in standard play. For P1c window 1, mean KL is
0.488 early, 0.586 middle, and 0.725 late. Placement is lower but still
nontrivial at 0.440. Iteration-1 window 2 follows the same pattern: 0.447 early,
0.562 middle, 0.665 late, and 0.242 placement.

KL establishes teachable distribution movement, not playing improvement, so a
paired arena measures the actual 0-to-96 operator. On 20 fixed standard
selection openings with seats swapped, deterministic legal-prior argmax scored
**2-38** against the same P1c checkpoint at deterministic 96-simulation Gumbel
search. Raw prior won no pair, lost 18, and split two; its game score was 5.0%
with a paired bootstrap interval of 0-12.5%. The raw player uses one root
evaluation, Gumbel scale zero, one symmetry, and no tactical shortcuts. The
searched player uses the deployed deterministic evaluation contract.

Therefore 96 simulations retain a very strong policy-improvement signal. The
failed P2 continuation is not explained by a near-zero-KL or ineffective
policy teacher, and increasing self-play search solely to create a teaching
signal is not presently justified.

## Value labels show a real semantic/variance hazard

Across all 25,505 archived observations, fitting the realized outcome improves
markedly along the failed branch: value MSE to `z` falls from 0.916 for P1c and
0.895 for iteration 1 to 0.631 for iteration 8; sign accuracy rises from
59.2%/59.9% to 74.5%. This improvement is not evidence that iteration 8 learned
a better value function.

There are 106 unique replay positions (490 observations) with an allowed
250K-node cached oracle label. Most are split-0 positions previously present in
the bootstrap corpus; only two unique positions are split 1. This makes the
comparison a semantic-drift diagnostic rather than a clean generalization
benchmark. On unique positions:

| Checkpoint | MSE to 250K oracle proxy | MAE to oracle proxy | Correlation |
| --- | ---: | ---: | ---: |
| P1c | 0.0112 | 0.0778 | 0.619 |
| Accepted iteration 1 | 0.0155 | 0.0921 | 0.550 |
| Failed iteration 8 | 0.2043 | 0.3736 | 0.141 |
| Realized `z` | 1.0210 | 1.0016 | -0.015 |

The replay itself independently exposes high label variance. Among all 415
standard positions repeated across different games, 65.8% have conflicting
`z` outcomes. Their unique-position-weighted within-position `z` variance is
0.610 (0.772 when observation weighted). Among the 51 repeated oracle-matched
positions, 62.7% conflict and the corresponding variances are 0.556/0.817.
These are in-archive repeat estimates, not unbiased irreducible-noise estimates,
but they demonstrate that a single game outcome is a high-variance target for
many states.

Taken together with iteration 8's frozen-objective decomposition (weighted
policy +0.00696, value +0.05316), the evidence strongly supports the proposed
mechanism: the inherited from-scratch learning rate acts on a converged network
while the main value head abruptly changes from calibrated bootstrap targets to
noisy single-game outcomes.

## Consequences for the next experiment

Do not run the previously proposed fixed-2x, 3e-4 continuation as the expected
production fix. Retain it only as the high-LR control. A four-arm closed-loop
diagnostic from the accepted iteration-1 checkpoint remains sensible:

1. **A — control:** 2x fixed reuse, global LR 3e-4, pure `z`.
2. **B — LR:** 2x fixed reuse, global LR 1e-4, pure `z`.
3. **C — differential heads:** shared trunk LR 1e-4, policy-head LR 3e-4,
   value-head LR 3e-5, pure `z`.
4. **D — LR plus value bridge:** global LR 1e-4 and
   `v_target=(1-beta)*v_P1c+beta*z`, beginning at beta 0.25 and annealing toward
   1 over approximately ten iterations.

The trunk qualification in C is essential. Policy and value share the entire
stem/residual tower, so “policy 3e-4 / value 3e-5” is not well-defined for the
shared parameters. Merely lowering the value-head parameter-group LR leaves
full-strength value gradients updating the trunk and cannot by itself confirm
a value-specific mechanism. C is best interpreted relative to B: it asks
whether extra policy-head plasticity helps while both shared representation and
value head remain protected.

A versus B isolates global step size; B versus D isolates the target bridge.
If strict factorial attribution is more important than testing differential
heads, add or substitute a fifth arm with 3e-4 plus the same value bridge. Save
every iteration. Run the first arena at iteration 4 against iteration 1; a
healthy four-iteration result is necessary but does not prove long-run safety,
because the prior failure accumulated through iteration 8. Continue promising
arms only under the existing per-step and cumulative frozen-teacher controls.

The value bridge is an explicit P2 transition exception, not a permanent change
of value semantics. Its checkpoint digest, beta schedule, current beta, and
anchor predictions must be reproducible and telemetered. Promotion still
requires paired playing strength; lower training loss against blended targets
is not a gate.

The shared one-arm-per-session Kaggle package and launch instructions are in
`P2_FINE_TUNING_ARMS.md`.

## Artifacts

- `temp/santorini_v4_p2_signal_diagnostic/replay-signal.json`
  (`84495815dfabf2878a5ba18e1a1b6b0162525d807b018032a48b200c87a204af`)
- `temp/santorini_v4_p2_signal_diagnostic/raw-vs-search96.json`
  (`09aebffd62fcf2d01c7c9846e5792666763f4318173d73300ca6d99fe128a834`)
