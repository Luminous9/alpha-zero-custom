# P2 4x dose branch: iterations 12-17

## Result

The branch, its local milestone evaluation, and the duplicate-aware follow-up
are complete. The training run was mechanically and diagnostically healthy.
The original 69.2% placement-inclusive result was real for the sampled learned
opening distribution, but its magnitude was badly inflated by repeated exact
and D4-equivalent openings. A fresh duplicate-aware decomposition and a frozen
60-opening D4-unique follow-up change the mechanistic conclusion: iteration
17's placement policy alone is neutral, while its standard-phase controller is
better on iteration-17-style openings. The diversity-controlled point estimate
is 55.8%, not the legacy 69-83%, and its interval still includes parity.

Use **iteration 17 as the provisional whole-agent checkpoint when playing
complete games from the empty board**. Keep **iteration 11 as the accepted
general-standard anchor and rollback checkpoint**. The evidence now favors
iteration 17 for the actual full-game objective, but does not establish broad
standard-play superiority or justify escalating to 6x/8x reuse.

### Training health

- All six requested iterations completed from the exact iteration-11 ancestor.
- Actual reuse was 4.00-4.06x, giving 47-48 optimizer steps per iteration.
- The run took 3,123.7 seconds (52.1 minutes) on a P100.
- Frozen teacher-objective step changes stayed between approximately 0 and
  `+0.0050`; the cumulative delta ended at `+0.0373`, far below the `+0.10`
  review gate.
- Every seam contrast delta was negative and no seam warning fired.
- At iteration 17, frozen deep-value Pearson was `+0.0185` relative to
  iteration 11 and MSE was `-0.0052`. The windows-9-11 slice was also healthy:
  Pearson `+0.0101`, MSE `-0.0021`. No deep-value warning fired.
- Standard prior-to-search-target KL stayed between 0.513 and 0.560 and ended
  at 0.560. The 96/32 search operator is not running out of policy signal.
- The block scored 63/144 (43.75%) in 5k live sparring and ended with a 46.25%
  rolling score. The ladder did not ratchet.

Relative to the separate 2x iterations-12-14 branch, 4x was at least as healthy
on the frozen value suite and generally better by iteration 14. More optimizer
dose is therefore not intrinsically destabilizing at the selected `1e-4` LR.

### Legacy iteration-17 versus iteration-11 arena

The local arena used 120 games per mode, 60 seat-paired openings/seeds, a new
seed (`20260816`), deterministic 96-simulation Gumbel evaluation, canonical-D4
FP32 inference, and no reserved final-test data.

| Mode | Iteration-17 score | Pair wins / splits / losses | Paired bootstrap 95% |
|---|---:|---:|---:|
| Fixed completed placements | 53-67 (44.2%) | 10 / 33 / 17 | 35.8%-52.5% |
| Placement-inclusive from the empty board | 83-37 (69.2%) | 28 / 27 / 5 | 60.8%-76.7% |

The standard-play direction independently resembles the earlier 2x
iteration-14 result (43.3% against iteration 11 on seed `20260815`). It is not
statistically resolved in this single iteration-17 suite, but it is not evidence
of standard-play progress. Conversely, the complete-game improvement is large
and its paired interval excludes parity.

The placement arena was seat-balanced (61 player-one wins and 59 player-two
wins), so the result is not a first-player artifact. However, its 43 exact and
22 D4-unique completed placements across 120 games were subsequently shown to
be far too concentrated for a promotion claim. Paired seeds and contestant
seat reversal control seat bias; they do not turn repeated openings into
independent evidence. Retain 69.2% only as the raw policy-weighted score of that
legacy suite.

### Expanded 50k external anchor

Iteration 11 and iteration 17 were also evaluated against the 50k oracle on the
same expanded set of 60 openings (120 games each), retaining the established
`20260921` suite as a prefix and adding 40 new openings.

| Checkpoint | V4 score | Pair 2-0 / 1-1 / 0-2 | Paired bootstrap 95% |
|---|---:|---:|---:|
| Iteration 11 | 58-62 (48.3%) | 8 / 42 / 10 | 41.7%-55.0% |
| Iteration 17 | 54-66 (45.0%) | 8 / 38 / 14 | 37.5%-52.5% |

On identical pairs, iteration 17 minus iteration 11 is **-3.3 points**, with a
95% interval of **-12.5 to +6.7** and 11 improved / 32 unchanged / 17 worsened
pairs. Even with 60 openings this instrument remains compressed, but it agrees
with the internal standard arena: no transferable standard-play gain is
demonstrated, and any regression is modest rather than catastrophic.

### Duplicate-aware placement-inclusive contract

Placement-inclusive neural arenas now use two passes:

1. Generate all requested stochastic placement occurrences and preserve their
   multiplicity for the raw policy-weighted score.
2. Hash the completed board plus physical-side-to-contestant mapping. Execute
   each exact deterministic continuation once, then reuse that result for
   repeated occurrences. Standard-play Gumbel scale must be zero for this
   reuse contract.

Every report includes exact and D4 opening counts, maximum multiplicity and
share, opening-frequency ESS, exact/full trajectory hashes, a raw score, and a
D4-capped score. The cap is normalized: no D4 family contributes more than 5%
of final score weight when at least 20 families exist. Bootstrap resampling
keeps the two seat assignments in one pair. A promotion requires raw and
capped directions to agree and resolve, plus natural D4 ESS of at least 75% of
the pair count and no naturally sampled D4 family above 5%.

This gives the same rule at every arena size. The raw view may contain any
number of naturally preferred repeats, but they are declared rather than
mistaken for independent games. The controlled view permits at most 2, 5, or 6
occurrences of one D4 family in a 40-, 100-, or 120-game suite respectively
before down-weighting. No artificial move noise is added after placement.

### Fresh duplicate-aware phase decomposition

The new suite used 120 games per matchup, seed `20260819`, deterministic
96-simulation standard play, paired contestant seats, and no reserved final
data. Score is for the first-named candidate in each matchup.

| Matchup | Raw policy score (95%) | 5%-capped D4 score (95%) | Natural D4 ESS / required | Unique exact continuations |
|---|---:|---:|---:|---:|
| iteration 17 vs iteration 11 | 83.3% (75.0-90.8) | 63.9% (47.9-80.0) | 4.48 / 45 | 71 / 120 |
| hybrid vs iteration 11 | 50.8% (45.8-55.8) | 49.6% (38.5-61.1) | 4.20 / 45 | 69 / 120 |
| hybrid vs iteration 17 | 14.2% (8.3-20.8) | 36.7% (21.0-51.6) | 4.41 / 45 | 67 / 120 |

The dominant D4 family occupied 42.5-45.8% of each natural sample. All three
natural-diversity gates therefore fail. The raw intervals are useful
distributional descriptions, not promotion intervals. Equal-weight D4-family
scores point in the same mechanistic direction but remain unresolved with only
21-22 families: 63.8% for iteration 17 versus iteration 11, 47.7% for the
hybrid versus iteration 11, and 37.0% for the hybrid versus iteration 17.

The execution contract removed 49, 51, and 53 redundant deterministic
continuation executions respectively. It did not discard their occurrence
weight from the raw estimator and did not perturb standard moves to force
different trajectories. Candidate 2-0 / split / 0-2 pair counts were 45/10/5,
5/51/4, and 2/13/45. The occurrence records contained 23, 23, and 27 repeated
full trajectories with maximum multiplicity 3, 4, and 4; these were counted in
the raw distributional view but were not physically replayed.

### Frozen 60-opening learned-policy robustness suite

The follow-up sampled placement with iteration 17 controlling both sides, then
retained the first representative of each D4 family. Sixty families were found
by occurrence 759; batched collection finished the active block at 840
occurrences, observing 63 D4 and 171 exact openings in total. The source
distribution itself remained concentrated: its largest D4 family occupied
33.6% of occurrences and natural D4 ESS was 5.62.

The selected suite contains exactly 60 D4-unique completed placements. From
each frozen board, iteration 17 and iteration 11 played both contestant seat
assignments at deterministic 96-simulation search. All 120 standard-game
trajectories were distinct, and physical-side results were balanced at 61-59.

| Iteration-17 score | Pair 2-0 / 1-1 / 0-2 | Paired bootstrap 95% |
|---:|---:|---:|
| 67-53 (55.8%) | 19 / 29 / 12 | 46.7%-65.0% |

This is the cleanest estimate of iteration 17's continuation strength in its
learned-opening domain. It favors iteration 17 by 5.8 points but does not
resolve parity. Together with the two independent raw whole-agent wins, it is
enough to make iteration 17 the provisional full-game choice; it is not enough
to claim general standard-play superiority. The progression from 83.3% raw to
63.9% D4-capped to 55.8% D4-unique quantifies how much the earlier headline
margin depended on opening frequency.

Artifact:

- `temp/santorini_v4_p2_unique_learned_openings/iteration17-vs-iteration11-60-d4-unique-seed20260821.json`

### Phase-switched hybrid

The proposed hybrid routes every placement-phase network evaluation to
iteration 17 and every standard-phase evaluation to iteration 11. The switch
occurs at each MCTS leaf, not only between played moves, so the last placement
search uses iteration 11 for descendants that have crossed into standard play.
Direct inference checks against the real checkpoints were numerically exact.

It was first evaluated twice with 120 games per matchup at 96 simulations. Seed
`20260817` deliberately replays the placement seeds used by the original
iteration-17-versus-iteration-11 arena, making it a matched-suite
decomposition. Seed `20260818` is a genuinely fresh replication.

| Suite | Opponent | Hybrid score | Pair wins / splits / losses | Paired bootstrap 95% |
|---|---|---:|---:|---:|
| Matched (`20260817`) | Iteration 11 | 57-63 (47.5%) | 3 / 51 / 6 | 42.5%-52.5% |
| Fresh (`20260818`) | Iteration 11 | 58-62 (48.3%) | 6 / 46 / 8 | 42.5%-54.2% |
| Matched (`20260817`) | Iteration 17 | 37-83 (30.8%) | 5 / 27 / 28 | 22.5%-39.2% |
| Fresh (`20260818`) | Iteration 17 | 38-82 (31.7%) | 4 / 30 / 26 | 24.2%-39.2% |

These legacy runs established that merely substituting iteration 17 during
placement does not preserve its raw full-game score, but they did not record
enough duplicate structure to attribute the gap safely. The fresh
duplicate-aware run above is the controlling result.

The legacy iteration-11 matchups were highly compressed by physical seat
(95-25 and 96-24 player-one records), producing 51 and 46 split pairs. That
makes them poor instruments for detecting a tiny placement effect, but it
cannot hide a large portable placement gain: neither point estimate resembles
the original iteration-17 full-agent score of 69.2%. The hybrid losses to
iteration 17 replicated within one point, but their raw magnitude was also
opening-concentration sensitive.

Artifacts:

- `temp/santorini_v4_p2_hybrid_eval/hybrid-vs-iterations11-and17-seed20260817.json`
- `temp/santorini_v4_p2_hybrid_eval/hybrid-vs-iterations11-and17-seed20260818.json`
- `temp/santorini_v4_p2_duplicate_aware_eval/iteration17-hybrid-seed20260819.json`

This was a closed, phase-specific diagnostic. Its result artifacts and analysis
are retained here, but the temporary hybrid wrapper and hard-coded iteration
11/17 runner were removed before commit because checkpoint routing was rejected
as a production direction.

### Revised interpretation and next gate

The combined evidence now rejects four simple stories:

1. 4x did not damage the value head or exhaust the search teacher; all frozen
   diagnostics remained healthy.
2. 4x did not demonstrate general post-placement progress on the existing
   fixed-opening or 50k-anchor instruments.
3. Its raw full-game gain cannot be attributed to an independently stronger
   placement policy. The iteration-17-placement/iteration-11-standard hybrid
   is at parity with iteration 11 under both raw and capped views.
4. Iteration 17's standard controller is the source of the advantage on the
   concentrated iteration-17-preferred opening distribution: replacing it
   with iteration 11 collapses the score in the reciprocal hybrid control.

This makes checkpoint routing an unsuitable production answer. It also removes
the earlier evidence for a special iteration-17 placement head: placement by
itself did not improve the hybrid. The remaining disagreement between the
existing fixed-opening/50k instruments and the learned-opening arena is now a
distribution question, not demonstrated phase incompatibility.

That diversity-controlled diagnostic is now complete. Its 55.8% point estimate
supports iteration 17 as the provisional full-game choice, while the
46.7-65.0% interval explains why iteration 11 remains the general-standard
anchor and rollback. If a formally resolved head-to-head promotion is needed,
predeclare and run a second fresh 60-D4-family suite; do not add games
selectively to the completed `20260821` result.

Preserve both checkpoints. Raising reuse to 6x/8x is not justified by the
current strength evidence; whether to continue another block at 4x from
iteration 17 is a separate training decision.

## Replay-prefix diversity audit and corrective design

The follow-up replay audit found that placement concentration is learned drift,
not an immutable property of the P1c handoff. On a matched 240-seed empty-board
self-play sample, the P1c checkpoint produced 80 D4 placement families with
opening-frequency ESS 29.21 and a 9.58% largest-family share. Iteration 1 had
47 families/ESS 13.75/12.92%; iteration 17 had 34/5.48/34.17%. Production
telemetry shows most of the collapse occurred during the early 2x bridge:
iteration 1's largest family was 6.67%, iteration 4's was 25.42%, and later
iterations fluctuated around 20-37% rather than worsening monotonically during
the 4x block.

Standard replay is not broadly duplicated. Across the 17 stored windows,
33,461 of 37,494 standard positions (89.2%) were D4-unique; iteration 17 alone
was 96.6% unique. No two of the 3,668 reconstructable ordinary episodes shared
the same retained board/value sequence, although compact replay does not store
complete action trajectories. The pathology is instead the mandatory
placement prefix: iteration 17 stored 864 placement positions (27.7% of its
self-play replay) even though placement is only four plies per game. The empty
root occurs once per empty-board game. Its 216 iteration-17 targets had mean
Jensen-Shannon divergence only 0.0111 from their average, so that repetition is
mostly redundant policy supervision plus high-variance outcome `z`.

The implemented correction is non-destructive and has two separate parts:

1. Raw replay files remain unchanged. Before optimization, placement states are
   grouped within each replay window and placement ply by their anonymous D4
   canonical state. Each group becomes one canonical example whose search
   policy and outcome target are arithmetic means. Optimizer draw mass is 85%
   standard and 15% placement, with exactly 3.75% assigned to each of the four
   placement plies. Within a window/ply stratum, group weight is
   `sqrt(occurrence_count)` instead of raw frequency. Count weighting was
   gradient-checked against the repeated-example objective; square-root
   weighting is therefore the one intentional change in state frequency.
2. Ten percent of ordinary games start after a placement generated by the
   current neural search and accepted only if it is D4-unique within that
   iteration's fixed-start quota. These candidates use the normal search,
   temperature, and root noise; no extra move noise is added. Rejected prefixes
   never enter replay. This breadth mixture is complementary to replay
   balancing and does not change placement Gumbel scale.

Telemetry now reports raw placement share, aggregate-group counts, per-ply draw
mass, sampling ESS, maximum group multiplicity, unique-start acceptance rate,
and D4 rejections. A generation failure is explicit after 32 candidates per
requested start rather than silently falling back to duplicates.

Future placement-inclusive comparisons use the checkpoint-generic
`arena_santorini_v4_placement.py`. Its `natural` mode retains the sampled
placement distribution while reporting duplicate-aware and D4-capped views;
its `unique-learned` mode builds a frozen D4-unique learned-opening suite.

The next matched diagnostic starts both arms from iteration 17 for four
iterations at the existing 4x dose and `1e-4` LR. The control sets both new
fractions to zero. The balanced arm uses 15% placement/square-root weighting
plus 10% unique neural starts. Keep all existing safety gates and compare the
two endpoints against iteration 17 and on the duplicate-aware placement suite.
Do not treat the new replay design as promoted production policy until this
short branch confirms safety and at least neutral strength.

Evaluation artifacts:

- `temp/santorini_v4_p2_dose4_eval/iteration17-vs-iteration11-seed20260816.json`
- `temp/santorini_v4_p2_iter11_oracle_50k_120/oracle-sweep-summary.json`
- `temp/santorini_v4_p2_iter17_oracle_50k_120/oracle-sweep-summary.json`
- `temp/santorini_v4_p2_unique_learned_openings/iteration17-vs-iteration11-60-d4-unique-seed20260821.json`

## Historical dose decision

Start from the accepted iteration-11 training checkpoint and its 11-window
replay. Run exactly six new iterations, 12 through 17, at fixed 4x fresh-data
reuse. Do not automatically escalate to 6x or 8x at the end of this job.

This is a single-variable dose experiment. It keeps the production architecture,
240 games per iteration, global AdamW learning rate `1e-4`, pure outcome-`z`
value target, 96/32 playout-cap search, and 10% 5k-node ladder-v2 oracle
sparring unchanged. The ancestor used 2x reuse; iteration 12 is the declared
2x-to-4x transition.

The branch deliberately disables the old endpoint arenas. Those use a small,
fixed suite that is no longer an adequate promotion instrument. After iteration
17, evaluate strength separately on expanded fresh suites, including the 20k,
50k, and 100k external anchors. Use at least 40 and preferably 60 paired
openings for the stronger oracle rungs, and preserve a fixed longitudinal suite
alongside rotated fresh neural self-match suites.

## In-loop controls

All existing controls remain active:

- frozen teacher objective: pause above a `+0.05` one-iteration increase or a
  `+0.10` cumulative increase from the iteration-1 reference;
- oracle ladder ratchet: pause when its rolling score reaches 55%;
- prior-to-search-target KL: warn after three eligible standard-play iterations
  below `0.15` (this does not pause); and
- canonical seam telemetry every iteration.

The run also adds the frozen A1 deep-value audit as standing telemetry. Its 480
standard-play positions are D4-unique, stratified by replay-window band and game
stage, and exclude overlaps with the held-out final corpus. Labels are the
already-cached cold-TT 250k-node oracle values. Every new checkpoint is compared
on exactly those positions with the frozen iteration-11 predictions.

One iteration warns if any of these paired point changes from iteration 11 is
reached:

- overall Pearson: `-0.02` or worse;
- overall MSE: `+0.015` or worse;
- windows 9-11 Pearson: `-0.04` or worse; or
- windows 9-11 MSE: `+0.02` or worse.

Two consecutive warning iterations pause after the new checkpoint, replay, and
telemetry have been written. The warning streak and suite fingerprint are saved
inside the training checkpoint, so resuming does not reset the guard. The
telemetry also records early/middle/late and all four replay-window slices plus
stratified paired bootstrap intervals. The engine values remain proxy labels,
so promotion should use the paired drift and playing-strength suites together,
not treat the absolute engine MSE as ground truth.

## Kaggle artifacts

- Notebook: `santorini/v4_p2_training_kaggle.ipynb`
- Runtime upload: `temp/santorini_v4_p2_runtime_bundle.zip`
- Resume checkpoint: `checkpoint_11-training.pth.zip`
- Resume replay: `checkpoint_11.examples.zip`

The notebook is now reusable. Its checked-in parameters describe the four-step
balanced arm from iteration 17; setting both replay/opening fractions to zero
runs the matched control. Optional expected-start/end checks prevent accidental
auto-resume during this diagnostic. Its final cell packages the complete output
directory into one ZIP. Kaggle extracts attached datasets automatically; the
notebook does not try to unpack the uploaded runtime or resume files.

Runtime bundle SHA-256:

`8bbf974f51f28e4e83ca2110c936ab6971dd1aab1235e3c6b5ded381a32a75c7`

The runtime includes the frozen deep-value suite with SHA-256:

`0323a03302862522928568a1076cdeac52e66d3f357ea01700cba10c305c1af2`

The telemetry implementation was checked against the actual iteration-11
checkpoint. The reproduced overall and windows-9-11 Pearson/MSE deltas were all
below `5e-7` in magnitude, as expected for the frozen reference.
