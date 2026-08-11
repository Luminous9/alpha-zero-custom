# Santorini V4 Design: Bootstrapped D4-Equivariant Network

Status: P0a and P0b are complete, including the representative P100 Run13 wall
baseline: 348.45 seconds per amortized iteration, of which 92.69% is self-play.
The P1a 100k corpus/conversion correctness gate passes. A deterministic
20/35/45 stage and 70/20/10 main-line/subgame/Run13
sampler plus a fresh, separately labeled Run13 component now pass the mixed-pilot
gate. P1a is complete through the hand-rolled path: the 13-plane encoder, exact
D4 reference, optimized regular tower, invariant auxiliary head, checkpoint
reload, and frozen ordinary-Conv2d TorchScript export pass. The first P1b pilot
selects global score+winner provisionally and rejects the tested equivariant
candidates against the ordinary 8x96 control at that scale. Its original terminal
stop conclusion was too strong: only 5,848 unique training positions and one
ordinary shape were tested. P1b is reopened for stage/source-correct learning
curves at roughly 100k/300k/1M examples, larger ordinary candidates, and one
capacity-aware equivariant control. The 100k and 300k unique-data gates and
matched P100 screens now pass with exact declared marginals, no repetition, and
no cross-corpus D4 overlap. At 300k, ordinary 6x192 retains the best supervised
objective (0.8923), while capacity-aware equivariant E reaches 33.4% policy
top-1 versus 33.9% for the ordinary winner and slightly improves value MSE. The
loss gap is real but small enough that exact symmetry remains decision-relevant.
Bare ordinary inference is no longer selectable: an exact, stabilizer-safe D4
canonical wrapper now passes unit and real-checkpoint transformation audits
without changing predictions on canonical selection positions. P1b.2 continues
to the 1M curve with equivariant E, canonical ordinary 6x192, and canonical
ordinary 10x128. The frozen-holdout-anchored 1M screen is complete. Ordinary
6x192 and 10x128 are statistically tied on the paired selection objective;
equivariant E trails both by a significant margin and remains slower, while
retaining a small early-stage advantage. The predeclared seam audit, end-to-end
P100 benchmark, and 240-game paired round robin are complete. The direct
6x192-versus-10x128 arena remains inconclusive; the predeclared observed-batch
end-to-end speed tie-break selects canonical ordinary 6x192. E's native wrapper
is faster at batched inference, but it retains the supervised deficit and loses
the round robin overall; no seam-specific supervised interaction was detected.
P1b.2 now requires the frozen 1M winner-only target ablation on 6x192. P1c
remains gated on that result. Final test data and final arena seeds remain untouched. See
`experiments/santorini_v4/P1B_SUPERVISED_SCREEN.md` and
`experiments/santorini_v4/P1B_SCALED_SCREEN.md`.
Prereq reading: `santorini/santorini_ai_architecture_v3.md` (V3 spec),
`santorini/SANTORINI_ORACLE.md` (oracle bridge), and
`experiments/santorini_oracle/RESULTS.md` (Run13 distillation outcomes).

V4 is a fresh network and training run: a candidate-selected tower with enriched
input encoding, supervised-bootstrapped from `santorini-ai` engine data, then
trained with the AlphaZero loop using the oracle as a value labeler and
distribution shaper from iteration 1. Exact search-facing D4 behavior is
required. It may be supplied by the equivariant architecture or by
stabilizer-safe canonical inference around an ordinary winner; augmentation or
sampled root averaging alone is insufficient. Run13 is retired to
benchmark anchor and placement teacher; it is not the starting checkpoint.

## 1. Why a fresh network instead of continuing Run 13

The Run13 oracle experiments (RESULTS.md) established that offline policy imitation from the minimax teacher does not work on a converged checkpoint: hard distillation failed to reproduce across budgets, soft distillation was neutral, and the adversarial fine-tunes were neutral at direct play and regressed at deep MCTS budgets (1-19 at 1024 sims vs 6-14 for Run13). The plausible mechanism is policy/value inconsistency: an alpha-beta engine has no policy distribution, and shifting the policy head toward invented soft targets while the value head describes old behavior is amplified - not washed out - by deeper search.

A corrected Run13 continuation plan exists (distribution shaping via sparring and seeded starts, plus an auxiliary oracle-value head). It is sound but no longer the best use of constrained GPU, for three reasons:

1. **The search-facing risky lever was deferred anyway.** Blending oracle values into the search-facing value head during AlphaZero training was already scoped to a _later_ run, informed by auxiliary-head calibration. Sparring and seeded starts are lower-risk, independently configurable distribution-shaping components that can be introduced in the fresh run from day 1. The auxiliary head still shares the trunk and is therefore not risk-free; it receives its own ablation and weight schedule.
2. **Detection power.** Continuation experiments produce small effects that 40-160-game arenas cannot resolve (the most interesting Run13 result was p=0.09). Bootstrap pretraining is cheaper to inspect because corpus generation is CPU-only and supervised pretraining has no MCTS in the loop. Staged corpus pilots and paired arenas can reject a broken transfer pipeline before committing a full self-play run.
3. **The architecture hypothesis.** V3 is ~1.35M parameters with a minimal 6-plane encoding, playing against an opponent (santorini-ai NNUE + alpha-beta) that Run13 beats ~10% of the time at only 20k nodes. It is not yet proven that parameter capacity, rather than data/search/optimization, is the active ceiling. V4 tests richer features and increased useful capacity per inference cost; exact D4 equivariance is one candidate inductive bias, not the definition of the upgrade. The ordinary-CNN controls and supervised sizing bake-off in §3.2 prevent either capacity or equivariance from being assumed effective.

Condition retained from that analysis: a fresh start is only justified _because_ it is paired with the architecture upgrade. Restarting the same network would re-buy Run13 at full price.

What Run13 remains for: benchmark/anchor opponent, placement-policy teacher (§5.3), and the source of the position distribution used in calibration (§4.2).

## 2. Goals and non-goals

Goals:

- Clearly exceed Run13 in direct paired arenas at 96/128 sims, without the deep-budget (1024-sim) regression pattern.
- Materially close the gap against the santorini-ai oracle at 20k nodes.
- D4 neural policy/value symmetry by construction up to documented floating-point tolerance. An equivariant winner supplies it architecturally. An ordinary winner uses deterministic D4-canonical inference at every neural evaluation, projects its policy over the canonical representative's stabilizer, and maps it back; augmentation remains a training aid, while root orientation averaging is disabled as redundant. Finite MCTS is audited separately: action-index tie breaking and uncoupled stochastic root noise can break whole-search equivariance even with an exact network, so stabilizer projection/coupled-noise handling gates the final self-play integration.
- Oracle in the loop from iteration 1 in its two trustworthy roles: scalar value labeler (auxiliary head first) and distribution shaper (sparring, seeded starts). After the explicitly declared bootstrap exception in §5.2, policy targets remain 100% native search targets produced by V4's configured MCTS/Gumbel search.

Non-goals:

- God powers (mortal-vs-mortal only, matching the oracle bridge).
- Placement supervision from the oracle - unsupported joint-vs-sequential placement boundary (`SANTORINI_ORACLE.md`, "Known boundary").
- Replicating NNUE input feature crosses. Those exist because NNUE has one hidden layer and cannot compose conjunctions; a deep conv tower can.

## 3. Network architecture candidates

### 3.1 D4-equivariant tower (G-CNN, regular representation)

The tower is a group-equivariant CNN over D4 (order 8):

- **Lift layer:** input planes → 8 orientation copies with tied weights.
- **Group conv blocks:** residual blocks convolving over (space x group), all weights tied across the 8 group elements. Group-aware BatchNorm (statistics shared across group copies).
- **Value head:** pool over the group dimension before the FC stack -> exactly invariant value. Same for the auxiliary oracle-value head (§3.4).
- **Policy head:** equivariant. Its output fiber is defined explicitly as the 64-dimensional permutation representation for the move/build direction pairs plus one trivial placement representation: `rho_policy = rho_move_build + rho_placement`. The spatial origin transforms with the board while both direction indices transform together. The existing V3 policy permutation tables are the executable reference used to test this representation. If implemented by inverse-permuting orientation copies and combining them, the result must be tested against those tables to ensure the operation did not accidentally produce an invariant policy.

Explicitly rejected shortcut; constraining kernels to be D4-symmetric (center/edge/corner parameterization). That yields equivariance but destroys direction sensitivity, and the policy is fundamentally directional. The full regular representation is required.

Implementation options, in preference order: (a) a pinned `escnn` version; (b) hand-rolled weight tying (small on a 5x5 board; a few hundred lines). The `escnn` prototype must prove checkpoint reload compatibility and export the inference model to ordinary PyTorch layers before throughput measurement, avoiding `GeometricTensor` and filter-expansion overhead in MCTS.

Ordinary-tower alternative: D4 canonicalization at inference. Canonicalization
is exact only when positions with a non-trivial stabilizer (especially the empty
board) average over every transform that reaches the selected representative
before the policy is mapped back. Arbitrary tie-breaking is not equivariant.
The implemented wrapper uses the corpus converter's representative and performs
that projection, so this is now a first-class architecture-system candidate
rather than an emergency fallback.

**Equivariance test (required before any training):** for random inputs and all 8 group elements, transformed input must produce the expected permuted policy and numerically equal value under a declared absolute/relative tolerance. Test the input encoder, trunk, policy representation, value/aux pooling, checkpoint reload, and exported inference model independently. This test gates everything downstream.

What architectural equivariance deletes from V3 (code + compute):

- Root orientation averaging in MCTS (currently 2-8 evals per root) - a direct self-play speedup that partially pays for the larger net.
- The root symmetry-refresh machinery and its interaction bugs (see commits d6dc09b, 3208ad9).
- The symmetry-consistency loss and paired-orientation batch fraction.
- D4 augmentation as a _necessity_. Training data is stored in one D4-canonical orientation and exact orbit duplicates are removed. Equivariance gives orbit-wide generalization, but naturally generated data should not be expected to shrink by 8x unless all eight copies were actually present.

### 3.2 Sizing: decided by supervised bake-off, not a priori

Channel counts below are _effective_ channels; with the regular
representation, multiplicity = effective/8 (e.g. 128 effective = 16 group
multiplicities), and parameter count is ~1/8 of the equivalent untied conv.

Candidates (identical corpus, schedule, and selection holdout):

| Candidate   | Blocks | Effective ch | ~FLOPs vs V3 | Rationale                                                                                  |
| ----------- | ------ | ------------ | ------------ | ------------------------------------------------------------------------------------------ |
| A           | 8      | 96           | 1.0x         | Smallest equivariant candidate                                                             |
| B           | 10     | 128          | ~2.2x        | Default next notch                                                                         |
| C           | 6      | 192          | ~2.5x        | Wide-shallow: tests whether width can use otherwise idle GPU capacity once receptive field is global |
| D           | 12     | 160          | ~4.2x        | Upper probe; retain only if the strength/cost curve has not flattened                       |

Ordinary-CNN candidates use the same selected input planes, bootstrap corpus,
targets, and schedule. The 8x96 V3 shape remains the baseline; 10x128 and 6x192
test the previously unmeasured capacity axis. They retain D4 augmentation during
training. A surviving ordinary checkpoint is deployed only through exact
stabilizer-safe canonical inference, with root-orientation sampling set to one.
A small 6-plane versus 13-plane comparison
on the baseline separately tests the feature additions.

Candidate C remains the equivariant continuity control. Add one roughly
learned-parameter-matched equivariant candidate (initial probe: 6 blocks, 320
effective channels / multiplicity 40). This does not replace the mathematically
valid group projection with unconstrained concatenation. If a richer head is
tested, it must remain equivariant and pass the full §3.1 transformation/export
suite. Report learned parameters, exported parameters, FLOPs, and measured cost;
no single matching notion is allowed to stand in for all four.

Selection protocol: (1) matched GPU learning curves at approximately 100k, 300k,
and 1M training examples, with supervised selection-holdout policy/value loss and
explicit unique/repetition statistics; (2) equal-simulation round-robin arenas
between surviving candidates and a diagnostic preview versus Run13; (3) measured
exported-model self-play throughput at FP32/FP16 and target batch size; (4) Gate
G1 versus Run13 for the selected scaled checkpoint; (5) an untouched final test
split and arena opening/seed set used only after selection. Report both
equal-simulation strength and strength per approximately equal wall-clock or
neural-evaluation budget. Pick the knee of strength per inference cost.

### 3.3 Input encoding

V3's 6 binary planes (my workers, opp workers, height 1/2/3, dome) are complete but minimal. V4 keeps them and adds exact, cheaply computed tactical structure (the KataGo lesson; don't make the net spend depth deriving what the rules engine computes for free). Side-to-move canonicalization is retained.

| #   | Plane(s)        | Definition                                                                                                                                         |
| --- | --------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1-6 | V3 planes       | unchanged                                                                                                                                          |
| 7   | Scalar height   | height/4 as one real-valued plane (restores ordinality that one-hots hide)                                                                         |
| 8   | My win threats  | squares where side to move wins next turn (own worker on h2 adjacent to an h3 square, square reachable/legal) - reuses `prepareTacticalRoot` logic |
| 9   | Opp win threats | same for the opponent on their next turn                                                                                                           |
| 10  | My mobility     | per-worker legal-move count, normalized, stamped on worker squares                                                                                 |
| 11  | Opp mobility    | same for opponent                                                                                                                                  |
| 12  | Climb access    | count of adjacent unoccupied/non-domed squares reachable by a hypothetical current-player worker from each non-dome square under the normal climb rule, divided by 8; origin occupancy is ignored, destination occupancy is respected |
| 13  | Phase           | broadcast scalar derived from board state only: workers placed / 4 during placement, then `min(sum(heights), 40) / 40` during standard play     |

All planes must be well-defined during the placement phase. Threat, mobility, and climb-derived planes are zero until all four workers exist so partially placed workers are not interpreted as standard-play states. Opponent threats are computed by an explicit opponent-perspective legality check. Every plane must be D4-covariant and tested against brute-force rule calculations as well as the end-to-end equivariance test.

### 3.4 Heads

- **Policy:** per-square 65 local-action channels - 1625 actions, unchanged
  action space (placement = local channel 64, direction-free, transforms
  spatially only).
- **Value:** group-pooled → FC → tanh, unchanged semantics.
- **Auxiliary oracle-value head (new):** small parallel invariant head off the trunk, trained only on oracle-labeled positions (§6.3). It is not consumed directly by search. It can still change the search-facing heads through shared-trunk gradients, so it receives an off-switch, a low initial weight, and a short ablation. Its purposes are representation shaping and measurement of how well the network can generalize the calibrated oracle signal. Agreement with the oracle measures learnability, not whether the oracle label itself is true.

## 4. Phase 0 - prerequisites and measurement

### 4.1 Deterministic oracle-query contract

P0a implemented reset-before-analyze in `ParallelOraclePool`, matching the
budget benchmark's independent-query behavior, plus a persistent label cache
keyed by `(D4-canonical FEN, node budget, engine binary/version digest,
calibration version)`. Cache writes are atomic and records contain score,
mate-band status, depth, actual nodes, and the mapped value. The cache cuts
repeated work and makes labels reproducible.

The santorini-ai `datagen` crate has the same issue at corpus scale: it keeps one transposition table per worker and its per-game reset is currently commented out. The V4 corpus mode must reset before each independent game (and before any separately adjudicated calibration rollout). The corpus header records this reset policy. No score generated with warm cross-game TT history is accepted into V4 training data.

### 4.2 Score-value calibration

`sigmoid(score/400)` is the engine's nominal convention, not established calibration on our distribution. Protocol:

- Sample positions from the Run13 replay distribution, stratified by phase.
- Compute the candidate label score at each budget under test, with an independent TT reset.
- Ground truth = **deeper oracle-vs-oracle adjudication outcomes** from each position, using a materially larger budget than the candidate label and a separate reset. _Not_ replay z: replay z is the outcome under weaker Run13 continuation. This still measures the deeper engine's game outcomes rather than perfect-play truth, so conclusions are stated in those terms.
- Split positions by D4-canonical key into fit and untouched test sets before fitting. Fit temperature T in `sigmoid(score/T)` on the fit set; report Brier score and calibration error on the test set by phase, score magnitude, and label budget. Clamp mate-band scores (|score| near 10,000) to ±1 separately; they are not on the sigmoid scale.
- Do not reuse the label search as the first move of its adjudication rollout. Record the adjudicator budget and engine digest with every result.

P0b result: global temperatures fitted to 300 deeper continuations were
T=261.8 at 20k and T=239.8 at 50k, and both improved held-out calibration over
the nominal T=400 mapping. Early-stage calibration remained materially worse
than middle/late calibration and exhibited a phase-dependent bias that one
global temperature cannot remove. The P1 pilot therefore compares the simple
global mapping against stage-aware calibration/downweighting before freezing
the score-plus-winner blend. See `experiments/santorini_v4/P0B_RESULTS.md`.

### 4.3 Throughput benchmark (moved here from §8 - no oracle dependency)

The benchmark has two stages. The Run13 baseline is measured in Phase 0; candidate measurements run immediately after the minimal V4/ordinary-control prototypes exist and before full corpus generation or pretraining commits to a winner.

- Measure the Run13 baseline first: separately instrument self-play, training, arena/telemetry, and serialization wall clock. Existing iteration totals are insufficient if they cannot recover this split. This quantifies how much headroom the "self-play GPU is idle-heavy, width is cheap" argument has.
- Measure exported-model NN evals/sec at real self-play batch shapes for the ordinary control and candidates A-D: FP32 versus FP16/autocast on P100. Only if P100+FP16 is insufficient, test T4 FP16 and decide whether dual-T4 process-parallel self-play is worth building.
- FP16 strength-neutrality check: policy/value agreement vs FP32 on a few thousand replay positions + one small paired arena.
- End-to-end smoke iterations remain the deciding throughput measurement: tiny 5x5 convolutions can be launch/CPU-bound, and theoretical FP16 FLOPs or removed root ensembles are not accepted as wall-clock estimates.

### 4.4 Score stability by stage and budget

Extend the existing budget benchmark to report score-sign agreement and score margins by phase (not only best-move agreement - the move flips in near-equal positions while the score barely moves; for a _value_ label, score stability is the relevant metric). Output: the cheapest node budget per phase that is stable enough for labeling. Prior report suggests ~86-87% sign agreement at modest budgets.

P0b result under the declared 90% sign-agreement and 100-point median-delta
thresholds: 20k passes for late positions, 100k passes for middle positions,
and no tested sub-250k budget passes for early positions. The 250k reference
passes against itself by construction and is not proven converged. The pilot
retains stage as label metadata and reports coverage/cost separately by stage.

## 5. Phase 1 - bootstrap corpus and supervised pretraining

### 5.1 Corpus

The existing datagen text format is insufficient for V4 policy bootstrap: it writes FEN, winner, score, ply, depth, and nodes, but not the searched best action or successor. Add a versioned Mortal-vs-Mortal V4 corpus mode before bulk generation.

Required per-record fields:

- schema version, engine binary/version digest, game id, and record id;
- FEN and side to move;
- searched best action path and/or best successor FEN;
- final game winner;
- root score, mate-band flag, completed depth, requested nodes, and actual nodes;
- game ply/build count and whether the record belongs to a main line or randomized subgame;
- TT policy (`reset-per-game`) and generation configuration in a file header/manifest.

Generation requirements:

- Explicitly select Mortal-vs-Mortal; default all-god datagen output is rejected.
- Reset the per-worker TT before every independent game. The currently commented-out reset is restored in V4 corpus mode.
- Make the number of random moves before recording configurable and include a meaningful zero/one-move fraction; current datagen always skips the first 2-6 standard moves.
- Run the rules/FEN/action differential validator on a sample from every generated shard. Verify that the recorded best successor maps to one or more legal V4 actions, including all winning-move aliases.
- Partition train/selection-holdout/final-test by D4-canonical position and game id before training so symmetric copies and same-game trajectories cannot leak across splits.

First generate a 100k-500k pilot. Validate schema round trips, score/winner perspective, action mapping, stage distribution, duplicate rate, and training-loader throughput. Only then scale to a 5M corpus; expand toward 20M only if the supervised learning curve has not flattened.

Conversion uses `SantoriniOracle.py` FEN mapping and stores one D4-canonical orientation. Exact orbit duplicates retain observation counts and aggregated target statistics rather than silently discarding frequency information. Datagen is still the engine's self-play distribution, so mix in separately oracle-labeled Run13-replay positions, with an explicit source field and controlled sampling weight, to hedge transfer risk.

### 5.2 Targets

- **Default bootstrap value:** a calibrated interpolation of engine evaluation and recorded engine-continuation winner, `v_boot = (1-alpha_boot) * z_engine + alpha_boot * v_score`, with `0 < alpha_boot < 1`, where `v_score = 2*sigmoid(score/T)-1` and mate-band scores are clamped. The P1 pilot compares the P0b global mapping against an explicitly recorded stage-aware calibration/downweighting rule because early-stage P0b calibration remained biased. Choose and freeze the mapping and `alpha_boot` using only calibration-fit/pilot-selection data before the reserved evaluation. `z_engine` is converted from the absolute recorded winner to the FEN side-to-move perspective. This is an explicitly declared bootstrap exception to V4's no-oracle-blend rule during AlphaZero training: the teacher policy, evaluation, and continuation outcome are coherent, and evaluation/result interpolation matches how the teacher's own NNUE was trained.
- **Required value ablation:** pretrain an otherwise identical winner-only variant (`alpha_boot=0`). If winner-only matches the blended model within the predeclared selection margin on the selection holdout and reserved selection arenas, prefer winner-only because it removes the target-semantics transition when the main value head switches to self-play z. If the blend wins clearly, record `alpha_boot`, T, and the transition explicitly in the checkpoint metadata. The final test split and final arena seeds remain untouched until the design is locked.
- **Policy:** the recorded engine best move is a legally smoothed hard target, weighted below the value loss. Put `1-epsilon` on the best V4-equivalent action set and distribute `epsilon` only over other legal actions; never smooth over all 1,625 logits. A winning no-build move divides best-move mass over its equivalent V4 aliases. Hard policy imitation is acceptable as an initialization with coherent teacher value/outcome supervision; this does not contradict rejecting post-hoc surgery on Run13.
- **Small pilot ablations:** on the pilot corpus, compare 6 versus 13 input planes, ordinary versus equivariant candidate A, and winner-only versus score+winner. These are screening experiments, not claims of final strength. The first P1b run confirmed that this limitation is substantive; architecture and target choices remain provisional until the scaled curves.

### 5.3 Placement (the oracle cannot teach it)

Distill placement from Run13: run the Run13 net with full evaluation-strength search over placement-phase states sampled from its replay and fresh self-play openings. Use its native policy/visit distribution as the policy target and completed-game outcome as the preferred main-value target; keep the Run13 search value only as optional auxiliary telemetry. Same-player sequential placement transitions retain the existing turn-aware value semantics.

Placement examples use a phase-balanced sampler/loss bucket so millions of standard-play records cannot swamp the four placement decisions. The target fraction is declared from the expected V4 self-play phase mix and reported for every pretraining epoch.

### 5.4 Gate G1 - bootstrap arena

Bake-off winner versus Run13 at 96 and 128 simulations, with two separate gates:

1. **Standard-play gate:** fixed, untouched completed openings with paired seats. This isolates engine-data transfer and standard play.
2. **Full-game gate:** empty-board games with reproducible placement/search randomness. This measures placement distillation plus standard play.

Report equal-simulation results and an approximately equal wall-clock/neural-evaluation comparison. Each opening/seat pair is one statistical block.

- **≥ ~35% score with an interval excluding the stop region on the standard gate:** green light. It does not need to win after zero self-play iterations.
- **< ~20%, or a material discrepancy between standard and full-game gates:** stop and debug distribution transfer, placement, encoding, calibration, or action mapping before spending loop GPU. If it cannot be closed, the Run13 continuation plan is the documented fallback.

## 6. Phase 2 - self-play loop with the oracle in-loop from day 1

Base config inherits Run13 (latest mode, Gumbel search, playout-cap randomization, 20-iteration replay window, AdamW schedule) with these changes.

### 6.1 Search and throughput

- No root orientation averaging, no symmetry refresh, no consistency loss (§3.1).
- Before self-play, transformed-root tests cover ordinary/equivariant neural outputs and complete deterministic searches. The returned root policy is projected over non-trivial stabilizers, and stochastic root noise is generated in canonical action coordinates, so finite search does not reintroduce orientation dependence after the network removes it.
- Exported-model FP16 inference for self-play. P100 peak FP16 arithmetic is 2x FP32, but no end-to-end speedup is assumed until measured; if Kaggle offers T4x2, benchmark it before building process-level parallelism.
- Self-play batch raised until the §4.3 benchmark identifies the end-to-end throughput knee.
- Candidate reduction of full sims to 48-64: test whether Gumbel target quality and playing strength hold at the smaller budget given the stronger bootstrap; adopt the reduction only if the benchmark supports it.

### 6.2 Distribution shaping

- **Sparring (~10% of games):** start from completed placements because the oracle cannot participate in V4's sequential placement phase. Play `SantoriniOraclePlayer` at a node budget laddered to keep the net's win rate ~35-50%, with paired seats/openings. Store only the net's own decisions: native MCTS policy + real game outcome. Raise the ladder when the net clears the current rung. Never spar at a budget that produces ~90% losses, where value labels collapse toward -1.
- **Disagreement-seeded starts (~10% of games):** mine high-margin, score-stable disagreements with the existing adversarial tooling and use them as starting positions only (no teacher labels). Seed pools are stage/value stratified, D4-deduplicated, versioned by source checkpoint, and replay-capped. Run13 disagreements may initialize the pool, but refresh it periodically from the current V4 checkpoint so training follows V4's blind spots rather than Run13's stale ones.
- Both components have independent configuration switches and telemetry. Run preplanned short matched branches with and without distribution shaping at an early checkpoint, rather than waiting for an ambiguous main-run trajectory to make the control decision.

### 6.3 Value labeling (auxiliary head only, this run)

- Label a subsample (25-50%) of stored post-placement positions via the §4.1
  cache at the §4.4 budget.
- Start the auxiliary loss weight at ~0.05-0.10. Ramp toward 0.2-0.3 only if main policy/value validation and strength telemetry remain healthy. Run a short matched auxiliary-off ablation because shared-trunk gradients can affect search even though the auxiliary output is not consumed by MCTS.
- After the supervised-bootstrap handoff, the main value head trains on self-play z only (`lambda_blend = 0`) for this run. If the blended bootstrap won §5.2, record the resulting target-semantics transition explicitly.
- **Storage:** keep z, v_oracle, node budget, engine/calibration version, source type, and an oracle-valid mask as separate replay fields. Blending is a training-time knob; changing lambda must never require regenerating data.
- Telemetry: auxiliary prediction versus oracle label, oracle label/main-head/z versus the deeper adjudication suite, main-loss changes with auxiliary gradients, sparring win rate by ladder rung, and seeded-start replay fraction/source age. Auxiliary agreement alone measures learnability, not oracle truth.

### 6.4 Deferred to a following run

Folding oracle value into the search-facing target,
`(1-lambda)*z + lambda*v_oracle`, with lambda chosen from the oracle label's measured accuracy against deeper adjudication plus a direct playing-strength ablation - not from auxiliary-head agreement and not by guess. Apart from the explicitly declared coherent-teacher bootstrap exception in §5.2, this search-facing blend stays out of V4's first AlphaZero run.

## 7. Evaluation and promotion

- Rules cross-validation (`validate_santorini_oracle.py`) must pass before any cross-engine result is trusted (existing policy, unchanged).
- Primary strength: paired arenas versus Run13 and versus the oracle at 96 and 128 simulations. Report equal-simulation results and approximately equal wall-clock/neural-evaluation results where architectures differ materially in inference cost.
- Stress test: 1024 sims - retained specifically because it caught the distillation collapse. A V4 candidate that wins at 128 and collapses at 1024 is rejected.
- Statistics: predeclare the effect and stop/accept regions. Use sequential testing over independent paired-opening blocks or fixed arenas sized for the effect sought. Report paired/block-bootstrap intervals (and ordinary Wilson intervals only as secondary per-game summaries), because the two seat assignments of one opening are correlated. The Run13 experiments showed that unplanned 40-160-game arenas cannot resolve the effects that matter.

## 8. Compute plan

The throughput benchmark itself is a Phase 0 deliverable (§4.3). This section records the reasoning and working expectations it will replace with measurements.

Working hypothesis: a wider net may be cheaper than FLOP counts suggest because self-play uses small batches of 5x5 inputs and can be CPU/launch-bound between GPU calls. Existing GPU traces suggest this, but the newly instrumented §4.3 wall-clock split is the authority. Extra per-eval compute may fill idle capacity or may simply add latency; the plan assumes neither outcome.

FP16 usage: exported self-play inference wrapped in `torch.autocast('cuda', dtype=torch.float16)`. P100 peak FP16 arithmetic is ~2x FP32, but tiny-convolution end-to-end speedup and strength neutrality are measured rather than inferred. Mixed-precision _training_ (autocast + `GradScaler`) is optional and low-priority. Dual-T4 is not a prerequisite: build process-parallel self-play (one worker per GPU, merged examples, single-GPU training) only if §4.3 shows T4x2+FP16 beating P100+FP16 by enough to repay implementation complexity.

Working expectation to be replaced by measurement: candidate B is ~2.2x V3 dense-convolution FLOPs. Removing root orientation averaging saves only the extra root evaluations, not the single-orientation interior simulations, so it is not counted as a 2-8x whole-loop speedup. FP16, exported inference, batching, and any simulation reduction are credited only after end-to-end smoke iterations. Bootstrap still has the separate compute benefit of skipping the weakest random-initialization region if Gate G1 passes.

## 9. Risks and fallbacks

| Risk                                     | Signal                                                                    | Response                                                                                                                                 |
| ---------------------------------------- | ------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| Corpus is invalid                         | action/FEN round trips fail, cross-game score drift, wrong god distribution | Stop before bulk generation; fix versioned Mortal-only schema, reset policy, and conversion tests                                       |
| Bootstrap transfers poorly               | G1 < 20% vs Run13                                                         | Debug encoding/calibration/action mapping and standard/full-game split; mix more Run13-distribution positions; fallback = Run13 continuation |
| G-CNN implementation stalls              | equivariance/export tests unstable, BN issues                             | pinned escnn ↔ hand-rolled swap; compare the first-class ordinary + stabilizer-safe canonical-inference candidate                        |
| Kaggle throughput worse than expected    | benchmark >1.5x per-iteration cost                                        | Drop to candidate A/C width, cut full sims (Gumbel), reduce sparring fraction                                                            |
| Oracle labeling too expensive            | labeler falls behind self-play                                            | Lower subsample rate; rely on cache hits; per-phase budgets from §4.4                                                                    |
| Placement regression                     | full-game gate/milestones decline while standard gate improves            | Increase phase-balanced Run13 placement corpus/weight and diagnose each sequential placement; seeded standard starts cannot repair placement |
| Bootstrap target transition hurts        | winner-only matches blend or blended handoff regresses                    | Prefer winner-only; otherwise reduce `alpha_boot` or add a short winner-only transition before self-play                                 |
| Auxiliary gradients hurt main heads      | auxiliary-on validation/arena branch regresses                            | Keep auxiliary weight low or disable it; stored labels and telemetry remain useful                                                       |
| Value-semantics mismatch shows up later  | future main-head blend regresses at deep search                           | `lambda` stays 0; auxiliary head retained only if its shared-trunk ablation is healthy                                                    |

## 10. Milestones

1. **P0a - deterministic data contract:** pool reset fix + versioned label cache; Mortal-only V4 datagen schema with best action/successor; reset-per-game TT policy; schema/action/FEN tests.
2. **P0b - measurement:** deeper-adjudicator calibration study; score-stability-by-phase report; rules cross-validation; instrumented Run13 wall-clock split.
3. **P1a - pilot and architecture feasibility:** 100k-500k corpus pilot + conversion pipeline; V4 encoder rule/covariance tests; pinned-escnn and hand-rolled feasibility spike; exported-model equivariance/checkpoint tests.
4. **P1b.1 - small screening ablations (complete, conclusion corrected):** the source-aware trainer, ordinary 6/13-plane baseline, per-epoch selection, Candidate A-D sizing, matched stage/global/winner targets, and paired standard/full selection arenas are complete. Global blend is provisional and winner-only fails at this scale. Candidate C loses the matched full-game pilot arena 10-30 to ordinary 8x96. This rejects the tested candidate at this scale, not V4.
5. **P1b.2 - scaled architecture screen (active; architecture selected, target ablation pending):** generate stage/source-correct supply; record stratum coverage and repetition; run matched 100k/300k/1M learning curves. At 1M, ordinary 6x192 and 10x128 score 0.8115 and 0.8138; their paired difference interval crosses zero. Equivariant E scores 0.8436, with an E-minus-6x192 paired interval of 0.0201-0.0439, and trains 28.7% slower. E retains slightly better early policy CE and winner MSE. Ordinary survivors use exact D4 canonical inference, which matches the frozen canonical holdout predictions and passes all eight transformed-position checks exactly on the audited checkpoint. A pre-arena seam audit finds no detectable high-versus-low frame-switch exposure interaction: E-minus-6x192 is +0.03257 in the lowest quartile and +0.03256 in the highest, with a high-minus-low interval of -0.03424 to +0.03448. The matched P100 wrapper benchmark shows that canonical preprocessing dominates raw network cost; at batch eight FP16, E/O6/O10 achieve 1,369/1,294/1,207 examples/s. The arenas execute mean batches around 6-10. In the combined standard/full seed-cluster bootstrap, O6 scores 45.0% against O10 (95% 35.0-53.75%), O6 scores 53.75% against E (45.0-63.75%), and O10 scores 67.5% against E (60.0-73.75%). The direct ordinary matchup is inconclusive, so the frozen end-to-end speed tie-break selects ordinary 6x192. The 1M plan contains exactly 1,000,000 distinct corpus positions with zero Run13 overlap. Run the required winner-only versus global-blend ablation on 6x192 next. Final test data and final arena seeds remain untouched.
6. **P1c - full corpus and pretraining (gated):** after P1b.2 selects an architecture, generate 5M valid records, expanding only if the learning curve justifies it; full pretraining includes phase-balanced Run13 placement distillation.
7. **Gate G1:** separate standard-play and full-game arenas versus Run13 at 96/128 simulations, plus equal-cost reporting. Stop/debug thresholds use paired-block uncertainty.
8. **P2 - self-play:** sparring + refreshed seeded starts + low-weight auxiliary head, each independently switchable; replay/telemetry schema extended.
9. **Review:** after ~20-30 iterations, run the 96/128/1024 battery and component ablations; decide whether to study a nonzero search-facing `lambda` in a following run.
