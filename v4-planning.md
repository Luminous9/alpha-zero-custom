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
was faster before canonical-path optimization, but it retains the supervised
deficit and loses the round robin overall; no seam-specific supervised
interaction was detected. Batched canonicalization plus on-device policy-frame
restoration subsequently raises 6x192 batch-eight P100 FP32 throughput from
1,388 to 3,285 examples/s. Against the identical uncanonicalized wrapper, the
remaining exact-D4 cost is 24% latency rather than the original dominant
overhead, so this implementation blocker is closed.
The frozen 1M winner-only ablation is also complete. Winner-only trails global
blend by +0.01031 on the common handoff objective, with a paired 95% interval of
+0.00169 to +0.01868, and therefore misses the +0.01 noninferiority margin. The
fixed arenas split 24-16 for global blend in standard play and 16-24 in full
games, producing an exact 40-40 combined score (95% 42.5-57.5%). Retain global
blend with `alpha_boot=0.5` and `T=261.8`; the main head switches explicitly to
pure self-play `z` at handoff. P1c and G1 are complete, and the P2 entry
implementation is now locally green. The selected P1c checkpoint migrates to a
trainable V4 checkpoint with bit-identical policy/value inference; complete
deterministic transformed searches pass on asymmetric, symmetric, and tactical
roots. Compact replay now preserves per-record source metadata while remaining
backward-compatible. The representative 240-game P100 FP32 ordinary and
10%-sparring smoke arms completed with valid replay/checkpoint provenance and
only a 2.9% mixed-arm wall-time premium. They also exposed two entry problems
before the production lineage began: the deterministic ladder-v1 proxy did not
match live 96/32 stochastic search, and 16x replay reuse overtrained the first
small replay window. Exact live-policy calibration now selects 5k nodes as
ladder version 2 (17-23 over 40 confirmation games), while an isolated 2x
first-iteration retrain preserves P1c strength at 22-18 and holds the frozen
objective shift to +0.0106. The corrected 5k transition passes: its teacher
objective step is +0.01414, seam contrast delta is -0.00409, and its equal-96
standard check against P1c is 19-21. It remains the only accepted production
checkpoint at iteration 1. Two continuation ramps are now discarded: 4x/6x
paused at iteration 3, and a slower +1x-per-iteration ramp paused at iteration
8 after standard strength collapsed 9-31 against iteration 1. The next proposed
diagnostic restarts from iteration 1 at fixed 2x for iterations 2-4, with paired
strength checks before any extension. Disagreement starts and the auxiliary
head remain disabled. Final test data and final arena seeds remain untouched. See
`experiments/santorini_v4/P1B_SUPERVISED_SCREEN.md` and
`experiments/santorini_v4/P1B_SCALED_SCREEN.md`, followed by
`experiments/santorini_v4/P1C_FULL_PRETRAIN.md`.
Prereq reading: `santorini/santorini_ai_architecture_v3.md` (V3 spec),
`santorini/SANTORINI_ORACLE.md` (oracle bridge), and
`experiments/santorini_oracle/RESULTS.md` (Run13 distillation outcomes).

V4 is a fresh network and training run: a candidate-selected tower with enriched
input encoding, supervised-bootstrapped from `santorini-ai` engine data, then
trained with the AlphaZero loop using the oracle as a value labeler and
distribution shaper from iteration 1. Exact search-facing D4 behavior is
required. It may be supplied by the equivariant architecture or by
stabilizer-safe canonical inference around an ordinary winner; augmentation or
sampled root averaging alone is insufficient. G1 establishes that Run13 is
clearly behind V4, so it is retired to an optional, manually invoked fixed
anchor plus its historical placement-teacher/control roles; it is neither the
starting checkpoint nor a P2 gate opponent.

## 1. Why a fresh network instead of continuing Run 13

The Run13 oracle experiments (RESULTS.md) established that offline policy imitation from the minimax teacher does not work on a converged checkpoint: hard distillation failed to reproduce across budgets, soft distillation was neutral, and the adversarial fine-tunes were neutral at direct play and regressed at deep MCTS budgets (1-19 at 1024 sims vs 6-14 for Run13). The plausible mechanism is policy/value inconsistency: an alpha-beta engine has no policy distribution, and shifting the policy head toward invented soft targets while the value head describes old behavior is amplified - not washed out - by deeper search.

A corrected Run13 continuation plan exists (distribution shaping via sparring and seeded starts, plus an auxiliary oracle-value head). It is sound but no longer the best use of constrained GPU, for three reasons:

1. **The search-facing risky lever was deferred anyway.** Blending oracle values into the search-facing value head during AlphaZero training was already scoped to a _later_ run, informed by auxiliary-head calibration. Calibrated sparring is the lower-risk distribution-shaping component enabled from P2's first block. V4 disagreement starts follow their loader/pool gate, and the auxiliary head follows checkpoint/replay migration plus label-semantic validation. Each remains independently configurable and receives its own ablation.
2. **Detection power.** Continuation experiments produce small effects that 40-160-game arenas cannot resolve (the most interesting Run13 result was p=0.09). Bootstrap pretraining is cheaper to inspect because corpus generation is CPU-only and supervised pretraining has no MCTS in the loop. Staged corpus pilots and paired arenas can reject a broken transfer pipeline before committing a full self-play run.
3. **The architecture hypothesis.** V3 is ~1.35M parameters with a minimal 6-plane encoding, playing against an opponent (santorini-ai NNUE + alpha-beta) that Run13 beats ~10% of the time at only 20k nodes. It is not yet proven that parameter capacity, rather than data/search/optimization, is the active ceiling. V4 tests richer features and increased useful capacity per inference cost; exact D4 equivariance is one candidate inductive bias, not the definition of the upgrade. The ordinary-CNN controls and supervised sizing bake-off in §3.2 prevent either capacity or equivariance from being assumed effective.

Condition retained from that analysis: a fresh start is only justified _because_ it is paired with the architecture upgrade. Restarting the same network would re-buy Run13 at full price.

What Run13 remains for: an optional fixed anchor that can be run manually when
an absolute historical comparison is useful, placement continuation outcomes
and policy control (§5.3), and the source of the position distribution used in
calibration (§4.2). It is disabled in routine P2 telemetry and cannot accept,
reject, roll back, or promote a P2 checkpoint.

## 2. Goals and non-goals

Goals:

- Preserve the already-established G1 advantage over Run13; use milestone
  self-matches and oracle/deep-search reviews, rather than repeated Run13 gates,
  to measure further P2 progress.
- Materially close the gap against the santorini-ai oracle at 20k nodes.
- D4 neural policy/value symmetry by construction up to documented floating-point tolerance. An equivariant winner supplies it architecturally. An ordinary winner uses deterministic D4-canonical inference at every neural evaluation, projects its policy over the canonical representative's stabilizer, and maps it back; augmentation remains a training aid, while root orientation averaging is disabled as redundant. Finite MCTS is audited separately: action-index tie breaking and uncoupled stochastic root noise can break whole-search equivariance even with an exact network, so stabilizer projection/coupled-noise handling gates the final self-play integration.
- Oracle in the loop from iteration 1 as a calibrated sparring opponent. The
  auxiliary scalar labeler and disagreement-seeded starts join only after their
  independent §6 integration gates; they are not allowed to delay or alter the
  ordinary-self-play control. After the explicitly declared bootstrap exception
  in §5.2, policy targets remain 100% native targets produced by V4's configured
  MCTS/Gumbel search.

Non-goals:

- God powers, including placement (mortal-vs-mortal only, matching the oracle
  bridge). Mortal placement is supported by the explicit joint-to-sequential
  factorization in §5.3.
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

### 5.3 Placement (joint oracle, sequential network)

Santorini-ai searches a complete unordered worker pair at each placement turn,
whereas V4 emits two same-player placement actions. This is representational,
not a loss of information. Query all 300 unordered pairs from the empty board
and all 253 pairs from each of the 49 D4-unique post-P1-pair boundaries. Convert
the joint distribution `q({a,b})` exactly into
`P(first=a)=0.5*sum_b q({a,b})` and
`P(second=b|first=a)=q({a,b})/sum_c q({a,c})`. The factorization makes the
arbitrary order unobservable while producing targets for all 1/6/49/904
sequential prefix orbits.

Each root pair is searched independently with a reset TT. Duplicate action
orders are collapsed by resulting FEN. Because finite search produced different
scores for D4-equivalent pairs, raw scores are projected by averaging over the
parent position's D4 stabilizer before softmax. The frozen one-time teacher uses
50,000 nodes per root pair. Raw responses are retained so temperature changes
do not require more search. A placement-only continuation tournament found the
initial T300 target too diffuse and selected T25. Oracle placement values are
telemetry only.

Run13 supplies the placement value target: start continuations from all 960
prefix orbits, use its native search policy, and record the completed-game
outcome. This is retained because the oracle's static placement evaluation is
not a game outcome, not because Run13 placement was shown to be superior.

Select the policy teacher before full GPU pretraining with a paired
placement-only tournament: sample each teacher through all four sequential
actions, swap teacher seats, then use the same deterministic santorini-ai
continuation for both sides with TT reset per opening. At 64 paired blocks,
oracle T25 matched Run13 (46.875%, paired 95% interval 39.0625%-54.6875%) and
matched the 50/50 T25/Run13 blend (46.875%, 38.2813%-55.4688%). The blend's
small point-estimate lead was inconclusive. Apply the predeclared preference for
pure oracle when it matches the mixed arm: P1c uses the pure T25 oracle policy
with the shared Run13 completed outcomes and frozen sampling plan. Same-player
sequential transitions keep the existing turn-aware value semantics.

Placement examples use a phase-balanced sampler/loss bucket so millions of
standard-play records cannot swamp the four placement decisions. The target
fraction is declared from the expected V4 self-play phase mix and reported for
every pretraining epoch.

### 5.4 Gate G1 - bootstrap arena

Bake-off winner versus Run13 at 96 and 128 simulations, with two separate gates:

1. **Standard-play gate:** fixed, untouched completed openings with paired seats. This isolates engine-data transfer and standard play.
2. **Full-game gate:** empty-board games with reproducible placement/search randomness. This measures placement distillation plus standard play.

Report equal-simulation results and an approximately equal wall-clock/neural-evaluation comparison. Each opening/seat pair is one statistical block.

- **≥ ~35% score with an interval excluding the stop region on the standard gate:** green light. It does not need to win after zero self-play iterations.
- **< ~20%, or a material discrepancy between standard and full-game gates:** stop and debug distribution transfer, placement, encoding, calibration, or action mapping before spending loop GPU. If it cannot be closed, the Run13 continuation plan is the documented fallback.

## 6. Phase 2 - self-play loop with the oracle in-loop from day 1

P2 starts from the selected P1c handoff checkpoint (SHA-256
`374f0b72adbdf009d19abaed87addbdfe89364ecc1e6a7246b423233be51b42e`)
after the green G1 transfer gate. The base loop inherits Run13's Gumbel search,
playout-cap randomization, 20-iteration replay window, fresh-data-reuse AdamW
schedule, tactical shortcuts, inference deduplication, and bounded MCTS
inference cache. Run13 itself is not a gate opponent: milestone self-matches are
the primary progress signal and Run13 remains only an optional fixed anchor.

The oracle is present from the first P2 block through calibrated sparring. The
disagreement-start and auxiliary-label components remain independently gated so
their integration or ablation cannot delay ordinary self-play or contaminate the
control branch.

### 6.1 Search and throughput

- Freeze the initial P2 search contract at 96 full simulations, 32 fast
  simulations, 25% standard full-search probability, and full search on all
  placement turns. Keep Gumbel scale 1.0 in standard self-play, placement scale
  1.5 for 90% of games and 2.25 for 10%, policy-target temperature 1, tactical
  shortcuts, inference deduplication, and the 4,096-entry per-move inference
  cache. A later 64-versus-96 full-cap bake-off is explicitly deferred; 96 is a
  validated starting point, not a claim of optimality. Recalibrate the oracle
  sparring rung if the live full-search budget changes.
- Use one neural orientation per leaf: no root orientation averaging, symmetry
  refresh, training augmentation, or consistency loss. The exact batched
  canonical wrapper and stabilizer projection are already verified for neural
  policy/value inference.
- Do not overstate that inference result as a complete-search result. Before the
  first live P2 job, add a transformed-root preflight covering a complete
  deterministic Gumbel-scale-zero search, including symmetric roots and tactical
  shortcuts. Deterministic analysis and promotion searches must transform their
  returned policies and, for asymmetric roots, selected actions correctly. On a
  symmetric root where no single action is fixed by the stabilizer, tied choices
  are compared by stabilizer orbit rather than demanding an impossible unique
  equivariant action. Stochastic self-play Gumbels need an
  equivariant distribution, not identical actions under a reused scalar seed;
  if pathwise same-seed equivariance is later required, generate the random
  vector in the D4-canonical root action frame and map it back.
- Use exported TorchScript **FP32** inference on P100 initially. The optimized
  canonical 6x192 wrapper measured 3,285 examples/s at batch eight in FP32
  versus 2,713 with autocast FP16 (FP16 was 17% slower). At batches 32 and 64,
  FP16 was respectively flat (+0.1%) and only 2.5% faster, despite essentially
  exact prediction agreement. The measured Run13 self-play workload averaged
  about 47 executed evaluations per inference batch, so neither peak P100 FP16
  FLOPS nor the batch-eight result alone predicts end-to-end P2 throughput.
- Begin with the inherited 128 concurrent self-play games, record the actual V4
  inference-batch distribution, games/hour, fresh positions/hour, cache reuse,
  GPU utilization, and complete wall-clock split, then test adjacent concurrency
  values around the observed knee. The end-to-end smoke is authoritative; the
  frozen-wrapper microbenchmark is only a diagnostic.
- Treat T4x2 as an optional throughput experiment, not a P2 dependency. First
  measure one T4 at FP32 and autocast FP16 on the observed V4 batch distribution
  and run the same end-to-end smoke contract as P100. A second T4 provides no
  automatic speedup to the current single-device loop; prototype one independent
  self-play worker per GPU, deterministic example merging, and resumable worker
  RNG only if the single-T4 result is promising. Adopt dual T4 only on a measured
  material games/hour gain after merge and training overhead, with unchanged
  search/replay semantics.

### 6.2 Distribution shaping

- **Sparring (~10% of games):** use `SantoriniOraclePlayer` at frozen ladder
  version 2, **5k nodes per move**, from symmetry-distinct completed
  placements. Reset the oracle at every game boundary. Store only the network's
  own full-search decisions, its native MCTS policy, and the real mixed-game
  outcome; never store oracle decisions as network policy targets. Record rung,
  engine digest, opening-pool version, seat, stage, and source on every game and
  replay record. The joint-to-sequential placement adapter remains outside live
  mixed-engine games until separately validated there.
- **P2-start calibration evidence:** the original deterministic evaluation
  proxy (96 simulations on every move, Gumbel scale zero) selected 100k, but it
  did not reproduce the live 96/32, 25%-full, Gumbel-scale-one player. In the
  100k P100 smoke that player scored only 2-22. An exact live-policy coarse
  sweep bracketed the target at 5k/7.5k; the 40-game confirmation scored 42.5%
  at 5k (paired bootstrap 27.5%-57.5%), 27.5% at 7.5k, and 30.0% at 10k.
  Freeze **5k as ladder version 2**. Treat ladder v1 as historical calibration
  evidence only. If the live neural search contract changes, recalibrate rather
  than carrying this rung forward. The separate 250k deterministic diagnostic
  improved V4 from 22.5% at 96 simulations
  to 45.0% at 1,024, showing useful search elasticity rather than a deep-search
  collapse; 1,024 remains an evaluation diagnostic, not a self-play budget.
  Full records are in
  `experiments/santorini_v4/P2_ORACLE_SWEEP.md`.
- **Bootstrap-to-self-play optimizer transition:** the first P100 arms applied
  the inherited 16x fresh-data budget to only 3,371/3,070 training positions.
  Their frozen teacher objective jumped by +0.346/+0.349, and equal-96 standard
  matches against P1c scored 35.0%/22.5% for ordinary/mixed. Retraining the
  exact ordinary replay from P1c at 2x reuse scored 55.0% against P1c and moved
  the frozen objective by only +0.0106. The accepted iteration-one transition
  also used 2x. The first continuation showed that 4x at iteration 2 was
  tolerable (+0.02522) but 6x at iteration 3 crossed the standing gate
  (+0.05651); that uncertain branch was discarded. A second branch restarted
  from iteration 1 and ramped more slowly from 2x at iteration 2 by +1x per
  iteration. It paused at iteration 8 (+0.06012), where equal-96 standard
  strength had collapsed 9-31 against iteration 1. The final objective step was
  primarily value loss, while seam and oracle-ratchet telemetry stayed clean.
  Therefore no reuse ramp is currently authorized. The next diagnostic holds
  2x through iterations 2-4, checks strength, then--only if healthy--through
  iteration 7 before testing 3x or a lower learning rate. This is an
  optimizer-dose transition, not retention of blended teacher labels; the P2
  main value target remains pure self-play `z`.
- **Sparring-rung ratchet:** retain the most recent 40 complete seat-swapped
  pair scores (80 games) in resumable checkpoint metadata. At 50%-55% V4 score,
  emit a watch signal. At **55% or higher**, finish and save the current
  iteration, then pause before the next one and recalibrate with the exact live
  search contract. Reset the rolling history only when the declared node rung
  or ladder version changes. Report the rolling paired bootstrap interval and
  2-0/1-1/0-2 counts, but use the predeclared point threshold for the ratchet.
- **Disagreement-seeded starts (target ~10% after activation):** adapt the existing
  Run13-specific adversarial miner to the canonical V4 checkpoint and use
  high-margin, score-stable disagreements only as starting positions--never as
  teacher policy/value labels. The initial pool is mined from the exact P1c
  handoff, not inherited from Run13. Pools are stage/value stratified,
  D4-deduplicated, versioned by source checkpoint and oracle settings,
  age-limited, and replay-capped. Do not enable this arm until the V4 loader,
  source metadata, start-state legality, and deterministic refresh tests pass.
- Sparring and seeded starts have independent switches and telemetry. Preserve a
  short ordinary-self-play control and run predeclared matched on/off branches at
  an early checkpoint rather than interpreting an uncontrolled main trajectory.

### 6.3 Value labeling (auxiliary head only, this run)

- The selected ordinary 6x192 checkpoint does not yet contain the auxiliary
  oracle-value head; only the equivariant prototype does. Before enabling this
  arm, add the parallel invariant head to the selected ordinary model and a
  versioned P1c checkpoint migration. Loading the migrated checkpoint must leave
  the main policy/value outputs bitwise or tightly numerically unchanged, while
  initializing only the new head. Self-play inference/export continues to return
  only the main policy and value.
- Extend the compact replay schema before collecting labels. Keep z,
  `v_oracle`, node budget, raw score, mate-band flag, engine/calibration version,
  source type, stage, and an oracle-valid mask as separate fields. Old replay
  remains loadable with the mask false. Blending or auxiliary weighting is a
  training-time choice and must never require regenerating replay.
- Do not silently treat the §4.4 stability ladder as a calibrated value ladder.
  P0b calibrated 20k/50k scores, while the stable middle/early operating points
  are 100k/250k and were not separately calibrated; early 250k also passed only
  by comparison with itself. The first safe online arm may label a 25% subsample
  of stored **late** positions at the calibrated/stable 20k setting. Middle and
  early labels remain masked or telemetry-only until their chosen budgets receive
  held-out calibration/adjudication. Any broader 25-50% plan is deferred until
  that semantic gap and labeler throughput are measured.
- Start the auxiliary loss weight at 0.05. Increase toward 0.10 only if the main
  policy/value validation, seam sentinel, and strength telemetry remain healthy;
  0.2-0.3 is no longer a default ramp target. Run a short matched auxiliary-off
  ablation because shared-trunk gradients can affect search even though the
  auxiliary output is not consumed by MCTS.
- After the supervised-bootstrap handoff, the main value head trains on self-play z only (`lambda_blend = 0`) for this run. If the blended bootstrap won §5.2, record the resulting target-semantics transition explicitly.
- Telemetry: auxiliary prediction versus oracle label, oracle label/main-head/z versus the deeper adjudication suite, main-loss changes with auxiliary gradients, sparring win rate by ladder rung, and seeded-start replay fraction/source age. Auxiliary agreement alone measures learnability, not oracle truth.

### 6.4 Frozen canonical-seam sentinel

Carry the P1b seam diagnostic through P2 as a fixed regression sentinel. The
suite is the same 3,000 canonical selection positions, the same stable
750-position exposure quartiles, and the same global-blend teacher policy/value
targets. It also freezes the P1c handoff checkpoint's per-position losses. P2's
main value target still switches to pure self-play `z`; changing that training
semantic must not change this diagnostic's semantic.

Evaluate the suite after every completed training iteration through the exact
canonical inference wrapper. Record overall and Q1-Q4 policy CE, value MSE,
combined objective (`0.25 * policy_ce + value_mse`), and policy top-1. The
primary sentinel is the change from P1c in the Q4-minus-Q1 objective contrast,
computed from paired per-position excess losses. Report a deterministic paired
bootstrap interval. A contrast increase above `+0.02` is an early warning; call
it confirmed only when the 95% interval also clears zero. This is telemetry, not
an automatic strength gate: one warning prompts inspection, while a confirmed
warning or a persistent upward trend pauses promotion for a targeted
frame-switch/search diagnostic and exposure-stratified arena. Do not silently
change the suite, targets, quartiles, baseline, or threshold during P2.

The suite's overall teacher objective is also a standing optimizer-dose gate,
separate from the seam contrast. Compare every checkpoint with the immediately
preceding checkpoint (the frozen P1c baseline for iteration one). If the
objective rises by more than **+0.05 in one iteration**, save the completed
checkpoint, replay, control history, and telemetry, then pause before further
self-play. Resume only after reviewing optimizer dose, effective replay reuse,
learning rate, replay-window composition, and a paired strength check. Persist
the preceding objective in resumable checkpoint metadata; a process restart
must not reset the comparison. The gate is one-sided: improvements and smaller
increases are recorded without pausing.

The next diagnostic should also add a cumulative review threshold near +0.10
from the accepted iteration-one objective. Crossing it saves state and requires
a paired standard/placement strength check before continuation, but does not by
itself reject a checkpoint. This catches slow accumulation that never crosses
+0.05 in one step; the second continuation entered that region well before its
iteration-eight pause.

The frozen P1c baseline objective is `0.737597`; Q1 is `0.699188`, Q4 is
`0.760605`, and the baseline high-minus-low contrast is `+0.061416`. The suite
artifact is `temp/santorini_v4_p2_preparation/v4-seam-telemetry-suite.npz`
(SHA-256 `a25633021d8cef71e87b307c9b2369e8df5d8599de52848a443046b37a5fcd7e`).
It is generated reproducibly by `prepare_santorini_v4_seam_telemetry.py` and
enabled with `--v4-seam-telemetry-suite`; its runtime is charged to the existing
arena/telemetry wall-clock bucket and its evaluation preserves all training RNG
states.

### 6.5 Deferred to a following run

Folding oracle value into the search-facing target,
`(1-lambda)*z + lambda*v_oracle`, with lambda chosen from the oracle label's measured accuracy against deeper adjudication plus a direct playing-strength ablation - not from auxiliary-head agreement and not by guess. Apart from the explicitly declared coherent-teacher bootstrap exception in §5.2, this search-facing blend stays out of V4's first AlphaZero run.

## 7. Evaluation and promotion

- Rules cross-validation (`validate_santorini_oracle.py`) must pass before any cross-engine result is trusted (existing policy, unchanged).
- **Primary longitudinal progress signal:** milestone self-matches, current P2
  checkpoint versus the checkpoint one milestone interval earlier. Keep the
  standard-play and placement-inclusive results separate. Each uses 20 fixed
  paired blocks (40 games), with the standard openings, placement seeds,
  evaluation search settings, and milestone interval frozen before the run.
  Report paired score/interval, 2-0/1-1/0-2 block counts, and early/middle/late
  standard strata. These moving-baseline matches estimate local progress; they
  are not an absolute rating and do not automatically roll back an update.
- Run13 is an optional manual fixed anchor only. Leave it disabled in the normal
  P2 job; invoking it does not constitute a gate or override milestone evidence.
- Scheduled review strength uses paired arenas versus the calibrated oracle at
  96 and 128 simulations. Report equal-simulation and approximately equal
  wall-clock/neural-evaluation results where costs differ materially.
- Stress test: 1024 sims - retained specifically because it caught the distillation collapse. A V4 candidate that wins at 128 and collapses at 1024 is rejected.
- Statistics: predeclare the effect and stop/accept regions. Use sequential testing over independent paired-opening blocks or fixed arenas sized for the effect sought. Report paired/block-bootstrap intervals (and ordinary Wilson intervals only as secondary per-game summaries), because the two seat assignments of one opening are correlated. The Run13 experiments showed that unplanned 40-160-game arenas cannot resolve the effects that matter.

## 8. Measured compute baseline and accelerator policy

The representative Run13 P100 loop takes 348.45 seconds per amortized
iteration; self-play accounts for 322.97 seconds (92.69%). The selected V4
wrapper benchmark establishes precision and canonicalization costs but is not a
substitute for a complete V4 iteration: at batch eight, optimized canonical
FP32 reaches 3,285 examples/s and carries a 24% latency premium over the matched
uncanonicalized wrapper. FP16 autocast is slower at batch eight and only ties or
slightly wins at batches 32-64. P100 production therefore starts in FP32.

The first P2 smoke measured 2,440 games/hour for ordinary self-play and 2,371
games/hour for the 100k mixed arm. Replacing 10% of games increased total wall
time by only 2.9%, so oracle process time is not the limiter; small four-game
sparring batches increased inference-batch count by 50%, which remains a useful
later batching optimization. The revised 5k/2x transition smoke must validate
the corrected curriculum and optimizer entry before the production lineage.
Concurrency, FP16, T4x2, or a lower
simulation cap receive compute credit only from matched end-to-end measurements.
Mixed-precision training remains optional and separate from inference precision.

Dual T4 requires explicit process-level parallelism because the two devices do
not accelerate the existing single-device worker automatically. The evaluation
order is: one-T4 FP32/FP16 wrapper benchmark at observed batches, one-T4
end-to-end smoke, then a minimal one-self-play-worker-per-GPU prototype only if
those measurements justify it. P100 remains the reference path and P2 is not
blocked on dual-GPU support.

## 9. Risks and fallbacks

| Risk                                     | Signal                                                                    | Response                                                                                                                                 |
| ---------------------------------------- | ------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| P2 migration changes the handoff player  | migrated P1c main outputs or 96-sim frozen-suite behavior differ          | Stop before self-play; fix the V4 trainable wrapper/checkpoint adapter and initialize only genuinely new state                           |
| Complete-search D4 mismatch              | transformed deterministic roots return non-transformed policies or non-orbit-equivalent tied actions | Fix search-frame/tie handling before live replay; do not hide it with root averaging                                                      |
| Kaggle throughput worse than expected    | measured V4 games/hour makes the iteration materially slower than budgeted | Tune concurrency/cache/precision from end-to-end data; benchmark T4x2; revisit 64 sims only through the declared bake-off                 |
| Oracle labeling too expensive or ambiguous | labeler falls behind self-play, or stage/budget calibration is missing    | Leave the auxiliary arm disabled or late-only; lower its subsample rate; never fill invalid labels with an unvalidated mapping            |
| Placement regression                     | placement-inclusive milestones decline while standard milestones improve | Diagnose each sequential placement and the sparring/seeded-start mix; use the optional Run13 placement control only as a targeted diagnostic |
| Bootstrap-to-z transition hurts          | ordinary-self-play control regresses immediately while migration checks pass | Pause shaping arms and isolate optimization/replay effects in a matched branch; do not reintroduce oracle blend into the main head ad hoc |
| Auxiliary gradients hurt main heads      | auxiliary-on validation/arena branch regresses                            | Keep auxiliary weight low or disable it; stored labels and telemetry remain useful                                                       |
| Value-semantics mismatch shows up later  | future main-head blend regresses at deep search                           | `lambda` stays 0; auxiliary head retained only if its shared-trunk ablation is healthy                                                    |
| Canonical seam emerges during self-play | frozen Q4-minus-Q1 excess-loss contrast rises by >0.02 or its interval clears zero | Keep the frozen suite unchanged; inspect exposure-stratified policy/value losses and run a targeted frame-switch/search diagnostic before promotion |

## 10. Milestones

1. **P0a - deterministic data contract:** pool reset fix + versioned label cache; Mortal-only V4 datagen schema with best action/successor; reset-per-game TT policy; schema/action/FEN tests.
2. **P0b - measurement:** deeper-adjudicator calibration study; score-stability-by-phase report; rules cross-validation; instrumented Run13 wall-clock split.
3. **P1a - pilot and architecture feasibility:** 100k-500k corpus pilot + conversion pipeline; V4 encoder rule/covariance tests; pinned-escnn and hand-rolled feasibility spike; exported-model equivariance/checkpoint tests.
4. **P1b.1 - small screening ablations (complete, conclusion corrected):** the source-aware trainer, ordinary 6/13-plane baseline, per-epoch selection, Candidate A-D sizing, matched stage/global/winner targets, and paired standard/full selection arenas are complete. Global blend is provisional and winner-only fails at this scale. Candidate C loses the matched full-game pilot arena 10-30 to ordinary 8x96. This rejects the tested candidate at this scale, not V4.
5. **P1b.2 - scaled architecture and target selection (complete):** the matched 100k/300k/1M curves select canonical ordinary 6x192 by supervised fit plus the frozen ordinary-family speed tie-break. Exact batched D4 inference reaches 3,285 examples/s at arena-relevant batch eight on P100, with a 24% latency premium over the identical uncanonicalized wrapper. Equivariant E retains a small early-stage advantage but loses the supervised/arena selection overall. The required 1M target ablation rejects winner-only noninferiority: its common handoff objective is +0.01031 worse, with a paired interval of +0.00169 to +0.01868, while the standard/full arenas cancel 40-40. Retain global blend (`alpha_boot=0.5`, `T=261.8`) and explicitly switch the main target to pure self-play `z` at handoff. Final data and arena seeds remain untouched.
6. **P1c - full corpus and pretraining (complete):** the selected canonical ordinary 6x192/global-blend model completed four epochs over the 10,640,649-draw frozen epoch with the pure T25 oracle placement policy. All input hashes match and no final data were touched. Epoch four wins the standard-only combined objective at 0.73775, improving 9.09% over the selected 1M-screen checkpoint; policy top-1 rises from 38.53% to 43.97%. The final epoch adds only 0.95% objective improvement and held-out value bottoms at epoch three, while the Run13-replay subgroup does not improve. Therefore do not authorize 20M datagen before the transfer gate. Placement fit is healthy (99.9963% legal mass and only 0.00889 policy CE above target entropy), but shared roots make this a pipeline check rather than generalization evidence. Proceed to G1 with checkpoint SHA-256 `374f0b72adbdf009d19abaed87addbdfe89364ecc1e6a7246b423233be51b42e`.
7. **Gate G1 (complete; green after diagnostic):** P1c beats Run13 28-12/30-10 on standard selection openings and 40-0/39-1 from sampled empty-board starts at equal 96/128 simulations. Equal-cost P1c-128 versus Run13-120 scores 31-9 standard and 40-0 full. The symmetric phase-gap rule correctly paused on the unexpectedly positive +30/+22.5-point full-game gaps. A separate 40-game-per-arm, selection-only decomposition resolves the stop: with shared P1c continuation, P1c placements score 50%/52.5%; with shared Run13, 45%/37.5%; replaying a balanced sample of those exact openings with normal contestants scores 97.5%/87.5%; greedy full placement at 128 remains 40-0. Controller substitution leaves placement boards identical. Thus the gap is natural-opening standard-play distribution/style compatibility, not mapping, seating, or sampling failure. Proceed to P2 while retaining placement-only telemetry. Final data remain untouched.
8. **P2 - self-play (two continuations discarded; fixed-2x diagnostic next):** the trainable canonical 6x192 wrapper, checkpoint migration, transformed-search preflight, replay/source metadata, paired reset-per-game sparring loop, frozen seam telemetry, and P100 throughput smoke pass. Exact live-policy calibration selects 5k ladder v2, and the corrected transition remains the only accepted production checkpoint at iteration 1. Attempt 1 used 4x/6x and paused at iteration 3; its strength was unresolved, so the branch was discarded conservatively. Attempt 2 restarted from iteration 1 at 2x and added 1x per iteration. Iteration 7 actually passed at +0.04365; iteration 8 paused at +0.06012. Frozen objective had accumulated from 0.75174 to 0.97853, validation worsened after iterations 3-4, and iteration 8 lost 9-31 standard and 17-23 placement-inclusive to iteration 1. The step was predominantly value loss; seam delta (-0.00277) and rolling oracle score (31.25%) were clean. Do not resume iteration 8. Next run a fixed-2x iterations 2-4 diagnostic from iteration 1 with per-iteration snapshots and paired standard/placement-inclusive checks, followed by a second fixed-2x block to iteration 7 only if healthy. Add a cumulative +0.10 teacher-review trigger alongside the +0.05 one-step gate. Disagreement starts and the low-weight auxiliary head remain separately gated as specified in §6.2-6.3; Run13 remains a manual anchor only.
9. **Review:** after ~20-30 iterations, run the 96/128/1024 battery and component ablations; decide whether to study a nonzero search-facing `lambda` in a following run.
