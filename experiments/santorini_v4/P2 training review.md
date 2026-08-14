# P2 iteration-14 strength review and measurement overhaul

Summary: iteration 11 remains the tentative production head. No routine continuation is authorized until the local value audit (action 1) completes and the decision matrix below selects a branch. This document synthesizes the `P2_PURE_Z_12_14-md` review with one new external-anchor measurement and replaces "run more iterations and re-arena" with an audit-first plan.

## Inputs

1. The iterations 12-14 pure-`z` review (`P2_PURE_Z_12_14.md`): all safety and optimization telemetry healthy; iteration 14 at parity with iteration 11 head-to-head (19-21 standard, 22-18 placement-inclusive); common-suite score against iteration 1 fell from iteration 11's 29-11 (72.5%) to 20-20, an anchored change of -22.5 points (interval = -42.5 to -5.0).
2. **external-anchor measurement (previously unrecorded):** iteration 11 was rerun against the 100k-node oracle under the exact original Plc sweep contract - deterministic 96-simulation Gumbel search at evaluation scale zero, canonical D4 inference, the same 20 completed-placement openings and seeds, reset-per-game TT.
   | Checkpoint | vs 100k oracle | Pairs 2-0 / 1-1 / 0-2 |
   | --- | ---: | ---: |
   | Plc (original sweep) | 17-23 (42.5%) | 3 / 11 / 6 |
   | Iteration 11 (rerun) | 13-27 (32.5%) | 0 / 13 / 7 |

    Summary JSON is found at ./temp/santorini_v4_p2_iter11_oracle_100k_fixed96/oracle-sweep-summary.json

## Findings

### F1 - Three iterations at this dose sit below the arena detection floor

Iterations 12-14 at fixed 2x reuse are ~25 optimizer steps each: ~75 total gradient steps at LR 1e-4. A 40-game paired arena resolves roughly +/- 12 points; the expected strength change from 75 steps is one or two points. "No improvenent from 12 to 14" is therefore the guaranteed null result of this cadence, not evidence of a stall. The original plan's milestone cadence (20-iteration spans) exists for exactly this reason.

### F2 - The -22.5-point anchored drop is mostly variance, modestly inflated

Iteration 11 was not itself selected among sibling checkpoints (its review was scheduled), so this was not classic winner's curse. Three channels still bias the anchored cmoparison, in decreasing order of expected size:

1. **Double-noise differencing (dominant):** -22.5 differences two independent 40-game arenas; even bias-free, that difference has an SD near +/- 11 points, so a 1-1.5σ swing explains most of the gap.
2. **Heritable arm selection:** Arm D was selected at iteration 4 partly on matchup scores against iteration 1 _on these same opening blocks_ (the four arms spanned 52.5-65% standard - noise level differences, so picking the max inflates the winner by roughly 6-8 points). Whatever portion reflects genuine style-fit to those 20 openings is inherited by D's lineage, and iteration 11 is still being measured on D's lucky suite.
3. **Outcome-conditional anchoring:** treating 72.5% as iteration 11's level is conditional on that single draw coming out high; had it rolled 55%, iteration 14's 50.0% would read as noise. Comparing successors against one retained-because-favorable measurement builds regression-to-the-mean into the narrative even without any checkpoint selection.

The direct arena agrees with this reading: iteration 14 at 47.5% head-to-head is parity, where a genuine 22-point regression should have shown ~30%. The true anchored change is likely smaller than the point estimate.

### F3 - The external anchor reads "no demonstrated general improvement"

The oracle rerun is the least-contaminated instrument in the P2 record: neither P1c nor iteration 11 was ever selected or gated on this suite, and the opponent is outside the training lineage. Its -10-point change is within noise (the paired interval straddles zero), so it does not establish a regression - but it removes the main evidence of _general_ improvement. The coherent joint reading of all three facts (iteration 1 ≈ P1c; iteration 11 ≫ iteration 1 family-internally; iteration 11 ≤ P1c versus the oracle) is that the iterations 2-11 gains are substantially family-specific - style and relative-exploitation gains against the checkpoint's own lineage - rather than transferable strength. A contributing mechanism: 10% of training games spar against the 5k-node oracle rung, which teaches punishing shallow-search errors that do not exist at 100k nodes.

### F4 - Improving `z`-fit is ambiguous and the current gates cannot resolve it

Replay validation value loss improved through the block (0.9101 → 0.8946). The signal diagnostic (`P2_SIGNAL_DIAGNOSTIC.md`) proved this metric is actively misleading in this regime: the failed iteration-8 branch improved its `z` MSE from 0.895 to 0.631 while its correlation with cached 250k oracle labels collapsed from 0.55 to 0.14. Fitting single-game outcomes better is consistent with both value learning and value noise memorization. The bridge was mostly `z` from iteration 9 onward (beta 0.833/0.917/1.0), so slow value erosion may have begun before iteration 12 - at 1e-4 it would run roughly 9x slower than the 3e-4 failure, slow enough to present as a plateau. No standing per-iteration metric currently compares the value head to oracle-anchored truth, so this hypothesis is invisible to every existing gate.

## Required next actions

Actions 1-2 are local/CPU and precede any further GPU training.

### A1 - Local oracle-anchored value audit (blocking)

Sample 300-500 D4-unique standard-play positions from recent replay windows (final-test positions excluded), label them at 250k nodes through the existing cache/pool, and evaluate the value heads of **P1c, iteration 4, iteration 11, and iteration 14** (iterations 12-13 optional, for onset localization) against those labels: correlation, MSE, MAE, stratified by stage. The tooling exists (`diagnose_santorini_v4_replay_signal-py`, `ParallelOraclePool`, the label cache).

Decision rule: a monotone decline in oracle correlation from iteration ~9 onward confirms slow value erosion under pure `z`; flat or improving correlation classifies iterations 12-14 as a plateau.

### A2 - Fresh-suite neural confirmation (as prescribed in `PZ_PURE_Z_12_14.md`)

80-120 paired-seat standard games on fresh fixed openings: iteration 14 versus iteration 11, and iteration 14 versus iteration 1. Placement-inclusive only as a secondary check. This resolves whether iterations 12-14 contain any real family-relative regression once suite-selection bias is removed.

### A3 - Formalize the external oracle anchor

Commit the iteration-11 rerun artifact (see "Inputs" section above), run the identical contract for iteration 14, and adopt deterministic oracle arenas at **20k and 100k nodes** as a standing component of every milestone review. This instrument is immune to both lineage non-transitivity and suite-selection bias, and is cheap (~10-15 minutes of local CPU per rung).

## Decision matrix

| A1 value audit                 | A2 fresh suite           | Classification                                     | Response                                                                                                                                                                                                                                                                                                                                                                                                                       |
| ------------------------------ | ------------------------ | -------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Correlation declines into 9-14 | any                      | Slow value erosion under pure `z`                  | Change the value-target semantics before any continuation: `v_target = (1-λ)•q_root + λ•z` with λ = 0.5, where `q_root` is the stored Gumbel completed-Q root value. This replaces "anneal to pure `z`" as the destination semantics - the root value is on-policy, far lower variance than a single outcome, and caps nothing (unlike a permanent P1c anchor). Restart point per the rule below, run a short block, re-audit. |
| Flat/improving                 | Even                     | Plateau at deliberately tiny dose                  | Pace levers under the existing gates: judge strength at 20-iteration milestones on rotated suites; raise games/iteration 240 → 360-480 first (variance reduction by data, the safest speedup); optionally test 4x reuse at 1e-4 (untested - the prior 4x failure was at 3e-4); diversify sparring across two rungs (e.g., 5k plus a deeper rung at low weight) to reduce shallow-exploitation bias.                            |
| Flat/improving                 | Real regression in 12-14 | Family-relative regression with healthy value head | Localize onset with the saved iteration-12/13 checkpoints, then branch from iteration 11 with either the root-Q value mix or the retained Arm C differential-LR fallback.                                                                                                                                                                                                                                                      |

**Restart-point rule.** The audit, not lineage seniority, chooses the restart checkpoint. Branch from iteration 11 only if its oracle-anchored value quality matches P1c _and_ some gain looks transferable (A3 anchors). If iteration 11's value quality is degraded, or the anchors confirm no general-strength gain through iteration 11, restart from **P1c** under the corrected value semantics - or from the last checkpoint whose audited value quality matches P1c (e.g., iteration 4 or 8) if the audit localizes a later onset. The durable assets of iterations 2-11 are the recipe, gates, calibrations, and instrumentation, all of which survice a restart; weights that bought no transferable strength do not justify continuing a lineage, and the replay window and reatchet history regenerate within one or two ~6-miute iterations.

## Standing instrumentation changes (all branches)

1. **Per-iteration oracle-anchored value telemetry:** ~200 cached-label positions per iteration; report value correlation/MSE against deep oracle labels; warn on a declining trend over a rolling three-iteration window. This is the instrument whose absence made both the iteration-8 failure and this review ambiguous.
2. **Rotate evaluation suites:** never reuse the openings that promoted the incumbent to judge its successor. Draw per-milestone opening seeds. Final arena seeds remain reserved and untouched.
3. **Milestone cadence:** strength promotion decisions at 20-iteration spans; interim reviews are safety-only (gates, seam, KL, oracle-value telemetry) and make no strength claims either direction.
4. **External anchor at milestones:** the A3 oracle rungs, alongside the neural milestone arenas.

# Protocol

- Iteration 11 remains the tentative production head; iterations 12-14 remain retained diagnostic branch state.
- No lineage continuation until Al completes and the matrix selects a branch.
- The root-Q value mix, if selected, is a declared change of target semantics and must be recorded in checkpoint metadata like the bridge was (A, source of root', start iteration).
- Neither the analyses here nor the proposed actions touch final-test data or final arena seeds.
