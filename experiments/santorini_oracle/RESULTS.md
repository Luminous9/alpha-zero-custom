# Run13 Oracle Training Results

Run13 remains the active checkpoint. None of the tested oracle-distilled
checkpoints showed evidence strong and consistent enough to replace it.

## Hard best-move distillation

- Direct distilled-vs-Run13 arenas at 96 and 128 simulations split in opposite
  directions: 23–17 and 15–25.
- Against the oracle, the distilled checkpoint scored 5–35 at 96 simulations
  and 2–38 at 128 simulations. The corresponding Run13 baselines were 4–36 and
  5–35.

The small direct-play result did not reproduce across search budgets, and
oracle performance did not improve. Hard score imitation is rejected as a
promotion path in its tested form.

## Confidence-filtered soft distillation

- Direct play was 81–79 over 160 games, effectively neutral.
- In matched low-budget oracle games it scored 16/160, versus 13/160 for the
  original checkpoint; this difference was too small to establish an effect.

The broader, softer labels avoided a clear direct-play regression but did not
demonstrate a meaningful playing-strength gain.

## Adversarial disagreement corrections

Training policy corrections with the observed game outcomes produced a 12–28
direct result against Run13. Those value labels describe the old behavior
policy, not the counterfactual oracle move, so policy and value supervision were
misaligned.

Anchoring teacher values to the source network mitigated that regression:

- Direct play against Run13 was 20–20.
- At 128 simulations it scored 12/80 against the oracle, versus 7/80 for Run13.
- At 1024 simulations it scored 1–19, versus 6–14 for Run13.

The low-budget direction was interesting, but the deeper-search collapse makes
this checkpoint unsuitable for promotion.

## What remains useful

- The Rust bridge and Python adapter provide a reproducible external opponent.
- Differential rule validation protects all cross-engine comparisons.
- Paired openings, fixed-node arenas, budget stability, and ranked-root
  diagnostics are useful evaluation infrastructure.
- Adversarial capture is useful as a hard-position miner.
- Frozen BatchNorm, rehearsal, source-value anchoring, and separate policy/value
  gates are reusable safeguards for small-data fine-tuning.

## Conditions for revisiting training

Another attempt should change the supervision, not merely scale the same
dataset. Promising directions include counterfactual rollouts that produce
values aligned with the oracle action, pairwise/ranking objectives over root
moves, or using oracle disagreements to prioritize self-play positions without
directly imitating incompatible search scores. Any candidate must pass direct
Run13 arenas and oracle arenas at both normal and deep MCTS budgets before it is
considered for promotion.
