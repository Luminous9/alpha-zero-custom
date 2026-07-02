# Engineering Plan: Santorini AI Architecture V2 Transition

## 1. Problem Statement: Worker Identity Variance

During evaluation testing of the Iteration 250 engine (Run #10, see training-log.md), a feature-representation flaw was identified. The current rules board labels the current player's workers as `1` and `2`, and the opponent's workers as `-1` and `-2`. The neural encoder currently converts those labels into four separate same-color identity planes: current worker 1, current worker 2, opponent worker 1, and opponent worker 2. Additionally, the policy uses a 128-length action vector: 64 local move/build actions for the first labeled worker and 64 for the second labeled worker.

### Technical Consequences:

- **Permutation Sensitivity:** We see that the engine assigns slightly different positional values and move probabilities to identical physical board states if the identities of Worker 1 and Worker 2 are inverted.
- **Label Dependence:** Even though the current neural input uses binary identity planes rather than raw integer magnitudes, it still asks the model to learn separate patterns for worker label 1 and worker label 2.
- **Capacity Waste:** The neural network is forced to utilize its parameter capacity to memorize identical tactical patterns twice (once for each worker identity combinations).
- **Non-Symmetric Policy:** The network exhibits an artificial "identity preference," hindering its progression toward a true Nash Equilibrium.

---

## 2. Proposed Solution: Identity-Free Architecture

We will restructure the network inputs and outputs to make the workers completely anonymous, ensuring that a specific physical arrangement of pieces on the board _always_ maps to a singular, identical mathematical representation.

```
Old Structure: [current worker 1] [current worker 2] [opponent worker 1] [opponent worker 2] [height planes] -> [128 Policy Array]
New Structure: [current workers] [opponent workers] [height planes] -> [1600 Spatial Policy Tensor]

```

### Architectural Changes:

1. **Input Tensor:** Collapse the four identity-specific worker planes into two anonymous worker planes. One channel will be a binary 5x5 grid containing `1`s on both squares where the current player's workers stand and `0`s elsewhere, while the other channel will be the same for the opponent's workers. Keep the four existing height planes (`height == 1`, `height == 2`, `height == 3`, `dome`). This changes the neural input from **8 channels** to **6 channels**.
2. **Output Policy Head:** Expand the policy vector from 128 discrete actions to a spatial tensor of shape **5x5x64 (1,600 flat values)**.
    - The **5x5 grid** represents the physical starting coordinate of the chosen worker.
    - The **64 depth channels** represent the combined directional permutations (8 move directions $\times$ 8 build directions).
    - The flat action index will be:

      ```python
      action = ((origin_x * 5 + origin_y) * 64) + move_direction * 8 + build_direction
      ```

    - Note: For PyTorch, one way to build this is to make the final policy layer a Conv2d with 64 filters, and then Flatten it to 1600.
3. **Game Interface:** Treat this as a Santorini game-interface migration, not just a neural-net migration. `getActionSize`, `getValidMoves`, `getNextState`, `getSymmetries`, human/evaluation helpers, and tests must all agree on the new physical-origin action indexing.

---

## 3. Transition & Execution Strategy

To avoid losing the days of computational time spent training Iteration 250, we will execute a "Network Surgery" and offline data translation pipeline to bootstrap the V2 model.

### Phase 1: Network Surgery (Weight Transfer)

We will initialize the V2 network structure and surgically copy the trained weights from the Iteration 250 checkpoint file.

| Layer Block          | Action            | Justification                                                    |
| -------------------- | ----------------- | ---------------------------------------------------------------- |
| **Input Conv Layer** | Partially transfer | Combine the old same-color worker-plane weights into the new anonymous worker planes, and copy height-plane weights directly. |
| **Input BatchNorm**  | Reset running stats | The first activation distribution changes after the input encoding changes. |
| **ResNet Core Body** | **Transfer 100%** | Retains all abstract spatial, tactical, and blocking heuristics. |
| **Value Head**       | **Transfer 100%** | Target prediction scalar (-1 to 1) remains identical.            |
| **Policy Head**      | Reinitialize      | Output layer dimensions expanded from 128 to 1,600.              |

### Phase 2: Offline Data Translation

We will write a one-time translation script to process the saved replay buffer examples (`latest.examples` file in `/temp/santorini_kaggle_training6`) from Iterations 231 to 250.

- The script will read the old 128-length policy vectors.
- It will identify the absolute coordinates of Worker 1 and Worker 2 for that frame.
- It will remap those probabilities into the corresponding blocks of the new 1,600 spatial array based on the physical square those workers occupied.
- This remapping is lossless for a single labeled training frame. Across label-swapped equivalent frames, V2 intentionally collapses contradictory worker-label preferences into one physical-action representation.

### Phase 3: Supervised Bootstrapping

We will not use MCTS and self-play for the bootstrapping. Run a pure supervised training pass using the translated dataset to train the reinitialized Input and Policy layers of the V2 network.

- **Duration:** 3 to 5 epochs.
- **Goal:** Distill the old Iteration 250 policy/value behavior into the V2 representation while removing same-color worker-label preferences.
- **Metrics:** Track held-out policy cross entropy/KL against translated targets, value MSE, legal-action probability mass after masking, and label-swap invariance.

---

## 4. Post-Transition & Reinforcement Learning

Once bootstrapping is complete, the V2 model will then resume the standard AlphaZero self-play loop to further train and improve.

## 5. Verification & Success Metrics

Before allowing the new model to run unsupervised, we will validate the transition using the Santorini evaluation and pit tooling:

1. **Label-Swap Invariance Test:** Pass identical board states with same-color worker labels swapped to the V2 network. Values and physical-action policy probabilities should match within a tight numerical tolerance, e.g. `np.allclose(..., atol=1e-6)`.
2. **Symmetry Test:** Verify that board rotations/reflections preserve legal-action masks and transform the 1,600-length policy correctly.
3. **Parity Match:** Run a pit match between the new bootstrapped V2 model and the old Iteration 250 model. A 100-game match is useful as a smoke test, but larger matches are needed for strong statistical confidence.
4. **Self-Play Smoke Test:** Run a short local training preset to ensure MCTS masking, self-play example generation, training, and arena comparison all work with 1,600 actions.

---

## 6. Implementation Notes

V2 migration tooling lives in `migrate_santorini_v2.py`.

Iteration 250 migration command:

```bash
.venv/bin/python migrate_santorini_v2.py \
  --source-folder temp/santorini_kaggle_training6 \
  --output-folder temp/santorini_kaggle_training6_v2
```

Generated artifacts:

- `temp/santorini_kaggle_training6_v2/best.pth.tar`: V2 checkpoint created from the Iteration 250 V1 checkpoint. The residual tower and value head are transferred; the input stem is partially transferred into anonymous worker channels; the V2 spatial policy head is newly initialized.
- `temp/santorini_kaggle_training6_v2/latest.examples`: V2 replay buffer translated from 128 worker-slot policies to 1,600 physical-origin policies.
- `temp/santorini_kaggle_training6/merged_20.examples`: V1 replay buffer merged from `training4/latest.examples`, `training6/checkpoint_28.pth.tar.examples`, `training6/checkpoint_34.pth.tar.examples`, and `training6/latest.examples`, capped at 20 exact-duplicate-free history windows.
- `temp/santorini_kaggle_training6_v2/merged_20.examples`: V2 translation of the merged 20-window replay buffer.

Validation completed:

- The migrated checkpoint loads into the current V2 network and predicts a `(1600,)` policy.
- The translated replay buffer contains 118,120 examples across 6 history windows.
- The merged translated replay buffer contains 399,368 examples across 20 history windows.
- Sampled translated policies sum to 1 and place all probability mass on legal V2 actions.
- Santorini unit tests pass with the V2 action interface.
