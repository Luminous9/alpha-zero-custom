# P2 frozen deep-value audit

## Result

The audit is complete. It does **not** support the proposed broad, monotone
value-head erosion from P1c through iteration 14. On the full frozen suite,
iterations 11 and 14 are statistically compatible with P1c and iteration 4;
their point estimates are slightly better in Pearson correlation and MSE and
essentially flat in MAE. This rules out a general value collapse as the root
cause of the stalled strength signal and does not justify changing to a root-Q
value target on its own.

The audit did reveal one localized warning. On the 120 positions sampled from
replay windows 9-11, P1c is materially better than the P2 lineage. Iteration 11
versus P1c loses 0.0885 Pearson correlation and adds 0.0315 MSE, with both
paired intervals excluding zero. The same pattern is already visible for
iteration 1 and iteration 4 on this position subset, so the window name must
not be interpreted as an onset date. It is evidence of distribution-specific
generalization loss on states that later appeared in windows 9-11, not evidence
that the value head first deteriorated during iteration 9.

## Frozen contract

- Source replay: the complete 14-window archive, SHA-256
  `4500335b1cc8f534160b6ea0ac55e7cebb7876c83a8bf710be818c2fbf6e0747`.
- Suite: 480 standard-play positions, all D4-unique.
- Sampling: 40 positions in each of 12 strata formed by three game stages and
  four replay bands (windows 1-4, 5-8, 9-11, and 12-14).
- Sources: 423 self-play and 57 oracle-sparring positions, also reported
  separately.
- Pretraining final-test protection: split-2 hashes from the final corpus were
  used only for exclusion; two replay positions were excluded before sampling.
- Suite seed: `20260814`.
- Frozen suite SHA-256:
  `d71ab1bda7d45421f9fd6a500706cc256c4bb8a1a160145bce65d8cedb21f7e6`.
- Labels: 250k nodes per position, reset/cold TT for every search, native oracle
  SHA-256 `aeca25af0b00a1e8e834223990f90c93a5dc8eb8bf1d39fb60e387e2eee8614c`.
- Value mapping: the established nominal score temperature 400, with mate-band
  scores mapped to ±1. These are calibrated engine proxies, not solved values.
- Uncertainty: 10,000 paired bootstrap samples, stratified by replay band and
  stage.

The audit covers P1c and iterations 1, 4, and 8-14 on the identical positions.
No final-test boards or targets were evaluated.

## Overall metrics

| Checkpoint | Pearson | Spearman | MSE | MAE |
|---|---:|---:|---:|---:|
| P1c | 0.7018 | 0.6998 | 0.13466 | 0.21502 |
| Iteration 1 | 0.6971 | 0.6953 | 0.13640 | 0.22255 |
| Iteration 4 | 0.7002 | 0.6955 | 0.13224 | 0.21803 |
| Iteration 8 | 0.7080 | 0.6885 | 0.12843 | 0.21544 |
| Iteration 9 | 0.7081 | 0.6875 | 0.12902 | 0.21573 |
| Iteration 10 | 0.7113 | 0.6868 | 0.12793 | 0.21585 |
| Iteration 11 | 0.7151 | 0.6915 | 0.12612 | 0.21423 |
| Iteration 12 | 0.7153 | 0.6900 | 0.12671 | 0.21574 |
| Iteration 13 | 0.7209 | 0.6968 | 0.12503 | 0.21630 |
| Iteration 14 | 0.7238 | 0.7015 | 0.12409 | 0.21683 |

The outcome target itself is much noisier against the oracle proxy: correlation
0.3601, MSE 0.8931, and MAE 0.8172. This confirms the motivation for guarding
the value head from small-window outcome noise, but it does not show that the
current 1e-4, 2x recipe has damaged the aggregate value function.

## Primary paired comparisons

All deltas are candidate minus reference. For correlation, positive is better;
for MSE and MAE, negative is better.

| Comparison | Pearson delta (95%) | MSE delta (95%) | MAE delta (95%) |
|---|---:|---:|---:|
| Iteration 11 − P1c | +0.0132 [-0.0204, +0.0460] | -0.00854 [-0.02398, +0.00694] | -0.00079 [-0.01209, +0.01062] |
| Iteration 14 − P1c | +0.0220 [-0.0126, +0.0576] | -0.01057 [-0.02795, +0.00584] | +0.00181 [-0.01093, +0.01452] |
| Iteration 11 − iteration 4 | +0.0149 [-0.0045, +0.0362] | -0.00612 [-0.01557, +0.00277] | -0.00380 [-0.01045, +0.00267] |
| Iteration 14 − iteration 4 | +0.0236 [+0.0004, +0.0483] | -0.00815 [-0.01976, +0.00286] | -0.00120 [-0.00919, +0.00678] |
| Iteration 14 − iteration 11 | +0.0088 [-0.0035, +0.0228] | -0.00203 [-0.00707, +0.00238] | +0.00260 [-0.00074, +0.00579] |

Stage-stratified results are consistent with the aggregate interpretation.
Relative to P1c, iteration 14 has slightly worse early correlation but lower
early MSE, essentially flat middle-stage metrics, and better late correlation
and MSE with slightly worse late MAE. There is no consistent stage-wide
collapse.

## Window 9-11 localization

| Comparison on windows 9-11 | Pearson delta (95%) | MSE delta (95%) | MAE delta (95%) |
|---|---:|---:|---:|
| Iteration 1 − P1c | -0.0432 [-0.0781, -0.0125] | +0.01586 [+0.00531, +0.02702] | +0.02438 [+0.01009, +0.03931] |
| Iteration 4 − P1c | -0.0421 [-0.0783, -0.0100] | +0.01530 [+0.00375, +0.02775] | +0.02138 [+0.00719, +0.03600] |
| Iteration 11 − P1c | -0.0885 [-0.1551, -0.0328] | +0.03154 [+0.01155, +0.05282] | +0.03211 [+0.01225, +0.05185] |
| Iteration 14 − P1c | -0.0780 [-0.1402, -0.0206] | +0.03015 [+0.00475, +0.05559] | +0.03432 [+0.01114, +0.05754] |

P1c is not similarly superior on the other three replay bands. Iterations 11
and 14 have better point estimates there, although most per-band intervals
overlap zero. The source split also does not establish a broad oracle-sparring
pathology: the 57-position sparring subset is too small and its paired
intervals overlap zero.

## Decision and next action

Classify A1 as **aggregate value quality flat/improving, with a localized
generalization warning**. Do not adopt root-Q blending or restart from P1c on
this evidence. Keep iteration 11 as the tentative production head while the
strength questions remain unresolved.

Proceed to the fresh 80-120 paired-seat P100 arenas (A2), followed by the 20k
and 100k external-oracle anchors (A3). The window-9-11 subset should remain a
standing value telemetry slice during any subsequent branch. If a future
training change worsens both the full frozen suite and this slice, it should
trigger review before another strength milestone.

## Artifacts

- `temp/santorini_v4_p2_deep_value_audit/frozen-value-suite.npz`
- `temp/santorini_v4_p2_deep_value_audit/frozen-value-suite-manifest.json`
- `temp/santorini_v4_p2_deep_value_audit/deep-oracle-labels.sqlite3`
- `temp/santorini_v4_p2_deep_value_audit/deep-value-audit-rows.npz`
- `temp/santorini_v4_p2_deep_value_audit/deep-value-audit-summary.json`

The runner is `audit_santorini_v4_deep_value.py`; reruns reuse the frozen suite
and cached labels and reject a changed sampling contract.
