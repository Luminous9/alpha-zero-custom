# P2 A3 external-oracle anchors

## Result

A3 is complete. P1c, iteration 11, and iteration 14 were evaluated on the same
20 paired openings at both 20k and 100k oracle nodes under the identical
deterministic 96-simulation contract.

The point estimates improve through the lineage at 20k, but not at 100k:

| Oracle budget | P1c | Iteration 11 | Iteration 14 |
|---:|---:|---:|---:|
| 20k | 22-18 (55.0%) | 23-17 (57.5%) | 25-15 (62.5%) |
| 100k | 17-23 (42.5%) | 13-27 (32.5%) | 16-24 (40.0%) |

No paired checkpoint-delta interval excludes zero. A3 therefore demonstrates
no transferable strength gain over P1c. Iteration 14 recovers most of iteration
11's 100k deficit, but remains statistically and practically compatible with
P1c at that rung.

## Frozen contract

- Oracle binary SHA-256:
  `aeca25af0b00a1e8e834223990f90c93a5dc8eb8bf1d39fb60e387e2eee8614c`.
- 40 games per budget: 20 completed-placement openings with both seats.
- Opening seed `20260921`, bootstrap seed `20260922`.
- Identical ordered opening hashes for every checkpoint and budget.
- V4 uses deterministic 96-simulation Gumbel search at scale zero, action
  temperature zero, canonical D4 inference, one root frame, and FP32 CPU.
- Oracle transposition table resets at every game boundary.
- No final-test data or reserved final-arena seeds were used.

Checkpoint SHA-256 identities:

- P1c: `374f0b72adbdf009d19abaed87addbdfe89364ecc1e6a7246b423233be51b42e`
- Iteration 11: `e49ef7fcf0f6a897bb87fbee3e04e951901cd2a5c1cc8575c80aea40b272710c`
- Iteration 14: `e91006a0919769d83b0e8097f6e94ba0732540239eab2bbc149bf2465ed4148f`

## Pair outcomes and uncertainty

| Budget | Checkpoint | V4 2-0 / 1-1 / 0-2 | Paired bootstrap 95% |
|---:|---|---:|---:|
| 20k | P1c | 5 / 12 / 3 | 42.5%-67.5% |
| 20k | Iteration 11 | 4 / 15 / 1 | 47.5%-67.5% |
| 20k | Iteration 14 | 7 / 11 / 2 | 50.0%-75.0% |
| 100k | P1c | 3 / 11 / 6 | 27.5%-57.5% |
| 100k | Iteration 11 | 0 / 13 / 7 | 22.5%-42.5% |
| 100k | Iteration 14 | 0 / 16 / 4 | 30.0%-47.5% |

All deltas below are candidate minus reference on the same opening pairs:

| Budget | Comparison | Score delta | Paired bootstrap 95% | Improved / same / worse pairs |
|---:|---|---:|---:|---:|
| 20k | Iteration 11 − P1c | +2.5 points | -12.5 to +17.5 | 5 / 11 / 4 |
| 20k | Iteration 14 − P1c | +7.5 points | -7.5 to +22.5 | 6 / 11 / 3 |
| 20k | Iteration 14 − iteration 11 | +5.0 points | -7.5 to +17.5 | 5 / 12 / 3 |
| 100k | Iteration 11 − P1c | -10.0 points | -25.0 to +5.0 | 2 / 13 / 5 |
| 100k | Iteration 14 − P1c | -2.5 points | -20.0 to +12.5 | 4 / 12 / 4 |
| 100k | Iteration 14 − iteration 11 | +7.5 points | -2.5 to +17.5 | 4 / 15 / 1 |

## Combined A1-A3 interpretation

1. A1 rejects a broad value-head erosion: aggregate deep-value metrics are
   flat-to-slightly improving through iteration 14.
2. A2 provides directional evidence that iteration 14 is modestly weaker than
   iteration 11 inside the checkpoint family, while remaining near iteration 1.
3. A3 finds no demonstrated improvement over P1c at the stronger 100k external
   anchor. The positive 20k point estimates are compatible with either modest
   shallow-budget progress or sampling variance.

The opposing A2 and A3 directions are further evidence that these checkpoints
are non-transitive and style-sensitive. A single internal matchup should not be
treated as a scalar strength measurement.

Keep iteration 11 as the tentative internal incumbent, but do not claim that
the current P2 recipe has produced transferable progress over P1c. Do not
resume routine 2x/1e-4 training or introduce root-Q targets based on these
results. The next branch should change the learning/data recipe and retain P1c
as the clean external baseline and iteration 11 as the internal incumbent.

## Artifacts

- Combined paired summary:
  `temp/santorini_v4_p2_a3/a3-summary.json`
- Iteration-14 sweep:
  `temp/santorini_v4_p2_iter14_oracle_20k_100k_fixed96/oracle-sweep-summary.json`
- Iteration-11 20k fill-in:
  `temp/santorini_v4_p2_iter11_oracle_20k_fixed96/oracle-sweep-summary.json`
- Iteration-11 100k rerun:
  `temp/santorini_v4_p2_iter11_oracle_100k_fixed96/oracle-sweep-summary.json`
- Original P1c sweep:
  `temp/santorini_v4_p2_oracle_sweep/oracle-sweep-summary.json`

The reusable comparison tool is `summarize_santorini_v4_p2_a3.py`.
