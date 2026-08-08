# Santorini V4 P1a Corpus Pilot

This pilot exercises the Mortal-vs-Mortal bootstrap data path at 100k records
before any multi-million-record generation or supervised training.

## Generation contract

The V4 datagen mode now supports a deterministic finite run:

```bash
cargo run -p datagen --release -- \
  --v4 --threads 8 --target-records 100000 --seed 20260808 \
  --output-dir /path/to/temp/santorini_v4_pilot_100k/raw-v3 \
  --random-moves-min 0 --random-moves-max 6 \
  --subgame-initial-chance 0.6
```

The total target is divided exactly across workers. Each worker owns a seeded
RNG and one shard, every independent main-line or randomized-subgame trajectory
starts with a cold transposition table, and all shards declare the executable
content digest and generation parameters. Existing output files are never
silently overwritten.

Two rejected attempts demonstrate the validation gate doing useful work:

1. The first attempt assigned randomized continuations their parent game ID,
   allowing records under one ID to disagree about the winner. Datagen now
   assigns every independently adjudicated continuation a trajectory game ID,
   while conversion groups all trajectories from the same root game into one
   data split.
2. The second attempt exposed terminal `no_moves` pseudo-actions. These states
   have value information but no AlphaZero policy target, so V4 emission now
   excludes them. Winning move-without-build rows remain and map to all legal V4
   action aliases.

The rejected raw directories are retained under `temp/` for diagnosis and are
not accepted by the converter.

## Conversion contract

`build_santorini_v4_corpus.py` performs full differential validation of every
record, converts each board and sparse best-action target to one deterministic
D4 orientation, averages policy targets over the board stabilizer, and
aggregates exact orbit duplicates without losing observation frequency.

The compact NPZ keeps winner outcome, raw score statistics, requested/actual
node counts, mate rate, source counts, stage, observation count, sparse policy,
and split ID separately. Train/selection/test assignment is made over connected
components of root game IDs and canonical position keys, preventing either
same-root trajectories or D4-equivalent duplicates from crossing splits.

## Pilot results

The corrected 100k run completed in 264 seconds on the local 8-worker CPU
setup. Full conversion and differential validation took 40.7 seconds; loading
and checking the resulting 6.46 MB NPZ took 0.72 seconds. The converter
preserved all 100,000 observations as 99,951 unique D4 positions (49 duplicate
observations, 0.049%), with no cross-split root-game or D4 leakage.

The correctness gate passes, but the default engine distribution does **not**
pass the scale-up gate:

| metric | 100k default (`0.6`) | 20k bake-off (`0.1`) |
|---|---:|---:|
| randomized-subgame records | 90.7% | 48.1% |
| early records | 1.2% | 4.0% |
| middle records | 11.5% | 17.0% |
| late records | 87.3% | 79.0% |
| score-sign/outcome agreement | 91.1% | 88.9% |

The seeded 20k bake-off completed in 72.8 seconds and independently passed the
same full conversion checks. Lowering the branch probability helps, but the
main-line distribution itself is about 70% late positions, so branch tuning
alone cannot create the intended phase balance. The production corpus must use
an explicit per-stage sampling/weighting rule and a controlled source weight;
it must also add the separately labeled Run13-replay component specified in
the plan. Until those rules are implemented and re-piloted, generation is held
at 100k rather than scaling to 5M.

The exact reports are
`temp/santorini_v4_pilot_100k/report.json` and
`temp/santorini_v4_pilot_branch_010/report.json`.

Raw shards and converted artifacts live under
`temp/santorini_v4_pilot_100k/`. They are measurement artifacts rather than
source-controlled training data.
