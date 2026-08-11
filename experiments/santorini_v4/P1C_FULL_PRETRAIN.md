# P1c full corpus and pretraining

Status: in progress. The placement-only tournament selected the pure
santorini-ai T25 policy, and the frozen full-corpus plan is complete. One
four-epoch P100 pretraining job is ready to run.

## Full-corpus audit

The completed 5.05M raw-record generation already satisfies the planned 5M
scale step. Its anchored train component is
`temp/santorini_v4_scaled/engine-corpus-1m-train.npz` (the historical `1m`
name refers to the screen size, not the underlying generated-record count). It
contains 4,166,074 D4-unique train positions and 4,752,217 retained raw
observations:

| Stage | Unique positions |
| --- | ---: |
| early | 229,085 |
| middle | 740,828 |
| late | 3,196,161 |

P1c therefore trains on this corpus before authorizing more datagen. Expansion
toward 20M raw records occurs only if the four-epoch full-corpus curve is still
materially improving.

The standard Run13 component contributes its 10,000 train positions once in the
coverage base. Repeats produce a 7,112,692-position standard epoch with the
frozen 20/35/45 stage and 77/22/1 source marginals while retaining every unique
engine and Run13 train position at least once.

## Placement contract

Run13's compact replay has 19,200 placement observations from 4,800 games, but
only 91 D4-unique placement positions. Exhaustive reachability contains 960
prefix orbits: 1/6/49/904 with zero/one/two/three workers placed.

Santorini-ai can search placement, but it chooses one unordered pair while V4
chooses two same-player actions. The oracle component queries the empty boundary
(300 legal pairs) and all 49 D4-unique post-P1-pair boundaries (253 pairs each).
It factors each joint distribution exactly into a first-square marginal and a
second-square conditional, covering all 960 sequential orbits without imposing
an arbitrary worker order. Every candidate pair receives an independent search
with a reset TT. Duplicate engine action orders are collapsed by resulting FEN.

Finite-budget engine scores were not D4-consistent, so scores equivalent under
the parent boundary's stabilizer are averaged before softmax. The raw teacher
search is frozen at 50,000 nodes per pair. Because every raw root response is
stored, policy temperature is a free deterministic rematerialization rather
than another engine search. The initial T300 policy was too diffuse in direct
placement play; the placement-only tournament selected T25. Oracle static
values remain telemetry only and are never substituted for a completed
outcome.

The Run13 labeler starts four continuations from every sequential orbit at 96
simulations, uses full search for every remaining placement choice, and combines
those fresh targets with replay observations. Its settings match Run13
pretraining behavior: Gumbel search, placement scale 1.5, all eight root
orientations during placement, policy-target temperature 1, and the normal
25%-at-96/otherwise-32 playout cap after setup.

The production component uses the pure santorini-ai T25 policy. Its completed
game `z`, observation weights, and position ordering come from the Run13
continuations. That is an explicit bootstrap compromise: Run13 is retained as
the source of valid game outcomes, not as evidence that its placement policy is
better. Standard engine records retain the selected global value blend
(`alpha_boot=0.5`, `T=261.8`).

Placement occupies 19,200/57,909 = 33.1555% of each P1c epoch, matching the
observed Run13 replay phase mix. Its 3,527,957 draws are split equally over the
four placement decisions. The resulting full epoch has 10,640,649 draws and is
repeated for four epochs.

A conventional placement holdout is not valid: the empty board and common
prefixes necessarily recur across games, so simultaneous exact-position and
game-level isolation collapses the holdout. Checkpoint selection therefore
uses the frozen standard selection set. Placement is checked through exhaustive
coverage diagnostics and the paired full-game Gate G1. The final test split and
final arena seeds remain untouched.

## Santorini-ai placement job

Build the local oracle service and generate its resumable 50-boundary corpus:

```bash
cargo build --release --manifest-path tools/santorini_oracle/Cargo.toml
.venv/bin/python label_santorini_v4_oracle_placement.py \
  --records-out temp/santorini_v4_oracle_placement/records.jsonl \
  --output temp/santorini_v4_oracle_placement/oracle-component.npz \
  --report-out temp/santorini_v4_oracle_placement/oracle-report.json \
  --nodes-per-move 50000 \
  --policy-temperature 25
```

The records contain every raw root response and make policy-temperature changes
free of additional engine searches. The component must report 50/50 boundaries,
960 unique positions, 1/6/49/904 coverage, reset-per-root-move TT, and no
completed outcomes. The frozen records SHA-256 is
`5c4dcb9f89a266857af7bf9bd97fb8c38d734e108567128ae063f4c9b501d691`.
The selected T25 rematerialization has SHA-256
`7d255c7cbbfa1b0f435c70d8b9bc3ea8c21ebd60eb2b1c8ccc72f8ae7cca1f16`.

## Run13 continuation Kaggle job

Build the upload locally:

```bash
.venv/bin/python prepare_santorini_v4_p1c_placement_bundle.py
```

Upload `temp/santorini_v4_p1c_placement_bundle.zip` as a Kaggle dataset and add
it to a GPU notebook. Kaggle exposes the zip contents already extracted. Do not
try to open it as a tar archive. Run this one cell; the entry point recursively
locates the extracted files beneath `/kaggle/input`:

```python
import subprocess
import sys
from pathlib import Path

entries = list(Path("/kaggle/input").rglob("run_santorini_v4_p1c_placement_kaggle.py"))
assert len(entries) == 1, entries
subprocess.run([sys.executable, str(entries[0])], check=True)
```

It writes:

- `/kaggle/working/santorini_v4_p1c_placement/placement-component.npz`
- `/kaggle/working/santorini_v4_p1c_placement/placement-report.json`

The local bundle has SHA-256
`7d64d3c8644d4fe08cee5683deca43c55893adf87ffe50e870cbf758473a1d30`.
The embedded Run13 checkpoint and replay digests are respectively
`cdc8ac1f396bed591fde419b63f8641cba403862c1812783fe14e6c048184f4e`
and `40df820594b630bc9e87f5f6a06b31d84d7179945e78aa25b3b94feb9727c2a9`.

This job completed in 4,970 seconds. It produced 23,296 aggregated placement
observations, including 4,096 fresh search observations, with exact
1/6/49/904 coverage. The returned NPZ SHA-256 is
`b168cad0f79781dd2ad3f4c6d08c412acf32e6e0ae062f28bfd7de88ca4278f4`.

After the Run13 output returns, use
`build_santorini_v4_mixed_placement.py` to attach its completed outcomes to the
selected oracle policy. For example, after copying its component to
`temp/santorini_v4_placement/run13-component.npz`:

```bash
.venv/bin/python build_santorini_v4_mixed_placement.py \
  --oracle-component temp/santorini_v4_placement/oracle-t25.npz \
  --run13-component temp/santorini_v4_placement/run13-component.npz \
  --policy-mode oracle \
  --output temp/santorini_v4_placement/oracle-t25-policy.npz
```

The resulting production component has SHA-256
`d7c9f48c23feb076a0ecb596db2d2acd5359a90486d751b86602445fc06414c9`.
Its T25 oracle and Run13 policies differ materially (mean TV 0.486, maximum TV
0.992), so the selection was made by play rather than target similarity.

## Placement-only teacher selection

The comparison samples each teacher's four sequential placement actions, swaps
teacher seats within every paired block, and then lets the same deterministic
santorini-ai instance play both sides from the completed opening. The engine TT
is reset for every opening. This isolates placement policy from standard-play
network strength.

The initial T300 screen made Run13 look clearly better, but a temperature sweep
showed that conclusion was caused by an overly diffuse oracle distribution.
At 64 paired blocks per matchup, Run13 scored 53.125% against oracle T25 with a
paired bootstrap 95% interval of 45.3125%-60.9375%. Their overall equal-opponent
scores in the T25/T50/T75/T100 round robin were 56.4453% and 55.8594%,
respectively. They are statistically tied.

The final three-way check compared Run13, pure oracle T25, and their 50/50
policy blend. The blend's overall point estimate was highest at 53.9063%, but
its direct advantages were inconclusive: oracle T25 scored 46.875% against the
blend (95% interval 38.2813%-55.4688%), and Run13 scored 45.3125% (36.7188%-
53.9063%). Greedy placement produced the same D4 opening for every teacher and
all greedy matchups split 1-1, so the useful difference is in the sampled
distribution rather than the top action.

The predeclared preference was pure oracle if it matched the blend. It did, so
P1c freezes oracle T25. This removes Run13 policy weight while respecting the
evidence: the tournament does not prove either pure teacher is superior, and it
does not justify three full GPU pretraining arms. The low-temperature and final
tournament reports are in
`temp/santorini_v4_placement_tournament_low_temperature/report.json` and
`temp/santorini_v4_placement_tournament_final/report.json`; neither touched the
final test set.

The plan builder rejects a placement component that does not explicitly declare
real completed outcomes, so the oracle-only raw component cannot accidentally
train the main value head on its zero placeholder. The frozen plan has
10,640,649 draws per epoch and SHA-256
`beb91cf9d068bc6bb7045f944002326a49273daa9646e2c4d89756633b36eec4`.
It is index-compatible with the selected T25 component and remains unchanged.

## P1c pretraining Kaggle job

The selected four-epoch run takes an estimated 6.4 P100 GPU-hours at the
measured 6x192 throughput. The placement tournament avoids spending roughly 19
GPU-hours on three full arms. Build and upload the dataset:

```bash
.venv/bin/python prepare_santorini_v4_p1c_pretraining_bundle.py
```

Upload `temp/santorini_v4_p1c_pretraining_bundle.zip` as a Kaggle dataset. Its
SHA-256 is
`93bfff4716213c26c5532bfae47ecd4c966ce85688ba2d65b5098df7b5c68969`.
Use a fresh P100 notebook with only this dataset attached. Kaggle exposes the
archive contents already extracted. Run:

```python
import subprocess
import sys
from pathlib import Path

entries = list(Path("/kaggle/input").rglob("run_santorini_v4_p1c_pretraining_kaggle.py"))
assert len(entries) == 1, entries
subprocess.run([sys.executable, str(entries[0]), "--arm", "oracle_t25"], check=True)
```

The job verifies all input hashes before training and writes beneath
`/kaggle/working/santorini_v4_p1c_pretraining/oracle_t25/`, including best and
final checkpoints, `results.json`, and `job-contract.json`.

After it returns, inspect the four-epoch curves and execute the standard/full-
game Gate G1 at 96/128 simulations.
