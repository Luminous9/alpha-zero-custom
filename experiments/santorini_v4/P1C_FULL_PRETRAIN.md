# P1c full corpus and pretraining

Status: in progress. The corpus contract and santorini-ai placement teacher are
implemented; the exhaustive Run13 continuation job supplies the shared
completed-outcome target and comparison policy.

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
the parent boundary's stabilizer are averaged before softmax. Pilot comparisons
froze 50,000 nodes per pair and policy temperature 300. At temperature 300,
10k-to-50k mean policy TV was 0.099 over the three deeper pilot roots; temperature
400 reduced it to 0.075 but flattened the signal to 0.072 nats of KL from
uniform, versus 0.121 at temperature 300. Oracle static values remain telemetry
only and are never substituted for a completed outcome.

The Run13 labeler starts four continuations from every sequential orbit at 96
simulations, uses full search for every remaining placement choice, and combines
those fresh targets with replay observations. Its settings match Run13
pretraining behavior: Gumbel search, placement scale 1.5, all eight root
orientations during placement, policy-target temperature 1, and the normal
25%-at-96/otherwise-32 playout cap after setup.

Three components are produced: Run13-only policy, santorini-ai-only policy, and
a 50/50 policy blend. Their boards, Run13 completed-game `z`, sampling weights,
and epoch plan are identical; only policy changes. The pure oracle arm is
preferred if it matches the mixed arm. Standard engine records retain the
selected global value blend (`alpha_boot=0.5`, `T=261.8`).

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
  --policy-temperature 300
```

The records contain every raw root response and make policy-temperature changes
free of additional engine searches. The component must report 50/50 boundaries,
960 unique positions, 1/6/49/904 coverage, reset-per-root-move TT, and no
completed outcomes. The frozen records and component SHA-256 digests are
`5c4dcb9f89a266857af7bf9bd97fb8c38d734e108567128ae063f4c9b501d691`
and `524dcb6284f2a0604c85f26017c52cb179416a6f75e3de68677a79e79aa362e2`.

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

After the Run13 output returns, use
`build_santorini_v4_mixed_placement.py` to materialize the three policy arms.
For example, after copying its component to
`temp/santorini_v4_p1c_placement/run13-component.npz`:

```bash
.venv/bin/python build_santorini_v4_mixed_placement.py \
  --oracle-component temp/santorini_v4_oracle_placement/oracle-component.npz \
  --run13-component temp/santorini_v4_p1c_placement/run13-component.npz \
  --policy-mode oracle \
  --output temp/santorini_v4_p1c_placement/oracle-policy.npz

.venv/bin/python build_santorini_v4_mixed_placement.py \
  --oracle-component temp/santorini_v4_oracle_placement/oracle-component.npz \
  --run13-component temp/santorini_v4_p1c_placement/run13-component.npz \
  --policy-mode run13 \
  --output temp/santorini_v4_p1c_placement/run13-policy.npz

.venv/bin/python build_santorini_v4_mixed_placement.py \
  --oracle-component temp/santorini_v4_oracle_placement/oracle-component.npz \
  --run13-component temp/santorini_v4_p1c_placement/run13-component.npz \
  --policy-mode blend \
  --oracle-policy-weight 0.5 \
  --output temp/santorini_v4_p1c_placement/blended-policy.npz
```

The plan builder rejects a placement component that does not explicitly declare
real completed outcomes, so the oracle-only raw component cannot accidentally
train the main value head on its zero placeholder. Then build one frozen plan
with `build_santorini_v4_p1c_plan.py` and reuse it unchanged for each arm, run
`run_santorini_v4_p1c_pretraining.sh` on P100, inspect the four-epoch curve,
and execute the standard/full-game Gate G1 at 96/128 simulations. The standard
gate should be insensitive to the placement arm; the paired full-game gate is
the selection measurement.
