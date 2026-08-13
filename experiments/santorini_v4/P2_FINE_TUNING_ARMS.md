# P2 fine-tuning arm execution

The local signal diagnostic selects one shared, parameterized Kaggle package
with one arm per GPU session. The bundle is
`temp/santorini_v4_p2_diagnostic_bundle.zip`. Kaggle extracts the uploaded
dataset; the notebook must not try to open it as a tarball or manually unpack
it.

## Frozen arms

| Arm | Trunk LR | Policy-head LR | Value-head LR | Main value target |
| --- | ---: | ---: | ---: | --- |
| A | 3e-4 | 3e-4 | 3e-4 | pure `z` control |
| B | 1e-4 | 1e-4 | 1e-4 | pure `z` |
| C | 1e-4 | 3e-4 | 3e-5 | pure `z` |
| D | 1e-4 | 1e-4 | 1e-4 | frozen P1c bridge to `z` |

Arm D uses beta 0.25 at iteration 2, 0.333333 at iteration 3, and
0.416667 at iteration 4. It would reach 1.0 at iteration 11, but this diagnostic
ends at iteration 4. Arm C preserves and migrates the accepted AdamW state; it
does not reset optimizer moments when introducing named parameter groups.

Every arm starts from the same accepted iteration-1 checkpoint, replay, RNG
state, and optimizer state. It runs iterations 2-4 at fixed 2x fresh-data reuse,
saves resumable/inference/replay snapshots after every iteration, and then runs
40 paired standard plus 40 paired placement-inclusive games against iteration
1. A +0.05 teacher-objective step, +0.10 cumulative rise, or oracle-rung ratchet
pauses the arm after saving and still runs the arena against the last completed
checkpoint.

## Kaggle launch

Attach the same extracted bundle dataset to four P100 GPU notebooks. Run A and
B concurrently, then C and D. Each notebook needs only this cell, changing the
arm letter:

```python
from pathlib import Path
import subprocess
import sys

matches = list(Path("/kaggle/input").rglob("run_santorini_v4_p2_diagnostic_kaggle.py"))
if len(matches) != 1:
    raise RuntimeError(f"Expected one diagnostic runner, found: {matches}")

subprocess.run(
    [sys.executable, str(matches[0]), "--arm", "A"],
    check=True,
)
```

Use `--arm B`, `--arm C`, or `--arm D` in the other notebooks. The runner
refuses non-P100 GPUs, altered inputs, unknown arms, configuration drift, and a
nonempty output directory. Results appear under:

```text
/kaggle/working/santorini_v4_p2_diagnostic/arm_A/
```

The directory contains `p2-diagnostic-contract.json`, `vs-iteration1.json`,
the latest artifacts, and iteration-specific checkpoint/replay snapshots.

## Shared bundle identity

The generated bundle is 66,194,470 bytes with SHA-256
`d9855737202bc938abc5964fa9f6c23e29d2737cae09d7783cfe7ea9f14f53ab`.
Regenerating the bundle after a source change intentionally changes this hash;
use the adjacent `.report.json` as the authoritative upload contract.

## Results and selection

All four arms completed iterations 2-4. Arm D had the smallest cumulative
frozen-objective movement (+0.00654) and was the only arm whose paired lower
bound exceeded 50% in both iteration-4 strength gates: 26-14 standard with a
55-75% paired interval, and 31-9 placement-inclusive with a 65-90% interval.
Arm C was also healthy at 22-18 and 28-12, and remains the fallback branch.

The controlled B-versus-D contrast at identical global 1e-4 LR strongly favors
the value bridge, especially in placement-inclusive play (20.0% versus 77.5%).
Select D as the P2 continuation and retain iteration 1 as the immutable rollback
anchor. The iterations 5-11 contract and Kaggle bundle are documented in
`P2_D_CONTINUATION.md`.
