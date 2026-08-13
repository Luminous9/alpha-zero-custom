# P2 Arm D continuation through iteration 11

## Decision

The four-arm diagnostic completed from the common accepted iteration-1 state.
All arms ran iterations 2-4 at fixed 2x reuse and passed their declared safety
contracts.

| Arm | Frozen-objective cumulative delta | Standard vs iteration 1 | Placement-inclusive vs iteration 1 |
| --- | ---: | ---: | ---: |
| A: global 3e-4, pure `z` | +0.02854 | 24-16 (60.0%) | 19-21 (47.5%) |
| B: global 1e-4, pure `z` | +0.00903 | 21-19 (52.5%) | 8-32 (20.0%) |
| C: differential LR, pure `z` | +0.00930 | 22-18 (55.0%) | 28-12 (70.0%) |
| D: global 1e-4, value bridge | **+0.00654** | **26-14 (65.0%)** | **31-9 (77.5%)** |

Arm D's paired records were 6/14/0 in standard play and 12/7/1 with
placement included. The corresponding paired bootstrap intervals were 55-75%
and 65-90%. It was also robust from both seats. D is therefore the selected P2
production continuation; C remains the fallback diagnostic branch.

The B-versus-D contrast isolates the value bridge at the same global 1e-4
learning rate. D improved by 12.5 points standard and 57.5 points
placement-inclusive on the common seed blocks. This is strong evidence that the
abrupt pure-`z` transition was harmful even after reducing the learning rate.
Ordinary value training/validation losses are not directly comparable for D
because its declared target is blended during the bridge.

## Frozen continuation contract

Resume the complete Arm D iteration-4 state, including its four replay windows,
AdamW moments, RNG state, oracle-ratchet history, iteration-1 frozen-teacher
reference, and current beta. Run iterations 5-11 with:

- 240 games per iteration at the unchanged 96/32 Gumbel search contract;
- fixed 2x fresh-data reuse and no warm-up or LR schedule;
- global trunk/policy/value LR 1e-4 and weight decay 1e-4;
- the same 5k-node ladder-v2 10% oracle sparring;
- the +0.05 step and +0.10-from-iteration-1 cumulative teacher gates; and
- the unchanged seam sentinel and 55% rolling oracle-ratchet pause.

The absolute bridge schedule is:

| Iteration | beta on self-play `z` |
| ---: | ---: |
| 5 | 0.500000 |
| 6 | 0.583333 |
| 7 | 0.666667 |
| 8 | 0.750000 |
| 9 | 0.833333 |
| 10 | 0.916667 |
| 11 | 1.000000 |

The runner saves training, inference, and replay snapshots after every completed
iteration. A declared safety trigger stops the run early after saving. Whether
it reaches iteration 11 or pauses early, it then runs 40 paired standard plus 40
paired placement-inclusive games against both iteration 4 and iteration 1.
Iteration 11 is a mandatory review point; this bundle cannot continue beyond
it.

## Kaggle launch

Upload `temp/santorini_v4_p2_d_continuation_5_11_bundle.zip` as a Kaggle dataset
and attach only that extracted continuation dataset to a P100 notebook. Kaggle
extracts the ZIP; do not open it as a tarball or unpack it in the notebook.

```python
from pathlib import Path
import subprocess
import sys

matches = list(Path("/kaggle/input").rglob(
    "run_santorini_v4_p2_d_continuation_kaggle.py"
))
if len(matches) != 1:
    raise RuntimeError(f"Expected one continuation runner, found: {matches}")

subprocess.run([sys.executable, str(matches[0])], check=True)
```

Results are written to:

```text
/kaggle/working/santorini_v4_p2_d_continuation/
```

Download the entire directory. The final result contract is
`p2-d-continuation-contract.json`; milestone arenas are `vs-iteration4.json`
and `vs-iteration1.json`.

## Bundle identity

The bundle is 112,131,832 bytes with SHA-256
`2d89bcde6a8af1001d1a5a7a45c80b73e7306c448baa63c08723409fe0169b96`.
Its adjacent `.report.json` records every input digest and the full frozen
protocol.
