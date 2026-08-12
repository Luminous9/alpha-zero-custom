# P2 entry, P100 smoke, and transition recheck

The implementation gates are complete: P1c migration preserves outputs,
deterministic complete-search D4 checks pass, compact replay schema 2 preserves
source metadata, paired oracle processes reset at every game boundary, and the
frozen seam diagnostic runs inside the training loop.

## First P100 smoke: informative, not a production checkpoint

Two 240-game arms started from the identical P1c handoff:

- `ordinary`: 240 ordinary games.
- `mixed`: 216 ordinary games and 24 paired 100k ladder-v1 sparring games.

Both completed valid checkpoint/replay round trips. Ordinary took 354.0 seconds
and mixed took 364.4 seconds, so replacing 10% of games added only 2.9% total
wall time. The seam exposure contrast remained clean in both arms. The mixed
arm did create 50% more inference batches because its four-game sparring worker
batch was much smaller than the 128-game ordinary batch; this is an optimization
opportunity, not an entry blocker.

The smoke also found two blockers before a production lineage was created:

1. The live 100k player scored only 2-22. Ladder v1 had been calibrated against
   deterministic 96-simulation search, not the actual stochastic 96/32
   playout-cap player. Exact live-policy calibration now freezes **5k nodes as
   ladder version 2**; its 40-game confirmation score was 17-23 (42.5%).
2. Applying 16x fresh-data reuse to the first small replay window caused a
   cold-start optimization shock. The ordinary and mixed frozen teacher
   objectives moved by +0.346/+0.349, and their equal-96 standard scores against
   P1c were 35.0% and 22.5%. An isolated retrain of the exact ordinary replay at
   2x reuse scored 22-18 (55.0%, paired interval 45-65%) and moved the frozen
   objective only +0.0106. P2 now ramps reuse over absolute iterations 1-8 as
   2x, 4x, 6x, 8x, 10x, 12x, 14x, 16x.

Do not continue from either first-smoke checkpoint. They are measurements only;
the corrected lineage starts again from `p2-start.pth.tar`.

Full measurements and hashes are in
`experiments/santorini_v4/P2_SMOKE_RESULTS.md`; live ladder records are in
`experiments/santorini_v4/P2_ORACLE_SWEEP.md`.

## Revised transition smoke

Upload `temp/santorini_v4_p2_smoke_bundle.zip` (SHA-256
`954e946a7a9efb901372ca6337267814adae43a98e9b96ae0e0837079ff5537f`)
as a Kaggle dataset and run only the corrected `transition` arm:

```python
from pathlib import Path
import subprocess
import sys

matches = list(Path("/kaggle/input").rglob("run_santorini_v4_p2_smoke_kaggle.py"))
assert len(matches) == 1, matches
subprocess.run(
    [sys.executable, str(matches[0]), "--arm", "transition"],
    check=True,
)
```

Kaggle extracts the uploaded zip automatically; do not try to open it as a
tarball. Results are written to
`/kaggle/working/santorini_v4_p2_smoke/transition/`. Retain the entire folder,
especially the contract, telemetry, resumable checkpoint, inference checkpoint,
and compact replay.

The transition arm uses 240 games, 10% paired 5k ladder-v2 sparring, FP32 P100
inference, the frozen 96/32 search contract, and iteration-one 2x reuse from the
declared eight-iteration warm-up. It includes the Linux oracle and requires no
Kaggle Cargo installation.

After download, run the same local 40-game equal-96 P1c comparison. Start the
production P2 lineage only if replay/rung provenance is correct, the seam
contrast remains clean, the frozen-objective shift remains small, and the
strength result is consistent with no material first-iteration regression.
Neither final-test data nor final arena seeds are used.
