# P2 entry and P100 smoke

The local P2 entry gates are complete:

- P1c-to-trainable-V4 migration preserves policy and value outputs exactly.
- Deterministic complete-search D4 checks pass on asymmetric, symmetric, and
  tactical roots.
- Compact replay schema 2 preserves per-position source metadata and still
  loads schema-1 replay.
- A real two-game, paired-seat, 100k-oracle Gumbel smoke completed one optimizer
  step and checkpoint/replay/telemetry round trip. Only neural decisions were
  stored; every record carried the declared source, rung, binary digest,
  opening, seat, and stage.

The representative P100 job has two arms from the exact same migrated handoff:

- `ordinary`: 240 ordinary self-play games.
- `mixed`: 216 ordinary games plus 24 paired 100k-node sparring games.

Both use FP32 canonical inference, 128-game ordinary concurrency, the 96/32
playout-cap contract, 16x fresh-data replay reuse, and the frozen seam sentinel.
Disagreement starts and the auxiliary oracle head are disabled.

Upload `temp/santorini_v4_p2_smoke_bundle.zip` as a Kaggle dataset. Kaggle
extracts the upload automatically; do not try to open it as a tarball. Run one
arm per session with:

```python
from pathlib import Path
import subprocess
import sys

matches = list(Path("/kaggle/input").rglob("run_santorini_v4_p2_smoke_kaggle.py"))
assert len(matches) == 1, matches
subprocess.run([sys.executable, str(matches[0]), "--arm", "mixed"], check=True)
```

Replace `mixed` with `ordinary` for the control. Results are written under
`/kaggle/working/santorini_v4_p2_smoke/<arm>/`. Retain the entire arm directory,
especially `p2-smoke-contract.json`, `telemetry/telemetry.jsonl`,
`latest-training.pth.tar`, and `latest.examples.npz`.

The job rejects a changed checkpoint/seam suite, an incomplete 240-game run, an
incorrect sparring-game count, or missing resumable outputs.
