# Santorini Oracle Experiments

This package preserves oracle-assisted training experiments that were useful
for learning what not to promote, while keeping them out of the supported
top-level workflow.

## Layout

- `RESULTS.md` records the observed arena outcomes and conclusions.
- `legacy/generate_santorini_oracle_replay.py` generates hard best-move targets.
- `legacy/generate_santorini_oracle_soft_replay.py` generates confidence-filtered
  ranked-root soft targets.

Run the legacy generators from the repository root as modules:

```bash
.venv/bin/python -m experiments.santorini_oracle.legacy.generate_santorini_oracle_replay --help
.venv/bin/python -m experiments.santorini_oracle.legacy.generate_santorini_oracle_soft_replay --help
```

Their output formats remain compatible with `finetune_santorini_oracle.py`, but
neither generator is recommended as a default training path. They are retained
for reproducibility and for controlled future ablations.

Reusable position canonicalization, policy conversion, confidence metrics, and
parallel oracle-process management live in `santorini/OracleResearch.py` so the
active diagnostic tools do not depend on legacy experiment scripts.
