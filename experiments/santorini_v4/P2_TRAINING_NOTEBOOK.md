# Reusable V4 P2 Kaggle training notebook

## Purpose

`santorini/v4_p2_training_kaggle.ipynb` replaces one-off continuation bundles
for the selected V4 P2 baseline. The code and immutable fixed inputs live in one
runtime dataset. Each future notebook run attaches that unchanged runtime plus
the previous run's output dataset, requests a number of new iterations, and
produces the next resumable output.

The runtime bundle is `temp/santorini_v4_p2_runtime_bundle.zip`. It contains the
frozen Python sources, Linux oracle, seam suite, P1c value anchor, and iteration-1
longitudinal arena anchor. It intentionally contains no mutable resume state.

## Routine workflow

1. Upload the runtime ZIP once as a Kaggle dataset. Kaggle extracts it; no
   notebook unpacking is required.
2. Import `santorini/v4_p2_training_kaggle.ipynb` into Kaggle and select a P100.
3. Attach the runtime dataset and the output dataset from the previous V4 run.
4. In the parameters cell, set `NUM_ITERATIONS` and a new `RUN_NAME`.
5. Run all cells. The final cell packages the complete output as the single
   `/kaggle/working/RUN_NAME.zip` download. Add that ZIP as the resume dataset
   for the next session; Kaggle extracts it automatically.

The notebook includes explicit GPU inspection, P100 enforcement, source setup,
and dependency validation. Current Kaggle images ship a CUDA 12.8 PyTorch wheel
without P100/Pascal `sm_60` kernels. The first setup cell therefore detects a
P100 before importing Torch and replaces that wheel with the same PyTorch
release's official CUDA 12.6 build. Enable Kaggle Internet for that installation;
the following CUDA smoke test verifies `sm_60` is present before training.
`SOURCE_MODE="bundled"` remains the routine source default: it runs the
manifest-validated source embedded in the runtime without a repository clone.
`SOURCE_MODE="git"` clones `alpha-zero-custom` at `REPOSITORY_REF`,
records the resolved commit, and refuses a commit mismatch. Use it only after
the intended training changes have been committed and pushed. The dependency
cell deliberately does not install the repository's legacy TensorFlow-era
`requirements.txt`; it checks the current PyTorch runtime and installs only
missing non-Torch packages.

The notebook normally auto-selects the unique highest-iteration
`latest-training` checkpoint under `/kaggle/input` and its sibling compact
replay. Explicit checkpoint/replay paths remain available for an intentional
rollback or branch. If two different checkpoints tie for the highest iteration,
auto-discovery refuses to guess.

The initial post-bridge request through iteration 14 is complete. Do not use
auto-discovery to continue from iteration 14 merely because it is the highest
attached checkpoint. Iteration 11 remains the production head; iteration 14 is
a diagnostic branch pending a larger fresh-suite arena. For an intentional new
branch, set `RESUME_CHECKPOINT` and `RESUME_REPLAY` explicitly so the notebook
cannot silently choose the non-promoted iteration-14 state. The evidence and
next gate are in `P2_PURE_Z_12_14.md`.

## Frozen normal-training contract

The reusable runner keeps the selected P2 baseline fixed:

- ordinary 6x192 V4 with canonical-D4 inference;
- 240 games per iteration;
- 96 full / 32 fast Gumbel search with 25% full-search probability;
- fixed 2x fresh-data reuse, global AdamW LR 1e-4, and no LR schedule;
- a rolling 20-iteration replay window;
- 10% 5k-node ladder-v2 oracle sparring;
- the frozen seam suite;
- +0.05 teacher-objective step and +0.10-from-iteration-1 cumulative review
  gates; and
- the 55% rolling oracle-ratchet review gate.

Every new iteration also reports the generating network's raw prior versus its
stored full-search target, before that target is trained:

- count, mean, median, p10, and p90 of `KL(target || prior)` separately for
  placement and standard positions;
- total variation and policy-argmax agreement; and
- explicit counts excluded because an exact tactical shortcut supplied the
  target or because a raw prior was unavailable.

The standard-play mean has a non-gating low-signal watch at 0.15. Three
consecutive eligible iterations (at least 256 measured standard positions
each) emit `prior_target_kl_warning`. The initial reference and warning streak
are resumable checkpoint metadata. This threshold is deliberately conservative:
the retrospective iteration-1 standard strata were about 0.77-0.82, and the
failed iteration-8 checkpoint was still about 0.42-0.53. A warning requests a
matched 96-vs-higher-search bake-off; it does not by itself authorize a budget
change. This metric directly covers the 96-simulation full-search teacher. It
does not isolate whether the 32-simulation fast trajectory policy has become
too weak, so that still needs an arena/search-budget diagnostic.

The training implementation is not limited to a small number of iterations.
`NUM_ITERATIONS` controls only the notebook's requested stopping point. The
runner invokes one fully resumable iteration at a time so it can validate the
telemetry chain, save state, and honor safety pauses. The replay window naturally
fills to 20 iterations and then rolls forward; fixed fresh-data reuse keeps the
optimizer dose tied to new data rather than growing with replay size.

This is the ready normal P2 baseline, not authorization for every deferred plan
feature. Disagreement-seeded starts, an auxiliary oracle-value head, higher
reuse, a different learning rate, and T4x2 remain separate experiments. A
declared safety pause still requires review even if `NUM_ITERATIONS` requested a
longer run.

## Outputs and snapshots

The output root always retains `latest-training.pth.tar`, `latest.pth.tar`,
`latest.examples.npz`, telemetry, and `p2-training-contract.json`. Selected
snapshots use ordinary flat names such as:

```text
checkpoint_12-training.pth.tar
checkpoint_12.pth.tar
checkpoint_12.examples.npz
```

`SNAPSHOT_INTERVAL=1` preserved all three checkpoints in the first pure-`z`
block. It may be increased for a future long stable run to reduce output size.
These files are not given special “protected” semantics; the runs saved their
snapshots correctly.
The final ZIP uses stored entries rather than recompressing model/replay
archives, so packaging is fast while preserving every output byte.

Unless disabled, the endpoint plays paired standard and placement-inclusive
arenas against both the run's starting checkpoint and iteration 1. The former
is the primary block-progress signal; the latter is the longitudinal anchor.

## Runtime identity

<!-- RUNTIME_IDENTITY_START -->
The current runtime bundle is 65,731,236 bytes with SHA-256
`1be4056bac21ecdbc4f9bc25693eff7fe373879e8ee2af9da6f7776a00e405f4`.
<!-- RUNTIME_IDENTITY_END -->
The adjacent `.report.json` records every bundled source and fixed-input digest.
