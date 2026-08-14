# P2 4x dose branch: iterations 12-17

## Decision

Start from the accepted iteration-11 training checkpoint and its 11-window
replay. Run exactly six new iterations, 12 through 17, at fixed 4x fresh-data
reuse. Do not automatically escalate to 6x or 8x at the end of this job.

This is a single-variable dose experiment. It keeps the production architecture,
240 games per iteration, global AdamW learning rate `1e-4`, pure outcome-`z`
value target, 96/32 playout-cap search, and 10% 5k-node ladder-v2 oracle
sparring unchanged. The ancestor used 2x reuse; iteration 12 is the declared
2x-to-4x transition.

The branch deliberately disables the old endpoint arenas. Those use a small,
fixed suite that is no longer an adequate promotion instrument. After iteration
17, evaluate strength separately on expanded fresh suites, including the 20k,
50k, and 100k external anchors. Use at least 40 and preferably 60 paired
openings for the stronger oracle rungs, and preserve a fixed longitudinal suite
alongside rotated fresh neural self-match suites.

## In-loop controls

All existing controls remain active:

- frozen teacher objective: pause above a `+0.05` one-iteration increase or a
  `+0.10` cumulative increase from the iteration-1 reference;
- oracle ladder ratchet: pause when its rolling score reaches 55%;
- prior-to-search-target KL: warn after three eligible standard-play iterations
  below `0.15` (this does not pause); and
- canonical seam telemetry every iteration.

The run also adds the frozen A1 deep-value audit as standing telemetry. Its 480
standard-play positions are D4-unique, stratified by replay-window band and game
stage, and exclude overlaps with the held-out final corpus. Labels are the
already-cached cold-TT 250k-node oracle values. Every new checkpoint is compared
on exactly those positions with the frozen iteration-11 predictions.

One iteration warns if any of these paired point changes from iteration 11 is
reached:

- overall Pearson: `-0.02` or worse;
- overall MSE: `+0.015` or worse;
- windows 9-11 Pearson: `-0.04` or worse; or
- windows 9-11 MSE: `+0.02` or worse.

Two consecutive warning iterations pause after the new checkpoint, replay, and
telemetry have been written. The warning streak and suite fingerprint are saved
inside the training checkpoint, so resuming does not reset the guard. The
telemetry also records early/middle/late and all four replay-window slices plus
stratified paired bootstrap intervals. The engine values remain proxy labels,
so promotion should use the paired drift and playing-strength suites together,
not treat the absolute engine MSE as ground truth.

## Kaggle artifacts

- Notebook: `santorini/v4_p2_training_kaggle.ipynb`
- Runtime upload: `temp/santorini_v4_p2_runtime_bundle.zip`
- Resume checkpoint: `checkpoint_11-training.pth.zip`
- Resume replay: `checkpoint_11.examples.zip`

The notebook is pinned to start iteration 11, request six iterations, use 4x
reuse, and end at iteration 17. Its final cell packages the complete output
directory into one ZIP. Kaggle extracts attached datasets automatically; the
notebook does not try to unpack the uploaded runtime or resume files.

Runtime bundle SHA-256:

`3612a2f51da5aeca1c86c82e49a6aebdc9db98116ffd43b2f76f1a9757b052da`

The runtime includes the frozen deep-value suite with SHA-256:

`0323a03302862522928568a1076cdeac52e66d3f357ea01700cba10c305c1af2`

The telemetry implementation was checked against the actual iteration-11
checkpoint. The reproduced overall and windows-9-11 Pearson/MSE deltas were all
below `5e-7` in magnitude, as expected for the frozen reference.
