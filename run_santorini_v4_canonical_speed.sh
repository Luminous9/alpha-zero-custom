#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
CHECKPOINT="temp/santorini_v4_screen_1m/ordinary_6x192_13_global_blend.pt"
ENGINE_CORPUS="temp/santorini_v4_scaled/engine-corpus.npz"
RUN13_COMPONENT="temp/santorini_v4_scaled/run13-component.npz"
SELECTION_PLAN="temp/santorini_v4_scaled/selection-3k.npz"
OUTPUT_DIR="temp/santorini_v4_canonical_speed"

mkdir -p "${OUTPUT_DIR}"

# Naming the same checkpoint twice gives a direct apples-to-apples measurement
# of the ordinary wrapper with and without exact D4 canonicalization.
"${PYTHON_BIN}" benchmark_santorini_v4_inference.py \
  --checkpoint "o6_uncanonicalized=${CHECKPOINT}" \
  --checkpoint "o6_canonical=${CHECKPOINT}" \
  --canonical-d4 o6_canonical --end-to-end-wrapper \
  --batch-sizes 1 8 32 64 \
  --warmup-iterations 10 --examples-per-case 2048 --minimum-iterations 20 \
  --agreement-examples 0 --device cuda \
  --engine-corpus "${ENGINE_CORPUS}" \
  --run13-component "${RUN13_COMPONENT}" \
  --selection-plan "${SELECTION_PLAN}" \
  --output-dir "${OUTPUT_DIR}"
