#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
DATA_DIR="${DATA_DIR:-temp/santorini_v4_scaled}"
P1C_DIR="${P1C_DIR:-temp/santorini_v4_p1c}"
OUTPUT_DIR="${OUTPUT_DIR:-temp/santorini_v4_p1c_results}"

mkdir -p "${OUTPUT_DIR}"

"${PYTHON_BIN}" screen_santorini_v4_bootstrap.py \
  --engine-corpus "${DATA_DIR}/engine-corpus-1m-train.npz" \
  --selection-engine-corpus "${DATA_DIR}/engine-corpus.npz" \
  --run13-component "${DATA_DIR}/run13-component.npz" \
  --placement-component "${P1C_DIR}/placement-component.npz" \
  --train-plan "${P1C_DIR}/train-plan.npz" \
  --selection-plan "${DATA_DIR}/selection-3k.npz" \
  --output-dir "${OUTPUT_DIR}" \
  --epochs 4 --batch-size 256 \
  --learning-rate 0.0003 --weight-decay 0.0001 \
  --policy-weight 0.25 --policy-epsilon 0.05 \
  --alpha-boot 0.5 --score-temperature 261.8 \
  --stage-reliability 0.25 0.75 1.0 \
  --seed 20260812 --device cuda --data-loading streaming \
  --configs ordinary_6x192_13_global_blend
