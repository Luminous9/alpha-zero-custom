#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
DATA_DIR="temp/santorini_v4_scaled"
CONTROL_DIR="temp/santorini_v4_screen_1m"
OUTPUT_DIR="temp/santorini_v4_value_ablation_1m"

TRAIN_ENGINE="${DATA_DIR}/engine-corpus-1m-train.npz"
SELECTION_ENGINE="${DATA_DIR}/engine-corpus.npz"
RUN13_COMPONENT="${DATA_DIR}/run13-component.npz"
TRAIN_PLAN="${DATA_DIR}/train-1m.npz"
SELECTION_PLAN="${DATA_DIR}/selection-3k.npz"
GLOBAL_CHECKPOINT="${CONTROL_DIR}/ordinary_6x192_13_global_blend.pt"
WINNER_CHECKPOINT="${OUTPUT_DIR}/ordinary_6x192_13_winner.pth.tar"

mkdir -p "${OUTPUT_DIR}"

"${PYTHON_BIN}" screen_santorini_v4_bootstrap.py \
  --engine-corpus "${TRAIN_ENGINE}" \
  --selection-engine-corpus "${SELECTION_ENGINE}" \
  --run13-component "${RUN13_COMPONENT}" \
  --train-plan "${TRAIN_PLAN}" \
  --selection-plan "${SELECTION_PLAN}" \
  --output-dir "${OUTPUT_DIR}" \
  --epochs 4 --batch-size 256 \
  --learning-rate 0.0003 --weight-decay 0.0001 \
  --policy-weight 0.25 --policy-epsilon 0.05 \
  --alpha-boot 0.5 --score-temperature 261.8 \
  --stage-reliability 0.25 0.75 1.0 \
  --seed 20260812 --device cuda --data-loading streaming \
  --configs ordinary_6x192_13_winner

"${PYTHON_BIN}" compare_santorini_v4_value_targets.py \
  --winner-checkpoint "${WINNER_CHECKPOINT}" \
  --blend-checkpoint "${GLOBAL_CHECKPOINT}" \
  --engine-corpus "${SELECTION_ENGINE}" \
  --run13-component "${RUN13_COMPONENT}" \
  --selection-plan "${SELECTION_PLAN}" \
  --batch-size 256 --bootstrap-samples 10000 --seed 20260820 \
  --policy-epsilon 0.05 --alpha-boot 0.5 --score-temperature 261.8 \
  --stage-reliability 0.25 0.75 1.0 \
  --noninferiority-margin 0.01 --device cuda \
  --output "${OUTPUT_DIR}/selection-comparison.json"

run_arena() {
  local gate="$1"
  "${PYTHON_BIN}" arena_santorini_v4_selection.py \
    --player1 "${GLOBAL_CHECKPOINT}" --player1-name global_blend \
    --player2 "${WINNER_CHECKPOINT}" --player2-name winner_only \
    --gate "${gate}" --games 40 --simulations 96 --batch-size 32 \
    --search-mode gumbel --gumbel-scale 0 --placement-gumbel-scale 1.5 \
    --player1-root-symmetries 1 --player2-root-symmetries 1 \
    --player1-canonical-d4 --player2-canonical-d4 \
    --inference-cache-size 4096 --device cuda --seed 20260814 \
    --engine-corpus "${SELECTION_ENGINE}" \
    --run13-component "${RUN13_COMPONENT}" \
    --selection-plan "${SELECTION_PLAN}" \
    --output "${OUTPUT_DIR}/arena_global_blend_vs_winner_${gate}.json"
}

run_arena standard
run_arena full

"${PYTHON_BIN}" summarize_santorini_v4_value_ablation.py \
  --comparison "${OUTPUT_DIR}/selection-comparison.json" \
  --standard-arena "${OUTPUT_DIR}/arena_global_blend_vs_winner_standard.json" \
  --full-arena "${OUTPUT_DIR}/arena_global_blend_vs_winner_full.json" \
  --bootstrap-samples 10000 --seed 20260821 \
  --output "${OUTPUT_DIR}/decision.json"
