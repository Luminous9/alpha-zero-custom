#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
CHECKPOINT_DIR="temp/santorini_v4_screen_1m"
OUTPUT_DIR="temp/santorini_v4_selection_1m"
ENGINE_CORPUS="temp/santorini_v4_scaled/engine-corpus.npz"
RUN13_COMPONENT="temp/santorini_v4_scaled/run13-component.npz"
SELECTION_PLAN="temp/santorini_v4_scaled/selection-3k.npz"

O10="${CHECKPOINT_DIR}/ordinary_10x128_13_global_blend.pth.zip"
O6="${CHECKPOINT_DIR}/ordinary_6x192_13_global_blend.pth.zip"
E="${CHECKPOINT_DIR}/equivariant_e_13_global_blend.pth.zip"

mkdir -p "${OUTPUT_DIR}"

"${PYTHON_BIN}" diagnose_santorini_v4_canonical_seams.py \
  --checkpoint "e=${E}" \
  --checkpoint "o6=${O6}" \
  --checkpoint "o10=${O10}" \
  --engine-corpus "${ENGINE_CORPUS}" \
  --run13-component "${RUN13_COMPONENT}" \
  --selection-plan "${SELECTION_PLAN}" \
  --output "${OUTPUT_DIR}/canonical-seams.json" \
  --device cpu --batch-size 256 --bootstrap-samples 10000 --seed 20260818

"${PYTHON_BIN}" benchmark_santorini_v4_inference.py \
  --checkpoint "o10=${O10}" \
  --checkpoint "o6=${O6}" \
  --checkpoint "e=${E}" \
  --batch-sizes 1 8 32 64 128 192 \
  --warmup-iterations 10 --examples-per-case 2048 --minimum-iterations 20 \
  --agreement-examples 512 --device cuda \
  --end-to-end-wrapper --canonical-d4 o10 --canonical-d4 o6 \
  --engine-corpus "${ENGINE_CORPUS}" \
  --run13-component "${RUN13_COMPONENT}" \
  --selection-plan "${SELECTION_PLAN}" \
  --output-dir "${OUTPUT_DIR}/benchmark"

run_arena() {
  local first_name="$1"
  local first_path="$2"
  local second_name="$3"
  local second_path="$4"
  local gate="$5"
  local first_canonical="$6"
  local second_canonical="$7"
  local canonical_flags=()
  if [[ "${first_canonical}" == "yes" ]]; then
    canonical_flags+=(--player1-canonical-d4)
  fi
  if [[ "${second_canonical}" == "yes" ]]; then
    canonical_flags+=(--player2-canonical-d4)
  fi
  "${PYTHON_BIN}" arena_santorini_v4_selection.py \
    --player1 "${first_path}" --player1-name "${first_name}" \
    --player2 "${second_path}" --player2-name "${second_name}" \
    --gate "${gate}" --games 40 --simulations 96 --batch-size 32 \
    --search-mode gumbel --gumbel-scale 0 --placement-gumbel-scale 1.5 \
    --player1-root-symmetries 1 --player2-root-symmetries 1 \
    "${canonical_flags[@]}" \
    --device cuda --fp16 --seed 20260814 \
    --engine-corpus "${ENGINE_CORPUS}" \
    --run13-component "${RUN13_COMPONENT}" \
    --selection-plan "${SELECTION_PLAN}" \
    --output "${OUTPUT_DIR}/arena_${first_name}_vs_${second_name}_${gate}.json"
}

for gate in standard full; do
  run_arena o6 "${O6}" o10 "${O10}" "${gate}" yes yes
  run_arena o6 "${O6}" e "${E}" "${gate}" yes no
  run_arena o10 "${O10}" e "${E}" "${gate}" yes no
done
