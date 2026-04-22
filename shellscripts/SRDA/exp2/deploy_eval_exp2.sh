#!/usr/bin/env bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

EXPERIMENT_NAME="srda_exp2"
TEST_NAME="base"
LOG_DIR="${ROOT_DIR}/work/SRDA/exp2/logs"
mkdir -p "${LOG_DIR}"

GPU_ID="${1:-0}"
SEED_START="${2:-9995}"
SEED_END="${3:-9999}"
BATCH_SIZE="${4:-30}"
RESPACING="${5:-10}"
ETA="${6:-1.0}"
CONFIG_PATH="${7:-configs/SRDA/exp2/srda_exp2_ldm_vqvae_ogi08.yaml}"
OUTPUTS_ROOT="${8:-${ROOT_DIR}/work/SRDA/exp2}"
TIMING="${TIMING:-0}"
TIMING_WARMUP_CYCLES="${TIMING_WARMUP_CYCLES:-1}"
TIMING_OUT_DIR="${TIMING_OUT_DIR:-}"
TIMING_SKIP_SAVE_OUTPUTS="${TIMING_SKIP_SAVE_OUTPUTS:-0}"

cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}/python"

echo "Launching SRDA Exp2 evaluation on GPU ${GPU_ID}"
echo "  Experiment : ${EXPERIMENT_NAME}/${TEST_NAME}"
echo "  Seeds      : ${SEED_START}..${SEED_END}"
echo "  Batch size : ${BATCH_SIZE}"
echo "  Respacing  : ${RESPACING}"
echo "  Eta        : ${ETA}"
echo "  Config     : ${CONFIG_PATH}"
echo "  Outputs    : ${OUTPUTS_ROOT}"
echo "  Timing     : TIMING=${TIMING} (warmup=${TIMING_WARMUP_CYCLES}, skip_save=${TIMING_SKIP_SAVE_OUTPUTS})"

ARGS=(
  pyscripts/SRDA/exp2/eval_srda_exp2_cycle.py
  --experiment_name "${EXPERIMENT_NAME}"
  --test_name "${TEST_NAME}"
  --seeds_start "${SEED_START}"
  --seeds_end "${SEED_END}"
  --batch_size "${BATCH_SIZE}"
  --config_path "${CONFIG_PATH}"
  --outputs_root "${OUTPUTS_ROOT}"
  --timestep_respacing "${RESPACING}"
  --eta "${ETA}"
)

if [[ "${TIMING}" == "1" ]]; then
  ARGS+=(--timing)
  ARGS+=(--timing_warmup_cycles "${TIMING_WARMUP_CYCLES}")
  if [[ -n "${TIMING_OUT_DIR}" ]]; then
    ARGS+=(--timing_out_dir "${TIMING_OUT_DIR}")
  fi
  if [[ "${TIMING_SKIP_SAVE_OUTPUTS}" == "1" ]]; then
    ARGS+=(--timing_skip_save_outputs)
  fi
fi

LOG_SUFFIX="gpu${GPU_ID}_tr${RESPACING}_eta${ETA}"
LOG_FILE="${LOG_DIR}/eval_srda_exp2_nohup_${LOG_SUFFIX}.log"

CUDA_VISIBLE_DEVICES="${GPU_ID}" nohup poetry run python "${ARGS[@]}" > "${LOG_FILE}" 2>&1 &

echo "SRDA Exp2 evaluation started (nohup). Log: ${LOG_FILE}"
