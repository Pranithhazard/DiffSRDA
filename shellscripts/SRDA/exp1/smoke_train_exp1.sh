#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

CONFIG_PATH="configs/SRDA/exp1/srda_exp1_ddim_sf4_smoke.yaml"
LOG_DIR="${ROOT_DIR}/work/SRDA/exp1_smoke/logs"
mkdir -p "${LOG_DIR}"

GPU_IDS="${1:-0}"
NUM_GPUS="${2:-}"

if [[ -z "${NUM_GPUS}" ]]; then
  IFS=',' read -r -a _GPU_ID_ARR <<< "${GPU_IDS}"
  NUM_GPUS="${#_GPU_ID_ARR[@]}"
fi

cd "${ROOT_DIR}"

LOG_FILE="${LOG_DIR}/train_srda_exp1_smoke_gpus${NUM_GPUS}.log"

echo "Running SRDA Exp1 smoke training"
echo "  Root dir   : ${ROOT_DIR}"
echo "  Config path: ${CONFIG_PATH}"
echo "  Log dir    : ${LOG_DIR}"
echo "  GPU ids    : ${GPU_IDS}"
echo "  Num GPUs   : ${NUM_GPUS}"
echo "  Log file   : ${LOG_FILE}"

time CUDA_VISIBLE_DEVICES="${GPU_IDS}" poetry run torchrun   --standalone   --nnodes 1   --nproc_per_node "${NUM_GPUS}"   pyscripts/SRDA/exp1/train_srda_exp1.py   --config_path "${CONFIG_PATH}" | tee "${LOG_FILE}"
