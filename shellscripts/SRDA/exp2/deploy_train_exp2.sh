#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

CONFIG_PATH="configs/SRDA/exp2/srda_exp2_ldm_vqvae_ogi08.yaml"
LOG_DIR="${ROOT_DIR}/work/SRDA/exp2/logs"
mkdir -p "${LOG_DIR}"

GPU_IDS="${1:-0}"
NUM_GPUS="${2:-}"

if [[ -z "${NUM_GPUS}" ]]; then
  IFS=',' read -r -a _GPU_ID_ARR <<< "${GPU_IDS}"
  NUM_GPUS="${#_GPU_ID_ARR[@]}"
fi

cd "${ROOT_DIR}"

echo "Launching SRDA Exp2 training"
echo "  Root dir   : ${ROOT_DIR}"
echo "  Config path: ${CONFIG_PATH}"
echo "  Log dir    : ${LOG_DIR}"
echo "  GPU ids    : ${GPU_IDS}"
echo "  Num GPUs   : ${NUM_GPUS}"

CUDA_VISIBLE_DEVICES="${GPU_IDS}" nohup poetry run torchrun   --standalone   --nnodes 1   --nproc_per_node "${NUM_GPUS}"   pyscripts/SRDA/exp2/train_srda_exp2.py   --config_path "${CONFIG_PATH}"   > "${LOG_DIR}/train_srda_exp2_nohup.log" 2>&1 &

echo "SRDA Exp2 training started (nohup)."
