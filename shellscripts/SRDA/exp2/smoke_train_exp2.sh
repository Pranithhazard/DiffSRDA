#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

CONFIG_PATH="configs/SRDA/exp2/srda_exp2_ldm_vqvae_ogi08_smoke.yaml"
LOG_DIR="${ROOT_DIR}/work/SRDA/exp2_smoke/logs"
mkdir -p "${LOG_DIR}"

GPU_IDS="${1:-0}"
NUM_GPUS="${2:-}"

if [[ -z "${NUM_GPUS}" ]]; then
  IFS=',' read -r -a _GPU_ID_ARR <<< "${GPU_IDS}"
  NUM_GPUS="${#_GPU_ID_ARR[@]}"
fi

cd "${ROOT_DIR}"

LOG_FILE="${LOG_DIR}/train_srda_exp2_smoke_gpus${NUM_GPUS}.log"

echo "Running SRDA Exp2 smoke training"
echo "  Root dir   : ${ROOT_DIR}"
echo "  Config path: ${CONFIG_PATH}"
echo "  Log dir    : ${LOG_DIR}"
echo "  GPU ids    : ${GPU_IDS}"
echo "  Num GPUs   : ${NUM_GPUS}"
echo "  Log file   : ${LOG_FILE}"
echo "  Requires   : saved_models/SRDA/latent_vqvae_ogi08/weight_latent.pth"

time CUDA_VISIBLE_DEVICES="${GPU_IDS}" poetry run torchrun   --standalone   --nnodes 1   --nproc_per_node "${NUM_GPUS}"   pyscripts/SRDA/exp2/train_srda_exp2.py   --config_path "${CONFIG_PATH}" | tee "${LOG_FILE}"
