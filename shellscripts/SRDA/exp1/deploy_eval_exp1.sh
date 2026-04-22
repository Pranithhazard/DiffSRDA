#!/usr/bin/env bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

EXPERIMENT_NAME="srda_exp1"
TEST_NAME="base"
LOG_DIR="${ROOT_DIR}/work/SRDA/exp1/logs"
mkdir -p "${LOG_DIR}"

GPU_ID="${1:-0}"
SEED_START="${2:-9995}"
SEED_END="${3:-9999}"
BATCH_SIZE="${4:-30}"
RESPACING="${5:-20}"
ETA="${6:-1.0}"
CONFIG_PATH="${7:-configs/SRDA/exp1/srda_exp1_ddim_sf4.yaml}"
OUTPUTS_ROOT="${8:-${ROOT_DIR}/work/SRDA/exp1}"
OBS_GUIDANCE_MODE="${9:-off}"
OBS_GUIDANCE_GAMMA="${10:-1.0}"
OBS_GUIDANCE_SIGMA="${11:-}"
OBS_GUIDANCE_EVERY="${12:-1}"
OBS_GUIDANCE_BLUR="${13:-0.0}"

# Optional suffix tag / model override
MODEL_TAG="${MODEL_TAG:-}"   # e.g., trainogi04
MODEL_DIR="${MODEL_DIR:-}"   # repo-root-relative dir containing weight_diffusion.pth

# --- Sensor shift (conditioning vs guidance observations) ---
# Set via env vars so positional args remain stable.
COND_GRID_INTERVAL="${COND_GRID_INTERVAL:-}"
GUIDANCE_GRID_INTERVAL="${GUIDANCE_GRID_INTERVAL:-}"
COND_OBS_NOISE_SIGMA="${COND_OBS_NOISE_SIGMA:-}"
GUIDANCE_OBS_NOISE_SIGMA="${GUIDANCE_OBS_NOISE_SIGMA:-}"
OBS_NOISE_SEED="${OBS_NOISE_SEED:-}"

# Optional timing (set via env vars; keeps positional args stable)
TIMING="${TIMING:-0}"                                   # 0|1
TIMING_WARMUP_CYCLES="${TIMING_WARMUP_CYCLES:-1}"
TIMING_OUT_DIR="${TIMING_OUT_DIR:-}"                    # empty => default under outputs_root
TIMING_SKIP_SAVE_OUTPUTS="${TIMING_SKIP_SAVE_OUTPUTS:-0}" # 0|1

cd "${ROOT_DIR}"

echo "Launching SRDA Exp1 evaluation on GPU ${GPU_ID}"
echo "  Experiment : ${EXPERIMENT_NAME}/${TEST_NAME}"
echo "  Seeds      : ${SEED_START}..${SEED_END}"
echo "  Batch size : ${BATCH_SIZE}"
echo "  Respacing  : ${RESPACING}"
echo "  Eta        : ${ETA}"
echo "  Config     : ${CONFIG_PATH}"
echo "  Outputs    : ${OUTPUTS_ROOT}"
echo "  Obs guide : mode=${OBS_GUIDANCE_MODE}, gamma=${OBS_GUIDANCE_GAMMA}, sigma=${OBS_GUIDANCE_SIGMA:-auto}, every=${OBS_GUIDANCE_EVERY}, blur=${OBS_GUIDANCE_BLUR}"
echo "  Sensor     : cond_ogi=${COND_GRID_INTERVAL:-default}, guid_ogi=${GUIDANCE_GRID_INTERVAL:-default}, cond_sigma=${COND_OBS_NOISE_SIGMA:-default}, guid_sigma=${GUIDANCE_OBS_NOISE_SIGMA:-default}, obs_seed=${OBS_NOISE_SEED:-default}"
echo "  Model tag  : ${MODEL_TAG:-none} (MODEL_DIR=${MODEL_DIR:-default})"
echo "  Timing     : TIMING=${TIMING} (warmup=${TIMING_WARMUP_CYCLES}, skip_save=${TIMING_SKIP_SAVE_OUTPUTS})"

export PYTHONPATH="${ROOT_DIR}/python"

ARGS=(
  pyscripts/SRDA/exp1/eval_srda_exp1_cycle.py
  --experiment_name "${EXPERIMENT_NAME}"
  --test_name "${TEST_NAME}"
  --seeds_start "${SEED_START}"
  --seeds_end "${SEED_END}"
  --batch_size "${BATCH_SIZE}"
  --config_path "${CONFIG_PATH}"
  --outputs_root "${OUTPUTS_ROOT}"
  --timestep_respacing "${RESPACING}"
  --eta "${ETA}"
  --obs_guidance_mode "${OBS_GUIDANCE_MODE}"
  --obs_guidance_gamma "${OBS_GUIDANCE_GAMMA}"
  --obs_guidance_every "${OBS_GUIDANCE_EVERY}"
  --obs_guidance_blur_sigma_px "${OBS_GUIDANCE_BLUR}"
)

if [[ -n "${MODEL_DIR}" ]]; then
  ARGS+=(--model_dir "${MODEL_DIR}")
fi
if [[ -n "${MODEL_TAG}" ]]; then
  ARGS+=(--model_tag "${MODEL_TAG}")
fi

if [[ -n "${OBS_GUIDANCE_SIGMA}" ]]; then
  ARGS+=(--obs_guidance_sigma "${OBS_GUIDANCE_SIGMA}")
fi

if [[ -n "${COND_GRID_INTERVAL}" ]]; then
  ARGS+=(--cond_grid_interval "${COND_GRID_INTERVAL}")
fi
if [[ -n "${GUIDANCE_GRID_INTERVAL}" ]]; then
  ARGS+=(--guidance_grid_interval "${GUIDANCE_GRID_INTERVAL}")
fi
if [[ -n "${COND_OBS_NOISE_SIGMA}" ]]; then
  ARGS+=(--cond_obs_noise_sigma "${COND_OBS_NOISE_SIGMA}")
fi
if [[ -n "${GUIDANCE_OBS_NOISE_SIGMA}" ]]; then
  ARGS+=(--guidance_obs_noise_sigma "${GUIDANCE_OBS_NOISE_SIGMA}")
fi
if [[ -n "${OBS_NOISE_SEED}" ]]; then
  ARGS+=(--obs_noise_seed "${OBS_NOISE_SEED}")
fi

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

LOG_SUFFIX="gpu${GPU_ID}_tr${RESPACING}_eta${ETA}_og${OBS_GUIDANCE_MODE}_blur${OBS_GUIDANCE_BLUR}"
if [[ -n "${COND_GRID_INTERVAL}" ]]; then
  LOG_SUFFIX+="_condogi${COND_GRID_INTERVAL}"
fi
if [[ -n "${GUIDANCE_GRID_INTERVAL}" ]]; then
  LOG_SUFFIX+="_guidogi${GUIDANCE_GRID_INTERVAL}"
fi
if [[ -n "${COND_OBS_NOISE_SIGMA}" ]]; then
  LOG_SUFFIX+="_condn${COND_OBS_NOISE_SIGMA//./p}"
fi
if [[ -n "${GUIDANCE_OBS_NOISE_SIGMA}" ]]; then
  LOG_SUFFIX+="_guidn${GUIDANCE_OBS_NOISE_SIGMA//./p}"
fi
if [[ -n "${OBS_NOISE_SEED}" ]]; then
  LOG_SUFFIX+="_obsseed${OBS_NOISE_SEED}"
fi
if [[ -n "${MODEL_TAG}" ]]; then
  LOG_SUFFIX+="_${MODEL_TAG}"
fi
LOG_FILE="${LOG_DIR}/eval_srda_exp1_nohup_${LOG_SUFFIX}.log"

CUDA_VISIBLE_DEVICES="${GPU_ID}" nohup poetry run python "${ARGS[@]}" \
  > "${LOG_FILE}" 2>&1 &

echo "SRDA Exp1 evaluation started (nohup). Log: ${LOG_FILE}"
