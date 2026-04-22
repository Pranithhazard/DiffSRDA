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

MODEL_TAG="${MODEL_TAG:-}"
MODEL_DIR="${MODEL_DIR:-}"

OBS_GUIDANCE_MODE="${OBS_GUIDANCE_MODE:-soft}"
OBS_GUIDANCE_GAMMA="${OBS_GUIDANCE_GAMMA:-1.0}"
OBS_GUIDANCE_SIGMA="${OBS_GUIDANCE_SIGMA:-}"
OBS_GUIDANCE_EVERY="${OBS_GUIDANCE_EVERY:-1}"
OBS_GUIDANCE_BLUR_START="${OBS_GUIDANCE_BLUR_START:-0.0}"
OBS_GUIDANCE_BLUR_FINAL="${OBS_GUIDANCE_BLUR_FINAL:-}"
OBS_GUIDANCE_BLUR_POWER="${OBS_GUIDANCE_BLUR_POWER:-1.0}"
OBS_GUIDANCE_TIGHTEN_FINAL_STEPS="${OBS_GUIDANCE_TIGHTEN_FINAL_STEPS:-0}"
OBS_GUIDANCE_RECOMPUTE_EPS="${OBS_GUIDANCE_RECOMPUTE_EPS:-0}"

COND_GRID_INTERVAL="${COND_GRID_INTERVAL:-}"
GUIDANCE_GRID_INTERVAL="${GUIDANCE_GRID_INTERVAL:-}"
COND_OBS_NOISE_SIGMA="${COND_OBS_NOISE_SIGMA:-}"
GUIDANCE_OBS_NOISE_SIGMA="${GUIDANCE_OBS_NOISE_SIGMA:-}"
OBS_NOISE_SEED="${OBS_NOISE_SEED:-}"

SENSOR_SCENARIO="${SENSOR_SCENARIO:-}"
SENSOR_SEED="${SENSOR_SEED:-}"
SENSOR_OBS_NOISE_SIGMA="${SENSOR_OBS_NOISE_SIGMA:-}"
SENSOR_GRID_INTERVAL="${SENSOR_GRID_INTERVAL:-}"
SENSOR_NUM_SENSORS="${SENSOR_NUM_SENSORS:-}"

TIMING="${TIMING:-0}"
TIMING_WARMUP_CYCLES="${TIMING_WARMUP_CYCLES:-1}"
TIMING_OUT_DIR="${TIMING_OUT_DIR:-}"
TIMING_SKIP_SAVE_OUTPUTS="${TIMING_SKIP_SAVE_OUTPUTS:-0}"

RUN_BASELINES="${RUN_BASELINES:-0}"

cd "${ROOT_DIR}"
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
  --obs_guidance_blur_sigma_px "${OBS_GUIDANCE_BLUR_START}"
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

if [[ -z "${SENSOR_SCENARIO}" ]]; then
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
else
  ARGS+=(--sensor_scenario "${SENSOR_SCENARIO}")
  if [[ -n "${SENSOR_SEED}" ]]; then
    ARGS+=(--sensor_seed "${SENSOR_SEED}")
  fi
  if [[ -n "${SENSOR_OBS_NOISE_SIGMA}" ]]; then
    ARGS+=(--sensor_obs_noise_sigma "${SENSOR_OBS_NOISE_SIGMA}")
  fi
  if [[ -n "${SENSOR_GRID_INTERVAL}" ]]; then
    ARGS+=(--sensor_grid_interval "${SENSOR_GRID_INTERVAL}")
  fi
  if [[ -n "${SENSOR_NUM_SENSORS}" ]]; then
    ARGS+=(--sensor_num_sensors "${SENSOR_NUM_SENSORS}")
  fi
fi

if [[ -n "${OBS_GUIDANCE_BLUR_FINAL}" ]]; then
  ARGS+=(--obs_guidance_blur_sigma_px_final "${OBS_GUIDANCE_BLUR_FINAL}")
  ARGS+=(--obs_guidance_blur_schedule_power "${OBS_GUIDANCE_BLUR_POWER}")
fi
if [[ "${OBS_GUIDANCE_TIGHTEN_FINAL_STEPS}" != "0" ]]; then
  ARGS+=(--obs_guidance_tighten_final_steps "${OBS_GUIDANCE_TIGHTEN_FINAL_STEPS}")
fi
if [[ "${OBS_GUIDANCE_RECOMPUTE_EPS}" == "1" ]]; then
  ARGS+=(--obs_guidance_recompute_eps)
fi
if [[ "${RUN_BASELINES}" != "1" ]]; then
  ARGS+=(--skip_enkf_hr --skip_enkf_bicubic)
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

if [[ "${GPU_ID}" == "cpu" ]]; then
  echo "Launching SRDA Exp1 guidance ablation on CPU"
else
  echo "Launching SRDA Exp1 guidance ablation on GPU ${GPU_ID}"
fi
echo "  Seeds      : ${SEED_START}..${SEED_END}"
echo "  Respacing  : ${RESPACING}"
echo "  Eta        : ${ETA}"
echo "  Batch size : ${BATCH_SIZE}"
echo "  Config     : ${CONFIG_PATH}"
echo "  Outputs    : ${OUTPUTS_ROOT}"
echo "  Guidance   : mode=${OBS_GUIDANCE_MODE}, gamma=${OBS_GUIDANCE_GAMMA}, sigma=${OBS_GUIDANCE_SIGMA:-auto}, every=${OBS_GUIDANCE_EVERY}"
echo "  Blur       : start=${OBS_GUIDANCE_BLUR_START}, final=${OBS_GUIDANCE_BLUR_FINAL:-none}, power=${OBS_GUIDANCE_BLUR_POWER}"
echo "  Tighten    : final_steps=${OBS_GUIDANCE_TIGHTEN_FINAL_STEPS}"
echo "  Eps-recomp : ${OBS_GUIDANCE_RECOMPUTE_EPS}"
echo "  Baselines  : RUN_BASELINES=${RUN_BASELINES}"
if [[ -z "${SENSOR_SCENARIO}" ]]; then
  echo "  Sensor     : cond_ogi=${COND_GRID_INTERVAL:-default}, guid_ogi=${GUIDANCE_GRID_INTERVAL:-default}, cond_sigma=${COND_OBS_NOISE_SIGMA:-default}, guid_sigma=${GUIDANCE_OBS_NOISE_SIGMA:-default}, obs_seed=${OBS_NOISE_SEED:-default}"
else
  echo "  Scenario   : scenario=${SENSOR_SCENARIO}, sseed=${SENSOR_SEED:-auto}, sn=${SENSOR_OBS_NOISE_SIGMA:-default}, ogi=${SENSOR_GRID_INTERVAL:-default}, Ns=${SENSOR_NUM_SENSORS:-}"
fi
echo "  Model tag  : ${MODEL_TAG:-none} (MODEL_DIR=${MODEL_DIR:-default})"
echo "  Timing     : TIMING=${TIMING} (warmup=${TIMING_WARMUP_CYCLES}, skip_save=${TIMING_SKIP_SAVE_OUTPUTS})"

LOG_SUFFIX="gpu${GPU_ID}_tr${RESPACING}_eta${ETA}_og${OBS_GUIDANCE_MODE}_blur${OBS_GUIDANCE_BLUR_START}"
if [[ -n "${OBS_GUIDANCE_BLUR_FINAL}" ]]; then
  LOG_SUFFIX+="_to${OBS_GUIDANCE_BLUR_FINAL}"
fi
if [[ "${OBS_GUIDANCE_TIGHTEN_FINAL_STEPS}" != "0" ]]; then
  LOG_SUFFIX+="_tight${OBS_GUIDANCE_TIGHTEN_FINAL_STEPS}"
fi
if [[ "${OBS_GUIDANCE_RECOMPUTE_EPS}" == "1" ]]; then
  LOG_SUFFIX+="_epsrec"
fi
if [[ -z "${SENSOR_SCENARIO}" ]]; then
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
else
  LOG_SUFFIX+="_sens${SENSOR_SCENARIO}"
  if [[ -n "${SENSOR_GRID_INTERVAL}" ]]; then
    LOG_SUFFIX+="_ogi${SENSOR_GRID_INTERVAL}"
  fi
  if [[ -n "${SENSOR_NUM_SENSORS}" ]]; then
    LOG_SUFFIX+="_Ns${SENSOR_NUM_SENSORS}"
  fi
  if [[ -n "${SENSOR_OBS_NOISE_SIGMA}" ]]; then
    LOG_SUFFIX+="_sn${SENSOR_OBS_NOISE_SIGMA//./p}"
  fi
  if [[ -n "${SENSOR_SEED}" ]]; then
    LOG_SUFFIX+="_sseed${SENSOR_SEED}"
  fi
fi
if [[ -n "${MODEL_TAG}" ]]; then
  LOG_SUFFIX+="_${MODEL_TAG}"
fi
LOG_FILE="${LOG_DIR}/eval_srda_exp1_ablation_nohup_${LOG_SUFFIX}.log"

if [[ "${GPU_ID}" == "cpu" ]]; then
  CUDA_VISIBLE_DEVICES="" nohup poetry run python "${ARGS[@]}" > "${LOG_FILE}" 2>&1 &
else
  CUDA_VISIBLE_DEVICES="${GPU_ID}" nohup poetry run python "${ARGS[@]}" > "${LOG_FILE}" 2>&1 &
fi

echo "Started (nohup). Log: ${LOG_FILE}"
