#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

RUN_DIR="${RUN_DIR:-runs/PE_20260520_110828}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/risk_dataset/D0_branch_full_v1_10ckpt_500ep_h150_k3_m16}"
DEVICE="${DEVICE:-cuda:0}"
SEED="${SEED:-1}"

# Per checkpoint. With 10 updates, the default is 500 source episodes total.
BRANCH_EPISODES="${BRANCH_EPISODES:-50}"
BRANCH_MAX_ANCHORS="${BRANCH_MAX_ANCHORS:-8}"
BRANCH_CANDIDATES="${BRANCH_CANDIDATES:-16}"
BRANCH_HORIZON="${BRANCH_HORIZON:-150}"
BRANCH_ANCHOR_STRIDE="${BRANCH_ANCHOR_STRIDE:-25}"
BRANCH_MIN_CANDIDATE_ACTION_L2="${BRANCH_MIN_CANDIDATE_ACTION_L2:-0.08}"

RISK_LIDAR_STACK_FRAMES="${RISK_LIDAR_STACK_FRAMES:-3}"
PERSIST_STEPS="${PERSIST_STEPS:-10}"
LABEL_MIN_VISIBLE_PIXELS="${LABEL_MIN_VISIBLE_PIXELS:-3}"
log_file="${LOG_FILE:-logs/D0_branch_full_v1_$(date +%Y%m%d_%H%M%S).log}"

BRANCH_UPDATES=(
  170
  220
  300
  350
  464
  610
  754
  900
  1110
  1300
)

OVERWRITE_ARGS=()
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  OVERWRITE_ARGS=(--overwrite)
fi

echo "Collecting D0 branch full dataset"
echo "  run_dir=${RUN_DIR}"
echo "  output_dir=${OUTPUT_DIR}"
echo "  branch_updates=${BRANCH_UPDATES[*]}"
echo "  branch_episodes_per_update=${BRANCH_EPISODES}"
echo "  expected_source_episodes=$((BRANCH_EPISODES * ${#BRANCH_UPDATES[@]}))"
echo "  branch_candidates=${BRANCH_CANDIDATES}"
echo "  branch_max_anchors=${BRANCH_MAX_ANCHORS}"
echo "  branch_horizon=${BRANCH_HORIZON}"
echo "  device=${DEVICE}"
echo "  log_file=${log_file}"

mkdir -p "$(dirname "${log_file}")"

PYTHONUNBUFFERED=1 conda run --no-capture-output -n aerialgym_v2 \
  python -u aerial_gym/rl_training/cleanrl/pursuit_risk/export_lidar_rollouts.py \
  --dataset-mode branch \
  --run-dir "${RUN_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --task pursuit_guidance_task \
  --robot-name base_quad_root_link_control_with_lidar \
  --controller-name thrust_bodyrate_control \
  --target-asset-type target_x500 \
  --device "${DEVICE}" \
  --seed "${SEED}" \
  --branch-updates "${BRANCH_UPDATES[@]}" \
  --branch-episodes "${BRANCH_EPISODES}" \
  --branch-candidates "${BRANCH_CANDIDATES}" \
  --branch-horizon "${BRANCH_HORIZON}" \
  --risk-horizons 50 150 \
  --risk-lidar-stack-frames "${RISK_LIDAR_STACK_FRAMES}" \
  --persist-steps "${PERSIST_STEPS}" \
  --label-min-visible-pixels "${LABEL_MIN_VISIBLE_PIXELS}" \
  --branch-anchor-stride "${BRANCH_ANCHOR_STRIDE}" \
  --branch-max-anchors "${BRANCH_MAX_ANCHORS}" \
  --branch-edge-margin-px 80 \
  --branch-low-pixel-max 50 \
  --branch-risk-anchor-min-score 0.25 \
  --branch-min-candidate-action-l2 "${BRANCH_MIN_CANDIDATE_ACTION_L2}" \
  --branch-min-severity-separation 0.001 \
  --branch-min-separated-anchors 1 \
  --branch-min-visible-anchor-rate 0.5 \
  --branch-min-detectable-anchor-rate 0.5 \
  "${OVERWRITE_ARGS[@]}" \
  2>&1 | tee -a "${log_file}"
