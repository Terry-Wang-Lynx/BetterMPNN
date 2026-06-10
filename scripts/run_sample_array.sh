#!/bin/bash
#SBATCH --job-name=bmpnn_sample
#SBATCH --array=0-3
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --output=logs/sampling_%A_%a.log
#SBATCH --error=logs/sampling_%A_%a.err

# BetterMPNN large-scale sampling via a SLURM job array.
# Splits `steps` (read from the YAML) across NUM_TASKS parallel GPU workers,
# all writing to one shared output dir. Merge afterwards with
#   bash scripts/merge_results.sh <output_dir>
#
# Usage:
#   sbatch scripts/run_sample_array.sh [config.yaml]
# Keep --array=0-(N-1) in sync with NUM_TASKS below.

set -euo pipefail

# --- Environment setup (edit for your cluster) ---
#   module load cuda/12.8 cudnn apptainer/1.2.4 miniforge3
# `module load <conda>` does NOT enable `conda activate` in a non-interactive
# shell; source the hook first (use the full conda path if it is not on PATH):
#   eval "$(conda shell.bash hook)"
#   conda activate bettermpnn

export PYTHONUNBUFFERED=1

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"
mkdir -p logs

CONFIG_FILE="${1:-configs/example_sampling.yaml}"
[ -f "${CONFIG_FILE}" ] || { echo "Config not found: ${CONFIG_FILE}"; exit 1; }

NUM_TASKS=4   # must match --array count
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

TOTAL_STEPS="$(awk '/^steps:/{print $2; exit}' "${CONFIG_FILE}")"
TOTAL_STEPS="${TOTAL_STEPS:-100}"

STEPS_PER_TASK=$(( (TOTAL_STEPS + NUM_TASKS - 1) / NUM_TASKS ))
STEP_START=$(( TASK_ID * STEPS_PER_TASK ))
STEP_END=$(( STEP_START + STEPS_PER_TASK - 1 ))
[ ${STEP_END} -ge ${TOTAL_STEPS} ] && STEP_END=$(( TOTAL_STEPS - 1 ))
[ ${STEP_START} -ge ${TOTAL_STEPS} ] && { echo "Task ${TASK_ID}: no work"; exit 0; }

OUTPUT_DIR="output_sampling_${SLURM_ARRAY_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"

echo "=== BetterMPNN Sampling: task ${TASK_ID}, steps ${STEP_START}-${STEP_END}/${TOTAL_STEPS} ==="
echo "Start:  $(date)   Node: $(hostname)"
echo "Config: ${CONFIG_FILE}   Output: ${OUTPUT_DIR}"
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# Capture the exit code so trailing/cosmetic commands can't mask success.
rc=0
python -m bettermpnn.cli \
    --config "${CONFIG_FILE}" \
    --mode sample \
    --output "${OUTPUT_DIR}" \
    --step-range "${STEP_START}-${STEP_END}" \
    -v || rc=$?

echo "=== Task ${TASK_ID} done: $(date) (exit ${rc}) ==="
echo "Merge with: bash scripts/merge_results.sh ${OUTPUT_DIR}"
exit "${rc}"
