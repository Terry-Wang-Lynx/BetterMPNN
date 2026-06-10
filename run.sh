#!/bin/bash
#SBATCH --job-name=bettermpnn
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --output=logs/train_%j.log
#SBATCH --error=logs/train_%j.err

# BetterMPNN training launcher (single GPU, GRPO fine-tuning).
#   sbatch run.sh [config.yaml]
# or run directly:
#   bash run.sh [config.yaml]

set -euo pipefail

# --- Environment setup (edit for your cluster) ---
# Load CUDA / cuDNN / the container runtime and your Python/conda module:
#   module load cuda/12.8 cudnn apptainer/1.2.4 miniforge3
# NOTE: `module load <conda>` alone does NOT enable `conda activate` in a
# non-interactive shell — you must source the shell hook first:
#   eval "$(conda shell.bash hook)"
#   conda activate bettermpnn
# (If `conda` is not on PATH after the module load, call its hook by full path,
#  e.g. eval "$(/path/to/miniforge/bin/conda shell.bash hook)".)

export PYTHONUNBUFFERED=1

# Resolve repo root from this script's location (portable; no hardcoded path).
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_DIR}"
mkdir -p logs

CONFIG_FILE="${1:-configs/example.yaml}"
OUTPUT_DIR="output_training_${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"

echo "=== BetterMPNN Training ==="
echo "Start:  $(date)"
echo "Node:   $(hostname)"
echo "Config: ${CONFIG_FILE}"
echo "Output: ${OUTPUT_DIR}"
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# Capture the training exit code explicitly so trailing/cosmetic commands can't
# mask success (e.g. a SIGPIPE under `pipefail` reporting a false failure).
rc=0
python -m bettermpnn.cli --config "${CONFIG_FILE}" --mode train --output "${OUTPUT_DIR}" || rc=$?

echo "=== Done: $(date) (exit ${rc}) ==="
exit "${rc}"
