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

# Activate your environment here (edit for your cluster), e.g.:
#   module load cuda cudnn apptainer
#   conda activate bettermpnn

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
