#!/bin/bash
#SBATCH --job-name=bettermpnn
#SBATCH --gpus=1
#SBATCH --time=72:00:00
#SBATCH --output=logs/train_%j.log
#SBATCH --error=logs/train_%j.err

set -euo pipefail

# --- Adjust these to your cluster environment ---
# module load cuda/12.x apptainer/1.x
# conda activate your_env

export PYTHONUNBUFFERED=1
cd /path/to/BetterMPNN

echo "=== BetterMPNN Training ==="
echo "Start: $(date)"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python -m bettermpnn.cli --config configs/example.yaml

echo "=== Done: $(date) ==="
