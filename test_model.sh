#!/bin/bash

# Set proxy configuration if needed for network access
export http_proxy=http://u-MtfrT7:vH5orjDV@127.0.0.1:3128
export https_proxy=http://u-MtfrT7:vH5orjDV@127.0.0.1:3128

# Load required system modules for the cluster environment
module load miniforge/25.3.0-3 cuda/12.8 cudnn/9.6.0.74_cuda12
module load apptainer/1.2.4

# Activate the conda environment and set Python output buffering
source activate protein-mpnn
export PYTHONUNBUFFERED=1

# Display GPU information for verification
nvidia-smi

# Execute the main optimization script
python scripts/test-model-1.py