#!/bin/bash

# Set proxy configuration if needed for network access
export http_proxy=http://u-MtfrT7:vH5orjDV@127.0.0.1:3128
export https_proxy=http://u-MtfrT7:vH5orjDV@127.0.0.1:3128

# Load required system modules for the cluster environment
module load miniforge/24.1.2 cuda/12.8 cudnn/9.6.0.74_cuda12

# Activate the conda environment and set Python output buffering
source activate evodiff
export PYTHONUNBUFFERED=1

# Display GPU information for verification
nvidia-smi

# Execute the main optimization script
python scripts/run_betterevodiff.py
