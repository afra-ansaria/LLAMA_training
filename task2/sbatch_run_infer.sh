#!/bin/bash
#SBATCH --job-name=llama-infer
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:10:00
#SBATCH --output=logs/O-%x_%j.txt
#SBATCH --error=logs/E-%x_%j.txt
#SBATCH -D .

# ---------------------------------------------------------
# ENVIRONMENT SETUP
# ---------------------------------------------------------
set -euo pipefail
echo "[$(date)] Starting inference job on node $(hostname)"


# GPU diagnostics
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi || echo "nvidia-smi not available."

# ---------------------------------------------------------
# RUN INFERENCE SCRIPT
# ---------------------------------------------------------
echo "[$(date)] Running inference..."
python inferenceWithExample.py

echo "[$(date)] Job complete."
