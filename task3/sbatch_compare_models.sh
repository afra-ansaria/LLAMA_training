#!/bin/bash
#SBATCH --job-name=infer-compare
#SBATCH --partition=gpu             # change to your GPU partition name
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=3                   # or --gres=gpu:1 on older Slurm configs
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out     # logs/infer-compare-<jobid>.out



# --- make sure caches & warnings don’t bite ---
export TOKENIZERS_PARALLELISM=false
export HF_HOME=${HF_HOME:-$PWD/.hf-cache}           # local HF cache

# --- run ---
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
echo "Running on host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# Use srun so Slurm binds resources correctly
srun python evalCompare2.py
