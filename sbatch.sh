#!/bin/bash

# # SBATCH --job-name=llama-finetune
# # SBATCH --nodes=1
# # SBATCH -D .
# # SBATCH --output=O-%x_%j.txt
# # SBATCH --error=E-%x_%j.txt
# # SBATCH --gres=gpu:4
# # SBATCH --cpus-per-task=120
# # SBATCH --export=ALL


# #SBATCH --job-name=llama-finetune
# #SBATCH --partition=gpu          # <-- real GPU partition
# #SBATCH --nodes=1
# #SBATCH --gpus-per-node=4        # or: --gres=gpu:4
# #SBATCH --ntasks-per-node=4
# #SBATCH --cpus-per-task=60
# #SBATCH --time=02:00:00
# #SBATCH --output=O-%x_%j.txt
# #SBATCH --error=E-%x_%j.txt
# #SBATCH --export=ALL


# export OMP_NUM_THREADS=120
# cd $SLURM_SUBMIT_DIR
# srun srun.sh

#SBATCH --job-name=llama-finetune
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4        # 1 process per GPU
#SBATCH --cpus-per-task=6          # tune to your CPU cores
#SBATCH --time=02:00:00
#SBATCH --output=O-%x_%j.txt
#SBATCH --error=E-%x_%j.txt
#SBATCH --export=ALL
set -euo pipefail

# Threads for each rank (match cpus-per-task)
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-6}

# NCCL for Ethernet clusters (no IB)
export NCCL_DEBUG=warn
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0      # change if iface != eth0

# Container image + mounts
CONTAINER="ghcr.io/yourorg/torch:2.4-cuda12.1"
MOUNTS="/mnt/data:/mnt/data"

cd "$SLURM_SUBMIT_DIR"
# Run the helper inside the container (ONE srun here)
srun --container-image="$CONTAINER" \
     --container-mounts="$MOUNTS" \
     bash srun.sh
