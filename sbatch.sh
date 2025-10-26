#!/bin/bash

# SBATCH --job-name=llama-finetune
# SBATCH --nodes=1
# SBATCH -D .
# SBATCH --output=O-%x_%j.txt
# SBATCH --error=E-%x_%j.txt
# SBATCH --gres=gpu:4
# SBATCH --cpus-per-task=120
# SBATCH --export=ALL


#SBATCH --job-name=llama-finetune
#SBATCH --partition=gpu          # <-- real GPU partition
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4        # or: --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=60
#SBATCH --time=02:00:00
#SBATCH --output=O-%x_%j.txt
#SBATCH --error=E-%x_%j.txt
#SBATCH --export=ALL


export OMP_NUM_THREADS=120
cd $SLURM_SUBMIT_DIR
srun srun.sh