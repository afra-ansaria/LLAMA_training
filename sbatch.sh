#!/bin/bash
mkdir -p logs
    #SBATCH --partition=main
    #SBATCH --job-name=llama-finetune
    #SBATCH --nodes=2
    #SBATCH -D .
    #SBATCH --output=logs/O-%x_%j.txt
    #SBATCH --error=logs/E-%x_%j.txt
    #SBATCH --gres=gpu:4
    #SBATCH --cpus-per-task=100
    #SBATCH --export=ALL

export OMP_NUM_THREADS=120
cd $SLURM_SUBMIT_DIR
srun srun.sh