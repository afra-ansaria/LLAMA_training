#! /bin/bash

# cd $SLURM_SUBMIT_DIR
# export GPUS_PER_NODE=4
# HOST_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
# MAIN_PROCESS_PORT=12345

# echo "SLURM_NNODES=$SLURM_NNODES"
# echo "SLURM_NODEID=$SLURM_NODEID"
# echo "HOST_ADDR=$HOST_ADDR"
# echo "MAIN_PROCESS_PORT=$MAIN_PROCESS_PORT"

# source .venv/bin/activate
# mkdir -p logs
# torchrun --nnodes $SLURM_NNODES \
# --nproc_per_node $GPUS_PER_NODE \
# --master_addr $HOST_ADDR \
# --master_port $MAIN_PROCESS_PORT \
# --node_rank=$SLURM_NODEID \
# finetuning.py \
# --model_name ./llama-3-8b \
# --output_dir saved_peft_model \
# --use_peft \
# --peft_method lora \
# --enable_fsdp \
# --use_fast_kernels \
# --dataset alpaca_dataset | tee -a logs/finetuning.log

#!/bin/bash
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

echo "== SLURM =="; scontrol show job "$SLURM_JOB_ID" | egrep -i 'Partition|TRES|Gres|Nodes'
echo "NNODES=$SLURM_NNODES NODEID=$SLURM_NODEID GPUS_ON_NODE=$SLURM_GPUS_ON_NODE"

# Sanity checks (must show GPUs and CUDA enabled)
which nvidia-smi || true
nvidia-smi || echo "no nvidia-smi"
python - <<'PY'
import torch, os
print("cuda.is_available:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("torch.version.cuda:", torch.version.cuda)
PY

# Rendezvous on Slurm-provided IP
export RDZV="${SLURM_LAUNCH_NODE_IP}:29500"

torchrun \
  --nnodes="$SLURM_NNODES" \
  --nproc_per_node="$SLURM_GPUS_ON_NODE" \
  --rdzv_backend=c10d \
  --rdzv_endpoint="$RDZV" \
  /root/LLAMA_training/finetuning.py \
    --model_name ./llama-3-8b \
    --output_dir /mnt/data/saved_peft_model \
    --use_peft \
    --peft_method lora \
    --enable_fsdp \
    --use_fast_kernels \
    --dataset alpaca_dataset | tee -a logs/finetuning.log

