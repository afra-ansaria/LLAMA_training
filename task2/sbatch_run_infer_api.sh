#!/bin/bash
#SBATCH --job-name=llama-infer
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/O-%x_%j.txt
#SBATCH --error=logs/E-%x_%j.txt

# === Setup ===
source ~/.bashrc
cd $SLURM_SUBMIT_DIR
source .venv/bin/activate

# Pick a port (can be any free one)
PORT=8000
NODE_IP=$(hostname -I | awk '{print $1}')

echo "🚀 Starting FastAPI server on GPU node..."
echo "Node IP: $NODE_IP"
echo "Port: $PORT"
echo "curl -X POST http://$NODE_IP:$PORT/generate -H 'Content-Type: application/json' -d '{\"prompt\":\"Explain quantum computing\",\"max_tokens\":150}'"

# === Run FastAPI ===
srun uvicorn inference_api:app --host 0.0.0.0 --port $PORT
