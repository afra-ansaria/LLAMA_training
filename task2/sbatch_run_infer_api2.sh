#!/bin/bash -l
#SBATCH --job-name=llama-infer
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/O-%x_%j.txt
#SBATCH --error=logs/E-%x_%j.txt

set -euo pipefail
source ~/.bashrc
cd "$SLURM_SUBMIT_DIR"
source /root/LLAMA_training/.venv/bin/activate

PORT=8000
export HF_HOME="${HF_HOME:-$PWD/.hf-cache}"

echo "PYTHON=$(which python)"
python -c 'import transformers, torch, sys; print("exe:", sys.executable); print("transformers:", transformers.__version__); print("CUDA avail:", torch.cuda.is_available())'

# --- start uvicorn on GPU node ---
srun --gres=gpu:1 --kill-on-bad-exit=1 \
  "$(which python)" -m uvicorn inference:app \
  --host 0.0.0.0 --port "$PORT" --log-level info --lifespan on &
UVICORN_PID=$!

# --- wait until port is open ---
echo "[wait] Waiting for Uvicorn to start (port $PORT)..."
srun --ntasks=1 --overlap bash -lc "
  for i in \$(seq 1 120); do
    if ss -ltn '( sport = :$PORT )' 2>/dev/null | grep -q LISTEN; then
      echo '[ready] Port $PORT is now listening.'
      exit 0
    fi
    sleep 0.5
  done
  echo '[timeout] Server did not start within 60s.'; exit 1
"

# --- now send the request ---
echo "[test] POST /generate"
srun --ntasks=1 --overlap bash -lc "
  echo '[diag] Host:' \$(hostname)
  echo '[diag] IP:  ' \$(hostname -I | awk '{print \$1}')
  curl -s --noproxy '*' http://127.0.0.1:${PORT}/generate \
    -H 'Content-Type: application/json' \
    -d '{\"prompt\":\"Explain quantum computing in one sentence.\",\"max_tokens\":64}'
  echo
"

NODE_IP=$(srun --ntasks=1 --overlap bash -lc "hostname -I | awk '{print \$1}'")
echo "🚀 Server running on ${NODE_IP}:${PORT}"
trap 'kill $UVICORN_PID 2>/dev/null || true' TERM INT
wait $UVICORN_PID

