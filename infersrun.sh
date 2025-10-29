#srun --gres=gpu:4 --cpus-per-task=4 --mem=32G   uvicorn serve_infer:app --host 0.0.0.0 --port 8000
