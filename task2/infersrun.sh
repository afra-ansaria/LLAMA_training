
#srun --gres=gpu:4 --cpus-per-task=4 --mem=32G   uvicorn inferAPI:app --host 0.0.0.0 --port 8000
#For curl run the inferAPI.py

# curl -X POST http://10.6.60.253:8000/generate   -H "Content-Type: application/json"   -d '{"prompt":"Explain quantum computing to a 15-year-old. Keep your answers to 150 words","max_tokens":200}'