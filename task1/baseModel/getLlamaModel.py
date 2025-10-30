from huggingface_hub import snapshot_download

snapshot_download(repo_id="meta-llama/Meta-Llama-3-8B", local_dir="../..models/llama-3-8b")