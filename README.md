

```bash
git clone <dir>
cd multi-node-llm-finetuning-slurm
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```



Before running any scripts, ensure you have set up the necessary environment variables:

```bash
export HF_TOKEN=your_huggingface_token
```

Models used - OLAMA
```bash
python download_model.py
```
Dataset used - Alpaca data


```bash
wget -P src/llama_cookbook/datasets https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json
```
