

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```


```
export HF_TOKEN=your_huggingface_token
```

## Download the Model and Dataset

```
# python task1/baseModel/getLlamaModel.py

wget -P src/llama_cookbook/datasets https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json

```

## Task 1: Fine-tuning

To fine-tune the model,

```
chmod +x srun.sh # o
sbatch sbatch.sh
```


## Task 2: Inference

To generate a sample of 5. You can add or change more prompts
```

sbatch sbatch_run_infer_api.sh
```

## Task 3: Compare

Compare the model

```
sbatch sbatch_compare_models.sh
```
