#! /bin/bash
# adjust partition name, image, and mount path to where your model folders live

IMAGE=ghcr.io/yourorg/torch:2.4-cuda12.1
MOUNT_SRC=/        # folder that contains llama-3-8b/ and saved_peft_model/
MOUNT_DST=/work

srun \
     --nodes=1 --gpus-per-node=1 --ntasks-per-node=1 --cpus-per-task=4 \
     --container-image="$IMAGE" \
     --container-mounts="$MOUNT_SRC:$MOUNT_DST" \
     bash -lc '
  which nvidia-smi && nvidia-smi
  python - <<PY
import torch, os
print("cuda.is_available:", torch.cuda.is_available(), "device_count:", torch.cuda.device_count())
PY

  cd /work
  python - <<PY
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE="/work/llama-3-8b"
ADAPTER="/work/saved_peft_model"

tokenizer = AutoTokenizer.from_pretrained(BASE, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16).to("cuda")
model = PeftModel.from_pretrained(model, ADAPTER).to("cuda")
model.eval()

prompt = "Explain diffusion models to a high school student."
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=256, temperature=0.2, do_sample=False)
print(tokenizer.decode(out[0], skip_special_tokens=True))
PY
'
