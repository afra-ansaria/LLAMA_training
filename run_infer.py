# run_infer.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

BASE = "./llama-3-8b"          # the base you downloaded with download_model.py
ADAPTER = "saved_peft_model"    # your finetune --output_dir

# 1) load base
tokenizer = AutoTokenizer.from_pretrained(BASE, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16).to("cuda")

# 2) load LoRA adapter (PEFT)
from peft import PeftModel
model = PeftModel.from_pretrained(model, ADAPTER).to("cuda")
model.eval()

# 3) generate
prompt = "Explain diffusion models to a high school student."
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
with torch.no_grad():
    out = model.generate(
        **inputs, max_new_tokens=256, temperature=0.2, do_sample=False
    )
print(tokenizer.decode(out[0], skip_special_tokens=True))
