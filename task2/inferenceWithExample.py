# run_infer.py. Simple example file. You can change the prompt below. 
import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models"))
BASE = os.path.join(BASE_DIR, "llama-3-8b")
ADAPTER = os.path.join(BASE_DIR, "save_finetuned_model")

print(f"Loading base model from: {BASE}")
print(f"Loading adapter from: {ADAPTER}")

# 1) load base
tokenizer = AutoTokenizer.from_pretrained(BASE, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16).to("cuda")

# 2) load LoRA adapter (PEFT)
from peft import PeftModel
model = PeftModel.from_pretrained(model, ADAPTER).to("cuda")
model.eval()

# 3) generate
prompt = "Explain the concept of transformers to a high school student."
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
with torch.no_grad():
    out = model.generate(
        **inputs, max_new_tokens=256, temperature=0.2, do_sample=False
    )
print(tokenizer.decode(out[0], skip_special_tokens=True))
