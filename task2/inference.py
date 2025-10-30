#This file is to call the inference API. You can call the /generate API to get a response for the prompt
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import os

app = FastAPI()

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models"))
BASE = os.path.join(BASE_DIR, "llama-3-8b")
ADAPTER = os.path.join(BASE_DIR, "save_finetuned_model")

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(BASE, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16).to("cuda")
model = PeftModel.from_pretrained(model, ADAPTER).to("cuda")
model.eval()
print("Model ready")

class Request(BaseModel):
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7

@app.post("/generate")
def generate(req: Request):
    inputs = tokenizer(req.prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
            do_sample=req.temperature > 0
        )
    result = tokenizer.decode(out[0], skip_special_tokens=True)
    return {"prompt": req.prompt, "response": result}
