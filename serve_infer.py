from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

app = FastAPI()

BASE = "./llama-3-8b"
ADAPTER = "./saved_peft_model"

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
