import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models"))
BASE = os.path.join(BASE_DIR, "llama-3-8b")
ADAPTER = os.path.join(BASE_DIR, "save_finetuned_model")

print(f"Loading base model from: {BASE}")
print(f"Loading adapter from: {ADAPTER}")


tokenizer = AutoTokenizer.from_pretrained(BASE, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16).to("cuda")
model = PeftModel.from_pretrained(model, ADAPTER).to("cuda")
model.eval()


prompts = [
    "Explain the concept of transformers to a high school student.",
    "What are the main differences between supervised and unsupervised learning?",
    "Summarize the importance of attention mechanisms in deep learning.",
    "Write a short paragraph about the future of AI in education.",
]


results = []
for i, prompt in enumerate(prompts, start=1):
    print(f"\n[Prompt {i}] {prompt}")
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.2,
            do_sample=False
        )
    response = tokenizer.decode(out[0], skip_special_tokens=True)
    print(f"[Response {i}] {response}\n")
    results.append((prompt, response))


output_path = os.path.join(os.path.dirname(__file__), "responses.txt")
with open(output_path, "w", encoding="utf-8") as f:
    for i, (prompt, response) in enumerate(results, start=1):
        f.write(f"Prompt {i}: {prompt}\n")
        f.write(f"Response {i}: {response}\n")
        f.write("=" * 80 + "\n\n")

print(f" All responses saved to {output_path}")
