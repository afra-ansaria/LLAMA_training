#!/usr/bin/env python
"""
Compare a base model (llama-3-8b) with its fine-tuned PEFT (LoRA) adapter.
Loads both from the shared models/ directory and writes results to results/models/compare_outputs.csv.
"""

import csv, os, time
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from peft import PeftModel


# ----------------------------------------------------------
# Helper functions
# ----------------------------------------------------------
def load_model(path_or_name, tokenizer_name=None, torch_dtype=torch.bfloat16, device="cuda"):
    tokenizer_name = tokenizer_name or path_or_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    cfg = AutoConfig.from_pretrained(path_or_name)
    model = AutoModelForCausalLM.from_pretrained(
        path_or_name,
        torch_dtype=torch_dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    if device == "cuda" and torch.cuda.is_available():
        model = model.to(device)
    model.eval()
    return model, tokenizer, cfg


def attach_peft(base_model, peft_dir):
    print(f"🔗 Attaching PEFT adapter from: {peft_dir}")
    ft_model = PeftModel.from_pretrained(base_model, peft_dir)
    try:
        ft_model = ft_model.merge_and_unload()
    except Exception:
        pass
    ft_model.eval()
    return ft_model


def gen_one(model, tokenizer, prompt, max_new_tokens=150, temperature=0.2, top_p=0.95):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        t0 = time.perf_counter()
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0),
            temperature=temperature,
            top_p=top_p,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        dt = time.perf_counter() - t0
    text = tokenizer.decode(out_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    toks = out_ids.shape[1] - inputs["input_ids"].shape[1]
    tps = toks / dt if dt > 0 else float("nan")
    return text.strip(), dt, tps, toks


# ----------------------------------------------------------
# Main
# ----------------------------------------------------------
def main():
    # Both models now live in ../models/
    base_model_path = "../models/llama-3-8b"
    peft_model_path = "../models/saved_peft_model"

    # Save comparison CSV under results/models/
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models"))
    os.makedirs(results_dir, exist_ok=True)
    out_csv = os.path.join(results_dir, "compare_outputs.csv")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base, tok, _ = load_model(base_model_path, device=device)
    ft = attach_peft(base, peft_model_path)

    prompts = [
        "Explain quantum computing to a 30-year-old in ≤150 words.",
        "Give three practical tips to optimize Python for numerical workloads.",
        "What’s the difference between data, tensor, and pipeline parallelism? Keep it concise.",
        "Summarize LayerNorm vs. RMSNorm and when to prefer each.",
        "Describe a simple caching strategy for high-throughput LLM inference APIs.",
    ]

    rows = []
    for i, p in enumerate(prompts, 1):
        base_txt, b_dt, b_tps, b_ntok = gen_one(base, tok, p)
        ft_txt, f_dt, f_tps, f_ntok = gen_one(ft, tok, p)
        rows.append({
            "id": i,
            "prompt": p,
            "base_output": base_txt,
            "ft_output": ft_txt,
            "base_latency_s": round(b_dt, 3),
            "ft_latency_s": round(f_dt, 3),
            "base_tok_per_s": round(b_tps, 2),
            "ft_tok_per_s": round(f_tps, 2),
            "base_new_tokens": b_ntok,
            "ft_new_tokens": f_ntok,
        })

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"comparison results to {out_csv} done")


if __name__ == "__main__":
    main()
