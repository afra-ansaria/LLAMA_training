#!/usr/bin/env python
import csv, os, time, torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from peft import PeftModel

def load_model(path_or_name, torch_dtype=torch.bfloat16, device="cuda:0"):
    tok = AutoTokenizer.from_pretrained(path_or_name, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    _ = AutoConfig.from_pretrained(path_or_name)

    model = AutoModelForCausalLM.from_pretrained(
        path_or_name,
        torch_dtype=torch_dtype,
        device_map=None,          # avoid cross-GPU sharding
    )
    if torch.cuda.is_available():
        model = model.to(device)
    model.eval()
    return model, tok, device

def attach_peft_on_copy(base_model_path, peft_dir, device="cuda:0"):
    # load a fresh copy for FT so we don't mutate the baseline
    base_for_ft, _, _ = load_model(base_model_path, device=device)
    ft = PeftModel.from_pretrained(base_for_ft, peft_dir)
    try:
        ft = ft.merge_and_unload()    # produce a plain merged model
    except Exception:
        pass
    if torch.cuda.is_available():
        ft = ft.to(device)
    ft.eval()
    return ft

def encode_prompt(tok, prompt, device, model=None):
    # Use chat template *only* if it exists
    if hasattr(tok, "chat_template") and tok.chat_template:
        msgs = [{"role": "user", "content": prompt}]
        ids = tok.apply_chat_template(msgs, return_tensors="pt")
        return {"input_ids": ids.to(device), "attention_mask": torch.ones_like(ids).to(device)}
    # Otherwise, plain text
    return tok(prompt, return_tensors="pt").to(device)

def gen_one(model, tok, prompt, device, max_new_tokens=200, min_new_tokens=32,
            temperature=0.7, top_p=0.9):
    inputs = encode_prompt(tok, prompt, device, model)
    with torch.no_grad():
        t0 = time.perf_counter()
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,   # <-- prevents 0–1 token stops
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id,
        )
        dt = time.perf_counter() - t0

    # Strip the prompt tokens
    gen_only = out[0][inputs["input_ids"].shape[1]:]
    text = tok.decode(gen_only, skip_special_tokens=True).strip()
    toks = int(gen_only.shape[0])
    tps = toks / dt if dt > 0 else float("nan")
    return text, dt, tps, toks

def main():
    base_model_path = "../models/llama-3-8b"
    peft_model_path = "../models/save_finetuned_model"

    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../results"))
    os.makedirs(results_dir, exist_ok=True)
    out_csv = os.path.join(results_dir, "compare_outputs.csv")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    base, tok, _ = load_model(base_model_path, device=device)
    ft = attach_peft_on_copy(base_model_path, peft_model_path, device=device)

    prompts = [
    "Explain how transformers revolutionized natural language processing in under 200 words.",
    "Describe the differences between supervised, unsupervised, and reinforcement learning with one practical example each.",
    "Write a short motivational message for a developer debugging code at 3 AM.",
    "Summarize the key steps involved in deploying a large language model to production.",
    "If you could give one piece of advice to future AI researchers, what would it be?"
    ]


    rows = []
    for i, p in enumerate(prompts, 1):
        base_txt, b_dt, b_tps, b_ntok = gen_one(base, tok, p, device)
        ft_txt, f_dt, f_tps, f_ntok   = gen_one(ft,   tok, p, device)
        print(f"\n[{i}] {p}\nBASE: {base_txt[:120]}...\nFT  : {ft_txt[:120]}...\n")
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

    # write once per run; if you want to accumulate across runs, switch "w"->"a" and guard the header
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"comparison results saved to: {out_csv}")

if __name__ == "__main__":
    main()
