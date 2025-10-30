#!/usr/bin/env python
import csv, os, time, math, torch, datetime, threading, subprocess, shutil
from collections import Counter
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, TextStreamer
from peft import PeftModel

# =========================
# Functional quality helpers
# =========================
def distinct_n(tokens, n=1):
    if not tokens:
        return 0.0
    if n == 1:
        return len(set(tokens)) / len(tokens)
    if n > len(tokens):
        return 0.0
    ngrams = set(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1))
    denom = max(1, len(tokens)-n+1)
    return len(ngrams) / denom

def repetition_ratio(tokens, k=3):
    if not tokens:
        return 0.0
    rep, run = 0, 1
    for i in range(1, len(tokens)):
        if tokens[i] == tokens[i-1]:
            run += 1
            if run == k:
                rep += k
            elif run > k:
                rep += 1
        else:
            run = 1
    return rep / len(tokens)

def token_entropy(tokens):
    if not tokens:
        return 0.0
    c = Counter(tokens)
    n = sum(c.values())
    probs = [v / n for v in c.values()]
    return -sum(p * math.log(p + 1e-12) for p in probs)

def jaccard_overlap(a_tokens, b_tokens):
    A, B = set(a_tokens), set(b_tokens)
    if not A and not B:
        return 0.0
    return len(A & B) / max(1, len(A | B))

# =========================
# NVML / nvidia-smi sampler
# =========================
class NVMLSampler:
    """
    Samples GPU utilization, mem, power every `interval_s` in a background thread.
    Uses pynvml if available else falls back to `nvidia-smi --query-gpu=utilization.gpu,power.draw --format=csv,noheader,nounits`.
    """
    def __init__(self, interval_s=0.05, device_index=0):
        self.interval_s = interval_s
        self.device_index = device_index
        self._stop = threading.Event()
        self._thread = None
        self.samples = []  # (t, util%, powerW)
        self.gpu_name = "cpu"
        self.mem_total_gb = None
        self._mode = "none"  # "pynvml" | "nvsmi" | "none"

        try:
            import pynvml
            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._h = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            self.gpu_name = pynvml.nvmlDeviceGetName(self._h).decode()
            self.mem_total_gb = pynvml.nvmlDeviceGetMemoryInfo(self._h).total / (1024**3)
            self._mode = "pynvml"
        except Exception:
            if shutil.which("nvidia-smi"):
                # name/total mem (best-effort once)
                try:
                    name = subprocess.check_output(
                        ["nvidia-smi", f"--id={device_index}", "--query-gpu=name,memory.total",
                         "--format=csv,noheader,nounits"], text=True
                    ).strip().split(",")
                    if name:
                        self.gpu_name = name[0].strip()
                        self.mem_total_gb = float(name[1].strip())/1024.0
                except Exception:
                    pass
                self._mode = "nvsmi"
            else:
                self._mode = "none"

    def _loop_pynvml(self):
        pn = self._pynvml
        while not self._stop.is_set():
            t = time.perf_counter()
            try:
                util = pn.nvmlDeviceGetUtilizationRates(self._h).gpu
                try:
                    power = pn.nvmlDeviceGetPowerUsage(self._h) / 1000.0  # W
                except Exception:
                    power = float("nan")
                self.samples.append((t, float(util), float(power)))
            except Exception:
                pass
            time.sleep(self.interval_s)

    def _loop_nvsmi(self):
        while not self._stop.is_set():
            t = time.perf_counter()
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", f"--id={self.device_index}",
                     "--query-gpu=utilization.gpu,power.draw",
                     "--format=csv,noheader,nounits"],
                    text=True
                ).strip()
                util_s, pow_s = out.split(",")
                util = float(util_s.strip())
                power = float(pow_s.strip())
                self.samples.append((t, util, power))
            except Exception:
                # try util only
                try:
                    out = subprocess.check_output(
                        ["nvidia-smi", f"--id={self.device_index}",
                         "--query-gpu=utilization.gpu",
                         "--format=csv,noheader,nounits"],
                        text=True
                    ).strip()
                    util = float(out)
                    self.samples.append((t, util, float("nan")))
                except Exception:
                    pass
            time.sleep(self.interval_s)

    def start(self):
        if self._mode == "none":
            return
        self._stop.clear()
        target = self._loop_pynvml if self._mode == "pynvml" else self._loop_nvsmi
        self._thread = threading.Thread(target=target, daemon=True)
        self._thread.start()

    def stop(self):
        if self._mode == "none":
            return
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)

    def summary(self):
        if not self.samples:
            return {
                "gpu_util_avg_pct": "",
                "gpu_util_max_pct": "",
                "power_avg_w": "",
                "power_max_w": "",
                "energy_joules": "",
            }
        util = [u for _, u, _ in self.samples if not math.isnan(u)]
        poww = [(t, p) for t, _, p in self.samples if not math.isnan(p)]
        util_avg = sum(util) / len(util) if util else float("nan")
        util_max = max(util) if util else float("nan")

        # Trapezoidal integration for energy (J = W * s)
        energy_j = float("nan")
        power_avg = float("nan")
        power_max = float("nan")
        if len(poww) >= 2:
            ts = [t for t, _ in poww]
            ps = [p for _, p in poww]
            dt = [ts[i+1]-ts[i] for i in range(len(ts)-1)]
            # avg power over interval
            energy_j = sum(0.5*(ps[i]+ps[i+1]) * dt[i] for i in range(len(dt)))
            power_avg = sum(ps)/len(ps)
            power_max = max(ps)

        def fmt(x):
            if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
                return ""
            return round(x, 3) if isinstance(x, float) else x

        return {
            "gpu_util_avg_pct": fmt(util_avg),
            "gpu_util_max_pct": fmt(util_max),
            "power_avg_w": fmt(power_avg),
            "power_max_w": fmt(power_max),
            "energy_joules": fmt(energy_j),
        }

# =========================
# Timing streamer for TTFT/TBT
# =========================
class TimingStreamer(TextStreamer):
    def __init__(self, tokenizer):
        super().__init__(tokenizer, skip_prompt=True)
        self.first_token_time = None
        self.token_times = []  # absolute times per new token

    def put(self, value):
        now = time.perf_counter()
        if self.first_token_time is None:
            self.first_token_time = now
        self.token_times.append(now)
        super().put(value)

# =========================
# Model loading
# =========================
def load_model(path_or_name, torch_dtype=torch.bfloat16, device="cuda:0"):
    tok = AutoTokenizer.from_pretrained(path_or_name, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    _ = AutoConfig.from_pretrained(path_or_name)

    model = AutoModelForCausalLM.from_pretrained(
        path_or_name,
        torch_dtype=torch_dtype,
        device_map=None,
        low_cpu_mem_usage=True,
    )
    if torch.cuda.is_available():
        model = model.to(device)
    model.eval()
    return model, tok, device

def attach_peft_on_copy(base_model_path, peft_dir, device="cuda:0"):
    base_for_ft, _, _ = load_model(base_model_path, device=device)
    ft = PeftModel.from_pretrained(base_for_ft, peft_dir)
    try:
        ft = ft.merge_and_unload()
    except Exception:
        pass
    if torch.cuda.is_available():
        ft = ft.to(device)
    ft.eval()
    return ft

def encode_prompt(tok, prompt, device, model=None):
    if hasattr(tok, "chat_template") and tok.chat_template:
        msgs = [{"role": "user", "content": prompt}]
        ids = tok.apply_chat_template(msgs, return_tensors="pt")
        return {"input_ids": ids.to(device), "attention_mask": torch.ones_like(ids).to(device)}
    return tok(prompt, return_tensors="pt").to(device)

# =========================
# Self NLL / perplexity of continuation
# =========================
@torch.no_grad()
def continuation_nll(model, ids_full, attn_full, prompt_len):
    if ids_full.shape[1] <= prompt_len + 1:
        return float("nan"), float("nan"), float("nan")
    inp = ids_full[:, :-1]
    tgt = ids_full[:, 1:]
    attn = attn_full[:, :-1]
    logits = model(input_ids=inp, attention_mask=attn, use_cache=False).logits
    logprobs = torch.log_softmax(logits, dim=-1)
    cont_mask = torch.zeros_like(tgt, dtype=torch.bool)
    cont_mask[:, prompt_len-1:] = True
    tgt_lp = logprobs.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    sel = tgt_lp[cont_mask]
    if sel.numel() == 0:
        return float("nan"), float("nan"), float("nan")
    avg_lp = sel.mean()
    avg_nll = -avg_lp.item()
    ppl = math.exp(avg_nll)
    return avg_nll, avg_nll, ppl  # (nll, xent=nll, ppl)

# =========================
# One generation with metrics
# =========================
def gen_one(model, tok, prompt, device, max_new_tokens=200, min_new_tokens=32,
            temperature=0.7, top_p=0.9, gpu_index=0):
    inputs = encode_prompt(tok, prompt, device, model)
    prompt_len = int(inputs["input_ids"].shape[1])

    # GPU metrics
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    # Samplers
    sampler = NVMLSampler(interval_s=0.05, device_index=gpu_index)
    streamer = TimingStreamer(tok)

    # Start sampling & timing
    sampler.start()
    t0 = time.perf_counter()

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id,
            streamer=streamer,   # captures TTFT + per-token times
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    sampler.stop()

    # Latency pieces
    ttft = (streamer.first_token_time - t0) if streamer.first_token_time else float("nan")
    total_time = t1 - t0
    # TBT (avg time between emitted tokens)
    if len(streamer.token_times) >= 2:
        gaps = [streamer.token_times[i+1]-streamer.token_times[i] for i in range(len(streamer.token_times)-1)]
        tbt = sum(gaps)/len(gaps)
    else:
        tbt = float("nan")

    # Token accounting
    out_ids = out[0]
    new_ids = out_ids[prompt_len:]
    total_tokens = int(out_ids.shape[0])
    new_tokens = int(new_ids.shape[0])

    # Text decode
    gen_text = tok.decode(new_ids, skip_special_tokens=True).strip()

    # Throughput
    steady_time = max(1e-9, total_time - (ttft if not math.isnan(ttft) else 0.0))
    throughput_tps = (new_tokens / steady_time) if steady_time > 0 else float("nan")

    # Memory
    peak_vram_gb = float("nan")
    if torch.cuda.is_available():
        peak_vram_gb = torch.cuda.max_memory_allocated() / (1024**3)

    # Quality / diversity
    new_ids_list = new_ids.tolist()
    d1 = distinct_n(new_ids_list, n=1)
    d2 = distinct_n(new_ids_list, n=2)
    rep = repetition_ratio(new_ids_list, k=3)
    ent = token_entropy(new_ids_list)
    prompt_ids_list = inputs["input_ids"][0].tolist()
    jacc = jaccard_overlap(prompt_ids_list, new_ids_list)

    attn_full = torch.ones_like(out_ids).unsqueeze(0).to(device)
    ids_full = out_ids.unsqueeze(0)
    avg_nll, _, ppl = continuation_nll(model, ids_full, attn_full, prompt_len)

    # GPU sampler summary
    sm = sampler.summary()

    return {
        # text
        "text": gen_text,

        # lengths
        "prompt_tokens": prompt_len,
        "new_tokens": new_tokens,
        "total_tokens": total_tokens,

        # latency/efficiency
        "ttft_s": ttft,
        "tbt_avg_s": tbt,
        "e2e_latency_s": total_time,
        "throughput_toks_per_s": throughput_tps,
        "peak_vram_gb": peak_vram_gb,
        **sm,  # gpu_util_avg_pct, gpu_util_max_pct, power_avg_w, power_max_w, energy_joules

        # functional quality
        "distinct_1": d1,
        "distinct_2": d2,
        "repetition_k3": rep,
        "token_entropy": ent,
        "prompt_jaccard": jacc,
        "avg_nll": avg_nll,
        "perplexity": ppl,
    }

# =========================
# Main
# =========================
def main():
    base_model_path = "../models/llama-3-8b"
    peft_model_path = "../models/save_finetuned_model"

    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../results"))
    os.makedirs(results_dir, exist_ok=True)
    out_csv = os.path.join(results_dir, "compare_outputs.csv")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    gpu_index = 0  # maps to CUDA_VISIBLE_DEVICES[0]
    base, tok, _ = load_model(base_model_path, device=device)
    ft = attach_peft_on_copy(base_model_path, peft_model_path, device=device)

    gpu_name = "cpu"
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        gpu_name = f"{props.name} ({props.total_memory/(1024**3):.0f} GB)"

    prompts = [
        "Explain how transformers revolutionized natural language processing in under 200 words.",
        "Describe the differences between supervised, unsupervised, and reinforcement learning with one practical example each.",
        "Write a short motivational message for a developer debugging code at 3 AM.",
        "Summarize the key steps involved in deploying a large language model to production.",
        "If you could give one piece of advice to future AI researchers, what would it be?"
    ]

    run_ts = datetime.datetime.utcnow().isoformat() + "Z"
    rows = []
    for i, p in enumerate(prompts, 1):
        base_m = gen_one(base, tok, p, device, gpu_index=gpu_index)
        ft_m   = gen_one(ft,   tok, p, device, gpu_index=gpu_index)

        print(f"\n[{i}] {p}\nBASE: {base_m['text'][:120]}...\nFT  : {ft_m['text'][:120]}...\n")

        rows.append({
            "run_timestamp_utc": run_ts,
            "gpu": gpu_name,
            "id": i,
            "prompt": p,

            # outputs
            "base_output": base_m["text"],
            "ft_output":   ft_m["text"],

            # lengths
            "prompt_tokens": base_m["prompt_tokens"],
            "base_new_tokens": base_m["new_tokens"],
            "ft_new_tokens":   ft_m["new_tokens"],
            "base_total_tokens": base_m["total_tokens"],
            "ft_total_tokens":   ft_m["total_tokens"],

            # latency & efficiency
            "base_ttft_s": round(base_m["ttft_s"], 4) if not math.isnan(base_m["ttft_s"]) else "",
            "ft_ttft_s":   round(ft_m["ttft_s"], 4) if not math.isnan(ft_m["ttft_s"]) else "",
            "base_tbt_avg_s": round(base_m["tbt_avg_s"], 4) if not math.isnan(base_m["tbt_avg_s"]) else "",
            "ft_tbt_avg_s":   round(ft_m["tbt_avg_s"], 4) if not math.isnan(ft_m["tbt_avg_s"]) else "",
            "base_e2e_latency_s": round(base_m["e2e_latency_s"], 3),
            "ft_e2e_latency_s":   round(ft_m["e2e_latency_s"], 3),
            "base_throughput_tps": round(base_m["throughput_toks_per_s"], 2) if not math.isnan(base_m["throughput_toks_per_s"]) else "",
            "ft_throughput_tps":   round(ft_m["throughput_toks_per_s"], 2) if not math.isnan(ft_m["throughput_toks_per_s"]) else "",
            "base_peak_vram_gb": round(base_m["peak_vram_gb"], 3) if not math.isnan(base_m["peak_vram_gb"]) else "",
            "ft_peak_vram_gb":   round(ft_m["peak_vram_gb"], 3) if not math.isnan(ft_m["peak_vram_gb"]) else "",

            # GPU util / power / energy
            "base_gpu_util_avg_pct": base_m["gpu_util_avg_pct"],
            "base_gpu_util_max_pct": base_m["gpu_util_max_pct"],
            "base_power_avg_w": base_m["power_avg_w"],
            "base_power_max_w": base_m["power_max_w"],
            "base_energy_j": base_m["energy_joules"],

            "ft_gpu_util_avg_pct": ft_m["gpu_util_avg_pct"],
            "ft_gpu_util_max_pct": ft_m["gpu_util_max_pct"],
            "ft_power_avg_w": ft_m["power_avg_w"],
            "ft_power_max_w": ft_m["power_max_w"],
            "ft_energy_j": ft_m["energy_joules"],

            # functional quality
            "base_distinct_1": round(base_m["distinct_1"], 4),
            "base_distinct_2": round(base_m["distinct_2"], 4),
            "base_repetition_k3": round(base_m["repetition_k3"], 4),
            "base_token_entropy": round(base_m["token_entropy"], 4),
            "base_prompt_jaccard": round(base_m["prompt_jaccard"], 4),
            "base_avg_nll": round(base_m["avg_nll"], 4) if not math.isnan(base_m["avg_nll"]) else "",
            "base_perplexity": round(base_m["perplexity"], 3) if not math.isnan(base_m["perplexity"]) else "",

            "ft_distinct_1": round(ft_m["distinct_1"], 4),
            "ft_distinct_2": round(ft_m["distinct_2"], 4),
            "ft_repetition_k3": round(ft_m["repetition_k3"], 4),
            "ft_token_entropy": round(ft_m["token_entropy"], 4),
            "ft_prompt_jaccard": round(ft_m["prompt_jaccard"], 4),
            "ft_avg_nll": round(ft_m["avg_nll"], 4) if not math.isnan(ft_m["avg_nll"]) else "",
            "ft_perplexity": round(ft_m["perplexity"], 3) if not math.isnan(ft_m["perplexity"]) else "",
        })

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"comparison results saved to: {out_csv}")

if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
