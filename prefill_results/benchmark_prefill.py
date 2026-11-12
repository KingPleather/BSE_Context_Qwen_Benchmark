"""
Prefill (context scaling) benchmark for Qwen3-4B.

Measures, for each prompt length S:
  - prefill latency (median of N timed runs, CUDA events)
  - peak GPU memory (allocated and reserved)
and saves them next to theoretical FLOP / memory predictions.

Usage (examples):
  python benchmark_prefill.py
  python benchmark_prefill.py --attn eager --out results/eager.csv
  python benchmark_prefill.py --logits-to-keep 1 --out results/last_logit.csv
"""

import argparse
import csv
import json
import os
import platform
import statistics

import torch
import transformers
from transformers import AutoModelForCausalLM

# ---- Qwen3-4B architecture (from config.json / Qwen3 Technical Report) ----
L = 36          # layers
D = 2560        # hidden size
N_Q = 32        # query heads
N_KV = 8        # key/value heads (GQA)
D_H = 128       # head dim
D_FF = 9728     # SwiGLU intermediate size
VOCAB = 151936  # embedding matrix rows (padded vocab)

PARAMS_PER_LAYER = (
    D * N_Q * D_H          # W_Q
    + 2 * D * N_KV * D_H   # W_K, W_V
    + N_Q * D_H * D        # W_O
    + 3 * D * D_FF         # gate, up, down
)
LAYER_PARAMS = L * PARAMS_PER_LAYER   # ~3.63B
EMB_PARAMS = VOCAB * D                # ~0.39B (tied with LM head)


def predicted_flops(S: int, n_logits: int) -> dict:
    """Forward-pass FLOPs for a prefill of S tokens (batch size 1)."""
    linear = 2 * S * LAYER_PARAMS                 # all weight matmuls in the layers
    lm_head = 2 * n_logits * VOCAB * D            # output projection
    attn = 2 * L * N_Q * D_H * S * S              # QK^T + AV with causal mask (half of 4*L*nq*dh*S^2)
    return {"flops_linear": linear, "flops_lm_head": lm_head,
            "flops_attn": attn, "flops_total": linear + lm_head + attn}


def predicted_memory(S: int, n_logits: int, bytes_per_el: int, weight_bytes: int) -> dict:
    """Bytes held at the end of a prefill (excluding transient activations)."""
    kv = S * 2 * L * N_KV * D_H * bytes_per_el    # K and V for every layer
    logits = n_logits * VOCAB * bytes_per_el
    return {"mem_kv_pred": kv, "mem_logits_pred": logits,
            "mem_total_pred": weight_bytes + kv + logits}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--lengths", default="16,32,64,128,256,512,1024,2048,4096,8192,16384,32768",
                   help="comma-separated prompt lengths")
    p.add_argument("--trials", type=int, default=5)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--attn", default="sdpa", choices=["sdpa", "eager", "flash_attention_2"])
    p.add_argument("--logits-to-keep", type=int, default=0,
                   help="0 = logits for every position (HF default), 1 = last position only")
    p.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp16"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/prefill_sdpa.csv")
    args = p.parse_args()

    assert torch.cuda.is_available(), "A CUDA GPU is required."
    device = "cuda"

    if args.dtype == "auto":
        # BF16 tensor cores need compute capability >= 8.0 (A100, L4, H100).
        # Older GPUs like the T4 (7.5) only *emulate* BF16, which is very slow,
        # even though torch.cuda.is_bf16_supported() may still return True.
        major, _ = torch.cuda.get_device_capability(0)
        dtype = torch.bfloat16 if major >= 8 else torch.float16
    else:
        dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    bytes_per_el = 2

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # ---- load model ----
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, attn_implementation=args.attn
    ).to(device).eval()
    torch.cuda.synchronize()
    weight_bytes = torch.cuda.memory_allocated()   # measured memory after loading = weights
    n_params = sum(p.numel() for p in model.parameters())

    meta = {
        "model": args.model, "gpu": torch.cuda.get_device_name(0),
        "dtype": str(dtype), "attn": args.attn, "logits_to_keep": args.logits_to_keep,
        "torch": torch.__version__, "transformers": transformers.__version__,
        "python": platform.python_version(), "n_params": n_params,
        "weight_bytes_measured": weight_bytes, "trials": args.trials, "warmup": args.warmup,
    }
    with open(os.path.splitext(args.out)[0] + "_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))

    gen = torch.Generator().manual_seed(args.seed)
    lengths = [int(x) for x in args.lengths.split(",")]

    fieldnames = ["S", "status", "latency_ms_median", "latency_ms_min", "latency_ms_std",
                  "peak_alloc_bytes", "peak_reserved_bytes",
                  "flops_linear", "flops_lm_head", "flops_attn", "flops_total",
                  "achieved_tflops", "mem_kv_pred", "mem_logits_pred", "mem_total_pred"]

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for S in lengths:
            n_logits = S if args.logits_to_keep == 0 else args.logits_to_keep
            row = {"S": S, **predicted_flops(S, n_logits),
                   **predicted_memory(S, n_logits, bytes_per_el, weight_bytes)}

            # random token ids so prompt content doesn't matter
            ids = torch.randint(0, 150000, (1, S), generator=gen).to(device)
            fwd_kwargs = {"use_cache": True}
            if args.logits_to_keep > 0:
                fwd_kwargs["logits_to_keep"] = args.logits_to_keep

            try:
                with torch.inference_mode():
                    for _ in range(args.warmup):
                        out = model(ids, **fwd_kwargs)
                        del out
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()

                    times = []
                    for _ in range(args.trials):
                        start = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)
                        start.record()
                        out = model(ids, **fwd_kwargs)
                        end.record()
                        torch.cuda.synchronize()
                        times.append(start.elapsed_time(end))  # milliseconds
                        del out

                med = statistics.median(times)
                row.update({
                    "status": "ok",
                    "latency_ms_median": med,
                    "latency_ms_min": min(times),
                    "latency_ms_std": statistics.stdev(times) if len(times) > 1 else 0.0,
                    "peak_alloc_bytes": torch.cuda.max_memory_allocated(),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                    "achieved_tflops": row["flops_total"] / (med / 1000) / 1e12,
                })
                print(f"S={S:>6}  {med:9.2f} ms  "
                      f"peak={row['peak_alloc_bytes']/1e9:6.2f} GB  "
                      f"{row['achieved_tflops']:6.1f} TFLOP/s")
            except torch.cuda.OutOfMemoryError:
                row["status"] = "oom"
                print(f"S={S:>6}  OOM — stopping sweep")
                writer.writerow(row)
                break
            finally:
                del ids
                torch.cuda.empty_cache()

            writer.writerow(row)
            f.flush()

    print(f"Saved results to {args.out}")


if __name__ == "__main__":
    main()
