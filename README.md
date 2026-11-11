# Context Scaling: Prefill Latency and Memory for Qwen3-4B

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KingPleather/BSE_Context_Qwen_Benchmark/blob/main/prefill_benchmark.ipynb)

**Authors:** Srikar Papineni, Jonathan Tan
**Course project:** Week 4, "Can we benchmark the current frontier?"

## Research question

How closely do prefill latency and GPU memory scale with prompt length compared with theoretical FLOP and memory predictions?

We sweep the prompt length S from 16 to 8192 tokens on Qwen3-4B. For each length we measure the latency of one prefill forward pass and the peak GPU memory, then compare both against analytical predictions derived from the model architecture.

## Model and hardware

| | |
|---|---|
| Model | [`Qwen/Qwen3-4B`](https://huggingface.co/Qwen/Qwen3-4B) (4,022,468,096 parameters) |
| Precision | bfloat16 |
| GPU | NVIDIA Tesla T4, 15 GB (Google Colab) |
| Software | PyTorch 2.14.1+cu130, Transformers 5.22.2, Python 3.13.16 |
| Batch size | 1 |
| Timed runs | 5 per prompt length, after 2 warmup runs |

### Architecture parameters used in the predictions

| Parameter | Value |
|---|---|
| Layers (L) | 36 |
| Hidden size (D) | 2560 |
| Query heads | 32 |
| Key/value heads (GQA) | 8 |
| Head dim | 128 |
| SwiGLU intermediate size | 9728 |
| Vocabulary (tied embedding / LM head) | 151,936 |

## Theoretical predictions

**FLOPs** for a prefill of S tokens (forward pass, batch 1):

- Linear layers: `2 · S · P_layers`, where P_layers ≈ 3.63B is the parameter count of the 36 transformer blocks. Linear in S.
- Attention scores and weighted sum: `2 · L · n_q · d_h · S²` (the causal mask halves the full `4·L·n_q·d_h·S²`). Quadratic in S.
- LM head: `2 · n_logits · V · D`, where n_logits is S (default) or 1 (`logits_to_keep=1`).

**Memory** held after prefill:

- Weights: measured after loading (8.10 GB)
- KV cache: `S · 2 · L · n_kv · d_h · 2 bytes` = 147,456 bytes per token (GQA with 8 KV heads makes this 4x smaller than full multi-head attention)
- Logits: `n_logits · V · 2 bytes`

Predicted latency assumes the GPU is compute bound and reaches 60% of its peak dense throughput.

## Experiments

| ID | Attention kernel | Logits computed | Purpose |
|---|---|---|---|
| A | SDPA | every position (Hugging Face default) | baseline |
| B | SDPA | last position only | removes the S × V logits tensor |
| C | eager | last position only | materializes the full S × S attention matrix |

**Independent variable:** prompt length S ∈ {16, 32, 64, ..., 8192} tokens of random token IDs (fixed seed 0).
**Dependent variables:** prefill latency (CUDA events, median of 5 runs, with min and std), peak allocated and reserved GPU memory (`torch.cuda.max_memory_allocated`).
**Held constant:** model, precision, GPU, batch size 1, `use_cache=True`, warmup count, and trial count.

## Key results

| S | A latency (ms) | B latency (ms) | C latency (ms) | B peak memory (GB) | B predicted memory (GB) | Attention share of FLOPs |
|---|---|---|---|---|---|---|
| 16 | 189.7 | 187.1 | 187.0 | 8.11 | 8.10 | 0.1% |
| 128 | 397.1 | 395.8 | 392.9 | 8.15 | 8.12 | 0.5% |
| 512 | 1652.2 | 1645.4 | 1650.5 | 8.31 | 8.18 | 2.0% |
| 1024 | 3606.7 | 3502.7 | 3536.8 | 8.69 | 8.25 | 4.0% |
| 2048 | 8241.3 | 7464.9 | 7647.5 | 9.84 | 8.40 | 7.7% |
| 4096 | 19279.2 | 18065.9 | 18524.4 | 14.01 | 8.71 | 14.3% |
| 8192 | OOM | OOM | not run | | | |

Run-to-run variation was small. The coefficient of variation (std / median) stayed at or below 1.1% at every point. Full raw data, including min and std, is in [`prefill_results/results/`](prefill_results/results/).

**Observations**

- Below roughly 64 tokens, latency is almost flat (about 187 to 217 ms). At these lengths the cost of streaming the 8 GB of weights and kernel launch overhead dominates, not compute.
- From 128 to 4096 tokens (32x more tokens), latency grows about 46x. This is faster than linear because the quadratic attention term grows from 0.5% to 14% of total FLOPs.
- Measured peak memory matches the weights + KV + logits prediction to within about 0.15 GB up to S = 512. Beyond that, transient activations (MLP intermediates, attention workspace) open a gap of about 5.3 GB at S = 4096, which the prediction does not include.
- Computing logits for only the last position (B vs A) saves about 1.2 s at S = 4096 and removes the S × 151,936 logits tensor.
- **Hardware caveat:** the T4 has no native bfloat16 support, so bf16 math is emulated. Achieved throughput peaked near 2.6 TFLOP/s, about 4% of the T4's 65 TFLOP/s fp16 peak. As a result, measured latency sits well above the 60%-of-peak prediction. Rerunning with `--dtype fp16` on the T4, or using a GPU with native bf16 (L4, A100, H100), should narrow this gap.

## Figures

| | |
|---|---|
| ![Prefill latency vs prompt length](prefill_results/figures/latency_vs_S.png) | ![Peak GPU memory vs prompt length](prefill_results/figures/memory_vs_S.png) |
| ![Achieved TFLOP/s vs prompt length](prefill_results/figures/tflops_vs_S.png) | ![Attention share of FLOPs](prefill_results/figures/attention_share.png) |

## Repository structure

```
.
├── README.md
├── prefill_benchmark.ipynb       # Colab notebook: runs all experiments and plots end to end
└── prefill_results/
    ├── benchmark_prefill.py      # benchmark script (latency, memory, FLOP/memory predictions)
    ├── plot_results.py           # generates the four figures from the CSVs
    ├── results/                  # raw CSVs and *_meta.json (environment, versions, settings)
    └── figures/                  # latency_vs_S, memory_vs_S, tflops_vs_S, attention_share
```

## Reproducing the results

### Option 1: Google Colab (easiest)

1. Click the **Open in Colab** badge above.
2. Choose `Runtime → Change runtime type → GPU` (T4 works; L4 or A100 is faster).
3. Choose `Runtime → Run all`. The notebook installs the dependencies, runs experiments A, B, and C, displays the figures, and downloads `prefill_results.zip`.

### Option 2: Local machine with a CUDA GPU

```bash
git clone https://github.com/KingPleather/BSE_Context_Qwen_Benchmark.git
cd BSE_Context_Qwen_Benchmark/prefill_results
pip install torch transformers accelerate pandas matplotlib

# Experiment A: SDPA, logits for every position
python benchmark_prefill.py --lengths 16,32,64,128,256,512,1024,2048,4096,8192 \
    --out results/A_sdpa_all_logits.csv

# Experiment B: SDPA, last-position logits only
python benchmark_prefill.py --lengths 16,32,64,128,256,512,1024,2048,4096,8192 \
    --logits-to-keep 1 --out results/B_sdpa_last_logit.csv

# Experiment C: eager attention, last-position logits only
python benchmark_prefill.py --attn eager --lengths 16,32,64,128,256,512,1024,2048,4096 \
    --logits-to-keep 1 --out results/C_eager_last_logit.csv

# Figures (set --peak-tflops to your GPU: T4=65, L4=121, A100=312, H100=989)
python plot_results.py results/A_sdpa_all_logits.csv results/B_sdpa_last_logit.csv \
    results/C_eager_last_logit.csv --peak-tflops 65
```

Useful flags for `benchmark_prefill.py`: `--trials` (default 5), `--warmup` (default 2), `--dtype {auto,bf16,fp16}`, `--seed`. The sweep stops cleanly at the first out-of-memory length and records it as `oom`.

## Limitations

- Single GPU (T4) with emulated bf16, so absolute latencies are not representative of production inference hardware.
- Batch size 1 only. Prefill throughput at larger batch sizes is not measured.
- The memory prediction excludes transient activations, so it underestimates peak memory at long contexts.
- Random token IDs are used. Prefill cost does not depend on token content, but this is not a realistic prompt distribution.
- FlashAttention 2 was not tested (not supported on the T4's Turing architecture).

## References

- Qwen Team. *Qwen3 Technical Report*, 2025. arXiv:2505.09388
- Model card: https://huggingface.co/Qwen/Qwen3-4B
