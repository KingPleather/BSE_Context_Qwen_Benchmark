"""
Make the report graphs from benchmark_prefill.py output.

Usage:
  python plot_results.py results/prefill_sdpa.csv --peak-tflops 312 --eff 0.6
  python plot_results.py results/prefill_sdpa.csv results/prefill_eager.csv --peak-tflops 65

--peak-tflops : your GPU's dense BF16/FP16 peak (A100=312, H100 SXM=989, L4=121, T4=65)
--eff         : assumed achievable fraction of peak for the prediction line
"""

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


def label_of(path):
    return os.path.splitext(os.path.basename(path))[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csvs", nargs="+")
    p.add_argument("--peak-tflops", type=float, required=True)
    p.add_argument("--eff", type=float, default=0.6)
    p.add_argument("--outdir", default="figures")
    args = p.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    dfs = {label_of(c): pd.read_csv(c).query("status == 'ok'") for c in args.csvs}

    # 1. Latency vs prompt length, with compute-bound prediction
    plt.figure(figsize=(6, 4.5))
    for name, df in dfs.items():
        plt.plot(df.S, df.latency_ms_median, "o-", label=f"measured ({name})")
    ref = next(iter(dfs.values()))
    pred_ms = ref.flops_total / (args.peak_tflops * 1e12 * args.eff) * 1000
    plt.plot(ref.S, pred_ms, "k--", label=f"predicted (FLOPs / {args.eff:.0%} of peak)")
    plt.xscale("log", base=2); plt.yscale("log")
    plt.xlabel("Prompt length S (tokens)"); plt.ylabel("Prefill latency (ms)")
    plt.title("Prefill latency vs prompt length"); plt.grid(True, which="both", alpha=0.3)
    plt.legend(); plt.tight_layout()
    plt.savefig(f"{args.outdir}/latency_vs_S.png", dpi=200)

    # 2. Peak memory vs prompt length, with prediction
    plt.figure(figsize=(6, 4.5))
    for name, df in dfs.items():
        plt.plot(df.S, df.peak_alloc_bytes / 1e9, "o-", label=f"measured peak ({name})")
        plt.plot(df.S, df.mem_total_pred / 1e9, "--", label=f"predicted: weights+KV+logits ({name})")
    plt.xscale("log", base=2)
    plt.xlabel("Prompt length S (tokens)"); plt.ylabel("Memory (GB)")
    plt.title("Peak GPU memory vs prompt length"); plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(f"{args.outdir}/memory_vs_S.png", dpi=200)

    # 3. Achieved throughput (TFLOP/s) vs prompt length
    # Runs often overlap almost exactly, so each gets its own marker/line style,
    # and the y-axis is zoomed to the data (GPU peak is stated in the title
    # instead of drawn, since a line at the peak squashes the data to the floor).
    styles = [("o", "-"), ("s", "--"), ("^", ":"), ("D", "-.")]
    plt.figure(figsize=(6, 4.5))
    for i, (name, df) in enumerate(dfs.items()):
        m, ls = styles[i % len(styles)]
        plt.plot(df.S, df.achieved_tflops, marker=m, ls=ls, markersize=7 - i,
                 fillstyle="none" if i else "full", label=name)
    ymax = max(df.achieved_tflops.max() for df in dfs.values())
    plt.ylim(0, ymax * 1.25)
    plt.xscale("log", base=2)
    plt.xlabel("Prompt length S (tokens)"); plt.ylabel("Achieved TFLOP/s")
    plt.title(f"Hardware utilization vs prompt length\n(GPU peak = {args.peak_tflops:g} TFLOP/s, "
              f"best achieved = {ymax / args.peak_tflops:.1%} of peak)")
    plt.grid(True, alpha=0.3)
    plt.legend(); plt.tight_layout()
    plt.savefig(f"{args.outdir}/tflops_vs_S.png", dpi=200)

    # 4. FLOP breakdown: how much is attention (S^2) vs linear layers
    plt.figure(figsize=(6, 4.5))
    plt.plot(ref.S, ref.flops_attn / ref.flops_total * 100, "o-")
    plt.xscale("log", base=2)
    plt.xlabel("Prompt length S (tokens)"); plt.ylabel("Attention share of FLOPs (%)")
    plt.title("When does the quadratic term matter?"); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{args.outdir}/attention_share.png", dpi=200)

    print(f"Saved figures to {args.outdir}/")


if __name__ == "__main__":
    main()
