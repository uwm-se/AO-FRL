"""Compare AO-FRL across the three DP noise settings.

Reads results/dp_{strong,medium,weak}_sigma*/AO-FRL_rounds.csv and
inversion_summary.json, produces two line plots:

  1. Classification utility — test accuracy vs round, one line per σ.
  2. Privacy protection    — reconstruction PSNR vs σ, with the clean
                             reconstruction ceiling for reference.

Usage:
    python plot_dp_comparison.py
    python plot_dp_comparison.py --results_root results --out_dir results/comparison
"""

import argparse
import csv
import glob
import json
import math
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def epsilon_from_sigma(sigma: float, delta: float = 1e-5,
                       clip_C: float = 1.0) -> float:
    """Inverse of the Gaussian-mechanism σ formula. Used to display the
    actual ε implied by a (σ, δ, clip_C) tuple, since experiments may
    override σ directly without re-deriving ε."""
    return clip_C * math.sqrt(2.0 * math.log(1.25 / delta)) / sigma


def find_runs(results_root: str):
    """Return list of (tag, sigma, dir) for runs found under results_root."""
    runs = []
    for d in sorted(glob.glob(os.path.join(results_root, "dp_*_sigma*"))):
        if not os.path.isdir(d):
            continue
        m = re.match(r"dp_(\w+)_sigma([\d.]+)$", os.path.basename(d))
        if not m:
            continue
        tag, sigma = m.group(1), float(m.group(2))
        runs.append((tag, sigma, d))
    # Order: strong (largest σ) -> weak (smallest σ)
    runs.sort(key=lambda r: -r[1])
    return runs


def load_rounds_csv(path: str):
    rounds, accs, f1s = [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            rounds.append(int(row["round"]))
            accs.append(float(row["accuracy"]))
            f1s.append(float(row["macro_f1"]))
    return np.array(rounds), np.array(accs), np.array(f1s)


def load_inversion(path: str):
    with open(path) as f:
        return json.load(f)


def plot_accuracy(runs, out_path: str):
    """Plot 1: Test accuracy vs round, one line per σ."""
    fig, ax = plt.subplots(figsize=(10, 6))

    color_map = {"strong": "tab:red", "medium": "tab:orange",
                 "weak": "tab:green"}

    for tag, sigma, d in runs:
        csv_path = os.path.join(d, "AO-FRL_rounds.csv")
        if not os.path.exists(csv_path):
            print(f"WARN: missing {csv_path}, skipping")
            continue
        rounds, accs, _ = load_rounds_csv(csv_path)
        inv = {}
        inv_path = os.path.join(d, "inversion_summary.json")
        if os.path.exists(inv_path):
            inv = load_inversion(inv_path)
        delta = inv.get("delta", 1e-5)
        clip_C = inv.get("clip_C", 1.0)
        # Derive actual ε from the σ that was used (not the arg default).
        eps_actual = epsilon_from_sigma(sigma, delta=delta, clip_C=clip_C)
        ax.plot(rounds, accs * 100, linewidth=2,
                color=color_map.get(tag, None),
                label=f"σ={sigma} ({tag}, ε≈{eps_actual:.1f})")

    ax.set_xlabel("Communication Round", fontsize=13)
    ax.set_ylabel("Test Accuracy (%)", fontsize=13)
    ax.set_title("AO-FRL Classification Utility under Different DP Noise",
                 fontsize=14)
    ax.legend(fontsize=11, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_psnr(runs, out_path: str):
    """Plot 2: PSNR vs σ — privacy protection level.

    Two lines:
      - Clean reconstruction (decoder applied to undistorted unit-norm
        embedding) — the ceiling, σ-independent in expectation but plotted
        across σ for visual reference.
      - DP-noisy reconstruction at each σ — drops as σ increases.
    """
    sigmas, clean_means, clean_stds = [], [], []
    noisy_means, noisy_stds = [], []
    tags = []
    eps_vals = []

    for tag, sigma, d in runs:
        inv_path = os.path.join(d, "inversion_summary.json")
        if not os.path.exists(inv_path):
            print(f"WARN: missing {inv_path}, skipping")
            continue
        inv = load_inversion(inv_path)
        sigmas.append(sigma)
        clean_means.append(inv["psnr_clean_mean"])
        clean_stds.append(inv["psnr_clean_std"])
        noisy_means.append(inv["psnr_noisy_mean"])
        noisy_stds.append(inv["psnr_noisy_std"])
        tags.append(tag)
        eps_vals.append(epsilon_from_sigma(sigma,
                                           delta=inv.get("delta", 1e-5),
                                           clip_C=inv.get("clip_C", 1.0)))

    if not sigmas:
        print("No inversion results found.")
        return

    sigmas = np.array(sigmas)
    clean_means = np.array(clean_means)
    clean_stds = np.array(clean_stds)
    noisy_means = np.array(noisy_means)
    noisy_stds = np.array(noisy_stds)

    # Sort by σ ascending for a clean line plot.
    order = np.argsort(sigmas)
    sigmas = sigmas[order]
    clean_means = clean_means[order]; clean_stds = clean_stds[order]
    noisy_means = noisy_means[order]; noisy_stds = noisy_stds[order]
    tags = [tags[i] for i in order]
    eps_vals = [eps_vals[i] for i in order]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.errorbar(sigmas, clean_means, yerr=clean_stds, marker="o",
                linewidth=2, markersize=8, capsize=4,
                color="tab:blue",
                label="Clean reconstruction (no DP) — ceiling")
    ax.errorbar(sigmas, noisy_means, yerr=noisy_stds, marker="s",
                linewidth=2, markersize=8, capsize=4,
                color="tab:red",
                label="DP-noisy reconstruction (privacy threat)")

    # Annotate each point with epsilon
    for s, ny, eps in zip(sigmas, noisy_means, eps_vals):
        if eps is not None:
            ax.annotate(f"ε≈{eps:.1f}", (s, ny),
                        textcoords="offset points", xytext=(8, -12),
                        fontsize=9, color="tab:red")

    ax.set_xlabel("DP Noise Scale σ (clip_C = 1.0, unit-norm embedding)",
                  fontsize=13)
    ax.set_ylabel("Reconstruction PSNR (dB)", fontsize=13)
    ax.set_xscale("log")
    ax.set_title("Privacy Protection: Decoder Inversion PSNR vs DP Noise",
                 fontsize=14)
    ax.legend(fontsize=11, loc="best")
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_root", default="results")
    p.add_argument("--out_dir", default="results/comparison")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    runs = find_runs(args.results_root)
    print(f"Found {len(runs)} DP runs:")
    for tag, sigma, d in runs:
        print(f"  σ={sigma} ({tag}) -> {d}")

    plot_accuracy(runs, os.path.join(args.out_dir, "dp_accuracy_vs_round.png"))
    plot_psnr(runs, os.path.join(args.out_dir, "dp_psnr_vs_sigma.png"))


if __name__ == "__main__":
    main()
