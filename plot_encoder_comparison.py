"""3-way encoder comparison plot at fixed σ=0.005.

Reads three AO-FRL_rounds.csv files (ImageNet baseline, Scheme A supervised,
Scheme B SupContrast) and plots accuracy vs round on one chart.
"""

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_rounds(path):
    rounds, accs = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            rounds.append(int(row["round"]))
            accs.append(float(row["accuracy"]) * 100)
    return np.array(rounds), np.array(accs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/comparison/encoder_compare_sigma0.005.png")
    args = p.parse_args()

    runs = [
        ("ImageNet ResNet-18 (no fine-tune)",
         "results/dp_weak_sigma0.005/AO-FRL_rounds.csv",
         "tab:blue", "-"),
        ("Scheme A: Supervised CE + anchor reg",
         "results/encoder_compare/scheme_a_supervised/AO-FRL_rounds.csv",
         "tab:orange", "-"),
        ("Scheme B: SupContrast",
         "results/encoder_compare/scheme_b_supcon/AO-FRL_rounds.csv",
         "tab:green", "-"),
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    for label, path, color, ls in runs:
        if not os.path.exists(path):
            print(f"WARN: missing {path}, skipping")
            continue
        rounds, accs = load_rounds(path)
        best = accs.max()
        best_round = rounds[accs.argmax()]
        ax.plot(rounds, accs, linewidth=2, color=color, linestyle=ls,
                label=f"{label} (best={best:.2f}% @ R{best_round})")

    ax.set_xlabel("Communication Round", fontsize=13)
    ax.set_ylabel("Test Accuracy (%)", fontsize=13)
    ax.set_title("Encoder Comparison @ σ=0.005 — 5000-image fine-tuning "
                 "of ResNet-18 does not beat the frozen ImageNet baseline",
                 fontsize=13)
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=150)
    plt.close(fig)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
