"""Compare AO-FRL across three datasets: CIFAR-100, CIFAR-10, SVHN.

All runs use σ=0.02, δ=1e-6, T_base=500, α=0.3 Dirichlet, 20 clients,
60 rounds, patience=10, replay default.
"""

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load(path, key):
    rounds, vals = [], []
    with open(path) as f:
        for r in csv.DictReader(f):
            rounds.append(int(r["round"]))
            vals.append(float(r[key]) * 100.0)
    return np.array(rounds), np.array(vals)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="results/dataset_compare/figures")
    args = p.parse_args()

    runs = [
        ("CIFAR-100  (100 classes)",
         "results/aofrl60/sigma0.02/AO-FRL_rounds.csv",
         "tab:red", "-", 2.5, "o", 3),
        ("CIFAR-10   (10 classes)",
         "results/dataset_cifar10/AO-FRL_rounds.csv",
         "tab:blue", "-", 2.5, "s", 3),
        ("SVHN       (10 classes, digits)",
         "results/dataset_svhn/AO-FRL_rounds.csv",
         "tab:green", "-", 2.5, "^", 3),
    ]

    os.makedirs(args.out_dir, exist_ok=True)

    for metric, ylabel, fname, title in [
        ("accuracy", "Test Accuracy (%)",
         "dataset_compare_acc_vs_round.png",
         "AO-FRL Generalization Across Datasets — Accuracy vs Round\n"
         "σ=0.02, ε≈265, T_base=500, non-IID α=0.3, 20 clients, replay enabled"),
        ("macro_f1", "Macro-F1 (%)",
         "dataset_compare_f1_vs_round.png",
         "AO-FRL Generalization Across Datasets — Macro-F1 vs Round\n"
         "σ=0.02, ε≈265, T_base=500, non-IID α=0.3, 20 clients, replay enabled"),
    ]:
        fig, ax = plt.subplots(figsize=(11, 6.5))
        for label, path, color, ls, lw, marker, mevery in runs:
            if not os.path.exists(path):
                print(f"WARN: missing {path}")
                continue
            rounds, vals = load(path, metric)
            best = vals.max()
            best_round = rounds[vals.argmax()]
            kw = dict(linewidth=lw, color=color, linestyle=ls,
                      label=f"{label}  (best={best:.2f}% @ R{best_round})",
                      marker=marker, markevery=mevery, markersize=5,
                      markerfacecolor="none")
            ax.plot(rounds, vals, **kw)

        ax.set_xlabel("Communication Round", fontsize=13)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=10.5, loc="lower right")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)
        ax.set_ylim(0, 100)
        fig.tight_layout()
        out = os.path.join(args.out_dir, fname)
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
