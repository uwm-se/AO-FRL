"""Plot accuracy + macro-F1 vs round for 4 baseline methods.

Reads per-round CSVs from results/baselines60/<Method>_rounds.csv.
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
        for row in csv.DictReader(f):
            rounds.append(int(row["round"]))
            vals.append(float(row[key]) * 100.0)
    return np.array(rounds), np.array(vals)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="results/baselines60")
    p.add_argument("--aofrl_root", default="results/aofrl60")
    p.add_argument("--out_dir", default="results/baselines60/figures")
    args = p.parse_args()

    runs = [
        ("Centralized (IID upper bound)",
         f"{args.root}/Centralized_rounds.csv",
         "tab:gray", "--", 2.5, None, None),
        ("FedAvg",
         f"{args.root}/FedAvg_rounds.csv",
         "tab:blue", "-", 2.2, None, None),
        ("FedProx (μ=1.0, local_epochs=10)",
         f"{args.root}/FedProx_rounds.csv",
         "tab:cyan", "--", 2.0, "o", 6),
        ("FedAdam",
         f"{args.root}/FedAdam_rounds.csv",
         "tab:purple", "-", 2.2, None, None),
        ("AO-FRL  σ=0.005 (ε≈1060)",
         f"{args.aofrl_root}/sigma0.005/AO-FRL_rounds.csv",
         "tab:green", "-", 2.5, None, None),
        ("AO-FRL  σ=0.02  (ε≈265)",
         f"{args.aofrl_root}/sigma0.02/AO-FRL_rounds.csv",
         "tab:orange", "-", 2.5, None, None),
        ("AO-FRL  σ=0.05  (ε≈106)",
         f"{args.aofrl_root}/sigma0.05/AO-FRL_rounds.csv",
         "tab:red", "-", 2.5, None, None),
    ]

    os.makedirs(args.out_dir, exist_ok=True)

    for metric, ylabel, fname, title in [
        ("accuracy", "Test Accuracy (%)",
         "baselines60_acc_vs_round.png",
         "Accuracy vs Round — 7 methods on CIFAR-100, non-IID α=0.3, 20 clients, 60 rounds (patience=10)"),
        ("macro_f1", "Macro-F1 (%)",
         "baselines60_f1_vs_round.png",
         "Macro-F1 vs Round — 7 methods on CIFAR-100, non-IID α=0.3, 20 clients, 60 rounds (patience=10)"),
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
                      label=f"{label}  (best={best:.2f}% @ R{best_round})")
            if marker:
                kw["marker"] = marker
                kw["markevery"] = mevery
                kw["markersize"] = 6
                kw["markerfacecolor"] = "none"
            ax.plot(rounds, vals, **kw)

        ax.set_xlabel("Communication Round", fontsize=13)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=10, loc="lower right")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)
        fig.tight_layout()
        out = os.path.join(args.out_dir, fname)
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
