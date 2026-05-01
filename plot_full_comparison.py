"""Combined accuracy + F1 plots across all 7 methods.

Reads per-round CSVs from results/full_compare/{baselines, aofrl_*}/<Method>_rounds.csv
and produces two line plots: accuracy vs round and macro-F1 vs round.
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
    p.add_argument("--root", default="results/full_compare")
    p.add_argument("--out_dir", default="results/comparison")
    args = p.parse_args()

    # Each tuple: (label, csv path, color, linestyle, linewidth, marker, marker_every)
    runs = [
        ("Centralized (IID upper bound)",
         f"{args.root}/baselines/Centralized_rounds.csv",
         "tab:gray", "--", 2.5, None, None),
        ("FedAvg",
         f"{args.root}/baselines/FedAvg_rounds.csv",
         "tab:blue", "-", 2.2, None, None),
        ("FedProx (μ=0.1)",
         f"{args.root}/baselines/FedProx_rounds.csv",
         "tab:cyan", "--", 2.0, "o", 8),     # dashed + circle markers
        ("FedAdam",
         f"{args.root}/baselines/FedAdam_rounds.csv",
         "tab:purple", "-", 2.0, None, None),
        ("AO-FRL  σ=0.005 (ε≈969)",
         f"{args.root}/aofrl_weak_sigma0.005/AO-FRL_rounds.csv",
         "tab:green", "-", 2.5, None, None),
        ("AO-FRL  σ=0.02 (ε≈242)",
         f"{args.root}/aofrl_medium_sigma0.02/AO-FRL_rounds.csv",
         "tab:orange", "-", 2.5, None, None),
        ("AO-FRL  σ=0.05 (ε≈97)",
         f"{args.root}/aofrl_strong_sigma0.05/AO-FRL_rounds.csv",
         "tab:red", "-", 2.5, None, None),
    ]

    os.makedirs(args.out_dir, exist_ok=True)

    for metric, ylabel, fname, title in [
        ("accuracy",  "Test Accuracy (%)",
         "full_compare_acc_vs_round.png",
         "Accuracy vs Round — 7 methods on CIFAR-100, non-IID α=0.3, 20 clients"),
        ("macro_f1",  "Macro-F1 (%)",
         "full_compare_f1_vs_round.png",
         "Macro-F1 vs Round — 7 methods on CIFAR-100, non-IID α=0.3, 20 clients"),
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
        ax.set_title(title, fontsize=13)
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
