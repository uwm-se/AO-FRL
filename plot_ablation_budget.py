"""Compare AO-FRL with vs. without dynamic per-class budget allocation.

Both runs share encoder=ImageNet frozen, σ=0.02 (ε≈265), 60 rounds,
patience=10, α_dirichlet=0.3, 20 clients. The only difference:
  - WITH dynamic budget:  --feedback_alpha 1.0  (default)
  - WITHOUT (ablation):   --feedback_alpha 0    (T_c stays at T_base=500)
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
    p.add_argument("--with_dyn",
                   default="results/aofrl60/sigma0.05/AO-FRL_rounds.csv")
    p.add_argument("--without_dyn",
                   default="results/ablation_no_budget_sigma0.05/AO-FRL_rounds.csv")
    p.add_argument("--out_dir",
                   default="results/ablation_no_budget_sigma0.05/figures")
    p.add_argument("--sigma_label", default="σ=0.05, ε≈106")
    args = p.parse_args()

    runs = [
        ("AO-FRL  (with dynamic budget,  α=1.0)",
         args.with_dyn, "tab:orange", "-", 2.5, "o", 2),
        ("AO-FRL  (no dynamic budget,    α=0)",
         args.without_dyn, "tab:gray", "--", 2.5, "s", 2),
    ]

    os.makedirs(args.out_dir, exist_ok=True)

    for metric, ylabel, fname, title in [
        ("accuracy", "Test Accuracy (%)",
         "ablation_budget_acc.png",
         f"Ablation — Dynamic Budget Allocation  (AO-FRL {args.sigma_label}, ImageNet frozen, 60 rounds, patience=10)"),
        ("macro_f1", "Macro-F1 (%)",
         "ablation_budget_f1.png",
         f"Ablation — Dynamic Budget Allocation  (AO-FRL {args.sigma_label}, ImageNet frozen, 60 rounds, patience=10)"),
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
                kw["markersize"] = 5
                kw["markerfacecolor"] = "none"
            ax.plot(rounds, vals, **kw)

        ax.set_xlabel("Communication Round", fontsize=13)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.set_title(title, fontsize=11.5)
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
