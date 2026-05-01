"""Plot AO-FRL accuracy + macro-F1 vs round for 3 encoder variants.

All runs use σ=0.02, δ=1e-6, 60 rounds, patience=10, α=0.3, 20 clients.
Encoders compared:
  - ImageNet frozen     (no fine-tune)
  - Scheme A: CE + L2 anchor (supervised fine-tune on 5K decoder pool)
  - Scheme B: SupCon (contrastive fine-tune on 5K decoder pool)
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
    p.add_argument("--imagenet",
                   default="results/aofrl60/sigma0.02/AO-FRL_rounds.csv")
    p.add_argument("--scheme_a",
                   default="results/encoder_compare/scheme_a/AO-FRL_rounds.csv")
    p.add_argument("--scheme_b",
                   default="results/encoder_compare/scheme_b/AO-FRL_rounds.csv")
    p.add_argument("--out_dir",
                   default="results/encoder_compare/figures")
    args = p.parse_args()

    runs = [
        ("ImageNet frozen (no fine-tune)",
         args.imagenet,
         "tab:gray", "-", 2.5, None, None),
        ("Scheme A: CE + L2 anchor  (5K aux)",
         args.scheme_a,
         "tab:blue", "-", 2.5, "o", 4),
        ("Scheme B: SupCon contrastive  (5K aux)",
         args.scheme_b,
         "tab:red", "--", 2.5, "s", 4),
    ]

    os.makedirs(args.out_dir, exist_ok=True)

    for metric, ylabel, fname, title in [
        ("accuracy", "Test Accuracy (%)",
         "encoder_compare_acc_vs_round.png",
         "Encoder Ablation — Accuracy vs Round  (AO-FRL σ=0.02, ε≈265, 60 rounds, patience=10)"),
        ("macro_f1", "Macro-F1 (%)",
         "encoder_compare_f1_vs_round.png",
         "Encoder Ablation — Macro-F1 vs Round  (AO-FRL σ=0.02, ε≈265, 60 rounds, patience=10)"),
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
