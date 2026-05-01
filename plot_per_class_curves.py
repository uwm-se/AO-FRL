"""Plot per-round mean / bottom-10 / top-10 accuracy for two methods.

Loads per-class accuracy matrices (rounds × 100) for the full AO-FRL run
and the K=5 partial + no-budget ablation; then for each round computes
the mean, bottom-10 mean (lowest 10 classes that round), and top-10 mean
(highest 10 classes that round). Plots all six curves on a single axis.
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def per_round_stats(npy_path):
    """Returns rounds-aligned arrays: mean, bottom-10 mean, top-10 mean (in %)."""
    mat = np.load(npy_path)  # (rounds, n_classes)
    n_rounds = mat.shape[0]
    means, bot10, top10 = [], [], []
    for r in range(n_rounds):
        row = np.sort(mat[r])
        means.append(row.mean())
        bot10.append(row[:10].mean())
        top10.append(row[-10:].mean())
    return (np.arange(1, n_rounds + 1),
            np.array(means) * 100,
            np.array(bot10) * 100,
            np.array(top10) * 100)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--full",
                   default="results/baseline_tbase500_per_class/AO-FRL_per_class.npy")
    p.add_argument("--partial",
                   default="results/partial_K5_no_budget/AO-FRL_per_class.npy")
    p.add_argument("--out",
                   default="results/partial_K5_compare_v2/per_class_curves.png")
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    rA, mA, bA, tA = per_round_stats(args.full)
    rB, mB, bB, tB = per_round_stats(args.partial)

    fig, ax = plt.subplots(figsize=(12, 6.5))

    # Three metric colors
    C_MEAN = "tab:blue"
    C_BOT  = "tab:red"
    C_TOP  = "tab:green"

    # Solid = full AO-FRL  ;  Dashed = K=5 partial + no budget
    ax.plot(rA, tA, "-",  color=C_TOP,  lw=2.2,
            label=f"Top-10 mean — Full AO-FRL  (final {tA[-1]:.1f}%)")
    ax.plot(rB, tB, "--", color=C_TOP,  lw=2.2,
            label=f"Top-10 mean — K=5 partial+no budget  (final {tB[-1]:.1f}%)")

    ax.plot(rA, mA, "-",  color=C_MEAN, lw=2.5,
            label=f"Mean acc — Full AO-FRL  (final {mA[-1]:.1f}%)")
    ax.plot(rB, mB, "--", color=C_MEAN, lw=2.5,
            label=f"Mean acc — K=5 partial+no budget  (final {mB[-1]:.1f}%)")

    ax.plot(rA, bA, "-",  color=C_BOT,  lw=2.2,
            label=f"Bottom-10 mean — Full AO-FRL  (final {bA[-1]:.1f}%)")
    ax.plot(rB, bB, "--", color=C_BOT,  lw=2.2,
            label=f"Bottom-10 mean — K=5 partial+no budget  (final {bB[-1]:.1f}%)")

    ax.set_xlabel("Communication Round", fontsize=13)
    ax.set_ylabel("Test Accuracy (%)", fontsize=13)
    ax.set_title("Per-Round Mean / Top-10 / Bottom-10 Accuracy  "
                 "(σ=0.02, AO-FRL vs K=5 partial + no budget)",
                 fontsize=12)
    ax.legend(loc="lower right", fontsize=9.5, ncol=1)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 100)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    plt.close(fig)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
