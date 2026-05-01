"""Per-class accuracy comparison: dynamic budget vs static budget.

Loads per-class accuracy matrices logged by FastEvaluator (rounds × 100)
for two AO-FRL runs differing only in --feedback_alpha (1.0 vs 0). Picks
the best round (argmax mean accuracy) for each, then produces:

  (a) sorted-by-baseline curve + difference shaded
  (b) histogram of per-class accuracies
  (c) summary table: mean / std / min / bottom-10 / bottom-20 means
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_best(npy_path):
    mat = np.load(npy_path)               # (rounds, n_classes)
    best_round = int(np.argmax(mat.mean(axis=1)))
    return mat[best_round], best_round, mat.shape[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--with_dyn",
                   default="results/per_class_with_dyn/AO-FRL_per_class.npy")
    p.add_argument("--no_dyn",
                   default="results/per_class_no_dyn/AO-FRL_per_class.npy")
    p.add_argument("--out_dir", default="results/per_class_compare")
    p.add_argument("--setup_label",
                   default="AO-FRL σ=0.05",
                   help="Setup string for plot titles")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with_acc, with_round, with_total = load_best(args.with_dyn)
    no_acc, no_round, no_total = load_best(args.no_dyn)

    print(f"WITH dynamic:  best round R{with_round+1} / {with_total}, "
          f"mean acc = {with_acc.mean()*100:.2f}%")
    print(f"NO   dynamic:  best round R{no_round+1} / {no_total}, "
          f"mean acc = {no_acc.mean()*100:.2f}%\n")

    # ---- Sort indices by NO-dyn ascending (worst classes leftmost) ----
    order = np.argsort(no_acc)
    with_sorted = with_acc[order] * 100
    no_sorted = no_acc[order] * 100
    diff = (with_acc - no_acc)[order] * 100

    # ============== Figure 1: sorted curve ==============
    fig, ax = plt.subplots(figsize=(13, 5.5))
    x = np.arange(100)
    ax.plot(x, no_sorted, color="tab:gray", lw=1.5, ls="--",
            label=f"No dynamic budget  (mean={no_sorted.mean():.2f}%)")
    ax.plot(x, with_sorted, color="tab:orange", lw=1.7,
            label=f"With dynamic budget  (mean={with_sorted.mean():.2f}%)")

    # Shade where dynamic helps
    ax.fill_between(x, no_sorted, with_sorted,
                    where=with_sorted >= no_sorted,
                    color="green", alpha=0.18, label="dynamic helps")
    ax.fill_between(x, no_sorted, with_sorted,
                    where=with_sorted < no_sorted,
                    color="red", alpha=0.18, label="dynamic hurts")

    ax.set_xlabel("Class rank  (sorted by no-dynamic accuracy ascending)",
                  fontsize=12)
    ax.set_ylabel("Per-class test accuracy (%)", fontsize=12)
    ax.set_title(f"Per-Class Accuracy at Best Round — Dynamic vs Static "
                 f"Budget  ({args.setup_label})", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc="lower right")
    fig.tight_layout()
    out = os.path.join(args.out_dir, "per_class_sorted.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")

    # ============== Figure 2: histogram ==============
    fig, ax = plt.subplots(figsize=(11, 5.5))
    bins = np.linspace(0, 1, 21)
    ax.hist(no_acc, bins=bins, alpha=0.55, color="tab:gray",
            edgecolor="black", label=f"No dynamic  (mean={no_acc.mean()*100:.2f}%)")
    ax.hist(with_acc, bins=bins, alpha=0.55, color="tab:orange",
            edgecolor="black", label=f"With dynamic  (mean={with_acc.mean()*100:.2f}%)")
    ax.set_xlabel("Per-class test accuracy", fontsize=12)
    ax.set_ylabel("# of classes", fontsize=12)
    ax.set_title(f"Distribution of Per-Class Accuracy "
                 f"({args.setup_label}, best round)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out = os.path.join(args.out_dir, "per_class_hist.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")

    # ============== Summary table ==============
    def stats(arr, label):
        s = np.sort(arr)
        return {
            "mean (all)":        f"{arr.mean()*100:.2f}%",
            "median":            f"{np.median(arr)*100:.2f}%",
            "std":               f"{arr.std()*100:.2f}",
            "min":               f"{arr.min()*100:.2f}%",
            "mean of bottom-5":  f"{s[:5].mean()*100:.2f}%",
            "mean of bottom-10": f"{s[:10].mean()*100:.2f}%",
            "mean of bottom-20": f"{s[:20].mean()*100:.2f}%",
            "mean of top-10":    f"{s[-10:].mean()*100:.2f}%",
        }

    s_no = stats(no_acc, "no")
    s_yes = stats(with_acc, "yes")
    print("=" * 68)
    print(f"{'metric':22s}  {'no-dyn':>14s}  {'with-dyn':>14s}  {'Δ (pp)':>10s}")
    print("-" * 68)
    for k in s_no:
        v_no = float(s_no[k].rstrip("%"))
        v_yes = float(s_yes[k].rstrip("%"))
        delta = v_yes - v_no
        sign = "+" if delta >= 0 else ""
        print(f"{k:22s}  {s_no[k]:>14s}  {s_yes[k]:>14s}  {sign}{delta:>9.2f}")
    print("=" * 68)

    # Save summary as text file
    out = os.path.join(args.out_dir, "per_class_summary.txt")
    with open(out, "w") as f:
        f.write(f"WITH dynamic best round: R{with_round+1}\n")
        f.write(f"NO   dynamic best round: R{no_round+1}\n\n")
        f.write(f"{'metric':22s}  {'no-dyn':>14s}  {'with-dyn':>14s}"
                f"  {'Δ (pp)':>10s}\n")
        f.write("-" * 68 + "\n")
        for k in s_no:
            v_no = float(s_no[k].rstrip("%"))
            v_yes = float(s_yes[k].rstrip("%"))
            delta = v_yes - v_no
            sign = "+" if delta >= 0 else ""
            f.write(f"{k:22s}  {s_no[k]:>14s}  {s_yes[k]:>14s}"
                    f"  {sign}{delta:>9.2f}\n")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
