"""For the T_base=50 + no-replay experiment, identify which CIFAR-100 classes
are in the bottom-5 / bottom-10 (worst-performing under static budget) and
analyze how their samples are distributed across the 20 clients.

This tells us: what kind of classes does dynamic budget allocation actually
help, and what is special about how their data is held by clients?
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torchvision

from utils import dirichlet_partition, split_decoder_pool


def main():
    # ---- Load CIFAR-100 class names + reconstruct partition ----
    ds = torchvision.datasets.CIFAR100(root="./data", train=True,
                                       download=False)
    class_names = ds.classes  # ['apple', 'aquarium_fish', ...]
    labels = np.array(ds.targets)
    decoder_pool, federated_pool = split_decoder_pool(labels, frac=0.1,
                                                      seed=42)
    fed_labels = labels[federated_pool]
    client_local = dirichlet_partition(fed_labels, n_clients=20, alpha=0.3,
                                       seed=42)
    # H[i, c] = client i's count of class c
    H = np.stack([np.bincount(fed_labels[idx], minlength=100)
                  for idx in client_local])  # (20, 100)

    # ---- Load per-class accuracy from no-dyn run, pick bottom-K ----
    pcs = np.load("results/tbase50_no_dyn/AO-FRL_per_class.npy")
    best_round = int(np.argmax(pcs.mean(axis=1)))
    static_acc = pcs[best_round]  # 100-d
    pcs_dyn = np.load("results/tbase50_with_dyn/AO-FRL_per_class.npy")
    best_round_dyn = int(np.argmax(pcs_dyn.mean(axis=1)))
    dyn_acc = pcs_dyn[best_round_dyn]

    print(f"Static budget best round R{best_round+1}, mean acc {static_acc.mean()*100:.2f}%")
    print(f"Dynamic budget best round R{best_round_dyn+1}, mean acc {dyn_acc.mean()*100:.2f}%\n")

    bottom10 = np.argsort(static_acc)[:10]
    print("=" * 90)
    print(f"{'Rank':4s}  {'class':5s}  {'name':16s}  "
          f"{'static':>7}  {'dynamic':>8}  {'Δ (pp)':>8}  "
          f"{'#clients holding':>16}  {'max/mean per client':>22}  {'gini':>7}")
    print("=" * 90)
    for rank, c in enumerate(bottom10):
        col = H[:, c]              # 20-d distribution
        n_holders = int((col > 0).sum())
        nz = col[col > 0]
        max_share = int(col.max())
        mean_share = float(col.mean())
        # Gini coefficient: how concentrated is the distribution
        if col.sum() > 0:
            sc = np.sort(col)
            n = len(sc)
            gini = (2 * np.arange(1, n+1) - n - 1).dot(sc) / (n * sc.sum())
        else:
            gini = 0.0
        print(f"{rank+1:4d}  c{c:<4d}  {class_names[c]:16s}  "
              f"{static_acc[c]*100:>6.1f}%  {dyn_acc[c]*100:>7.1f}%  "
              f"{(dyn_acc[c]-static_acc[c])*100:>+7.2f}  "
              f"{n_holders:>16d}  {f'{max_share}/{mean_share:.1f}':>22}  "
              f"{gini:>7.3f}")
    print("=" * 90)

    # ---- Compare with top-10 (best classes under static) ----
    top10 = np.argsort(static_acc)[-10:][::-1]
    print(f"\n--- TOP-10 reference (best classes under static) ---")
    print(f"{'Rank':4s}  {'class':5s}  {'name':16s}  "
          f"{'static':>7}  {'#clients holding':>16}  {'max/mean':>14}  {'gini':>7}")
    for rank, c in enumerate(top10):
        col = H[:, c]
        n_holders = int((col > 0).sum())
        max_share = int(col.max())
        mean_share = float(col.mean())
        if col.sum() > 0:
            sc = np.sort(col)
            n = len(sc)
            gini = (2 * np.arange(1, n+1) - n - 1).dot(sc) / (n * sc.sum())
        else:
            gini = 0.0
        print(f"{rank+1:4d}  c{c:<4d}  {class_names[c]:16s}  "
              f"{static_acc[c]*100:>6.1f}%  {n_holders:>16d}  "
              f"{f'{max_share}/{mean_share:.1f}':>14}  {gini:>7.3f}")

    # ---- Aggregate stats: bottom-10 vs top-10 ----
    bottom_holders = (H[:, bottom10] > 0).sum(axis=0)  # 10-d
    top_holders = (H[:, top10] > 0).sum(axis=0)
    bottom_gini = []
    top_gini = []
    for c in bottom10:
        col = H[:, c]
        if col.sum() > 0:
            sc = np.sort(col)
            n = len(sc)
            bottom_gini.append((2 * np.arange(1, n+1) - n - 1).dot(sc) / (n * sc.sum()))
    for c in top10:
        col = H[:, c]
        if col.sum() > 0:
            sc = np.sort(col)
            n = len(sc)
            top_gini.append((2 * np.arange(1, n+1) - n - 1).dot(sc) / (n * sc.sum()))

    print(f"\n--- Aggregate comparison ---")
    print(f"Bottom-10 (worst classes):  mean #clients holding = {bottom_holders.mean():.1f}/20,  "
          f"mean Gini = {np.mean(bottom_gini):.3f}")
    print(f"Top-10  (best classes):    mean #clients holding = {top_holders.mean():.1f}/20,  "
          f"mean Gini = {np.mean(top_gini):.3f}")

    # ---- Visualization: distribution heatmap ----
    out_dir = "results/tbase50_compare"
    os.makedirs(out_dir, exist_ok=True)

    # 2-row grid: bottom-10 (top), top-10 (bottom)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8.5))
    bot_mat = H[:, bottom10].T   # (10, 20)
    top_mat = H[:, top10].T

    im1 = ax1.imshow(bot_mat, aspect="auto", cmap="Reds")
    ax1.set_yticks(range(10))
    ax1.set_yticklabels([f"c{c}: {class_names[c]} ({static_acc[c]*100:.0f}%→{dyn_acc[c]*100:.0f}%)"
                         for c in bottom10], fontsize=9)
    ax1.set_xticks(range(20))
    ax1.set_xticklabels([f"C{i}" for i in range(20)], fontsize=8)
    ax1.set_xlabel("Client index")
    ax1.set_title("Bottom-10 classes (lowest accuracy under static budget) — "
                  "samples per client", fontsize=11)
    plt.colorbar(im1, ax=ax1, label="# samples")
    # Annotate cells
    for i in range(10):
        for j in range(20):
            if bot_mat[i, j] > 0:
                ax1.text(j, i, str(bot_mat[i, j]), ha="center", va="center",
                         fontsize=7, color="white" if bot_mat[i, j] > bot_mat.max()*0.5 else "black")

    im2 = ax2.imshow(top_mat, aspect="auto", cmap="Greens")
    ax2.set_yticks(range(10))
    ax2.set_yticklabels([f"c{c}: {class_names[c]} ({static_acc[c]*100:.0f}%)"
                         for c in top10], fontsize=9)
    ax2.set_xticks(range(20))
    ax2.set_xticklabels([f"C{i}" for i in range(20)], fontsize=8)
    ax2.set_xlabel("Client index")
    ax2.set_title("Top-10 classes (highest accuracy under static budget) — "
                  "samples per client (reference)", fontsize=11)
    plt.colorbar(im2, ax=ax2, label="# samples")
    for i in range(10):
        for j in range(20):
            if top_mat[i, j] > 0:
                ax2.text(j, i, str(top_mat[i, j]), ha="center", va="center",
                         fontsize=7, color="white" if top_mat[i, j] > top_mat.max()*0.5 else "black")

    fig.tight_layout()
    out = os.path.join(out_dir, "bottom_vs_top_distribution.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nSaved: {out}")

    # ---- Also: number of holders bar plot ----
    fig, ax = plt.subplots(figsize=(11, 4.5))
    x_b = np.arange(10)
    bars1 = ax.bar(x_b - 0.2, bottom_holders, 0.4, color="tab:red", alpha=0.8,
                   label=f"bottom-10 (mean={bottom_holders.mean():.1f}/20 holders)")
    bars2 = ax.bar(x_b + 0.2, top_holders, 0.4, color="tab:green", alpha=0.8,
                   label=f"top-10 (mean={top_holders.mean():.1f}/20 holders)")
    ax.set_xticks(x_b)
    ax.set_xticklabels([f"#{i+1}" for i in range(10)])
    ax.set_xlabel("Rank within group")
    ax.set_ylabel("# of clients holding ≥1 sample")
    ax.set_title("How concentrated are bottom-10 vs top-10 classes "
                 "across 20 clients? (out of 20)", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, 22)
    for x_, n_ in zip(x_b - 0.2, bottom_holders):
        ax.text(x_, n_ + 0.3, str(n_), ha="center", fontsize=8)
    for x_, n_ in zip(x_b + 0.2, top_holders):
        ax.text(x_, n_ + 0.3, str(n_), ha="center", fontsize=8)
    fig.tight_layout()
    out = os.path.join(out_dir, "holder_count_compare.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
