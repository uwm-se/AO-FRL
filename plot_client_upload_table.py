"""Per-round client upload table for the paper.

For a single client at the moment of an upload (right after a sync round),
show the four columns the upload decision is based on:
  1. local histogram (how many samples this client holds per class)
  2. received budget (per-class upload quota from server)
  3. per-class val acc reported at last sync (the feedback signal)
  4. actually uploaded count this round

Saves table to results/comparison/client_upload_table.png.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torchvision

from utils import (allocate_budgets, dirichlet_partition, set_seed,
                   split_decoder_pool, update_per_class_target)


def main():
    set_seed(42)
    ds = torchvision.datasets.CIFAR100(root="./data", train=True,
                                       download=False)
    labels = np.array(ds.targets)
    decoder_pool, federated_pool = split_decoder_pool(labels, frac=0.1, seed=42)
    fed_labels = labels[federated_pool]
    client_local = dirichlet_partition(fed_labels, n_clients=20, alpha=0.3,
                                       seed=42)
    client_indices = [federated_pool[idx] for idx in client_local]

    # Build histograms
    H = np.stack([np.bincount(labels[idx], minlength=100)
                  for idx in client_indices])
    client_id = 0
    hist = H[client_id]

    # Mock per-class acc reported at last sync (Round 5).
    # Spread covers strong / weak / unseen classes for the client.
    acc = np.full(100, 0.65)
    acc[32] = 0.85   # strong
    acc[99] = 0.30   # weak
    acc[78] = 0.45   # below-average
    acc[7]  = 0.70

    # Server updates T_c using feedback (alpha=1.0)
    T_base = 500
    T_new = update_per_class_target(acc, T_base=T_base, alpha=1.0)
    B_new = allocate_budgets(H, T_new)
    budget = B_new[client_id]

    # Actually uploaded each round = min(budget, hist) — already enforced by
    # AllocateBudget invariant, but show explicitly in table.
    uploaded = np.minimum(budget, hist)

    # ---- Pick representative classes for the table ----
    rows_show = [
        # (class_id, comment)
        (32, "strong (acc↑→budget cut)"),
        (99, "weak (acc↓→budget boost)"),
        (78, "below-avg (acc↓→budget boost)"),
        (23, "average"),
        (46, "average"),
        (22, "small holding"),
        (7,  "not held (capped to 0)"),
        (41, "not held (capped to 0)"),
    ]

    headers = ["Class", "Local hist h_i[c]", "Budget b_i[c]",
               "Sync val acc a_i[c]", "Uploaded u_i[c]", "Note"]
    rows = []
    row_colors = []
    for c, note in rows_show:
        rows.append([
            f"c{c}",
            str(int(hist[c])),
            str(int(budget[c])),
            f"{acc[c]:.2f}",
            str(int(uploaded[c])),
            note,
        ])
        # Color rows by acc level
        if acc[c] >= 0.80:
            row_colors.append(["#cfe8ff"] * len(headers))
        elif acc[c] <= 0.50 and hist[c] > 0:
            row_colors.append(["#ffd9d9"] * len(headers))
        elif hist[c] == 0:
            row_colors.append(["#eeeeee"] * len(headers))
        else:
            row_colors.append(["#ffffff"] * len(headers))

    # Total / aggregate row
    rows.append([
        "Σ (all 100)",
        f"{int(hist.sum())}",
        f"{int(budget.sum())}",
        "—",
        f"{int(uploaded.sum())}",
        f"covers {(hist > 0).sum()} classes (74 / 100)",
    ])
    row_colors.append(["#fffacc"] * len(headers))

    # ---- Render ----
    fig, ax = plt.subplots(figsize=(13, 5.0))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=headers,
        cellColours=row_colors,
        cellLoc="center",
        loc="center",
        colWidths=[0.07, 0.14, 0.13, 0.16, 0.14, 0.30],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 2.2)

    # Header styling
    for j in range(len(headers)):
        cell = table[0, j]
        cell.set_facecolor("#444444")
        cell.set_text_props(color="white", fontweight="bold")

    # Bold first column + last row
    for i in range(1, len(rows) + 1):
        table[i, 0].set_text_props(fontweight="bold")
    for j in range(len(headers)):
        table[len(rows), j].set_text_props(fontweight="bold")

    title = ("Client 0's upload state at Round 6 "
             "(just after Round-5 sync, T_base=500, σ=0.02, ε≈242)")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)

    legend_lines = [
        "  blue rows  : strong class (acc ≥ 0.80) — server cuts budget",
        "  red rows   : weak class (acc ≤ 0.50, held) — server boosts budget",
        "  gray rows  : not held by this client (b_i[c] always 0)",
        "  yellow row : aggregate over all 100 classes",
    ]
    ax.text(0.5, -0.08, "\n".join(legend_lines),
            ha="center", va="top", transform=ax.transAxes, fontsize=10,
            family="monospace", color="#444")

    out_dir = "results/comparison"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "client_upload_table.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
