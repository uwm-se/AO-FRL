"""Visualize per-class budget evolution across rounds for AO-FRL × 3 σ.

Each AO-FRL run logs the per-class upload target T_c stats per round in
aofrl_history.csv:
    round, synced, total_uploaded, T_min, T_max, T_mean, T_std

Two standalone figures are produced:
  - budget_spread.png      — [T_min, T_max] band + T_mean line
  - budget_dispersion.png  — std of T_c across 100 classes
"""

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_history(path):
    rows = {"round": [], "synced": [], "T_min": [], "T_max": [],
            "T_mean": [], "T_std": []}
    with open(path) as f:
        for r in csv.DictReader(f):
            rows["round"].append(int(r["round"]))
            rows["synced"].append(int(r["synced"]))
            rows["T_min"].append(int(r["T_min"]))
            rows["T_max"].append(int(r["T_max"]))
            rows["T_mean"].append(float(r["T_mean"]))
            rows["T_std"].append(float(r["T_std"]))
    return {k: np.array(v) for k, v in rows.items()}


RUNS = [
    ("σ=0.005  (ε≈1060)",
     "results/aofrl60/sigma0.005/aofrl_history.csv",
     "tab:green"),
    ("σ=0.02   (ε≈265)",
     "results/aofrl60/sigma0.02/aofrl_history.csv",
     "tab:orange"),
    ("σ=0.05   (ε≈106)",
     "results/aofrl60/sigma0.05/aofrl_history.csv",
     "tab:red"),
]


def plot_spread(out_path):
    fig, ax = plt.subplots(figsize=(11, 6))
    for label, path, color in RUNS:
        h = load_history(path)
        ax.fill_between(h["round"], h["T_min"], h["T_max"],
                        alpha=0.18, color=color)
        ax.plot(h["round"], h["T_mean"], color=color, lw=2.2,
                label=f"{label}   (mean over 100 classes)")
        ax.plot(h["round"], h["T_min"], color=color, lw=0.9,
                ls=":", alpha=0.7)
        ax.plot(h["round"], h["T_max"], color=color, lw=0.9,
                ls=":", alpha=0.7)
        sync_idx = np.where(h["synced"] == 1)[0]
        ax.scatter(h["round"][sync_idx], h["T_mean"][sync_idx],
                   marker="v", s=55, color=color, zorder=5,
                   edgecolors="black", linewidths=0.6)

    ax.axhline(500, color="black", ls="--", lw=1, alpha=0.5,
               label="T_base = 500 (initial)")
    ax.set_xlabel("Communication Round", fontsize=13)
    ax.set_ylabel("Per-class upload target  $T_c$", fontsize=13)
    ax.set_title(
        "AO-FRL Per-Class Budget Spread — [$T_{\\min}$, $T_{\\max}$] "
        "across rounds  (▼ = feedback-sync round)",
        fontsize=12.5)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_dispersion(out_path):
    fig, ax = plt.subplots(figsize=(11, 6))
    for label, path, color in RUNS:
        h = load_history(path)
        ax.plot(h["round"], h["T_std"], color=color, lw=2.2,
                marker="o", markersize=5, markerfacecolor="white",
                label=label)
        sync_idx = np.where(h["synced"] == 1)[0]
        ax.scatter(h["round"][sync_idx], h["T_std"][sync_idx],
                   marker="v", s=70, color=color, zorder=5,
                   edgecolors="black", linewidths=0.6)

    ax.set_xlabel("Communication Round", fontsize=13)
    ax.set_ylabel("Std of $T_c$ across 100 classes", fontsize=13)
    ax.set_title(
        "AO-FRL Budget Dispersion over Time — std($T_c$)  "
        "(▼ = feedback-sync round)",
        fontsize=12.5)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    out_dir = "results/aofrl60/figures"
    os.makedirs(out_dir, exist_ok=True)
    plot_spread(os.path.join(out_dir, "budget_spread.png"))
    plot_dispersion(os.path.join(out_dir, "budget_dispersion.png"))


if __name__ == "__main__":
    main()
