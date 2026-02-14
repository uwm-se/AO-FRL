#!/usr/bin/env python3
"""
Plot comparison between baseline AO-FRL and ablation experiments.
"""

import argparse
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_ablation_comparison(baseline_dir, ablation_dirs, output_dir):
    """Plot comparison between baseline and ablation experiments."""
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    baseline_acc_df = pd.read_csv(os.path.join(baseline_dir, "AO-FRL_rounds.csv"))
    baseline_server_df = pd.read_csv(os.path.join(baseline_dir, "server_instructions.csv"))

    # Merge baseline dataframes
    baseline_df = pd.merge(baseline_acc_df, baseline_server_df, on="round", how="left")
    baseline_df.rename(columns={
        "accuracy": "test_acc",
        "cumulative_comm_bytes": "comm_bytes",
        "avg_reject_ratio": "avg_reject_ratio",
        "avg_budget": "avg_budget",
        "avg_sigma": "avg_sigma",
        "n_conservative": "conservative_count"
    }, inplace=True)

    with open(os.path.join(baseline_dir, "final_summary.json"), "r") as f:
        baseline_summary = json.load(f)

    ablation_data = {}
    for name, path in ablation_dirs.items():
        df = pd.read_csv(os.path.join(path, "metrics.csv"))
        with open(os.path.join(path, "summary.json"), "r") as f:
            summary = json.load(f)
        ablation_data[name] = {"df": df, "summary": summary}

    # Figure 1: Accuracy Comparison
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(baseline_df["round"], baseline_df["test_acc"],
            label="AO-FRL (Full)", linewidth=2.5, color="#2E86AB")

    colors = ["#A23B72", "#F18F01"]
    for i, (name, data) in enumerate(ablation_data.items()):
        df = data["df"]
        ax.plot(df["round"], df["test_acc"],
                label=name, linewidth=2.5, color=colors[i], linestyle="--")

    ax.set_xlabel("Round", fontsize=12)
    ax.set_ylabel("Test Accuracy", fontsize=12)
    ax.set_title("Ablation Study: Test Accuracy vs Training Round", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ablation_accuracy_comparison.png"), dpi=300)
    print(f"✓ Saved: ablation_accuracy_comparison.png")
    plt.close()

    # Figure 2: Final Accuracy Bar Chart
    fig, ax = plt.subplots(figsize=(10, 6))

    methods = ["AO-FRL (Full)"] + list(ablation_data.keys())
    final_accs = [baseline_df["test_acc"].iloc[-1]]
    for name, data in ablation_data.items():
        final_accs.append(data["df"]["test_acc"].iloc[-1])

    colors_bar = ["#2E86AB", "#A23B72", "#F18F01"]
    bars = ax.bar(methods, final_accs, color=colors_bar, edgecolor="black", linewidth=1.5)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2%}', ha='center', va='bottom', fontsize=11, fontweight="bold")

    ax.set_ylabel("Final Test Accuracy", fontsize=12)
    ax.set_title("Ablation Study: Final Test Accuracy Comparison", fontsize=14, fontweight="bold")
    ax.set_ylim([0, max(final_accs) * 1.15])
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ablation_final_accuracy.png"), dpi=300)
    print(f"✓ Saved: ablation_final_accuracy.png")
    plt.close()

    # Figure 3: Budget Evolution Comparison
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(baseline_df["round"], baseline_df["avg_budget"],
            label="AO-FRL (Full)", linewidth=2.5, color="#2E86AB")

    for i, (name, data) in enumerate(ablation_data.items()):
        df = data["df"]
        ax.plot(df["round"], df["avg_budget"],
                label=name, linewidth=2.5, color=colors[i], linestyle="--")

    ax.axhline(500, color="gray", linestyle=":", linewidth=2, label="Base Budget (500)")
    ax.set_xlabel("Round", fontsize=12)
    ax.set_ylabel("Average Upload Budget", fontsize=12)
    ax.set_title("Ablation Study: Upload Budget Evolution", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ablation_budget_evolution.png"), dpi=300)
    print(f"✓ Saved: ablation_budget_evolution.png")
    plt.close()

    # Figure 4: Rejection Ratio Comparison
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(baseline_df["round"], baseline_df["avg_reject_ratio"],
            label="AO-FRL (Full)", linewidth=2.5, color="#2E86AB")

    for i, (name, data) in enumerate(ablation_data.items()):
        df = data["df"]
        ax.plot(df["round"], df["avg_reject_ratio"],
                label=name, linewidth=2.5, color=colors[i], linestyle="--")

    ax.set_xlabel("Round", fontsize=12)
    ax.set_ylabel("Average Rejection Ratio", fontsize=12)
    ax.set_title("Ablation Study: Privacy Gate Rejection Ratio", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ablation_rejection_ratio.png"), dpi=300)
    print(f"✓ Saved: ablation_rejection_ratio.png")
    plt.close()

    # Figure 5: Communication Cost Comparison
    fig, ax = plt.subplots(figsize=(10, 6))

    baseline_comm = baseline_summary["AO-FRL"]["total_comm_bytes"] / 1e9

    comm_costs = [baseline_comm]
    for name, data in ablation_data.items():
        comm_costs.append(data["summary"]["total_comm_gb"])

    bars = ax.bar(methods, comm_costs, color=colors_bar, edgecolor="black", linewidth=1.5)

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f} GB', ha='center', va='bottom', fontsize=11, fontweight="bold")

    ax.set_ylabel("Total Communication Cost (GB)", fontsize=12)
    ax.set_title("Ablation Study: Total Communication Cost", fontsize=14, fontweight="bold")
    ax.set_ylim([0, max(comm_costs) * 1.15])
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ablation_communication_cost.png"), dpi=300)
    print(f"✓ Saved: ablation_communication_cost.png")
    plt.close()

    # Generate summary report
    print("\n" + "="*70)
    print("ABLATION STUDY SUMMARY")
    print("="*70)
    print(f"\nBaseline: AO-FRL (Full)")
    print(f"  Final Accuracy: {baseline_df['test_acc'].iloc[-1]:.4f}")
    print(f"  Total Communication: {baseline_comm:.3f} GB")
    print(f"  Avg Rejection Ratio: {baseline_df['avg_reject_ratio'].mean():.4f}")
    print(f"  Avg Upload Budget: {baseline_df['avg_budget'].mean():.1f}")

    for name, data in ablation_data.items():
        df = data["df"]
        summary = data["summary"]
        print(f"\n{name}:")
        print(f"  Final Accuracy: {df['test_acc'].iloc[-1]:.4f} " +
              f"({df['test_acc'].iloc[-1] - baseline_df['test_acc'].iloc[-1]:+.4f})")
        print(f"  Total Communication: {summary['total_comm_gb']:.3f} GB " +
              f"({summary['total_comm_gb'] - baseline_comm:+.3f} GB)")
        print(f"  Avg Rejection Ratio: {df['avg_reject_ratio'].mean():.4f}")
        print(f"  Avg Upload Budget: {df['avg_budget'].mean():.1f}")

    print("="*70)


def main():
    parser = argparse.ArgumentParser(description="Plot ablation study results")
    parser.add_argument("--baseline", default="results",
                        help="Path to baseline AO-FRL results")
    parser.add_argument("--no_orchestration", default="ablation_results/no_orchestration",
                        help="Path to no orchestration ablation results")
    parser.add_argument("--no_privacy_gate", default="ablation_results/no_privacy_gate",
                        help="Path to no privacy gate ablation results")
    parser.add_argument("--output", default="ablation_results/figures",
                        help="Output directory for figures")
    args = parser.parse_args()

    ablation_dirs = {
        "w/o Server Orchestration": args.no_orchestration,
        "w/o Privacy Gate": args.no_privacy_gate,
    }

    plot_ablation_comparison(args.baseline, ablation_dirs, args.output)


if __name__ == "__main__":
    main()
