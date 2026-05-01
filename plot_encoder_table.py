"""Generate a summary table image comparing encoder fine-tuning approaches.

Outputs PNG to results/comparison/encoder_methods_table.png.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    headers = [
        "Method",
        "Objective",
        "Trainable\nparams",
        "Holdout val\n(500 imgs)",
        "AO-FRL global\ntest acc (best)",
        "vs ImageNet\nbaseline",
        "Verdict",
    ]
    rows = [
        ["Reconstruction (prior)",
         "MSE pixel\n(autoencoder)",
         "~100% encoder\n+ decoder",
         "20.95 dB\n(PSNR)",
         "30.45 %",
         "-35.68 pp",
         "Catastrophic"],
        ["ImageNet (frozen)",
         "- (no fine-tune)",
         "0\n(only MLP head)",
         "-",
         "66.13 %",
         "baseline",
         "Recommended"],
        ["Scheme A:\nCE + anchor reg",
         "Cross-entropy\n+ L2 anchor",
         "75 %\n(layer4 only)",
         "47.80 %",
         "61.90 %",
         "-4.23 pp",
         "Overfits"],
        ["Scheme B:\nSupContrast",
         "Supervised\ncontrastive",
         "75 %\n(layer4 only)",
         "58.20 %\n(linear probe)",
         "65.58 %",
         "-0.55 pp",
         "Roughly tied"],
    ]

    # Color rows by verdict
    row_colors = [
        ["#ffcccc"] * len(headers),  # reconstruction - red
        ["#ccffcc"] * len(headers),  # ImageNet - green (recommended)
        ["#ffe6cc"] * len(headers),  # Scheme A - orange
        ["#fffacc"] * len(headers),  # Scheme B - yellow
    ]

    fig, ax = plt.subplots(figsize=(15, 5.5))
    ax.axis("off")

    table = ax.table(
        cellText=rows,
        colLabels=headers,
        cellColours=row_colors,
        cellLoc="center",
        loc="center",
        colWidths=[0.16, 0.13, 0.10, 0.13, 0.13, 0.10, 0.10],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 2.4)

    # Bold headers
    for j in range(len(headers)):
        cell = table[0, j]
        cell.set_facecolor("#444444")
        cell.set_text_props(color="white", fontweight="bold")

    # Bold method-name column
    for i in range(1, len(rows) + 1):
        cell = table[i, 0]
        cell.set_text_props(fontweight="bold")

    # Title and footnote
    ax.set_title("Encoder Fine-Tuning Comparison — AO-FRL @ σ=0.005, "
                 "100-class CIFAR-100, 20 clients, α=0.3",
                 fontsize=13, fontweight="bold", pad=12)
    ax.text(0.5, -0.05,
            "Holdout = 5000 CIFAR-100 train images held out from federated "
            "clients (server-side public auxiliary data).\n"
            "Each fine-tuning method trains on 4500 of these images, "
            "validates on 500, then is used as the federated encoder.",
            ha="center", va="top", transform=ax.transAxes, fontsize=10,
            style="italic", color="#444")

    out = "results/comparison/encoder_methods_table.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
