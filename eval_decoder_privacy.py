"""Privacy evaluation: server-side reconstruction attack on client data.

Threat model:
  - Server has access to public auxiliary data (the 5000 holdout) and trains
    a decoder using that, with the encoder frozen at ImageNet weights.
  - Client uploads embeddings of its OWN training images (drawn from the
    45000 federated pool, disjoint from the decoder pool) after L2 clip + DP
    Gaussian noise.
  - Server attempts to invert these uploads with its decoder.

This script measures how well the server's decoder can reconstruct CLIENT
training images, comparing:
  - Clean reconstruction (encoder→decoder, no DP) — strongest possible
    attack from the server, ignoring the DP layer entirely.
  - DP-noisy reconstruction (encoder→clip→+N(0, σ²)→decoder), at three σ
    levels matching the AO-FRL utility experiments.

Per-sigma privacy gain (dB) = PSNR_clean − PSNR_noisy(σ). Higher = better
privacy preserved by the DP step beyond the encoder's intrinsic information
loss.

Outputs (under --out_dir):
  privacy_eval.json          per-condition mean/std PSNR
  privacy_psnr_table.png     summary table image
  privacy_grid.png           qualitative grid (original / clean / each σ)
"""

import argparse
import csv
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

from models import Decoder
from run_experiment import build_encoder
from utils import set_seed, split_decoder_pool


def epsilon_from_sigma(sigma, delta=1e-5, clip_C=1.0):
    if sigma <= 0:
        return float("inf")
    return clip_C * math.sqrt(2.0 * math.log(1.25 / delta)) / sigma


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--decoder_weights", default="models/decoder.pt")
    p.add_argument("--clip_C", type=float, default=1.0)
    p.add_argument("--delta", type=float, default=1e-5)
    p.add_argument("--n_samples", type=int, default=300,
                   help="Client images to evaluate on (drawn from "
                        "federated_pool, disjoint from decoder pool).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", default="results/comparison")
    args = p.parse_args()

    sigmas = [0.005, 0.02, 0.05]
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    encoder, embed_dim = build_encoder(device, weights_path=None)  # ImageNet
    decoder = Decoder(embed_dim=embed_dim).to(device).eval()
    decoder.load_state_dict(torch.load(args.decoder_weights,
                                       map_location=device))
    print(f"Loaded decoder: {args.decoder_weights}")

    enc_tf = T.Compose([
        T.Resize(256), T.CenterCrop(224), T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    tgt_tf = T.ToTensor()

    train_ds = torchvision.datasets.CIFAR100(
        root="./data", train=True, download=True, transform=None)
    labels = np.array(train_ds.targets)
    decoder_pool, federated_pool = split_decoder_pool(
        labels, frac=0.1, seed=args.seed)

    # Verify disjointness — sanity check
    assert len(set(decoder_pool.tolist()) & set(federated_pool.tolist())) == 0
    print(f"Decoder training pool: {len(decoder_pool)} (server-side)")
    print(f"Federated client pool: {len(federated_pool)} (clients)")

    n = min(args.n_samples, len(federated_pool))
    rng = np.random.default_rng(args.seed + 1)  # different from decoder split
    pick = rng.choice(len(federated_pool), size=n, replace=False)
    sample_idx = federated_pool[pick]
    print(f"Sampling {n} images from federated_pool for privacy eval")

    enc_inputs, targets, sample_labels = [], [], []
    for i in sample_idx:
        img, label = train_ds[int(i)]
        enc_inputs.append(enc_tf(img))
        targets.append(tgt_tf(img))
        sample_labels.append(label)
    enc_inputs = torch.stack(enc_inputs).to(device)
    targets = torch.stack(targets).to(device)

    # Encoder + clip (these are deterministic, σ-independent)
    z = F.normalize(encoder(enc_inputs), dim=1)
    norms = z.norm(dim=1, keepdim=True).clamp(min=1e-12)
    z_clip = z * torch.clamp(args.clip_C / norms, max=1.0)

    def per_sample_psnr(rec, tgt):
        mse = (rec - tgt).pow(2).mean(dim=(1, 2, 3)).clamp(min=1e-12)
        return (10.0 * torch.log10(1.0 / mse)).cpu().numpy()

    # Clean reconstruction (no DP)
    recon_clean = decoder(z_clip).clamp(0, 1)
    psnr_clean = per_sample_psnr(recon_clean, targets)
    psnr_clean_mean = float(psnr_clean.mean())
    psnr_clean_std = float(psnr_clean.std())
    print(f"\nClean (no DP):  PSNR = {psnr_clean_mean:.2f} ± "
          f"{psnr_clean_std:.2f} dB")

    # DP-noisy at each σ
    psnr_noisy_per_sigma = {}
    recon_noisy_per_sigma = {}
    for sigma in sigmas:
        # Use a fixed RNG seed per σ so re-runs are reproducible
        gen = torch.Generator(device=device).manual_seed(args.seed + 100 +
                                                          int(sigma * 1e6))
        noise = torch.randn(z_clip.shape, generator=gen, device=device) * sigma
        z_noisy = z_clip + noise
        recon = decoder(z_noisy).clamp(0, 1)
        recon_noisy_per_sigma[sigma] = recon
        psnr = per_sample_psnr(recon, targets)
        psnr_noisy_per_sigma[sigma] = {
            "mean": float(psnr.mean()), "std": float(psnr.std()),
            "per_sample": psnr.tolist(),
        }
        eps = epsilon_from_sigma(sigma, args.delta, args.clip_C)
        print(f"σ={sigma:.3f} (ε≈{eps:.1f}):  PSNR = "
              f"{psnr.mean():.2f} ± {psnr.std():.2f} dB    "
              f"privacy gain = {psnr_clean_mean - psnr.mean():+.2f} dB")

    # ----- Save JSON summary -----
    os.makedirs(args.out_dir, exist_ok=True)
    summary = {
        "n_samples": n,
        "data_source": "federated_pool (client training images)",
        "encoder": "ImageNet ResNet-18 (frozen)",
        "decoder": args.decoder_weights,
        "clip_C": args.clip_C,
        "delta": args.delta,
        "psnr_clean_mean": psnr_clean_mean,
        "psnr_clean_std": psnr_clean_std,
        "per_sigma": {
            f"{s}": {
                "epsilon": epsilon_from_sigma(s, args.delta, args.clip_C),
                "psnr_noisy_mean": psnr_noisy_per_sigma[s]["mean"],
                "psnr_noisy_std": psnr_noisy_per_sigma[s]["std"],
                "privacy_gain_db": psnr_clean_mean -
                                   psnr_noisy_per_sigma[s]["mean"],
            }
            for s in sigmas
        },
    }
    with open(os.path.join(args.out_dir, "privacy_eval.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary JSON: {args.out_dir}/privacy_eval.json")

    # ----- Generate summary table image -----
    headers = ["Condition", "σ", "ε (δ=1e-5)", "PSNR (mean ± std)",
               "Privacy gain vs clean"]
    rows = [["Clean (no DP)", "0", "∞", f"{psnr_clean_mean:.2f} ± "
            f"{psnr_clean_std:.2f} dB", "— (baseline)"]]
    row_colors = [["#cfe8ff"] * len(headers)]
    for s in sigmas:
        m = psnr_noisy_per_sigma[s]["mean"]
        sd = psnr_noisy_per_sigma[s]["std"]
        eps = epsilon_from_sigma(s, args.delta, args.clip_C)
        gain = psnr_clean_mean - m
        if s <= 0.01:
            rc = "#fffacc"
        elif s <= 0.03:
            rc = "#ffe6cc"
        else:
            rc = "#ffcccc"
        rows.append([
            f"DP (σ={s})",
            f"{s}",
            f"≈ {eps:.1f}",
            f"{m:.2f} ± {sd:.2f} dB",
            f"−{gain:.2f} dB" if gain >= 0 else f"+{-gain:.2f} dB"
        ])
        row_colors.append([rc] * len(headers))

    fig, ax = plt.subplots(figsize=(13, 4.0))
    ax.axis("off")
    table = ax.table(
        cellText=rows, colLabels=headers, cellColours=row_colors,
        cellLoc="center", loc="center",
        colWidths=[0.16, 0.08, 0.13, 0.22, 0.22])
    table.auto_set_font_size(False); table.set_fontsize(11); table.scale(1.0, 2.2)
    for j in range(len(headers)):
        cell = table[0, j]
        cell.set_facecolor("#444444")
        cell.set_text_props(color="white", fontweight="bold")
    for i in range(1, len(rows) + 1):
        table[i, 0].set_text_props(fontweight="bold")
    ax.set_title("Server-Side Reconstruction Attack on CLIENT Training Data\n"
                 "(decoder trained on 5000-image public holdout, frozen "
                 "ImageNet encoder)",
                 fontsize=13, fontweight="bold", pad=14)
    ax.text(0.5, -0.05,
            f"n={n} client images | clip_C={args.clip_C} | unit-norm "
            f"embedding | per-σ noise sample fixed by seed for reproducibility",
            ha="center", va="top", transform=ax.transAxes,
            fontsize=10, style="italic", color="#444")
    fig.savefig(os.path.join(args.out_dir, "privacy_psnr_table.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Table:        {args.out_dir}/privacy_psnr_table.png")

    # ----- Generate visualization grid -----
    n_show = min(8, n)
    rows_v = [("Original", targets[:n_show]),
              ("Clean recon", recon_clean[:n_show])]
    for s in sigmas:
        rows_v.append((f"σ={s}", recon_noisy_per_sigma[s][:n_show]))
    fig, axes = plt.subplots(len(rows_v), n_show,
                             figsize=(2 * n_show, 1.5 * len(rows_v) + 0.5))
    psnrs_for_grid = {0.0: psnr_clean[:n_show]}
    for s in sigmas:
        psnrs_for_grid[s] = np.array(psnr_noisy_per_sigma[s]["per_sample"])[:n_show]
    for r, (label, panel) in enumerate(rows_v):
        for c in range(n_show):
            ax = axes[r, c] if n_show > 1 else axes[r]
            ax.imshow(panel[c].cpu().permute(1, 2, 0).numpy())
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(label, fontsize=10)
            if r == 0:
                ax.set_title(f"id {int(sample_idx[c])}", fontsize=8)
            elif label == "Clean recon":
                ax.set_xlabel(f"{psnrs_for_grid[0.0][c]:.1f} dB", fontsize=8)
            else:
                # extract sigma value from label
                s_val = float(label.split("=")[1])
                ax.set_xlabel(f"{psnrs_for_grid[s_val][c]:.1f} dB",
                              fontsize=8)
    fig.suptitle("Reconstruction of Client Training Images — clean vs DP-noisy",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "privacy_grid.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Grid:         {args.out_dir}/privacy_grid.png")


if __name__ == "__main__":
    main()
