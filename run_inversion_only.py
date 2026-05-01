"""Standalone decoder-inversion PSNR evaluation.

Used to retroactively add inversion_summary.json to existing sweep result
directories without re-running the full federated experiment. Server-side
attack model: server holds the public holdout decoder-pool, trains a decoder
with the encoder frozen, then attempts to invert noisy uploads.

Per-sample MSE → PSNR for two scenarios:
  - clean:  decoder(z)              — ceiling, σ-independent
  - noisy:  decoder(clip(z) + N(0, σ²))  — actual privacy threat

Writes inversion_summary.json + inversion_grid.png + per-sample CSV into the
target dir, matching the format of run_inversion_eval in run_experiment.py.
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
from utils import set_seed


@torch.no_grad()
def evaluate(args):
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Frozen ImageNet ResNet-18 encoder (no fine-tune)
    encoder, embed_dim = build_encoder(device, weights_path=None)

    # Public-holdout decoder
    decoder = Decoder(embed_dim=embed_dim).to(device).eval()
    decoder.load_state_dict(torch.load(args.decoder_weights,
                                       map_location=device))

    enc_tf = T.Compose([
        T.Resize(256), T.CenterCrop(224), T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    tgt_tf = T.ToTensor()

    test_ds = torchvision.datasets.CIFAR100(
        root="./data", train=False, download=True, transform=None)

    n = min(args.n_samples, len(test_ds))
    rng = np.random.default_rng(args.seed)
    sample_idx = rng.choice(len(test_ds), size=n, replace=False).tolist()

    enc_inputs, targets, labels = [], [], []
    for i in sample_idx:
        img, label = test_ds[i]
        enc_inputs.append(enc_tf(img))
        targets.append(tgt_tf(img))
        labels.append(label)
    enc_inputs = torch.stack(enc_inputs).to(device)
    targets = torch.stack(targets).to(device)

    z = F.normalize(encoder(enc_inputs), dim=1)
    norms = z.norm(dim=1, keepdim=True).clamp(min=1e-12)
    z_clip = z * torch.clamp(args.clip_C / norms, max=1.0)

    recon_clean = decoder(z_clip).clamp(0, 1)
    z_noisy = z_clip + torch.randn_like(z_clip) * args.sigma
    recon_noisy = decoder(z_noisy).clamp(0, 1)

    def per_sample_psnr(rec, tgt):
        mse = (rec - tgt).pow(2).mean(dim=(1, 2, 3)).clamp(min=1e-12)
        return (10.0 * torch.log10(1.0 / mse)).cpu().numpy()

    psnr_clean = per_sample_psnr(recon_clean, targets)
    psnr_noisy = per_sample_psnr(recon_noisy, targets)

    eps_implied = (args.clip_C * math.sqrt(2.0 * math.log(1.25 / args.delta))
                   / max(args.sigma, 1e-12))

    summary = {
        "n_samples": n,
        "sigma": float(args.sigma),
        "epsilon": float(eps_implied),
        "delta": float(args.delta),
        "clip_C": float(args.clip_C),
        "psnr_clean_mean": float(psnr_clean.mean()),
        "psnr_clean_std": float(psnr_clean.std()),
        "psnr_noisy_mean": float(psnr_noisy.mean()),
        "psnr_noisy_std": float(psnr_noisy.std()),
        "encoder": "ImageNet ResNet-18 (frozen)",
        "decoder": args.decoder_weights,
    }
    print(json.dumps(summary, indent=2))

    os.makedirs(args.results_dir, exist_ok=True)
    with open(os.path.join(args.results_dir,
                           "inversion_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    csv_path = os.path.join(args.results_dir, "inversion_per_sample.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_idx", "label", "psnr_clean", "psnr_noisy"])
        for j, idx in enumerate(sample_idx):
            w.writerow([idx, int(labels[j]),
                        float(psnr_clean[j]), float(psnr_noisy[j])])

    n_show = min(8, n)
    fig, axes = plt.subplots(3, n_show, figsize=(2 * n_show, 6))
    rows = ["Original", "Recon (clean)", f"Recon (σ={args.sigma:.3f})"]
    panels = [targets[:n_show], recon_clean[:n_show], recon_noisy[:n_show]]
    for r, (label, panel) in enumerate(zip(rows, panels)):
        for c in range(n_show):
            ax = axes[r, c] if n_show > 1 else axes[r]
            ax.imshow(panel[c].cpu().permute(1, 2, 0).numpy())
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(label, fontsize=11)
            if r == 0:
                ax.set_title(f"id {sample_idx[c]}", fontsize=9)
            elif r == 1:
                ax.set_xlabel(f"{psnr_clean[c]:.1f}dB", fontsize=9)
            elif r == 2:
                ax.set_xlabel(f"{psnr_noisy[c]:.1f}dB", fontsize=9)
    fig.suptitle(f"Decoder Inversion — σ={args.sigma:.3f} "
                 f"(ε≈{eps_implied:.1f}, δ={args.delta})", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(args.results_dir, "inversion_grid.png"), dpi=150)
    plt.close(fig)
    print(f"Saved to: {args.results_dir}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sigma", type=float, required=True)
    p.add_argument("--delta", type=float, default=1e-5)
    p.add_argument("--clip_C", type=float, default=1.0)
    p.add_argument("--results_dir", required=True,
                   help="Existing sweep dir; inversion outputs land here.")
    p.add_argument("--decoder_weights", default="models/decoder.pt")
    p.add_argument("--n_samples", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
