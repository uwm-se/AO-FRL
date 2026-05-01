"""PSNR sweep on CIFAR-100 test set: 3 encoders × 3 σ + clean baseline.

Decoder $g_\\psi$ is trained on (ImageNet-frozen encoder, image) pairs from the
5K decoder pool. We evaluate the decoder's reconstruction quality on the 10K
CIFAR-100 test set (never seen during decoder/encoder training, never used
for federated training) under combinations of:

  Encoder ∈ {ImageNet frozen, Scheme A (CE+anchor), Scheme B (SupCon)}
  σ       ∈ {0.005, 0.02, 0.05}    (plus a no-noise reference for ImageNet)

Lower PSNR = better privacy. (ε at δ=1e-6: 1060 / 265 / 106).
"""

import argparse
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


def epsilon_from_sigma(sigma, delta=1e-6, clip_C=1.0):
    if sigma <= 0:
        return float("inf")
    return clip_C * math.sqrt(2.0 * math.log(1.25 / delta)) / sigma


@torch.no_grad()
def encode_clip_optionally_noisy(encoder, imgs, sigma, clip_C, gen):
    z = encoder(imgs)
    z = F.normalize(z, dim=1)
    norms = z.norm(dim=1, keepdim=True).clamp(min=1e-12)
    z_clip = z * torch.clamp(clip_C / norms, max=1.0)
    if sigma <= 0:
        return z_clip
    noise = torch.randn(z_clip.shape, generator=gen,
                        device=z_clip.device) * sigma
    return z_clip + noise


def per_sample_psnr(rec, tgt):
    mse = (rec - tgt).pow(2).mean(dim=(1, 2, 3)).clamp(min=1e-12)
    return (10.0 * torch.log10(1.0 / mse)).cpu().numpy()


@torch.no_grad()
def evaluate(encoder, decoder, ds, sigma, clip_C, batch_size, gen_seed,
             device):
    enc_tf = T.Compose([
        T.Resize(256), T.CenterCrop(224), T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    tgt_tf = T.ToTensor()

    psnr_all = []
    n = len(ds)
    rec_first = None
    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        enc_in = torch.stack(
            [enc_tf(ds[i][0]) for i in range(start, end)]).to(device)
        tgt = torch.stack(
            [tgt_tf(ds[i][0]) for i in range(start, end)]).to(device)
        gen = torch.Generator(device=device).manual_seed(gen_seed + start)
        z = encode_clip_optionally_noisy(encoder, enc_in, sigma, clip_C, gen)
        rec = decoder(z).clamp(0, 1)
        psnr_all.append(per_sample_psnr(rec, tgt))
        if rec_first is None:
            rec_first = (rec[:8].cpu(), tgt[:8].cpu())
    return np.concatenate(psnr_all), rec_first


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--decoder_weights", default="models/decoder_frozen_enc.pt")
    p.add_argument("--scheme_a_encoder",
                   default="models/encoder_supervised.pt")
    p.add_argument("--scheme_b_encoder",
                   default="models/encoder_supcon.pt")
    p.add_argument("--clip_C", type=float, default=1.0)
    p.add_argument("--delta", type=float, default=1e-6)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", default="results/psnr_sweep")
    args = p.parse_args()

    sigmas = [0.005, 0.02, 0.05]

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    decoder = Decoder(embed_dim=512).to(device).eval()
    decoder.load_state_dict(torch.load(args.decoder_weights,
                                       map_location=device))
    print(f"Loaded decoder: {args.decoder_weights}")

    test_ds = torchvision.datasets.CIFAR100(
        root="./data", train=False, download=False, transform=None)
    print(f"CIFAR-100 test set: {len(test_ds)} images")

    encoders = [
        ("ImageNet frozen", None, "tab:gray"),
        ("Scheme A (CE+anchor)", args.scheme_a_encoder, "tab:orange"),
        ("Scheme B (SupCon)",     args.scheme_b_encoder, "tab:red"),
    ]

    summary = {
        "delta": args.delta,
        "clip_C": args.clip_C,
        "n_test_samples": len(test_ds),
        "decoder_weights": args.decoder_weights,
        "sigmas": sigmas,
        "epsilons_at_delta": {f"{s}": epsilon_from_sigma(s, args.delta,
                                                         args.clip_C)
                              for s in sigmas},
        "results": {},
    }

    print("\n" + "=" * 78)
    print(f"{'Encoder':22s}  {'σ':>7}  {'ε':>8}  "
          f"{'PSNR mean':>10}  {'std':>7}  {'median':>7}")
    print("=" * 78)

    grid_per_enc = {}

    # Clean baseline (ImageNet, no noise) — done first as reference
    encoder_im, _ = build_encoder(device, weights_path=None)
    psnr_clean, rec_clean = evaluate(
        encoder_im, decoder, test_ds, 0.0, args.clip_C,
        args.batch_size, args.seed + 100, device)
    base_mean = float(psnr_clean.mean())
    summary["results"]["ImageNet clean (σ=0)"] = {
        "encoder": "ImageNet pretrained",
        "sigma": 0.0,
        "epsilon": float("inf"),
        "psnr_mean": base_mean,
        "psnr_std": float(psnr_clean.std()),
        "psnr_median": float(np.median(psnr_clean)),
    }
    print(f"{'ImageNet':22s}  {'0 (clean)':>9}  {'∞':>8}  "
          f"{base_mean:10.2f}  {psnr_clean.std():7.2f}  "
          f"{np.median(psnr_clean):7.2f}")

    # Sweep encoders × sigmas
    for enc_name, enc_path, _color in encoders:
        encoder, _ = build_encoder(device, weights_path=enc_path)
        grid_per_enc[enc_name] = {"clean": None, "sigmas": {}}
        for s in sigmas:
            psnr, rec_pair = evaluate(
                encoder, decoder, test_ds, s, args.clip_C,
                args.batch_size, args.seed + 100, device)
            eps = epsilon_from_sigma(s, args.delta, args.clip_C)
            key = f"{enc_name} σ={s}"
            summary["results"][key] = {
                "encoder": enc_path or "ImageNet pretrained",
                "sigma": s,
                "epsilon": eps,
                "psnr_mean": float(psnr.mean()),
                "psnr_std": float(psnr.std()),
                "psnr_median": float(np.median(psnr)),
                "privacy_gain_db_vs_clean": base_mean - float(psnr.mean()),
            }
            grid_per_enc[enc_name]["sigmas"][s] = rec_pair
            print(f"{enc_name:22s}  {s:>7.3f}  {eps:>8.1f}  "
                  f"{psnr.mean():10.2f}  {psnr.std():7.2f}  "
                  f"{np.median(psnr):7.2f}")
        del encoder
        if device == "cuda":
            torch.cuda.empty_cache()

    print("=" * 78 + "\n")

    # Save JSON
    json_path = os.path.join(args.out_dir, "psnr_sweep.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"JSON: {json_path}")

    # ------ Grouped bar chart ------
    enc_names = [e[0] for e in encoders]
    fig, ax = plt.subplots(figsize=(11.5, 6.5))
    width = 0.25
    x = np.arange(len(enc_names))
    sigma_colors = {0.005: "#9ed99e", 0.02: "#ff9966", 0.05: "#cc3333"}

    for i, s in enumerate(sigmas):
        means = [summary["results"][f"{e} σ={s}"]["psnr_mean"]
                 for e in enc_names]
        stds = [summary["results"][f"{e} σ={s}"]["psnr_std"]
                for e in enc_names]
        eps = epsilon_from_sigma(s, args.delta, args.clip_C)
        bars = ax.bar(x + (i - 1) * width, means, width, yerr=stds,
                      capsize=4, color=sigma_colors[s],
                      edgecolor="black", linewidth=0.6,
                      label=f"σ={s}  (ε≈{eps:.0f})")
        for xi, m in zip(x + (i - 1) * width, means):
            ax.text(xi, m + 0.1, f"{m:.2f}", ha="center", fontsize=8.5,
                    fontweight="bold")

    ax.axhline(base_mean, color="black", ls="--", lw=1, alpha=0.6,
               label=f"ImageNet clean baseline = {base_mean:.2f} dB")

    ax.set_xticks(x)
    ax.set_xticklabels(enc_names, fontsize=10)
    ax.set_ylabel("Reconstruction PSNR (dB)   ↓ = better privacy",
                  fontsize=12)
    ax.set_title(
        f"Privacy Sweep on CIFAR-100 Test Set ({summary['n_test_samples']} "
        f"images)\n3 encoders × 3 DP levels   |   δ={args.delta}, "
        f"clip C={args.clip_C}",
        fontsize=12)
    ax.legend(loc="upper right", fontsize=9.5)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    bar_path = os.path.join(args.out_dir, "psnr_sweep_bar.png")
    fig.savefig(bar_path, dpi=150)
    plt.close(fig)
    print(f"Bar:  {bar_path}")

    # ------ Visualization grid ------
    n_show = 8
    rows = [("Original",
             grid_per_enc[enc_names[0]]["sigmas"][sigmas[0]][1])]
    for enc_name in enc_names:
        for s in sigmas:
            rec, _ = grid_per_enc[enc_name]["sigmas"][s]
            rows.append((f"{enc_name}\nσ={s}", rec))
    fig, axes = plt.subplots(len(rows), n_show,
                              figsize=(2 * n_show, 1.4 * len(rows) + 0.5))
    for r, (label, panel) in enumerate(rows):
        for c in range(n_show):
            ax = axes[r, c] if n_show > 1 else axes[r]
            ax.imshow(panel[c].permute(1, 2, 0).numpy())
            ax.set_xticks([])
            ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(label, fontsize=8)
    fig.suptitle("Reconstruction grid on CIFAR-100 test set "
                 "(8 random images)", fontsize=11)
    fig.tight_layout()
    grid_path = os.path.join(args.out_dir, "psnr_sweep_grid.png")
    fig.savefig(grid_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Grid: {grid_path}")


if __name__ == "__main__":
    main()
