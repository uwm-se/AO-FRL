"""Compare reconstruction PSNR across encoder/noise conditions.

Decoder is trained on (ImageNet-frozen encoder, image) pairs from the 5K
decoder pool — the test images here come from the disjoint 45K federated
pool, so the decoder has never seen ANY of these images during training.

Conditions evaluated:
  (0) ImageNet frozen encoder + NO noise   ← clean baseline / OOD upper bound
  (1) ImageNet frozen encoder + σ=0.02
  (2) Scheme A encoder        + σ=0.02   (CE + L2 anchor, fine-tuned on aux)
  (3) Scheme B encoder        + σ=0.02   (SupCon, fine-tuned on aux)

Comparison (0) vs (1) isolates the privacy contribution from noise alone.
Comparing (1) vs (2)/(3) shows whether encoder fine-tuning adds further
"distribution-shift" privacy on top.

Test set: federated_pool, stratified-sampled by class (default 1/5 ratio →
9000 images, 90/class).
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
from utils import set_seed, split_decoder_pool


def epsilon_from_sigma(sigma, delta=1e-6, clip_C=1.0):
    if sigma <= 0:
        return float("inf")
    return clip_C * math.sqrt(2.0 * math.log(1.25 / delta)) / sigma


def stratified_subset(labels: np.ndarray, fraction: float,
                      rng: np.random.Generator):
    """Pick `fraction` of indices per class for balanced subset."""
    out = []
    n_classes = int(labels.max()) + 1
    for c in range(n_classes):
        idx_c = np.where(labels == c)[0]
        n_take = max(1, int(round(len(idx_c) * fraction)))
        pick = rng.choice(idx_c, size=n_take, replace=False)
        out.append(pick)
    return np.concatenate(out)


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
def evaluate_condition(encoder, decoder, train_ds, sample_idx, sigma,
                       clip_C, batch_size, gen_seed, device):
    """Run encode+(optional noise)+decode on all sample_idx; return per-sample PSNR."""
    enc_tf = T.Compose([
        T.Resize(256), T.CenterCrop(224), T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    tgt_tf = T.ToTensor()

    psnr_all = []
    n = len(sample_idx)
    rec_for_grid = None  # save first batch for visualization

    for start in range(0, n, batch_size):
        chunk = sample_idx[start:start + batch_size]
        enc_inputs = torch.stack(
            [enc_tf(train_ds[int(i)][0]) for i in chunk]).to(device)
        targets = torch.stack(
            [tgt_tf(train_ds[int(i)][0]) for i in chunk]).to(device)

        # Per-batch generator for reproducibility
        gen = torch.Generator(device=device).manual_seed(gen_seed + start)
        z = encode_clip_optionally_noisy(encoder, enc_inputs, sigma,
                                          clip_C, gen)
        recon = decoder(z).clamp(0, 1)
        psnr_all.append(per_sample_psnr(recon, targets))
        if rec_for_grid is None:
            rec_for_grid = (recon[:8].cpu(), targets[:8].cpu())

    return np.concatenate(psnr_all), rec_for_grid


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--decoder_weights", default="models/decoder_frozen_enc.pt")
    p.add_argument("--scheme_a_encoder",
                   default="models/encoder_supervised.pt")
    p.add_argument("--scheme_b_encoder",
                   default="models/encoder_supcon.pt")
    p.add_argument("--sigma", type=float, default=0.02)
    p.add_argument("--clip_C", type=float, default=1.0)
    p.add_argument("--delta", type=float, default=1e-6)
    p.add_argument("--fraction", type=float, default=0.2,
                   help="Fraction of federated_pool to evaluate (stratified "
                        "by class). 0.2 → 9000 images / 90 per class.")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", default="results/encoder_privacy")
    args = p.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    eps = epsilon_from_sigma(args.sigma, args.delta, args.clip_C)
    print(f"σ={args.sigma}  δ={args.delta}  ε≈{eps:.1f}\n")

    decoder = Decoder(embed_dim=512).to(device).eval()
    decoder.load_state_dict(torch.load(args.decoder_weights,
                                       map_location=device))
    print(f"Loaded decoder: {args.decoder_weights}\n")

    # ---- Stratified subset of federated_pool ----
    train_ds = torchvision.datasets.CIFAR100(
        root="./data", train=True, download=False, transform=None)
    labels = np.array(train_ds.targets)
    decoder_pool, federated_pool = split_decoder_pool(
        labels, frac=0.1, seed=args.seed)
    assert len(set(decoder_pool.tolist()) & set(federated_pool.tolist())) == 0

    fed_labels = labels[federated_pool]
    rng = np.random.default_rng(args.seed + 1)
    pick_local = stratified_subset(fed_labels, args.fraction, rng)
    sample_idx = federated_pool[pick_local]
    print(f"Stratified subset: {len(sample_idx)} images "
          f"(~{len(sample_idx)//100} per class) from federated_pool\n")

    # ---- Conditions ----
    conditions = [
        ("ImageNet frozen,  no noise",
         None, 0.0, "tab:gray"),
        ("ImageNet frozen,  σ=0.02",
         None, args.sigma, "tab:blue"),
        ("Scheme A: CE+anchor, σ=0.02",
         args.scheme_a_encoder, args.sigma, "tab:orange"),
        ("Scheme B: SupCon,    σ=0.02",
         args.scheme_b_encoder, args.sigma, "tab:red"),
    ]

    summary = {
        "sigma": args.sigma,
        "delta": args.delta,
        "epsilon": eps,
        "clip_C": args.clip_C,
        "fraction_of_federated_pool": args.fraction,
        "n_samples": int(len(sample_idx)),
        "decoder_weights": args.decoder_weights,
        "conditions": [],
    }

    grid_rows = [("Original", None)]

    for name, enc_path, sigma, _color in conditions:
        encoder, _ = build_encoder(device, weights_path=enc_path)
        psnr, (rec, tgt) = evaluate_condition(
            encoder, decoder, train_ds, sample_idx, sigma,
            args.clip_C, args.batch_size, gen_seed=args.seed + 100,
            device=device)
        del encoder
        torch.cuda.empty_cache() if device == "cuda" else None

        summary["conditions"].append({
            "name": name,
            "encoder": enc_path or "ImageNet pretrained",
            "sigma": sigma,
            "psnr_mean": float(psnr.mean()),
            "psnr_std": float(psnr.std()),
            "psnr_median": float(np.median(psnr)),
        })
        if grid_rows[0][1] is None:
            grid_rows[0] = ("Original", tgt)
        grid_rows.append((name, rec))

        print(f"[{name:35s}]  PSNR = {psnr.mean():.2f} ± {psnr.std():.2f} dB"
              f"   (median {np.median(psnr):.2f})")

    # Privacy gain (vs ImageNet clean baseline)
    base = summary["conditions"][0]["psnr_mean"]
    print()
    for cond in summary["conditions"][1:]:
        cond["privacy_gain_db"] = base - cond["psnr_mean"]
        print(f"  {cond['name']:35s}  privacy gain = "
              f"{cond['privacy_gain_db']:+.2f} dB (vs ImageNet clean)")

    json_path = os.path.join(args.out_dir, "encoder_privacy.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nJSON: {json_path}")

    # ---- Bar plot ----
    names = [c["name"] for c in summary["conditions"]]
    psnrs = [c["psnr_mean"] for c in summary["conditions"]]
    stds = [c["psnr_std"] for c in summary["conditions"]]
    colors = [c[3] for c in conditions]

    fig, ax = plt.subplots(figsize=(11, 6.5))
    bars = ax.bar(np.arange(len(names)), psnrs, yerr=stds, capsize=6,
                  color=colors, edgecolor="black", alpha=0.85)

    # Reference line: clean baseline
    ax.axhline(base, color="black", ls="--", lw=1, alpha=0.5,
               label=f"clean baseline = {base:.2f} dB")

    for i, (m, s) in enumerate(zip(psnrs, stds)):
        ax.text(i, m + 0.18, f"{m:.2f} dB", ha="center",
                fontsize=10, fontweight="bold")
        if i > 0:
            gain = base - m
            ax.text(i, m / 2, f"−{gain:.2f} dB\nprivacy", ha="center",
                    va="center", fontsize=9, color="darkred",
                    fontweight="bold")

    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels([n.replace(", ", "\n") for n in names], fontsize=9)
    ax.set_ylabel("Reconstruction PSNR (dB)   ↓ = better privacy",
                  fontsize=12)
    ax.set_title(
        f"Privacy Evaluation — encoder + noise vs clean baseline\n"
        f"σ={args.sigma},  ε≈{eps:.0f},  δ={args.delta}   |   "
        f"{summary['n_samples']} client images ({int(args.fraction*100)}% of "
        f"federated_pool, stratified)",
        fontsize=11.5)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    bar_path = os.path.join(args.out_dir, "encoder_privacy_bar.png")
    fig.savefig(bar_path, dpi=150)
    plt.close(fig)
    print(f"Bar:  {bar_path}")

    # ---- Visualization grid ----
    n_show = 8
    fig, axes = plt.subplots(len(grid_rows), n_show,
                              figsize=(2 * n_show, 1.6 * len(grid_rows) + 0.5))
    for r, (label, panel) in enumerate(grid_rows):
        for c in range(n_show):
            ax = axes[r, c] if n_show > 1 else axes[r]
            ax.imshow(panel[c].permute(1, 2, 0).numpy())
            ax.set_xticks([])
            ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(label, fontsize=8)
    fig.suptitle(
        f"Reconstruction grid (8 random client images) — σ={args.sigma}, "
        f"ε≈{eps:.0f}, δ={args.delta}",
        fontsize=11)
    fig.tight_layout()
    grid_path = os.path.join(args.out_dir, "encoder_privacy_grid.png")
    fig.savefig(grid_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Grid: {grid_path}")


if __name__ == "__main__":
    main()
