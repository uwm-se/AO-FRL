"""Evaluate two decoders side-by-side on CIFAR-100 test set:

  (A) decoder trained on F.normalize'd embeddings  → current pipeline
  (B) decoder trained on raw (un-normalized) embeddings → no-normalize

Both use the same frozen ImageNet encoder. σ=0 (no DP noise) — we are
isolating the privacy contribution of the L2-normalize step alone, on top
of the encoder's lossy compression.
"""
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

from models import Decoder
from run_experiment import build_encoder
from utils import set_seed


@torch.no_grad()
def encode(encoder, imgs, normalize_z, clip_C=1.0):
    z = encoder(imgs)
    if normalize_z:
        z = F.normalize(z, dim=1)
        norms = z.norm(dim=1, keepdim=True).clamp(min=1e-12)
        z = z * torch.clamp(clip_C / norms, max=1.0)
    return z


def per_sample_psnr(rec, tgt):
    mse = (rec - tgt).pow(2).mean(dim=(1, 2, 3)).clamp(min=1e-12)
    return (10.0 * torch.log10(1.0 / mse)).cpu().numpy()


@torch.no_grad()
def evaluate(encoder, decoder, ds, normalize_z, batch_size, device):
    enc_tf = T.Compose([
        T.Resize(256), T.CenterCrop(224), T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    tgt_tf = T.ToTensor()

    psnrs = []
    n = len(ds)
    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        enc_in = torch.stack(
            [enc_tf(ds[i][0]) for i in range(start, end)]).to(device)
        tgt = torch.stack(
            [tgt_tf(ds[i][0]) for i in range(start, end)]).to(device)
        z = encode(encoder, enc_in, normalize_z=normalize_z)
        rec = decoder(z).clamp(0, 1)
        psnrs.append(per_sample_psnr(rec, tgt))
    return np.concatenate(psnrs)


def stats(psnr, label):
    return (f"{label:30s}  mean={psnr.mean():.3f} dB  "
            f"std={psnr.std():.3f}  "
            f"median={np.median(psnr):.3f}  "
            f"p5={np.percentile(psnr, 5):.3f}  "
            f"p95={np.percentile(psnr, 95):.3f}")


def main():
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    encoder, _ = build_encoder(device, weights_path=None)
    encoder.eval()

    test_ds = torchvision.datasets.CIFAR100(
        root="./data", train=False, download=False, transform=None)
    print(f"CIFAR-100 test set: {len(test_ds)} images\n")

    # (A) decoder trained on F.normalize'd embeddings (current baseline)
    dec_norm = Decoder(embed_dim=512).to(device).eval()
    dec_norm.load_state_dict(torch.load("models/decoder_frozen_enc.pt",
                                         map_location=device))
    psnr_norm = evaluate(encoder, dec_norm, test_ds, normalize_z=True,
                          batch_size=128, device=device)

    # (B) decoder trained on raw (un-normalized) embeddings
    dec_raw = Decoder(embed_dim=512).to(device).eval()
    dec_raw.load_state_dict(torch.load(
        "models/decoder_frozen_enc_no_normalize.pt", map_location=device))
    psnr_raw = evaluate(encoder, dec_raw, test_ds, normalize_z=False,
                         batch_size=128, device=device)

    print("=" * 80)
    print("PSNR on CIFAR-100 test set (10,000 images), σ=0 (no DP noise)")
    print("=" * 80)
    print(stats(psnr_norm, "(A) WITH F.normalize"))
    print(stats(psnr_raw,  "(B) WITHOUT F.normalize (raw embedding)"))
    print("-" * 80)
    diff_mean = psnr_raw.mean() - psnr_norm.mean()
    diff_med  = np.median(psnr_raw) - np.median(psnr_norm)
    print(f"Δ mean (raw − norm)   = {diff_mean:+.3f} dB")
    print(f"Δ median (raw − norm) = {diff_med:+.3f} dB")
    print()
    print("Interpretation:")
    if abs(diff_mean) < 0.5:
        print("  → F.normalize contributes almost no privacy on top of the "
              "encoder. The encoder mapping is essentially the entire "
              "single-release privacy mechanism.")
    elif abs(diff_mean) < 2.0:
        print(f"  → F.normalize contributes {diff_mean:+.2f} dB of additional "
              "privacy hardening. The encoder mapping is the dominant "
              "mechanism; normalize is a minor secondary contributor.")
    else:
        print(f"  → F.normalize contributes a substantial {diff_mean:+.2f} dB "
              "— larger than expected from a 1-scalar projection. Worth a "
              "deeper look.")


if __name__ == "__main__":
    main()
