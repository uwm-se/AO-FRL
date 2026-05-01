"""Train ResNet-18 encoder + Decoder on a held-out 1/10 slice of CIFAR-100.

The decoder pool (5000 images, 50/class) is disjoint from the federated client
pool used by run_experiment.py — they share the same seed, so the same
split_decoder_pool() call produces the same partition in both scripts.

Encoder is fine-tuned (NOT frozen). After training, the decoder is used by the
server agent to invert client-uploaded embeddings (clipped + noised + gated)
back to images for PSNR-based privacy evaluation.
"""

import argparse
import csv
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset

from models import Decoder
from run_experiment import build_encoder
from utils import set_seed, split_decoder_pool, split_train_val


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class AEPairDataset(Dataset):
    """Yields (encoder_input_224, target_32) for the same CIFAR-100 image.

    encoder_input: 224x224, ImageNet-normalized (matches client_agent.py).
    target: 32x32, in [0, 1] — what the decoder must reproduce.
    """

    def __init__(self, dataset, indices: np.ndarray, augment: bool = False):
        self.dataset = dataset
        self.indices = indices
        self.augment = augment

        self.enc_tf = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        self.tgt_tf = T.ToTensor()  # 32x32 PIL -> [0,1] tensor

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        img, _ = self.dataset[self.indices[i]]  # PIL 32x32
        # Optional horizontal flip applied to BOTH views consistently.
        if self.augment and torch.rand(1).item() < 0.5:
            img = TF.hflip(img)
        return self.enc_tf(img), self.tgt_tf(img)


def psnr(mse: float, max_val: float = 1.0) -> float:
    if mse <= 0:
        return float("inf")
    return 10.0 * math.log10(max_val * max_val / mse)


def run_epoch(encoder, decoder, loader, device, optimizer=None,
              normalize_z=True):
    train = optimizer is not None
    encoder.train(train)
    decoder.train(train)

    total_mse, total_n = 0.0, 0
    for enc_in, tgt in loader:
        enc_in = enc_in.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            z = encoder(enc_in)
            if normalize_z:
                z = F.normalize(z, dim=1)  # match federated DP path
            recon = decoder(z)
            loss = F.mse_loss(recon, tgt)

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        bs = tgt.size(0)
        total_mse += loss.item() * bs
        total_n += bs

    avg_mse = total_mse / max(total_n, 1)
    return avg_mse, psnr(avg_mse)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--encoder_lr", type=float, default=1e-4)
    p.add_argument("--decoder_lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--frac", type=float, default=0.1,
                   help="Fraction of CIFAR-100 train set per class for decoder pool")
    p.add_argument("--val_ratio", type=float, default=0.1,
                   help="Within decoder pool, fraction held out for validation")
    p.add_argument("--seed", type=int, default=42,
                   help="MUST match run_experiment.py --seed for disjoint federated pool")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", default="auto")
    p.add_argument("--out_dir", default="models")
    p.add_argument("--no_augment", action="store_true")
    p.add_argument("--freeze_encoder", action="store_true",
                   help="Freeze the ImageNet-pretrained encoder; train decoder "
                        "only. Use this when downstream classification has "
                        "shown that fine-tuning hurts utility.")
    p.add_argument("--encoder_out", default="models/encoder_finetuned.pt")
    p.add_argument("--decoder_out", default="models/decoder.pt")
    p.add_argument("--no_normalize", action="store_true",
                   help="Skip F.normalize on encoder output. For ablation: "
                        "train decoder on raw (un-normalized) embeddings to "
                        "isolate the privacy contribution of L2 normalize.")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Device: {device}")

    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading CIFAR-100...")
    train_ds = torchvision.datasets.CIFAR100(
        root="./data", train=True, download=True, transform=None)
    labels = np.array(train_ds.targets)

    decoder_pool, federated_pool = split_decoder_pool(
        labels, frac=args.frac, seed=args.seed)
    print(f"Decoder pool: {len(decoder_pool)} | "
          f"Federated pool (excluded): {len(federated_pool)}")

    # Sanity: disjoint
    assert len(set(decoder_pool.tolist()) & set(federated_pool.tolist())) == 0

    train_idx, val_idx = split_train_val(
        decoder_pool, val_ratio=args.val_ratio, seed=args.seed)
    print(f"  -> train {len(train_idx)} / val {len(val_idx)}")

    train_loader = DataLoader(
        AEPairDataset(train_ds, train_idx, augment=not args.no_augment),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device == "cuda"))
    val_loader = DataLoader(
        AEPairDataset(train_ds, val_idx, augment=False),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device == "cuda"))

    # Encoder: same architecture as client_agent.py uses.
    encoder, embed_dim = build_encoder(device)
    decoder = Decoder(embed_dim=embed_dim).to(device)

    if args.freeze_encoder:
        for p in encoder.parameters():
            p.requires_grad = False
        encoder.eval()
        optimizer = torch.optim.Adam(decoder.parameters(),
                                     lr=args.decoder_lr,
                                     weight_decay=args.weight_decay)
        print("Encoder FROZEN — training decoder only.")
    else:
        for p in encoder.parameters():
            p.requires_grad = True
        optimizer = torch.optim.Adam([
            {"params": encoder.parameters(), "lr": args.encoder_lr},
            {"params": decoder.parameters(), "lr": args.decoder_lr},
        ], weight_decay=args.weight_decay)

    history_path = os.path.join(args.out_dir, "autoencoder_history.csv")
    with open(history_path, "w", newline="") as f:
        csv.writer(f).writerow(
            ["epoch", "train_mse", "train_psnr", "val_mse", "val_psnr",
             "elapsed_s"])

    best_val_psnr = -float("inf")
    enc_path = args.encoder_out
    dec_path = args.decoder_out

    normalize_z = not args.no_normalize
    print(f"Normalize encoder output: {normalize_z}")

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        tr_mse, tr_psnr = run_epoch(encoder, decoder, train_loader, device,
                                    optimizer, normalize_z=normalize_z)
        with torch.no_grad():
            va_mse, va_psnr = run_epoch(encoder, decoder, val_loader, device,
                                        normalize_z=normalize_z)

        elapsed = time.time() - t0
        print(f"[{epoch:3d}/{args.epochs}] "
              f"train MSE={tr_mse:.5f} PSNR={tr_psnr:.2f}dB | "
              f"val MSE={va_mse:.5f} PSNR={va_psnr:.2f}dB | "
              f"{elapsed:.0f}s")

        with open(history_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [epoch, tr_mse, tr_psnr, va_mse, va_psnr, elapsed])

        if va_psnr > best_val_psnr:
            best_val_psnr = va_psnr
            if not args.freeze_encoder:
                torch.save(encoder.state_dict(), enc_path)
            torch.save(decoder.state_dict(), dec_path)
            print(f"  -> saved best (val PSNR={va_psnr:.2f}dB)")

    print(f"\nDone. Best val PSNR: {best_val_psnr:.2f} dB")
    print(f"Encoder weights: {enc_path}")
    print(f"Decoder weights: {dec_path}")
    print(f"History: {history_path}")


if __name__ == "__main__":
    main()
