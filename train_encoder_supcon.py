"""Scheme B: Supervised Contrastive (SupCon) fine-tune of the encoder.

Trains the encoder + a small projection head with the SupCon loss
(Khosla et al. 2020, "Supervised Contrastive Learning"):
  L_i = -1/|P(i)| * Σ_{p∈P(i)} log[exp(z_i·z_p/τ) /
                                   Σ_{a∈A(i)} exp(z_i·z_a/τ)]
where for each anchor i, P(i) is the set of same-class samples (excluding
self) within the 2N-batch (2 augmentation views per image), and A(i) is all
samples except i.

The projection head is discarded after training; only encoder weights are
saved for downstream AO-FRL use. Same partial-unfreeze + anchor-reg + early
stop guards as scheme A, since 5000 images is small.
"""

import argparse
import csv
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset

from run_experiment import build_encoder
from train_encoder_supervised import set_partial_unfreeze, anchor_l2
from utils import set_seed, split_decoder_pool, split_train_val


class TwoViewSubset(Dataset):
    """Returns two independently-augmented views of the same image + label."""

    def __init__(self, dataset, indices, transform):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        img, label = self.dataset[self.indices[i]]
        return self.transform(img), self.transform(img), label


def supcon_loss(features: torch.Tensor, labels: torch.Tensor,
                temperature: float = 0.07) -> torch.Tensor:
    """SupCon loss. features: (B, D) L2-normalized; labels: (B,)."""
    device = features.device
    B = features.size(0)

    sim = features @ features.T / temperature                  # (B, B)
    self_mask = torch.eye(B, dtype=torch.bool, device=device)
    label_mask = labels.unsqueeze(0) == labels.unsqueeze(1)
    pos_mask = label_mask & ~self_mask                         # same class & not self

    # Numerical stability
    sim_max, _ = sim.max(dim=1, keepdim=True)
    sim = sim - sim_max.detach()

    exp_sim = torch.exp(sim) * (~self_mask).float()            # zero-out self
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)

    pos_count = pos_mask.float().sum(dim=1).clamp(min=1.0)
    mean_log_prob_pos = (pos_mask.float() * log_prob).sum(dim=1) / pos_count

    return -mean_log_prob_pos.mean()


def linear_probe_eval(encoder, train_imgs, train_labs, val_imgs, val_labs,
                      device, embed_dim=512, n_classes=100,
                      epochs=20, batch_size=128):
    """Train a fresh linear classifier on encoder features and report val acc.

    Encoder is in eval mode and features are extracted under no_grad; only
    the freshly-initialized head learns (with grad).
    """
    encoder.eval()
    with torch.no_grad():
        def feats(imgs):
            out = []
            for i in range(0, imgs.size(0), batch_size):
                z = F.normalize(encoder(imgs[i:i+batch_size].to(device)), dim=1)
                out.append(z.cpu())
            return torch.cat(out)
        z_train = feats(train_imgs).to(device)
        z_val = feats(val_imgs).to(device)
    train_labs = train_labs.to(device)
    val_labs = val_labs.to(device)

    head = nn.Linear(embed_dim, n_classes).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-2, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(epochs):
        idx = torch.randperm(z_train.size(0), device=device)
        for i in range(0, z_train.size(0), batch_size):
            sel = idx[i:i+batch_size]
            head.train()
            logits = head(z_train[sel])
            loss = loss_fn(logits, train_labs[sel])
            opt.zero_grad(); loss.backward(); opt.step()

    head.eval()
    with torch.no_grad():
        val_preds = head(z_val).argmax(1)
    return (val_preds == val_labs).float().mean().item()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=128,
                   help="Per-batch image count (each image becomes 2 views).")
    p.add_argument("--encoder_lr", type=float, default=1e-4)
    p.add_argument("--proj_lr", type=float, default=1e-3)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--anchor_weight", type=float, default=1e-3)
    p.add_argument("--unfreeze_block", type=int, default=4, choices=[1, 2, 3, 4])
    p.add_argument("--early_stop_patience", type=int, default=8)
    p.add_argument("--probe_every", type=int, default=2,
                   help="Run linear-probe eval every N epochs (slow).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="models/encoder_supcon.pt")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--history_csv",
                   default="models/encoder_supcon_history.csv")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    train_ds = torchvision.datasets.CIFAR100(
        root="./data", train=True, download=True, transform=None)
    labels_all = np.array(train_ds.targets)

    decoder_pool, _ = split_decoder_pool(labels_all, frac=0.1, seed=args.seed)
    train_idx, val_idx = split_train_val(decoder_pool, val_ratio=0.1,
                                         seed=args.seed)
    print(f"Holdout split: train {len(train_idx)} / val {len(val_idx)}")

    aug = T.Compose([
        T.RandomResizedCrop(224, scale=(0.4, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        T.RandomGrayscale(p=0.2),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_tf = T.Compose([
        T.Resize(256), T.CenterCrop(224), T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_loader = DataLoader(TwoViewSubset(train_ds, train_idx, aug),
                              batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=(device == "cuda"), drop_last=True)

    # For periodic linear-probe eval: precompute eval-tf'd train + val tensors
    def collect_imgs(indices):
        out_imgs, out_labs = [], []
        for i in indices:
            img, lab = train_ds[i]
            out_imgs.append(eval_tf(img))
            out_labs.append(lab)
        return torch.stack(out_imgs), torch.tensor(out_labs, dtype=torch.long)
    probe_train_imgs, probe_train_labs = collect_imgs(train_idx)
    probe_val_imgs, probe_val_labs = collect_imgs(val_idx)

    encoder, embed_dim = build_encoder(device, weights_path=None)
    proj = nn.Sequential(
        nn.Linear(embed_dim, embed_dim),
        nn.ReLU(inplace=True),
        nn.Linear(embed_dim, 128),
    ).to(device)

    n_unfrozen = set_partial_unfreeze(encoder, args.unfreeze_block)
    n_total = sum(p.numel() for p in encoder.parameters())
    print(f"Unfrozen encoder params: {n_unfrozen}/{n_total} "
          f"({100*n_unfrozen/n_total:.1f}%)")

    anchor = {n: p.detach().clone()
              for n, p in encoder.named_parameters() if p.requires_grad}

    optimizer = torch.optim.AdamW([
        {"params": [p for p in encoder.parameters() if p.requires_grad],
         "lr": args.encoder_lr},
        {"params": proj.parameters(), "lr": args.proj_lr},
    ])

    os.makedirs(os.path.dirname(args.history_csv), exist_ok=True)
    with open(args.history_csv, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_supcon", "anchor_l2",
                                "val_probe_acc", "elapsed_s"])

    best_val_acc = -1.0
    best_epoch = 0
    rounds_since_best = 0
    last_probe_acc = -1.0
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        encoder.train(); proj.train()
        sum_loss, sum_anchor, n_batches = 0.0, 0.0, 0
        for v1, v2, labs in train_loader:
            v1 = v1.to(device, non_blocking=True)
            v2 = v2.to(device, non_blocking=True)
            labs = labs.to(device, non_blocking=True)

            x = torch.cat([v1, v2], dim=0)              # (2B, ...)
            y = torch.cat([labs, labs], dim=0)          # (2B,)
            z = encoder(x)
            z = proj(z)
            z = F.normalize(z, dim=1)

            sc = supcon_loss(z, y, temperature=args.temperature)
            anc = anchor_l2(encoder, anchor, args.anchor_weight)
            loss = sc + anc

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sum_loss += sc.item()
            sum_anchor += anc.item()
            n_batches += 1

        avg_supcon = sum_loss / max(n_batches, 1)
        avg_anchor = sum_anchor / max(n_batches, 1)
        elapsed = time.time() - t0

        if epoch % args.probe_every == 0 or epoch == args.epochs:
            last_probe_acc = linear_probe_eval(
                encoder, probe_train_imgs, probe_train_labs,
                probe_val_imgs, probe_val_labs, device,
                embed_dim=embed_dim, n_classes=100, epochs=15)

        print(f"[{epoch:3d}/{args.epochs}] supcon={avg_supcon:.3f} "
              f"anchor={avg_anchor:.3f} | "
              f"val probe acc={last_probe_acc:.3f} | {elapsed:.0f}s")

        with open(args.history_csv, "a", newline="") as f:
            csv.writer(f).writerow(
                [epoch, avg_supcon, avg_anchor, last_probe_acc, elapsed])

        if last_probe_acc > best_val_acc:
            best_val_acc = last_probe_acc
            best_epoch = epoch
            rounds_since_best = 0
            os.makedirs(os.path.dirname(args.output), exist_ok=True)
            torch.save(encoder.state_dict(), args.output)
            print(f"  -> saved best (val probe acc={last_probe_acc:.4f})")
        else:
            rounds_since_best += 1
            if rounds_since_best >= args.early_stop_patience:
                print(f"Early stop at epoch {epoch} (best="
                      f"{best_val_acc:.4f} @ epoch {best_epoch})")
                break

    print(f"\nDone. Best val probe acc: {best_val_acc:.4f} @ epoch {best_epoch}")
    print(f"Encoder weights: {args.output}")


if __name__ == "__main__":
    main()
