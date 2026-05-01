"""Scheme A: supervised classification fine-tune of the encoder.

Fine-tunes ImageNet-pretrained ResNet-18 on the 5000-image holdout (the same
public auxiliary set used for decoder training, disjoint from federated
clients) with three guards against overfitting on tiny data:
  1. L2 anchor regularization to the original ImageNet weights
  2. Partial unfreeze (only `--unfreeze_block` and later)
  3. Strong augmentation (RandomResizedCrop + Flip + ColorJitter)
  4. Early stopping on val acc

Output: encoder weights at --output. The temporary classifier head is
discarded — downstream the AO-FRL MLPHead is trained from scratch on top of
this encoder.
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
from torch.utils.data import DataLoader

from agents.client_agent import _TransformSubset
from run_experiment import build_encoder
from utils import set_seed, split_decoder_pool, split_train_val


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--encoder_lr", type=float, default=1e-4)
    p.add_argument("--head_lr", type=float, default=1e-3)
    p.add_argument("--anchor_weight", type=float, default=1e-3,
                   help="L2 reg coefficient toward the original ImageNet "
                        "weights. Higher = stay closer to pretrain.")
    p.add_argument("--unfreeze_block", type=int, default=4, choices=[1, 2, 3, 4],
                   help="Unfreeze layer{N..4}. 4 = only the last block.")
    p.add_argument("--early_stop_patience", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="models/encoder_supervised.pt")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--history_csv",
                   default="models/encoder_supervised_history.csv")
    return p.parse_args()


def set_partial_unfreeze(encoder: nn.Sequential, unfreeze_block: int):
    """ResNet-18 stem + 4 blocks + avgpool + flatten in nn.Sequential.

    Children indices: 0=conv1, 1=bn1, 2=relu, 3=maxpool, 4=layer1,
    5=layer2, 6=layer3, 7=layer4, 8=avgpool, 9=Flatten.

    unfreeze_block=4 unfreezes only layer4 (index 7).
    unfreeze_block=1 unfreezes layer1+layer2+layer3+layer4.
    """
    layer_first_idx = {1: 4, 2: 5, 3: 6, 4: 7}[unfreeze_block]
    n_unfrozen = 0
    for i, child in enumerate(encoder.children()):
        unfreeze = i >= layer_first_idx
        for p in child.parameters():
            p.requires_grad = unfreeze
            if unfreeze:
                n_unfrozen += p.numel()
    return n_unfrozen


def anchor_l2(encoder: nn.Module, anchor: dict, weight: float):
    if weight <= 0:
        return torch.zeros((), device=next(encoder.parameters()).device)
    loss = 0.0
    for n, p in encoder.named_parameters():
        if p.requires_grad and n in anchor:
            loss = loss + ((p - anchor[n]) ** 2).sum()
    return weight * loss


def evaluate(encoder, head, loader, device):
    encoder.eval(); head.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labs in loader:
            imgs = imgs.to(device, non_blocking=True)
            labs = labs.to(device, non_blocking=True)
            z = F.normalize(encoder(imgs), dim=1)
            preds = head(z).argmax(1)
            correct += (preds == labs).sum().item()
            total += labs.size(0)
    return correct / max(total, 1)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    train_ds = torchvision.datasets.CIFAR100(
        root="./data", train=True, download=True, transform=None)
    labels = np.array(train_ds.targets)

    decoder_pool, _ = split_decoder_pool(labels, frac=0.1, seed=args.seed)
    train_idx, val_idx = split_train_val(decoder_pool, val_ratio=0.1,
                                         seed=args.seed)
    print(f"Holdout split: train {len(train_idx)} / val {len(val_idx)}")

    train_tf = T.Compose([
        T.RandomResizedCrop(224, scale=(0.6, 1.0)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.4, 0.4, 0.4, 0.1),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_tf = T.Compose([
        T.Resize(256), T.CenterCrop(224), T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    train_loader = DataLoader(_TransformSubset(train_ds, train_idx, train_tf),
                              batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=(device == "cuda"))
    val_loader = DataLoader(_TransformSubset(train_ds, val_idx, eval_tf),
                            batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers,
                            pin_memory=(device == "cuda"))

    encoder, embed_dim = build_encoder(device, weights_path=None)
    head = nn.Linear(embed_dim, 100).to(device)

    n_unfrozen = set_partial_unfreeze(encoder, args.unfreeze_block)
    n_total = sum(p.numel() for p in encoder.parameters())
    print(f"Unfrozen encoder params: {n_unfrozen}/{n_total} "
          f"({100*n_unfrozen/n_total:.1f}%)")

    anchor = {n: p.detach().clone()
              for n, p in encoder.named_parameters() if p.requires_grad}

    optimizer = torch.optim.AdamW([
        {"params": [p for p in encoder.parameters() if p.requires_grad],
         "lr": args.encoder_lr},
        {"params": head.parameters(), "lr": args.head_lr},
    ])
    loss_fn = nn.CrossEntropyLoss()

    os.makedirs(os.path.dirname(args.history_csv), exist_ok=True)
    with open(args.history_csv, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "train_acc",
                                "val_acc", "anchor_l2", "elapsed_s"])

    best_val_acc = -1.0
    best_epoch = 0
    rounds_since_best = 0
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        encoder.train(); head.train()
        loss_sum, anchor_sum, correct, total = 0.0, 0.0, 0, 0
        for imgs, labs in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            labs = labs.to(device, non_blocking=True)
            z = F.normalize(encoder(imgs), dim=1)
            logits = head(z)
            ce = loss_fn(logits, labs)
            anc = anchor_l2(encoder, anchor, args.anchor_weight)
            loss = ce + anc

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bs = labs.size(0)
            loss_sum += ce.item() * bs
            anchor_sum += anc.item()
            correct += (logits.argmax(1) == labs).sum().item()
            total += bs

        train_loss = loss_sum / total
        train_acc = correct / total
        val_acc = evaluate(encoder, head, val_loader, device)
        elapsed = time.time() - t0

        print(f"[{epoch:3d}/{args.epochs}] train CE={train_loss:.3f} "
              f"acc={train_acc:.3f} | val acc={val_acc:.3f} | "
              f"anchor_l2={anchor_sum:.3f} | {elapsed:.0f}s")

        with open(args.history_csv, "a", newline="") as f:
            csv.writer(f).writerow(
                [epoch, train_loss, train_acc, val_acc,
                 anchor_sum, elapsed])

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            rounds_since_best = 0
            os.makedirs(os.path.dirname(args.output), exist_ok=True)
            torch.save(encoder.state_dict(), args.output)
            print(f"  -> saved best (val acc={val_acc:.4f})")
        else:
            rounds_since_best += 1
            if rounds_since_best >= args.early_stop_patience:
                print(f"Early stop at epoch {epoch} (best={best_val_acc:.4f} "
                      f"@ epoch {best_epoch})")
                break

    print(f"\nDone. Best val acc: {best_val_acc:.4f} @ epoch {best_epoch}")
    print(f"Encoder weights: {args.output}")


if __name__ == "__main__":
    main()
