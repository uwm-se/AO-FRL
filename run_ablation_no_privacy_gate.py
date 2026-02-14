#!/usr/bin/env python3
"""
Ablation 2: No Privacy Gate
Clients add Gaussian noise but skip the cosine similarity filtering.
All noisy embeddings are accepted (up to budget limit).
"""

import argparse
import copy
import json
import os
import sys
import time
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
import torchvision.transforms as T
from torch.utils.data import DataLoader, TensorDataset

from utils import (set_seed, dirichlet_partition, split_train_val,
                   estimate_comm_bytes, load_skill_files, log_skills)
from agents.server_agent import ServerAgent, MLPHead
from agents.evaluator_agent import EvaluatorAgent
from agents.client_agent import _TransformSubset
from a2a import A2ABus, AgentCard, Part, Artifact

# Import the full run_experiment module to reuse components
import run_experiment as base


class ProposedClientNoPrivacyGate(base.ProposedClient):
    """ProposedClient with Privacy Gate disabled."""

    def extract_gated_embeddings(self, n_views=2):
        """Extract embeddings with noise but WITHOUT privacy gate filtering.
        All noisy embeddings are accepted (no cosine similarity check).
        """
        N = self.embs.size(0)
        labels_np = self.labels.numpy()
        embs_normed = self.embs  # already L2-normalized

        # Generate all noised embeddings WITHOUT filtering
        all_candidates = []  # list of (z_tilde, label)
        for v in range(n_views):
            view_noise = torch.randn(N, self.embed_dim) * 0.01 * (v + 1)
            z_views = embs_normed + view_noise

            # Clipping
            norms = z_views.norm(dim=1, keepdim=True)
            clip_mask = (norms > self.clip_C).squeeze()
            if clip_mask.any():
                z_views[clip_mask] = z_views[clip_mask] * (
                    self.clip_C / norms[clip_mask])

            # Gaussian noise
            z_tilde = z_views + torch.randn_like(z_views) * self.sigma

            # Accept ALL samples (no privacy gate filtering)
            for i in range(N):
                label = int(labels_np[i])
                zt = z_tilde[i]
                all_candidates.append((zt, label))

        # No rejection - all samples accepted
        reject_ratio = 0.0
        total_count = len(all_candidates)

        # Collect embeddings up to upload budget
        all_z, all_y = [], []
        for zt, label in all_candidates:
            all_z.append(zt)
            all_y.append(label)
            if len(all_z) >= self.upload_budget:
                break

        # Apply hooks
        self._apply_hooks(reject_ratio)

        # Fallback (should never happen since we accept all)
        if not all_z:
            # Compute prototypes as fallback
            prototypes = torch.zeros(self.n_classes, self.embed_dim)
            proto_valid = torch.zeros(self.n_classes, dtype=torch.bool)
            for c in range(self.n_classes):
                mask = self.labels == c
                if mask.any():
                    prototypes[c] = F.normalize(embs_normed[mask].mean(0), dim=0)
                    proto_valid[c] = True

            for c in range(self.n_classes):
                if proto_valid[c] and self.label_counts[c] > 0:
                    all_z.append(prototypes[c] +
                                 torch.randn(self.embed_dim) * self.sigma)
                    all_y.append(c)

        if all_z:
            embeddings = torch.stack(all_z)
            labels_out = torch.tensor(all_y, dtype=torch.long)
        else:
            embeddings = torch.zeros(0, self.embed_dim)
            labels_out = torch.zeros(0, dtype=torch.long)

        hist = np.bincount(all_y, minlength=self.n_classes) if all_y else np.zeros(self.n_classes, dtype=int)

        summary = {
            "client_id": self.id,
            "label_histogram": hist.tolist(),
            "reject_ratio": reject_ratio,  # Always 0 (no gate)
            "sigma": self.sigma,
            "n_uploaded": len(all_z),
            "augmentation_mode": self.augmentation_mode,
        }
        return embeddings, labels_out, summary


def main():
    # Parse args
    p = argparse.ArgumentParser(description="Ablation: No Privacy Gate")
    p.add_argument("--rounds", type=int, default=100)
    p.add_argument("--n_clients", type=int, default=20)
    p.add_argument("--alpha", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--results_dir", default="ablation_results/no_privacy_gate")
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    # Setup
    os.makedirs(args.results_dir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu")

    print("="*70)
    print("Ablation Study 2: No Privacy Gate")
    print("="*70)
    print(f"Configuration:")
    print(f"  Rounds: {args.rounds}")
    print(f"  Clients: {args.n_clients}")
    print(f"  Alpha: {args.alpha}")
    print(f"  Device: {device}")
    print(f"  Results: {args.results_dir}")
    print(f"  Privacy: Gaussian noise (σ=0.02) without gate filtering")
    print("="*70)

    # Load CIFAR-100
    train_full = torchvision.datasets.CIFAR100(
        "data", train=True, download=True, transform=None)
    test_full = torchvision.datasets.CIFAR100(
        "data", train=False, download=True, transform=None)

    n_classes = 100

    # Build frozen encoder
    encoder, embed_dim = base.build_encoder(device)

    # Partition data using Dirichlet
    from utils import dirichlet_partition, split_train_val
    train_labels = np.array(train_full.targets)
    client_indices = dirichlet_partition(train_labels, args.n_clients,
                                          args.alpha, args.seed)

    # Precompute ALL train embeddings once
    print("\nPrecomputing train embeddings...")
    all_train_indices = np.arange(len(train_full))
    train_embs, train_labs = base.precompute_embeddings(
        encoder, train_full, all_train_indices, device, batch_size=256)
    # L2 normalize
    train_embs = F.normalize(train_embs, dim=1)

    # Precompute test embeddings
    print("Precomputing test embeddings...")
    test_indices = np.arange(len(test_full))
    test_embs, test_labels = base.precompute_embeddings(
        encoder, test_full, test_indices, device, batch_size=256)
    test_embs = F.normalize(test_embs, dim=1)

    # Create config
    cfg = {
        "sigma": 0.02,
        "clip_C": 1.0,
        "tau_percentile": 0.15,  # Not used (no gate)
        "tau_min": 0.5,  # Not used (no gate)
        "upload_budget": 500,
        "n_views": 2,
        "low_data_k": 10,
        "high_risk_r": 0.30,
        "server_lr": 1e-3,
        "server_train_epochs": 3,
        "head_hidden": 256,
        "replay_decay": 0.995,
        "replay_min_weight": 0.3,
        "server_lr_decay": 0.98,
        "server_lr_min": 1e-4,
    }

    # Create server (standard orchestration)
    server = ServerAgent(
        embed_dim, n_classes, args.n_clients, device, cfg
    )

    # Create clients with NO privacy gate
    clients = {}
    for cid in range(args.n_clients):
        # Split client's data into train/val
        tr_idx, va_idx = split_train_val(client_indices[cid], 0.1, args.seed)

        client = ProposedClientNoPrivacyGate(
            cid, train_embs[tr_idx], train_labs[tr_idx],
            train_embs[va_idx], train_labs[va_idx],
            n_classes, embed_dim, device, cfg
        )
        clients[cid] = client

    # Training loop
    metrics = {
        "round": [],
        "test_acc": [],
        "test_loss": [],
        "comm_bytes": [],
        "avg_reject_ratio": [],
        "avg_budget": [],
        "avg_sigma": [],
        "conservative_count": [],
    }

    total_comm = 0

    for r in range(args.rounds):
        print(f"\n{'='*70}")
        print(f"Round {r+1}/{args.rounds}")
        print(f"{'='*70}")

        # Collect embeddings and summaries from all clients
        all_embeddings = []
        all_labels = []
        summaries = []

        round_comm = 0

        for cid in range(args.n_clients):
            client = clients[cid]
            embs, labs, summary = client.extract_gated_embeddings(n_views=cfg["n_views"])

            all_embeddings.append(embs.to(device))
            all_labels.append(labs.to(device))
            summaries.append(summary)

            # Communication cost
            n_emb = embs.size(0)
            emb_bytes = n_emb * embed_dim * 4
            summary_bytes = 100 * 4 + 32  # histogram + metadata
            round_comm += emb_bytes + summary_bytes

        total_comm += round_comm

        # Server trains head on collected embeddings
        if all_embeddings:
            combined_embs = torch.cat(all_embeddings, dim=0)
            combined_labels = torch.cat(all_labels, dim=0)
            broadcast_bytes = server.train_head(combined_embs, combined_labels,
                                                 epochs=cfg["server_train_epochs"])
            round_comm += broadcast_bytes * args.n_clients
        else:
            combined_embs = torch.zeros(0, embed_dim)
            combined_labels = torch.zeros(0, dtype=torch.long)
            broadcast_bytes = server.train_head(combined_embs, combined_labels,
                                                 epochs=cfg["server_train_epochs"])
            round_comm += broadcast_bytes * args.n_clients

        # Server generates instructions (standard orchestration)
        instructions_list = server.orchestrate(summaries)

        # Clients apply instructions
        for instr in instructions_list:
            cid = instr["client_id"]
            clients[cid].apply_server_instructions(instr)

        # Clients evaluate locally
        for cid in range(args.n_clients):
            clients[cid].evaluate_local(server.get_head())

        # Server evaluates on test set
        head = server.get_head()
        head.eval()
        with torch.no_grad():
            all_preds = []
            bs = 1024
            for i in range(0, test_embs.size(0), bs):
                z = test_embs[i:i+bs].to(device)
                preds = head.to(device)(z).argmax(1).cpu()
                all_preds.append(preds)
            all_preds = torch.cat(all_preds).numpy()
            all_labels = test_labels.numpy()
            test_acc = (all_preds == all_labels).mean()
            # Compute loss
            z = test_embs.to(device)
            y = test_labels.to(device)
            logits = head.to(device)(z)
            test_loss = F.cross_entropy(logits, y).item()

        # Log metrics
        avg_reject = np.mean([s["reject_ratio"] for s in summaries])  # Should be 0
        avg_budget = np.mean([s["n_uploaded"] for s in summaries])
        avg_sigma = np.mean([s["sigma"] for s in summaries])
        conservative_count = sum(1 for s in summaries if s["augmentation_mode"] == "conservative")

        metrics["round"].append(r + 1)
        metrics["test_acc"].append(test_acc)
        metrics["test_loss"].append(test_loss)
        metrics["comm_bytes"].append(total_comm)
        metrics["avg_reject_ratio"].append(avg_reject)
        metrics["avg_budget"].append(avg_budget)
        metrics["avg_sigma"].append(avg_sigma)
        metrics["conservative_count"].append(conservative_count)

        print(f"  Test Acc: {test_acc:.4f}")
        print(f"  Avg Reject Ratio: {avg_reject:.4f} (should be 0.0)")
        print(f"  Avg Budget: {avg_budget:.1f}")
        print(f"  Conservative Clients: {conservative_count}/{args.n_clients}")
        print(f"  Total Comm: {total_comm/1e9:.3f} GB")

    # Save results
    import pandas as pd
    df = pd.DataFrame(metrics)
    csv_path = os.path.join(args.results_dir, "metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Saved metrics to {csv_path}")

    # Save summary
    summary = {
        "ablation": "no_privacy_gate",
        "description": "Gaussian noise added but no Privacy Gate filtering (all samples accepted)",
        "privacy_mechanism": "Gaussian noise (sigma=0.02) only, no cosine similarity filtering",
        "rounds": args.rounds,
        "n_clients": args.n_clients,
        "alpha": args.alpha,
        "final_test_acc": float(test_acc),
        "total_comm_gb": float(total_comm / 1e9),
    }

    with open(os.path.join(args.results_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Ablation 2 Complete!")
    print(f"  Final Test Accuracy: {test_acc:.4f}")
    print(f"  Total Communication: {total_comm/1e9:.3f} GB")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
