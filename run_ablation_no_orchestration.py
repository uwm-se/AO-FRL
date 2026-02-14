#!/usr/bin/env python3
"""
Ablation 1: No Server Orchestration
Server does not generate personalized instructions. All clients use fixed default parameters.
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


class ServerAgentNoOrchestration(ServerAgent):
    """ServerAgent that returns fixed default instructions (no orchestration)."""

    def orchestrate(self, summaries: list) -> list:
        """Return fixed default instructions for all clients."""
        instructions = []
        for s in summaries:
            cid = s["client_id"]
            # Fixed default parameters - no adaptation
            instructions.append({
                "client_id": cid,
                "upload_budget": 500,  # Fixed base budget
                "sigma": 0.02,         # Fixed noise
                "augmentation_mode": "normal",  # Fixed mode
            })
        return instructions


def main():
    # Parse args
    p = argparse.ArgumentParser(description="Ablation: No Server Orchestration")
    p.add_argument("--rounds", type=int, default=100)
    p.add_argument("--n_clients", type=int, default=20)
    p.add_argument("--alpha", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--results_dir", default="ablation_results/no_orchestration")
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    # Setup
    os.makedirs(args.results_dir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu")

    print("="*70)
    print("Ablation Study 1: No Server Orchestration")
    print("="*70)
    print(f"Configuration:")
    print(f"  Rounds: {args.rounds}")
    print(f"  Clients: {args.n_clients}")
    print(f"  Alpha: {args.alpha}")
    print(f"  Device: {device}")
    print(f"  Results: {args.results_dir}")
    print(f"  Fixed params: budget=500, sigma=0.02, aug=normal")
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
        "tau_percentile": 0.15,
        "tau_min": 0.5,
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

    # Create server with NO orchestration
    server = ServerAgentNoOrchestration(
        embed_dim, n_classes, args.n_clients, device, cfg
    )

    # Create clients
    clients = {}
    for cid in range(args.n_clients):
        # Split client's data into train/val
        tr_idx, va_idx = split_train_val(client_indices[cid], 0.1, args.seed)

        client = base.ProposedClient(
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

        # Server generates instructions (fixed defaults, no orchestration)
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
        avg_reject = np.mean([s["reject_ratio"] for s in summaries])
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
        print(f"  Avg Reject Ratio: {avg_reject:.4f}")
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
        "ablation": "no_orchestration",
        "description": "Server returns fixed default instructions (no personalized orchestration)",
        "fixed_params": {
            "upload_budget": 500,
            "sigma": 0.02,
            "augmentation_mode": "normal"
        },
        "rounds": args.rounds,
        "n_clients": args.n_clients,
        "alpha": args.alpha,
        "final_test_acc": float(test_acc),
        "total_comm_gb": float(total_comm / 1e9),
    }

    with open(os.path.join(args.results_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Ablation 1 Complete!")
    print(f"  Final Test Accuracy: {test_acc:.4f}")
    print(f"  Total Communication: {total_comm/1e9:.3f} GB")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
