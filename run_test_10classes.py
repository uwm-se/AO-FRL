#!/usr/bin/env python3
"""
Test: Agent-Orchestrated Privacy-Gated Federated Learning on 10-class CIFAR-100.

Demonstrates the full agent interaction loop:
  1. ClientAgent  → extract privacy-gated embeddings + summary
  2. ClientAgent  → upload (embeddings, labels, summary) to ServerAgent
  3. ServerAgent  → aggregate embeddings, train MLP head
  4. ServerAgent  → orchestrate: analyze summaries → per-client instructions
  5. ServerAgent  → broadcast updated head to all clients
  6. ClientAgent  → apply server instructions, evaluate locally
  7. EvaluatorAgent → global test evaluation, logging, plotting

Usage:
    python run_test_10classes.py
"""

import json
import os
import sys
import time
import logging

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T

from utils import (set_seed, dirichlet_partition, split_train_val,
                   estimate_comm_bytes, load_skill_files, log_skills)
from agents.server_agent import ServerAgent
from agents.client_agent import ClientAgent
from agents.evaluator_agent import EvaluatorAgent

# ── Config ──────────────────────────────────────────────────────────────
N_CLASSES_SUBSET = 10       # classes 0-9
N_CLIENTS = 5
ROUNDS = 20
ALPHA = 0.3
LOCAL_EPOCHS = 2
SEED = 42
RESULTS_DIR = "test_results"

CFG = {
    "n_clients": N_CLIENTS,
    "alpha": ALPHA,
    "rounds": ROUNDS,
    "local_epochs": LOCAL_EPOCHS,
    "batch_size": 64,
    "server_lr": 1e-3,
    "server_optimizer": "adam",
    "server_train_epochs": 3,
    "head_hidden": 256,
    "sigma": 0.02,
    "clip_C": 1.0,
    "tau_high": 0.95,
    "upload_budget": 500,
    "n_views": 2,
    "low_data_k": 10,
    "high_risk_r": 0.30,
    "val_ratio": 0.1,
    "seed": SEED,
}


def setup_logging(results_dir):
    os.makedirs(results_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(results_dir, "experiment.log"),
                                mode='w'),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
    return logging.getLogger("AgentFL")


def build_encoder(device):
    """Build frozen ResNet-18 encoder (dim=512)."""
    import torchvision.models as models
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    encoder = nn.Sequential(*list(model.children())[:-1], nn.Flatten())
    encoder = encoder.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder, 512


class FilteredDataset(torch.utils.data.Dataset):
    """Wraps a CIFAR-100 dataset, keeping only samples whose label is in
    `keep_classes` and remapping labels to 0..len(keep_classes)-1."""

    def __init__(self, base_dataset, keep_classes):
        self.base = base_dataset
        targets = np.array(base_dataset.targets)
        self.remap = {orig: new for new, orig in enumerate(sorted(keep_classes))}
        mask = np.isin(targets, list(keep_classes))
        self.indices = np.where(mask)[0]
        # Rebuild .targets for downstream code that reads it
        self.targets = [self.remap[targets[i]] for i in self.indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img, label = self.base[self.indices[idx]]
        return img, self.remap[label]


# ── Main ────────────────────────────────────────────────────────────────

def main():
    logger = setup_logging(RESULTS_DIR)
    logger.info("=" * 60)
    logger.info("Agent-Orchestrated FL — 10-class CIFAR-100 test")
    logger.info("=" * 60)
    logger.info(f"Config: {N_CLIENTS} clients, {ROUNDS} rounds, "
                f"alpha={ALPHA}, seed={SEED}")

    # Load skill files for auditability
    skills = load_skill_files("skills")
    log_skills(skills, logger)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    set_seed(SEED)

    n_classes = N_CLASSES_SUBSET
    keep_classes = list(range(n_classes))

    # ── Data ────────────────────────────────────────────────────────────
    logger.info("Loading CIFAR-100 → filtering to 10 classes (0-9)...")
    raw_train = torchvision.datasets.CIFAR100(
        root="./data", train=True, download=True, transform=None)
    raw_test = torchvision.datasets.CIFAR100(
        root="./data", train=False, download=True, transform=None)

    train_ds = FilteredDataset(raw_train, keep_classes)
    test_ds = FilteredDataset(raw_test, keep_classes)
    logger.info(f"  Train: {len(train_ds)} samples | Test: {len(test_ds)} samples")

    # ── Encoder ─────────────────────────────────────────────────────────
    logger.info("Building frozen ResNet-18 encoder...")
    encoder, embed_dim = build_encoder(device)

    # ── Partition among clients ─────────────────────────────────────────
    logger.info(f"Dirichlet partition (alpha={ALPHA}) → {N_CLIENTS} clients")
    labels_np = np.array(train_ds.targets)
    all_indices = np.arange(len(train_ds))
    client_indices = dirichlet_partition(labels_np, N_CLIENTS, ALPHA, SEED)

    # ── Create Agents ───────────────────────────────────────────────────
    logger.info("Creating agents...")

    # Server Agent
    server = ServerAgent(embed_dim, n_classes, N_CLIENTS, device, CFG)
    logger.info("  [ServerAgent] created — MLP head, orchestrator ready")

    # Evaluator Agent
    evaluator = EvaluatorAgent(test_ds, encoder, n_classes, device, RESULTS_DIR)
    logger.info("  [EvaluatorAgent] created — will cache test embeddings on first eval")

    # Client Agents
    clients = []
    for i in range(N_CLIENTS):
        tr_idx, va_idx = split_train_val(client_indices[i],
                                          CFG["val_ratio"], SEED)
        c = ClientAgent(
            client_id=i,
            train_indices=tr_idx,
            val_indices=va_idx,
            dataset=train_ds,
            encoder=encoder,
            embed_dim=embed_dim,
            n_classes=n_classes,
            device=device,
            cfg=CFG,
        )
        nc = len(set(labels_np[tr_idx]))
        logger.info(f"  [ClientAgent {i}] {len(tr_idx)} train / "
                     f"{len(va_idx)} val samples, {nc} classes")
        clients.append(c)

    # ── Agent Interaction Loop ──────────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("Starting Agent Interaction Loop")
    logger.info("=" * 60)

    cumulative_comm = 0

    for rnd in range(1, ROUNDS + 1):
        t0 = time.time()
        round_comm = 0

        # ── Step 1-2: ClientAgents extract & upload ─────────────────────
        all_embs, all_labs, summaries = [], [], []
        for c in clients:
            embs, labs, summary = c.extract_gated_embeddings(
                n_views=CFG["n_views"])
            all_embs.append(embs)
            all_labs.append(labs)
            summaries.append(summary)
            upload_bytes = estimate_comm_bytes(embs)
            round_comm += upload_bytes

        # ── Step 3: ServerAgent aggregates & trains head ────────────────
        # Summary comm cost
        round_comm += N_CLIENTS * 4 * (n_classes + 5)  # histogram + scalars

        valid_e = [e for e in all_embs if e.numel() > 0]
        valid_l = [l for l in all_labs if l.numel() > 0]
        if valid_e:
            merged_embs = torch.cat(valid_e)
            merged_labs = torch.cat(valid_l)
        else:
            merged_embs = torch.zeros(0, embed_dim)
            merged_labs = torch.zeros(0, dtype=torch.long)

        broadcast_bytes = server.train_head(
            merged_embs, merged_labs, epochs=CFG["server_train_epochs"])

        # ── Step 4: ServerAgent orchestrates → per-client instructions ──
        instructions = server.orchestrate(summaries)

        # ── Step 5: Broadcast head + instructions to clients ────────────
        round_comm += broadcast_bytes * N_CLIENTS
        cumulative_comm += round_comm

        # ── Step 6: ClientAgents apply instructions & evaluate locally ──
        local_accs = []
        for instr in instructions:
            cid = instr["client_id"]
            clients[cid].apply_server_instructions(instr)
            acc_local, _ = clients[cid].evaluate_local(server.get_head())
            local_accs.append(acc_local)

        # ── Step 7: EvaluatorAgent global test ──────────────────────────
        acc, f1 = evaluator.evaluate(server.get_head(), "Proposed",
                                      rnd, round_comm, cumulative_comm)

        dt = time.time() - t0
        total_uploaded = sum(s["n_uploaded"] for s in summaries)
        avg_reject = np.mean([s["reject_ratio"] for s in summaries])
        avg_local_acc = np.mean(local_accs) if local_accs else 0.0

        if rnd % 5 == 0 or rnd == 1:
            logger.info(
                f"[Round {rnd:2d}] "
                f"GlobalAcc:{acc:.4f} F1:{f1:.4f} | "
                f"AvgLocalAcc:{avg_local_acc:.4f} | "
                f"Uploaded:{total_uploaded} Rejected:{avg_reject:.2f} | "
                f"Comm:{cumulative_comm/1e6:.1f}MB ({dt:.1f}s)")

            # Log a sample of server instructions for visibility
            sample = instructions[0]
            logger.info(
                f"  └─ Server→Client0 instructions: "
                f"budget={sample['upload_budget']}, "
                f"sigma={sample['sigma']:.4f}, "
                f"aug={sample['augmentation_mode']}")

    # ── Save Results ────────────────────────────────────────────────────
    evaluator.save_csv("Proposed")
    evaluator.save_final_json()
    evaluator.plot_comparisons()

    summary_path = os.path.join(RESULTS_DIR, "final_summary.json")
    with open(summary_path) as f:
        summary = json.load(f)

    logger.info("")
    logger.info("=" * 60)
    logger.info("FINAL RESULTS")
    logger.info("=" * 60)
    for method, stats in summary.items():
        logger.info(f"  {method}:")
        for k, v in stats.items():
            logger.info(f"    {k}: {v:.4f}" if isinstance(v, float)
                        else f"    {k}: {v}")

    logger.info(f"\nOutputs in {RESULTS_DIR}/:")
    for f_name in sorted(os.listdir(RESULTS_DIR)):
        fpath = os.path.join(RESULTS_DIR, f_name)
        size = os.path.getsize(fpath)
        logger.info(f"  {f_name} ({size:,} bytes)")

    logger.info("\nDone.")


if __name__ == "__main__":
    main()
