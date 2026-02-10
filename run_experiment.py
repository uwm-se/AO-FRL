#!/usr/bin/env python3
"""
Federated Learning Experiment: FedAvg vs Agent-Orchestrated Representation Sharing.

Usage:
    python run_experiment.py --rounds 100 --n_clients 20 --alpha 0.3
    python run_experiment.py --rounds 50 --n_clients 10 --alpha 0.5 --seed 123
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


def parse_args():
    p = argparse.ArgumentParser(description="Federated Learning Experiment")
    p.add_argument("--n_clients", type=int, default=20)
    p.add_argument("--alpha", type=float, default=0.3)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--rounds", type=int, default=100)
    p.add_argument("--local_epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--server_lr", type=float, default=1e-3)
    p.add_argument("--fedavg_lr", type=float, default=1e-3)
    p.add_argument("--server_optimizer", default="adam", choices=["adam", "sgd"])
    p.add_argument("--server_train_epochs", type=int, default=3)
    p.add_argument("--head_hidden", type=int, default=256)
    p.add_argument("--sigma", type=float, default=0.02)
    p.add_argument("--clip_C", type=float, default=1.0)
    p.add_argument("--tau_high", type=float, default=0.95)
    p.add_argument("--upload_budget", type=int, default=500)
    p.add_argument("--n_views", type=int, default=2)
    p.add_argument("--low_data_k", type=int, default=10)
    p.add_argument("--high_risk_r", type=float, default=0.30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--results_dir", default="results")
    p.add_argument("--device", default="auto")
    p.add_argument("--methods", nargs="+", default=["fedavg", "proposed"])
    return p.parse_args()


def build_encoder(device):
    """Build a frozen ResNet-18 encoder (penultimate features, dim=512)."""
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    encoder = nn.Sequential(*list(model.children())[:-1], nn.Flatten())
    encoder = encoder.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder, 512


@torch.no_grad()
def precompute_embeddings(encoder, dataset, indices, device, batch_size=256):
    """Precompute embeddings for a set of indices using frozen encoder.
    Returns (embeddings_tensor, labels_tensor).
    """
    transform = T.Compose([
        T.Resize(256), T.CenterCrop(224), T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    subset = _TransformSubset(dataset, indices, transform)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=False)
    embs, labs = [], []
    for imgs, labels in loader:
        z = encoder(imgs.to(device)).cpu()
        z = F.normalize(z, dim=1)
        embs.append(z)
        labs.append(labels)
    return torch.cat(embs), torch.cat(labs)


def setup_logging(results_dir):
    os.makedirs(results_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(results_dir, "experiment.log"),
                                mode='w'),
            logging.StreamHandler(sys.stdout),
        ]
    )
    return logging.getLogger("FedExp")


# ====================================================================== #
#  Lightweight Client for FedAvg (uses precomputed embeddings)            #
# ====================================================================== #
class FedAvgClient:
    """Minimal client for FedAvg: local SGD on cached embeddings."""

    def __init__(self, client_id, embs, labels, val_embs, val_labels,
                 n_classes, device):
        self.id = client_id
        self.embs = embs
        self.labels = labels
        self.val_embs = val_embs
        self.val_labels = val_labels
        self.n_samples = len(embs)
        self.n_classes = n_classes
        self.device = device

    def local_train(self, head, local_epochs, lr, batch_size=64):
        head_local = copy.deepcopy(head).to(self.device)
        head_local.train()
        opt = torch.optim.SGD(head_local.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()

        ds = TensorDataset(self.embs, self.labels)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

        for _ in range(local_epochs):
            for z, y in loader:
                z, y = z.to(self.device), y.to(self.device)
                loss = loss_fn(head_local(z), y)
                opt.zero_grad()
                loss.backward()
                opt.step()

        n_params = sum(p.numel() for p in head_local.parameters())
        return head_local.state_dict(), n_params * 4

    @torch.no_grad()
    def evaluate_local(self, head):
        if self.val_embs.size(0) == 0:
            return 0.0
        head.eval()
        z = self.val_embs.to(self.device)
        y = self.val_labels.to(self.device)
        preds = head.to(self.device)(z).argmax(1)
        return (preds == y).float().mean().item()


# ====================================================================== #
#  Lightweight Client for Proposed Method                                 #
# ====================================================================== #
class ProposedClient:
    """Client for proposed method: privacy-gated embedding upload."""

    SKILL_FILE = "skills/client_agent.md"

    def __init__(self, client_id, embs, labels, val_embs, val_labels,
                 n_classes, embed_dim, device, cfg):
        self.id = client_id
        self.embs = embs  # precomputed clean embeddings
        self.labels = labels
        self.val_embs = val_embs
        self.val_labels = val_labels
        self.n_classes = n_classes
        self.embed_dim = embed_dim
        self.device = device
        self.cfg = cfg

        self.sigma = cfg.get("sigma", 0.02)
        self.clip_C = cfg.get("clip_C", 1.0)
        self.tau_high = cfg.get("tau_high", 0.95)
        self.upload_budget = cfg.get("upload_budget", 500)
        self.augmentation_mode = "normal"
        self.prev_val_accs = []

        # Label info
        self.label_counts = np.bincount(labels.numpy(),
                                         minlength=n_classes)

    def extract_gated_embeddings(self, n_views=2):
        """Extract privacy-gated embeddings from cached clean embeddings."""
        N = self.embs.size(0)
        labels_np = self.labels.numpy()

        # Compute per-class prototypes
        prototypes = torch.zeros(self.n_classes, self.embed_dim)
        proto_valid = torch.zeros(self.n_classes, dtype=torch.bool)
        embs_normed = self.embs  # already L2-normalized at precompute time

        for c in range(self.n_classes):
            mask = self.labels == c
            if mask.any():
                prototypes[c] = F.normalize(embs_normed[mask].mean(0), dim=0)
                proto_valid[c] = True

        all_z, all_y = [], []
        reject_count = 0
        total_count = 0

        for v in range(n_views):
            # View-specific perturbation (simulates augmentation diversity)
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

            # Privacy gate (vectorized where possible)
            for i in range(N):
                total_count += 1
                label = int(labels_np[i])
                zt = z_tilde[i]

                if proto_valid[label]:
                    sim = F.cosine_similarity(
                        zt.unsqueeze(0), prototypes[label].unsqueeze(0)
                    ).item()
                    if sim > self.tau_high:
                        reject_count += 1
                        continue

                all_z.append(zt)
                all_y.append(label)

                if len(all_z) >= self.upload_budget:
                    break
            if len(all_z) >= self.upload_budget:
                break

        reject_ratio = reject_count / max(total_count, 1)

        # Apply hooks
        self._apply_hooks(reject_ratio)

        # Fallback to prototypes
        if not all_z:
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
            "reject_ratio": reject_ratio,
            "sigma": self.sigma,
            "n_uploaded": len(all_z),
            "augmentation_mode": self.augmentation_mode,
        }
        return embeddings, labels_out, summary

    def _apply_hooks(self, reject_ratio):
        # low_data_hook
        k = self.cfg.get("low_data_k", 10)
        if (self.label_counts[self.label_counts > 0] < k).any():
            self.augmentation_mode = "conservative"

        # high_risk_hook
        r = self.cfg.get("high_risk_r", 0.30)
        if reject_ratio > r:
            self.sigma = min(self.sigma * 1.5, 0.5)
            self.upload_budget = max(self.upload_budget // 2, 50)

        # drift_hook
        if len(self.prev_val_accs) >= 2:
            if all(self.prev_val_accs[-i] < self.prev_val_accs[-i - 1]
                   for i in range(1, min(3, len(self.prev_val_accs)))):
                self.upload_budget = int(self.upload_budget * 1.3)

    @torch.no_grad()
    def evaluate_local(self, head):
        if self.val_embs.size(0) == 0:
            self.prev_val_accs.append(0.0)
            return 0.0
        head.eval()
        z = self.val_embs.to(self.device)
        y = self.val_labels.to(self.device)
        preds = head.to(self.device)(z).argmax(1)
        acc = (preds == y).float().mean().item()
        self.prev_val_accs.append(acc)
        return acc

    def apply_server_instructions(self, instructions):
        if not instructions:
            return
        self.upload_budget = instructions.get("upload_budget", self.upload_budget)
        self.sigma = instructions.get("sigma", self.sigma)
        self.augmentation_mode = instructions.get("augmentation_mode",
                                                   self.augmentation_mode)


# ====================================================================== #
#  Evaluator using cached embeddings                                      #
# ====================================================================== #
class FastEvaluator:
    """Evaluator that uses precomputed test embeddings."""

    SKILL_FILE = "skills/evaluator_agent.md"

    def __init__(self, test_embs, test_labels, n_classes, device, results_dir):
        self.test_embs = test_embs
        self.test_labels = test_labels
        self.n_classes = n_classes
        self.device = device
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        self.metrics = {}

    @torch.no_grad()
    def evaluate(self, head, method_name, round_num, comm_bytes,
                 cumulative_comm):
        from sklearn.metrics import f1_score
        head.eval()
        head_dev = head.to(self.device)

        all_preds = []
        bs = 1024
        for i in range(0, self.test_embs.size(0), bs):
            z = self.test_embs[i:i+bs].to(self.device)
            preds = head_dev(z).argmax(1).cpu()
            all_preds.append(preds)

        all_preds = torch.cat(all_preds).numpy()
        all_labels = self.test_labels.numpy()

        acc = (all_preds == all_labels).mean()
        macro_f1 = f1_score(all_labels, all_preds, average="macro",
                            zero_division=0)

        record = {
            "round": round_num,
            "accuracy": float(acc),
            "macro_f1": float(macro_f1),
            "comm_bytes": comm_bytes,
            "cumulative_comm_bytes": cumulative_comm,
        }

        if method_name not in self.metrics:
            self.metrics[method_name] = []
        self.metrics[method_name].append(record)
        return acc, macro_f1

    def save_csv(self, method_name):
        import csv
        records = self.metrics.get(method_name, [])
        if not records:
            return
        path = os.path.join(self.results_dir, f"{method_name}_rounds.csv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=records[0].keys())
            w.writeheader()
            w.writerows(records)

    def save_final_json(self):
        summary = {}
        for method, records in self.metrics.items():
            if records:
                best = max(records, key=lambda r: r["accuracy"])
                summary[method] = {
                    "best_accuracy": best["accuracy"],
                    "best_macro_f1": best["macro_f1"],
                    "best_round": best["round"],
                    "final_accuracy": records[-1]["accuracy"],
                    "final_macro_f1": records[-1]["macro_f1"],
                    "total_comm_bytes": records[-1]["cumulative_comm_bytes"],
                    "total_rounds": len(records),
                }
        path = os.path.join(self.results_dir, "final_summary.json")
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)

    def plot_comparisons(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if not self.metrics:
            return

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        for method, records in self.metrics.items():
            rounds = [r["round"] for r in records]
            accs = [r["accuracy"] * 100 for r in records]
            ax.plot(rounds, accs, label=method, linewidth=2)
        ax.set_xlabel("Communication Round", fontsize=13)
        ax.set_ylabel("Global Test Accuracy (%)", fontsize=13)
        ax.set_title("Accuracy vs. Communication Rounds", fontsize=14)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(self.results_dir, "acc_vs_rounds.png"),
                    dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        for method, records in self.metrics.items():
            comm_mb = [r["cumulative_comm_bytes"] / 1e6 for r in records]
            accs = [r["accuracy"] * 100 for r in records]
            ax.plot(comm_mb, accs, label=method, linewidth=2)
        ax.set_xlabel("Cumulative Communication (MB)", fontsize=13)
        ax.set_ylabel("Global Test Accuracy (%)", fontsize=13)
        ax.set_title("Communication Cost vs. Accuracy", fontsize=14)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(self.results_dir, "comm_vs_acc.png"),
                    dpi=150)
        plt.close(fig)


# ====================================================================== #
#  Run FedAvg                                                             #
# ====================================================================== #
def run_fedavg(args, clients, server, evaluator, logger):
    logger.info("=" * 60)
    logger.info("Starting FedAvg baseline")
    logger.info("=" * 60)

    cumulative_comm = 0
    for rnd in range(1, args.rounds + 1):
        t0 = time.time()
        round_comm = 0
        client_sds, client_sizes = [], []

        for c in clients:
            sd, comm = c.local_train(server.get_fedavg_head(),
                                      args.local_epochs, args.fedavg_lr,
                                      args.batch_size)
            client_sds.append(sd)
            client_sizes.append(c.n_samples)
            round_comm += comm

        server.fedavg_aggregate(client_sds, client_sizes)
        head_params = sum(p.numel() for p in server.get_fedavg_head().parameters())
        round_comm += head_params * 4 * args.n_clients
        cumulative_comm += round_comm

        acc, f1 = evaluator.evaluate(server.get_fedavg_head(), "FedAvg",
                                      rnd, round_comm, cumulative_comm)

        dt = time.time() - t0
        if rnd % 10 == 0 or rnd == 1:
            logger.info(f"[FedAvg] R{rnd:3d} | Acc:{acc:.4f} F1:{f1:.4f} "
                        f"Comm:{cumulative_comm/1e6:.1f}MB ({dt:.1f}s)")

    evaluator.save_csv("FedAvg")
    logger.info(f"[FedAvg] Final acc={acc:.4f}")


# ====================================================================== #
#  Run Proposed                                                           #
# ====================================================================== #
def run_proposed(args, clients, server, evaluator, logger):
    logger.info("=" * 60)
    logger.info("Starting Proposed: Agent-Orchestrated Fed Rep Sharing")
    logger.info("=" * 60)

    cumulative_comm = 0
    for rnd in range(1, args.rounds + 1):
        t0 = time.time()
        round_comm = 0
        all_embs, all_labs, summaries = [], [], []

        for c in clients:
            embs, labs, summary = c.extract_gated_embeddings(
                n_views=args.n_views)
            all_embs.append(embs)
            all_labs.append(labs)
            summaries.append(summary)
            round_comm += estimate_comm_bytes(embs)
            c.evaluate_local(server.get_head())

        # Summary comm cost (small)
        round_comm += args.n_clients * 4 * 105

        valid_e = [e for e in all_embs if e.numel() > 0]
        valid_l = [l for l in all_labs if l.numel() > 0]
        if valid_e:
            merged_embs = torch.cat(valid_e)
            merged_labs = torch.cat(valid_l)
        else:
            merged_embs = torch.zeros(0, server.embed_dim)
            merged_labs = torch.zeros(0, dtype=torch.long)

        instructions = server.orchestrate(summaries)
        for instr in instructions:
            clients[instr["client_id"]].apply_server_instructions(instr)

        broadcast_bytes = server.train_head(merged_embs, merged_labs,
                                             epochs=args.server_train_epochs)
        round_comm += broadcast_bytes * args.n_clients
        cumulative_comm += round_comm

        acc, f1 = evaluator.evaluate(server.get_head(), "Proposed",
                                      rnd, round_comm, cumulative_comm)

        dt = time.time() - t0
        if rnd % 10 == 0 or rnd == 1:
            total_up = sum(s["n_uploaded"] for s in summaries)
            avg_rej = np.mean([s["reject_ratio"] for s in summaries])
            logger.info(
                f"[Proposed] R{rnd:3d} | Acc:{acc:.4f} F1:{f1:.4f} "
                f"Comm:{cumulative_comm/1e6:.1f}MB Up:{total_up} "
                f"Rej:{avg_rej:.2f} ({dt:.1f}s)")

    evaluator.save_csv("Proposed")
    logger.info(f"[Proposed] Final acc={acc:.4f}")


# ====================================================================== #
#  Main                                                                   #
# ====================================================================== #
def main():
    args = parse_args()
    logger = setup_logging(args.results_dir)

    logger.info("Experiment Configuration:")
    for k, v in vars(args).items():
        logger.info(f"  {k}: {v}")

    skills = load_skill_files("skills")
    log_skills(skills, logger)

    device = "cuda" if (args.device == "auto" and
                        torch.cuda.is_available()) else "cpu"
    if args.device not in ("auto", "cpu", "cuda"):
        device = args.device
    logger.info(f"Device: {device}")

    set_seed(args.seed)
    n_classes = 100

    # Load data
    logger.info("Loading CIFAR-100...")
    train_ds = torchvision.datasets.CIFAR100(
        root="./data", train=True, download=True, transform=None)
    test_ds = torchvision.datasets.CIFAR100(
        root="./data", train=False, download=True, transform=None)
    train_labels = np.array(train_ds.targets)

    # Partition
    logger.info(f"Partitioning: {args.n_clients} clients, alpha={args.alpha}")
    client_indices = dirichlet_partition(train_labels, args.n_clients,
                                          args.alpha, args.seed)
    for i, idx in enumerate(client_indices):
        nc = len(set(train_labels[idx]))
        logger.info(f"  Client {i}: {len(idx)} samples, {nc} classes")

    # Build encoder
    logger.info("Building frozen ResNet-18 encoder...")
    encoder, embed_dim = build_encoder(device)

    # PRECOMPUTE ALL EMBEDDINGS ONCE
    logger.info("Precomputing train embeddings (one-time cost)...")
    t_start = time.time()
    all_train_indices = np.arange(len(train_ds))
    train_embs, train_labs = precompute_embeddings(
        encoder, train_ds, all_train_indices, device, batch_size=256)
    logger.info(f"  Train embeddings: {train_embs.shape} "
                f"({time.time()-t_start:.1f}s)")

    logger.info("Precomputing test embeddings...")
    t_start = time.time()
    test_indices = np.arange(len(test_ds))
    test_embs, test_labs = precompute_embeddings(
        encoder, test_ds, test_indices, device, batch_size=256)
    logger.info(f"  Test embeddings: {test_embs.shape} "
                f"({time.time()-t_start:.1f}s)")

    # Build evaluator with cached test embeddings
    evaluator = FastEvaluator(test_embs, test_labs, n_classes, device,
                               args.results_dir)

    cfg = {k: v for k, v in vars(args).items()}

    # Run FedAvg
    if "fedavg" in args.methods:
        set_seed(args.seed)
        server_fa = ServerAgent(embed_dim, n_classes, args.n_clients,
                                 device, cfg)
        clients_fa = []
        for i in range(args.n_clients):
            tr_idx, va_idx = split_train_val(client_indices[i],
                                              args.val_ratio, args.seed)
            c = FedAvgClient(i, train_embs[tr_idx], train_labs[tr_idx],
                              train_embs[va_idx], train_labs[va_idx],
                              n_classes, device)
            clients_fa.append(c)
        run_fedavg(args, clients_fa, server_fa, evaluator, logger)

    # Run Proposed
    if "proposed" in args.methods:
        set_seed(args.seed)
        server_pr = ServerAgent(embed_dim, n_classes, args.n_clients,
                                 device, cfg)
        clients_pr = []
        for i in range(args.n_clients):
            tr_idx, va_idx = split_train_val(client_indices[i],
                                              args.val_ratio, args.seed)
            c = ProposedClient(
                i, train_embs[tr_idx], train_labs[tr_idx],
                train_embs[va_idx], train_labs[va_idx],
                n_classes, embed_dim, device, cfg)
            clients_pr.append(c)
        run_proposed(args, clients_pr, server_pr, evaluator, logger)

    # Final outputs
    evaluator.save_final_json()
    evaluator.plot_comparisons()

    summary_path = os.path.join(args.results_dir, "final_summary.json")
    with open(summary_path) as f:
        summary = json.load(f)
    logger.info("=" * 60)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 60)
    for method, stats in summary.items():
        logger.info(f"  {method}:")
        for k, v in stats.items():
            logger.info(f"    {k}: {v:.4f}" if isinstance(v, float)
                        else f"    {k}: {v}")

    logger.info("Experiment complete.")


if __name__ == "__main__":
    main()
