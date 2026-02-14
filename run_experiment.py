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
from a2a import A2ABus, AgentCard, Part, Artifact


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
    p.add_argument("--tau_percentile", type=float, default=0.15,
                   help="Reject top tau_percentile fraction most similar to prototype")
    p.add_argument("--tau_min", type=float, default=0.5,
                   help="Minimum cosine similarity threshold (floor for adaptive gate)")
    p.add_argument("--upload_budget", type=int, default=500)
    p.add_argument("--n_views", type=int, default=2)
    p.add_argument("--low_data_k", type=int, default=10)
    p.add_argument("--high_risk_r", type=float, default=0.30)
    p.add_argument("--replay_decay", type=float, default=0.995,
                   help="Exponential decay factor for replay buffer sample weights")
    p.add_argument("--replay_min_weight", type=float, default=0.3,
                   help="Minimum weight for oldest replay buffer samples")
    p.add_argument("--server_lr_decay", type=float, default=0.98,
                   help="Per-round multiplicative LR decay for server head training")
    p.add_argument("--server_lr_min", type=float, default=1e-4,
                   help="Minimum LR floor for server head training")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--results_dir", default="results")
    p.add_argument("--device", default="auto")
    p.add_argument("--centralized_epochs", type=int, default=50,
                   help="Total training epochs for centralized baseline")
    p.add_argument("--methods", nargs="+",
                   default=["fedavg", "ao-frl", "centralized"])
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
        """Extract privacy-gated embeddings from cached clean embeddings.
        Uses adaptive percentile-based privacy gate: for each class, reject
        the top tau_percentile fraction of samples most similar to the prototype.
        """
        N = self.embs.size(0)
        labels_np = self.labels.numpy()
        tau_percentile = self.cfg.get("tau_percentile", 0.15)
        tau_min = self.cfg.get("tau_min", 0.5)

        # Compute per-class prototypes
        prototypes = torch.zeros(self.n_classes, self.embed_dim)
        proto_valid = torch.zeros(self.n_classes, dtype=torch.bool)
        embs_normed = self.embs  # already L2-normalized at precompute time

        for c in range(self.n_classes):
            mask = self.labels == c
            if mask.any():
                prototypes[c] = F.normalize(embs_normed[mask].mean(0), dim=0)
                proto_valid[c] = True

        # Phase 1: Generate all noised embeddings and compute similarities
        all_candidates = []  # list of (z_tilde, label, similarity)
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

            # Compute per-sample similarity to class prototype
            for i in range(N):
                label = int(labels_np[i])
                zt = z_tilde[i]
                if proto_valid[label]:
                    sim = F.cosine_similarity(
                        zt.unsqueeze(0), prototypes[label].unsqueeze(0)
                    ).item()
                else:
                    sim = 0.0  # no prototype → always accept
                all_candidates.append((zt, label, sim))

        # Phase 2: Adaptive percentile gate per class
        # Group similarities by class to find per-class threshold
        from collections import defaultdict
        class_sims = defaultdict(list)
        for idx, (_, label, sim) in enumerate(all_candidates):
            class_sims[label].append((idx, sim))

        # Determine per-class thresholds
        accepted_indices = set()
        reject_count = 0
        total_count = len(all_candidates)

        for c, sim_list in class_sims.items():
            if not proto_valid[c]:
                # No prototype → accept all
                for idx, _ in sim_list:
                    accepted_indices.add(idx)
                continue

            sims = np.array([s for _, s in sim_list])
            # Threshold = (1 - tau_percentile) quantile, but no lower than tau_min
            threshold = max(np.percentile(sims, (1 - tau_percentile) * 100),
                            tau_min)

            for idx, sim in sim_list:
                if sim > threshold:
                    reject_count += 1
                else:
                    accepted_indices.add(idx)

        reject_ratio = reject_count / max(total_count, 1)

        # Collect accepted embeddings up to upload budget
        all_z, all_y = [], []
        for idx in sorted(accepted_indices):
            zt, label, _ = all_candidates[idx]
            all_z.append(zt)
            all_y.append(label)
            if len(all_z) >= self.upload_budget:
                break

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

        method_styles = {
            "Centralized": {"color": "tab:green", "linestyle": "--"},
            "FedAvg": {"color": "tab:blue", "linestyle": "-"},
            "AO-FRL": {"color": "tab:red", "linestyle": "-"},
        }

        def _style(method):
            return method_styles.get(method, {"color": None, "linestyle": "-"})

        # --- Plot 1: Accuracy vs Rounds ---
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        for method, records in self.metrics.items():
            rounds = [r["round"] for r in records]
            accs = [r["accuracy"] * 100 for r in records]
            s = _style(method)
            ax.plot(rounds, accs, label=method, linewidth=2, **s)
        ax.set_xlabel("Communication Round", fontsize=13)
        ax.set_ylabel("Global Test Accuracy (%)", fontsize=13)
        ax.set_title("Accuracy vs. Communication Rounds", fontsize=14)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(self.results_dir, "acc_vs_rounds.png"),
                    dpi=150)
        plt.close(fig)

        # --- Plot 2: Macro-F1 vs Rounds ---
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        for method, records in self.metrics.items():
            rounds = [r["round"] for r in records]
            f1s = [r["macro_f1"] * 100 for r in records]
            s = _style(method)
            ax.plot(rounds, f1s, label=method, linewidth=2, **s)
        ax.set_xlabel("Communication Round", fontsize=13)
        ax.set_ylabel("Macro-F1 (%)", fontsize=13)
        ax.set_title("Macro-F1 vs. Communication Rounds", fontsize=14)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(self.results_dir, "f1_vs_rounds.png"),
                    dpi=150)
        plt.close(fig)

        # --- Plot 3: Cumulative Communication Cost vs Rounds ---
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        for method, records in self.metrics.items():
            rounds = [r["round"] for r in records]
            comm_mb = [r["cumulative_comm_bytes"] / 1e6 for r in records]
            s = _style(method)
            ax.plot(rounds, comm_mb, label=method, linewidth=2, **s)
        ax.set_xlabel("Communication Round", fontsize=13)
        ax.set_ylabel("Cumulative Communication (MB)", fontsize=13)
        ax.set_title("Communication Cost vs. Rounds", fontsize=14)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(self.results_dir, "comm_vs_rounds.png"),
                    dpi=150)
        plt.close(fig)

        # --- Plot 4: Communication Cost vs Accuracy ---
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        for method, records in self.metrics.items():
            comm_mb = [r["cumulative_comm_bytes"] / 1e6 for r in records]
            accs = [r["accuracy"] * 100 for r in records]
            s = _style(method)
            ax.plot(comm_mb, accs, label=method, linewidth=2, **s)
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
#  Run Centralized (upper-bound baseline)                                 #
# ====================================================================== #
def run_centralized(args, train_embs, train_labs, evaluator, embed_dim,
                    n_classes, device, logger, bus=None):
    """Train a single MLPHead on ALL training data — no federation, no privacy.
    This is the theoretical upper bound for the frozen-encoder setup.
    We report results per-epoch so the curve is comparable to per-round plots.
    """
    logger.info("=" * 60)
    logger.info("Starting Centralized baseline (upper bound)")
    logger.info("=" * 60)

    head = MLPHead(embed_dim, n_classes,
                   hidden=args.head_hidden).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=args.server_lr)
    loss_fn = nn.CrossEntropyLoss()

    ds = TensorDataset(train_embs, train_labs)
    loader = DataLoader(ds, batch_size=256, shuffle=True)

    total_epochs = args.centralized_epochs
    # Map epochs to "rounds" so the plot x-axis is comparable
    # If rounds=100 and centralized_epochs=50, each epoch maps to 2 rounds
    rounds_per_epoch = max(1, args.rounds / total_epochs)

    for epoch in range(1, total_epochs + 1):
        t0 = time.time()
        head.train()
        for z_batch, y_batch in loader:
            z_batch = z_batch.to(device)
            y_batch = y_batch.to(device)
            loss = loss_fn(head(z_batch), y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Centralized has zero communication cost
        mapped_round = int(epoch * rounds_per_epoch)

        if bus:
            t = bus.send_task("server", "evaluator", "evaluate",
                              [Part(type="json", content={"method": "Centralized",
                                                          "round": mapped_round})])
        acc, f1 = evaluator.evaluate(head, "Centralized", mapped_round,
                                      0, 0)
        if bus:
            bus.complete_task(t.task_id,
                              response_parts=[Part(type="json",
                                                   content={"acc": acc, "f1": f1})])

        dt = time.time() - t0
        if epoch % 10 == 0 or epoch == 1:
            logger.info(f"[Centralized] Epoch {epoch:3d} (→R{mapped_round}) | "
                        f"Acc:{acc:.4f} F1:{f1:.4f} ({dt:.1f}s)")

    evaluator.save_csv("Centralized")
    logger.info(f"[Centralized] Final acc={acc:.4f}")


# ====================================================================== #
#  Run FedAvg                                                             #
# ====================================================================== #
def run_fedavg(args, clients, server, evaluator, logger, bus=None):
    logger.info("=" * 60)
    logger.info("Starting FedAvg baseline")
    logger.info("=" * 60)

    cumulative_comm = 0
    for rnd in range(1, args.rounds + 1):
        t0 = time.time()
        round_comm = 0
        client_sds, client_sizes = [], []

        for c in clients:
            # A2A: Server → Client local_train task
            if bus:
                t = bus.send_task("server", f"client_{c.id}", "local_train",
                                  [Part(type="json", content={"round": rnd,
                                                              "local_epochs": args.local_epochs})])
            sd, comm = c.local_train(server.get_fedavg_head(),
                                      args.local_epochs, args.fedavg_lr,
                                      args.batch_size)
            if bus:
                bus.complete_task(t.task_id,
                                  artifacts=[Artifact(artifact_id=f"sd_r{rnd}_c{c.id}",
                                                      name="state_dict",
                                                      data=None,
                                                      size_bytes=comm)])
            client_sds.append(sd)
            client_sizes.append(c.n_samples)
            round_comm += comm

        server.fedavg_aggregate(client_sds, client_sizes)
        head_params = sum(p.numel() for p in server.get_fedavg_head().parameters())
        round_comm += head_params * 4 * args.n_clients
        cumulative_comm += round_comm

        # A2A: Server → Evaluator evaluate task
        if bus:
            t = bus.send_task("server", "evaluator", "evaluate",
                              [Part(type="json", content={"method": "FedAvg",
                                                          "round": rnd})])
        acc, f1 = evaluator.evaluate(server.get_fedavg_head(), "FedAvg",
                                      rnd, round_comm, cumulative_comm)
        if bus:
            bus.complete_task(t.task_id,
                              response_parts=[Part(type="json",
                                                   content={"acc": acc, "f1": f1})])

        dt = time.time() - t0
        if rnd % 10 == 0 or rnd == 1:
            logger.info(f"[FedAvg] R{rnd:3d} | Acc:{acc:.4f} F1:{f1:.4f} "
                        f"Comm:{cumulative_comm/1e6:.1f}MB ({dt:.1f}s)")

    evaluator.save_csv("FedAvg")
    logger.info(f"[FedAvg] Final acc={acc:.4f}")


# ====================================================================== #
#  Run Proposed                                                           #
# ====================================================================== #
def run_proposed(args, clients, server, evaluator, logger, bus=None):
    logger.info("=" * 60)
    logger.info("Starting AO-FRL: Agent-Orchestrated Fed Rep Sharing")
    logger.info("=" * 60)

    cumulative_comm = 0
    instruction_history = []  # track server→client instructions per round

    for rnd in range(1, args.rounds + 1):
        t0 = time.time()
        round_comm = 0
        all_embs, all_labs, summaries = [], [], []

        for c in clients:
            # A2A: Server → Client extract_embeddings task
            if bus:
                t = bus.send_task("server", f"client_{c.id}",
                                  "extract_embeddings",
                                  [Part(type="json",
                                        content={"round": rnd,
                                                 "n_views": args.n_views})])
            embs, labs, summary = c.extract_gated_embeddings(
                n_views=args.n_views)
            if bus:
                bus.complete_task(t.task_id,
                                  artifacts=[Artifact(
                                      artifact_id=f"emb_r{rnd}_c{c.id}",
                                      name="gated_embeddings",
                                      data=None,
                                      size_bytes=estimate_comm_bytes(embs))],
                                  response_parts=[Part(type="json",
                                                       content=summary)])
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
            cid = instr["client_id"]
            # A2A: Server → Client apply_instructions task
            if bus:
                t = bus.send_task("server", f"client_{cid}",
                                  "apply_instructions",
                                  [Part(type="json", content=instr)])
            clients[cid].apply_server_instructions(instr)
            if bus:
                bus.complete_task(t.task_id)

        # Record per-round instruction stats
        budgets = [instr["upload_budget"] for instr in instructions]
        sigmas = [instr["sigma"] for instr in instructions]
        n_conservative = sum(1 for instr in instructions
                             if instr["augmentation_mode"] == "conservative")
        total_up = sum(s["n_uploaded"] for s in summaries)
        avg_rej = np.mean([s["reject_ratio"] for s in summaries])
        instruction_history.append({
            "round": rnd,
            "avg_budget": np.mean(budgets),
            "min_budget": int(np.min(budgets)),
            "max_budget": int(np.max(budgets)),
            "avg_sigma": np.mean(sigmas),
            "max_sigma": float(np.max(sigmas)),
            "n_conservative": n_conservative,
            "n_normal": args.n_clients - n_conservative,
            "total_uploaded": total_up,
            "avg_reject_ratio": avg_rej,
        })

        broadcast_bytes = server.train_head(merged_embs, merged_labs,
                                             epochs=args.server_train_epochs)
        round_comm += broadcast_bytes * args.n_clients
        cumulative_comm += round_comm

        # A2A: Server → Evaluator evaluate task
        if bus:
            t = bus.send_task("server", "evaluator", "evaluate",
                              [Part(type="json", content={"method": "AO-FRL",
                                                          "round": rnd})])
        acc, f1 = evaluator.evaluate(server.get_head(), "AO-FRL",
                                      rnd, round_comm, cumulative_comm)
        if bus:
            bus.complete_task(t.task_id,
                              response_parts=[Part(type="json",
                                                   content={"acc": acc, "f1": f1})])

        dt = time.time() - t0
        if rnd % 10 == 0 or rnd == 1:
            logger.info(
                f"[AO-FRL] R{rnd:3d} | Acc:{acc:.4f} F1:{f1:.4f} "
                f"Comm:{cumulative_comm/1e6:.1f}MB Up:{total_up} "
                f"Rej:{avg_rej:.2f} | AvgBudget:{np.mean(budgets):.0f} "
                f"AvgSigma:{np.mean(sigmas):.4f} "
                f"Conservative:{n_conservative}/{args.n_clients} ({dt:.1f}s)")

    evaluator.save_csv("AO-FRL")
    logger.info(f"[AO-FRL] Final acc={acc:.4f}")

    # Save instruction history CSV
    save_instruction_history(instruction_history, args.results_dir)
    # Plot instruction trends
    plot_instruction_trends(instruction_history, args.results_dir,
                            args.n_clients)


def save_instruction_history(history, results_dir):
    """Save per-round server instruction stats to CSV."""
    import csv
    if not history:
        return
    path = os.path.join(results_dir, "server_instructions.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=history[0].keys())
        w.writeheader()
        w.writerows(history)


def plot_instruction_trends(history, results_dir, n_clients):
    """Plot how server instructions to clients change over rounds."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not history:
        return

    rounds = [h["round"] for h in history]
    avg_budgets = [h["avg_budget"] for h in history]
    min_budgets = [h["min_budget"] for h in history]
    max_budgets = [h["max_budget"] for h in history]
    avg_sigmas = [h["avg_sigma"] for h in history]
    max_sigmas = [h["max_sigma"] for h in history]
    n_conservative = [h["n_conservative"] for h in history]
    total_uploaded = [h["total_uploaded"] for h in history]
    avg_reject = [h["avg_reject_ratio"] for h in history]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (0,0) Upload budget over rounds
    ax = axes[0, 0]
    ax.plot(rounds, avg_budgets, label="Avg budget", linewidth=2, color="tab:blue")
    ax.fill_between(rounds, min_budgets, max_budgets, alpha=0.2, color="tab:blue",
                    label="Min–Max range")
    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Upload Budget (embeddings)")
    ax.set_title("Server-assigned Upload Budget per Client")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (0,1) Sigma (noise scale) over rounds
    ax = axes[0, 1]
    ax.plot(rounds, avg_sigmas, label="Avg sigma", linewidth=2, color="tab:orange")
    ax.plot(rounds, max_sigmas, label="Max sigma", linewidth=1.5,
            linestyle="--", color="tab:red")
    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Noise Scale (sigma)")
    ax.set_title("Server-assigned DP Noise Scale")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (1,0) Augmentation mode distribution
    ax = axes[1, 0]
    n_normal = [n_clients - nc for nc in n_conservative]
    ax.stackplot(rounds, n_conservative, n_normal,
                 labels=["Conservative", "Normal"],
                 colors=["tab:red", "tab:green"], alpha=0.7)
    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Number of Clients")
    ax.set_title("Augmentation Mode Distribution")
    ax.legend(loc="center right")
    ax.set_ylim(0, n_clients)
    ax.grid(True, alpha=0.3)

    # (1,1) Actual uploads & rejection ratio
    ax = axes[1, 1]
    ax.plot(rounds, total_uploaded, label="Total uploaded", linewidth=2,
            color="tab:blue")
    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Total Uploaded Embeddings", color="tab:blue")
    ax.tick_params(axis='y', labelcolor="tab:blue")
    ax.grid(True, alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(rounds, avg_reject, label="Avg reject ratio", linewidth=2,
             color="tab:red", linestyle="--")
    ax2.set_ylabel("Avg Rejection Ratio", color="tab:red")
    ax2.tick_params(axis='y', labelcolor="tab:red")
    ax2.set_ylim(-0.05, 1.05)
    ax.set_title("Client Upload Volume & Privacy Gate Rejection")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="center right")

    fig.suptitle("Server Orchestration — Per-Round Instruction Trends",
                 fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "server_instructions.png"), dpi=150)
    plt.close(fig)


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

    # Initialize A2A bus
    bus = A2ABus()
    bus.register_agent(AgentCard("server", "ServerAgent",
                                 "Orchestration & head training",
                                 ["orchestrate", "train_head", "fedavg_aggregate"]))
    bus.register_agent(AgentCard("evaluator", "EvaluatorAgent",
                                 "Global model evaluation",
                                 ["evaluate"]))
    for i in range(args.n_clients):
        bus.register_agent(AgentCard(f"client_{i}", f"ClientAgent_{i}",
                                     f"Client {i} — local training & embedding extraction",
                                     ["local_train", "extract_embeddings",
                                      "apply_instructions"]))
    logger.info(f"A2A bus initialized: {len(bus._agents)} agents registered")

    # Run Centralized (upper bound)
    if "centralized" in args.methods:
        set_seed(args.seed)
        run_centralized(args, train_embs, train_labs, evaluator,
                        embed_dim, n_classes, device, logger, bus=bus)

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
        run_fedavg(args, clients_fa, server_fa, evaluator, logger, bus=bus)

    # Run AO-FRL
    if "ao-frl" in args.methods:
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
        run_proposed(args, clients_pr, server_pr, evaluator, logger, bus=bus)

    # Final outputs
    evaluator.save_final_json()
    evaluator.plot_comparisons()

    # Save A2A audit log
    bus.save_log(os.path.join(args.results_dir, "a2a_communication.json"))
    logger.info(f"A2A audit log saved: {bus.summary()['total_tasks']} tasks recorded")

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
