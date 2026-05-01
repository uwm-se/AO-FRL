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
                   split_decoder_pool, gaussian_dp_sigma,
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
    # AO-FRL DP knobs (sigma is derived from epsilon/delta unless explicitly set)
    p.add_argument("--epsilon", type=float, default=2.0,
                   help="Target DP epsilon for the Gaussian mechanism. Used to "
                        "derive sigma when --sigma is not given.")
    p.add_argument("--delta", type=float, default=1e-5)
    p.add_argument("--sigma", type=float, default=None,
                   help="If set, overrides the sigma derived from epsilon/delta.")
    p.add_argument("--clip_C", type=float, default=1.0)
    p.add_argument("--per_class_target", type=int, default=500,
                   help="T_base — global per-class total upload target per round.")
    p.add_argument("--head_sync_every", type=int, default=5,
                   help="Sync head down to clients every N rounds for feedback.")
    p.add_argument("--feedback_alpha", type=float, default=1.0,
                   help="Weight for boosting low-acc classes in T_c update.")
    p.add_argument("--encoder_weights", default="models/encoder_finetuned.pt",
                   help="Path to fine-tuned encoder weights. Falls back to "
                        "ImageNet-pretrained if file missing.")
    p.add_argument("--decoder_weights", default="models/decoder.pt",
                   help="Path to decoder weights for final-round inversion eval.")
    p.add_argument("--inversion_n_samples", type=int, default=200,
                   help="Test images used for final-round PSNR evaluation.")
    # Early stopping (AO-FRL only — head plateau / decline is common after
    # the early peak, so we stop when no improvement for `patience` rounds.)
    p.add_argument("--early_stop_patience", type=int, default=10,
                   help="Stop AO-FRL when test acc has not improved for this "
                        "many rounds. 0 disables early stopping.")
    # Legacy gate args (no longer used on AO-FRL path; retained for ablations)
    p.add_argument("--tau_high", type=float, default=0.95)
    p.add_argument("--tau_percentile", type=float, default=0.15,
                   help="LEGACY: cosine gate (no longer used on AO-FRL path)")
    p.add_argument("--tau_min", type=float, default=0.5,
                   help="LEGACY: cosine gate floor (no longer used on AO-FRL path)")
    p.add_argument("--upload_budget", type=int, default=500,
                   help="LEGACY: replaced by --per_class_target on AO-FRL path")
    p.add_argument("--n_views", type=int, default=2,
                   help="LEGACY: replaced by per-class budget sampling")
    p.add_argument("--low_data_k", type=int, default=10,
                   help="Threshold for low_data_hook (re-enabled when --legacy_hooks).")
    p.add_argument("--high_risk_r", type=float, default=0.30,
                   help="LEGACY: hook removed on AO-FRL path")
    p.add_argument("--dataset", default="cifar100",
                   choices=["cifar100", "cifar10", "svhn"],
                   help="Dataset to run on. Default: cifar100.")
    p.add_argument("--clients_per_round", type=int, default=-1,
                   help="If > 0, only this many randomly-selected clients "
                        "upload each round (partial participation). Defaults "
                        "to -1 = all n_clients participate.")
    p.add_argument("--random_upload_fraction", type=float, default=None,
                   help="If set (e.g. 1.0), each client samples this fraction "
                        "of its own data UNIFORMLY AT RANDOM each round, "
                        "ignoring server per-class budget allocation. Used to "
                        "ablate the impact of any budget management at all "
                        "vs server-coordinated upload schedules.")
    p.add_argument("--legacy_hooks", action="store_true",
                   help="Re-enable client-side low_data + drift hooks. "
                        "When set, each ProposedClient (a) forces full upload "
                        "of any class with 0 < hist[c] < low_data_k and "
                        "(b) boosts all per-class budgets by 1.3× when its "
                        "overall val acc has dropped for two consecutive sync "
                        "rounds.")
    p.add_argument("--replay_max", type=int, default=500_000,
                   help="Server replay buffer cap. Should be >= n_classes * "
                        "T_base * 5 so cross-round noise averaging works "
                        "(otherwise per-round uploads overwrite the buffer).")
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
    # FedProx / FedAdam baselines
    p.add_argument("--fedprox_mu", type=float, default=0.01,
                   help="Proximal term coefficient μ in FedProx loss.")
    p.add_argument("--fedadam_lr", type=float, default=1e-2,
                   help="Server-side Adam learning rate for FedAdam.")
    p.add_argument("--fedadam_beta1", type=float, default=0.9)
    p.add_argument("--fedadam_beta2", type=float, default=0.99)
    p.add_argument("--fedadam_tau", type=float, default=1e-3,
                   help="Adaptivity / numerical stability term in FedAdam.")
    p.add_argument("--methods", nargs="+",
                   default=["fedavg", "ao-frl", "centralized"],
                   choices=["centralized", "fedavg", "fedprox", "fedadam",
                            "ao-frl"])
    return p.parse_args()


def build_encoder(device, weights_path: str = None):
    """Build a frozen ResNet-18 encoder (penultimate features, dim=512).

    If weights_path is given and exists, load fine-tuned weights from
    train_autoencoder.py instead of the ImageNet-pretrained baseline.
    The encoder is always frozen for federated experiments.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    encoder = nn.Sequential(*list(model.children())[:-1], nn.Flatten())

    if weights_path and os.path.exists(weights_path):
        sd = torch.load(weights_path, map_location="cpu")
        encoder.load_state_dict(sd)

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
#  FedProx Client: FedAvg + proximal regularization toward global head    #
# ====================================================================== #
class FedProxClient(FedAvgClient):
    """FedProx (Li et al. 2020): adds μ/2 ||w − w_global||² to local loss
    to limit drift on heterogeneous clients."""

    def local_train(self, head, local_epochs, lr, batch_size=64, mu=0.01):
        head_local = copy.deepcopy(head).to(self.device)
        # Snapshot global params for the proximal term
        global_params = [p.detach().clone().to(self.device)
                         for p in head.parameters()]
        head_local.train()
        opt = torch.optim.SGD(head_local.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()

        ds = TensorDataset(self.embs, self.labels)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

        for _ in range(local_epochs):
            for z, y in loader:
                z, y = z.to(self.device), y.to(self.device)
                logits = head_local(z)
                ce = loss_fn(logits, y)
                prox = sum(((p - g) ** 2).sum()
                           for p, g in zip(head_local.parameters(),
                                            global_params))
                loss = ce + 0.5 * mu * prox
                opt.zero_grad()
                loss.backward()
                opt.step()

        n_params = sum(p.numel() for p in head_local.parameters())
        return head_local.state_dict(), n_params * 4


# ====================================================================== #
#  Lightweight Client for Proposed Method                                 #
# ====================================================================== #
class ProposedClient:
    """AO-FRL client: image -> encoder -> embedding -> clip -> DP noise -> upload.

    Uses precomputed clean embeddings (encoder is frozen across federated
    rounds, so a one-shot precompute is equivalent to per-round encoder
    forward passes — but cheaper). Each round, the client applies fresh
    Gaussian DP noise and uploads up to per-class budget assigned by server.
    """

    SKILL_FILE = "skills/client_agent.md"

    def __init__(self, client_id, embs, labels, n_classes, embed_dim,
                 device, cfg):
        self.id = client_id
        self.embs = embs              # precomputed clean embeddings (N, D)
        self.labels = labels          # (N,)
        self.n_classes = n_classes
        self.embed_dim = embed_dim
        self.device = device
        self.cfg = cfg

        self.sigma = float(cfg.get("sigma", 0.1))
        self.clip_C = float(cfg.get("clip_C", 1.0))

        self.label_hist = np.bincount(labels.numpy(),
                                      minlength=n_classes).astype(np.int64)
        self._class_indices = [
            np.where(labels.numpy() == c)[0] for c in range(n_classes)
        ]

        # Per-class budget (assigned by server). Default = full availability.
        self.budget = self.label_hist.copy()

        # Legacy hook state (low_data + drift). Activated by cfg["legacy_hooks"].
        self.legacy_hooks = bool(cfg.get("legacy_hooks", False))
        self.low_data_k = int(cfg.get("low_data_k", 10))
        self.val_acc_history = []  # per-client overall val acc, appended each sync

    @torch.no_grad()
    def extract_dp_embeddings(self, rng: np.random.Generator = None):
        """Sample per-class up to budget (or uniform random if configured),
        apply L2 clip + Gaussian DP noise.

        If cfg["random_upload_fraction"] is set, the per-class budget is
        ignored and the client uploads `fraction × |own_data|` samples drawn
        uniformly at random from its own pool — preserving the client's
        non-IID class distribution at the server.

        Returns (embeddings, labels, summary).
        """
        rng = rng or np.random.default_rng()

        sel_z, sel_y = [], []
        per_class_uploaded = np.zeros(self.n_classes, dtype=np.int64)

        random_frac = self.cfg.get("random_upload_fraction")
        if random_frac is not None:
            # ---- No-budget mode: uniform random sampling, ignores per-class B ----
            # When fraction > 1.0, sample WITH replacement (each draw still gets
            # independent fresh DP noise — valid under per-release DP).
            n_total = self.embs.size(0)
            n_take = max(0, int(round(random_frac * n_total)))
            if n_take > 0:
                replace = (n_take > n_total)
                picks = rng.choice(n_total, size=n_take, replace=replace)
                z = self.embs[picks]
                y = self.labels[picks]
                norms = z.norm(dim=1, keepdim=True).clamp(min=1e-12)
                scale = torch.clamp(self.clip_C / norms, max=1.0)
                z = z * scale
                z = z + torch.randn_like(z) * self.sigma
                sel_z.append(z)
                sel_y.append(y)
                # Track uploaded count per class for logging
                for c_id in range(self.n_classes):
                    per_class_uploaded[c_id] = int((y == c_id).sum())
        else:
            # ---- Standard budget mode: per-class sampling ----
            for c in range(self.n_classes):
                avail = self._class_indices[c]
                n_take = min(int(self.budget[c]), len(avail))
                if n_take <= 0:
                    continue
                picks = rng.choice(avail, size=n_take, replace=False)
                z = self.embs[picks]                     # (n_take, D)

                # L2 clipping (sensitivity = clip_C)
                norms = z.norm(dim=1, keepdim=True).clamp(min=1e-12)
                scale = torch.clamp(self.clip_C / norms, max=1.0)
                z = z * scale

                # Gaussian DP noise
                z = z + torch.randn_like(z) * self.sigma

                sel_z.append(z)
                sel_y.append(torch.full((n_take,), c, dtype=torch.long))
                per_class_uploaded[c] = n_take

        if sel_z:
            embeddings = torch.cat(sel_z)
            labels_out = torch.cat(sel_y)
        else:
            embeddings = torch.zeros(0, self.embed_dim)
            labels_out = torch.zeros(0, dtype=torch.long)

        # Logging hook (no parameter mutation): per-class candidate / upload
        # counts, DP params, and observed clip rate.
        candidates_per_class = self.label_hist.tolist()
        summary = {
            "client_id": self.id,
            "label_histogram": candidates_per_class,         # what client has
            "uploaded_histogram": per_class_uploaded.tolist(),# what was sent
            "n_uploaded": int(per_class_uploaded.sum()),
            "sigma": self.sigma,
            "clip_C": self.clip_C,
            "epsilon": self.cfg.get("epsilon"),
            "delta": self.cfg.get("delta"),
        }
        return embeddings, labels_out, summary

    @torch.no_grad()
    def evaluate_per_class_on_train(self, head):
        """Per-class accuracy on the client's CLEAN train embeddings.

        Used as feedback signal to server (returns 100 floats). Safe: only
        scalar accuracies leave the client, not the embeddings themselves.
        """
        if self.embs.size(0) == 0:
            return np.zeros(self.n_classes), np.zeros(self.n_classes,
                                                       dtype=np.int64)

        head.eval()
        head_dev = head.to(self.device)
        z = self.embs.to(self.device)
        y = self.labels.to(self.device)
        preds = head_dev(z).argmax(1).cpu().numpy()
        y_np = self.labels.numpy()

        per_class_correct = np.zeros(self.n_classes, dtype=np.int64)
        per_class_total = self.label_hist.copy()
        for c in range(self.n_classes):
            if per_class_total[c] == 0:
                continue
            mask = y_np == c
            per_class_correct[c] = int((preds[mask] == c).sum())

        per_class_acc = np.divide(
            per_class_correct, per_class_total,
            out=np.zeros(self.n_classes, dtype=np.float64),
            where=per_class_total > 0)
        return per_class_acc, per_class_total

    def apply_budget(self, budget_vec: np.ndarray):
        """Server -> client: per-class upload budget for this client."""
        budget_vec = np.asarray(budget_vec, dtype=np.int64)
        # Cap by what the client actually has — server should already ensure
        # this, but be defensive.
        self.budget = np.minimum(budget_vec, self.label_hist)

    def apply_legacy_hooks(self, current_overall_val_acc=None):
        """Re-enabled legacy client-side hooks: low_data + drift only.

        - low_data_hook: for any class with 0 < hist[c] < low_data_k, set
          budget[c] := hist[c] so all available samples are uploaded.
        - drift_hook: if this client's overall val acc has dropped for two
          consecutive sync rounds, multiply all per-class budgets by 1.3
          (capped by histogram). Requires at least 3 history points.
        """
        if not self.legacy_hooks:
            return
        # low_data_hook
        low_classes = (self.label_hist > 0) & (self.label_hist < self.low_data_k)
        if low_classes.any():
            self.budget = np.where(low_classes, self.label_hist, self.budget)
        # drift_hook (only when caller passed in a fresh val_acc)
        if current_overall_val_acc is not None:
            self.val_acc_history.append(float(current_overall_val_acc))
            if len(self.val_acc_history) >= 3:
                a, b, c = self.val_acc_history[-3:]
                if a > b > c:
                    self.budget = np.minimum(
                        (self.budget * 1.3).astype(np.int64), self.label_hist)


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

        # Per-class accuracy (np.float64, length n_classes)
        per_class = np.zeros(self.n_classes, dtype=np.float64)
        for c in range(self.n_classes):
            mask = all_labels == c
            if mask.any():
                per_class[c] = float((all_preds[mask] == c).mean())

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
        # Per-class array stored separately to keep CSV columns clean
        self.metrics.setdefault(f"_per_class_{method_name}", []).append(
            per_class)
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

        # Save per-class accuracy matrix (rounds × n_classes)
        pcs = self.metrics.get(f"_per_class_{method_name}", [])
        if pcs:
            mat = np.stack(pcs)  # (n_rounds, n_classes)
            np.save(os.path.join(self.results_dir,
                                 f"{method_name}_per_class.npy"), mat)

    def save_final_json(self):
        summary = {}
        for method, records in self.metrics.items():
            if method.startswith("_per_class_"):
                continue
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

        # Skip auxiliary per-class arrays which share self.metrics dict.
        plot_metrics = {k: v for k, v in self.metrics.items()
                        if not k.startswith("_per_class_")}
        if not plot_metrics:
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
        for method, records in plot_metrics.items():
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
        for method, records in plot_metrics.items():
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
        for method, records in plot_metrics.items():
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
        for method, records in plot_metrics.items():
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

    # Early stopping helper. The Centralized loop is epoch-paced; we let
    # patience be measured in *rounds* (mapped from epochs) so it's
    # comparable to federated runs.
    stopper = EarlyStopper(getattr(args, "early_stop_patience", 0))

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

        if stopper.update(acc, mapped_round):
            logger.info(f"[Centralized] Early stop at epoch {epoch} "
                        f"(→R{mapped_round}) — no improvement for "
                        f"{stopper.patience} rounds (best acc="
                        f"{stopper.best_acc:.4f} at R{stopper.best_round})")
            break

    evaluator.save_csv("Centralized")
    logger.info(f"[Centralized] Final acc={acc:.4f} "
                f"(best={stopper.best_acc:.4f} @ R{stopper.best_round})")


class EarlyStopper:
    """Track best test acc and signal stop after `patience` rounds of no
    improvement. patience=0 disables (always returns False)."""
    def __init__(self, patience: int):
        self.patience = int(patience)
        self.best_acc = -1.0
        self.best_round = 0
        self.rounds_since_best = 0

    def update(self, acc: float, rnd: int) -> bool:
        if acc > self.best_acc:
            self.best_acc = acc
            self.best_round = rnd
            self.rounds_since_best = 0
        else:
            self.rounds_since_best += 1
        return (self.patience > 0 and
                self.rounds_since_best >= self.patience)


# ====================================================================== #
#  Run FedAvg / FedProx (shared loop, only client.local_train differs)    #
# ====================================================================== #
def _run_fedavg_like(args, clients, server, evaluator, logger,
                     method_name, train_fn, bus=None):
    """Common loop for FedAvg-style baselines. `train_fn(client, head)`
    returns (state_dict, comm_bytes)."""
    logger.info("=" * 60)
    logger.info(f"Starting {method_name} baseline")
    logger.info("=" * 60)

    cumulative_comm = 0
    stopper = EarlyStopper(getattr(args, "early_stop_patience", 0))
    acc, f1 = 0.0, 0.0

    for rnd in range(1, args.rounds + 1):
        t0 = time.time()
        round_comm = 0
        client_sds, client_sizes = [], []

        for c in clients:
            if bus:
                t = bus.send_task("server", f"client_{c.id}", "local_train",
                                  [Part(type="json",
                                        content={"round": rnd,
                                                 "local_epochs": args.local_epochs})])
            sd, comm = train_fn(c, server.get_fedavg_head())
            if bus:
                bus.complete_task(t.task_id, artifacts=[Artifact(
                    artifact_id=f"sd_r{rnd}_c{c.id}", name="state_dict",
                    data=None, size_bytes=comm)])
            client_sds.append(sd)
            client_sizes.append(c.n_samples)
            round_comm += comm

        if method_name == "FedAdam":
            server.fedadam_aggregate(client_sds, client_sizes)
        else:
            server.fedavg_aggregate(client_sds, client_sizes)
        head_params = sum(p.numel() for p in server.get_fedavg_head().parameters())
        round_comm += head_params * 4 * args.n_clients
        cumulative_comm += round_comm

        if bus:
            t = bus.send_task("server", "evaluator", "evaluate",
                              [Part(type="json",
                                    content={"method": method_name, "round": rnd})])
        acc, f1 = evaluator.evaluate(server.get_fedavg_head(), method_name,
                                     rnd, round_comm, cumulative_comm)
        if bus:
            bus.complete_task(t.task_id, response_parts=[Part(type="json",
                content={"acc": acc, "f1": f1})])

        dt = time.time() - t0
        if rnd % 10 == 0 or rnd == 1:
            logger.info(f"[{method_name}] R{rnd:3d} | Acc:{acc:.4f} F1:{f1:.4f} "
                        f"Comm:{cumulative_comm/1e6:.1f}MB ({dt:.1f}s)")

        if stopper.update(acc, rnd):
            logger.info(f"[{method_name}] Early stop at R{rnd} — no "
                        f"improvement for {stopper.patience} rounds (best "
                        f"acc={stopper.best_acc:.4f} at R{stopper.best_round})")
            break

    evaluator.save_csv(method_name)
    logger.info(f"[{method_name}] Final acc={acc:.4f} "
                f"(best={stopper.best_acc:.4f} @ R{stopper.best_round})")


def run_fedavg(args, clients, server, evaluator, logger, bus=None):
    def train_fn(c, head):
        return c.local_train(head, args.local_epochs, args.fedavg_lr,
                             args.batch_size)
    _run_fedavg_like(args, clients, server, evaluator, logger,
                     "FedAvg", train_fn, bus=bus)


def run_fedprox(args, clients, server, evaluator, logger, bus=None):
    def train_fn(c, head):
        return c.local_train(head, args.local_epochs, args.fedavg_lr,
                             args.batch_size, mu=args.fedprox_mu)
    logger.info(f"  FedProx μ = {args.fedprox_mu}")
    _run_fedavg_like(args, clients, server, evaluator, logger,
                     "FedProx", train_fn, bus=bus)


def run_fedadam(args, clients, server, evaluator, logger, bus=None):
    server.init_fedadam(lr=args.fedadam_lr, beta1=args.fedadam_beta1,
                        beta2=args.fedadam_beta2, tau=args.fedadam_tau)
    logger.info(f"  FedAdam η={args.fedadam_lr} β1={args.fedadam_beta1} "
                f"β2={args.fedadam_beta2} τ={args.fedadam_tau}")
    def train_fn(c, head):
        return c.local_train(head, args.local_epochs, args.fedavg_lr,
                             args.batch_size)
    _run_fedavg_like(args, clients, server, evaluator, logger,
                     "FedAdam", train_fn, bus=bus)


# ====================================================================== #
#  Run Proposed                                                           #
# ====================================================================== #
def run_proposed(args, clients, server, evaluator, logger, bus=None):
    logger.info("=" * 60)
    logger.info("Starting AO-FRL: Agent-Orchestrated Fed Rep Sharing")
    logger.info(f"  sigma={args.sigma:.4f} (eps={args.epsilon}, delta={args.delta}, "
                f"clip_C={args.clip_C})")
    logger.info(f"  T_base={args.per_class_target}, "
                f"head_sync_every={args.head_sync_every}, "
                f"feedback_alpha={args.feedback_alpha}")
    logger.info("=" * 60)

    n_classes = server.n_classes
    n_clients = len(clients)

    # ---- Round 0: collect histograms, server allocates initial budgets ----
    label_hist_matrix = np.stack([c.label_hist for c in clients], axis=0)
    server.init_budgets(label_hist_matrix, T_base=args.per_class_target)
    for i, c in enumerate(clients):
        c.apply_budget(server.get_budgets()[i])
    # Histogram upload comm: 100 ints * 4B per client (one-time)
    init_comm = n_clients * n_classes * 4
    # Initial budget broadcast: 100 ints * 4B per client
    init_comm += n_clients * n_classes * 4

    cumulative_comm = init_comm
    history = []
    rng = np.random.default_rng(args.seed)

    # Early stopping state.
    best_acc = -1.0
    best_round = 0
    rounds_since_best = 0
    patience = int(getattr(args, "early_stop_patience", 0))

    for rnd in range(1, args.rounds + 1):
        t0 = time.time()
        round_comm = 0
        sync_this_round = (rnd % args.head_sync_every == 0)

        # ---- Optional sync: head down -> client per-class acc -> rebalance ----
        if sync_this_round:
            head_params = sum(p.numel() for p in server.get_head().parameters())
            head_bytes = head_params * 4
            # Server -> all clients: head download
            round_comm += head_bytes * n_clients

            per_class_correct = np.zeros(n_classes, dtype=np.int64)
            per_class_total = np.zeros(n_classes, dtype=np.int64)
            per_client_overall = []  # for legacy drift_hook
            for c in clients:
                if bus:
                    t = bus.send_task("server", f"client_{c.id}", "evaluate",
                                      [Part(type="json", content={"round": rnd})])
                acc_vec, total_vec = c.evaluate_per_class_on_train(
                    server.get_head())
                if bus:
                    bus.complete_task(t.task_id,
                                      response_parts=[Part(type="json",
                                          content={"per_class_acc": acc_vec.tolist()})])
                # Aggregate across clients, weighted by client sample count
                per_class_correct += (acc_vec * total_vec).astype(np.int64)
                per_class_total += total_vec
                # Per-client overall val acc (used by legacy drift_hook)
                tot = int(total_vec.sum())
                overall = float((acc_vec * total_vec).sum() / tot) if tot > 0 else 0.0
                per_client_overall.append(overall)
                # Client -> server: 100 floats per client
                round_comm += n_classes * 4

            global_per_class_acc = np.divide(
                per_class_correct, per_class_total,
                out=np.zeros(n_classes), where=per_class_total > 0)

            # Server updates per-class targets and reallocates budgets
            new_budget = server.update_budgets_from_feedback(
                global_per_class_acc, alpha=args.feedback_alpha)
            for i, c in enumerate(clients):
                c.apply_budget(new_budget[i])
                # Re-enabled legacy client-side hooks (low_data + drift)
                c.apply_legacy_hooks(current_overall_val_acc=per_client_overall[i])
            # Budget broadcast: 100 ints * 4B per client
            round_comm += n_clients * n_classes * 4

        # ---- Each round: select participating clients (partial participation
        # if args.clients_per_round > 0) ----
        k_per_round = int(getattr(args, "clients_per_round", -1))
        if k_per_round > 0 and k_per_round < n_clients:
            participating_idx = rng.choice(n_clients, size=k_per_round,
                                           replace=False)
            participating = [clients[i] for i in sorted(participating_idx)]
        else:
            participating = clients

        # ---- Each participating client extracts DP-noised embeddings ----
        all_embs, all_labs, summaries = [], [], []
        for c in participating:
            if bus:
                t = bus.send_task("server", f"client_{c.id}",
                                  "extract_embeddings",
                                  [Part(type="json", content={"round": rnd})])
            embs, labs, summary = c.extract_dp_embeddings(rng=rng)
            if bus:
                bus.complete_task(
                    t.task_id,
                    artifacts=[Artifact(artifact_id=f"emb_r{rnd}_c{c.id}",
                                        name="dp_embeddings", data=None,
                                        size_bytes=estimate_comm_bytes(embs))],
                    response_parts=[Part(type="json", content=summary)])
            all_embs.append(embs)
            all_labs.append(labs)
            summaries.append(summary)
            round_comm += estimate_comm_bytes(embs)

        merged_embs = (torch.cat([e for e in all_embs if e.numel() > 0])
                       if any(e.numel() > 0 for e in all_embs)
                       else torch.zeros(0, server.embed_dim))
        merged_labs = (torch.cat([l for l in all_labs if l.numel() > 0])
                       if any(l.numel() > 0 for l in all_labs)
                       else torch.zeros(0, dtype=torch.long))

        # ---- Server trains head on accumulated noisy embeddings ----
        server.train_head(merged_embs, merged_labs,
                          epochs=args.server_train_epochs)

        cumulative_comm += round_comm

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

        # Per-round logging
        T_vec = server.get_per_class_target()
        total_uploaded = sum(s["n_uploaded"] for s in summaries)
        history.append({
            "round": rnd,
            "synced": int(sync_this_round),
            "total_uploaded": total_uploaded,
            "T_min": int(T_vec.min()),
            "T_max": int(T_vec.max()),
            "T_mean": float(T_vec.mean()),
            "T_std": float(T_vec.std()),
        })

        dt = time.time() - t0
        if rnd % 10 == 0 or rnd == 1 or sync_this_round:
            sync_tag = " [SYNC]" if sync_this_round else ""
            logger.info(
                f"[AO-FRL]{sync_tag} R{rnd:3d} | Acc:{acc:.4f} F1:{f1:.4f} "
                f"Comm:{cumulative_comm/1e6:.1f}MB Up:{total_uploaded} "
                f"T:[{T_vec.min()}, {T_vec.max()}] mean={T_vec.mean():.0f} "
                f"({dt:.1f}s)")

        # Early stopping: track best acc, stop if no improvement for `patience`
        # rounds (configurable, 0 disables).
        if acc > best_acc:
            best_acc = acc
            best_round = rnd
            rounds_since_best = 0
        else:
            rounds_since_best += 1
        if patience > 0 and rounds_since_best >= patience:
            logger.info(f"[AO-FRL] Early stop at R{rnd} — no improvement for "
                        f"{patience} rounds (best acc={best_acc:.4f} at "
                        f"R{best_round})")
            break

    evaluator.save_csv("AO-FRL")
    logger.info(f"[AO-FRL] Final acc={acc:.4f} (best={best_acc:.4f} "
                f"@ R{best_round})")

    # Save and plot per-round target/upload trends
    save_aofrl_history(history, args.results_dir)
    plot_aofrl_history(history, args.results_dir)


@torch.no_grad()
def run_inversion_eval(args, head, encoder, train_ds, test_ds, decoder_pool,
                       embed_dim, device, logger):
    """Final-round privacy/utility evaluation via decoder inversion.

    For a fixed pool of test images, run encoder -> embedding -> [clip + DP
    noise] -> decoder -> reconstruction. Compare reconstruction to the original
    32x32 image via PSNR. Reports both:
      - clean_psnr: encoder -> decoder (no noise) — reconstruction ceiling
      - noisy_psnr: encoder -> clip + N(0, sigma^2) -> decoder — what an
        attacker recovers from a single noisy upload at this DP setting

    Decoder weights from train_autoencoder.py. Skipped silently if missing.
    """
    from models import Decoder

    if not os.path.exists(args.decoder_weights):
        logger.warning(f"Decoder weights not found at {args.decoder_weights}; "
                       "skipping inversion eval.")
        return

    decoder = Decoder(embed_dim=embed_dim).to(device).eval()
    decoder.load_state_dict(torch.load(args.decoder_weights,
                                       map_location=device))

    enc_tf = T.Compose([
        T.Resize(256), T.CenterCrop(224), T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    tgt_tf = T.ToTensor()

    n = min(args.inversion_n_samples, len(test_ds))
    rng = np.random.default_rng(args.seed)
    sample_idx = rng.choice(len(test_ds), size=n, replace=False).tolist()

    logger.info(f"Running inversion eval on {n} test images...")

    enc_inputs, targets, labels = [], [], []
    for i in sample_idx:
        img, label = test_ds[i]
        enc_inputs.append(enc_tf(img))
        targets.append(tgt_tf(img))
        labels.append(label)
    enc_inputs = torch.stack(enc_inputs).to(device)
    targets = torch.stack(targets).to(device)

    # Encode
    encoder.eval()
    z_raw = encoder(enc_inputs)
    z_unit = F.normalize(z_raw, dim=1)

    # L2 clip (no-op when norm <= clip_C; unit-norm @ clip_C=1 is no-op)
    norms = z_unit.norm(dim=1, keepdim=True).clamp(min=1e-12)
    z_clip = z_unit * torch.clamp(args.clip_C / norms, max=1.0)

    # Two reconstructions: clean vs DP-noisy
    recon_clean = decoder(z_clip).clamp(0, 1)
    z_noisy = z_clip + torch.randn_like(z_clip) * args.sigma
    recon_noisy = decoder(z_noisy).clamp(0, 1)

    # Per-sample MSE -> PSNR
    def per_sample_psnr(rec, tgt):
        mse = (rec - tgt).pow(2).mean(dim=(1, 2, 3)).clamp(min=1e-12)
        return (10.0 * torch.log10(1.0 / mse)).cpu().numpy()

    psnr_clean = per_sample_psnr(recon_clean, targets)
    psnr_noisy = per_sample_psnr(recon_noisy, targets)

    # Per-class predictions on the head, for utility-on-noisy reference
    head.eval()
    head_dev = head.to(device)
    preds_clean = head_dev(z_clip).argmax(1).cpu().numpy()
    preds_noisy = head_dev(z_noisy).argmax(1).cpu().numpy()
    labels_np = np.array(labels)

    summary = {
        "n_samples": n,
        "sigma": float(args.sigma),
        "epsilon": float(args.epsilon),
        "delta": float(args.delta),
        "clip_C": float(args.clip_C),
        "psnr_clean_mean": float(psnr_clean.mean()),
        "psnr_clean_std": float(psnr_clean.std()),
        "psnr_noisy_mean": float(psnr_noisy.mean()),
        "psnr_noisy_std": float(psnr_noisy.std()),
        "head_acc_on_clean": float((preds_clean == labels_np).mean()),
        "head_acc_on_noisy": float((preds_noisy == labels_np).mean()),
    }
    logger.info("Inversion eval results:")
    for k, v in summary.items():
        logger.info(f"  {k}: {v}")

    # Save summary JSON
    out_path = os.path.join(args.results_dir, "inversion_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Save per-sample CSV
    import csv
    csv_path = os.path.join(args.results_dir, "inversion_per_sample.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_idx", "label", "psnr_clean", "psnr_noisy",
                    "pred_clean", "pred_noisy"])
        for j, idx in enumerate(sample_idx):
            w.writerow([idx, int(labels[j]),
                        float(psnr_clean[j]), float(psnr_noisy[j]),
                        int(preds_clean[j]), int(preds_noisy[j])])

    # Save visualization grid: original | clean recon | noisy recon for 8 imgs
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n_show = min(8, n)
    fig, axes = plt.subplots(3, n_show, figsize=(2 * n_show, 6))
    rows = ["Original", "Recon (clean)", f"Recon (σ={args.sigma:.2f})"]
    panels = [targets[:n_show], recon_clean[:n_show], recon_noisy[:n_show]]
    for r, (label, panel) in enumerate(zip(rows, panels)):
        for c in range(n_show):
            ax = axes[r, c] if n_show > 1 else axes[r]
            ax.imshow(panel[c].cpu().permute(1, 2, 0).numpy())
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(label, fontsize=11)
            if r == 0:
                ax.set_title(f"id {sample_idx[c]}", fontsize=9)
            elif r == 1:
                ax.set_xlabel(f"{psnr_clean[c]:.1f}dB", fontsize=9)
            elif r == 2:
                ax.set_xlabel(f"{psnr_noisy[c]:.1f}dB", fontsize=9)
    fig.suptitle(f"Decoder Inversion — σ={args.sigma:.3f} "
                 f"(ε={args.epsilon}, δ={args.delta})", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(args.results_dir, "inversion_grid.png"), dpi=150)
    plt.close(fig)


def save_aofrl_history(history, results_dir):
    """Per-round AO-FRL stats: sync flag, total uploads, T_c distribution."""
    import csv
    if not history:
        return
    path = os.path.join(results_dir, "aofrl_history.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=history[0].keys())
        w.writeheader()
        w.writerows(history)


def plot_aofrl_history(history, results_dir):
    """Plot per-class target T_c spread and total uploads over rounds."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not history:
        return
    rounds = [h["round"] for h in history]
    t_min = [h["T_min"] for h in history]
    t_max = [h["T_max"] for h in history]
    t_mean = [h["T_mean"] for h in history]
    total_up = [h["total_uploaded"] for h in history]
    sync_rounds = [h["round"] for h in history if h["synced"]]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(rounds, t_mean, color="tab:blue", linewidth=2, label="mean T_c")
    ax.fill_between(rounds, t_min, t_max, alpha=0.25, color="tab:blue",
                    label="min–max T_c")
    for sr in sync_rounds:
        ax.axvline(sr, color="tab:red", alpha=0.2, linewidth=0.8)
    ax.set_xlabel("Round")
    ax.set_ylabel("Per-class target T_c")
    ax.set_title("Server's per-class upload target (red = sync round)")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(rounds, total_up, color="tab:green", linewidth=2)
    ax.set_xlabel("Round")
    ax.set_ylabel("Total uploaded embeddings")
    ax.set_title("Total noisy embeddings collected per round")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "aofrl_history.png"), dpi=150)
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

    # Derive sigma from (epsilon, delta) unless explicitly overridden.
    if args.sigma is None:
        args.sigma = gaussian_dp_sigma(args.epsilon, args.delta, args.clip_C)
        logger.info(f"DP sigma derived: sigma={args.sigma:.4f} "
                    f"from epsilon={args.epsilon}, delta={args.delta}, "
                    f"clip_C={args.clip_C}")
    else:
        logger.info(f"DP sigma explicitly set: sigma={args.sigma:.4f}")

    # ---- Load dataset ----
    ds_name = args.dataset.lower()
    logger.info(f"Loading dataset: {ds_name.upper()}...")
    if ds_name == "cifar100":
        train_ds = torchvision.datasets.CIFAR100(
            root="./data", train=True, download=True, transform=None)
        test_ds = torchvision.datasets.CIFAR100(
            root="./data", train=False, download=True, transform=None)
        train_labels = np.array(train_ds.targets)
        n_classes = 100
    elif ds_name == "cifar10":
        train_ds = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=None)
        test_ds = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=None)
        train_labels = np.array(train_ds.targets)
        n_classes = 10
    elif ds_name == "svhn":
        train_ds = torchvision.datasets.SVHN(
            root="./data", split="train", download=True, transform=None)
        test_ds = torchvision.datasets.SVHN(
            root="./data", split="test", download=True, transform=None)
        train_labels = np.array(train_ds.labels)
        n_classes = 10
    else:
        raise ValueError(f"Unsupported dataset: {ds_name}")
    logger.info(f"  → train {len(train_ds)}, test {len(test_ds)}, "
                f"n_classes={n_classes}")

    # Decoder pool is held out from clients (disjoint from federated data).
    # Same seed as train_autoencoder.py → identical split.
    decoder_pool, federated_pool = split_decoder_pool(
        train_labels, frac=0.1, seed=args.seed)
    logger.info(f"Held out {len(decoder_pool)} decoder-pool images. "
                f"Federated pool: {len(federated_pool)} images.")
    fed_labels = train_labels[federated_pool]

    # Partition the FEDERATED POOL across clients (Dirichlet non-IID).
    logger.info(f"Partitioning: {args.n_clients} clients, alpha={args.alpha}")
    client_local_indices = dirichlet_partition(fed_labels, args.n_clients,
                                               args.alpha, args.seed)
    # Translate local positions in fed_labels back to absolute indices.
    client_indices = [federated_pool[idx] for idx in client_local_indices]
    for i, idx in enumerate(client_indices):
        nc = len(set(train_labels[idx]))
        logger.info(f"  Client {i}: {len(idx)} samples, {nc} classes")

    # Build encoder (loads fine-tuned weights from train_autoencoder.py if
    # present; otherwise falls back to ImageNet-pretrained baseline).
    if os.path.exists(args.encoder_weights):
        logger.info(f"Loading fine-tuned encoder: {args.encoder_weights}")
    else:
        logger.info("Encoder weights not found; using ImageNet-pretrained.")
    encoder, embed_dim = build_encoder(device, args.encoder_weights)

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

    # Run Centralized (upper bound) — also restricted to federated pool
    # for a fair comparison with FedAvg / AO-FRL.
    if "centralized" in args.methods:
        set_seed(args.seed)
        run_centralized(args, train_embs[federated_pool],
                        train_labs[federated_pool], evaluator,
                        embed_dim, n_classes, device, logger, bus=bus)

    def _build_fedavg_clients(client_class, **client_kwargs):
        clients_out = []
        for i in range(args.n_clients):
            tr_idx, va_idx = split_train_val(client_indices[i],
                                             args.val_ratio, args.seed)
            c = client_class(i, train_embs[tr_idx], train_labs[tr_idx],
                             train_embs[va_idx], train_labs[va_idx],
                             n_classes, device, **client_kwargs)
            clients_out.append(c)
        return clients_out

    # Run FedAvg
    if "fedavg" in args.methods:
        set_seed(args.seed)
        server_fa = ServerAgent(embed_dim, n_classes, args.n_clients,
                                 device, cfg)
        clients_fa = _build_fedavg_clients(FedAvgClient)
        run_fedavg(args, clients_fa, server_fa, evaluator, logger, bus=bus)

    # Run FedProx — adds proximal term to client local loss.
    if "fedprox" in args.methods:
        set_seed(args.seed)
        server_fp = ServerAgent(embed_dim, n_classes, args.n_clients,
                                device, cfg)
        clients_fp = _build_fedavg_clients(FedProxClient)
        run_fedprox(args, clients_fp, server_fp, evaluator, logger, bus=bus)

    # Run FedAdam — server-side Adam over client deltas.
    if "fedadam" in args.methods:
        set_seed(args.seed)
        server_fad = ServerAgent(embed_dim, n_classes, args.n_clients,
                                  device, cfg)
        clients_fad = _build_fedavg_clients(FedAvgClient)
        run_fedadam(args, clients_fad, server_fad, evaluator, logger, bus=bus)

    # Run AO-FRL — DP-noised embedding sharing with budget orchestration.
    if "ao-frl" in args.methods:
        set_seed(args.seed)
        server_pr = ServerAgent(embed_dim, n_classes, args.n_clients,
                                 device, cfg)
        clients_pr = []
        for i in range(args.n_clients):
            # AO-FRL client uses ALL of its assigned data (no train/val split):
            # validation feedback comes from running head on clean train embs.
            idx = client_indices[i]
            c = ProposedClient(
                i, train_embs[idx], train_labs[idx],
                n_classes, embed_dim, device, cfg)
            clients_pr.append(c)
        run_proposed(args, clients_pr, server_pr, evaluator, logger, bus=bus)

        # Final-round inversion + PSNR evaluation with held-out decoder pool.
        run_inversion_eval(args, server_pr.get_head(), encoder,
                           train_ds, test_ds, decoder_pool,
                           embed_dim, device, logger)

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
