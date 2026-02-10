"""EvaluatorAgent: global evaluation, metric logging, and visualization."""

import json
import csv
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as T
from sklearn.metrics import f1_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class EvaluatorAgent:
    """Global evaluation agent — metrics, logging, plotting."""

    SKILL_FILE = "skills/evaluator_agent.md"

    def __init__(self, test_dataset, encoder, n_classes: int,
                 device: str, results_dir: str = "results"):
        self.test_dataset = test_dataset
        self.encoder = encoder
        self.n_classes = n_classes
        self.device = device
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)

        self.transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        # Per-method tracking
        self.metrics = {}  # method_name -> list of per-round dicts

        # Cache test embeddings (encoder is frozen)
        self._cached_test_embs = None
        self._cached_test_labels = None

    def _make_loader(self, batch_size=256):
        from agents.client_agent import _TransformSubset
        indices = list(range(len(self.test_dataset)))
        ds = _TransformSubset(self.test_dataset, indices, self.transform)
        return DataLoader(ds, batch_size=batch_size, shuffle=False,
                          num_workers=2, pin_memory=True)

    @torch.no_grad()
    def _precompute_test_embeddings(self):
        """Precompute and cache test set embeddings."""
        self.encoder.eval()
        loader = self._make_loader()
        embs_list, labels_list = [], []
        for imgs, labels in loader:
            imgs = imgs.to(self.device)
            z = self.encoder(imgs).cpu()
            z = F.normalize(z, dim=1)
            embs_list.append(z)
            labels_list.append(labels)
        self._cached_test_embs = torch.cat(embs_list, dim=0)
        self._cached_test_labels = torch.cat(labels_list, dim=0)

    @torch.no_grad()
    def evaluate(self, head, method_name: str, round_num: int,
                 comm_bytes: int, cumulative_comm: int):
        """Evaluate encoder+head on global test set, log metrics."""
        if self._cached_test_embs is None:
            self._precompute_test_embeddings()

        head.eval()
        head_dev = head.to(self.device)

        # Evaluate in batches using cached embeddings
        all_preds, all_labels = [], []
        bs = 512
        for i in range(0, self._cached_test_embs.size(0), bs):
            z = self._cached_test_embs[i:i+bs].to(self.device)
            preds = head_dev(z).argmax(dim=1).cpu()
            all_preds.append(preds)
            all_labels.append(self._cached_test_labels[i:i+bs])

        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()

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

    # ------------------------------------------------------------------ #
    #  Logging                                                             #
    # ------------------------------------------------------------------ #
    def save_csv(self, method_name: str):
        """Save per-round metrics to CSV."""
        records = self.metrics.get(method_name, [])
        if not records:
            return
        path = os.path.join(self.results_dir, f"{method_name}_rounds.csv")
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)

    def save_final_json(self):
        """Save final metrics summary for all methods."""
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

    # ------------------------------------------------------------------ #
    #  Visualization                                                       #
    # ------------------------------------------------------------------ #
    def plot_comparisons(self):
        """Generate comparison plots across all methods."""
        if not self.metrics:
            return

        # --- Accuracy vs Rounds ---
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

        # --- Communication vs Accuracy ---
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

        print(f"Plots saved to {self.results_dir}/")
