"""ServerAgent: aggregation, head training, orchestration, and FedAvg."""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from utils import estimate_comm_bytes


class MLPHead(nn.Module):
    """Lightweight MLP classification head."""
    def __init__(self, in_dim: int, n_classes: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)


class ServerAgent:
    """Federated server agent — trains head, orchestrates clients."""

    SKILL_FILE = "skills/server_agent.md"

    def __init__(self, embed_dim: int, n_classes: int, n_clients: int,
                 device: str, cfg: dict):
        self.embed_dim = embed_dim
        self.n_classes = n_classes
        self.n_clients = n_clients
        self.device = device
        self.cfg = cfg

        # Create the global head
        self.head = MLPHead(embed_dim, n_classes,
                            hidden=cfg.get("head_hidden", 256)).to(device)

        # Optimizer
        lr = cfg.get("server_lr", 1e-3)
        opt_name = cfg.get("server_optimizer", "adam")
        if opt_name == "sgd":
            self.optimizer = torch.optim.SGD(self.head.parameters(), lr=lr,
                                             momentum=0.9)
        else:
            self.optimizer = torch.optim.Adam(self.head.parameters(), lr=lr)
        self.loss_fn = nn.CrossEntropyLoss()

        # For FedAvg baseline
        self.fedavg_head = MLPHead(embed_dim, n_classes,
                                    hidden=cfg.get("head_hidden", 256)).to(device)

        # Orchestration state
        self.client_summaries = {}

        # Replay buffer: accumulate embeddings across rounds so noise averages out
        self._replay_embs = []
        self._replay_labs = []
        self._replay_max = cfg.get("replay_max", 50000)

    # ------------------------------------------------------------------ #
    #  Proposed method: train head on collected embeddings                 #
    # ------------------------------------------------------------------ #
    def train_head(self, all_embeddings: torch.Tensor,
                   all_labels: torch.Tensor, epochs: int = 3):
        """Train head on server-side collected embeddings.
        Accumulates embeddings in a replay buffer across rounds so that
        independent noise from different rounds averages out.
        Returns communication cost (bytes) for broadcasting head.
        """
        if len(all_embeddings) == 0:
            n_params = sum(p.numel() for p in self.head.parameters())
            return n_params * 4

        # Append current round's data to replay buffer
        self._replay_embs.append(all_embeddings.detach().cpu())
        self._replay_labs.append(all_labels.detach().cpu())

        # Concatenate full buffer
        buf_embs = torch.cat(self._replay_embs)
        buf_labs = torch.cat(self._replay_labs)

        # Cap buffer size: keep most recent data
        if buf_embs.size(0) > self._replay_max:
            buf_embs = buf_embs[-self._replay_max:]
            buf_labs = buf_labs[-self._replay_max:]
            # Rebuild list so memory doesn't grow unbounded
            self._replay_embs = [buf_embs]
            self._replay_labs = [buf_labs]

        # Reset optimizer each round to avoid stale momentum on noisy data,
        # but keep head weights for continuity across rounds.
        lr = self.cfg.get("server_lr", 1e-3)
        opt_name = self.cfg.get("server_optimizer", "adam")
        if opt_name == "sgd":
            optimizer = torch.optim.SGD(self.head.parameters(), lr=lr, momentum=0.9)
        else:
            optimizer = torch.optim.Adam(self.head.parameters(), lr=lr)

        ds = TensorDataset(buf_embs, buf_labs)
        loader = DataLoader(ds, batch_size=256, shuffle=True)

        self.head.train()
        for _ in range(epochs):
            for z_batch, y_batch in loader:
                z_batch = z_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                logits = self.head(z_batch)
                loss = self.loss_fn(logits, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Comm cost: broadcast head params to all clients
        n_params = sum(p.numel() for p in self.head.parameters())
        return n_params * 4  # bytes for one download

    # ------------------------------------------------------------------ #
    #  Orchestration: generate per-client instructions                    #
    # ------------------------------------------------------------------ #
    def orchestrate(self, summaries: list) -> list:
        """Analyze summaries and generate per-client instructions."""
        self.client_summaries = {s["client_id"]: s for s in summaries}

        # Compute global label distribution
        global_hist = np.zeros(self.n_classes)
        for s in summaries:
            global_hist += np.array(s["label_histogram"])

        # Target: uniform
        target_per_class = global_hist.sum() / self.n_classes if global_hist.sum() > 0 else 1
        label_gap = np.clip(target_per_class - global_hist, 0, None)
        label_gap_normalized = label_gap / (label_gap.sum() + 1e-8)

        instructions = []
        for s in summaries:
            cid = s["client_id"]
            hist = np.array(s["label_histogram"])
            reject_ratio = s["reject_ratio"]
            sigma = s["sigma"]

            # Compute client-specific budget
            # Clients with rare classes get higher budget
            client_classes = np.where(hist > 0)[0]
            rarity_score = label_gap_normalized[client_classes].sum() if len(client_classes) > 0 else 0

            base_budget = self.cfg.get("upload_budget", 500)
            budget = int(base_budget * (1 + rarity_score))

            # Noise: increase for high-risk clients
            new_sigma = sigma
            aug_mode = "normal"

            # Hook: high_risk
            if reject_ratio > self.cfg.get("high_risk_r", 0.30):
                new_sigma = min(sigma * 1.5, 0.5)
                budget = max(budget // 2, 50)
                aug_mode = "conservative"

            # Hook: low_data
            low_k = self.cfg.get("low_data_k", 10)
            has_low = any(hist[c] < low_k and hist[c] > 0
                          for c in range(self.n_classes))
            if has_low:
                budget = int(budget * 1.2)
                aug_mode = "conservative"

            instructions.append({
                "client_id": cid,
                "upload_budget": budget,
                "sigma": new_sigma,
                "augmentation_mode": aug_mode,
            })

        return instructions

    # ------------------------------------------------------------------ #
    #  FedAvg baseline                                                     #
    # ------------------------------------------------------------------ #
    def fedavg_aggregate(self, client_state_dicts: list,
                         client_sizes: list):
        """Weighted average of client head parameters."""
        total = sum(client_sizes)
        if total == 0:
            return
        avg_state = {}
        for key in client_state_dicts[0]:
            avg_state[key] = sum(
                sd[key].float() * (n / total)
                for sd, n in zip(client_state_dicts, client_sizes)
            )
        self.fedavg_head.load_state_dict(avg_state)

    def get_fedavg_head(self):
        return self.fedavg_head

    def get_head(self):
        return self.head
