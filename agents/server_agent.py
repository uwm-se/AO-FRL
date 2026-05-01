"""ServerAgent: aggregation, head training, orchestration, and FedAvg."""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from utils import allocate_budgets, update_per_class_target


class MLPHead(nn.Module):
    """Lightweight MLP classification head.

    LayerNorm (not BatchNorm) — the head is trained on DP-noised embeddings
    (||z||≈√(1+σ²d)) but evaluated on clean unit-norm test embeddings.
    BatchNorm would capture the noisy training distribution in its running
    stats and corrupt clean inference; LayerNorm normalizes per-sample so
    train/test distributions are decoupled.
    """
    def __init__(self, in_dim: int, n_classes: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
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

        # Replay buffer: flat per-sample storage so age tracking survives
        # truncation. Without this, the previous chunk-based design reset all
        # surviving sample ages to the current round on each truncation,
        # destroying cross-round noise averaging.
        self._replay_embs = torch.zeros(0, embed_dim)              # (N, D)
        self._replay_labs = torch.zeros(0, dtype=torch.long)        # (N,)
        self._replay_rounds = torch.zeros(0, dtype=torch.float32)   # (N,)
        self._replay_max = int(cfg.get("replay_max", 500_000))
        self._current_round = 0

    # ------------------------------------------------------------------ #
    #  Proposed method: train head on collected embeddings                 #
    # ------------------------------------------------------------------ #
    def train_head(self, all_embeddings: torch.Tensor,
                   all_labels: torch.Tensor, epochs: int = 3):
        """Train head on server-side collected embeddings.

        Accumulates noisy embeddings in a flat replay buffer across rounds so
        independent DP noise from different rounds can be averaged out via
        WeightedRandomSampler with exponential age decay.

        Returns: bytes for broadcasting head params (one download).
        """
        self._current_round += 1
        n_params = sum(p.numel() for p in self.head.parameters())
        head_bytes = n_params * 4

        # Append current round's samples
        if all_embeddings.numel() > 0:
            embs_cpu = all_embeddings.detach().cpu()
            labs_cpu = all_labels.detach().cpu()
            rnds = torch.full((embs_cpu.size(0),),
                              float(self._current_round),
                              dtype=torch.float32)
            self._replay_embs = torch.cat([self._replay_embs, embs_cpu])
            self._replay_labs = torch.cat([self._replay_labs, labs_cpu])
            self._replay_rounds = torch.cat([self._replay_rounds, rnds])

        # Cap buffer (keep most recent samples; per-sample ages preserved)
        if self._replay_embs.size(0) > self._replay_max:
            keep = -self._replay_max
            self._replay_embs = self._replay_embs[keep:]
            self._replay_labs = self._replay_labs[keep:]
            self._replay_rounds = self._replay_rounds[keep:]

        if self._replay_embs.size(0) == 0:
            return head_bytes

        # Per-sample weights: decay^age, floored at replay_min_weight
        decay = self.cfg.get("replay_decay", 0.995)
        min_weight = self.cfg.get("replay_min_weight", 0.3)
        ages = self._current_round - self._replay_rounds
        weights = torch.clamp(decay ** ages, min=min_weight)

        # LR decay: lr * server_lr_decay^round, floored at server_lr_min
        base_lr = self.cfg.get("server_lr", 1e-3)
        lr_decay = self.cfg.get("server_lr_decay", 0.98)
        lr_min = self.cfg.get("server_lr_min", 1e-4)
        lr = max(base_lr * (lr_decay ** (self._current_round - 1)), lr_min)

        opt_name = self.cfg.get("server_optimizer", "adam")
        if opt_name == "sgd":
            optimizer = torch.optim.SGD(self.head.parameters(), lr=lr,
                                         momentum=0.9)
        else:
            optimizer = torch.optim.Adam(self.head.parameters(), lr=lr)

        from torch.utils.data import WeightedRandomSampler
        ds = TensorDataset(self._replay_embs, self._replay_labs)
        sampler = WeightedRandomSampler(weights.tolist(), num_samples=len(ds),
                                         replacement=True)
        loader = DataLoader(ds, batch_size=256, sampler=sampler)

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

        return head_bytes

    # ------------------------------------------------------------------ #
    #  AO-FRL orchestration: histogram-driven budget allocation           #
    # ------------------------------------------------------------------ #
    def init_budgets(self, label_histograms: np.ndarray, T_base: int):
        """Round-1 budget allocation from client label histograms.

        label_histograms: int (n_clients, n_classes), sample count per (i, c).
        T_base: per-class total upload target across all clients.
        Returns: budget matrix (n_clients, n_classes).
        """
        self._T_base = T_base
        self._per_class_target = np.full(self.n_classes, T_base, dtype=np.int64)
        self._label_hist = np.asarray(label_histograms, dtype=np.int64)
        self._budget = allocate_budgets(self._label_hist, self._per_class_target)
        return self._budget

    def update_budgets_from_feedback(self, per_class_acc: np.ndarray,
                                     alpha: float = 1.0):
        """Re-balance budgets from aggregated per-class validation accuracy.

        per_class_acc: float (n_classes,) — global avg val acc per class
        (weighted across clients by sample count, computed by caller).
        Returns: new budget matrix (n_clients, n_classes).
        """
        self._per_class_target = update_per_class_target(
            per_class_acc, self._T_base, alpha=alpha)
        self._budget = allocate_budgets(self._label_hist,
                                        self._per_class_target)
        return self._budget

    def get_budgets(self):
        return self._budget

    def get_per_class_target(self):
        return self._per_class_target

    # ------------------------------------------------------------------ #
    #  Legacy orchestration (used by 10-class test + ablations)           #
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

    # ------------------------------------------------------------------ #
    #  FedAdam (Reddi et al. 2020): server-side Adam over client deltas   #
    # ------------------------------------------------------------------ #
    def init_fedadam(self, lr: float = 1e-3, beta1: float = 0.9,
                     beta2: float = 0.99, tau: float = 1e-3):
        """Initialize server-side Adam moment estimates."""
        self._fa_lr = lr
        self._fa_b1 = beta1
        self._fa_b2 = beta2
        self._fa_tau = tau
        self._fa_m = {n: torch.zeros_like(p)
                      for n, p in self.fedavg_head.named_parameters()}
        self._fa_v = {n: torch.zeros_like(p)
                      for n, p in self.fedavg_head.named_parameters()}

    def fedadam_aggregate(self, client_state_dicts: list,
                          client_sizes: list):
        """Aggregate via server-side Adam: average client deltas, apply Adam."""
        total = sum(client_sizes)
        if total == 0:
            return
        with torch.no_grad():
            for name, server_p in self.fedavg_head.named_parameters():
                avg_client = sum(
                    sd[name].float().to(server_p.device) * (n / total)
                    for sd, n in zip(client_state_dicts, client_sizes)
                )
                delta = avg_client - server_p          # pseudo-gradient
                self._fa_m[name] = (self._fa_b1 * self._fa_m[name]
                                    + (1 - self._fa_b1) * delta)
                self._fa_v[name] = (self._fa_b2 * self._fa_v[name]
                                    + (1 - self._fa_b2) * delta ** 2)
                server_p.add_(self._fa_lr * self._fa_m[name]
                              / (torch.sqrt(self._fa_v[name]) + self._fa_tau))

    def get_fedavg_head(self):
        return self.fedavg_head

    def get_head(self):
        return self.head
