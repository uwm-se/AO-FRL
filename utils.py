"""Utility functions: seeding, data partitioning, communication estimation."""

import math
import os
import random
import json
import numpy as np
import torch
from pathlib import Path


def set_seed(seed: int = 42):
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def dirichlet_partition(labels: np.ndarray, n_clients: int, alpha: float,
                        seed: int = 42):
    """Partition dataset indices into n_clients using Dirichlet label-skew.

    Returns: list of np.arrays, each containing sample indices for a client.
    """
    rng = np.random.default_rng(seed)
    n_classes = int(labels.max()) + 1
    class_indices = [np.where(labels == c)[0] for c in range(n_classes)]

    client_indices = [[] for _ in range(n_clients)]
    for c in range(n_classes):
        idx = class_indices[c].copy()
        rng.shuffle(idx)
        proportions = rng.dirichlet(np.repeat(alpha, n_clients))
        proportions = proportions / proportions.sum()
        splits = (np.cumsum(proportions) * len(idx)).astype(int)
        splits = np.clip(splits, 0, len(idx))
        parts = np.split(idx, splits[:-1])
        for i, part in enumerate(parts):
            client_indices[i].append(part)

    client_indices = [np.concatenate(parts) for parts in client_indices]
    # Shuffle within each client
    for idx in client_indices:
        rng.shuffle(idx)
    return client_indices


def split_train_val(indices: np.ndarray, val_ratio: float = 0.1,
                    seed: int = 42):
    """Split indices into train and val subsets."""
    rng = np.random.default_rng(seed)
    idx = indices.copy()
    rng.shuffle(idx)
    n_val = max(1, int(len(idx) * val_ratio))
    return idx[n_val:], idx[:n_val]


def split_decoder_pool(labels: np.ndarray, frac: float = 0.1,
                       seed: int = 42):
    """Split CIFAR-100 train indices into (decoder_pool, federated_pool).

    Per class, take `frac` fraction (e.g. 50/500 = 1/10) for decoder training.
    Federated pool is the disjoint complement — clients never see decoder data.
    Deterministic given seed; same seed → same split across train_autoencoder
    and run_experiment.

    Returns: (decoder_pool_idx, federated_pool_idx) — both np.int64 arrays.
    """
    rng = np.random.default_rng(seed)
    n_classes = int(labels.max()) + 1
    decoder_idx, federated_idx = [], []
    for c in range(n_classes):
        cls_idx = np.where(labels == c)[0]
        cls_idx = cls_idx.copy()
        rng.shuffle(cls_idx)
        n_dec = int(round(len(cls_idx) * frac))
        decoder_idx.append(cls_idx[:n_dec])
        federated_idx.append(cls_idx[n_dec:])
    return (np.concatenate(decoder_idx).astype(np.int64),
            np.concatenate(federated_idx).astype(np.int64))


def gaussian_dp_sigma(epsilon: float, delta: float, clip_C: float = 1.0):
    """Noise scale for the Gaussian mechanism satisfying (ε, δ)-DP.

    For an L2-clipped vector with sensitivity Δ = clip_C, releasing one sample
    via x + N(0, σ² I) is (ε, δ)-DP when
        σ ≥ clip_C * sqrt(2 ln(1.25/δ)) / ε.
    """
    if epsilon <= 0 or delta <= 0 or delta >= 1:
        raise ValueError(f"invalid (eps, delta) = ({epsilon}, {delta})")
    return clip_C * math.sqrt(2.0 * math.log(1.25 / delta)) / epsilon


def allocate_budgets(label_hist: np.ndarray,
                     per_class_target: np.ndarray) -> np.ndarray:
    """Distribute per-class upload budgets across clients to balance classes.

    Args:
        label_hist: int array (n_clients, n_classes) — # class-c samples held
            by client i.
        per_class_target: int array (n_classes,) — desired total uploads per
            class across all clients.

    Returns:
        budget: int array (n_clients, n_classes) — per-client per-class cap,
        with budget[i, c] ≤ label_hist[i, c] always. Two-phase allocation:
        (1) divide T_c evenly across holders, (2) redistribute leftover to
        clients with remaining capacity.
    """
    n_clients, n_classes = label_hist.shape
    budget = np.zeros_like(label_hist, dtype=np.int64)
    target = np.asarray(per_class_target, dtype=np.int64)

    for c in range(n_classes):
        holders = np.where(label_hist[:, c] > 0)[0]
        if len(holders) == 0:
            continue
        T = int(target[c])
        if T <= 0:
            continue

        # Phase 1: even split across holders, capped by what each has.
        per = T // len(holders)
        remainder = T - per * len(holders)
        for j, i in enumerate(holders):
            ask = per + (1 if j < remainder else 0)
            budget[i, c] = min(ask, label_hist[i, c])

        # Phase 2: redistribute deficit to holders with remaining capacity.
        for _ in range(8):  # bounded fixed-point
            deficit = T - int(budget[:, c].sum())
            if deficit <= 0:
                break
            slack = [i for i in holders if budget[i, c] < label_hist[i, c]]
            if not slack:
                break
            per_slack = max(1, deficit // len(slack))
            for i in slack:
                room = label_hist[i, c] - budget[i, c]
                give = min(per_slack, int(room), deficit)
                budget[i, c] += give
                deficit -= give
                if deficit == 0:
                    break

    return budget


def update_per_class_target(per_class_acc: np.ndarray, T_base: int,
                            alpha: float = 1.0) -> np.ndarray:
    """Boost upload budget for low-accuracy classes; total is preserved.

    weight_c = 1 + alpha * (1 - acc_c), then T_c = T_base * weight_c / mean(weight)
    so that mean(T_c) == T_base. acc clamped to [0, 1].
    """
    acc = np.clip(np.asarray(per_class_acc, dtype=np.float64), 0.0, 1.0)
    weight = 1.0 + alpha * (1.0 - acc)
    weight = weight / weight.mean()
    return np.maximum(1, np.round(T_base * weight)).astype(np.int64)


def estimate_comm_bytes(tensor_or_count, dtype_bytes: int = 4):
    """Estimate communication cost in bytes.
    tensor_or_count: a Tensor (use numel) or an int (number of scalars).
    """
    if isinstance(tensor_or_count, torch.Tensor):
        return tensor_or_count.numel() * dtype_bytes
    return int(tensor_or_count) * dtype_bytes


def load_skill_files(skills_dir: str = "skills") -> dict:
    """Load all skill .md files and return {filename: content}."""
    skills = {}
    p = Path(skills_dir)
    if p.exists():
        for f in sorted(p.glob("*.md")):
            skills[f.name] = f.read_text()
    return skills


def log_skills(skills: dict, logger=None):
    """Log loaded skill files for auditability."""
    msg = f"Loaded {len(skills)} skill files: {list(skills.keys())}"
    if logger:
        logger.info(msg)
    else:
        print(msg)
    for name, content in skills.items():
        header = content.split('\n')[0] if content else "(empty)"
        submsg = f"  {name}: {header}"
        if logger:
            logger.info(submsg)
        else:
            print(submsg)
