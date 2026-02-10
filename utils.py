"""Utility functions: seeding, data partitioning, communication estimation."""

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
