# Hyperparameter Reference

All methods share a frozen **ResNet-18** encoder (ImageNet-pretrained, output dim=512) and a trainable **MLPHead** (512 -> 256 -> 100, with BatchNorm + ReLU + Dropout(0.2)).

---

## 1. Shared (Global) Hyperparameters

| Parameter | CLI Flag | Default | Description |
|---|---|---|---|
| n_clients | `--n_clients` | 20 | Number of federated clients |
| alpha | `--alpha` | 0.3 | Dirichlet concentration for non-IID partitioning (lower = more heterogeneous) |
| val_ratio | `--val_ratio` | 0.1 | Fraction of each client's data reserved for local validation |
| rounds | `--rounds` | 100 | Number of communication rounds (FedAvg & AO-FRL) |
| head_hidden | `--head_hidden` | 256 | Hidden layer dimension of MLPHead |
| seed | `--seed` | 42 | Random seed for reproducibility |
| device | `--device` | auto | Compute device (auto / cpu / cuda) |

---

## 2. Centralized (Upper Bound)

Trains a single MLPHead on **all** training embeddings — no federation, no privacy. Serves as the theoretical performance ceiling.

| Parameter | CLI Flag | Default | Description |
|---|---|---|---|
| centralized_epochs | `--centralized_epochs` | 50 | Total training epochs |
| server_lr | `--server_lr` | 1e-3 | Learning rate (Adam optimizer) |
| batch_size | *(hardcoded)* | 256 | Training batch size |

> Epochs are mapped to "rounds" on the x-axis for plot comparability: `rounds_per_epoch = rounds / centralized_epochs`.
> Communication cost is always 0.

---

## 3. FedAvg (Baseline)

Standard Federated Averaging — each client trains a local copy of MLPHead via SGD, then the server performs weighted averaging of parameters.

| Parameter | CLI Flag | Default | Description |
|---|---|---|---|
| local_epochs | `--local_epochs` | 3 | Local SGD epochs per client per round |
| fedavg_lr | `--fedavg_lr` | 1e-3 | Client-side SGD learning rate |
| batch_size | `--batch_size` | 64 | Client-side training batch size |

**Communication per round:**
- Upload: each client sends full head state_dict (`n_params * 4` bytes)
- Download: server broadcasts aggregated head to all clients (`n_params * 4 * n_clients` bytes)

> Note: This is "FedAvg on head only" — the encoder is frozen and embeddings are precomputed. This differs from the original FedAvg paper which trains the full model.

---

## 4. AO-FRL (Agent-Orchestrated Federated Representation Learning)

Clients upload **privacy-gated embeddings** to the server, which trains the head centrally. An orchestration loop adaptively adjusts per-client upload budgets, noise levels, and augmentation modes.

### 4a. Privacy Pipeline

| Parameter | CLI Flag | Default | Description |
|---|---|---|---|
| sigma | `--sigma` | 0.02 | Gaussian noise scale added to embeddings (DP mechanism) |
| clip_C | `--clip_C` | 1.0 | L2 clipping bound for embeddings before noise addition |
| tau_high | `--tau_high` | 0.95 | Cosine similarity threshold (legacy, used in agent classes) |
| tau_percentile | `--tau_percentile` | 0.15 | Privacy gate: reject top fraction most similar to class prototype |
| tau_min | `--tau_min` | 0.5 | Minimum cosine similarity threshold (floor for adaptive gate) |
| upload_budget | `--upload_budget` | 500 | Max embeddings per client per round |
| n_views | `--n_views` | 2 | Number of augmented views per embedding (multi-view noise injection) |

**Privacy gate logic (per class):**
1. Compute class prototype (mean of L2-normalized embeddings)
2. For each candidate embedding, compute cosine similarity to its class prototype
3. Set per-class threshold = `max(percentile(sims, (1 - tau_percentile) * 100), tau_min)`
4. Reject embeddings with similarity > threshold
5. Accept remaining up to `upload_budget`

### 4b. Server-Side Head Training

| Parameter | CLI Flag | Default | Description |
|---|---|---|---|
| server_lr | `--server_lr` | 1e-3 | Base learning rate for server head optimizer |
| server_optimizer | `--server_optimizer` | adam | Optimizer type (adam / sgd) |
| server_train_epochs | `--server_train_epochs` | 3 | Training epochs per round on collected embeddings |
| server_lr_decay | `--server_lr_decay` | 0.98 | Per-round multiplicative LR decay |
| server_lr_min | `--server_lr_min` | 1e-4 | Minimum LR floor |
| replay_decay | `--replay_decay` | 0.995 | Exponential decay factor for replay buffer sample weights |
| replay_min_weight | `--replay_min_weight` | 0.3 | Minimum weight for oldest replay buffer samples |

**Replay buffer:** Embeddings from previous rounds are kept (up to 50,000) and resampled with age-based exponential decay weighting, so noise from different rounds averages out over time.

**Effective LR at round `r`:** `max(server_lr * server_lr_decay^(r-1), server_lr_min)`

### 4c. Adaptive Hooks (Client-Side)

Three hooks dynamically adjust client behavior based on runtime conditions:

| Hook | Trigger | Effect |
|---|---|---|
| `low_data_hook` | Any local class has < `low_data_k` (default 10) samples | Switch to conservative augmentation |
| `high_risk_hook` | Rejection ratio > `high_risk_r` (default 0.30) | sigma *= 1.5 (capped at 0.5), budget //= 2 (min 50) |
| `drift_hook` | Validation accuracy declines for 2+ consecutive rounds | budget *= 1.3 |

| Parameter | CLI Flag | Default | Description |
|---|---|---|---|
| low_data_k | `--low_data_k` | 10 | Threshold for low-data hook |
| high_risk_r | `--high_risk_r` | 0.30 | Rejection ratio threshold for high-risk hook |

### 4d. Server Orchestration

The server also adjusts per-client parameters each round via `orchestrate()`:

- **Budget adjustment**: Clients holding rare classes (relative to uniform target) get higher upload budgets (`base_budget * (1 + rarity_score)`)
- **High-risk override**: If `reject_ratio > high_risk_r` → sigma *= 1.5, budget //= 2, augmentation = conservative
- **Low-data override**: If any class has < `low_data_k` uploaded samples → budget *= 1.2, augmentation = conservative

### 4e. Communication per Round

- Upload: variable-size gated embeddings (`n_uploaded * 512 * 4` bytes) + small summary (~420 bytes per client)
- Download: server broadcasts trained head to all clients (`n_params * 4 * n_clients` bytes)

---

## Quick Reference: Parameter → Method Mapping

| Parameter | Centralized | FedAvg | AO-FRL |
|---|:---:|:---:|:---:|
| n_clients / alpha / rounds | x | x | x |
| head_hidden | x | x | x |
| centralized_epochs | x | | |
| local_epochs / fedavg_lr / batch_size | | x | |
| server_lr / server_optimizer | x | | x |
| server_train_epochs | | | x |
| server_lr_decay / server_lr_min | | | x |
| replay_decay / replay_min_weight | | | x |
| sigma / clip_C / tau_* | | | x |
| upload_budget / n_views | | | x |
| low_data_k / high_risk_r | | | x |
