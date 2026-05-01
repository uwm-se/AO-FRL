# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**AO-FRL** (Agent-Orchestrated Federated Representation Learning): a federated learning experiment comparing **FedAvg** and **Centralized** baselines against the proposed **AO-FRL** method. Evaluated on CIFAR-100 with non-IID Dirichlet partitioning, using a frozen ImageNet-pretrained ResNet-18 encoder (512-dim) with a trainable MLP head.

## Running Experiments

```bash
# Default run (all three methods: fedavg, ao-frl, centralized)
python run_experiment.py --rounds 100 --n_clients 20 --alpha 0.3

# Quick test
python run_experiment.py --rounds 10 --n_clients 5 --alpha 0.5 --seed 123

# Run one or more methods (--methods is nargs=+; valid values: fedavg, ao-frl, centralized)
python run_experiment.py --methods fedavg
python run_experiment.py --methods ao-frl centralized

# 10-class subset test using full agent classes (hardcoded config in the script)
python run_test_10classes.py
```

Key CLI parameters (defaults in parens): `--alpha` 0.3 (Dirichlet non-IID degree), `--sigma` 0.02 (DP noise scale), `--tau_percentile` 0.15 / `--tau_min` 0.5 (adaptive gate; see "AO-FRL Privacy Pipeline"), `--upload_budget` 500 (max embeddings/client/round), `--server_lr_decay` 0.98 (per-round LR decay), `--replay_decay` 0.995 (replay buffer age weighting), `--centralized_epochs` 50 (centralized total epochs, stretched across `--rounds` for plotting). Note: README.md lists some stale defaults — trust the argparse block in [run_experiment.py:35-75](run_experiment.py#L35-L75).

### Ablation Studies

Each ablation `import run_experiment as base` and subclasses/swaps `ProposedClient` (or `ServerAgent`) — they only run the AO-FRL variant, not FedAvg or Centralized. Compare against `results/AO-FRL_rounds.csv` from a baseline run.

```bash
bash run_all_ablations.sh  # runs both ablations sequentially

python run_ablation_no_orchestration.py --rounds 100 --n_clients 20 --alpha 0.3
python run_ablation_no_privacy_gate.py  --rounds 100 --n_clients 20 --alpha 0.3

# Plot ablation comparisons
python plot_ablation_results.py \
    --baseline results \
    --no_orchestration ablation_results/no_orchestration \
    --no_privacy_gate ablation_results/no_privacy_gate \
    --output ablation_results/figures
```

### Visualization

```bash
python plot_all_figures.py         # generate all paper figures from results/
python plot_convergence_focus.py   # convergence-focused comparison
```

## Dependencies

```bash
pip install torch torchvision scikit-learn matplotlib numpy
```

CIFAR-100 auto-downloads to `data/` on first run.

## Architecture

### Dual Implementation Pattern

Two parallel implementations of client logic exist for different use cases. **Default to editing `run_experiment.py`** for any change that should affect the headline results — `agents/` is only exercised by `run_test_10classes.py`.

1. **`run_experiment.py`** (primary) — `FedAvgClient`, `ProposedClient`, `FastEvaluator`. These operate on **embeddings precomputed once at startup** from the frozen encoder, making all rounds fast. This is the script all ablations and headline plots are built on.

2. **`agents/`** — `ClientAgent`, `EvaluatorAgent`, plus shared `ServerAgent`/`MLPHead` from `agents/server_agent.py`. Full agents with encoder access and augmentation pipelines, used by `run_test_10classes.py`. `ClientAgent` caches embeddings after first extraction per round.

`ServerAgent` and `MLPHead` are the only code shared across both paths — when changing head architecture or server-side training, the change applies to both.

`_TransformSubset` (defined in `agents/client_agent.py`) is imported by both `run_experiment.py` and `agents/evaluator_agent.py` — it applies transforms to PIL dataset subsets at load time.

### ServerAgent (`agents/server_agent.py`)

Holds two separate `MLPHead` instances:
- `self.head` — AO-FRL, trained server-side on collected embeddings via `train_head()`
- `self.fedavg_head` — FedAvg, updated by `fedavg_aggregate()` (weighted average of client state dicts)

`train_head()` maintains a **replay buffer** across rounds so Gaussian noise from different rounds averages out. Uses `WeightedRandomSampler` with exponential age decay (`replay_decay`) and per-round LR decay (`server_lr_decay`).

`orchestrate()` computes per-class label gaps vs. uniform target distribution and returns per-client instructions adjusting `upload_budget`, `sigma`, and `augmentation_mode`.

### MLPHead Architecture

`Linear(512, 256) → BatchNorm1d → ReLU → Dropout(0.2) → Linear(256, 100)`

### Centralized Baseline

`run_centralized()` is *not* federated training — it trains a single `MLPHead` on all clients' precomputed embeddings concatenated together, treated as an upper-bound reference. `--centralized_epochs` (default 50) is stretched across `--rounds` so per-round CSVs align for plotting (e.g., 50 epochs over 100 rounds = 1 epoch per 2 rounds).

### AO-FRL Privacy Pipeline

Per round, each client's embeddings go through:
1. **L2 clipping** at bound `clip_C`
2. **Gaussian noise** (scale `sigma`)
3. **Adaptive percentile privacy gate** — per class, reject the top `tau_percentile` fraction most similar to class prototype; threshold floored at `tau_min`. The `tau_high` arg (default 0.95) is a legacy fixed-threshold fallback and is *not* used on the active code path — keep this in mind when tuning.

Three adaptive hooks run after gate (both in `ProposedClient._apply_hooks()` and `ClientAgent._apply_hooks()`):
- `low_data_hook`: conservative augmentation when any class has < `low_data_k` samples
- `high_risk_hook`: sigma × 1.5, budget ÷ 2 when rejection rate > `high_risk_r`
- `drift_hook`: budget × 1.3 when validation accuracy has declined for ≥ 2 consecutive rounds

### A2A Protocol Layer (`a2a/`)

In-process agent-to-agent communication for audit logging only — all actual computation is synchronous Python. `A2ABus` records tasks (send/complete/fail) to a JSON log at `results/a2a_communication.json`.

### Results Directory

`results/` holds per-method round CSVs (`FedAvg_rounds.csv`, `AO-FRL_rounds.csv`, `Centralized_rounds.csv` — CamelCase, written by `evaluator.save_csv(method_name)`), a `final_summary.json` (best/final accuracy per method), AO-FRL-only orchestration outputs (`server_instructions.csv` + `.png`), the full A2A audit log (`a2a_communication.json`), and comparison plots. Ablations write to `ablation_results/{no_orchestration,no_privacy_gate}/` with the same per-method CSV layout, and `ablation_results/figures/` holds comparison plots produced by `plot_ablation_results.py`.

### `utils.py`

Shared utilities: `set_seed`, `dirichlet_partition`, `split_train_val`, `estimate_comm_bytes`, `load_skill_files`, `log_skills`. Communication cost is always float32 (4 bytes/scalar).

### `skills/*.md`

`server_agent.md`, `client_agent.md`, `evaluator_agent.md` — loaded at startup via `load_skill_files()` for audit logging. Not executed.
