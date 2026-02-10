# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Federated learning experiment comparing **FedAvg** (baseline) against a **proposed agent-orchestrated privacy-gated representation sharing** method. Evaluated on CIFAR-100 with non-IID data partitioning (Dirichlet distribution). Uses a frozen ResNet-18 encoder with a trainable MLP classification head.

## Running Experiments

```bash
# Default run (100 rounds, 20 clients, alpha=0.3)
python run_experiment.py --rounds 100 --n_clients 20 --alpha 0.3

# Quick test run
python run_experiment.py --rounds 10 --n_clients 5 --alpha 0.5 --seed 123

# Run only one method
python run_experiment.py --methods fedavg
python run_experiment.py --methods proposed

# 10-class subset test using full agent classes (hardcoded config, no CLI args)
python run_test_10classes.py

# Key parameters
#   --alpha          Dirichlet concentration (lower = more non-IID)
#   --sigma          DP noise scale for privacy gating
#   --tau_high       Cosine similarity threshold for privacy gate rejection
#   --upload_budget  Max embeddings per client per round
```

Results go to `results/` (main) or `test_results/` (10-class test). Outputs: CSV per method, `final_summary.json`, comparison plots (`acc_vs_rounds.png`, `comm_vs_acc.png`).

## Dependencies

PyTorch, torchvision, scikit-learn, matplotlib, numpy. CIFAR-100 auto-downloads to `data/`.

## Architecture

### Dual implementation pattern

There are **two parallel implementations** of the client/evaluator logic:

1. **`run_experiment.py`** — contains lightweight classes (`FedAvgClient`, `ProposedClient`, `FastEvaluator`) that operate on **precomputed embeddings** for speed. Embeddings are computed once from the frozen ResNet-18 encoder at startup, then reused across all rounds and methods. This is the main experiment script.

2. **`agents/`** — full-featured agent classes (`ServerAgent`, `ClientAgent`, `EvaluatorAgent`) with encoder access, augmentation pipelines, and embedding caching. Used by `run_test_10classes.py`. `ClientAgent` caches clean embeddings after first extraction; `EvaluatorAgent` caches test embeddings similarly.

Both share `ServerAgent` and `MLPHead` from `agents/server_agent.py`. The `_TransformSubset` helper in `agents/client_agent.py` is imported by both `run_experiment.py` and `agents/evaluator_agent.py`.

### ServerAgent (`agents/server_agent.py`)

Maintains two separate heads: `self.head` (proposed method, trained on collected embeddings) and `self.fedavg_head` (FedAvg, weighted-averaged from client state dicts). The `orchestrate()` method analyzes client summaries, computes per-class label gaps against a uniform target, and generates per-client instructions adjusting budget/noise/augmentation.

### MLPHead

Shared between both methods. Architecture: Linear(512, 256) → BatchNorm → ReLU → Dropout(0.2) → Linear(256, 100). Input dim=512 (ResNet-18 features), output dim=100 (CIFAR-100 classes).

### Privacy pipeline (proposed method)

Client embeddings go through 3 stages before upload:
1. **L2 clipping** to bound `C`
2. **Gaussian noise** addition (scale `sigma`)
3. **Privacy gate** — reject samples with cosine similarity > `tau_high` to class prototype

Three adaptive hooks modify client behavior across rounds:
- `low_data_hook`: switches to conservative augmentation when any class has < `k` samples
- `high_risk_hook`: increases noise and halves upload budget when rejection rate > `r`
- `drift_hook`: increases upload budget by 30% when validation accuracy declines for consecutive rounds

### Skill files (`skills/*.md`)

Markdown files describing each agent's role and constraints. Loaded at startup via `load_skill_files()` for auditability logging. Not executed — they define the agent "contract".

### Communication cost tracking

Both methods track per-round and cumulative bytes transferred (float32 = 4 bytes/scalar). FedAvg sends full head parameters; proposed method sends variable-size gated embeddings + small summaries.
