# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**AO-FRL** (Agent-Orchestrated Federated Representation Learning): a federated learning experiment that operates on the **upload channel** rather than gradients. Clients send privacy-protected embeddings produced by a shared frozen encoder under per-record (ε, δ)-DP; a server agent dynamically allocates per-class upload budgets to address non-IID heterogeneity and long-tail fairness. Compared against FedAvg, FedProx, FedAdam, and a Centralized upper-bound on CIFAR-100 / CIFAR-10 / SVHN with Dirichlet non-IID partitioning.

## Running Experiments

```bash
# All five methods in one process (shares the encoder precompute)
python run_experiment.py --rounds 100 --n_clients 20 --alpha 0.3 \
    --methods centralized fedavg fedprox fedadam ao-frl

# AO-FRL only at a specific σ
python run_experiment.py --methods ao-frl --sigma 0.02 --rounds 60

# Headline reproduction: 8-method comparison (4 baselines + 4 σ levels of AO-FRL: 0, 0.005, 0.02, 0.05)
bash run_full_comparison.sh

# σ sweep for AO-FRL only (strong / medium / weak)
bash run_dp_sweep.sh

# 10-class subset test using full agent classes (hardcoded config in the script)
python run_test_10classes.py
```

Valid `--methods` values: `centralized`, `fedavg`, `fedprox`, `fedadam`, `ao-frl` (nargs=+).

Trust the argparse block in [run_experiment.py:36-140](run_experiment.py#L36-L140) over any defaults table — README.md and this file may lag.

### Key CLI Parameters

- DP knobs: `--epsilon` 2.0 / `--delta` 1e-5 → `σ` is **derived** via `gaussian_dp_sigma(ε, δ, clip_C)` ([utils.py:87](utils.py#L87)). Pass `--sigma` to override the derived value (most reproduction scripts do).
- Budget orchestration: `--per_class_target` 500 (T_base) — global per-class total upload target per round, allocated across clients by [utils.py:99 `allocate_budgets`](utils.py#L99). Re-balanced each round by [utils.py:154 `update_per_class_target`](utils.py#L154) which boosts low-accuracy classes (`--feedback_alpha`).
- Head sync: `--head_sync_every` 5 (server pushes head down to clients for client-side per-class val acc feedback).
- Replay buffer: `--replay_max` 500_000, `--replay_decay` 0.995, `--replay_min_weight` 0.3.
- Server head training: `--server_lr_decay` 0.98, `--server_lr_min` 1e-4, `--server_optimizer` adam.
- Encoder/decoder: `--encoder_weights models/encoder_finetuned.pt` (falls back to ImageNet-pretrained if file missing), `--decoder_weights models/decoder.pt`, `--inversion_n_samples` 200.
- Multi-dataset: `--dataset cifar100|cifar10|svhn`.
- Partial participation: `--clients_per_round K`. No-budget mode: `--random_upload_fraction F`.
- Early stopping (AO-FRL only): `--early_stop_patience` 10.
- Centralized: `--centralized_epochs` 50 (stretched across `--rounds` for plot alignment).

### Encoder fine-tuning + PSNR privacy evaluation

```bash
# Train reconstruction decoder used by PSNR privacy eval (frozen encoder)
python train_autoencoder.py --freeze_encoder --epochs 30 \
    --decoder_out models/decoder_frozen_enc.pt

# Encoder fine-tuning ablations
python train_encoder_supervised.py --epochs 30   # Scheme A: CE + L2 anchor
python train_encoder_supcon.py     --epochs 30   # Scheme B: SupCon

# PSNR privacy evaluation: 3 encoders × 3 σ + clean reference
python eval_psnr_sweep.py
python eval_normalize_ablation.py    # isolates L2-normalize contribution
python eval_decoder_privacy.py
python eval_encoder_privacy.py

# Diagnostic: embedding magnitude probe
python probe_encoder_norms.py
```

### Legacy Ablations — broken on current code path

[run_ablation_no_orchestration.py](run_ablation_no_orchestration.py) and [run_ablation_no_privacy_gate.py](run_ablation_no_privacy_gate.py) target the **legacy cosine-gate pipeline**: they call `client.extract_gated_embeddings(...)`, but `ProposedClient` now exposes `extract_dp_embeddings(...)` instead ([run_experiment.py:323](run_experiment.py#L323)). Don't run these as-is — port the ablation idea into a new script that subclasses the current `ProposedClient` / `ServerAgent` if needed. `run_all_ablations.sh` is similarly stale.

### Visualization

`plot_*.py` scripts read CSVs from `results/` (or a custom dir) and write PNGs. The most-used:

- [plot_all_figures.py](plot_all_figures.py) — paper figure bundle from `results/`
- [plot_full_comparison.py](plot_full_comparison.py) — 8-method comparison from `results/full_compare/`
- [plot_dp_comparison.py](plot_dp_comparison.py) — σ sweep
- [plot_dataset_compare.py](plot_dataset_compare.py) — cifar100 / cifar10 / svhn transfer
- [plot_encoder_compare60.py](plot_encoder_compare60.py), [plot_encoder_table.py](plot_encoder_table.py) — encoder fine-tuning ablation
- [plot_per_class_curves.py](plot_per_class_curves.py), [plot_per_class_compare.py](plot_per_class_compare.py) — long-tail fairness
- [plot_convergence_focus.py](plot_convergence_focus.py), [plot_baselines60.py](plot_baselines60.py)
- [analyze_bottom_classes.py](analyze_bottom_classes.py) — bottom-10 / top-10 accuracy & partition stats

## Dependencies

```bash
pip install torch torchvision scikit-learn matplotlib numpy
```

CIFAR-100 / CIFAR-10 / SVHN auto-download to `data/` on first run.

## Architecture

### Dual Implementation Pattern

Two parallel implementations of client logic exist. **Default to editing [run_experiment.py](run_experiment.py)** for any change that should affect headline results — `agents/` is only exercised by [run_test_10classes.py](run_test_10classes.py).

1. **[run_experiment.py](run_experiment.py)** (primary) — `FedAvgClient`, `FedProxClient`, `ProposedClient`, `FastEvaluator`. Operates on **embeddings precomputed once at startup** from the frozen encoder, making rounds cheap. All headline experiments and σ sweeps use this path.

2. **`agents/`** — `ClientAgent`, `EvaluatorAgent`, plus shared `ServerAgent`/`MLPHead`. Full agents with encoder access and augmentation pipelines, used only by `run_test_10classes.py`. `ClientAgent` caches embeddings after first extraction per round.

`ServerAgent` and `MLPHead` ([agents/server_agent.py](agents/server_agent.py)) are shared across both paths — head-architecture or server-side-training changes apply to both. `_TransformSubset` (defined in [agents/client_agent.py](agents/client_agent.py)) is imported by both paths.

### MLPHead

`Linear(512, 256) → LayerNorm → ReLU → Dropout(0.2) → Linear(256, n_classes)` ([agents/server_agent.py:10-30](agents/server_agent.py#L10-L30)).

**LayerNorm, not BatchNorm**: the head is trained on DP-noised embeddings (`||z||≈√(1+σ²·d)`) but evaluated on clean unit-norm test embeddings. BatchNorm would capture the noisy training distribution in its running stats and corrupt clean inference. If you replace LayerNorm, retain that train/test decoupling property.

### ServerAgent

Holds two separate `MLPHead` instances:
- `self.head` — AO-FRL, trained server-side on collected embeddings via `train_head()`.
- `self.fedavg_head` — FedAvg/FedProx/FedAdam, updated by `fedavg_aggregate()` (weighted state-dict average) or by FedAdam's server-side optimizer (`init_fedadam`).

`train_head()` maintains a **flat replay buffer** (FIFO truncation at `replay_max`) across rounds so independent Gaussian noise from different rounds averages out across replays. Uses `WeightedRandomSampler` with exponential age decay (`replay_decay`, floored at `replay_min_weight`) and per-round LR decay (`server_lr_decay`, floored at `server_lr_min`).

**Per-class budget orchestration** (active path): `init_budgets()` + `update_per_class_target()` + `allocate_budgets()`. The legacy `ServerAgent.orchestrate()` method (sigma/aug-mode adjustment + cosine-gate hooks) is **not called** by `run_proposed()` — only by the broken legacy ablation scripts.

### AO-FRL Privacy Pipeline (current, active path)

Per round, each active client transforms data via `ProposedClient.extract_dp_embeddings()` ([run_experiment.py:323](run_experiment.py#L323)):
1. Sample per-class up to server-assigned budget (or `random_upload_fraction × N` uniformly if set).
2. **L2 clip** at `clip_C` (default 1.0) — bounds per-record sensitivity Δ for the Gaussian mechanism.
3. **Gaussian noise** `z ← z + N(0, σ²·I)` with σ either user-provided or derived from (ε, δ).

Embeddings are precomputed once with `F.normalize` at startup, so the L2 clip is operating on already-unit-norm vectors — `clip_C` effectively scales the noise budget, not the geometry.

### Legacy Privacy Mechanisms (still in argparse, NOT on active path)

These flags exist for legacy ablation comparison. The active AO-FRL path ignores them:
- `--tau_high`, `--tau_percentile`, `--tau_min` — old cosine-similarity privacy gate
- `--upload_budget` — replaced by `--per_class_target`
- `--n_views` — replaced by per-class budget sampling
- `--high_risk_r` — old hook; removed
- `--legacy_hooks` — opt-in flag re-enabling client-side `low_data` (force full upload of any class with `0 < hist[c] < low_data_k`) and `drift` (boost all per-class budgets ×1.3 after two consecutive val-acc drops). Off by default.

When tuning AO-FRL, **don't touch the legacy flags** unless you're explicitly running a legacy-vs-current comparison.

### Centralized Baseline

[run_centralized()](run_experiment.py#L662) is *not* federated — it trains a single `MLPHead` on all clients' precomputed embeddings concatenated, treated as an upper bound. `--centralized_epochs` (default 50) is stretched across `--rounds` so per-round CSVs align for plotting (e.g., 50 epochs over 100 rounds = 1 epoch per 2 rounds).

### A2A Protocol Layer (`a2a/`)

In-process agent-to-agent messaging used **only for audit logging** — all actual computation is synchronous Python. `A2ABus` records send/complete/fail tasks to `results/a2a_communication.json`.

### Results Layout

`results/` (or `--results_dir`):
- Per-method round CSVs in CamelCase: `FedAvg_rounds.csv`, `FedProx_rounds.csv`, `FedAdam_rounds.csv`, `AO-FRL_rounds.csv`, `Centralized_rounds.csv` (written by `evaluator.save_csv(method_name)`).
- `final_summary.json` — best/final accuracy per method.
- `*_per_class.npy` — per-round, per-class accuracy matrices.
- AO-FRL-only: `server_instructions.csv` + `.png`.
- `a2a_communication.json` — full audit log.
- `experiment.log` — full stdout/stderr.

`results/full_compare/`, `results/dp_<tag>_sigma*/`, `results/dataset_<name>/`, `results/encoder_compare/` etc. are produced by the orchestration shell scripts.

### `utils.py`

Shared utilities:
- `set_seed`, `dirichlet_partition`, `split_train_val`, `split_decoder_pool`
- `gaussian_dp_sigma(ε, δ, clip_C)` — DP calibration
- `allocate_budgets(label_hist, per_class_target)` — two-phase greedy budget allocation across clients (even-split + deficit redistribution)
- `update_per_class_target(per_class_acc, T_base, alpha)` — boost low-accuracy classes while preserving total
- `estimate_comm_bytes(...)` — always float32 = 4 bytes/scalar
- `load_skill_files`, `log_skills`

### `skills/*.md`

`server_agent.md`, `client_agent.md`, `evaluator_agent.md` — loaded at startup via `load_skill_files()` for audit logging. **Not executed.**

### `models/`

Trained weights checked into the repo (used by inversion / PSNR eval and encoder ablations):
- `decoder.pt`, `decoder_frozen_enc.pt`, `decoder_frozen_enc_no_normalize.pt`
- `encoder_supervised.pt`, `encoder_supcon.pt`
- `decoder.py` — reconstruction decoder architecture
