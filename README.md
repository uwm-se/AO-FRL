# AO-FRL: Agent-Orchestrated Federated Representation Learning

A federated learning framework that operates on the **upload channel** rather than the gradient channel: clients send privacy-protected representations of their own data — produced by a shared frozen encoder under per-record differential privacy — and a server agent dynamically allocates per-class upload budgets to address non-IID heterogeneity and long-tail fairness.

## Headline Results (CIFAR-100, 20 non-IID clients, Dirichlet α=0.3, 60 rounds)

| Method | Best Acc | Best Round | Notes |
|---|---|---|---|
| FedAvg | 56.60% | R60 | still climbing |
| FedProx (μ=1.0) | 54.98% | R60 | still climbing |
| FedAdam | 61.19% | R60 | still climbing |
| **AO-FRL (σ=0.02)** | **66.66%** | **R8** | per-release ε≈265, δ=10⁻⁶ |
| AO-FRL (σ=0.005) | 66.13% | R3 | total comm 1.22 GB (lower than FedAvg) |
| AO-FRL (σ=0.05) | 63.21% | R40 | strong DP |

- **+10.06 pp over FedAvg, +11.68 pp over FedProx, +5.47 pp over FedAdam**
- A single AO-FRL round at σ=0.02 already surpasses FedAvg's 60-round plateau
- Per-class budget allocation lifts **bottom-10 accuracy by +11.0 pp** and the worst class from **4% to 29%**, while preserving top-10 (+1.5 pp)
- The same hyperparameters transfer cleanly to **CIFAR-10** and **SVHN**

## Method

### Architecture

- **Encoder** (frozen, shared): ImageNet-pretrained ResNet-18 → 512-dim embedding
- **Server-side classifier** (trained): two-layer MLP head, `Linear(512, 256) → LayerNorm → ReLU → Dropout(0.2) → Linear(256, 100)`
- Clients perform **no local model training** — only encoder forward + DP transformation per round

LayerNorm (instead of BatchNorm) is used in the head: it is trained on DP-noised embeddings (`||z|| ≈ √(1+σ²·d)`) but evaluated on clean unit-norm test embeddings. BatchNorm would capture the noisy training distribution in its running stats and corrupt clean inference.

### Client-side DP pipeline

Each round, every active client transforms its local data into uploadable embeddings via:

1. **Encoder**: image (224×224×3) → 512-dim raw embedding
2. **L2 normalize + clip** at bound `clip_C` (default 1.0): bounds per-record sensitivity so the Gaussian mechanism's (ε, δ) guarantee holds
3. **Gaussian noise**: `z ← z + N(0, σ²·I)` calibrated to a target (ε, δ) per release

Empirically, this pipeline holds reconstruction PSNR (decoder-based adversary trained on a public auxiliary holdout) at **11.80 dB median at σ=0.02** on the CIFAR-100 test set — well below the visual-recognition threshold. An ablation isolates the encoder mapping itself as the dominant single-release privacy mechanism (Δ vs no L2 normalize ≈ 0 dB).

### Server agent

Coordinates clients via in-process A2A messaging and runs three responsibilities each round:

- **Aggregation**: receives per-client noisy embeddings + label histograms; appends to a flat per-sample replay buffer with FIFO truncation at `replay_max` (default 500K samples).
- **Head training**: trains the MLPHead via `WeightedRandomSampler` over the replay buffer with weights `0.995^age` (floored at 0.3), so independent DP noise from different rounds averages out across replays. Adam (default) or SGD+momentum, with per-round LR decay (`server_lr_decay`, default 0.98).
- **Per-class upload-budget orchestration**: computes per-class label gaps against a uniform target distribution and assigns each client a per-class upload budget for the next round, redistributing communication toward classes the model finds harder to learn (two-phase greedy allocation + deficit redistribution; see `utils.allocate_budgets`).

## Installation

```bash
pip install torch torchvision scikit-learn matplotlib numpy
```

CIFAR-100 / CIFAR-10 / SVHN auto-download to `data/` on first run.

## Reproducing the Main Experiments

### 1. Train the reconstruction decoder (used for PSNR privacy evaluation)

```bash
python train_autoencoder.py --freeze_encoder --epochs 30 \
    --decoder_out models/decoder_frozen_enc.pt
```

The trained weights are also tracked in this repo (`models/decoder_frozen_enc.pt`) so the privacy evaluation can run without retraining.

### 2. Run the 60-round federated baselines + AO-FRL

```bash
# 7-method comparison: Centralized + FedAvg + FedProx + FedAdam + AO-FRL × 3σ
bash run_full_comparison.sh

# Or just one AO-FRL operating point
python run_experiment.py \
    --rounds 60 --n_clients 20 --alpha 0.3 \
    --sigma 0.02 --upload_budget 500 \
    --early_stop_patience 10 \
    --methods ao-frl
```

### 3. Run the σ sweep for AO-FRL

```bash
bash run_dp_sweep.sh
```

### 4. Cross-dataset transfer

```bash
python run_experiment.py --dataset cifar10 --methods ao-frl ...
python run_experiment.py --dataset svhn    --methods ao-frl ...
python plot_dataset_compare.py
```

### 5. Encoder fine-tuning ablation (Scheme A / Scheme B)

```bash
python train_encoder_supervised.py --epochs 30   # Scheme A: CE + L2 anchor
python train_encoder_supcon.py     --epochs 30   # Scheme B: SupCon
# Then run AO-FRL with --encoder_weights pointing at each, and compare
python plot_encoder_compare60.py
```

### 6. PSNR privacy evaluation

```bash
python eval_psnr_sweep.py            # 3 encoders × 3 σ + clean reference
python eval_normalize_ablation.py    # isolates L2-normalize contribution (≈ 0 dB)
```

## Key CLI Parameters

| Parameter | Description | Default |
|---|---|---|
| `--dataset` | `cifar100` / `cifar10` / `svhn` | `cifar100` |
| `--rounds` | Federated rounds (with early stopping) | 60 |
| `--n_clients` | Number of clients | 20 |
| `--alpha` | Dirichlet concentration (lower = more non-IID) | 0.3 |
| `--sigma` | DP Gaussian noise scale | 0.02 |
| `--clip_C` | L2 clip bound for per-record sensitivity | 1.0 |
| `--upload_budget` | Per-client per-class baseline budget | 500 |
| `--clients_per_round` | Partial participation cohort size (0 = full) | 0 |
| `--replay_max` | Server replay buffer capacity | 500000 |
| `--replay_decay` | Per-round age-decay for sampling weights | 0.995 |
| `--server_lr_decay` | Per-round LR decay | 0.98 |
| `--early_stop_patience` | Rounds without improvement before stopping | 10 |
| `--methods` | Subset of `{centralized, fedavg, fedprox, fedadam, ao-frl}` | all |

Trust the argparse block in `run_experiment.py` over this table for the exact set of flags.

## Project Structure

```
.
├── agents/
│   ├── client_agent.py         # ClientAgent (full-encoder path; agents/ exercise: run_test_10classes.py)
│   ├── server_agent.py         # ServerAgent + MLPHead (shared by all paths)
│   └── evaluator_agent.py      # EvaluatorAgent
├── a2a/                        # In-process agent-to-agent audit log
├── models/
│   ├── decoder.py              # Reconstruction decoder (for PSNR privacy eval)
│   └── decoder_frozen_enc.pt   # Trained decoder weights (frozen encoder)
├── run_experiment.py           # Primary driver (precomputed-embedding path)
├── train_autoencoder.py        # Decoder training
├── train_encoder_supervised.py # Scheme A encoder fine-tune (CE + L2 anchor)
├── train_encoder_supcon.py     # Scheme B encoder fine-tune (SupCon)
├── eval_psnr_sweep.py          # PSNR privacy evaluation across encoders × σ
├── eval_normalize_ablation.py  # L2-normalize vs encoder isolation
├── probe_encoder_norms.py      # Embedding magnitude diagnostic
├── analyze_bottom_classes.py   # Bottom-10 / top-10 class + partition analysis
├── plot_*.py                   # Figures
├── run_full_comparison.sh      # 7-method baseline orchestrator
├── run_dp_sweep.sh             # σ sweep orchestrator
└── CLAUDE.md                   # Project notes for Claude Code
```

### Outputs

Per-method round CSVs (`FedAvg_rounds.csv`, `AO-FRL_rounds.csv`, `Centralized_rounds.csv`, …) plus `final_summary.json` are written to `results/` (or a `--out_dir` you specify). Per-class accuracy matrices are saved as `*_per_class.npy`. The A2A audit log is at `results/a2a_communication.json`.

## License

This project is released for academic research purposes.

## Citation

If you use this code or its findings, please cite:

```bibtex
@misc{aofrl2026,
  title  = {AO-FRL: Agent-Orchestrated Federated Representation Learning with Per-Record Differential Privacy},
  author = {Lei Yao},
  year   = {2026},
  howpublished = {\url{https://github.com/uwm-se/AO-FRL}}
}
```
