# AO-FRL: Agent-Orchestrated Federated Representation Learning

A federated learning framework featuring **agent-orchestrated privacy-gated representation sharing** for non-IID data scenarios. This project compares the proposed method against the FedAvg baseline on CIFAR-100 with Dirichlet-distributed non-IID partitioning.

## Overview

This repository implements a novel federated learning approach that combines:
- **Privacy-preserving representation sharing** with adaptive noise and gating mechanisms
- **Server-side orchestration** that dynamically adjusts client behavior based on global insights
- **Adaptive hooks** that respond to data scarcity, privacy risks, and model drift
- **Communication-efficient** embedding transmission vs. full model parameters

## Key Features

### Proposed Method
- **Frozen ResNet-18 encoder** for feature extraction (512-dim embeddings)
- **Trainable MLP head** (512 → 256 → 100 classes)
- **Three-stage privacy pipeline**:
  1. L2 clipping (bound C)
  2. Gaussian noise addition (scale σ)
  3. Privacy gate (cosine similarity threshold τ_high)

### Adaptive Hooks
- **Low-data hook**: Switches to conservative augmentation when class samples < k
- **High-risk hook**: Increases noise and reduces upload budget when rejection rate > r
- **Drift hook**: Boosts upload budget by 30% when validation accuracy declines

### Server Orchestration
- Analyzes client class distributions and label gaps
- Generates per-client instructions (budget, noise, augmentation mode)
- Maintains separate heads for FedAvg and proposed method

## Installation

### Dependencies
```bash
pip install torch torchvision scikit-learn matplotlib numpy
```

### Dataset
CIFAR-100 will be automatically downloaded to `data/` on first run.

## Usage

### Basic Experiments

```bash
# Default run (100 rounds, 20 clients, α=0.3)
python run_experiment.py --rounds 100 --n_clients 20 --alpha 0.3

# Quick test run
python run_experiment.py --rounds 10 --n_clients 5 --alpha 0.5 --seed 123

# Run only one method
python run_experiment.py --methods fedavg
python run_experiment.py --methods proposed
```

### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--rounds` | Number of federated learning rounds | 100 |
| `--n_clients` | Number of clients | 20 |
| `--alpha` | Dirichlet concentration (lower = more non-IID) | 0.3 |
| `--sigma` | DP noise scale for privacy gating | 0.1 |
| `--tau_high` | Cosine similarity threshold for rejection | 0.95 |
| `--upload_budget` | Max embeddings per client per round | 100 |

### 10-Class Subset Test

```bash
python run_test_10classes.py
```

Uses full agent classes with encoder access and augmentation pipelines (hardcoded config, no CLI args).

### Ablation Studies

```bash
# Run all ablation experiments
bash run_all_ablations.sh

# Individual ablations
python run_ablation_no_orchestration.py
python run_ablation_no_privacy_gate.py
```

## Project Structure

```
.
├── agents/
│   ├── client_agent.py       # Full ClientAgent with encoder & caching
│   ├── server_agent.py       # ServerAgent + MLPHead
│   └── evaluator_agent.py    # EvaluatorAgent for validation
├── run_experiment.py         # Main experiment script (precomputed embeddings)
├── run_test_10classes.py     # 10-class test with full agents
├── run_ablation_*.py         # Ablation study scripts
├── plot_*.py                 # Visualization scripts
├── CLAUDE.md                 # Project instructions for Claude Code
└── *.md                      # Documentation and flowcharts
```

### Dual Implementation Pattern

1. **`run_experiment.py`**: Lightweight classes (`FedAvgClient`, `ProposedClient`, `FastEvaluator`) operating on **precomputed embeddings** for speed
2. **`agents/`**: Full-featured agent classes with encoder access, augmentation pipelines, and embedding caching

Both share `ServerAgent` and `MLPHead` from `agents/server_agent.py`.

## Results

Results are saved to:
- **Main experiments**: `results/`
- **10-class tests**: `test_results/`
- **Ablation studies**: `ablation_results/`

### Output Files
- `{method}_results.csv`: Per-round metrics (accuracy, loss, communication)
- `final_summary.json`: Aggregated statistics
- `acc_vs_rounds.png`: Accuracy comparison plot
- `comm_vs_acc.png`: Communication efficiency plot

## Architecture Details

### MLPHead
```
Linear(512, 256) → BatchNorm → ReLU → Dropout(0.2) → Linear(256, 100)
```

### Communication Cost Tracking
- **FedAvg**: Sends full head parameters (float32 × num_params)
- **Proposed**: Sends variable-size gated embeddings + small summaries

## Documentation

Comprehensive documentation available in markdown files:
- `framework_description.md`: Overall framework explanation
- `method_adaptive_hooks.md`: Detailed hook mechanisms
- `server_orchestration_flow.md`: Server orchestration logic
- `privacy_gate_*.md`: Privacy pipeline flowcharts
- `hyperparameters.md`: Parameter tuning guide

## License

This project is released for academic research purposes.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{aofrl2025,
  title={Agent-Orchestrated Federated Representation Learning with Privacy-Gated Sharing},
  author={Your Name},
  year={2025},
  howpublished={\url{https://github.com/WillnotchooseC/AO-FRL}}
}
```

## Contact

For questions or issues, please open an issue on GitHub.
