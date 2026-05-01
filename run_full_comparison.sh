#!/usr/bin/env bash
# Full method comparison: Centralized + FedAvg + FedProx + FedAdam
# + AO-FRL @ σ ∈ {0.005, 0.02, 0.05}
#
# All methods share:
#   - Frozen ImageNet ResNet-18 encoder (no fine-tune)
#   - Federated pool (45000) — decoder pool of 5000 held out
#   - 20 clients, Dirichlet α=0.3
#   - Up to 100 rounds, early stopping with patience=10
#   - Same CIFAR-100 test set (10000) for all evaluations
#
# Per-method outputs land in results/full_compare/<run>/<Method>_rounds.csv.

set -euo pipefail

ROOT="results/full_compare"
mkdir -p "${ROOT}"

# ----- Baselines (4 methods in one process: shares the encoder precompute) -----
echo "============================================================"
echo "[1/4] Centralized + FedAvg + FedProx + FedAdam"
echo "============================================================"
python run_experiment.py \
  --methods centralized fedavg fedprox fedadam \
  --rounds 100 --n_clients 20 --alpha 0.3 \
  --centralized_epochs 50 \
  --early_stop_patience 10 \
  --decoder_weights /nonexistent_d --encoder_weights /nonexistent_e \
  --inversion_n_samples 0 \
  --results_dir "${ROOT}/baselines"

# ----- AO-FRL across 3 σ levels -----
declare -a SIGMAS=("0.005" "0.02" "0.05")
declare -a TAGS=("weak" "medium" "strong")

for i in "${!SIGMAS[@]}"; do
  s="${SIGMAS[$i]}"
  tag="${TAGS[$i]}"
  out="${ROOT}/aofrl_${tag}_sigma${s}"
  echo "============================================================"
  echo "[$((i+2))/4] AO-FRL  σ=${s} (${tag})  -> ${out}"
  echo "============================================================"
  python run_experiment.py \
    --methods ao-frl --sigma "${s}" \
    --rounds 100 --n_clients 20 --alpha 0.3 \
    --early_stop_patience 10 \
    --decoder_weights /nonexistent_d --encoder_weights /nonexistent_e \
    --inversion_n_samples 0 \
    --results_dir "${out}"
done

echo "Full comparison complete. Results in ${ROOT}/"
