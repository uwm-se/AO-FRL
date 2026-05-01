#!/usr/bin/env bash
# Sweep AO-FRL across three DP noise strengths (strong / medium / weak).
# Each run produces its own results directory with per-round CSVs, the final
# inversion grid + PSNR summary, and the FedAvg/Centralized baselines.
#
# Usage: bash run_dp_sweep.sh [extra args...]
# Anything you append (e.g. --rounds 50 --n_clients 10) is forwarded to all
# three runs.

set -euo pipefail

EXTRA_ARGS=("$@")

# σ values: strong → weak. Corresponding (ε, δ=1e-5) labels are derived in
# run_experiment.py for logging. clip_C=1.0 (default) on unit-norm embeddings.
declare -a SIGMAS=("0.05" "0.02" "0.005")
declare -a TAGS=("strong" "medium" "weak")

# AO-FRL only per σ; FedAvg/Centralized are σ-independent — run once separately.
for i in "${!SIGMAS[@]}"; do
  s="${SIGMAS[$i]}"
  tag="${TAGS[$i]}"
  out="results/dp_${tag}_sigma${s}"
  echo "============================================================"
  echo "[sweep] σ=${s} (${tag}) -> ${out}"
  echo "============================================================"
  python run_experiment.py --sigma "${s}" \
                           --methods ao-frl \
                           --results_dir "${out}" \
                           "${EXTRA_ARGS[@]}"
done

echo "Sweep complete. Results in:"
for tag in "${TAGS[@]}"; do
  for s in "${SIGMAS[@]}"; do :; done
  echo "  results/dp_${tag}_sigma*"
done
