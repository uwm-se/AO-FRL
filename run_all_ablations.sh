#!/bin/bash
# Run all ablation experiments sequentially

set -e  # Exit on error

echo "======================================================================"
echo "Running Ablation Studies"
echo "======================================================================"

# Ablation 1: No Server Orchestration
echo ""
echo "Starting Ablation 1: No Server Orchestration..."
echo "----------------------------------------------------------------------"
python run_ablation_no_orchestration.py \
    --rounds 100 \
    --n_clients 20 \
    --alpha 0.3 \
    --seed 42 \
    --results_dir ablation_results/no_orchestration

echo ""
echo "✓ Ablation 1 Complete"
echo ""

# Ablation 2: No Privacy Gate
echo "Starting Ablation 2: No Privacy Gate..."
echo "----------------------------------------------------------------------"
python run_ablation_no_privacy_gate.py \
    --rounds 100 \
    --n_clients 20 \
    --alpha 0.3 \
    --seed 42 \
    --results_dir ablation_results/no_privacy_gate

echo ""
echo "✓ Ablation 2 Complete"
echo ""

# Generate comparison plots
echo "Generating comparison plots..."
echo "----------------------------------------------------------------------"
python plot_ablation_results.py \
    --baseline results \
    --no_orchestration ablation_results/no_orchestration \
    --no_privacy_gate ablation_results/no_privacy_gate \
    --output ablation_results/figures

echo ""
echo "======================================================================"
echo "All Ablation Studies Complete!"
echo "======================================================================"
echo ""
echo "Results saved to:"
echo "  - ablation_results/no_orchestration/"
echo "  - ablation_results/no_privacy_gate/"
echo "  - ablation_results/figures/"
echo ""
