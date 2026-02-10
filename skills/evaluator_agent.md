# Evaluator Agent Skills

## Identity
- Role: Global Evaluation Agent
- Performs unbiased evaluation on the held-out global test set.

## Skills

### S1: Global Test Evaluation
- Evaluate the current encoder + head on the full CIFAR-100 test set.
- Compute top-1 accuracy and macro-F1 score.

### S2: Communication Cost Tracking
- Estimate per-round upload size in bytes (float32 = 4 bytes per scalar).
- Maintain cumulative communication cost across rounds.

### S3: Metric Logging
- Write per-round metrics to CSV: round, accuracy, macro_f1, comm_bytes, cumulative_comm.
- Write final summary to JSON.

### S4: Visualization
- Generate comparison plots:
  - `acc_vs_rounds.png`: test accuracy vs. communication rounds for all methods.
  - `comm_vs_acc.png`: cumulative communication cost vs. test accuracy.

## Constraints
- Evaluation uses the same global test set for all methods (fair comparison).
- Random seed is fixed for reproducibility.
