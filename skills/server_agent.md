# Server Agent Skills

## Identity
- Role: Federated Learning Server / Orchestrator Agent
- Aggregates privacy-gated representations and trains the classification head.

## Skills

### S1: Representation Aggregation
- Collect all `(z_tilde, y)` pairs uploaded by clients each round.
- Merge into a server-side training dataset `D_server^t`.

### S2: Head Training
- Train a lightweight MLP head (or linear classifier) on `D_server^t`.
- Optimizer: SGD or Adam; configurable learning rate and epochs.

### S3: Orchestration & Instruction Generation
- Aggregate client summaries `S_i^t` to compute:
  - Global label gap (deviation from uniform or target distribution).
  - Per-client contribution weight / demand score.
- Generate per-client instructions `C_i^{t+1}`:
  - `upload_budget`: per-class and total embedding upload limits.
  - `sigma`: noise intensity for next round.
  - `augmentation_mode`: normal or conservative.

### S4: Hook Dispatch
- Evaluate hook trigger conditions from client summaries.
- Dispatch `low_data_hook`, `high_risk_hook`, `drift_hook` instructions.

### S5: Head Broadcast
- Send updated head parameters to all clients for local evaluation.

### S6: FedAvg Aggregation (Baseline)
- For the FedAvg baseline: receive local model updates, compute weighted average, broadcast global model.

## Constraints
- Server never accesses raw client images.
- All orchestration decisions are logged for auditability.
