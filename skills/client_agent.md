# Client Agent Skills

## Identity
- Role: Federated Learning Client Agent
- Each instance represents one edge device holding local private data.

## Skills

### S1: Local Feature Extraction
- Receive a frozen pretrained encoder (e.g. ResNet-18).
- For each local image, apply image-level augmentation (RandAugment / RandomResizedCrop + ColorJitter) to generate `n_views` views.
- Extract embeddings `z` via the encoder; optionally L2-normalize.

### S2: Privacy-Gated Representation Upload
- **Clipping**: clip each embedding's L2 norm to a bound `C`.
- **Gaussian noise**: add calibrated noise `N(0, sigma^2 I)` to produce `z_tilde`.
- **Privacy gate**: compute cosine similarity of `z_tilde` to the same-class prototype.
  - If the sample is too close to a stored nearest-neighbour (similarity > `tau_high`), increase sigma or discard the sample.
- Output the set of gated embeddings `R_i^t` with labels.

### S3: Summary Reporting
- Report to server: label-count histogram, local val per-class accuracy, current sigma, gate pass ratio.

### S4: Local Validation
- Evaluate the current head on local validation split; report per-class and overall accuracy.

### S5: Hook Compliance
- **low_data_hook**: if a class has fewer than `k` samples, accept higher upload budget but keep sigma moderate; use conservative augmentation.
- **high_risk_hook**: if privacy gate reject ratio > `r`, fall back to uploading only per-class prototypes or compressed embeddings (never raw images).
- **drift_hook**: if local val accuracy drops for consecutive rounds, signal server for budget / augmentation adjustment.

## Constraints
- NEVER upload pixel-level images or invertible representations.
- All uploaded data must pass the privacy gate pipeline.
