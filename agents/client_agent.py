"""ClientAgent: local feature extraction, privacy gating, and summary reporting."""

import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as T


class ClientAgent:
    """Federated client agent — extracts gated embeddings or trains local model."""

    SKILL_FILE = "skills/client_agent.md"

    def __init__(self, client_id: int, train_indices: np.ndarray,
                 val_indices: np.ndarray, dataset, encoder, embed_dim: int,
                 n_classes: int, device: str, cfg: dict):
        self.id = client_id
        self.dataset = dataset
        self.train_idx = train_indices
        self.val_idx = val_indices
        self.encoder = encoder
        self.embed_dim = embed_dim
        self.n_classes = n_classes
        self.device = device
        self.cfg = cfg

        # Privacy / gating parameters (will be updated by server instructions)
        self.sigma = cfg.get("sigma", 0.02)
        self.clip_C = cfg.get("clip_C", 1.0)
        self.tau_high = cfg.get("tau_high", 0.95)
        self.upload_budget = cfg.get("upload_budget", 9999)
        self.augmentation_mode = "normal"  # or "conservative"

        # Hooks tracking
        self.prev_val_accs = []  # for drift detection

        # Augmentation transforms
        self._build_augmentations()

        # Precompute label array for convenience
        self.train_labels = np.array([dataset[i][1] for i in train_indices])
        self.val_labels = np.array([dataset[i][1] for i in val_indices])

        # Cache: precomputed clean embeddings (computed once, reused)
        self._cached_clean_embs = None
        self._cached_clean_labels = None

    # ------------------------------------------------------------------ #
    #  Augmentations                                                      #
    # ------------------------------------------------------------------ #
    def _build_augmentations(self):
        self.aug_normal = T.Compose([
            T.RandomResizedCrop(224, scale=(0.6, 1.0)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.4, 0.4, 0.4, 0.1),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.aug_conservative = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.aug_eval = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def _get_aug(self):
        return self.aug_conservative if self.augmentation_mode == "conservative" else self.aug_normal

    # ------------------------------------------------------------------ #
    #  Proposed method: extract privacy-gated embeddings (BATCHED)        #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def extract_gated_embeddings(self, n_views: int = 2):
        """Extract embeddings with clipping, noise, and privacy gate.
        Uses batched encoder forward passes for efficiency.
        Returns (embeddings, labels, summary_dict).
        """
        self.encoder.eval()

        # Step 1: Get clean embeddings (cached after first call)
        if self._cached_clean_embs is None:
            self._precompute_clean_embeddings()

        clean_embs = self._cached_clean_embs  # (N, D)
        clean_labels = self._cached_clean_labels  # (N,)

        # Compute per-class prototypes from clean embeddings
        prototypes = {}
        for c in range(self.n_classes):
            mask = clean_labels == c
            if mask.any():
                proto = clean_embs[mask].mean(dim=0)
                prototypes[c] = F.normalize(proto, dim=0)

        # Step 2: Generate multi-view embeddings from cached clean embeddings.
        # Since encoder is frozen, augmentation in embedding space is simulated
        # by adding small random perturbations (view noise) on top of DP noise.
        # This avoids re-running the encoder on augmented images every round.
        all_z, all_y = [], []
        reject_count = 0
        total_count = 0
        N = clean_embs.size(0)

        # Stack prototypes for vectorized gate check
        proto_tensor = torch.zeros(self.n_classes, self.embed_dim)
        proto_mask = torch.zeros(self.n_classes, dtype=torch.bool)
        for c, p in prototypes.items():
            proto_tensor[c] = p
            proto_mask[c] = True

        for view_idx in range(n_views):
            # Simulate multi-view by adding small view-specific perturbation
            view_noise = torch.randn(N, self.embed_dim) * 0.01 * (view_idx + 1)
            z_views = clean_embs + view_noise

            # Clipping (vectorized)
            norms = z_views.norm(dim=1, keepdim=True)
            clip_mask = norms.squeeze() > self.clip_C
            z_views[clip_mask] = z_views[clip_mask] * (
                self.clip_C / norms[clip_mask])

            # Gaussian noise (vectorized)
            z_tilde_all = z_views + torch.randn(N, self.embed_dim) * self.sigma

            # Privacy gate (vectorized)
            labels_np = clean_labels.numpy()
            for i in range(N):
                total_count += 1
                label = labels_np[i]
                z_tilde = z_tilde_all[i]

                if proto_mask[label]:
                    sim = F.cosine_similarity(
                        z_tilde.unsqueeze(0),
                        proto_tensor[label].unsqueeze(0)
                    ).item()
                    if sim > self.tau_high:
                        reject_count += 1
                        continue

                all_z.append(z_tilde)
                all_y.append(int(label))

                if len(all_z) >= self.upload_budget:
                    break
            if len(all_z) >= self.upload_budget:
                break

        reject_ratio = reject_count / max(total_count, 1)

        # Apply hooks
        self._apply_hooks(reject_ratio, all_z, all_y, prototypes)

        # Fallback: upload prototypes if nothing passed gate
        if len(all_z) == 0 and prototypes:
            for c, proto in prototypes.items():
                if c in set(self.train_labels):
                    all_z.append(proto + torch.randn_like(proto) * self.sigma)
                    all_y.append(c)

        if all_z:
            embeddings = torch.stack(all_z)
            labels_t = torch.tensor(all_y, dtype=torch.long)
        else:
            embeddings = torch.zeros(0, self.embed_dim)
            labels_t = torch.zeros(0, dtype=torch.long)

        # Label histogram
        hist = np.zeros(self.n_classes, dtype=int)
        for y in all_y:
            hist[y] += 1

        summary = {
            "client_id": self.id,
            "label_histogram": hist.tolist(),
            "reject_ratio": reject_ratio,
            "sigma": self.sigma,
            "n_uploaded": len(all_z),
            "augmentation_mode": self.augmentation_mode,
        }

        return embeddings, labels_t, summary

    @torch.no_grad()
    def _precompute_clean_embeddings(self):
        """Precompute and cache clean embeddings for all training data."""
        self.encoder.eval()
        loader = self._make_loader(self.train_idx, self.aug_eval,
                                   batch_size=128, shuffle=False)
        embs_list, labels_list = [], []
        for imgs, labels in loader:
            imgs = imgs.to(self.device)
            z = self.encoder(imgs).cpu()
            z = F.normalize(z, dim=1)
            embs_list.append(z)
            labels_list.append(labels)
        self._cached_clean_embs = torch.cat(embs_list, dim=0)
        self._cached_clean_labels = torch.cat(labels_list, dim=0)

    def _apply_hooks(self, reject_ratio, all_z, all_y, prototypes):
        """Apply low_data, high_risk, and drift hooks."""
        # low_data_hook
        label_counts = np.bincount(self.train_labels,
                                   minlength=self.n_classes)
        k = self.cfg.get("low_data_k", 10)
        low_classes = set(np.where(label_counts < k)[0].tolist())
        if low_classes:
            self.augmentation_mode = "conservative"

        # high_risk_hook
        r = self.cfg.get("high_risk_r", 0.30)
        if reject_ratio > r:
            self.sigma = min(self.sigma * 1.5, 0.5)
            self.upload_budget = max(self.upload_budget // 2, 50)

        # drift_hook
        if len(self.prev_val_accs) >= 2:
            if all(self.prev_val_accs[-i] < self.prev_val_accs[-i - 1]
                   for i in range(1, min(3, len(self.prev_val_accs)))):
                self.upload_budget = int(self.upload_budget * 1.3)

    # ------------------------------------------------------------------ #
    #  FedAvg baseline: local SGD training                                #
    # ------------------------------------------------------------------ #
    def local_train_fedavg(self, head, local_epochs: int, lr: float,
                           batch_size: int = 64):
        """Train head locally for FedAvg using cached embeddings.
        Returns updated head state_dict and comm cost (bytes uploaded).
        """
        # Ensure embeddings are cached
        if self._cached_clean_embs is None:
            self._precompute_clean_embeddings()

        head_local = copy.deepcopy(head).to(self.device)
        head_local.train()
        loss_fn = torch.nn.CrossEntropyLoss()
        opt = torch.optim.SGD(head_local.parameters(), lr=lr)

        # Use cached embeddings directly (no encoder forward pass needed)
        from torch.utils.data import TensorDataset, DataLoader as DL
        emb_ds = TensorDataset(self._cached_clean_embs,
                               self._cached_clean_labels)
        loader = DL(emb_ds, batch_size=batch_size, shuffle=True)

        for _ in range(local_epochs):
            for z_batch, y_batch in loader:
                z_batch = z_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                logits = head_local(z_batch)
                loss = loss_fn(logits, y_batch)
                opt.zero_grad()
                loss.backward()
                opt.step()

        n_params = sum(p.numel() for p in head_local.parameters())
        comm_bytes = n_params * 4  # float32
        return head_local.state_dict(), comm_bytes

    # ------------------------------------------------------------------ #
    #  Local validation                                                    #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def evaluate_local(self, head):
        """Evaluate on local val set using precomputed val embeddings."""
        head.eval()
        # Precompute val embeddings if needed
        if not hasattr(self, '_cached_val_embs') or self._cached_val_embs is None:
            self.encoder.eval()
            loader = self._make_loader(self.val_idx, self.aug_eval,
                                       batch_size=128, shuffle=False)
            embs_list, labels_list = [], []
            for imgs, labels in loader:
                imgs = imgs.to(self.device)
                z = self.encoder(imgs).cpu()
                embs_list.append(z)
                labels_list.append(labels)
            if embs_list:
                self._cached_val_embs = torch.cat(embs_list, dim=0)
                self._cached_val_labels = torch.cat(labels_list, dim=0)
            else:
                self._cached_val_embs = torch.zeros(0, self.embed_dim)
                self._cached_val_labels = torch.zeros(0, dtype=torch.long)

        if self._cached_val_embs.size(0) == 0:
            self.prev_val_accs.append(0.0)
            return 0.0, np.zeros(self.n_classes)

        z_all = self._cached_val_embs.to(self.device)
        y_all = self._cached_val_labels.to(self.device)
        preds = head.to(self.device)(z_all).argmax(dim=1)
        correct = (preds == y_all).sum().item()
        total = y_all.size(0)
        acc = correct / total

        per_class_correct = np.zeros(self.n_classes)
        per_class_total = np.zeros(self.n_classes)
        preds_np = preds.cpu().numpy()
        y_np = y_all.cpu().numpy()
        for c in range(self.n_classes):
            mask = y_np == c
            per_class_total[c] = mask.sum()
            per_class_correct[c] = (preds_np[mask] == c).sum()

        per_class_acc = np.divide(per_class_correct, per_class_total,
                                  out=np.zeros_like(per_class_correct),
                                  where=per_class_total > 0)
        self.prev_val_accs.append(acc)
        return acc, per_class_acc

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #
    def _make_loader(self, indices, transform, batch_size=64, shuffle=False):
        subset = _TransformSubset(self.dataset, indices, transform)
        return DataLoader(subset, batch_size=batch_size, shuffle=shuffle,
                          num_workers=2, pin_memory=False)

    def apply_server_instructions(self, instructions: dict):
        """Update local parameters based on server-agent instructions."""
        if instructions is None:
            return
        self.upload_budget = instructions.get("upload_budget", self.upload_budget)
        self.sigma = instructions.get("sigma", self.sigma)
        aug_mode = instructions.get("augmentation_mode", self.augmentation_mode)
        self.augmentation_mode = aug_mode


class _TransformSubset(torch.utils.data.Dataset):
    """Subset with custom transform applied to PIL images."""
    def __init__(self, dataset, indices, transform):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img, label = self.dataset[self.indices[idx]]
        if self.transform:
            img = self.transform(img)
        return img, label
