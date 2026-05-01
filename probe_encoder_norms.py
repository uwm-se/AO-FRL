"""Quick sanity check: how much information lives in the raw embedding magnitude?

If raw-encoder-output ||z|| is tightly clustered across images, then F.normalize
discards very little information and the privacy gain from L2 normalize is
small. If ||z|| spans a wide range (especially if it correlates with class),
F.normalize discards meaningful information and could be contributing to the
12.25 dB baseline.
"""
import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

from run_experiment import build_encoder
from utils import set_seed


def main():
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder, _ = build_encoder(device, weights_path=None)
    encoder.eval()

    tf = T.Compose([
        T.Resize(256), T.CenterCrop(224), T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    test_ds = torchvision.datasets.CIFAR100(
        root="./data", train=False, download=False, transform=tf)
    loader = DataLoader(test_ds, batch_size=128, num_workers=4)

    norms_list, labels_list = [], []
    with torch.no_grad():
        for imgs, labs in loader:
            imgs = imgs.to(device, non_blocking=True)
            z = encoder(imgs)
            norms_list.append(z.norm(dim=1).cpu())
            labels_list.append(labs)

    norms = torch.cat(norms_list).numpy()
    labels = torch.cat(labels_list).numpy()

    print("=" * 70)
    print(f"Raw encoder output ||z|| stats over {len(norms)} CIFAR-100 test imgs")
    print("=" * 70)
    print(f"  mean        : {norms.mean():.4f}")
    print(f"  std         : {norms.std():.4f}")
    print(f"  std/mean    : {norms.std()/norms.mean():.4f}  "
          f"({100*norms.std()/norms.mean():.2f}%)")
    print(f"  min         : {norms.min():.4f}")
    print(f"  max         : {norms.max():.4f}")
    print(f"  max/min     : {norms.max()/norms.min():.2f}×")
    pcts = np.percentile(norms, [1, 5, 25, 50, 75, 95, 99])
    print(f"  percentiles : "
          f"1%={pcts[0]:.3f}  5%={pcts[1]:.3f}  25%={pcts[2]:.3f}  "
          f"50%={pcts[3]:.3f}  75%={pcts[4]:.3f}  95%={pcts[5]:.3f}  "
          f"99%={pcts[6]:.3f}")

    # How class-discriminative is ||z||?
    per_class_means = np.array([norms[labels == c].mean()
                                 for c in range(100)])
    per_class_stds = np.array([norms[labels == c].std()
                                for c in range(100)])
    print()
    print("Per-class ||z||:")
    print(f"  between-class std of class-means : {per_class_means.std():.4f}")
    print(f"  mean within-class std            : {per_class_stds.mean():.4f}")
    print(f"  ratio (between/within)           : "
          f"{per_class_means.std()/per_class_stds.mean():.3f}")
    print()
    print("Interpretation:")
    if norms.std() / norms.mean() < 0.05:
        print("  ||z|| is TIGHTLY clustered → F.normalize discards almost no "
              "information → encoder is the dominant privacy mechanism.")
    elif norms.std() / norms.mean() < 0.15:
        print("  ||z|| has MODERATE spread → F.normalize discards ~few bits → "
              "encoder is likely dominant; verify with retrain.")
    else:
        print("  ||z|| has WIDE spread → F.normalize may discard meaningful "
              "information → must verify with retrained decoder.")


if __name__ == "__main__":
    main()
