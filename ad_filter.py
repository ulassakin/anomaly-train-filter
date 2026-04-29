"""
Stage 1: Separate normal vs. anomalous images from contaminated training data.
Textile categories only: carpet, grid, leather, tile, wood.

What this does:
  1. Load all training images (contaminated — mix of normal + some defective)
  2. Extract DINOv2 patch features for each image
  3. Fit PCA + GMM on all patch features (iterative trimming to handle contamination)
  4. Score each image by the fraction of its patches that score above an anomaly threshold
  5. Split into normal / anomalous using a second GMM on the image scores
  6. Save two lists: {category}_normal_paths.txt and {category}_anomalous_paths.txt

Usage:
    python ad_filter.py --data_root /path/to/mvtec --category carpet
    python ad_filter.py --data_root /path/to/mvtec --category all
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DINO_MODEL           = "vit_small_patch14_dinov2.lvd142m"
IMG_SIZE             = 518
PCA_DIM              = 64
GMM_COMPONENTS       = 9
TRIM_PERCENTILE      = 85   # Keep bottom X% of patches each trimming round
TRIM_ITERATIONS      = 3    # Number of GMM refitting rounds
PATCH_ANOMALY_PCTILE = 90   # Patch score percentile used as anomaly threshold
RANDOM_SEED          = 42

# All five textile categories
TEXTILE_CATEGORIES = ["carpet", "grid", "leather", "tile", "wood"]

# Brightness normalization reduces false positives caused by lighting variation.
# Grid has very consistent, uniform lighting so normalization hurts it — excluded.
BRIGHTNESS_NORM_CATEGORIES = {"carpet", "leather", "tile", "wood"}


# ─────────────────────────────────────────────
# DEVICE
# ─────────────────────────────────────────────
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────
class ImageFolderDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, str(self.image_paths[idx])


def load_training_images(data_root: Path, category: str, contamination: float = 0.1):
    """
    Builds a contaminated training set:
      - Normal images from train/good
      - Defective images injected from test/ subfolders (~10% by default)
    Also returns ground truth labels so separation quality can be evaluated.
    """
    train_dir = data_root / category / "train" / "good"
    test_dir  = data_root / category / "test"

    normal_paths = []
    for ext in ["*.png", "*.jpg", "*.PNG", "*.JPG"]:
        normal_paths += sorted(train_dir.glob(ext))

    anomaly_paths = []
    for defect_dir in sorted(test_dir.iterdir()):
        if defect_dir.is_dir() and defect_dir.name != "good":
            for ext in ["*.png", "*.jpg", "*.PNG", "*.JPG"]:
                anomaly_paths += sorted(defect_dir.glob(ext))

    n_inject = int(len(normal_paths) * contamination / (1 - contamination))
    n_inject = min(n_inject, len(anomaly_paths))
    rng = np.random.RandomState(RANDOM_SEED)
    injected = [Path(p) for p in rng.choice(anomaly_paths, n_inject, replace=False)]

    all_paths  = normal_paths + injected
    all_labels = [0] * len(normal_paths) + [1] * len(injected)

    print(f"  Normal images    : {len(normal_paths)}")
    print(f"  Injected defects : {len(injected)}  ({contamination*100:.0f}% contamination)")
    print(f"  Total            : {len(all_paths)}")
    return all_paths, all_labels


# ─────────────────────────────────────────────
# DINOV2
# ─────────────────────────────────────────────
def load_dinov2(device):
    import timm
    model = timm.create_model(DINO_MODEL, pretrained=True, num_classes=0)
    model = model.to(device).eval()
    print(f"  Loaded {DINO_MODEL}")
    return model


@torch.no_grad()
def extract_features(model, dataloader, device):
    """
    Extracts DINOv2 patch features for every image in the dataloader.

    Returns:
        all_patch_features : list of (N_patches, D) arrays, one per image
        all_paths          : list of str, one per image
    """
    all_patch_features = []
    all_paths = []

    for imgs, paths in tqdm(dataloader, desc="Extracting features"):
        imgs = imgs.to(device)
        feats = model.get_intermediate_layers(imgs, n=1)[0]  # (B, N_patches+1, D)
        patch_feats = feats[:, 1:, :].cpu().numpy()          # drop CLS token → (B, N_patches, D)

        for i in range(len(paths)):
            all_patch_features.append(patch_feats[i])
            all_paths.append(paths[i])

    return all_patch_features, all_paths


# ─────────────────────────────────────────────
# GMM WITH ITERATIVE TRIMMING
# ─────────────────────────────────────────────
def fit_gmm_iterative(all_patch_features):
    """
    Fits a GMM on patch features with iterative trimming to handle contamination:
      - Round 1: fit on all patches
      - Score all patches, discard the top (100 - TRIM_PERCENTILE)% highest scorers
      - Rounds 2+: refit only on the surviving patches
    Each round pushes anomalous patches further from the fitted distribution.

    Returns:
        pca      : fitted PCA object
        gmm      : fitted GaussianMixture object
        trim_mask: boolean array — True = patch kept in final fit
    """
    flat_features = np.vstack(all_patch_features)  # (total_patches, D)
    print(f"  Total patches: {flat_features.shape[0]:,}  |  dim: {flat_features.shape[1]}")

    print(f"  Fitting PCA ({flat_features.shape[1]}D → {PCA_DIM}D) ...")
    pca = PCA(n_components=PCA_DIM, random_state=RANDOM_SEED)
    reduced = pca.fit_transform(flat_features)
    print(f"  Explained variance: {pca.explained_variance_ratio_.cumsum()[-1]:.3f}")

    mask = np.ones(len(reduced), dtype=bool)  # start: keep all patches

    for round_idx in range(TRIM_ITERATIONS):
        print(f"  GMM round {round_idx + 1}/{TRIM_ITERATIONS}  ({mask.sum():,} patches kept) ...")
        gmm = GaussianMixture(
            n_components=GMM_COMPONENTS,
            covariance_type="full",
            max_iter=300,
            random_state=RANDOM_SEED,
        )
        gmm.fit(reduced[mask])

        # Score ALL patches so the mask update is unbiased
        scores = -gmm.score_samples(reduced)   # higher = more anomalous
        threshold = np.percentile(scores[mask], TRIM_PERCENTILE)
        mask = scores < threshold

    print(f"  Final GMM fit on {mask.sum():,} patches  ({100 * mask.mean():.1f}% of total)")
    return pca, gmm, mask


# ─────────────────────────────────────────────
# IMAGE-LEVEL SCORING  (texture mode only)
# ─────────────────────────────────────────────
def score_images(pca, gmm, all_patch_features):
    """
    Scores each image by the fraction of its patches that exceed a global
    anomaly threshold. This is appropriate for textile / texture categories
    where defects (scratches, holes, stains) are spatially spread across
    the image rather than localized to one region.

    Returns:
        image_scores : (N_images,) array of floats in [0, 1]
    """
    # Compute patch-level anomaly scores for every image
    all_patch_scores = []
    for patch_features in all_patch_features:
        reduced = pca.transform(patch_features)
        patch_scores = -gmm.score_samples(reduced)
        all_patch_scores.append(patch_scores)

    # Global threshold: a patch is "anomalous" if it exceeds this percentile
    patch_threshold = np.percentile(np.concatenate(all_patch_scores), PATCH_ANOMALY_PCTILE)
    print(f"  Patch anomaly threshold (p{PATCH_ANOMALY_PCTILE}): {patch_threshold:.3f}")

    # Image score = fraction of anomalous patches
    image_scores = np.array([
        (patch_scores > patch_threshold).mean()
        for patch_scores in all_patch_scores
    ])
    return image_scores


# ─────────────────────────────────────────────
# PER-CATEGORY PIPELINE
# ─────────────────────────────────────────────
def run_category(category, data_root, out_dir, model, device):
    print(f"\n{'=' * 55}")
    print(f"  Category: {category}")
    print(f"{'=' * 55}")

    # 1. Load images
    print("[1/4] Loading training images ...")
    paths, gt_labels = load_training_images(data_root, category)

    # Build transform — brightness normalization on for categories with lighting variation
    base_transforms = [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ]
    if category in BRIGHTNESS_NORM_CATEGORIES:
        base_transforms.append(
            transforms.Lambda(lambda x: x * (0.5 / (x.mean() + 1e-6)))
        )
        print(f"  Brightness normalization: ON")
    else:
        print(f"  Brightness normalization: OFF")
    base_transforms.append(
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    )
    transform = transforms.Compose(base_transforms)

    dataset = ImageFolderDataset(paths, transform)
    loader  = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)

    # 2. Extract features
    print("\n[2/4] Extracting DINOv2 patch features ...")
    all_patch_features, all_paths = extract_features(model, loader, device)

    # 3. Fit GMM with iterative trimming
    print("\n[3/4] Fitting GMM (iterative trimming) ...")
    pca, gmm, trim_mask = fit_gmm_iterative(all_patch_features)

    # 4. Score images and split into normal / anomalous
    print("\n[4/4] Scoring images ...")
    image_scores = score_images(pca, gmm, all_patch_features)

    # Fit a 2-component GMM on image scores to find the natural split
    score_gmm = GaussianMixture(n_components=2, random_state=RANDOM_SEED)
    score_gmm.fit(image_scores.reshape(-1, 1))

    normal_component    = np.argmin(score_gmm.means_)
    anomalous_component = np.argmax(score_gmm.means_)
    assignments = score_gmm.predict(image_scores.reshape(-1, 1))

    pred_normal    = [p for p, a in zip(all_paths, assignments) if a == normal_component]
    pred_anomalous = [p for p, a in zip(all_paths, assignments) if a == anomalous_component]

    print(f"\n  Normal cluster mean   : {score_gmm.means_[normal_component][0]:.3f}")
    print(f"  Anomalous cluster mean: {score_gmm.means_[anomalous_component][0]:.3f}")

    # Evaluate against injected ground truth
    pred_anomalous_set = set(pred_anomalous)
    total_injected  = sum(gt_labels)
    true_positives  = sum(1 for p, l in zip(all_paths, gt_labels) if l == 1 and p in pred_anomalous_set)
    false_positives = sum(1 for p, l in zip(all_paths, gt_labels) if l == 0 and p in pred_anomalous_set)
    false_negatives = sum(1 for p, l in zip(all_paths, gt_labels) if l == 1 and p not in pred_anomalous_set)
    precision = true_positives / max(len(pred_anomalous), 1)
    recall    = true_positives / max(total_injected, 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-8)

    print(f"  Recall={recall:.2f}  Precision={precision:.2f}  F1={f1:.2f}  "
          f"(caught {true_positives}/{total_injected}, fp={false_positives})")

    # Save path lists
    (out_dir / f"{category}_normal_paths.txt").write_text("\n".join(pred_normal))
    (out_dir / f"{category}_anomalous_paths.txt").write_text("\n".join(pred_anomalous))

    return {
        "category":        category,
        "injected":        total_injected,
        "true_positives":  true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "recall":          recall,
        "precision":       precision,
        "f1":              f1,
    }


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main(args):
    device    = get_device()
    data_root = Path(args.data_root)
    out_dir   = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    categories = TEXTILE_CATEGORIES if args.category == "all" else \
                 [c.strip() for c in args.category.split(",")]

    # Validate that only textile categories are requested
    invalid = [c for c in categories if c not in TEXTILE_CATEGORIES]
    if invalid:
        raise ValueError(f"Non-textile categories requested: {invalid}. "
                         f"Valid options: {TEXTILE_CATEGORIES}")

    print(f"\n{'=' * 55}")
    print(f"  Stage 1 — GMM Training Set Separation (Textile)")
    print(f"  Categories : {categories}")
    print(f"  Device     : {device}")
    print(f"{'=' * 55}")

    print("\nLoading DINOv2 (once for all categories) ...")
    model = load_dinov2(device)

    results = []
    for category in categories:
        try:
            r = run_category(category, data_root, out_dir, model, device)
            results.append(r)
        except Exception as e:
            print(f"  ERROR on {category}: {e}")

    # Summary table
    print("\n" + "=" * 70)
    print(f"  {'CATEGORY':<18} {'INJECTED':>8} {'CAUGHT':>7} {'FP':>5} {'RECALL':>7} {'PREC':>7} {'F1':>6}")
    print("  " + "-" * 68)
    for r in results:
        print(f"  {r['category']:<18} {r['injected']:>8} {r['true_positives']:>7} "
              f"{r['false_positives']:>5} {r['recall']:>7.2f} {r['precision']:>7.2f} {r['f1']:>6.2f}")
    if len(results) > 1:
        avg_recall    = sum(r["recall"]    for r in results) / len(results)
        avg_precision = sum(r["precision"] for r in results) / len(results)
        avg_f1        = sum(r["f1"]        for r in results) / len(results)
        print("  " + "-" * 68)
        print(f"  {'AVERAGE':<18} {'':>8} {'':>7} {'':>5} "
              f"{avg_recall:>7.2f} {avg_precision:>7.2f} {avg_f1:>6.2f}")
    print("=" * 70 + "\n")
    print("  Done. Output saved to:", out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stage 1: GMM-based training set cleaning for textile anomaly detection."
    )
    parser.add_argument("--data_root",   type=str, required=True,
                        help="Path to MVTec root directory")
    parser.add_argument("--category",    type=str, default="all",
                        help="Textile category name, comma-separated list, or 'all' (default). "
                             f"Valid: {TEXTILE_CATEGORIES}")
    parser.add_argument("--output_dir",  type=str, default="./stage1_output",
                        help="Where to save normal/anomalous path lists")
    args = parser.parse_args()
    main(args)
