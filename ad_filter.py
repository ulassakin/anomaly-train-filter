"""
Stage 1: Separate normal vs. anomalous images from contaminated training data.

What this does:
  1. Load all training images (contaminated — mix of normal + some defective)
  2. Extract DINOv2 patch features for each image
  3. Fit PCA + GMM on all patch features (iterative trimming to handle contamination)
  4. Score each IMAGE by aggregating its patch scores (mean of top-k worst patches)
  5. Threshold: bottom X% of images = normal, top = anomalous
  6. Save two lists: normal_paths.txt and anomalous_paths.txt

Usage:
    python gmm_stage1.py --data_root /path/to/mvtec --category carpet
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
from scipy.ndimage import binary_dilation, binary_closing

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DINO_MODEL      = "vit_small_patch14_dinov2.lvd142m"
IMG_SIZE        = 518
PCA_DIM         = 64
GMM_COMPONENTS  = 9
TRIM_PERCENTILE      = 85   # During iterative trimming, keep bottom X% patches each round
TRIM_ITERATIONS      = 3    # How many refitting rounds
RANDOM_SEED          = 42


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
    Automatically builds a contaminated training set:
      - Normal images from train/good
      - Defective images injected from test/ subfolders (~10% by default)
    Also returns ground truth labels so you can evaluate separation quality.
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


def get_grid_dims(n_patches):
    """
    Find (rows, cols) for the patch grid given total patch count.
    DINOv2 with patch_size=14 on a square image of size S gives
    grid = (S // 14) x (S // 14) patches. But different categories
    may use different image sizes, so we find the closest square
    factorisation rather than assuming sqrt is exact.
    """
    side = int(np.sqrt(n_patches))
    # Try side x side first (perfect square)
    if side * side == n_patches:
        return side, side
    # Try side x (side+1) and (side+1) x side
    if side * (side + 1) == n_patches:
        return side, side + 1
    if (side + 1) * side == n_patches:
        return side + 1, side
    # Fallback: trim to nearest perfect square
    return side, side


def compute_foreground_mask(patch_features):
    """
    Compute a foreground mask using the first PCA component of patch features.
    This is exactly the AnomalyDINO masking approach — no extra model needed,
    DINOv2 patch features separate foreground/background in their first PC.

    Steps:
      1. PCA on patch features → take first component
      2. Reshape into spatial grid (handles non-square grids)
      3. Threshold at mean to get binary mask
      4. Ensure foreground is the center (not background) — flip if needed
      5. Apply dilation + morphological closing to fill holes and gaps

    Returns a boolean mask of shape (N_patches,) — True = foreground (keep)
    """
    n_patches = patch_features.shape[0]
    rows, cols = get_grid_dims(n_patches)
    n_use = rows * cols  # may be less than n_patches if non-square fallback

    # Step 1: first PCA component
    pca_mask = PCA(n_components=1)
    first_pc = pca_mask.fit_transform(patch_features[:n_use]).reshape(rows, cols)

    # Step 2: threshold at mean
    binary = first_pc > first_pc.mean()

    # Step 3: ensure center is foreground — flip if needed
    cr, cc = rows // 4, cols // 4
    center = binary[cr:-cr, cc:-cc]
    if center.mean() < 0.5:
        binary = ~binary

    # Step 4: morphological cleanup
    binary = binary_dilation(binary, iterations=2)
    binary = binary_closing(binary, iterations=2)

    # Build final mask — any trimmed patches default to True (keep)
    mask = np.ones(n_patches, dtype=bool)
    mask[:n_use] = binary.reshape(-1)
    return mask


@torch.no_grad()
def extract_features(model, dataloader, device, use_mask=False):
    """
    Returns:
        all_patch_features : list of (N_patches, D) arrays — one per image
                             If use_mask=True, background patches are zeroed
                             and a separate mask list is returned
        all_paths          : list of str, length N_images
        all_masks          : list of (N_patches,) boolean arrays (None if use_mask=False)
    """
    all_patch_features = []
    all_paths = []
    all_masks = []

    for imgs, paths in tqdm(dataloader, desc="Extracting features"):
        imgs = imgs.to(device)
        feats = model.get_intermediate_layers(imgs, n=1)[0]  # (B, N_patches+1, D)
        patch_feats = feats[:, 1:, :].cpu().numpy()          # (B, N_patches, D)

        for i in range(len(paths)):
            pf = patch_feats[i]   # (N_patches, D)

            if use_mask:
                mask = compute_foreground_mask(pf)
                all_masks.append(mask)
            else:
                all_masks.append(None)

            all_patch_features.append(pf)
            all_paths.append(paths[i])

    return all_patch_features, all_paths, all_masks


# ─────────────────────────────────────────────
# GMM WITH ITERATIVE TRIMMING
# ─────────────────────────────────────────────
def fit_gmm_iterative(all_patch_features):
    """
    Fit GMM iteratively to handle contamination:
      - Round 1: fit on everything
      - Score all patches, discard top (100-TRIM_PERCENTILE)% highest-scoring ones
      - Round 2+: refit on remaining patches only
    This gradually pushes anomalous patches out of the fit.
    """
    # Stack all patches into one big matrix
    flat_features = np.vstack(all_patch_features)  # (total_patches, D)
    print(f"  Total patches: {flat_features.shape[0]:,}  |  dim: {flat_features.shape[1]}")

    # PCA once — fit on everything, don't refit each round
    print(f"  Fitting PCA ({flat_features.shape[1]}D → {PCA_DIM}D) ...")
    pca = PCA(n_components=PCA_DIM, random_state=RANDOM_SEED)
    reduced = pca.fit_transform(flat_features)
    print(f"  Explained variance: {pca.explained_variance_ratio_.cumsum()[-1]:.3f}")

    # Iterative trimming
    mask = np.ones(len(reduced), dtype=bool)  # start: keep all patches

    for round_idx in range(TRIM_ITERATIONS):
        print(f"  GMM round {round_idx+1}/{TRIM_ITERATIONS}  "
              f"({mask.sum():,} patches kept) ...")
        gmm = GaussianMixture(
            n_components=GMM_COMPONENTS,
            covariance_type="full",
            max_iter=300,
            random_state=RANDOM_SEED,
        )
        gmm.fit(reduced[mask])

        # Score ALL patches (not just kept ones) so we can update the mask
        scores = -gmm.score_samples(reduced)  # higher = more anomalous
        threshold = np.percentile(scores[mask], TRIM_PERCENTILE)
        mask = scores < threshold

    print(f"  Final GMM fit on {mask.sum():,} patches  "
          f"({100*mask.mean():.1f}% of total)")
    return pca, gmm, mask


# ─────────────────────────────────────────────
# IMAGE-LEVEL SCORING
# ─────────────────────────────────────────────
def score_images(pca, gmm, all_patch_features, all_masks, category):
    """
    Scoring strategy depends on category type:

    TEXTURES (carpet, grid, leather, tile, wood):
      No masking. Fraction of anomalous patches above global threshold.
      Captures spatially spread defects (scratches, holes, stains).

    OBJECTS with masking (capsule, hazelnut, pill, screw, toothbrush):
      Apply foreground mask — ignore background patches entirely.
      Then top-1 max patch score on foreground patches only.
      Masking removes the background noise that was drowning out defect signal.

    OBJECTS without masking (bottle, cable, metal_nut, transistor, zipper):
      DINOv2 masking fails for these (per AnomalyDINO paper).
      Fall back to top-1 max patch score on all patches.
    """
    if category in TEXTURE_CATEGORIES:
        print(f"  Scoring strategy: fraction of anomalous patches  (texture mode)")

        all_patch_scores = []
        for patch_features in all_patch_features:
            reduced = pca.transform(patch_features)
            patch_scores = -gmm.score_samples(reduced)
            all_patch_scores.append(patch_scores)

        patch_threshold = np.percentile(np.concatenate(all_patch_scores), PATCH_ANOMALY_PCTILE)
        print(f"  Patch anomaly threshold (p{PATCH_ANOMALY_PCTILE}): {patch_threshold:.3f}")

        image_scores = []
        for patch_scores in all_patch_scores:
            image_scores.append((patch_scores > patch_threshold).mean())

    elif category in MASK_CATEGORIES:
        print(f"  Scoring strategy: top-1 max patch score + foreground mask  (object mode)")

        image_scores = []
        for patch_features, mask in zip(all_patch_features, all_masks):
            reduced = pca.transform(patch_features)
            patch_scores = -gmm.score_samples(reduced)
            # Only consider foreground patches
            fg_scores = patch_scores[mask] if mask.sum() > 0 else patch_scores
            image_scores.append(fg_scores.max())

    else:
        # NO_MASK_CATEGORIES — masking fails, use all patches
        print(f"  Scoring strategy: top-1 max patch score  (object mode, no mask)")

        image_scores = []
        for patch_features in all_patch_features:
            reduced = pca.transform(patch_features)
            patch_scores = -gmm.score_samples(reduced)
            image_scores.append(patch_scores.max())

    return np.array(image_scores)


# ─────────────────────────────────────────────
# ALL MVTEC CATEGORIES
# ─────────────────────────────────────────────
MVTEC_CATEGORIES = [
    "carpet", "grid", "leather", "tile", "wood",          # textures
    "bottle", "cable", "capsule", "hazelnut", "metal_nut", # objects
    "pill", "screw", "toothbrush", "transistor", "zipper", # objects
]

# Texture categories: spread-out defects, no masking needed
# Object categories:  localized defects, mask out background
# Some object categories fail the DINOv2 masking test (per AnomalyDINO paper Table 8)
TEXTURE_CATEGORIES = {"carpet", "grid", "leather", "tile", "wood"}
MASK_CATEGORIES    = {"capsule", "hazelnut", "pill", "screw", "toothbrush"}  # masking works
NO_MASK_CATEGORIES = {"bottle", "cable", "metal_nut", "transistor", "zipper"}  # masking fails
PATCH_ANOMALY_PCTILE = 90   # threshold percentile for texture fraction scoring

# Brightness normalization helps categories with lighting variation (e.g. carpet)
# but hurts categories with uniform, consistent lighting (e.g. grid)
BRIGHTNESS_NORM_CATEGORIES = {"carpet", "leather", "tile", "wood"}


# ─────────────────────────────────────────────
# PER-CATEGORY LOGIC
# ─────────────────────────────────────────────
def run_category(category, data_root, out_dir, model, device):
    print(f"\n{'='*55}")
    print(f"  Category: {category}")
    print(f"{'='*55}")

    # 1. Load
    print("[1/4] Loading training images ...")
    paths, gt_labels = load_training_images(data_root, category)

    # Build transform — brightness normalization only for categories
    # where lighting variation is a known source of false positives.
    # Grid has very consistent lighting so normalization hurts it.
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
    use_mask = category in MASK_CATEGORIES
    all_patch_features, all_paths, all_masks = extract_features(model, loader, device, use_mask=use_mask)
    if use_mask:
        n_masked = sum(m.mean() for m in all_masks) / len(all_masks)
        print(f"  Foreground mask applied — avg foreground: {100*n_masked:.1f}% of patches")

    # 3. Fit GMM
    print("\n[3/4] Fitting GMM (iterative trimming) ...")
    pca, gmm, trim_mask = fit_gmm_iterative(all_patch_features)

    # 4. Score and split
    print("\n[4/4] Scoring images ...")
    image_scores = score_images(pca, gmm, all_patch_features, all_masks, category)

    score_gmm = GaussianMixture(n_components=2, random_state=RANDOM_SEED)
    score_gmm.fit(image_scores.reshape(-1, 1))

    normal_component    = np.argmin(score_gmm.means_)
    anomalous_component = np.argmax(score_gmm.means_)
    assignments = score_gmm.predict(image_scores.reshape(-1, 1))

    pred_normal    = [p for p, a in zip(all_paths, assignments) if a == normal_component]
    pred_anomalous = [p for p, a in zip(all_paths, assignments) if a == anomalous_component]

    print(f"\n  Normal cluster mean   : {score_gmm.means_[normal_component][0]:.3f}")
    print(f"  Anomalous cluster mean: {score_gmm.means_[anomalous_component][0]:.3f}")

    # Evaluate
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

    # Save
    (out_dir / f"{category}_normal_paths.txt").write_text("\n".join(pred_normal))
    (out_dir / f"{category}_anomalous_paths.txt").write_text("\n".join(pred_anomalous))

    return {
        "category":       category,
        "injected":       total_injected,
        "true_positives": true_positives,
        "false_positives":false_positives,
        "false_negatives":false_negatives,
        "recall":         recall,
        "precision":      precision,
        "f1":             f1,
    }


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main(args):
    device    = get_device()
    data_root = Path(args.data_root)
    out_dir   = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Decide which categories to run
    if args.category == "all":
        categories = MVTEC_CATEGORIES
    else:
        categories = [c.strip() for c in args.category.split(",")]

    print(f"\n{'='*55}")
    print(f"  Stage 1 — GMM Training Set Separation")
    print(f"  Categories : {categories}")
    print(f"  Device     : {device}")
    print(f"{'='*55}")

    # Load model once, reuse across all categories
    print("\nLoading DINOv2 (once for all categories) ...")
    model = load_dinov2(device)

    # Run
    results = []
    for category in categories:
        try:
            r = run_category(category, data_root, out_dir, model, device)
            results.append(r)
        except Exception as e:
            print(f"  ERROR on {category}: {e}")

    # Summary table
    print("\n" + "="*70)
    print(f"  {'CATEGORY':<18} {'INJECTED':>8} {'CAUGHT':>7} {'FP':>5} {'RECALL':>7} {'PREC':>7} {'F1':>6}")
    print("  " + "-"*68)
    for r in results:
        print(f"  {r['category']:<18} {r['injected']:>8} {r['true_positives']:>7} "
              f"{r['false_positives']:>5} {r['recall']:>7.2f} {r['precision']:>7.2f} {r['f1']:>6.2f}")
    if len(results) > 1:
        avg_recall    = sum(r['recall']    for r in results) / len(results)
        avg_precision = sum(r['precision'] for r in results) / len(results)
        avg_f1        = sum(r['f1']        for r in results) / len(results)
        print("  " + "-"*68)
        print(f"  {'AVERAGE':<18} {'':>8} {'':>7} {'':>5} {avg_recall:>7.2f} {avg_precision:>7.2f} {avg_f1:>6.2f}")
    print("="*70 + "\n")

    print("  Done. normal_paths.txt files saved to:", out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stage 1: GMM-based training set cleaning for anomaly detection."
    )
    parser.add_argument("--data_root",   type=str, required=True,
                        help="Path to MVTec root directory")
    parser.add_argument("--category",    type=str, default="all",
                        help="Category name, comma-separated list, or 'all' (default)")
    parser.add_argument("--output_dir",  type=str, default="./stage1_output",
                        help="Where to save normal/anomalous path lists")
    args = parser.parse_args()
    main(args)
