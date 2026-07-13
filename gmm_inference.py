"""
Stage 1 Evaluation: GMM inference on MVTec test set.

Loads the PCA + GMM fitted by ad_filter.py and scores unseen test images.
The GMM is never refit here — it runs exactly as it would in deployment.

What this measures:
  - AUROC : image-level detection  (is this image defective?)
  - AUPRO : pixel-level localization  (where is the defect?)

AUPRO (Area Under Per-Region Overlap) is the standard localization metric
for MVTec-AD. It measures how well the anomaly heatmap aligns with the
ground truth pixel mask, averaged over overlap thresholds.

Usage:
    python gmm_inference.py --data_root /path/to/mvtec --category carpet
    python gmm_inference.py --data_root /path/to/mvtec --category all

Output per category:
  - AUROC + AUPRO printed to terminal
  - {category}_scores.csv              — per-image scores + labels + subtype
  - {category}_score_distribution.png  — normal vs defective score histogram
  - heatmaps/{category}/{name}.png     — anomaly heatmap overlaid on original
                                         (only for defective test images)

Prerequisites:
    Run ad_filter.py first to generate {category}_gmm.pkl files.
    pip install scikit-image scipy  (for AUPRO and Gaussian smoothing)
"""

import argparse
import csv
import warnings
from pathlib import Path

import joblib
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# CONFIG  (must match ad_filter.py exactly)
# ─────────────────────────────────────────────
DINO_MODEL             = "vit_small_patch14_dinov2.lvd142m"
IMG_SIZE               = 518
PATCH_SIZE             = 14
GRID_SIZE              = IMG_SIZE // PATCH_SIZE   # 37 patches per side
PATCH_ANOMALY_PCTILE   = 90
RANDOM_SEED            = 42

TEXTILE_CATEGORIES         = ["carpet", "grid", "leather", "tile", "wood"]
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
        self.transform   = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, str(self.image_paths[idx])


def load_test_images(data_root: Path, category: str):
    """
    Loads all test images with labels and defect subtype names.

    MVTec test structure:
        test/good/           -> label 0 (normal)
        test/{defect_type}/  -> label 1 (anomalous)

    Returns:
        paths    : list of Path
        labels   : list of int  (0=normal, 1=defective)
        subtypes : list of str  (e.g. "hole", "cut", "good")
    """
    test_dir = data_root / category / "test"
    paths, labels, subtypes = [], [], []

    for subdir in sorted(test_dir.iterdir()):
        if not subdir.is_dir():
            continue
        label = 0 if subdir.name == "good" else 1
        for ext in ["*.png", "*.jpg", "*.PNG", "*.JPG"]:
            for p in sorted(subdir.glob(ext)):
                paths.append(p)
                labels.append(label)
                subtypes.append(subdir.name)

    print(f"  Test images — normal: {labels.count(0)}  "
          f"defective: {labels.count(1)}  total: {len(paths)}")
    return paths, labels, subtypes


def load_gt_mask(data_root: Path, category: str,
                 image_path: Path) -> np.ndarray:
    """
    Loads the pixel-level ground truth segmentation mask for a defective image.

    MVTec ground truth structure:
        ground_truth/{defect_type}/{image_stem}_mask.png

    Returns binary float32 mask (H, W), or None if not found.
    """
    defect_type = image_path.parent.name
    if defect_type == "good":
        return None

    mask_path = (data_root / category / "ground_truth" /
                 defect_type / (image_path.stem + "_mask.png"))

    if not mask_path.exists():
        return None

    mask = Image.open(mask_path).convert("L").resize(
        (IMG_SIZE, IMG_SIZE), Image.NEAREST
    )
    return (np.array(mask) > 0).astype(np.float32)


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
    Extracts DINOv2 patch features for all images.

    Also reads the actual patch grid shape (H_patches, W_patches) directly
    from the model's patch embedding layer — avoids guessing from sqrt(N).

    Returns:
        all_patch_features : list of (N_patches, D) arrays
        all_paths          : list of str
        grid_shape         : (H_patches, W_patches) tuple
    """
    all_patch_features, all_paths = [], []
    grid_shape = None

    for imgs, paths in tqdm(dataloader, desc="  Extracting features"):
        imgs        = imgs.to(device)
        feats       = model.get_intermediate_layers(imgs, n=1)[0]  # (B, N+1, D)
        patch_feats = feats[:, 1:, :].cpu().numpy()                # (B, N, D)

        # Read actual grid shape from model once
        if grid_shape is None:
            if hasattr(model, "patch_embed") and hasattr(model.patch_embed, "grid_size"):
                grid_shape = tuple(model.patch_embed.grid_size)
            else:
                # Fallback: assume square grid, trim to nearest perfect square
                n = patch_feats.shape[1]
                side = int(np.sqrt(n))
                grid_shape = (side, side)

        for i in range(len(paths)):
            all_patch_features.append(patch_feats[i])
            all_paths.append(paths[i])

    print(f"  Patch grid shape: {grid_shape[0]}h x {grid_shape[1]}w "
          f"= {grid_shape[0]*grid_shape[1]} patches")
    return all_patch_features, all_paths, grid_shape


# ─────────────────────────────────────────────
# SCORING + HEATMAP GENERATION
# ─────────────────────────────────────────────
def patch_scores_to_heatmap(patch_scores: np.ndarray,
                             grid_shape: tuple,
                             output_size: int = IMG_SIZE) -> np.ndarray:
    """
    Converts flat patch score vector to a spatial (H, W) heatmap.

    Uses the actual grid shape from the model (e.g. 36x38) rather than
    assuming a square grid — this ensures correct spatial alignment with
    the original image and with ground truth masks.

    Steps:
      1. Reshape to (grid_h, grid_w) using actual model grid dimensions
      2. Gaussian smoothing (sigma=1) to soften blocky patch boundaries
      3. Bilinear upsample to (output_size, output_size)

    Returns float32 array of shape (output_size, output_size).
    """
    from scipy.ndimage import gaussian_filter

    grid_h, grid_w = grid_shape
    n_expected = grid_h * grid_w

    # Pad with zero if one patch is missing (occasional DINOv2 behaviour)
    scores = patch_scores.astype(np.float32)
    if len(scores) < n_expected:
        scores = np.pad(scores, (0, n_expected - len(scores)))

    # 1. Reshape using actual grid dimensions
    grid = scores[:n_expected].reshape(grid_h, grid_w)

    # 2. Smooth
    grid = gaussian_filter(grid, sigma=1)

    # 3. Upsample
    heatmap = np.array(
        Image.fromarray(grid).resize((output_size, output_size), Image.BILINEAR)
    )
    return heatmap


TOPK_FRACTION = 0.05   # top 5% of patches used for image-level aggregation


def score_all_images(pca, gmm, all_patch_features, grid_shape):
    """
    Scores all images and returns image-level scores + per-image heatmaps.

    Image score = mean of the top-k worst patch scores per image,
                  where k = TOPK_FRACTION * total patches (~68 of 1369).

    Why top-k mean instead of global mean or fraction-above-threshold:

      Global mean: defect patches (~5% of the image) get averaged with
      ~1300 normal patches, diluting the signal by ~20x. The gap between
      normal and defective image scores becomes tiny.

      Fraction-above-threshold: binarizes patch scores against a global
      percentile, compressing all image scores near zero and creating a
      tradeoff between AUROC and score spread.

      Top-k mean: focuses on the worst-scoring patch region (where the
      defect is) while averaging over enough patches to be robust to
      single-patch noise. Preserves full magnitude of the anomaly signal
      without diluting it with surrounding normal patches. The gap between
      normal and defective image scores is ~18x larger than global mean
      on typical localized defect patterns.

    The heatmap still uses raw per-patch scores — top-k only affects
    the single image-level aggregate value.

    Returns:
        image_scores : (N,) float array  — top-k mean patch score per image
        all_heatmaps : list of (IMG_SIZE, IMG_SIZE) float32 arrays
        None         : placeholder (patch_threshold no longer used)
    """
    image_scores = []
    all_heatmaps = []

    for pf in all_patch_features:
        reduced      = pca.transform(pf)
        patch_scores = -gmm.score_samples(reduced)      # higher = more anomalous
        k            = max(1, int(len(patch_scores) * TOPK_FRACTION))
        topk_score   = np.partition(patch_scores, -k)[-k:].mean()
        image_scores.append(topk_score)
        all_heatmaps.append(patch_scores_to_heatmap(patch_scores, grid_shape))

    return np.array(image_scores), all_heatmaps, None


# ─────────────────────────────────────────────
# AUPRO COMPUTATION
# ─────────────────────────────────────────────
def compute_aupro(heatmaps, gt_masks, num_thresholds=100):
    """
    Computes AUPRO: Area Under the Per-Region Overlap curve.

    Standard localization metric for MVTec-AD. Unlike pixel-level AUROC,
    AUPRO weights each connected defect region equally regardless of its
    size — so a small pin scratch contributes as much as a large stain.

    Integration is performed up to FPR=0.3 (MVTec convention), then
    normalized to [0, 1].

    Args:
        heatmaps      : list of (H, W) float arrays — anomaly heatmaps
        gt_masks      : list of (H, W) float arrays or None
                        (None for normal images that have no defect)
        num_thresholds: how many threshold values to sweep

    Returns:
        aupro : float, or None if no masks were found
    """
    from skimage.measure import label as skimage_label

    defect_heatmaps = [h for h, m in zip(heatmaps, gt_masks) if m is not None]
    defect_masks    = [m for m in gt_masks if m is not None]
    normal_heatmaps = [h for h, m in zip(heatmaps, gt_masks) if m is None]

    if len(defect_masks) == 0:
        print("  WARNING: No ground truth masks found — skipping AUPRO")
        return None

    # Threshold sweep range
    all_vals   = np.concatenate([h.ravel() for h in heatmaps])
    thresholds = np.linspace(all_vals.min(), all_vals.max(), num_thresholds)

    pro_values = []
    fpr_values = []

    for thresh in thresholds:
        # ── Per-region overlap ──────────────────────────────────────────
        region_overlaps = []
        for heatmap, gt_mask in zip(defect_heatmaps, defect_masks):
            pred_mask  = (heatmap > thresh).astype(np.uint8)
            labeled_gt, n_regions = skimage_label(
                gt_mask, return_num=True, connectivity=2
            )
            for rid in range(1, n_regions + 1):
                region  = (labeled_gt == rid)
                overlap = pred_mask[region].mean()
                region_overlaps.append(float(overlap))

        pro = float(np.mean(region_overlaps)) if region_overlaps else 0.0

        # ── False positive rate on normal pixels ────────────────────────
        if normal_heatmaps:
            normal_pixels = np.concatenate([h.ravel() for h in normal_heatmaps])
            fpr = float((normal_pixels > thresh).mean())
        else:
            # Fallback: use non-defect pixels from defective images
            fp_pixels = []
            for heatmap, gt_mask in zip(defect_heatmaps, defect_masks):
                bg = (gt_mask == 0)
                if bg.any():
                    fp_pixels.append(heatmap[bg])
            fpr = float((np.concatenate(fp_pixels) > thresh).mean()) \
                  if fp_pixels else 0.0

        pro_values.append(pro)
        fpr_values.append(fpr)

    fpr_arr = np.array(fpr_values)
    pro_arr = np.array(pro_values)

    # Sort by FPR for integration
    order    = np.argsort(fpr_arr)
    fpr_sort = fpr_arr[order]
    pro_sort = pro_arr[order]

    # Integrate up to FPR = 0.3 and normalize
    fpr_limit = 0.3
    valid     = fpr_sort <= fpr_limit
    if valid.sum() < 2:
        return float(np.mean(pro_sort))

    aupro = float(np.trapz(pro_sort[valid], fpr_sort[valid]) / fpr_limit)
    return aupro


# ─────────────────────────────────────────────
# HEATMAP VISUALIZATION
# ─────────────────────────────────────────────
def save_heatmap_overlay(original_path: Path, heatmap: np.ndarray,
                          gt_mask, save_path: Path, image_score: float):
    """
    Saves a side-by-side visualization:
      [Original] | [Heatmap overlay] | [GT mask]

    The heatmap is blended over the original image using a red colormap.
    Brighter red = higher anomaly score.
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize

    orig     = np.array(
        Image.open(original_path).convert("RGB").resize(
            (IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    )
    norm     = Normalize(vmin=heatmap.min(), vmax=heatmap.max())
    colored  = cm.get_cmap("Reds")(norm(heatmap))[:, :, :3]
    overlay  = np.clip(0.6 * (orig / 255.0) + 0.4 * colored, 0, 1)

    n_panels = 3 if gt_mask is not None else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))

    axes[0].imshow(orig);          axes[0].set_title("Original");   axes[0].axis("off")
    axes[1].imshow(overlay);       axes[1].set_title(f"Heatmap  (score: {image_score:.4f})"); axes[1].axis("off")
    if gt_mask is not None:
        axes[2].imshow(gt_mask, cmap="gray")
        axes[2].set_title("GT mask")
        axes[2].axis("off")

    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)



# ─────────────────────────────────────────────
# VALLEY THRESHOLD  (KDE + persistence + dip-test)
# ─────────────────────────────────────────────
def find_valley_threshold(image_scores: np.ndarray,
                           n_normal: int = None,
                           n_bins: int = 200,
                           save_path: Path = None,
                           category: str = "") -> float:
    """
    Finds the natural valley between normal and anomalous image scores
    using a multi-bandwidth KDE persistence method, with a percentile-
    based fallback for degenerate (non-bimodal) cases.

    WHY NOT THE OLD RAW-HISTOGRAM APPROACH:
    With only ~20-120 image scores per category, a 200-bin raw histogram
    is mostly noise — most bins hold 0-2 points. "First local minimum"
    on that noisy curve reliably lands on the first small wiggle right
    past the search-start cutoff, not on the true gap between normal and
    defective populations. This was validated against MVTec runs: the
    old method's threshold consistently sat within ~10% of search_start,
    while the true normal/defective boundary was often much further out.

    THE NEW METHOD, IN THREE STAGES:

    1. DIP TEST GATEKEEPER (Hartigan's dip test)
       Before looking for any valley, test whether the score distribution
       is even statistically multimodal. If it isn't (p > dip_alpha), a
       "valley" would just be noise — don't trust the KDE search at all,
       fall back to a conservative percentile.

    2. MULTI-BANDWIDTH KDE PERSISTENCE
       Instead of one histogram, fit a smooth Kernel Density Estimate at
       MANY bandwidths, from narrow (close to raw histogram, may show
       spurious noise dips) to wide (heavily smoothed, may merge real
       structure away). At each bandwidth, record every interior local
       minimum (a "candidate valley") along with how deep it is relative
       to its flanking peaks.

       A genuine structural valley (the real gap between normal and
       defective) tends to appear at roughly the same location across
       MOST bandwidths — it's real signal, so smoothing doesn't erase it
       until the bandwidth gets very large. A noise-driven dip only shows
       up at one or two narrow bandwidths and disappears as soon as you
       smooth slightly more.

       We cluster nearby candidates (across all bandwidths) by location,
       then score each cluster by (how many distinct bandwidths it
       persisted across) x (its average relative depth). The highest-
       scoring cluster wins — this is the valley that is both DEEP and
       STABLE, not just the first dip encountered.

    3. FALLBACK
       If the dip test says the data is unimodal, or no cluster survives
       the minimum-persistence bar, fall back to the old percentile-based
       heuristic (search above a class-balance-informed percentile, clamp
       with a normal-region safety floor). This guarantees the function
       always returns a usable threshold, even on distributions with no
       genuine bimodal structure.

    Args:
        image_scores : all image-level anomaly scores (normal + defective)
        n_normal     : number of normal images, used both by the fallback
                       percentile method and to sanity-check the KDE result
                       against a safety floor (see step 3 below in code).
        n_bins       : histogram resolution, only used for the fallback
                       and for the diagnostic plot's raw-histogram panel.
        save_path    : if given, saves a diagnostic plot of the search
                       process to this path (directory).
        category     : category name, used in the plot title/filename.

    Returns:
        threshold : float
    """
    result = _kde_persistence_valley(image_scores)

    normal_region = image_scores
    search_start = None
    fallback_threshold = None
    fallback_used = False

    if result is not None:
        threshold, n_persist, depth, dip_p = result
        print(f"  KDE valley found: {threshold:.2f}  "
              f"(persisted {n_persist} bandwidths, depth={depth:.3f}, dip p={dip_p:.4f})")
    else:
        fallback_used = True
        threshold, search_start = _percentile_fallback_valley(image_scores, n_normal, n_bins)
        fallback_threshold = threshold
        print(f"  KDE found no reliable valley — using percentile fallback: {threshold:.2f}")

    # ── Safety floor, same spirit as before: threshold should not sit
    # below the bulk of the normal-appearing region. Approximate the
    # normal region as everything below the KDE/fallback threshold's own
    # lower-percentile neighborhood, then require the threshold to clear
    # its p99 — protects against a KDE valley accidentally landing inside
    # the normal cluster on a weird distribution.
    below = image_scores[image_scores <= threshold]
    if n_normal is not None and n_normal > 0:
        approx_normal_ceiling = float(np.percentile(np.sort(image_scores)[:n_normal], 99)) \
            if n_normal <= len(image_scores) else None
        if approx_normal_ceiling is not None and threshold < approx_normal_ceiling:
            print(f"  Threshold below normal-p99 ceiling ({approx_normal_ceiling:.2f}) — raising.")
            threshold = approx_normal_ceiling

    print(f"  Valley threshold: {threshold:.2f}")

    if save_path is not None:
        try:
            _plot_valley_diagnostic_kde(
                image_scores=image_scores,
                threshold=threshold,
                fallback_used=fallback_used,
                kde_result=result,
                n_normal=n_normal,
                category=category,
                save_path=save_path,
            )
        except ImportError:
            print("  (matplotlib not available — skipping valley diagnostic plot)")

    return threshold


def _kde_persistence_valley(image_scores: np.ndarray,
                             n_bandwidths: int = 15,
                             bw_range: tuple = (0.08, 0.5),
                             min_persistence_frac: float = 0.3,
                             dip_alpha: float = 0.10):
    """
    Core multi-bandwidth KDE persistence search. See find_valley_threshold
    docstring for the full rationale.

    Returns (threshold, n_bandwidths_persisted, relative_depth, dip_pvalue)
    on success, or None if the data doesn't show reliable bimodal structure.
    """
    try:
        import diptest
    except ImportError:
        print("  (diptest not installed — skipping KDE gatekeeper, "
              "install with `pip install diptest` for the improved valley method)")
        return None

    from scipy.stats import gaussian_kde
    from scipy.signal import argrelextrema

    scores = np.asarray(image_scores, dtype=float)
    if len(scores) < 8:
        return None  # too few points for KDE bandwidth sweep to be meaningful

    # ── Stage 1: dip test gatekeeper ──
    _, dip_p = diptest.diptest(scores)
    if dip_p > dip_alpha:
        return None

    lo, hi = scores.min(), scores.max()
    if hi <= lo:
        return None
    xs = np.linspace(lo, hi, 3000)
    bandwidths = np.linspace(bw_range[0], bw_range[1], n_bandwidths)

    all_candidates = []  # (x_location, bandwidth, relative_depth)
    for bw in bandwidths:
        try:
            kde = gaussian_kde(scores, bw_method=bw)
        except Exception:
            continue
        density = kde(xs)
        minima_idx = argrelextrema(density, np.less)[0]
        maxima_idx = argrelextrema(density, np.greater)[0]
        for m_idx in minima_idx:
            left_peaks = maxima_idx[maxima_idx < m_idx]
            right_peaks = maxima_idx[maxima_idx > m_idx]
            if len(left_peaks) == 0 or len(right_peaks) == 0:
                continue  # not an interior valley (no peak on one side)
            left_peak, right_peak = left_peaks[-1], right_peaks[0]
            valley_h = density[m_idx]
            peak_h = min(density[left_peak], density[right_peak])
            if peak_h <= 0:
                continue
            relative_depth = 1.0 - (valley_h / peak_h)
            all_candidates.append((xs[m_idx], bw, relative_depth))

    if not all_candidates:
        return None

    # ── Stage 2: cluster candidates by location, score by persistence x depth ──
    all_candidates.sort(key=lambda c: c[0])
    tolerance = (hi - lo) * 0.03
    clusters, current = [], [all_candidates[0]]
    for cand in all_candidates[1:]:
        if cand[0] - current[-1][0] <= tolerance:
            current.append(cand)
        else:
            clusters.append(current)
            current = [cand]
    clusters.append(current)

    min_persist_count = max(1, int(min_persistence_frac * n_bandwidths))
    best, best_score = None, -1.0
    for cluster in clusters:
        n_bw = len(set(round(c[1], 6) for c in cluster))
        avg_depth = float(np.mean([c[2] for c in cluster]))
        avg_loc = float(np.mean([c[0] for c in cluster]))
        if n_bw < min_persist_count:
            continue
        score = n_bw * avg_depth
        if score > best_score:
            best_score = score
            best = (avg_loc, n_bw, avg_depth)

    if best is None:
        return None

    loc, n_bw, depth = best
    return loc, n_bw, depth, dip_p


def _percentile_fallback_valley(image_scores: np.ndarray,
                                 n_normal: int = None,
                                 n_bins: int = 200):
    """
    The original percentile + first-local-minimum heuristic, kept as a
    fallback for cases where the KDE method finds no reliable bimodal
    structure (e.g. diptest not installed, or genuinely unimodal data).
    Same logic as the prior implementation and as ad_filter.py.
    """
    from scipy.ndimage import uniform_filter1d

    n_total = len(image_scores)
    if n_normal is not None:
        search_pctile = (n_normal / n_total) * 100 + 10
        search_pctile = min(search_pctile, 60)
    else:
        search_pctile = 30.0

    search_start = float(np.percentile(image_scores, search_pctile))

    counts, bin_edges = np.histogram(image_scores, bins=n_bins)
    smoothed = uniform_filter1d(counts.astype(float), size=5)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    start_idx = int(np.searchsorted(bin_centers, search_start))
    start_idx = min(start_idx, len(smoothed) - 4)

    threshold = None
    for j in range(start_idx + 1, len(smoothed) - 1):
        if smoothed[j] <= smoothed[j - 1] and smoothed[j] <= smoothed[j + 1]:
            threshold = float(bin_centers[j])
            break

    if threshold is None:
        threshold = search_start

    normal_region = image_scores[image_scores <= search_start]
    if len(normal_region) > 0:
        floor = float(np.percentile(normal_region, 99))
        if threshold < floor:
            threshold = floor

    return threshold, search_start


def _plot_valley_diagnostic_kde(image_scores, threshold, fallback_used,
                                 kde_result, n_normal, category, save_path):
    """
    Saves a diagnostic plot of the KDE persistence search:
      - raw histogram for context
      - the KDE curve at the bandwidth closest to the winning cluster's
        (or, for the fallback case, a note that the percentile method
        was used instead)
      - the final threshold
    """
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    scores = np.asarray(image_scores, dtype=float)
    fig, ax = plt.subplots(figsize=(10, 5.5))

    # Raw histogram for context (light bars, density-normalized to match KDE scale)
    ax.hist(scores, bins=40, color="#c9c9c9", alpha=0.5, density=True,
            label="Raw histogram (density)")

    # Overlay KDE curves at a few representative bandwidths so the
    # smoothing behavior is visible
    xs = np.linspace(scores.min(), scores.max(), 1000)
    for bw, style, lw, lbl in [
        (0.15, "--", 1.0, "KDE bw=0.15 (narrow)"),
        (0.3, "-", 1.8, "KDE bw=0.30 (mid)"),
        (0.5, ":", 1.0, "KDE bw=0.50 (wide)"),
    ]:
        try:
            kde = gaussian_kde(scores, bw_method=bw)
            ax.plot(xs, kde(xs), linestyle=style, linewidth=lw, color="#333333",
                    alpha=0.8, label=lbl)
        except Exception:
            pass

    if fallback_used:
        ax.axvline(threshold, color="#e08020", linewidth=2.4, linestyle="-.",
                   label=f"FINAL threshold (percentile fallback) = {threshold:.2f}")
    else:
        _, n_persist, depth, dip_p = kde_result
        ax.axvline(threshold, color="#20a040", linewidth=2.4,
                   label=f"FINAL threshold (KDE valley) = {threshold:.2f}\n"
                         f"persisted {n_persist} bandwidths, depth={depth:.2f}, dip p={dip_p:.3f}")

    ax.set_xlabel("Image anomaly score")
    ax.set_ylabel("Density")
    ax.set_title(f"{category} — valley threshold search (KDE persistence method)", fontsize=12)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()

    out_path = Path(save_path) / f"{category}_valley_search.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Valley search diagnostic: {out_path}")


# ─────────────────────────────────────────────
# PER-CATEGORY EVALUATION
# ─────────────────────────────────────────────
def run_category(category, data_root, models_dir, out_dir, model, device,
                 save_heatmaps=True):
    print(f"\n{'=' * 55}")
    print(f"  Category: {category}")
    print(f"{'=' * 55}")

    # 1. Load saved PCA + GMM (never refit)
    gmm_path = models_dir / f"{category}_gmm.pkl"
    if not gmm_path.exists():
        raise FileNotFoundError(
            f"No saved GMM at {gmm_path}. "
            f"Run ad_filter.py --category {category} first."
        )
    saved = joblib.load(gmm_path)
    pca, gmm = saved["pca"], saved["gmm"]
    print(f"  Loaded GMM: {gmm.n_components} components  "
          f"PCA: {pca.n_components_} dims")

    # 2. Load test images + ground truth masks
    print("\n[1/4] Loading test images ...")
    paths, labels, subtypes = load_test_images(data_root, category)

    gt_masks = []
    for path, label in zip(paths, labels):
        gt_masks.append(None if label == 0
                        else load_gt_mask(data_root, category, path))
    print(f"  Ground truth masks loaded: "
          f"{sum(1 for m in gt_masks if m is not None)}")

    # Build transform — must match ad_filter.py exactly
    base_transforms = [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ]
    if category in BRIGHTNESS_NORM_CATEGORIES:
        base_transforms.append(
            transforms.Lambda(lambda x: x * (0.5 / (x.mean() + 1e-6)))
        )
        print("  Brightness normalization: ON")
    else:
        print("  Brightness normalization: OFF")
    base_transforms.append(
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    )
    transform = transforms.Compose(base_transforms)

    dataset = ImageFolderDataset(paths, transform)
    loader  = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)

    # 3. Extract features
    print("\n[2/4] Extracting DINOv2 patch features ...")
    all_patch_features, all_paths, grid_shape = extract_features(
        model, loader, device
    )

    # 4. Score + heatmaps
    print("\n[3/4] Scoring images and building heatmaps ...")
    image_scores, all_heatmaps, _ = score_all_images(
        pca, gmm, all_patch_features, grid_shape
    )

    # 5. AUROC (detection)
    labels_arr    = np.array(labels)
    auroc         = roc_auc_score(labels_arr, image_scores)
    normal_scores = image_scores[labels_arr == 0]
    defect_scores = image_scores[labels_arr == 1]

    # 6. AUPRO (localization)
    print("\n[4/4] Computing AUPRO ...")
    aupro = compute_aupro(all_heatmaps, gt_masks)

    # ── Helper: compute threshold-based metrics from a given threshold value
    def threshold_metrics(t):
        preds_ = (image_scores > t).astype(int)
        tp_  = int(((preds_ == 1) & (labels_arr == 1)).sum())
        fp_  = int(((preds_ == 1) & (labels_arr == 0)).sum())
        fn_  = int(((preds_ == 0) & (labels_arr == 1)).sum())
        tn_  = int(((preds_ == 0) & (labels_arr == 0)).sum())
        prec_ = tp_ / max(tp_ + fp_, 1)
        rec_  = tp_ / max(tp_ + fn_, 1)
        f1_   = 2 * prec_ * rec_ / max(prec_ + rec_, 1e-8)
        return tp_, fp_, fn_, tn_, prec_, rec_, f1_

    # ── Oracle threshold (upper-bound, uses ground truth)
    # Midpoint in the gap for separated distributions; F1-maximizing sweep
    # for overlapping ones. Reports the best possible threshold-based metrics.
    highest_normal   = image_scores[labels_arr == 0].max()
    lowest_defective = image_scores[labels_arr == 1].min()
    if lowest_defective > highest_normal:
        threshold_oracle = float((highest_normal + lowest_defective) / 2)
        oracle_method    = "midpoint (gap)"
    else:
        candidates = np.unique(image_scores)
        best_f1_o, best_t_o = 0.0, candidates[len(candidates) // 2]
        for t in candidates:
            *_, _, _, _, f1_c = threshold_metrics(t)
            if f1_c > best_f1_o:
                best_f1_o, best_t_o = f1_c, t
        threshold_oracle = float(best_t_o)
        oracle_method    = "F1-maximizing sweep (overlap)"
    tp_o, fp_o, fn_o, tn_o, prec_o, rec_o, f1_o = threshold_metrics(threshold_oracle)

    # ── Valley threshold (deployment-realistic, uses no ground truth labels)
    # Same histogram valley logic as ad_filter.py — finds the minimum-density
    # bin above the 10th percentile of image scores. No ground truth needed.
    threshold_valley = find_valley_threshold(
        image_scores, n_normal=int((labels_arr == 0).sum()),
        save_path=out_dir, category=category,
    )
    tp_v, fp_v, fn_v, tn_v, prec_v, rec_v, f1_v = threshold_metrics(threshold_valley)

    # ── Per-subtype breakdown — computed for both thresholds
    subtype_stats = {}
    for path, label, subtype, score in zip(all_paths, labels, subtypes, image_scores):
        if subtype not in subtype_stats:
            subtype_stats[subtype] = {"total": 0, "caught_oracle": 0, "caught_valley": 0, "scores": []}
        subtype_stats[subtype]["total"] += 1
        subtype_stats[subtype]["scores"].append(float(score))
        if label == 1 and score > threshold_oracle:
            subtype_stats[subtype]["caught_oracle"] += 1
        if label == 1 and score > threshold_valley:
            subtype_stats[subtype]["caught_valley"] += 1

    print(f"\n  {'─'*60}")
    print(f"  AUROC  : {auroc:.4f}   (detection ranking — threshold-free)")
    if aupro is not None:
        print(f"  AUPRO  : {aupro:.4f}   (localization overlap — threshold-free)")
    print(f"  Normal scores — mean: {normal_scores.mean():.4f}  std: {normal_scores.std():.4f}")
    print(f"  Defect scores — mean: {defect_scores.mean():.4f}  std: {defect_scores.std():.4f}")

    print(f"\n  ── Oracle threshold ({oracle_method})")
    print(f"  Threshold : {threshold_oracle:.4f}")
    print(f"  Recall    : {rec_o:.3f}   TP={tp_o}  FN={fn_o}")
    print(f"  Precision : {prec_o:.3f}   FP={fp_o}  TN={tn_o}")
    print(f"  F1        : {f1_o:.3f}")

    print(f"\n  ── Valley threshold (deployment-realistic, no GT used)")
    print(f"  Threshold : {threshold_valley:.4f}")
    print(f"  Recall    : {rec_v:.3f}   TP={tp_v}  FN={fn_v}")
    print(f"  Precision : {prec_v:.3f}   FP={fp_v}  TN={tn_v}")
    print(f"  F1        : {f1_v:.3f}")
    print(f"  {'─'*60}")

    # ── Per-subtype table (both thresholds side-by-side)
    print(f"\n  Per-defect-type breakdown:")
    print(f"  {'SUBTYPE':<20} {'N':>4}  {'ORACLE':>8}  {'VALLEY':>8}  {'AVG SCORE':>10}")
    print(f"  {'─'*58}")
    for subtype, stats in sorted(subtype_stats.items()):
        n        = stats["total"]
        avg_s    = np.mean(stats["scores"])
        if subtype == "good":
            oracle_str = f"{'(normal)':>8}"
            valley_str = f"{'(normal)':>8}"
        else:
            rec_oracle = stats["caught_oracle"] / max(n, 1)
            rec_valley = stats["caught_valley"] / max(n, 1)
            oracle_str = f"{rec_oracle:>8.3f}"
            valley_str = f"{rec_valley:>8.3f}"
        print(f"  {subtype:<20} {n:>4}  {oracle_str}  {valley_str}  {avg_s:>10.4f}")
    print(f"  {'─'*58}")

    # Use oracle threshold for downstream outputs (heatmap saving uses threshold
    # only for the score annotation — both thresholds are already printed above)
    threshold = threshold_oracle

    # 7. Save CSV
    csv_path = out_dir / f"{category}_scores.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "gt_label", "subtype", "anomaly_score"])
        for path, label, subtype, score in zip(
                all_paths, labels, subtypes, image_scores):
            writer.writerow([path, label, subtype, f"{score:.6f}"])
    print(f"  Scores saved to: {csv_path}")

    # 8. Score distribution plot
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        all_scores  = image_scores
        score_min   = float(all_scores.min())
        score_max   = float(all_scores.max())

        # Shared bin edges so both histograms are on the same scale
        n_bins    = max(30, len(all_scores) // 3)
        bin_edges = np.linspace(score_min, score_max, n_bins + 1)

        fig = plt.figure(figsize=(12, 8))
        gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

        # ── Panel 1: overlapping histograms, shared x-axis (full range) ──
        ax1 = fig.add_subplot(gs[0, :])
        ax1.hist(normal_scores, bins=bin_edges, alpha=0.65, color="#30c0a0",
                 label=f"Normal (n={len(normal_scores)})  "
                       f"mean={normal_scores.mean():.1f}  std={normal_scores.std():.1f}",
                 density=True)
        ax1.hist(defect_scores, bins=bin_edges, alpha=0.65, color="#e05060",
                 label=f"Defective (n={len(defect_scores)})  "
                       f"mean={defect_scores.mean():.1f}  std={defect_scores.std():.1f}",
                 density=True)
        ax1.axvline(threshold_oracle, color="#c03050", linewidth=1.8,
                    linestyle="--",
                    label=f"Oracle  t={threshold_oracle:.1f}  F1={f1_o:.3f}  "
                          f"Rec={rec_o:.3f}  Prec={prec_o:.3f}")
        ax1.axvline(threshold_valley, color="#2060c0", linewidth=1.8,
                    linestyle=":",
                    label=f"Valley  t={threshold_valley:.1f}  F1={f1_v:.3f}  "
                          f"Rec={rec_v:.3f}  Prec={prec_v:.3f}")
        ax1.set_xlabel("Anomaly score")
        ax1.set_ylabel("Density")
        ax1.legend(fontsize=8, loc="upper right")
        title = f"{category} — AUROC: {auroc:.4f}"
        if aupro is not None:
            title += f"   AUPRO: {aupro:.4f}"
        ax1.set_title(title, fontsize=11)

        # ── Panel 2: normal scores zoomed in ──
        ax2 = fig.add_subplot(gs[1, 0])
        n_bins_zoom = max(15, len(normal_scores) // 2)
        ax2.hist(normal_scores, bins=n_bins_zoom, color="#30c0a0", alpha=0.8, density=True)
        ax2.axvline(threshold_oracle, color="#c03050", linewidth=1.5, linestyle="--")
        ax2.axvline(threshold_valley, color="#2060c0", linewidth=1.5, linestyle=":")
        ax2.set_title("Normal scores (zoomed)", fontsize=9)
        ax2.set_xlabel("Anomaly score")
        ax2.set_ylabel("Density")
        # Annotate with percentiles
        for p, pv in [(25, np.percentile(normal_scores, 25)),
                      (50, np.percentile(normal_scores, 50)),
                      (75, np.percentile(normal_scores, 75)),
                      (95, np.percentile(normal_scores, 95))]:
            ax2.axvline(pv, color="gray", linewidth=0.8, linestyle="-", alpha=0.5)
            ax2.text(pv, ax2.get_ylim()[1] * 0.5, f"p{p}", fontsize=7,
                     color="gray", ha="center", rotation=90)

        # ── Panel 3: defective scores zoomed in ──
        ax3 = fig.add_subplot(gs[1, 1])
        n_bins_zoom = max(15, len(defect_scores) // 4)
        ax3.hist(defect_scores, bins=n_bins_zoom, color="#e05060", alpha=0.8, density=True)
        ax3.axvline(threshold_oracle, color="#c03050", linewidth=1.5, linestyle="--",
                    label=f"Oracle t={threshold_oracle:.1f}")
        ax3.axvline(threshold_valley, color="#2060c0", linewidth=1.5, linestyle=":",
                    label=f"Valley t={threshold_valley:.1f}")
        ax3.set_title("Defective scores (zoomed)", fontsize=9)
        ax3.set_xlabel("Anomaly score")
        ax3.set_ylabel("Density")
        ax3.legend(fontsize=7)
        for p, pv in [(5,  np.percentile(defect_scores, 5)),
                      (25, np.percentile(defect_scores, 25)),
                      (50, np.percentile(defect_scores, 50))]:
            ax3.axvline(pv, color="gray", linewidth=0.8, linestyle="-", alpha=0.5)
            ax3.text(pv, ax3.get_ylim()[1] * 0.5, f"p{p}", fontsize=7,
                     color="gray", ha="center", rotation=90)

        fig.suptitle(
            f"Normal: [{normal_scores.min():.1f}, {normal_scores.max():.1f}]   "
            f"Defective: [{defect_scores.min():.1f}, {defect_scores.max():.1f}]   "
            f"Gap: {defect_scores.min() - normal_scores.max():.1f}",
            fontsize=9, y=1.01, color="#444444"
        )

        plot_path = out_dir / f"{category}_score_distribution.png"
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Distribution plot: {plot_path}")
    except ImportError:
        print("  (matplotlib not available — skipping plot)")

    # 9. Heatmap overlays for defective images
    if save_heatmaps:
        try:
            heatmap_dir = out_dir / "heatmaps" / category
            heatmap_dir.mkdir(parents=True, exist_ok=True)
            defect_idx = [i for i, l in enumerate(labels) if l == 1]
            print(f"  Saving {len(defect_idx)} heatmap overlays ...")
            for idx in defect_idx:
                save_heatmap_overlay(
                    original_path=Path(all_paths[idx]),
                    heatmap=all_heatmaps[idx],
                    gt_mask=gt_masks[idx],
                    save_path=heatmap_dir / f"{Path(all_paths[idx]).stem}.png",
                    image_score=float(image_scores[idx]),
                )
            print(f"  Heatmaps saved to: {heatmap_dir}")
        except ImportError:
            print("  (matplotlib not available — skipping heatmaps)")

    return {
        "category":    category,
        "auroc":       auroc,
        "aupro":       aupro,
        # Oracle (upper bound — uses GT labels to pick threshold)
        "recall_o":    rec_o,
        "precision_o": prec_o,
        "f1_o":        f1_o,
        "tp_o":        tp_o,
        "fp_o":        fp_o,
        "fn_o":        fn_o,
        # Valley (deployment-realistic — no GT used)
        "recall_v":    rec_v,
        "precision_v": prec_v,
        "f1_v":        f1_v,
        "tp_v":        tp_v,
        "fp_v":        fp_v,
        "fn_v":        fn_v,
        "n_normal":    int((labels_arr == 0).sum()),
        "n_defect":    int((labels_arr == 1).sum()),
        "normal_mean": float(normal_scores.mean()),
        "defect_mean": float(defect_scores.mean()),
    }


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main(args):
    device     = get_device()
    data_root  = Path(args.data_root)
    models_dir = Path(args.models_dir)
    out_dir    = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    categories = TEXTILE_CATEGORIES if args.category == "all" else \
                 [c.strip() for c in args.category.split(",")]

    invalid = [c for c in categories if c not in TEXTILE_CATEGORIES]
    if invalid:
        raise ValueError(f"Non-textile categories: {invalid}. "
                         f"Valid: {TEXTILE_CATEGORIES}")

    print(f"\n{'=' * 55}")
    print(f"  Stage 1 Evaluation — GMM Inference (Test Set)")
    print(f"  Categories : {categories}")
    print(f"  Device     : {device}")
    print(f"  Models dir : {models_dir}")
    print(f"{'=' * 55}")

    print("\nLoading DINOv2 (once for all categories) ...")
    model = load_dinov2(device)

    results = []
    for category in categories:
        try:
            r = run_category(
                category, data_root, models_dir, out_dir, model, device,
                save_heatmaps=not args.no_heatmaps,
            )
            results.append(r)
        except Exception as e:
            print(f"  ERROR on {category}: {e}")
            import traceback; traceback.print_exc()

    # Summary table — oracle and valley thresholds shown side-by-side
    W = 102
    print("\n" + "=" * W)
    print(f"  {'':18}  {'':7}  {'':7}  "
          f"{'── oracle (GT) ──':^24}  {'── valley (no GT) ──':^26}")
    print(f"  {'CATEGORY':<18}  {'AUROC':>7}  {'AUPRO':>7}  "
          f"{'REC':>6}  {'PREC':>6}  {'F1':>6}  "
          f"{'REC':>6}  {'PREC':>6}  {'F1':>6}")
    print("  " + "-" * (W - 2))
    for r in results:
        aupro_str = f"{r['aupro']:.4f}" if r["aupro"] is not None else "   N/A "
        print(f"  {r['category']:<18}  {r['auroc']:>7.4f}  {aupro_str:>7}  "
              f"{r['recall_o']:>6.3f}  {r['precision_o']:>6.3f}  {r['f1_o']:>6.3f}  "
              f"{r['recall_v']:>6.3f}  {r['precision_v']:>6.3f}  {r['f1_v']:>6.3f}")
    if len(results) > 1:
        avg_auroc    = sum(r["auroc"]       for r in results) / len(results)
        valid_aupro  = [r["aupro"]          for r in results if r["aupro"] is not None]
        avg_aupro    = sum(valid_aupro) / len(valid_aupro) if valid_aupro else None
        avg_rec_o    = sum(r["recall_o"]    for r in results) / len(results)
        avg_prec_o   = sum(r["precision_o"] for r in results) / len(results)
        avg_f1_o     = sum(r["f1_o"]        for r in results) / len(results)
        avg_rec_v    = sum(r["recall_v"]    for r in results) / len(results)
        avg_prec_v   = sum(r["precision_v"] for r in results) / len(results)
        avg_f1_v     = sum(r["f1_v"]        for r in results) / len(results)
        aupro_str    = f"{avg_aupro:.4f}" if avg_aupro is not None else "   N/A "
        print("  " + "-" * (W - 2))
        print(f"  {'AVERAGE':<18}  {avg_auroc:>7.4f}  {aupro_str:>7}  "
              f"{avg_rec_o:>6.3f}  {avg_prec_o:>6.3f}  {avg_f1_o:>6.3f}  "
              f"{avg_rec_v:>6.3f}  {avg_prec_v:>6.3f}  {avg_f1_v:>6.3f}")
    print("=" * W + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stage 1 evaluation: GMM inference on MVTec test set."
    )
    parser.add_argument("--data_root",   type=str, required=True,
                        help="Path to MVTec root directory")
    parser.add_argument("--models_dir",  type=str, default="./stage1_output",
                        help="Directory with {category}_gmm.pkl files")
    parser.add_argument("--category",    type=str, default="all",
                        help="Category name, comma-separated list, or 'all'")
    parser.add_argument("--output_dir",  type=str, default="./inference_output",
                        help="Where to save scores, plots, heatmaps")
    parser.add_argument("--no_heatmaps", action="store_true",
                        help="Skip saving heatmap images (faster)")
    args = parser.parse_args()
    main(args)
