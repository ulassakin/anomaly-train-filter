"""
Stage 1: Separate normal vs. anomalous images from contaminated training data.

What this does:
  1. Load all training images (contaminated — mix of normal + some defective)
  2. Extract DINOv2 patch features for each image
  3. Fit PCA + GMM on all patch features (iterative trimming to handle contamination)
  4. Score each IMAGE by aggregating its patch scores (mean of top-k worst patches)
  5. Split into normal/anomalous using TWO independent methods, each aimed at a
     different downstream consumer with a different error cost:

       Method A — Valley threshold (precision-favoring)
         Finds the natural low-density gap in the score histogram. Rarely
         misclassifies a normal image as anomalous, at some cost to recall.
         Output: {category}_anomalous_paths.txt
         Intended use: input to zero-shot defect-type clustering, where a
         false positive (a normal patch polluting a defect cluster) is
         costly, but missing some genuine defects is fine since clustering
         only needs representative examples of each defect type.

       Method B — 2-component GMM on image scores (recall-favoring)
         Fits two Gaussians to the score distribution and assigns each
         image to whichever it's more likely under. Its decision boundary
         tends to sit lower (more permissive) than the valley method,
         catching more of the spread-out anomalous tail at some cost to
         precision.
         Output: {category}_normal_paths.txt
         Intended use: input to downstream detector training (e.g.
         Dinomaly), where a false negative (contamination reaching the
         training set) is permanent and uncorrectable, while a false
         positive (a normal image wrongly excluded) is cheap — it just
         shrinks the training set slightly.

     Both methods are computed from the same already-scored image_scores
     array — no extra feature extraction or GMM refitting is needed for
     the second method, so this costs milliseconds, not minutes.
  6. Save four lists per category:
       {category}_anomalous_paths.txt        (valley method — for clustering)
       {category}_normal_paths_valley.txt     (valley method's complement, informational)
       {category}_normal_paths.txt            (2-GMM method — for Dinomaly)
       {category}_anomalous_paths_2gmm.txt    (2-GMM method's complement, informational)

Usage:
    python ad_filter.py --data_root /path/to/mvtec --category carpet
"""

import argparse
import warnings
from pathlib import Path

import joblib
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
DEFECT_PCA_DIM  = 16   # lower-dim PCA fit fresh on defect patches only (much
                        # smaller population than normal patches, so a lower
                        # dimensionality avoids overfitting the defect-specific
                        # PCA/GMM to too little data)
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
# VALLEY THRESHOLD
# ─────────────────────────────────────────────
def find_valley_threshold(image_scores: np.ndarray,
                           n_bins: int = 200,
                           search_pctile: float = 50.0) -> float:
    """
    Finds the natural valley between the normal and anomalous image score
    distributions without assuming Gaussian shapes.

    Why this is better than a 2-component GMM:
      The normal image score distribution has a long right tail — some normal
      images score slightly higher due to fabric variation or lighting. A GMM
      fits Gaussians to both sides and places its boundary ambiguously in that
      tail. Valley detection instead finds the lowest-density point between
      the two modes, which is exactly where a threshold should be.

    Steps:
      1. Build a smooth histogram of image scores (KDE via histogram + smoothing)
      2. Search only in the upper half of scores (scores > 50th percentile)
         — the valley must be above the bulk of normal images
      3. Find the bin with minimum density in that search range
      4. Return the score value at that bin as the threshold

    In the degenerate case where no clear valley exists (all scores similar),
    falls back to the 95th percentile — conservative but safe.
    """
    from scipy.ndimage import uniform_filter1d

    # 1. Smooth histogram
    counts, bin_edges = np.histogram(image_scores, bins=n_bins)
    smoothed = uniform_filter1d(counts.astype(float), size=5)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # 2. Search range: above the search_pctile of scores
    search_start = np.percentile(image_scores, search_pctile)
    search_mask  = bin_centers > search_start

    if search_mask.sum() < 3:
        # Fallback: no meaningful search range — use 95th percentile
        print("  WARNING: no clear valley found — falling back to p95 threshold")
        return float(np.percentile(image_scores, 95))

    # 3. Find minimum density bin in search range
    search_counts  = smoothed[search_mask]
    search_centers = bin_centers[search_mask]
    valley_idx     = np.argmin(search_counts)
    threshold      = float(search_centers[valley_idx])

    # Sanity check: threshold must be above at least 70% of scores
    # (otherwise it's too low and will produce too many false positives)
    min_threshold = float(np.percentile(image_scores, 70))
    if threshold < min_threshold:
        threshold = min_threshold
        print(f"  Valley too low — raised to p70: {threshold:.4f}")

    return threshold


# ─────────────────────────────────────────────
# 2-COMPONENT GMM SPLIT  (recall-favoring, for Dinomaly training set)
# ─────────────────────────────────────────────
def find_2gmm_split(image_scores: np.ndarray):
    """
    Alternate splitting decision: fit a 2-component GMM directly on the
    1D image scores and hard-assign each image to whichever component
    (normal or anomalous) it's more likely under.

    Unlike the valley method (which searches for the single deepest,
    lowest-density gap), a 2-component GMM's decision boundary is where
    the two fitted Gaussians' densities cross — a point that depends on
    each component's fitted variance, not just its mean. When the
    anomalous score population is wide and spread out (as it typically
    is here, since defect severity varies a lot across subtypes), the
    anomalous Gaussian's left tail reaches further down toward the
    normal cluster, pulling the crossover point lower. A lower boundary
    means more images get called anomalous — higher recall, lower
    precision — which is exactly the behavior wanted for a training-set
    cleaner where missing contamination is the costly error.

    This is cheap: it's a tiny 1D GMM fit on at most a few hundred
    numbers, not a refit of anything expensive.

    Returns:
        assignments : (N,) int array, 0 or 1 (component index per image)
        normal_component : int, which component index means "normal"
    """
    score_gmm = GaussianMixture(n_components=2, random_state=RANDOM_SEED)
    score_gmm.fit(image_scores.reshape(-1, 1))

    normal_component = int(np.argmin(score_gmm.means_))
    assignments = score_gmm.predict(image_scores.reshape(-1, 1))

    return assignments, normal_component, score_gmm


# ─────────────────────────────────────────────
# ZERO-SHOT DEFECT-TYPE CLUSTERING
# ─────────────────────────────────────────────
def fit_defect_type_clusters(pca, gmm, anomalous_patch_features,
                              image_paths=None,
                              n_clusters: int = 5,
                              patch_anomaly_pctile: float = 90.0):
    """
    Fits a second, separate clustering model on genuinely anomalous
    PATCHES, to discover zero-shot defect-type groups — without ever
    being told what a "scratch" or "hole" is.

    WHY THIS IS A SEPARATE MODEL, NOT A REUSE OF THE NORMAL GMM:
    `gmm` (9 components) was fit ONLY on normal patches — its components
    represent different flavors of NORMAL texture variation (lighting,
    weave angle, etc.), not defect types. Calling `.predict()` on an
    anomalous patch against `gmm` would just tell you which flavor of
    normal it's LEAST unlike — not what kind of defect it is. Defect-type
    discovery needs a model fit purely on the anomalous population, in
    isolation from normal patches, so its clusters can only reflect
    genuine structure among defects (a scratch looks structurally
    different from a stain, which looks different from a hole) rather
    than incidental normal variation.

    WHY THIS IS PER-CATEGORY, NOT POOLED ACROSS CATEGORIES:
    What a "scratch" looks like in DINOv2 feature space on carpet's
    woven fibers is not the same as on wood's grain or tile's glaze —
    these are different materials with different textures. Pooling
    would let the clusters organize partly by material rather than
    purely by defect type. Each category gets its own independently
    fit anomalous_gmm, exactly mirroring how the normal `gmm` is
    already fit per-category, never shared.

    WHY INPUT COMES FROM THE VALLEY METHOD'S OUTPUT (not the 2-GMM's):
    The valley method is precision-favoring — it rarely misclassifies a
    normal image as anomalous. Feeding this function the 2-GMM method's
    (recall-favoring, lower-precision) anomalous set instead would let
    more normal-image noise into the clustering fit, since every false-
    positive image's patches would be candidates for the clustering
    population. This choice was made based on the STRUCTURAL cost
    asymmetry of the two methods (precision-favoring vs recall-favoring),
    decided before looking at any per-category clustering outcome — not
    picked after comparing which produces prettier clusters.

    STEPS:
      1. For every image in the (precision-favoring) anomalous set,
         score its patches with the already-fitted PCA + normal GMM —
         this decides which patches even qualify as "defective".
      2. Keep only patches that INDIVIDUALLY score above the patch
         anomaly threshold — this isolates genuine defect-region
         patches, discarding the majority of patches in a defective
         image that are still normal-looking background texture. Kept
         in their RAW (384D) feature form, not the normal-PCA-reduced
         form — see step 3.
      3. Fit a FRESH PCA (DEFECT_PCA_DIM components) on the raw defect
         patches specifically, then a FRESH GMM in that new space, for
         the actual clustering.

         WHY A FRESH PCA rather than reusing the normal-fitted PCA: the
         normal PCA's dimensions were chosen to capture directions
         where NORMAL texture varies most — a coordinate system built
         to answer "how different is this from normal," not "how
         different is this defect from other defects." Real defect-
         type-distinguishing structure can be compressed away in that
         space. A PCA fit fresh on the defect-patch population finds
         the directions where DEFECTS vary most among themselves,
         which is a much closer match to the clustering goal.

         WHY A GMM rather than K-means: K-means assumes every cluster
         is roughly spherical and similarly sized — a poor fit when
         different defect types plausibly have different shapes and
         spreads (e.g. a "hole" might form a tight, compact cluster
         while diffuse contamination might be elongated). A
         full-covariance GMM can capture that per-cluster shape and
         gives genuine probabilistic membership, consistent with the
         same generative-modeling approach already used for the
         normal-patch model.
      4. IF image_paths is provided, evaluate cluster PURITY against
         MVTec's known ground-truth defect subtype folders (e.g.
         test/color/, test/cut/, test/hole/). This is evaluation only —
         exactly the same "labels grade, never decide" boundary already
         used by evaluate_split() elsewhere in this script. The
         clustering itself (steps 1-3) never sees or uses these labels;
         this step only measures, after the fact, how well the
         zero-shot clusters happen to line up with real defect types —
         so you have a concrete signal for whether this mechanism is
         actually working, rather than flying blind.

    Args:
        pca      : the already-fitted PCA (from fit_gmm_iterative)
        gmm      : the already-fitted normal-patch GMM
        anomalous_patch_features : list of (N_patches, D) arrays, one
                   per image in the valley method's anomalous set —
                   RAW (un-PCA-reduced) patch features, same format as
                   what extract_features() returns.
        image_paths : list of str/Path, same length and order as
                   anomalous_patch_features — the source image path for
                   each entry. Used ONLY for the post-hoc purity report
                   (extracts the true defect subtype from MVTec's
                   test/{subtype}/ folder structure). Pass None to skip
                   the purity report entirely.
        n_clusters : number of defect-type clusters to discover. This
                   is a design choice, not something the data determines
                   automatically.
        patch_anomaly_pctile : percentile (over the anomalous images'
                   own patches) used as the per-patch anomaly cutoff.
                   Deliberately computed fresh here, from only the
                   anomalous image set, rather than reusing any
                   percentile computed elsewhere — this function
                   should be self-contained and not depend on Method
                   A/B's internal thresholds.

    Returns:
        defect_cluster_model : fitted GaussianMixture object (in the
                   fresh defect-specific PCA space), or None if there
                   were too few qualifying patches to cluster
                   meaningfully (fewer than n_clusters * 5).
        defect_pca            : the freshly-fit PCA used to project raw
                   defect patches before clustering. Needed at inference
                   time to project new patches into the same space
                   before calling defect_cluster_model.predict(). None
                   if clustering was skipped.
        cluster_patch_counts : dict {cluster_id: n_patches}, or None
        cluster_purity        : dict {cluster_id: {"dominant_subtype": str,
                   "purity": float, "counts": {subtype: n}}}, or None
                   if image_paths was not provided.
    """
    # (defect_cluster_model is a GaussianMixture, imported at module level)

    if len(anomalous_patch_features) == 0:
        print("  No anomalous images available — skipping defect-type clustering.")
        return None, None, None, None, None, None

    # Step 1: score every anomalous image's patches using the NORMAL
    # pca+gmm (this part must stay — it's what decides which patches
    # even qualify as "defective" in the first place). We keep BOTH the
    # raw 384D features and the normal-pca-reduced 64D scores here: the
    # reduced version is only needed for scoring in this step; the RAW
    # version is what step 3 will use to fit a fresh, defect-specific
    # PCA (see rationale below).
    all_scores = []
    all_raw_patches = []
    for patch_features in anomalous_patch_features:
        reduced = pca.transform(patch_features)
        patch_scores = -gmm.score_samples(reduced)
        all_scores.append(patch_scores)
        all_raw_patches.append(patch_features)  # RAW 384D, not reduced

    pooled_scores = np.concatenate(all_scores)
    patch_threshold = np.percentile(pooled_scores, patch_anomaly_pctile)
    print(f"  Defect-patch anomaly threshold (p{patch_anomaly_pctile:.0f}): {patch_threshold:.4f}")

    # Step 2: keep only genuinely anomalous-scoring patches, in their
    # RAW 384D form (not the normal-PCA-reduced 64D form) — this is
    # what step 3 needs to fit a fresh, defect-specific PCA.
    # Track each surviving patch's SOURCE IMAGE INDEX so we can later
    # look up its true subtype for the purity report (only if
    # image_paths was provided — this tracking is otherwise free/unused).
    defect_patches_raw = []
    source_image_idx = []
    for img_idx, (raw, scores) in enumerate(zip(all_raw_patches, all_scores)):
        keep = scores > patch_threshold
        n_keep = int(keep.sum())
        if n_keep > 0:
            defect_patches_raw.append(raw[keep])
            source_image_idx.extend([img_idx] * n_keep)

    if len(defect_patches_raw) == 0:
        print("  No patches cleared the anomaly threshold — skipping clustering.")
        return None, None, None, None, None, None

    defect_patches_raw = np.vstack(defect_patches_raw)
    source_image_idx = np.array(source_image_idx)
    print(f"  Defect patches collected: {defect_patches_raw.shape[0]:,} "
          f"(from {len(anomalous_patch_features)} anomalous images)")

    min_required = n_clusters * 5
    if defect_patches_raw.shape[0] < min_required:
        print(f"  Only {defect_patches_raw.shape[0]} defect patches available — "
              f"need at least {min_required} for {n_clusters} clusters. Skipping.")
        return None, None, None, None, None, None

    # Step 3: fit a FRESH PCA on the raw defect patches (not the
    # normal-fitted pca), then a FRESH GMM (not K-means) in that new
    # space, for the actual clustering.
    #
    # WHY A FRESH PCA: the `pca` passed into this function was fit on
    # NORMAL patches — its 64 dimensions were chosen to capture the
    # directions where normal texture varies most. That's a coordinate
    # system built to answer "how different is this from normal," not
    # "how different is this defect from other defects." Reusing it for
    # clustering means K-means (or any clustering) measures distance in
    # a space that was never optimized to separate defect TYPES from
    # each other — real defect-distinguishing structure can be
    # compressed away. Fitting a new PCA on the raw 384D defect-patch
    # population specifically finds the directions where DEFECTS vary
    # most among themselves, which is a much closer match to the actual
    # clustering goal.
    #
    # WHY A FRESH GMM INSTEAD OF K-MEANS: K-means assumes every cluster
    # is roughly spherical and similar in size/spread — a poor fit when
    # different defect types plausibly have different shapes and
    # variances in feature space (e.g. a "hole" might form a tight,
    # compact cluster while "thread" contamination might be diffuse and
    # elongated). A GMM with full covariance can capture that
    # per-cluster shape, and — just like the existing normal-patch GMM
    # — gives a genuine probabilistic membership rather than a rigid
    # nearest-center assignment, which fits the same underlying
    # generative-modeling philosophy already used throughout this
    # pipeline.
    defect_pca_dim = min(DEFECT_PCA_DIM, defect_patches_raw.shape[0] - 1, defect_patches_raw.shape[1])
    defect_pca = PCA(n_components=defect_pca_dim, random_state=RANDOM_SEED)
    defect_patches_reduced = defect_pca.fit_transform(defect_patches_raw)
    print(f"  Fresh defect-PCA ({defect_patches_raw.shape[1]}D → {defect_pca_dim}D)  "
          f"explained variance: {defect_pca.explained_variance_ratio_.cumsum()[-1]:.3f}")

    # n_init=10: fit the GMM from 10 different random starting positions
    # and keep whichever converged result has the best log-likelihood.
    # EM (the algorithm GaussianMixture uses internally) only ever moves
    # cluster centers to nearby better positions from wherever they
    # started -- a single bad starting arrangement can get permanently
    # stuck in a mediocre local optimum even though "better" cluster
    # placements exist elsewhere. Visual inspection of a single-init fit
    # showed several cluster ellipses heavily overlapping the same
    # territory rather than spreading out over the visibly-separated
    # true defect-type groups -- a classic symptom of this. Multiple
    # random restarts make it much more likely at least one attempt
    # starts in a good position and finds the true best arrangement,
    # at a small, worthwhile computational cost (this GMM fits on only
    # a few thousand defect patches, not the full normal-patch dataset).
    defect_cluster_model = GaussianMixture(
        n_components=n_clusters,
        covariance_type="full",
        max_iter=300,
        n_init=10,
        random_state=RANDOM_SEED,
    )
    cluster_assignments = defect_cluster_model.fit_predict(defect_patches_reduced)

    cluster_patch_counts = {
        int(c): int((cluster_assignments == c).sum())
        for c in range(n_clusters)
    }
    print(f"  Fitted {n_clusters} defect-type clusters:")
    for c, count in sorted(cluster_patch_counts.items()):
        print(f"    Cluster {c}: {count:,} patches")

    # Step 4: purity report against ground-truth subtype folders
    # (EVALUATION ONLY — see docstring. Never influences clustering.)
    cluster_purity = None
    if image_paths is not None:
        cluster_purity = _evaluate_cluster_purity(
            cluster_assignments, source_image_idx, image_paths, n_clusters
        )

    # Step 5: aggregate patch-level cluster assignments into ONE
    # defect-type label PER IMAGE, via majority vote.
    #
    # WHY VOTE AFTER CLUSTERING, RATHER THAN AVERAGING PATCHES BEFORE
    # CLUSTERING: the actual goal is a per-image label ("this carpet has
    # a cut"), not a per-patch label — a real deployment wants one
    # answer per product, not a scattered list of patch-level guesses.
    # But averaging an image's anomalous patches into one vector BEFORE
    # clustering would blur away exactly the structural detail (the
    # shape/texture of the defect) that best distinguishes one defect
    # type from another — a thin cut and a round hole could average out
    # to a similar-looking blurred vector even if their individual
    # patches look quite different. Clustering on the raw, un-averaged
    # patches lets each patch express its strongest individual signal;
    # aggregation only happens AFTER clustering has already done its
    # best work, via majority vote — collapsing many patch-level votes
    # into one image-level answer. This is also more robust to noise:
    # a handful of stray patches voting for the wrong cluster (edge
    # effects, lighting) get outvoted by the majority within that same
    # image, rather than corrupting a single averaged feature vector.
    image_cluster_labels = _aggregate_patch_clusters_to_images(
        cluster_assignments, source_image_idx, len(anomalous_patch_features)
    )

    image_purity = None
    if image_paths is not None:
        image_purity = _evaluate_image_cluster_purity(
            image_cluster_labels, image_paths, n_clusters
        )

    return (defect_cluster_model, defect_pca, cluster_patch_counts, cluster_purity,
            image_cluster_labels, image_purity)


def _extract_true_subtype(image_path) -> str:
    """
    Extracts the ground-truth defect subtype from an MVTec-AD image path.
    MVTec structure: .../{category}/test/{subtype}/{image}.png
    Returns the {subtype} folder name (e.g. "color", "cut", "hole"),
    or "unknown" if the path doesn't match the expected structure
    (e.g. the image came from train/good, which shouldn't happen for
    a genuinely anomalous image, but handled defensively).
    """
    parts = Path(image_path).parts
    try:
        test_idx = parts.index("test")
        return parts[test_idx + 1]
    except (ValueError, IndexError):
        return "unknown"


def _evaluate_cluster_purity(cluster_assignments, source_image_idx, image_paths, n_clusters):
    """
    For each discovered cluster, looks up the TRUE defect subtype (from
    MVTec's folder structure) of every patch's source image, and reports
    the dominant subtype and what fraction of the cluster's patches
    belong to it — a purity score. This is purely diagnostic: it tells
    you whether the zero-shot clusters happen to correspond to real,
    distinct defect types, without the clustering process itself ever
    having used this information.

    A high-purity cluster (e.g. 80%+ dominated by one true subtype)
    suggests the clustering is finding genuine defect-type structure.
    A low-purity cluster (close to 1/n_true_subtypes, i.e. no better
    than random) suggests that cluster isn't capturing anything
    meaningful — could be a mix of several defect types, or mostly
    false-positive normal patches that happened to cluster together.

    Returns:
        dict {cluster_id: {"dominant_subtype": str, "purity": float,
                            "n_patches": int, "counts": {subtype: n}}}
    """
    import collections

    # Map each patch to its true subtype via its source image
    true_subtypes_per_patch = np.array([
        _extract_true_subtype(image_paths[img_idx]) for img_idx in source_image_idx
    ])

    purity_report = {}
    print(f"\n  Cluster purity (vs. ground-truth defect subtypes):")
    for c in range(n_clusters):
        cluster_mask = cluster_assignments == c
        n_patches = int(cluster_mask.sum())
        if n_patches == 0:
            purity_report[c] = {"dominant_subtype": None, "purity": 0.0,
                                 "n_patches": 0, "counts": {}}
            print(f"    Cluster {c}: empty")
            continue

        subtypes_in_cluster = true_subtypes_per_patch[cluster_mask]
        counts = dict(collections.Counter(subtypes_in_cluster))
        dominant_subtype, dominant_count = max(counts.items(), key=lambda kv: kv[1])
        purity = dominant_count / n_patches

        purity_report[c] = {
            "dominant_subtype": dominant_subtype,
            "purity": purity,
            "n_patches": n_patches,
            "counts": counts,
        }

        counts_str = ", ".join(f"{k}={v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1]))
        print(f"    Cluster {c}: {n_patches} patches — dominant='{dominant_subtype}' "
              f"({100*purity:.1f}% purity)  [{counts_str}]")

    avg_purity = np.mean([r["purity"] for r in purity_report.values() if r["n_patches"] > 0])
    print(f"  Average cluster purity: {100*avg_purity:.1f}%")

    return purity_report


def _aggregate_patch_clusters_to_images(cluster_assignments, source_image_idx, n_images):
    """
    Collapses patch-level cluster assignments into ONE defect-type label
    per image, via majority vote among that image's own qualifying
    (anomalous-scoring) patches.

    Images with zero qualifying patches (none of their patches cleared
    the anomaly threshold — can happen for borderline valley-method
    false positives) get label None, meaning "no defect-type assigned."

    Returns:
        image_cluster_labels : list of length n_images, image_cluster_labels[i]
                   is the majority-vote cluster ID for image i (matching
                   the index order of the anomalous_patch_features/
                   image_paths passed into fit_defect_type_clusters), or
                   None if that image had no qualifying patches.
    """
    import collections

    image_cluster_labels = [None] * n_images
    for img_idx in range(n_images):
        patch_mask = source_image_idx == img_idx
        if patch_mask.sum() == 0:
            continue  # no qualifying patches for this image
        votes = cluster_assignments[patch_mask]
        majority_cluster, _ = collections.Counter(votes).most_common(1)[0]
        image_cluster_labels[img_idx] = int(majority_cluster)

    return image_cluster_labels


def _evaluate_image_cluster_purity(image_cluster_labels, image_paths, n_clusters):
    """
    Per-image version of the purity report: for each discovered cluster,
    looks at which IMAGES were assigned to it (via majority vote), looks
    up each image's TRUE defect subtype (from MVTec's folder structure),
    and reports the dominant subtype and purity — this time counting
    whole images, not individual patches. This is the number that
    actually answers "if I show this system a new defective product,
    how often does its assigned defect-type bucket match the real
    defect type" — the practically meaningful question, since the
    deployed system's output IS one label per image, not per patch.

    EVALUATION ONLY — see fit_defect_type_clusters() docstring for the
    same "labels grade, never decide" boundary. This function only
    looks at labels AFTER majority voting has already happened.

    Returns:
        dict {cluster_id: {"dominant_subtype": str, "purity": float,
                            "n_images": int, "counts": {subtype: n}}}
    """
    import collections

    purity_report = {}
    print(f"\n  Per-IMAGE cluster purity (vs. ground-truth defect subtypes):")
    for c in range(n_clusters):
        images_in_cluster = [
            image_paths[i] for i, label in enumerate(image_cluster_labels)
            if label == c
        ]
        n_images = len(images_in_cluster)
        if n_images == 0:
            purity_report[c] = {"dominant_subtype": None, "purity": 0.0,
                                 "n_images": 0, "counts": {}}
            print(f"    Cluster {c}: empty")
            continue

        subtypes = [_extract_true_subtype(p) for p in images_in_cluster]
        counts = dict(collections.Counter(subtypes))
        dominant_subtype, dominant_count = max(counts.items(), key=lambda kv: kv[1])
        purity = dominant_count / n_images

        purity_report[c] = {
            "dominant_subtype": dominant_subtype,
            "purity": purity,
            "n_images": n_images,
            "counts": counts,
        }

        counts_str = ", ".join(f"{k}={v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1]))
        print(f"    Cluster {c}: {n_images} images — dominant='{dominant_subtype}' "
              f"({100*purity:.1f}% purity)  [{counts_str}]")

    n_unassigned = sum(1 for label in image_cluster_labels if label is None)
    if n_unassigned > 0:
        print(f"    ({n_unassigned} images had no qualifying defect patches — unassigned)")

    valid_clusters = {c: r for c, r in purity_report.items() if r["n_images"] > 0}

    if valid_clusters:
        # Naive average: treats every cluster equally regardless of size.
        # Misleading when some clusters have only 1-2 images — a
        # trivially "100% pure" cluster of 1 image counts exactly as
        # much as a genuinely messy 14-image cluster, which inflates
        # the overall picture. Reported for reference only.
        naive_avg = np.mean([r["purity"] for r in valid_clusters.values()])

        # Weighted average: what actually matters — the fraction of
        # ALL anomalous images that landed in a cluster whose majority
        # subtype matches their own true subtype. This is the honest
        # number: it answers "if I hand this system a random defective
        # image, how often does its assigned bucket agree with its true
        # type," not "how often does an arbitrary cluster agree with
        # itself regardless of how many images it actually represents."
        total_images = sum(r["n_images"] for r in valid_clusters.values())
        weighted_avg = sum(
            r["n_images"] * r["purity"] for r in valid_clusters.values()
        ) / total_images

        print(f"  Average per-image cluster purity (naive, unweighted): {100*naive_avg:.1f}%")
        print(f"  Average per-image cluster purity (weighted by cluster size): {100*weighted_avg:.1f}%  <- the honest number")

        # Flag small clusters explicitly — their purity numbers are not
        # statistically meaningful on their own.
        small_clusters = [c for c, r in valid_clusters.items() if r["n_images"] < 5]
        if small_clusters:
            print(f"  NOTE: clusters {small_clusters} have fewer than 5 images — "
                  f"their purity numbers are not statistically meaningful in isolation.")

    return purity_report



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

    total_injected = sum(gt_labels)

    def evaluate_split(pred_anomalous):
        pred_anomalous_set = set(pred_anomalous)
        tp = sum(1 for p, l in zip(all_paths, gt_labels) if l == 1 and p in pred_anomalous_set)
        fp = sum(1 for p, l in zip(all_paths, gt_labels) if l == 0 and p in pred_anomalous_set)
        fn = sum(1 for p, l in zip(all_paths, gt_labels) if l == 1 and p not in pred_anomalous_set)
        prec = tp / max(len(pred_anomalous), 1)
        rec  = tp / max(total_injected, 1)
        f1   = 2 * prec * rec / max(prec + rec, 1e-8)
        return tp, fp, fn, prec, rec, f1

    # ── Method A: valley threshold (precision-favoring) ──
    threshold_valley = find_valley_threshold(image_scores)
    valley_assignments = (image_scores > threshold_valley).astype(int)
    valley_normal    = [p for p, a in zip(all_paths, valley_assignments) if a == 0]
    valley_anomalous = [p for p, a in zip(all_paths, valley_assignments) if a == 1]
    v_tp, v_fp, v_fn, v_prec, v_rec, v_f1 = evaluate_split(valley_anomalous)

    print(f"\n  ── Method A: Valley threshold (precision-favoring) ──")
    print(f"  Threshold             : {threshold_valley:.4f}")
    print(f"  Normal images         : {len(valley_normal)}")
    print(f"  Anomalous images      : {len(valley_anomalous)}")
    print(f"  Recall={v_rec:.2f}  Precision={v_prec:.2f}  F1={v_f1:.2f}  "
          f"(caught {v_tp}/{total_injected}, fp={v_fp})")

    # ── Zero-shot defect-type clustering ──
    # Uses Method A's (precision-favoring) anomalous set specifically —
    # see fit_defect_type_clusters() docstring for why. Pull the RAW
    # patch features (not yet PCA-reduced) for exactly those images.
    # valley_anomalous paths are passed too, purely so cluster quality
    # can be checked against MVTec's known ground-truth subtype folders
    # AFTER clustering — the clustering itself never sees these labels.
    print(f"\n  ── Zero-shot defect-type clustering ──")
    path_to_patches = dict(zip(all_paths, all_patch_features))
    valley_anomalous_patch_features = [path_to_patches[p] for p in valley_anomalous]
    (defect_cluster_model, defect_pca, cluster_patch_counts, cluster_purity,
     image_cluster_labels, image_cluster_purity) = fit_defect_type_clusters(
        pca, gmm, valley_anomalous_patch_features, image_paths=valley_anomalous
    )

    # ── Method B: 2-component GMM (recall-favoring) ──
    gmm_assignments, gmm_normal_component, score_gmm = find_2gmm_split(image_scores)
    gmm_normal    = [p for p, a in zip(all_paths, gmm_assignments) if a == gmm_normal_component]
    gmm_anomalous = [p for p, a in zip(all_paths, gmm_assignments) if a != gmm_normal_component]
    g_tp, g_fp, g_fn, g_prec, g_rec, g_f1 = evaluate_split(gmm_anomalous)

    print(f"\n  ── Method B: 2-component GMM (recall-favoring) ──")
    print(f"  Normal cluster mean   : {score_gmm.means_[gmm_normal_component][0]:.4f}")
    print(f"  Anomalous cluster mean: {score_gmm.means_[1 - gmm_normal_component][0]:.4f}")
    print(f"  Normal images         : {len(gmm_normal)}")
    print(f"  Anomalous images      : {len(gmm_anomalous)}")
    print(f"  Recall={g_rec:.2f}  Precision={g_prec:.2f}  F1={g_f1:.2f}  "
          f"(caught {g_tp}/{total_injected}, fp={g_fp})")

    # ── Save path lists — separate files per method, per intended consumer ──
    # Valley method -> anomalous set feeds zero-shot defect-type clustering
    (out_dir / f"{category}_anomalous_paths.txt").write_text("\n".join(str(p) for p in valley_anomalous))
    (out_dir / f"{category}_normal_paths_valley.txt").write_text("\n".join(str(p) for p in valley_normal))

    # 2-GMM method -> normal set feeds Dinomaly training
    (out_dir / f"{category}_normal_paths.txt").write_text("\n".join(str(p) for p in gmm_normal))
    (out_dir / f"{category}_anomalous_paths_2gmm.txt").write_text("\n".join(str(p) for p in gmm_anomalous))

    # Save fitted PCA + GMM + both thresholds/models for inference and clustering
    joblib.dump(
        {
            "pca": pca,
            "gmm": gmm,
            "threshold_valley": threshold_valley,
            "score_gmm": score_gmm,
            "score_gmm_normal_component": gmm_normal_component,
            "defect_cluster_model": defect_cluster_model,
            "defect_pca": defect_pca,
            "defect_cluster_patch_counts": cluster_patch_counts,
            "defect_cluster_purity": cluster_purity,
            "image_cluster_labels": dict(zip(valley_anomalous, image_cluster_labels))
                if image_cluster_labels is not None else None,
            "image_cluster_purity": image_cluster_purity,
        },
        out_dir / f"{category}_gmm.pkl"
    )
    print(f"\n  Saved model to: {out_dir}/{category}_gmm.pkl")

    return {
        "category":        category,
        "injected":        total_injected,
        # Valley method results
        "valley_tp": v_tp, "valley_fp": v_fp, "valley_fn": v_fn,
        "valley_recall": v_rec, "valley_precision": v_prec, "valley_f1": v_f1,
        # 2-GMM method results
        "gmm_tp": g_tp, "gmm_fp": g_fp, "gmm_fn": g_fn,
        "gmm_recall": g_rec, "gmm_precision": g_prec, "gmm_f1": g_f1,
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

    # Summary table — both methods side by side
    print("\n" + "=" * 104)
    print(f"                              ── Valley (precision-favoring) ──      ── 2-GMM (recall-favoring) ──")
    print(f"  {'CATEGORY':<12} {'INJ':>5}   {'REC':>6} {'PREC':>6} {'F1':>6}   {'FP':>4}     "
          f"{'REC':>6} {'PREC':>6} {'F1':>6}   {'FP':>4}")
    print("  " + "-" * 102)
    for r in results:
        print(f"  {r['category']:<12} {r['injected']:>5}   "
              f"{r['valley_recall']:>6.2f} {r['valley_precision']:>6.2f} {r['valley_f1']:>6.2f}   {r['valley_fp']:>4}     "
              f"{r['gmm_recall']:>6.2f} {r['gmm_precision']:>6.2f} {r['gmm_f1']:>6.2f}   {r['gmm_fp']:>4}")
    if len(results) > 1:
        avg_v_rec  = sum(r['valley_recall']    for r in results) / len(results)
        avg_v_prec = sum(r['valley_precision'] for r in results) / len(results)
        avg_v_f1   = sum(r['valley_f1']        for r in results) / len(results)
        avg_g_rec  = sum(r['gmm_recall']       for r in results) / len(results)
        avg_g_prec = sum(r['gmm_precision']    for r in results) / len(results)
        avg_g_f1   = sum(r['gmm_f1']           for r in results) / len(results)
        print("  " + "-" * 102)
        print(f"  {'AVERAGE':<12} {'':>5}   "
              f"{avg_v_rec:>6.2f} {avg_v_prec:>6.2f} {avg_v_f1:>6.2f}   {'':>4}     "
              f"{avg_g_rec:>6.2f} {avg_g_prec:>6.2f} {avg_g_f1:>6.2f}   {'':>4}")
    print("=" * 104 + "\n")

    print("  Done. Output saved to:", out_dir)
    print("    {category}_normal_paths.txt          <- 2-GMM method, feed into Dinomaly")
    print("    {category}_anomalous_paths.txt       <- valley method, feed into defect clustering")
    print("    {category}_normal_paths_valley.txt   <- valley method's normal set (informational)")
    print("    {category}_anomalous_paths_2gmm.txt  <- 2-GMM method's anomalous set (informational)")
    print("    {category}_gmm.pkl                   <- fitted PCA + GMM + both split models")


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
