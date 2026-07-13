# anomaly-train-filter

A lightweight two-stage pipeline for unsupervised anomaly detection in textile/texture inspection. Stage 1 cleans a contaminated training set using DINOv2 patch features and iterative GMM fitting. Stage 2 evaluates the fitted model on held-out test data, reporting detection (AUROC), localization (AUPRO), and threshold-selection quality under fully unsupervised conditions — no labels used anywhere except for evaluation.

---

## The Problem

State-of-the-art unsupervised anomaly detection methods (e.g. [Dinomaly](https://arxiv.org/abs/2405.14529)) assume a **pure normal training set**. In real industrial deployments this assumption breaks — a camera mounted above a factory conveyor belt will inevitably capture some defective products during the data collection phase, contaminating the training data with anomalies. This is particularly relevant for textile manufacturing, where a camera above a conveyor belt captures both normal and defective fabric during data collection.

Training on contaminated data causes a downstream model to learn defects as normal, directly hurting detection performance at inference time.

**Stage 1 of this pipeline solves that problem** — it automatically separates normal images from anomalous ones in a contaminated set, producing a clean training set you can feed into any anomaly detection method (e.g. Dinomaly).

**Stage 2 evaluates the whole approach honestly** — it measures how well a model fit purely on the Stage-1-cleaned data performs on genuinely unseen test images, and specifically measures the cost of having no ground-truth labels available at deployment time (see "Oracle vs. Valley Threshold" below).

---

## Pipeline Overview

```
Contaminated training set
        │
        ▼
┌───────────────────────────────────────────────────┐
│  STAGE 1 — ad_filter.py                            │
│                                                     │
│  [1] Extract DINOv2 patch features for every image │
│  [2] Fit PCA (384D → 64D) + GMM with iterative      │
│      trimming (discard highest-scoring patches      │
│      each round, refit on the rest — prevents       │
│      contamination from corrupting the learned      │
│      normal distribution)                            │
│  [3] Score each image (category-dependent strategy: │
│      texture = fraction of anomalous patches,        │
│      masked object = top-1 foreground patch,          │
│      unmasked object = top-1 patch)                    │
│  [4] Find valley threshold, split normal/anomalous   │
└───────────────────────────────────────────────────┘
        │                              │
        ▼                              ▼
normal_paths.txt              {category}_gmm.pkl
anomalous_paths.txt           (saved PCA + GMM + threshold)
(feed into Dinomaly)                  │
                                       ▼
┌───────────────────────────────────────────────────┐
│  STAGE 2 — gmm_inference.py                        │
│                                                     │
│  Loads the frozen PCA + GMM (never refit) and       │
│  scores held-out MVTec test images.                 │
│                                                     │
│  [1] Extract DINOv2 patch features                  │
│  [2] Score patches + build spatial anomaly heatmaps  │
│  [3] Compute AUROC (detection) and AUPRO             │
│      (localization) — threshold-free metrics          │
│  [4] Pick an operating threshold two ways:            │
│        Oracle  — sweeps every cutoff using ground     │
│                  truth, reports the F1-best one        │
│                  (upper bound; not deployable)         │
│        Valley  — finds the threshold from the score    │
│                  distribution alone, no labels used     │
│                  (deployment-realistic)                  │
└───────────────────────────────────────────────────┘
        │
        ▼
{category}_scores.csv, distribution + valley-search plots,
heatmap overlays, per-defect-subtype breakdown
```

---

## Why Threshold Selection Is the Hard Part

Ranking quality (AUROC) is consistently near-perfect on MVTec textures — telling normal and defective apart, on average, is not the bottleneck. The hard problem is picking a **single cutoff** without access to labels.

Both stages need this at some point (Stage 1 to split the contaminated set, Stage 2 to report a deployment-realistic operating point) and both use the same underlying method:

1. **Hartigan's dip test** — a statistical test asking "does this score distribution actually look bimodal?" before trusting any valley search. If the data doesn't show statistically detectable structure, the method falls back safely rather than guessing.
2. **Multi-bandwidth KDE persistence search** — instead of a single noisy histogram (unreliable with only ~20-120 test images per category), a smooth density curve is built at 15 different smoothing levels. A genuine valley between normal and defective populations survives across most smoothing levels; a noise-driven dip only survives a couple. The deepest, most persistent valley wins.
3. **Percentile fallback** — used when the dip test finds no reliable bimodal structure (this happens on categories where defect severity varies too much across subtypes for a single clean second cluster to form — see Results).

This replaced an earlier, simpler "first local minimum in a raw histogram" approach, which was found to consistently land right at the edge of its own search-start cutoff rather than on any real structural gap.

---

## Requirements

```
torch
torchvision
timm
scikit-learn
scipy
scikit-image
numpy
Pillow
tqdm
joblib
matplotlib
diptest
```

Install with:

```bash
pip install torch torchvision timm scikit-learn scipy scikit-image numpy Pillow tqdm joblib matplotlib diptest
```

(`diptest` is required for the improved valley-threshold method in both stages; without it, the code falls back to the percentile method automatically with a printed warning.)

---

## Usage

### Stage 1 — clean a contaminated training set

```bash
# Single category
python ad_filter.py --data_root /path/to/mvtec --category carpet --output_dir ./stage1_output

# Multiple categories
python ad_filter.py --data_root /path/to/mvtec --category carpet,grid,leather --output_dir ./stage1_output

# All 5 texture categories
python ad_filter.py --data_root /path/to/mvtec --category all --output_dir ./stage1_output
```

| Argument | Default | Description |
|---|---|---|
| `--data_root` | required | Path to MVTec-AD root directory |
| `--category` | `all` | Category name, comma-separated list, or `all` |
| `--output_dir` | `./stage1_output` | Where to save path lists and fitted models |

**Output** (per category, saved to `output_dir`):
- `{category}_normal_paths.txt` — cleaned training set, feed into your anomaly detector
- `{category}_anomalous_paths.txt` — flagged images for inspection
- `{category}_gmm.pkl` — fitted PCA + GMM + threshold, used by Stage 2

### Stage 2 — evaluate on held-out test data

```bash
python gmm_inference.py --data_root /path/to/mvtec --category all --models_dir ./stage1_output --output_dir ./inference_output
```

| Argument | Default | Description |
|---|---|---|
| `--data_root` | required | Path to MVTec-AD root directory |
| `--models_dir` | `./stage1_output` | Directory containing `{category}_gmm.pkl` files from Stage 1 |
| `--category` | `all` | Category name, comma-separated list, or `all` |
| `--output_dir` | `./inference_output` | Where to save scores, plots, heatmaps |
| `--no_heatmaps` | off | Skip saving heatmap overlays (faster) |

**Output** (per category, saved to `output_dir`):
- `{category}_scores.csv` — per-image scores, labels, defect subtype
- `{category}_score_distribution.png` — normal vs. defective score histogram with oracle/valley thresholds marked
- `{category}_valley_search.png` — diagnostic plot of the valley-threshold search itself
- `heatmaps/{category}/*.png` — anomaly heatmap overlaid on each defective test image, alongside ground truth

---

## Results on MVTec-AD (Texture Categories)

### Stage 1 — Training Set Cleaning
10% contamination injected from test defect folders. Recall measures the fraction of injected defects correctly flagged; precision measures the fraction of flagged images that were genuinely defective.

| Category | Normal Images | Injected | Caught | FP | Recall | Precision | F1 |
|---|---|---|---|---|---|---|---|
| carpet  | 280 | 31 | 28 | 1 | 0.90 | 0.97 | 0.93 |
| grid    | 264 | 29 | 29 | 1 | 1.00 | 0.97 | 0.98 |
| leather | 245 | 27 | 27 | 2 | 1.00 | 0.93 | 0.96 |
| tile    | 230 | 25 | 24 | 1 | 0.96 | 0.96 | 0.96 |
| wood    | 247 | 27 | 20 | 3 | 0.74 | 0.87 | 0.80 |
| **Average** | | | | | **0.92** | **0.94** | **0.93** |

### Stage 2 — Test Set Evaluation

AUROC/AUPRO are threshold-free. Oracle uses ground-truth labels to pick the best possible cutoff (upper bound). Valley uses the KDE-persistence method (or percentile fallback) with no labels — this is the deployment-realistic number.

| Category | AUROC | AUPRO | Oracle Rec | Oracle Prec | Oracle F1 | Valley Rec | Valley Prec | Valley F1 |
|---|---|---|---|---|---|---|---|---|
| carpet  | 1.0000 | 0.0411 | 1.000 | 1.000 | 1.000 | 0.854 | 1.000 | 0.921 |
| grid    | 1.0000 | 0.3470 | 1.000 | 1.000 | 1.000 | 0.982 | 1.000 | 0.991 |
| leather | 1.0000 | 0.6313 | 1.000 | 1.000 | 1.000 | 0.989 | 1.000 | 0.995 |
| tile    | 1.0000 | 0.2535 | 1.000 | 1.000 | 1.000 | 0.786 | 1.000 | 0.880 |
| wood    | 0.9877 | 0.6728 | 0.950 | 1.000 | 0.974 | 0.850 | 1.000 | 0.919 |
| **Average** | **0.9975** | **0.3892** | **0.990** | **1.000** | **0.995** | **0.892** | **1.000** | **0.941** |

**Reading these results:**
- **AUROC is near-perfect everywhere** — the underlying anomaly *ranking* is excellent; separating normal from defective is not the bottleneck.
- **Valley precision is 1.000 across every category** — the unsupervised threshold never misclassifies a normal image as defective. The gap to oracle is paid entirely in recall (missed detections), never in false alarms.
- **grid and leather see the largest oracle-to-valley gap closed** by the KDE persistence method (F1 within 0.01–0.02 of oracle) because their defect-severity scores form one identifiable second cluster.
- **carpet, tile, wood remain harder** — their defect subtypes span a wide range of severities (e.g. tile: `gray_stroke` avg score ≈481 vs. `crack` avg score ≈2892), which produces a smeared, non-bimodal score distribution that the dip test correctly refuses to treat as two clean clusters. This is a genuine property of the data, not a threshold-search failure — a single global cutoff has a hard ceiling here regardless of method.
- **AUPRO is low for carpet (0.04) and tile (0.25)** despite perfect AUROC — the anomaly heatmap's spatial localization is noticeably weaker than the image-level detection signal for these two categories, a separate open question from thresholding.

---

## Dataset

Evaluated on the [MVTec Anomaly Detection Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad). The dataset must be downloaded separately and is not included in this repository.

Expected folder structure:

```
mvtec/
├── carpet/
│   ├── train/
│   │   └── good/
│   ├── test/
│   │   ├── good/
│   │   ├── color/
│   │   ├── cut/
│   │   └── ...
│   └── ground_truth/
│       ├── color/
│       └── ...
├── grid/
├── leather/
├── tile/
├── wood/
└── ...
```

---

## Context

This tool is **Stage 1 + Stage 2 of a larger research pipeline** for unsupervised anomaly detection under real factory conditions. The cleaned training set produced by Stage 1 is intended to be fed into [Dinomaly](https://arxiv.org/abs/2405.14529) for the actual defect detection model; Stage 2 provides an honest, no-labels-used evaluation of the density-modeling approach on its own.

Texture categories (carpet, grid, leather, tile, wood) are the primary focus because they match the target deployment scenario (fabric inspection) and because the GMM scoring strategy — fraction of anomalous patches — is well-suited to spatially distributed defects. Object categories require foreground masking and localized scoring, which are currently supported in Stage 1 but not evaluated in Stage 2.

The full pipeline is under active development. Current open directions include closing the oracle-vs-valley gap on high-severity-variance categories (carpet, tile, wood) and improving spatial localization (AUPRO) independent of thresholding.

---

## Citation

If you use this tool in your research, please cite:

```bibtex
@misc{anomaly-train-filter,
  author = {Ulas Sakin},
  title  = {anomaly-train-filter: Unsupervised Training Set Cleaning and Evaluation for Anomaly Detection},
  year   = {2025},
  url    = {https://github.com/ulassakin/anomaly-train-filter}
}
```

---

## License

MIT
