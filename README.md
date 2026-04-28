# anomaly-train-filter

A lightweight tool for cleaning contaminated training sets in unsupervised anomaly detection. Uses DINOv2 patch features and iterative GMM fitting to separate normal from anomalous images — no labels required. Evaluated on all 15 MVTec-AD categories.

---

## The Problem

State-of-the-art unsupervised anomaly detection methods (e.g. [Dinomaly](https://arxiv.org/abs/2405.14529)) assume a **pure normal training set**. In real industrial deployments this assumption breaks — a camera mounted above a factory conveyor belt will inevitably capture some defective products during the data collection phase, contaminating the training data with anomalies.

Training on contaminated data causes the model to learn defects as normal, directly hurting detection performance at inference time.

**This tool solves that problem.** Given a contaminated training set, it automatically separates normal images from anomalous ones — producing a clean training set you can feed directly into any anomaly detection method.

---

## How It Works

```
Contaminated training set
        │
        ▼
[1] Extract DINOv2 patch features for every image
        │
        ▼
[2] Fit PCA (384D → 64D) + GMM with iterative trimming
    Iterative trimming: refit GMM after removing highest-scoring
    patches each round, preventing anomalous patches from corrupting
    the learned normal distribution
        │
        ▼
[3] Score each image
    Textures → fraction of anomalous patches
               (defects are spatially spread, fraction captures cumulative evidence)
    Objects  → max patch score (top-1)
               (defects are localized, one outlier patch is enough)
        │
        ▼
[4] Fit 2-component GMM on image scores
    Finds natural boundary between normal and anomalous clusters
    No hardcoded threshold — the data decides
        │
        ▼
normal_paths.txt        anomalous_paths.txt
(feed into Dinomaly)    (discarded)
```

---

## Requirements

```
torch
torchvision
timm
scikit-learn
numpy
Pillow
tqdm
```

Install with:

```bash
pip install torch torchvision timm scikit-learn numpy Pillow tqdm
```

---

## Usage

**Single category:**
```bash
python filter.py --data_root /path/to/mvtec --category carpet
```

**Multiple categories:**
```bash
python filter.py --data_root /path/to/mvtec --category carpet,grid,leather
```

**All 15 MVTec-AD categories:**
```bash
python filter.py --data_root /path/to/mvtec --category all
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--data_root` | required | Path to MVTec-AD root directory |
| `--category` | `all` | Category name, comma-separated list, or `all` |
| `--output_dir` | `./stage1_output` | Where to save output path lists |

### Output

For each category, two files are saved to `output_dir`:

- `{category}_normal_paths.txt` — one path per line, feed this into your anomaly detector
- `{category}_anomalous_paths.txt` — flagged images for inspection

---

## Results on MVTec-AD

Evaluated with 10% contamination (injected from test defect folders). Recall measures what fraction of injected defects were correctly flagged. Precision measures what fraction of flagged images were actually defective.

| Category | Injected | Caught | FP | Recall | Precision | F1 |
|---|---|---|---|---|---|---|
| carpet | 31 | 31 | 15 | 1.00 | 0.67 | 0.81 |
| grid | — | — | — | — | — | — |
| leather | — | — | — | — | — | — |
| tile | — | — | — | — | — | — |
| wood | — | — | — | — | — | — |
| bottle | — | — | — | — | — | — |
| cable | — | — | — | — | — | — |
| capsule | — | — | — | — | — | — |
| hazelnut | — | — | — | — | — | — |
| metal_nut | — | — | — | — | — | — |
| pill | — | — | — | — | — | — |
| screw | — | — | — | — | — | — |
| toothbrush | — | — | — | — | — | — |
| transistor | — | — | — | — | — | — |
| zipper | — | — | — | — | — | — |
| **Average** | | | | **—** | **—** | **—** |

> Results will be filled in after full evaluation run.

---

## Dataset

This tool is evaluated on [MVTec Anomaly Detection Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad). The dataset must be downloaded separately and is not included in this repository.

Expected folder structure:

```
mvtec/
├── carpet/
│   ├── train/
│   │   └── good/
│   └── test/
│       ├── good/
│       ├── color/
│       ├── cut/
│       └── ...
├── grid/
└── ...
```

---

## Context

This tool is **Stage 1 of a larger research pipeline** for unsupervised anomaly detection under real factory conditions. The cleaned training set produced here is intended to be fed into [Dinomaly](https://arxiv.org/abs/2405.14529) for the actual defect detection model.

The full pipeline is under active development.

---

## Citation

If you use this tool in your research, please cite:

```bibtex
@misc{anomaly-train-filter,
  author = {Ulas Sakin},
  title  = {anomaly-train-filter: Training Set Cleaning for Unsupervised Anomaly Detection},
  year   = {2025},
  url    = {https://github.com/ulassakin/anomaly-train-filter}
}
```

---

## License

MIT
