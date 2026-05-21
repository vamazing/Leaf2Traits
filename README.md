# Leaf2Traits: Hyperspectral Leaf Analysis with Deep Learning

A multi-stage machine learning pipeline for analyzing hyperspectral leaf reflectance
spectra from five public EcoSIS plant databases. The pipeline trains a self-supervised
autoencoder, predicts biochemical leaf traits, evaluates cross-domain transfer across
all five datasets, and discovers spectrally unusual candidate plants.

---

## Project Structure

```
.
├── ecosis_loaders.py              # EcoSIS API loader (AQGRAD, Talladega, CABO, LOPEX, Sierra)
├── explore_lopex93_columns.py     # Stage 0: LOPEX93 column inspector
├── step1_pretrain_downstream.py   # Stage 1: Autoencoder pretraining + downstream tasks
├── step2_transfer_learning.py     # Stage 2: Cross-domain transfer (4 datasets, 12 directions)
├── step3_integrated_discovery.py  # Stage 3: Full A->C discovery pipeline
├── lopex1993metadata.xlsx         # LOPEX93 biochemical trait metadata (required)
└── README.md
```

### Outputs (auto-created)

```
outputs/
├── stage1/
│   ├── autoencoder_loss.png
│   ├── embedding_pca.png                      # PCA coloured by all 5 datasets
│   ├── trait_compare_LOPEX93_<trait>.png      # Per-trait predicted-vs-true plots (LOPEX93)
│   ├── trait_compare_SierraNevada_<trait>.png # Per-trait predicted-vs-true plots (Sierra Nevada)
│   ├── trait_compare_CABOv2_<trait>.png       # Per-trait predicted-vs-true plots (CABO v2)
│   ├── encoder.pt / scaler_mean.npy / scaler_std.npy / target_wl.npy
│   └── embeddings_all.npz
├── stage2/
│   ├── baseline_comparison.png    # 12-direction raw vs embedding bar chart
│   └── transfer_summary.csv
└── stage3/
    ├── novelty_distribution.png
    ├── ranked_candidates.csv
    ├── trait_comparison.png
    └── embedding_validation.png
```

---

## Datasets

| Name | Year | Source | Description |
|---|---|---|---|
| AQGRAD | 2021 | EcoSIS | Leaf spectra and traits along a water-affinity gradient, Mantua lakes, Italy |
| Talladega 2018 | 2018 | EcoSIS | Leaf reflectance and foliar traits, Talladega National Forest, Alabama |
| CABO 2018-2019 v2 | 2023 update | EcoSIS | Broad multispecies leaf-level spectra with associated trait data |
| LOPEX93 | 1993 | JRC / EcoSIS | Lab spectra + biochemical traits (via metadata xlsx) |
| Sierra Nevada 2023 | 2023 | EcoSIS | Field spectra + measured traits, California (drought/fire stress) |

All datasets download automatically. Sierra Nevada 2023 and CABO v2 include inline
trait fields in the EcoSIS export, while LOPEX93 uses the local metadata workbook for
trait attachment.

---

## Setup

```bash
pip install numpy pandas matplotlib scikit-learn torch torchvision scipy openpyxl
```

Place `lopex1993metadata.xlsx` in the project root before running.

---

## Running the Pipeline

```bash
python explore_lopex93_columns.py   # optional, verify LOPEX93 download
python step1_pretrain_downstream.py # trains autoencoder on all 5 datasets
python step2_transfer_learning.py   # pairwise cross-domain transfer directions
python step3_integrated_discovery.py
```

---

## Architecture

### Autoencoder
- Input: 201-band spectrum (400–2400 nm, 10 nm step)
- Encoder: Linear(201→1024) → BN → GELU → Dropout(0.05) → Linear(1024→512) → BN → GELU → Dropout(0.05) → Linear(512→256) → BN → GELU → Dropout(0.05) → Linear(256→128) → BN → GELU → Dropout(0.05) → Linear(128→192)
- Decoder: symmetric mirror of the encoder back to 201 bands
- Latent space: 192-dimensional embedding
- Training: Adam + Cosine LR, MSE loss, 140 epochs, batch 64, lr=7.5e-4, weight decay=1e-5, Gaussian input noise std=0.03

### Preprocessing
1. Savitzky-Golay smoothing (window=11, polyorder=3)
2. Standard Normal Variate (SNV) normalisation
3. Linear interpolation to 400–2400 nm, 10 nm grid

### Discovery
- Isolation Forest (200 trees, contamination=10%) on 192-dim embeddings
- Novelty score = -isolation_score, normalized to [0,1]
- Composite = novelty + 0.15 × known_hyperaccumulator_genus_prior
