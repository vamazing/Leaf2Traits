# Leaf2Traits: Hyperspectral Leaf Analysis with Deep Learning

A multi-stage machine learning pipeline for analyzing hyperspectral leaf reflectance
spectra from five public EcoSIS plant databases. The pipeline trains a self-supervised
autoencoder, predicts biochemical leaf traits, evaluates cross-domain transfer across
all five datasets, and discovers spectrally unusual candidate plants.

---

## Project Structure

```text
.
|-- ecosis_loaders.py              # EcoSIS API loader (AQGRAD, Talladega, CABO, LOPEX, Sierra)
|-- step1_pretrain_downstream.py   # Stage 1: Autoencoder pretraining + downstream tasks
|-- step2_transfer_learning.py     # Stage 2: Cross-domain transfer
|-- step3_integrated_discovery.py  # Stage 3: Integrated discovery pipeline
|-- lopex1993metadata.xlsx         # LOPEX93 biochemical trait metadata (required)
|-- ca2023_freshspectra_supplementaldata.csv
|-- requirements.txt
|-- package.json
`-- README.md
```

### Outputs

```text
outputs/
|-- stage1/
|   |-- autoencoder_loss.png
|   |-- embedding_pca.png
|   |-- trait_compare_LOPEX93_<trait>.png
|   |-- trait_compare_SierraNevada_<trait>.png
|   |-- trait_compare_CABOv2_<trait>.png
|   |-- trait_regression_summary_LOPEX93.csv
|   |-- trait_regression_summary_SierraNevada.csv
|   |-- trait_regression_summary_CABOv2.csv
|   |-- encoder.pt
|   |-- scaler_mean.npy
|   |-- scaler_std.npy
|   |-- target_wl.npy
|   `-- embeddings_all.npz
|-- stage2/
|   |-- baseline_comparison.png
|   `-- transfer_summary.csv
`-- stage3/
    |-- novelty_distribution.png
    |-- ranked_candidates.csv
    |-- trait_comparison.png
    `-- embedding_validation.png
```

Each stage clears only its own output folder before writing new artifacts, so stale
plots from prior runs do not mix with current results.

---

## Datasets

| Name | Year | Source | Description |
|---|---|---|---|
| AQGRAD | 2021 | EcoSIS | Leaf spectra and traits along a water-affinity gradient in Mantua lakes, Italy |
| Talladega 2018 | 2018 | EcoSIS | Leaf reflectance and foliar traits from Talladega National Forest, Alabama |
| CABO 2018-2019 v2 | 2023 update | EcoSIS | Broad multispecies leaf-level spectra with inline trait metadata |
| LOPEX93 | 1993 | JRC / EcoSIS | Lab spectra with biochemical traits attached from `lopex1993metadata.xlsx` |
| Sierra Nevada 2023 | 2023 | EcoSIS | Field spectra with measured traits from California |

CABO and Sierra Nevada use inline trait fields from the EcoSIS export. LOPEX93 uses
the local metadata workbook for trait attachment.

---

## Setup

```bash
pip install -r requirements.txt
```

Optional on Windows with DirectML:

```bash
python -m pip install torch-directml
```

Place `lopex1993metadata.xlsx` in the project root before running.

---

## Running the Pipeline

```bash
python step1_pretrain_downstream.py
python step2_transfer_learning.py
python step3_integrated_discovery.py
```

---

## Stage 1

`step1_pretrain_downstream.py` downloads all five datasets, preprocesses spectra onto
a shared 400-2400 nm grid, trains the autoencoder, extracts 192-dimensional
embeddings, and compares raw spectra vs embeddings for trait regression.

LOPEX93 is evaluated with sample aggregation and species-held-out splits. CABO trait
regression uses the full normalized EcoSIS trait set, including structural, pigment,
biochemical, and elemental traits when present.

## Stage 2

`step2_transfer_learning.py` measures pairwise dataset separability using raw spectra
and learned embeddings. High accuracy here indicates strong domain differences between
datasets rather than easy transfer.

## Stage 3

`step3_integrated_discovery.py` reloads the trained encoder, rebuilds the pooled
dataset, compares trait performance again, and ranks unusual candidate species by
embedding-space novelty with a small hyperaccumulator-genus prior.

---

## Autoencoder

- Input: 201-band spectrum sampled every 10 nm from 400 to 2400 nm
- Encoder: 201 -> 1024 -> 512 -> 256 -> 128 -> 192
- Decoder: symmetric mirror back to 201 bands
- Latent space: 192-dimensional embedding
- Training: 140 epochs, batch size 64, learning rate `7.5e-4`, weight decay `1e-5`,
  Gaussian input noise `std=0.03`

## Preprocessing

1. Savitzky-Golay smoothing with window `11` and polynomial order `3`
2. Standard normal variate normalization
3. Linear interpolation to the shared 400-2400 nm grid

## Discovery

- Isolation Forest with 200 trees on the learned embeddings
- Novelty score from the negated isolation score
- Composite score = novelty + hyperaccumulator-genus prior
