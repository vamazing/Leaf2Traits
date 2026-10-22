# Leaf2Traits — 2-Minute Video Script

**Total runtime: ~2:00 (≈300 words spoken at natural pace)**

---

## INTRODUCTION (~20 sec)

"Hi, I'm Vidyut Krishnan, and I'm a student at [School Name]. My project is called **Leaf2Traits** — a deep learning pipeline that reads the fingerprint of light reflected off a leaf to tell us what's happening inside it, without ever touching the leaf itself."

---

## RESEARCH PROBLEM (~30 sec)

"Every plant leaf reflects sunlight in a unique pattern across hundreds of wavelengths — that's called a **hyperspectral signature**. Researchers want to use these signatures to measure leaf traits like chlorophyll content, water content, and nitrogen — but there's a problem.

Each field or lab uses a **different spectrometer**, producing slightly different measurements. A model trained on one dataset often fails on another. My goal was to build a system that learns **universal leaf features** from many datasets at once, and then prove whether those features actually transfer."

---

## HOW IT WORKS (~40 sec)

"The pipeline runs in **three stages**.

In **Stage 1**, I preprocess spectra from eight public plant databases — over 13,000 samples — and train an **autoencoder** that compresses each 201-band spectrum into a compact **192-dimensional representation**. A supervised probe then predicts six biochemical traits from that representation, and I compare it against a PCA baseline.

In **Stage 2**, I test whether those learned features **transfer across datasets** — training on one dataset and testing on every other, plus a leave-one-dataset-out validation with balanced accuracy.

In **Stage 3**, I combine reconstruction error and latent-space distance into a **novelty detector** that flags spectrally unusual leaves for manual inspection."

---

## KEY FINDINGS (~30 sec)

"Three headline results:

**First** — the autoencoder's latent space organized leaves by species family and plant functional type — biologically meaningful structure that no one explicitly taught it.

**Second** — transfer performance drops across datasets, but I showed this is driven by **sensor differences, not biology**. Spectral distribution analysis confirmed all eight datasets are statistically similar after preprocessing.

**Third** — the autoencoder reached an R-squared of **0.856** on trait prediction, while PCA-192 only reached **0.645** — proving the network captures **nonlinear biological variation** that linear methods miss."

---

## IMPACT & CONCLUSION (~20 sec)

"These findings matter because they show that self-supervised learning can build **reusable, cross-dataset representations** for plant science — reducing the need to retrain models every time a new field campaign adds data. This could power faster crop monitoring, forest health assessment, and biodiversity surveys — anywhere leaves need to be measured at scale.

Thank you."

---

## PRODUCTION NOTES

| Element | Suggestion |
|---|---|
| **On-screen text** | Show "13,000+ samples · 8 datasets · 100+ species" during intro |
| **B-roll / slides** | Mermaid pipeline diagram during stages; PCA plot; reconstruction examples figure |
| **Key stat callouts** | R² = 0.856 (autoencoder) vs 0.645 (PCA) — animate side by side |
| **Tone** | Confident, conversational — avoid reading word-for-word |
| **Pacing** | ~150 words/min; pause briefly after each numbered finding |
