"""
step1_pretrain_downstream.py
Stage 1: Pretraining + Downstream Prediction

Datasets:
  - AQGRAD               (EcoSIS, water-affinity gradient, Mantua lakes, Italy)
  - Talladega 2018       (EcoSIS, leaf spectra + foliar traits, Alabama)
  - LOPEX93              (EcoSIS spectra + lopex1993metadata.xlsx traits)
  - Sierra Nevada 2023   (EcoSIS, field, California - includes measured traits)

Pipeline:
  1. Download all four datasets.
  2. Preprocess: Savitzky-Golay smooth -> SNV normalise -> align to 400-2400nm grid.
  3. Train deep autoencoder (self-supervised, 60 epochs).
  4. Extract 64-dim embeddings; visualise with PCA.
  5. Trait regression on LOPEX93 (via metadata xlsx) and Sierra Nevada traits.

Run:
    python step1_pretrain_downstream.py
"""

import io, urllib.request, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, classification_report
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from ecosis_loaders import AQGRAD_PACKAGE_ID, TALL2018_PACKAGE_ID, load_aqgrad, load_tall2018

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
OUT_DIR   = Path("outputs/stage1")
OUT_DIR.mkdir(parents=True, exist_ok=True)
SEED      = 42
torch.manual_seed(SEED); np.random.seed(SEED)

TARGET_WL  = np.arange(400, 2401, 10)   # 201 bands
LATENT_DIM = 64
HIDDEN     = [256, 128]
BATCH_SIZE = 64
N_EPOCHS   = 60
LR         = 1e-3

LOPEX93_METADATA   = "lopex1993metadata.xlsx"
LOPEX93_TRAIT_COLS = [
    "Chlorophyll_a+b (µg/cm²)",
    "Equivalent Water Thickness (g/cm²)",
    "Leaf mass per area (g/cm²)",
    "Protein_1 (% DW)",
    "Cellulose_1 (% DW)",
    "Carotenoid (µg/cm²)",
]

# Sierra Nevada trait columns (confirmed from EcoSIS package description)
SIERRA_METADATA    = "ca2023_freshspectra_supplementaldata.csv"
# Exact column names in the supplemental CSV
SIERRA_TRAIT_COLS  = [
    "LMA_value_g/m2",
    "leaf_water_content_percent",
    "chlorophyll _mg/m2",
]

ECOSIS_API = "https://ecosis.org/api/package/{id}/export?format=csv"
DATASETS = {
    "aqgrad":        [AQGRAD_PACKAGE_ID],
    "talladega2018":  [TALL2018_PACKAGE_ID],
    "lopex93":       ["leaf-optical-properties-experiment-database--lopex93-", "3849_2"],
    "sierranevada2023": ["fresh-leaf-spectra-and-measured-traits-from-the-sierra-nevada--ca--in-july-2023"],
}
LABEL_MAP   = {"aqgrad": 0, "talladega2018": 1, "lopex93": 2, "sierranevada2023": 3}
LABEL_NAMES = {0: "AQGRAD", 1: "Talladega 2018", 2: "LOPEX93", 3: "Sierra Nevada 2023"}


# ── I/O ───────────────────────────────────────────────────────────────────────
def fetch_csv(did, timeout=120):
    url = ECOSIS_API.format(id=did)
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 Leaf2Traits/1.0",
            "Accept": "text/csv,text/plain,*/*",
        })
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode("utf-8", errors="replace")
        if raw.strip().startswith("{") or raw.strip().startswith("["):
            return pd.DataFrame()
        df = pd.read_csv(io.StringIO(raw))
        return df if not df.empty else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def load_dataset(name, ids):
    print(f"  {name} ...")
    if name == "aqgrad":
        df = load_aqgrad()
        if not df.empty:
            print(f"    OK {AQGRAD_PACKAGE_ID}: {df.shape}")
            df["_dataset"] = name
            return df
        print(f"    WARN: AQGRAD download failed -> synthetic fallback")
        return pd.DataFrame()
    if name == "talladega2018":
        df = load_tall2018()
        if not df.empty:
            print(f"    OK {TALL2018_PACKAGE_ID}: {df.shape}")
            df["_dataset"] = name
            return df
        print(f"    WARN: Talladega 2018 download failed -> synthetic fallback")
        return pd.DataFrame()
    for did in ids:
        df = fetch_csv(did)
        if not df.empty:
            print(f"    OK {did}: {df.shape}")
            if name == "sierranevada2023":
                # Join metadata BEFORE adding _dataset so non-spectral cols are visible
                df = load_sierra_metadata(df)
            df["_dataset"] = name
            return df
    print(f"    WARN: download failed -> synthetic fallback")
    return pd.DataFrame()


def load_lopex93_metadata(n_spectra):
    p = Path(LOPEX93_METADATA)
    if not p.exists():
        print(f"  INFO: {LOPEX93_METADATA} not found.")
        return None
    meta = pd.read_excel(p).replace(-999, np.nan)
    sample_ids = np.arange(n_spectra) // 2 + 1
    joined = pd.DataFrame({"Sample_#": sample_ids}).merge(meta, on="Sample_#", how="left")
    print(f"  Loaded LOPEX93 metadata: {len(meta)} samples -> {joined['Chlorophyll_a+b (µg/cm²)'].notna().sum()} spectra with traits")
    return joined


def make_synthetic(name, n=120):
    rng  = np.random.default_rng(abs(hash(name)) % (2**31))
    wl   = np.arange(400, 2401, 10)
    base = 0.05 + 0.30 * np.exp(-((wl - 800)**2) / (2*200**2))
    spec = np.clip(base + rng.normal(0, 0.025, (n, len(wl))), 0, 1)
    meta = {"_dataset": name, "Species": [f"{name}_sp{j%6}" for j in range(n)]}
    if name == "lopex93":
        meta.update({c: rng.uniform(1, 50, n) for c in LOPEX93_TRAIT_COLS})
    if name == "sierranevada2023":
        meta.update({"LMA": rng.uniform(50, 300, n),
                     "Nmass": rng.uniform(1, 4, n),
                     "EWT": rng.uniform(0.005, 0.04, n)})
    return pd.DataFrame({**meta, **{str(w): spec[:, i] for i, w in enumerate(wl)}})


# ── Preprocessing ─────────────────────────────────────────────────────────────
def is_wl(col):
    try:
        v = float(col); return 300.0 <= v <= 2600.0
    except (ValueError, TypeError):
        return False


def preprocess_dataset(df):
    wl_cols = sorted([c for c in df.columns if is_wl(c)], key=float)
    wl  = np.array([float(c) for c in wl_cols])
    mat = df[wl_cols].values.astype(np.float32)
    bad = np.isnan(mat).mean(1) > 0.5
    mat = mat[~bad]; df = df.loc[df.index[~bad]].reset_index(drop=True)
    mat = np.nan_to_num(mat, nan=0.0)
    mat = savgol_filter(mat, window_length=11, polyorder=3, axis=1)
    mu  = mat.mean(1, keepdims=True); sd = mat.std(1, keepdims=True) + 1e-8
    mat = (mat - mu) / sd
    out = np.zeros((mat.shape[0], len(TARGET_WL)), dtype=np.float32)
    for i, row in enumerate(mat):
        out[i] = np.interp(TARGET_WL, wl, row)
    return out, df


def preprocess_all(datasets):
    all_X, all_y, all_meta = [], [], []
    for name, df in datasets.items():
        Xp, dfp = preprocess_dataset(df)
        if Xp.shape[0] == 0:
            continue
        all_X.append(Xp)
        all_y.extend([LABEL_MAP.get(name, -1)] * Xp.shape[0])
        mc = [c for c in dfp.columns if not is_wl(c)]
        all_meta.append(dfp[mc])
    return (np.vstack(all_X).astype(np.float32),
            np.array(all_y, dtype=np.int64),
            pd.concat(all_meta, ignore_index=True))


# ── Autoencoder ───────────────────────────────────────────────────────────────
class Autoencoder(nn.Module):
    def __init__(self, inp=201, hidden=None, latent=64):
        super().__init__()
        if hidden is None: hidden = [256, 128]
        enc = []; d = inp
        for h in hidden:
            enc += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.GELU()]; d = h
        enc.append(nn.Linear(d, latent))
        self.encoder = nn.Sequential(*enc)
        dec = []; d = latent
        for h in reversed(hidden):
            dec += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.GELU()]; d = h
        dec.append(nn.Linear(d, inp))
        self.decoder = nn.Sequential(*dec)

    def forward(self, x): z = self.encoder(x); return self.decoder(z), z
    def encode(self, x):  return self.encoder(x)


def train_ae(X):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    mu = X.mean(0); std = X.std(0) + 1e-8
    Xn = (X - mu) / std
    dl = DataLoader(TensorDataset(torch.tensor(Xn, dtype=torch.float32)),
                    batch_size=BATCH_SIZE, shuffle=True)
    model = Autoencoder(X.shape[1], HIDDEN, LATENT_DIM).to(device)
    opt   = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_EPOCHS)
    fn    = nn.MSELoss(); hist = []
    for ep in range(1, N_EPOCHS+1):
        model.train(); el = 0.0
        for (b,) in dl:
            b = b.to(device); r, _ = model(b)
            loss = fn(r, b); opt.zero_grad(); loss.backward(); opt.step()
            el += loss.item() * len(b)
        el /= len(Xn); hist.append(el); sched.step()
        if ep % 10 == 0 or ep == 1:
            print(f"    Epoch {ep:3d}/{N_EPOCHS}  loss={el:.5f}")
    return model, hist, mu, std


def get_embeddings(model, X, mu, std):
    model.eval()
    device = next(model.parameters()).device
    Xn = (X - mu) / std
    with torch.no_grad():
        z = model.encode(torch.tensor(Xn, dtype=torch.float32).to(device))
    return z.cpu().numpy()


# ── Plots ─────────────────────────────────────────────────────────────────────
def plot_loss(hist):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, len(hist)+1), hist, lw=2, color="#2563eb")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss")
    ax.set_title("Autoencoder Training Loss"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT_DIR/"autoencoder_loss.png", dpi=150); plt.close(fig)
    print("  Saved: autoencoder_loss.png")


def plot_pca(Z, labels):
    pca = PCA(n_components=2, random_state=SEED)
    Z2  = pca.fit_transform(Z)
    colours = ["#e74c3c", "#2ecc71", "#3498db", "#f39c12"]
    fig, ax = plt.subplots(figsize=(9, 6))
    for i, u in enumerate(np.unique(labels)):
        m = labels == u
        ax.scatter(Z2[m, 0], Z2[m, 1], label=LABEL_NAMES.get(u, str(u)),
                   alpha=0.6, s=25, color=colours[i % len(colours)])
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title("PCA of Spectral Embeddings (by dataset)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT_DIR/"embedding_pca.png", dpi=150); plt.close(fig)
    print("  Saved: embedding_pca.png")


# ── Downstream ────────────────────────────────────────────────────────────────
def run_trait_regression(Z, trait_df, tag=""):
    present = [c for c in trait_df.columns
               if pd.api.types.is_numeric_dtype(trait_df[c])
               and trait_df[c].notna().sum() >= 15]
    if not present:
        return False
    print(f"  Traits ({tag}): {present}")
    for trait in present:
        y = trait_df[trait].values.astype(float)
        ok = ~np.isnan(y)
        if ok.sum() < 15: continue
        Xt, yt = Z[ok], y[ok]
        Xtr, Xte, ytr, yte = train_test_split(Xt, yt, test_size=0.25, random_state=SEED)
        reg = Ridge(alpha=1.0); reg.fit(Xtr, ytr)
        yp  = reg.predict(Xte)
        r2  = r2_score(yte, yp); rmse = mean_squared_error(yte, yp)**0.5
        print(f"    {trait:<40}  R2={r2:.3f}  RMSE={rmse:.4g}")
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(yte, yp, alpha=0.7, s=30, color="#2563eb")
        mn, mx = min(yte.min(), yp.min()), max(yte.max(), yp.max())
        ax.plot([mn, mx], [mn, mx], "r--", lw=1.5)
        ax.set_xlabel(f"True {trait}"); ax.set_ylabel("Predicted")
        ax.set_title(f"{trait}  R2={r2:.3f}"); fig.tight_layout()
        safe = trait.replace("/","_").replace(" ","_").replace("(","").replace(")","").replace("%","pct")
        fig.savefig(OUT_DIR/f"trait_scatter_{tag}_{safe}.png", dpi=150); plt.close(fig)
    return True


def run_classification_fallback(Z, labels):
    print("  Dataset classification fallback:")
    Xtr, Xte, ytr, yte = train_test_split(Z, labels, test_size=0.25,
                                            stratify=labels, random_state=SEED)
    clf = LogisticRegression(max_iter=500, random_state=SEED)
    clf.fit(Xtr, ytr)
    names = [LABEL_NAMES[i] for i in sorted(LABEL_NAMES)]
    print(classification_report(yte, clf.predict(Xte), target_names=names))



# ── Sierra Nevada 2023 metadata loader ────────────────────────────────────────
def load_sierra_metadata(df_sierra):
    """
    Join ca2023_freshspectra_supplementaldata.csv to the EcoSIS Sierra Nevada
    spectra by positional tiling.

    The EcoSIS export is pure spectra with no metadata columns.
    363 rows = 33 samples x 11 scans each. The supplemental CSV has 35 samples;
    samples 75 and 76 (the two highest sample_numbers) are absent from EcoSIS.
    We sort by sample_number, take the first 33, and tile each 11 times.
    """
    p = Path(SIERRA_METADATA)
    if not p.exists():
        print(f"    INFO: {SIERRA_METADATA} not found.")
        return df_sierra

    meta = pd.read_csv(p)
    meta["genus"] = meta["genus"].str.strip()
    meta["Species"] = meta.apply(
        lambda r: f"{r['genus']} {r['species']}"
                  if pd.notna(r["species"]) and str(r["species"]).strip() not in ("", "nan")
                  else r["genus"],
        axis=1
    )
    meta = meta.rename(columns={
        "LMA_value_g/m2":           "LMA_g_m2",
        "leaf_water_content_percent":"LWC_percent",
        "chlorophyll _mg/m2":       "Chlorophyll_mg_m2",
    })
    meta = meta.sort_values("sample_number").reset_index(drop=True)

    n_ecosis = len(df_sierra)   # 363
    # Find scans_per_sample: largest divisor of n_ecosis <= n_samples
    n_samples = len(meta)       # 35
    scans_per = 1
    for c in range(2, 30):
        if n_ecosis % c == 0 and n_ecosis // c <= n_samples:
            scans_per = c
    n_used   = n_ecosis // scans_per   # 33
    meta_used = meta.head(n_used).reset_index(drop=True)

    tiled = meta_used.loc[meta_used.index.repeat(scans_per)].reset_index(drop=True)
    keep  = [c for c in ["Species","genus","LMA_g_m2","LWC_percent",
                          "Chlorophyll_mg_m2","common_name","plant_functional_type","family"]
             if c in tiled.columns]

    df_out = df_sierra.reset_index(drop=True)
    for col in keep:
        df_out[col] = tiled[col].values

    n_matched = df_out["Species"].notna().sum()
    print(f"    Positional join: {n_used} samples x {scans_per} scans = "
          f"{n_used*scans_per} rows, {n_matched} species matched")
    return df_out


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*65)
    print("  Stage 1 - Pretraining + Downstream Prediction")
    print("="*65)

    print("\n[1/6] Loading datasets ...")
    raws = {}
    for name, ids in DATASETS.items():
        df = load_dataset(name, ids)
        if df.empty:
            df = make_synthetic(name); print(f"    Using synthetic {name}: {df.shape}")
        raws[name] = df

    print("\n[2/6] Preprocessing ...")
    X, labels, meta = preprocess_all(raws)
    print(f"  Pooled: {X.shape}")

    print("\n[3/6] Training autoencoder ...")
    model, hist, mu, std = train_ae(X)
    torch.save(model.state_dict(), OUT_DIR/"encoder.pt")
    np.save(OUT_DIR/"scaler_mean.npy", mu)
    np.save(OUT_DIR/"scaler_std.npy",  std)
    np.save(OUT_DIR/"target_wl.npy",   TARGET_WL)
    print(f"  Final loss: {hist[-1]:.5f}")

    print("\n[4/6] Extracting embeddings ...")
    Z = get_embeddings(model, X, mu, std)
    np.savez(OUT_DIR/"embeddings_all.npz", Z=Z, labels=labels, X=X)
    print(f"  Embedding shape: {Z.shape}")

    print("\n[5/6] Plots ...")
    plot_loss(hist); plot_pca(Z, labels)

    print("\n[6/6] Downstream tasks ...")
    trait_found = False

    # LOPEX93 traits via metadata xlsx
    lmask = labels == LABEL_MAP["lopex93"]
    if lmask.sum() > 0:
        lopex_meta = load_lopex93_metadata(int(lmask.sum()))
        if lopex_meta is not None:
            trait_df = lopex_meta[[c for c in LOPEX93_TRAIT_COLS if c in lopex_meta.columns]]
            ok = run_trait_regression(Z[lmask], trait_df, tag="LOPEX93")
            if ok: trait_found = True

    # Sierra Nevada traits (inline in CSV)
    smask = labels == LABEL_MAP["sierranevada2023"]
    if smask.sum() > 0:
        sm = meta[smask].reset_index(drop=True)
        # Use renamed trait cols first, then any numeric column
        renamed = ["LMA_g_m2", "LWC_percent", "Chlorophyll_mg_m2"]
        sierra_traits = [c for c in renamed if c in sm.columns and sm[c].notna().sum() >= 10]
        if not sierra_traits:
            sierra_traits = [c for c in sm.columns
                             if pd.api.types.is_numeric_dtype(sm[c])
                             and sm[c].notna().sum() >= 10
                             and c not in ("_dataset",)]
        if sierra_traits:
            ok = run_trait_regression(Z[smask], sm[sierra_traits], tag="SierraNevada")
            if ok: trait_found = True

    if not trait_found:
        print("  No trait columns found. Running dataset classification fallback.")
        run_classification_fallback(Z, labels)

    print(f"\n{'='*65}")
    print(f"  Stage 1 complete. Outputs in: {OUT_DIR.resolve()}")
    print("="*65 + "\n")


if __name__ == "__main__":
    main()