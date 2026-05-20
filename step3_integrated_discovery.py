"""
step3_integrated_discovery.py
Stage 3: Integrated Discovery Pipeline

Full A->C pipeline across all four datasets:
  A. Representation validation  - PCA of embeddings + reconstruction error.
  B. Trait regression comparison - raw spectra vs embeddings (5-fold CV).
     Uses LOPEX93 traits (from metadata xlsx) and Sierra Nevada traits (inline).
  C. Novel candidate discovery  - Isolation Forest novelty scoring + taxonomic prior.

Run:
    python step3_integrated_discovery.py
"""

import io, json, urllib.request, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor, IsolationForest
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import torch, torch.nn as nn
try:
    import torch_directml
except ImportError:
    torch_directml = None

from ecosis_loaders import (
    AQGRAD_PACKAGE_ID,
    TALL2018_PACKAGE_ID,
    filter_lopex_spectral_outliers,
    load_aqgrad,
    load_tall2018,
)

warnings.filterwarnings("ignore")

STAGE1_DIR = Path("outputs/stage1")
OUT_DIR    = Path("outputs/stage3")
OUT_DIR.mkdir(parents=True, exist_ok=True)
SEED = 42; np.random.seed(SEED)
TARGET_WL = np.arange(400, 2401, 10)
DEFAULT_AE_CONFIG = {"hidden": [1024, 512, 256, 128], "latent_dim": 192, "dropout": 0.05}
RIDGE_ALPHAS = np.logspace(-3, 3, 13)

LOPEX93_METADATA   = "lopex1993metadata.xlsx"
LOPEX93_TRAIT_COLS = [
    "Chlorophyll_a+b (µg/cm²)", "Equivalent Water Thickness (g/cm²)",
    "Leaf mass per area (g/cm²)", "Protein_1 (% DW)",
    "Cellulose_1 (% DW)", "Carotenoid (µg/cm²)",
]
SIERRA_METADATA   = "ca2023_freshspectra_supplementaldata.csv"
SIERRA_TRAIT_COLS = ["LMA_g_m2", "LWC_percent", "Chlorophyll_mg_m2"]

KNOWN_HYPERACCUMULATOR_GENERA = {
    "Noccaea","Thlaspi","Arabidopsis","Alyssum","Brassica",
    "Pteris","Sedum","Mesembryanthemum","Haumaniastrum",
    "Viola","Minuartia","Silene","Astragalus",
}

ECOSIS_API = "https://ecosis.org/api/package/{id}/export?format=csv"
DATASETS = {
    "aqgrad":           [AQGRAD_PACKAGE_ID],
    "talladega2018":    [TALL2018_PACKAGE_ID],
    "lopex93":          ["leaf-optical-properties-experiment-database--lopex93-", "3849_2"],
    "sierranevada2023": ["fresh-leaf-spectra-and-measured-traits-from-the-sierra-nevada--ca--in-july-2023"],
}
LABEL_MAP = {"aqgrad": 0, "talladega2018": 1, "lopex93": 2, "sierranevada2023": 3}


# ── Autoencoder ───────────────────────────────────────────────────────────────
class Autoencoder(nn.Module):
    def __init__(self, inp=201, hidden=None, latent=64, dropout=0.0):
        super().__init__()
        if hidden is None: hidden = [256, 128]
        enc = []; d = inp
        for h in hidden:
            enc += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.GELU()]
            if dropout > 0:
                enc.append(nn.Dropout(dropout))
            d = h
        enc.append(nn.Linear(d, latent))
        self.encoder = nn.Sequential(*enc)
        dec = []; d = latent
        for h in reversed(hidden):
            dec += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.GELU()]
            if dropout > 0:
                dec.append(nn.Dropout(dropout))
            d = h
        dec.append(nn.Linear(d, inp))
        self.decoder = nn.Sequential(*dec)

    def forward(self, x): z = self.encoder(x); return self.decoder(z), z
    def encode(self, x):  return self.encoder(x)


class PLS1Regressor(BaseEstimator, RegressorMixin):
    def __init__(self, n_components=8):
        self.n_components = n_components

    def fit(self, X, y):
        self.pipeline_ = Pipeline(
            [
                ("scale", StandardScaler()),
                ("reg", PLSRegression(n_components=self.n_components)),
            ]
        )
        self.pipeline_.fit(X, np.asarray(y).reshape(-1))
        return self

    def predict(self, X):
        return np.asarray(self.pipeline_.predict(X)).reshape(-1)


def resolve_device():
    if torch.cuda.is_available():
        return torch.device("cuda"), "cuda"
    if torch_directml is not None:
        return torch_directml.device(), "directml"
    return torch.device("cpu"), "cpu"


def load_encoder():
    pt = STAGE1_DIR/"encoder.pt"; mn = STAGE1_DIR/"scaler_mean.npy"; ms = STAGE1_DIR/"scaler_std.npy"
    if not all(p.exists() for p in [pt, mn, ms]):
        print("  WARN: Stage 1 encoder not found."); return None, None, None
    cfg_path = STAGE1_DIR / "ae_config.json"
    cfg = DEFAULT_AE_CONFIG
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    device, device_name = resolve_device()
    m = Autoencoder(
        hidden=cfg.get("hidden"),
        latent=cfg.get("latent_dim", 64),
        dropout=cfg.get("dropout", 0.0),
    )
    m.load_state_dict(torch.load(pt, map_location="cpu"))
    m.to(device)
    m.eval()
    print(f"  Loaded encoder on {device_name}")
    return m, np.load(mn), np.load(ms)


# ── Data helpers ──────────────────────────────────────────────────────────────
def is_wl(col):
    try: v = float(col); return 300.0 <= v <= 2600.0
    except (ValueError, TypeError): return False


def fetch_csv(did, timeout=120):
    url = ECOSIS_API.format(id=did)
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 Leaf2Traits/1.0", "Accept": "text/csv,text/plain,*/*"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode("utf-8", errors="replace")
        if raw.strip().startswith("{") or raw.strip().startswith("["): return pd.DataFrame()
        df = pd.read_csv(io.StringIO(raw)); return df if not df.empty else pd.DataFrame()
    except Exception: return pd.DataFrame()



def is_wl(c):
    try: v = float(c); return 300.0 <= v <= 2600.0
    except: return False

def load_sierra_metadata(df_sierra):
    """Positional tile join: 363 EcoSIS rows = 33 samples x 11 scans."""
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

    n_ecosis  = len(df_sierra)
    n_samples = len(meta)
    scans_per = 1
    for c in range(2, 30):
        if n_ecosis % c == 0 and n_ecosis // c <= n_samples:
            scans_per = c
    n_used    = n_ecosis // scans_per
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


def load_all():
    frames = []
    for name, ids in DATASETS.items():
        print(f"  {name} ...")
        if name == "aqgrad":
            df = load_aqgrad()
            if not df.empty:
                df["_dataset"] = name
                print(f"    OK: {df.shape}")
                frames.append(df)
                continue
            print(f"    WARN: download failed -> synthetic")
            frames.append(make_synthetic(name))
            continue
        if name == "talladega2018":
            df = load_tall2018()
            if not df.empty:
                df["_dataset"] = name
                print(f"    OK: {df.shape}")
                frames.append(df)
                continue
            print(f"    WARN: download failed -> synthetic")
            frames.append(make_synthetic(name))
            continue
        for did in ids:
            df = fetch_csv(did)
            if not df.empty:
                # Join metadata BEFORE adding _dataset so non-spectral cols are visible
                if name == "lopex93":
                    lm = load_lopex93_metadata(len(df))
                    if lm is not None:
                        for col in ["Latin Name"] + LOPEX93_TRAIT_COLS:
                            if col in lm.columns:
                                df[col] = lm[col].values
                        df = df.rename(columns={"Latin Name": "Species"})
                        print(f"    Joined LOPEX93 metadata")
                if name == "sierranevada2023":
                    df = load_sierra_metadata(df)
                df["_dataset"] = name
                print(f"    OK: {df.shape}")
                frames.append(df); break
        else:
            print(f"    WARN: download failed -> synthetic")
            frames.append(make_synthetic(name))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_lopex93_metadata(n_spectra):
    p = Path(LOPEX93_METADATA)
    if not p.exists(): return None
    meta = pd.read_excel(p).replace(-999, np.nan)
    sample_ids = np.arange(n_spectra) // 2 + 1
    return pd.DataFrame({"Sample_#": sample_ids}).merge(meta, on="Sample_#", how="left")


def make_synthetic(name, n=120):
    rng = np.random.default_rng(abs(hash(name)) % (2**31))
    wl  = np.arange(400, 2401, 10)
    base = 0.05 + 0.30 * np.exp(-((wl-800)**2)/(2*200**2))
    spec = np.clip(base + rng.normal(0, 0.025, (n, len(wl))), 0, 1)
    meta = {"_dataset": name, "Species": [f"{name}_sp{j%6}" for j in range(n)]}
    if name == "sierranevada2023":
        meta.update({"LMA": rng.uniform(50,300,n), "Nmass": rng.uniform(1,4,n)})
    return pd.DataFrame({**meta, **{str(w): spec[:,i] for i,w in enumerate(wl)}})


def preprocess(df):
    wl_cols = sorted([c for c in df.columns if is_wl(c)], key=float)
    wl = np.array([float(c) for c in wl_cols])
    mat = df[wl_cols].values.astype(np.float32)
    bad = np.isnan(mat).mean(1) > 0.5
    mat = mat[~bad]; dfok = df.loc[df.index[~bad]].reset_index(drop=True)
    mat = np.nan_to_num(mat, nan=0.0)
    mat = savgol_filter(mat, window_length=11, polyorder=3, axis=1)
    mu  = mat.mean(1, keepdims=True); sd = mat.std(1, keepdims=True) + 1e-8
    mat = (mat - mu) / sd
    out = np.zeros((mat.shape[0], len(TARGET_WL)), dtype=np.float32)
    for i, row in enumerate(mat): out[i] = np.interp(TARGET_WL, wl, row)
    if "_dataset" in dfok.columns:
        lopex_ix = np.where(dfok["_dataset"].values == "lopex93")[0]
        if len(lopex_ix):
            Xl, dfl = filter_lopex_spectral_outliers(
                out[lopex_ix], dfok.iloc[lopex_ix].reset_index(drop=True)
            )
            keep = np.ones(len(dfok), dtype=bool)
            keep[lopex_ix] = False
            out = np.vstack([out[keep], Xl])
            dfok = pd.concat(
                [dfok.iloc[keep].reset_index(drop=True), dfl], ignore_index=True
            )
    return out, dfok


def encode(X, model, mu, std):
    if model is None: return X
    device = next(model.parameters()).device
    Xn = (X - mu) / std
    with torch.no_grad(): z = model.encode(torch.tensor(Xn, dtype=torch.float32).to(device))
    return z.cpu().numpy()


def reconstruct(X, model, mu, std):
    if model is None: return X
    device = next(model.parameters()).device
    Xn = (X - mu) / std
    with torch.no_grad(): r, _ = model(torch.tensor(Xn, dtype=torch.float32).to(device))
    return r.cpu().numpy() * std + mu


# ── Option A ──────────────────────────────────────────────────────────────────
def validate_representations(X_raw, Z, meta, model, mu, std):
    print("  [A] Representation validation ...")
    datasets = meta.get("_dataset", pd.Series(["unknown"]*len(meta)))
    colours  = ["#e74c3c","#2ecc71","#3498db","#f39c12"]
    pca = PCA(n_components=2, random_state=SEED)
    Z2  = pca.fit_transform(Z)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    for i, ds in enumerate(datasets.unique()):
        m = datasets == ds
        ax.scatter(Z2[m,0], Z2[m,1], label=ds, alpha=0.55, s=20, color=colours[i%4])
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title("Embedding PCA (4 datasets)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1]
    npz = STAGE1_DIR/"embeddings_all.npz"
    if npz.exists() and model is not None:
        data = np.load(npz); X_s = data["X"]
        recon = reconstruct(X_s, model, mu, std)
        recon_err = np.mean((X_s - recon)**2, axis=1)
        ax.hist(recon_err, bins=40, color="#2563eb", alpha=0.7, edgecolor="white")
        ax.set_xlabel("Reconstruction MSE (per sample)")
    else:
        ax.hist(np.random.exponential(0.01, 400), bins=40, color="#2563eb", alpha=0.7)
        ax.set_xlabel("Reconstruction MSE (demo)")
    ax.set_ylabel("Count"); ax.set_title("Reconstruction Error Distribution"); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR/"embedding_validation.png", dpi=150); plt.close(fig)
    print("  Saved: embedding_validation.png")


# ── Option B ──────────────────────────────────────────────────────────────────
def build_regressor_candidates(n_samples, n_features):
    n_components = max(2, min(16, n_features, n_samples - 1))
    min_leaf = max(4, n_samples // 20)
    return {
        "ridge_cv": Pipeline(
            [("scale", StandardScaler()), ("reg", RidgeCV(alphas=RIDGE_ALPHAS))]
        ),
        "pls": PLS1Regressor(n_components=n_components),
        "hist_gbr": HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_depth=4,
            max_iter=400,
            min_samples_leaf=min_leaf,
            l2_regularization=1e-3,
            random_state=SEED,
        ),
    }


def select_best_regressor_cv(X, y):
    n_samples, n_features = X.shape
    cv_splits = min(5, max(3, n_samples // 10))
    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=SEED)
    best_name, best_score = None, -np.inf
    for name, model in build_regressor_candidates(n_samples, n_features).items():
        score = cross_val_score(clone(model), X, y, cv=cv, scoring="r2").mean()
        if score > best_score:
            best_name, best_score = name, score
    return best_name, best_score


def compare_trait_regression(X_raw, Z, meta):
    print("  [B] Trait regression comparison (raw vs embedding) ...")

    # Collect trait columns: LOPEX93 traits + renamed Sierra Nevada traits
    known = LOPEX93_TRAIT_COLS + SIERRA_TRAIT_COLS
    all_traits = [c for c in known if c in meta.columns and meta[c].notna().sum() >= 10]
    skip = set(known) | {"_dataset", "sample_id", "sample_number",
                          "latitude_of_plot", "longitude_of_plot"}
    for col in meta.columns:
        if col not in all_traits and pd.api.types.is_numeric_dtype(meta[col]) \
                and meta[col].notna().sum() >= 10 and col not in skip:
            all_traits.append(col)

    if not all_traits:
        print("    No trait columns found. Skipping.")
        return

    rows = []
    for trait in all_traits[:8]:   # cap at 8 to keep plot readable
        y  = meta[trait].values.astype(float)
        ok = ~np.isnan(y)
        if ok.sum() < 15: continue
        yr, Xr, Xz = y[ok], X_raw[ok], Z[ok]
        raw_name, r2_raw = select_best_regressor_cv(Xr, yr)
        emb_name, r2_emb = select_best_regressor_cv(Xz, yr)
        rows.append(
            {
                "trait": trait,
                "R2_raw": r2_raw,
                "R2_emb": r2_emb,
                "model_raw": raw_name,
                "model_emb": emb_name,
            }
        )
        print(
            f"    {trait:<42}  raw={raw_name}:{r2_raw:.3f}  "
            f"emb={emb_name}:{r2_emb:.3f}  gain={r2_emb-r2_raw:+.3f}"
        )

    if not rows: return
    df_r = pd.DataFrame(rows)
    x = np.arange(len(df_r)); w = 0.35
    fig, ax = plt.subplots(figsize=(max(8, 2*len(df_r)), 5))
    ax.bar(x - w/2, df_r["R2_raw"], w, label="Raw Spectra",  color="#94a3b8")
    ax.bar(x + w/2, df_r["R2_emb"], w, label="Embeddings",   color="#2563eb")
    ax.set_xticks(x)
    labels = [t[:30] for t in df_r["trait"]]
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("R² (5-fold CV)"); ax.set_title("Trait Prediction: Raw vs Embedding")
    ax.axhline(0, color="black", lw=0.8); ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR/"trait_comparison.png", dpi=150); plt.close(fig)
    print("  Saved: trait_comparison.png")


# ── Option C ──────────────────────────────────────────────────────────────────
def genus_from_species(sp):
    if pd.isna(sp): return "Unknown"
    return str(sp).split("_")[0].split(" ")[0]


def run_discovery(X_raw, Z, meta):
    print("  [C] Novel candidate discovery ...")
    iso = IsolationForest(n_estimators=200, contamination=0.1, random_state=SEED, n_jobs=-1)
    iso.fit(Z)
    novelty = -iso.score_samples(Z)
    novelty = (novelty - novelty.min()) / (novelty.max() - novelty.min() + 1e-8)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(novelty, bins=50, color="#e74c3c", alpha=0.75, edgecolor="white")
    ax.axvline(np.percentile(novelty, 90), color="black", ls="--", lw=1.5, label="90th pct")
    ax.set_xlabel("Novelty Score (higher = more spectrally unusual)")
    ax.set_ylabel("Count"); ax.set_title("Novelty Score Distribution (4 datasets)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT_DIR/"novelty_distribution.png", dpi=150); plt.close(fig)
    print("  Saved: novelty_distribution.png")

    # Species / genus
    sp_col = meta["Species"] if "Species" in meta.columns else pd.Series(["Unknown"]*len(meta))
    ds_col = meta.get("_dataset", pd.Series(["unknown"]*len(meta)))
    genera = sp_col.apply(genus_from_species)

    # Full per-spectrum table (saved to CSV for reference)
    all_spectra = pd.DataFrame({
        "species":        sp_col.values,
        "genus":          genera.values,
        "dataset":        ds_col.values,
        "novelty_score":  novelty,
        "known_hyperacc": genera.apply(lambda g: g in KNOWN_HYPERACCUMULATOR_GENERA).values,
    })
    all_spectra["prior_boost"]     = all_spectra["known_hyperacc"].astype(float) * 0.15
    all_spectra["composite_score"] = all_spectra["novelty_score"] + all_spectra["prior_boost"]
    all_spectra.to_csv(OUT_DIR/"all_spectra_novelty.csv", index=False)

    # Collapse to one row per species: take the highest novelty scan
    # This prevents any single species (e.g. Quercus pubescens with 22 scans)
    # from dominating the rankings.
    species_ranked = (
        all_spectra
        .sort_values("novelty_score", ascending=False)
        .groupby("species", sort=False)
        .first()           # highest-novelty scan per species
        .reset_index()
        .sort_values("composite_score", ascending=False)
        .reset_index(drop=True)
    )
    species_ranked["rank"]       = species_ranked.index + 1
    species_ranked["n_spectra"]  = (
        all_spectra.groupby("species")["novelty_score"]
        .count()
        .reindex(species_ranked["species"])
        .values
    )
    species_ranked.to_csv(OUT_DIR/"ranked_candidates.csv", index=False)
    print(f"  Saved: ranked_candidates.csv  ({len(species_ranked)} unique species)")
    print(f"  Saved: all_spectra_novelty.csv  ({len(all_spectra)} individual spectra)")

    # Novelty histogram — one bar per species (max score)
    fig, ax = plt.subplots(figsize=(max(10, len(species_ranked)*0.4), 5))
    colours = species_ranked["known_hyperacc"].map({True: "#e74c3c", False: "#3498db"})
    ax.bar(range(len(species_ranked)), species_ranked["composite_score"],
           color=colours, edgecolor="white", linewidth=0.4)
    ax.axhline(np.percentile(species_ranked["composite_score"], 75),
               color="black", ls="--", lw=1.2, label="75th percentile")
    ax.set_xticks(range(len(species_ranked)))
    ax.set_xticklabels(species_ranked["species"], rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("Composite Novelty Score (max scan per species)")
    ax.set_title("Species-Level Novelty Ranking")
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color="#e74c3c", label="Known hyperaccumulator genus"),
        Patch(color="#3498db", label="Other species"),
        plt.Line2D([0],[0], color="black", ls="--", label="75th percentile"),
    ])
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR/"novelty_distribution.png", dpi=150); plt.close(fig)
    print("  Saved: novelty_distribution.png")

    print(f"\n  Top-10 candidate species (one entry per species, highest novelty scan):")
    print(f"  {'Rank':>4}  {'Species':<30} {'Genus':<15} {'Dataset':<16} {'Novelty':>8}  {'N scans':>7}  {'HyperAcc?':>9}")
    print("  " + "-"*95)
    for _, row in species_ranked.head(10).iterrows():
        ha = "YES" if row["known_hyperacc"] else "-"
        print(f"  {int(row['rank']):>4}  {str(row['species']):<30} "
              f"{str(row['genus']):<15} {str(row['dataset']):<16} "
              f"{row['novelty_score']:>8.4f}  {int(row['n_spectra']):>7}  {ha:>9}")


# ── Custom candidates ─────────────────────────────────────────────────────────
def load_custom_candidates(path="candidates_local_wildplants.csv"):
    p = Path(path)
    if not p.exists(): return None, None
    df = pd.read_csv(p)
    print(f"  Custom wildplant file loaded: {df.shape}")
    Xp, dfp = preprocess(df)
    return Xp, dfp


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*65)
    print("  Stage 3 - Integrated Discovery Pipeline")
    print("="*65)

    print("\n[1/5] Loading encoder ...")
    model, mu, std = load_encoder()

    print("\n[2/5] Loading datasets ...")
    df = load_all()

    print("\n[3/5] Preprocessing ...")
    X_raw, meta = preprocess(df)
    print(f"  Spectra: {X_raw.shape}")
    Z = encode(X_raw, model, mu, std)
    print(f"  Embeddings: {Z.shape}")

    X_cust, meta_cust = load_custom_candidates()
    if X_cust is not None:
        Z_cust = encode(X_cust, model, mu, std)
        X_raw  = np.vstack([X_raw, X_cust])
        Z      = np.vstack([Z, Z_cust])
        meta   = pd.concat([meta, meta_cust], ignore_index=True)
        print(f"  After custom candidates: {X_raw.shape}")

    print("\n[4/5] Running pipeline A->C ...")
    validate_representations(X_raw, Z, meta, model, mu, std)
    compare_trait_regression(X_raw, Z, meta)
    run_discovery(X_raw, Z, meta)

    print(f"\n{'='*65}")
    print(f"  Stage 3 complete. Outputs in: {OUT_DIR.resolve()}")
    print("="*65)
    print("""
  Custom wildplant input (candidates_local_wildplants.csv):
    species, 400, 410, ..., 2400
    Viola_riviniana, 0.042, 0.041, ..., 0.031
  Add a 'species' column; wavelength columns as bare nm integers.
""")


if __name__ == "__main__":
    main()
