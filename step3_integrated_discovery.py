"""
step3_integrated_discovery.py
Stage 3: Integrated Discovery Pipeline

Full A->C pipeline across all datasets:
  A. Representation validation  - PCA of embeddings + reconstruction error.
  B. Trait regression comparison - raw spectra vs embeddings (5-fold CV).
     Uses API-provided LOPEX93, CABO, NGEE Panama, and Sierra Nevada traits.
  C. Novel candidate discovery  - Isolation Forest novelty scoring + taxonomic prior.

Run:
    python step3_integrated_discovery.py
"""

import io, json, os, shutil, urllib.request, warnings
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
from sklearn.model_selection import GroupKFold, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import torch, torch.nn as nn
try:
    import torch_directml
except ImportError:
    torch_directml = None

from ecosis_loaders import (
    AQGRAD_PACKAGE_ID,
    CABO_V2_PACKAGE_ID,
    NGEE_PANAMA_PACKAGE_ID,
    LOPEX93_PACKAGE_IDS,
    SIERRA_NEVADA_2023_PACKAGE_ID,
    TALL2018_PACKAGE_ID,
    filter_lopex_spectral_outliers,
    load_aqgrad,
    load_cabo_v2,
    load_ngee_panama,
    load_lopex93,
    load_sierra_nevada_2023,
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

LOPEX93_TRAIT_COLS = [
    "Chlorophyll_a+b (µg/cm²)", "Equivalent Water Thickness (g/cm²)",
    "Leaf mass per area (g/cm²)", "Protein_1 (% DW)",
    "Cellulose_1 (% DW)", "Carotenoid (µg/cm²)",
]
CABO_V2_TRAIT_COLS = [
    "LMA_g_m2",
    "EWT_g_cm2",
    "LDMC_mg_g",
    "C_mass_g_g",
    "N_mass_g_g",
    "C_area_g_m2",
    "N_area_g_m2",
    "CN_ratio",
    "Carotenoid_mass_g_g",
    "Carotenoid_area_g_m2",
    "Chlorophyll_A_mass_g_g",
    "Chlorophyll_A_area_g_m2",
    "Chlorophyll_B_mass_g_g",
    "Chlorophyll_B_area_g_m2",
    "Cellulose_mass_g_g",
    "Cellulose_area_g_m2",
    "Hemicellulose_mass_g_g",
    "Hemicellulose_area_g_m2",
    "Lignin_mass_g_g",
    "Lignin_area_g_m2",
    "Solubles_mass_g_g",
    "Solubles_area_g_m2",
    "Al_mass_mg_g",
    "Al_area_g_m2",
    "B2089_mass_mg_g",
    "B2498_mass_mg_g",
    "Ca_mass_mg_g",
    "Ca_area_g_m2",
    "Cu_mass_mg_g",
    "Cu_area_g_m2",
    "Fe_mass_mg_g",
    "Fe_area_g_m2",
    "K_mass_mg_g",
    "K_area_g_m2",
    "Mg_mass_mg_g",
    "Mg_area_g_m2",
    "Mn_mass_mg_g",
    "Mn_area_g_m2",
    "Na_mass_mg_g",
    "Na_area_g_m2",
    "P_mass_mg_g",
    "P_area_g_m2",
    "Zn_mass_mg_g",
    "Zn_area_g_m2",
]
NGEE_PANAMA_TRAIT_COLS = [
    "CNratio",
    "Carea_g_m2",
    "Cmass_pc",
    "H20_pc",
    "LMA_gDW_m2",
    "LWP_bar",
    "Narea_g_m2",
    "Nmass_pc",
    "SLA_cm2_gDW",
]
SIERRA_TRAIT_COLS = ["LMA_g_m2", "LWC_percent", "Chlorophyll_mg_m2"]

KNOWN_HYPERACCUMULATOR_GENERA = {
    "Noccaea","Thlaspi","Arabidopsis","Alyssum","Brassica",
    "Pteris","Sedum","Mesembryanthemum","Haumaniastrum",
    "Viola","Minuartia","Silene","Astragalus",
}
DATASET_TRAIT_MAP = {
    "lopex93": LOPEX93_TRAIT_COLS,
    "cabo_v2": CABO_V2_TRAIT_COLS,
    "ngee_panama": NGEE_PANAMA_TRAIT_COLS,
    "sierranevada2023": SIERRA_TRAIT_COLS,
}

ECOSIS_API = "https://ecosis.org/api/package/{id}/export?format=csv"
DATASETS = {
    "aqgrad":           [AQGRAD_PACKAGE_ID],
    "talladega2018":    [TALL2018_PACKAGE_ID],
    "cabo_v2":          [CABO_V2_PACKAGE_ID],
    "ngee_panama":      [NGEE_PANAMA_PACKAGE_ID],
    "lopex93":          LOPEX93_PACKAGE_IDS,
    "sierranevada2023": [SIERRA_NEVADA_2023_PACKAGE_ID],
}
LABEL_MAP = {
    "aqgrad": 0,
    "talladega2018": 1,
    "cabo_v2": 2,
    "ngee_panama": 3,
    "lopex93": 4,
    "sierranevada2023": 5,
}


def clear_stage_outputs(out_dir):
    """Remove only this stage's previously generated artifacts."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in out_dir.iterdir():
        if path.is_file() or path.is_symlink():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)


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
    requested = os.getenv("LEAF2TRAITS_DEVICE", "").strip().lower()
    if requested in {"cuda", "gpu"}:
        if torch.cuda.is_available():
            return torch.device("cuda"), "cuda"
        print("  WARN: LEAF2TRAITS_DEVICE=cuda requested but CUDA is unavailable.")
    elif requested == "directml":
        if torch_directml is not None:
            return torch_directml.device(), "directml"
        print("  WARN: LEAF2TRAITS_DEVICE=directml requested but torch-directml is unavailable.")
    elif requested == "cpu":
        return torch.device("cpu"), "cpu"
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
        if name == "cabo_v2":
            df = load_cabo_v2()
            if not df.empty:
                df["_dataset"] = name
                print(f"    OK: {df.shape}")
                frames.append(df)
                continue
            print(f"    WARN: download failed -> synthetic")
            frames.append(make_synthetic(name))
            continue
        if name == "ngee_panama":
            df = load_ngee_panama()
            if not df.empty:
                df["_dataset"] = name
                print(f"    OK: {df.shape}")
                frames.append(df)
                continue
            print(f"    WARN: download failed -> synthetic")
            frames.append(make_synthetic(name))
            continue
        if name == "lopex93":
            df = load_lopex93()
            if not df.empty:
                df["_dataset"] = name
                print(f"    OK: {df.shape}")
                frames.append(df)
                continue
            print(f"    WARN: download failed -> synthetic")
            frames.append(make_synthetic(name))
            continue
        if name == "sierranevada2023":
            df = load_sierra_nevada_2023()
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
                df["_dataset"] = name
                print(f"    OK: {df.shape}")
                frames.append(df); break
        else:
            print(f"    WARN: download failed -> synthetic")
            frames.append(make_synthetic(name))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def make_synthetic(name, n=120):
    rng = np.random.default_rng(abs(hash(name)) % (2**31))
    wl  = np.arange(400, 2401, 10)
    base = 0.05 + 0.30 * np.exp(-((wl-800)**2)/(2*200**2))
    spec = np.clip(base + rng.normal(0, 0.025, (n, len(wl))), 0, 1)
    meta = {"_dataset": name, "Species": [f"{name}_sp{j%6}" for j in range(n)]}
    if name == "sierranevada2023":
        meta.update({"LMA": rng.uniform(50,300,n), "Nmass": rng.uniform(1,4,n)})
    return pd.DataFrame({**meta, **{str(w): spec[:,i] for i,w in enumerate(wl)}})


def preprocess_dataset(df):
    wl_cols = []
    for c in sorted([c for c in df.columns if is_wl(c)], key=float):
        col = pd.to_numeric(df[c], errors="coerce")
        if col.notna().mean() >= 0.5:
            wl_cols.append(c)
    if not wl_cols:
        return np.zeros((0, len(TARGET_WL)), dtype=np.float32), df.iloc[0:0].copy()
    wl = np.array([float(c) for c in wl_cols])
    mat = df[wl_cols].values.astype(np.float64)
    bad = np.isnan(mat).mean(1) > 0.5
    mat = mat[~bad]; dfok = df.loc[df.index[~bad]].reset_index(drop=True)
    if mat.shape[0] == 0:
        return np.zeros((0, len(TARGET_WL)), dtype=np.float32), dfok
    mat = np.nan_to_num(mat, nan=0.0)
    n_bands = mat.shape[1]
    if n_bands >= 5:
        window = min(11, n_bands if n_bands % 2 == 1 else n_bands - 1)
        polyorder = min(3, window - 1)
        if window >= 3 and polyorder >= 1:
            try:
                mat = savgol_filter(mat, window_length=window, polyorder=polyorder, axis=1)
            except ValueError as exc:
                dataset = (
                    str(dfok["_dataset"].iloc[0])
                    if "_dataset" in dfok.columns and len(dfok)
                    else "unknown"
                )
                print(
                    f"    WARN: Savitzky-Golay smoothing skipped for {dataset} "
                    f"(bands={n_bands}): {exc}"
                )
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


def preprocess_all(df):
    if "_dataset" not in df.columns:
        return preprocess_dataset(df)

    all_X = []
    all_meta = []
    for dataset in df["_dataset"].dropna().unique():
        df_part = df.loc[df["_dataset"].eq(dataset)].reset_index(drop=True)
        Xp, meta_p = preprocess_dataset(df_part)
        if Xp.shape[0] == 0:
            continue
        all_X.append(Xp)
        meta_cols = [c for c in meta_p.columns if not is_wl(c)]
        all_meta.append(meta_p[meta_cols])

    if not all_X:
        return np.zeros((0, len(TARGET_WL)), dtype=np.float32), pd.DataFrame()
    return np.vstack(all_X).astype(np.float32), pd.concat(all_meta, ignore_index=True)


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


def choose_group_column(meta_df, prefer_species=False):
    if prefer_species:
        candidates = (
            "Plot_ID",
            "Species",
            "USDA_Species_Code",
            "sample_number",
            "Sample_#",
            "sample_id",
            "Sample_ID",
        )
    else:
        candidates = (
            "Plot_ID",
            "sample_number",
            "Sample_#",
            "sample_id",
            "Sample_ID",
            "Species",
            "USDA_Species_Code",
        )
    for col in candidates:
        if col in meta_df.columns and meta_df[col].notna().sum() >= 15:
            return col
    return None


def choose_entity_label(meta_df):
    for col in ("Species", "Plot_ID", "USDA_Species_Code", "sample_number", "Sample_ID", "sample_id"):
        if col in meta_df.columns and meta_df[col].notna().sum() > 0:
            return col
    return None


def build_entity_series(meta_df):
    entity = pd.Series(["Unknown"] * len(meta_df), index=meta_df.index, dtype=object)
    for col in ("Species", "Plot_ID", "USDA_Species_Code", "sample_number", "Sample_ID", "sample_id"):
        if col not in meta_df.columns:
            continue
        vals = meta_df[col]
        mask = entity.eq("Unknown") & vals.notna() & vals.astype(str).str.strip().ne("")
        entity.loc[mask] = vals.loc[mask].astype(str).str.strip()
    return entity


def build_cv(y, groups=None):
    if groups is not None:
        n_groups = pd.Series(groups).nunique()
        if n_groups >= 3:
            return GroupKFold(n_splits=min(5, n_groups)), groups
    cv_splits = min(5, max(3, len(y) // 10))
    return KFold(n_splits=cv_splits, shuffle=True, random_state=SEED), None


def aggregate_sample_level(X_raw, Z, meta_df, sample_col="Sample_#"):
    if sample_col not in meta_df.columns:
        return X_raw, Z, meta_df.reset_index(drop=True), False
    sample_ids = pd.to_numeric(meta_df[sample_col], errors="coerce")
    keep = sample_ids.notna().to_numpy()
    if keep.sum() < 10:
        return X_raw, Z, meta_df.reset_index(drop=True), False

    X_raw = X_raw[keep]
    Z = Z[keep]
    meta = meta_df.loc[keep].reset_index(drop=True).copy()
    meta["__sample_group"] = sample_ids.loc[keep].astype(int).to_numpy()
    grouped = meta.groupby("__sample_group", sort=True)
    sample_keys = list(grouped.indices.keys())
    X_raw_agg = np.vstack(
        [X_raw[np.asarray(grouped.indices[key], dtype=int)].mean(axis=0) for key in sample_keys]
    ).astype(np.float32)
    Z_agg = np.vstack(
        [Z[np.asarray(grouped.indices[key], dtype=int)].mean(axis=0) for key in sample_keys]
    ).astype(np.float32)
    meta_agg = grouped.first().reset_index()
    meta_agg[sample_col] = np.asarray(sample_keys, dtype=int)
    meta_agg = meta_agg.drop(columns=["__sample_group"], errors="ignore")
    return X_raw_agg, Z_agg, meta_agg, True


def select_best_regressor_cv(X, y, groups=None):
    n_samples, n_features = X.shape
    cv, cv_groups = build_cv(y, groups)
    best_name, best_score = None, -np.inf
    for name, model in build_regressor_candidates(n_samples, n_features).items():
        score = cross_val_score(
            clone(model), X, y, cv=cv, groups=cv_groups, scoring="r2"
        ).mean()
        if score > best_score:
            best_name, best_score = name, score
    return best_name, best_score


def compare_trait_regression(X_raw, Z, meta):
    print("  [B] Trait regression comparison (raw vs embedding) ...")

    # Collect trait columns: LOPEX93 + CABO + renamed Sierra Nevada traits
    known = LOPEX93_TRAIT_COLS + CABO_V2_TRAIT_COLS + SIERRA_TRAIT_COLS
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
        yr = y[ok]
        Xr = X_raw[ok]
        Xz = Z[ok]
        meta_t = meta.loc[ok].reset_index(drop=True).copy()
        aggregated = False
        if meta_t.get("_dataset", pd.Series(dtype=object)).eq("lopex93").all():
            Xr, Xz, meta_t, aggregated = aggregate_sample_level(Xr, Xz, meta_t)
            yr = meta_t[trait].values.astype(float)
        group_col = choose_group_column(meta_t, prefer_species=aggregated)
        groups = None
        if group_col is not None:
            group_series = meta_t[group_col].copy()
            group_series = group_series.where(group_series.notna(), "__missing_group__")
            groups = group_series.astype(str).values
        raw_name, r2_raw = select_best_regressor_cv(Xr, yr, groups)
        emb_name, r2_emb = select_best_regressor_cv(Xz, yr, groups)
        rows.append(
            {
                "trait": trait,
                "R2_raw": r2_raw,
                "R2_emb": r2_emb,
                "model_raw": raw_name,
                "model_emb": emb_name,
                "aggregation": "sample_mean" if aggregated else "none",
                "group_col": group_col or "",
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
def compare_trait_regression_v2(X_raw, Z, meta):
    print("  [B] Trait regression comparison (raw vs embedding) ...")
    if "_dataset" not in meta.columns:
        print("    No dataset labels found. Skipping.")
        return

    rows = []
    for dataset, trait_cols in DATASET_TRAIT_MAP.items():
        ds_mask = meta["_dataset"].eq(dataset).to_numpy()
        if ds_mask.sum() == 0:
            continue
        meta_ds = meta.loc[ds_mask].reset_index(drop=True).copy()
        Xr_ds = X_raw[ds_mask]
        Xz_ds = Z[ds_mask]
        traits = [c for c in trait_cols if c in meta_ds.columns and meta_ds[c].notna().sum() >= 10]
        if not traits:
            continue
        print(f"    Dataset={dataset}  traits={len(traits)}")
        for trait in traits:
            y = meta_ds[trait].values.astype(float)
            ok = ~np.isnan(y)
            if ok.sum() < 15:
                continue
            yr = y[ok]
            Xr = Xr_ds[ok]
            Xz = Xz_ds[ok]
            meta_t = meta_ds.loc[ok].reset_index(drop=True).copy()
            aggregated = False
            if dataset == "lopex93":
                Xr, Xz, meta_t, aggregated = aggregate_sample_level(Xr, Xz, meta_t)
                yr = meta_t[trait].values.astype(float)
            group_col = choose_group_column(meta_t, prefer_species=aggregated)
            groups = None
            if group_col is not None:
                group_series = meta_t[group_col].copy()
                group_series = group_series.where(group_series.notna(), "__missing_group__")
                candidate_groups = group_series.astype(str).values
                if pd.Series(candidate_groups).nunique() >= 2:
                    groups = candidate_groups
            raw_name, r2_raw = select_best_regressor_cv(Xr, yr, groups)
            emb_name, r2_emb = select_best_regressor_cv(Xz, yr, groups)
            rows.append(
                {
                    "dataset": dataset,
                    "trait": trait,
                    "R2_raw": r2_raw,
                    "R2_emb": r2_emb,
                    "model_raw": raw_name,
                    "model_emb": emb_name,
                    "aggregation": "sample_mean" if aggregated else "none",
                    "group_col": group_col or "",
                }
            )
            print(
                f"      {trait:<38}  raw={raw_name}:{r2_raw:.3f}  "
                f"emb={emb_name}:{r2_emb:.3f}  gain={r2_emb-r2_raw:+.3f}"
            )

    if not rows:
        print("    No trait rows available after dataset-specific filtering.")
        return

    df_r = pd.DataFrame(rows)
    df_r["label"] = df_r["dataset"] + ":" + df_r["trait"]
    df_r.to_csv(OUT_DIR / "trait_comparison_summary.csv", index=False)

    x = np.arange(len(df_r))
    w = 0.35
    fig, ax = plt.subplots(figsize=(max(10, 0.45 * len(df_r)), 6))
    ax.bar(x - w / 2, df_r["R2_raw"], w, label="Raw Spectra", color="#94a3b8")
    ax.bar(x + w / 2, df_r["R2_emb"], w, label="Embeddings", color="#2563eb")
    ax.set_xticks(x)
    ax.set_xticklabels(df_r["label"], rotation=55, ha="right", fontsize=8)
    ax.set_ylabel("R2 (dataset-specific CV)")
    ax.set_title("Trait Prediction by Dataset: Raw vs Embedding")
    ax.axhline(0, color="black", lw=0.8)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "trait_comparison.png", dpi=150)
    plt.close(fig)
    print("  Saved: trait_comparison.png")
    print("  Saved: trait_comparison_summary.csv")


def normalize_zero_one(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values
    vmin = np.nanmin(values)
    vmax = np.nanmax(values)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or abs(vmax - vmin) < 1e-12:
        return np.zeros_like(values, dtype=float)
    return (values - vmin) / (vmax - vmin)


def run_discovery_v2(X_raw, Z, meta):
    print("  [C] Novel candidate discovery ...")
    ds_col = meta.get("_dataset", pd.Series(["unknown"] * len(meta))).fillna("unknown")
    novelty_raw = np.zeros(len(meta), dtype=float)
    novelty_norm = np.zeros(len(meta), dtype=float)

    global_iso = IsolationForest(
        n_estimators=200, contamination=0.1, random_state=SEED, n_jobs=-1
    ).fit(Z)
    global_scores = -global_iso.score_samples(Z)

    for dataset, idx in ds_col.groupby(ds_col).groups.items():
        idx = np.asarray(list(idx), dtype=int)
        if len(idx) < 10:
            novelty_raw[idx] = global_scores[idx]
            novelty_norm[idx] = normalize_zero_one(global_scores[idx])
            continue
        iso = IsolationForest(
            n_estimators=200, contamination=0.1, random_state=SEED, n_jobs=-1
        )
        scores = -iso.fit(Z[idx]).score_samples(Z[idx])
        novelty_raw[idx] = scores
        novelty_norm[idx] = normalize_zero_one(scores)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(novelty_norm, bins=50, color="#e74c3c", alpha=0.75, edgecolor="white")
    ax.axvline(
        np.percentile(novelty_norm, 90), color="black", ls="--", lw=1.5, label="90th pct"
    )
    ax.set_xlabel("Within-dataset normalized novelty score")
    ax.set_ylabel("Count")
    ax.set_title("Novelty Score Distribution (combined normalized by dataset)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "novelty_distribution.png", dpi=150)
    plt.close(fig)
    print("  Saved: novelty_distribution.png")

    entity_col = choose_entity_label(meta)
    entity_label = entity_col or "Label"
    sp_col = build_entity_series(meta)
    genera = sp_col.apply(genus_from_species)
    if entity_col != "Species":
        genera = pd.Series(["Unknown"] * len(sp_col), index=sp_col.index)
    known_hyper = genera.apply(lambda g: g in KNOWN_HYPERACCUMULATOR_GENERA).values

    all_spectra = pd.DataFrame(
        {
            "species": sp_col.values,
            "genus": genera.values,
            "dataset": ds_col.values,
            "novelty_raw": novelty_raw,
            "novelty_score": novelty_norm,
            "known_hyperacc": known_hyper,
        }
    )
    all_spectra["prior_boost"] = all_spectra["known_hyperacc"].astype(float) * 0.15
    all_spectra["dataset_novelty_percentile"] = (
        all_spectra.groupby("dataset")["novelty_score"].rank(pct=True)
    )
    species_q95 = all_spectra.groupby("species")["novelty_score"].quantile(0.95)
    all_spectra["species_q95_novelty"] = all_spectra["species"].map(species_q95).astype(float)
    all_spectra["composite_score"] = all_spectra["novelty_score"] + all_spectra["prior_boost"]
    all_spectra["species_q95_composite"] = (
        all_spectra["species_q95_novelty"] + all_spectra["prior_boost"]
    )
    all_spectra.to_csv(OUT_DIR / "all_spectra_novelty.csv", index=False)

    representative = (
        all_spectra.sort_values("novelty_score", ascending=False)
        .groupby("species", sort=False)
        .first()[["genus", "dataset", "known_hyperacc", "prior_boost"]]
    )
    species_ranked = pd.DataFrame(
        {
            "species": species_q95.index,
            "genus": representative["genus"].reindex(species_q95.index).values,
            "dataset": representative["dataset"].reindex(species_q95.index).values,
            "known_hyperacc": representative["known_hyperacc"].reindex(species_q95.index).values,
            "prior_boost": representative["prior_boost"].reindex(species_q95.index).values,
            "novelty_score": species_q95.values,
        }
    )
    species_ranked["composite_score"] = species_ranked["novelty_score"] + species_ranked["prior_boost"]
    species_ranked["n_spectra"] = (
        all_spectra.groupby("species")["novelty_score"]
        .count()
        .reindex(species_ranked["species"])
        .values
    )
    species_ranked = species_ranked.sort_values(
        ["composite_score", "novelty_score"], ascending=False
    ).reset_index(drop=True)
    species_ranked["rank"] = species_ranked.index + 1
    species_ranked.to_csv(OUT_DIR / "ranked_candidates.csv", index=False)
    print(f"  Saved: ranked_candidates.csv  ({len(species_ranked)} unique species)")
    print(f"  Saved: all_spectra_novelty.csv  ({len(all_spectra)} individual spectra)")

    fig, ax = plt.subplots(figsize=(max(10, len(species_ranked) * 0.4), 5))
    colours = species_ranked["known_hyperacc"].map({True: "#e74c3c", False: "#3498db"})
    ax.bar(
        range(len(species_ranked)),
        species_ranked["composite_score"],
        color=colours,
        edgecolor="white",
        linewidth=0.4,
    )
    ax.axhline(
        np.percentile(species_ranked["composite_score"], 75),
        color="black",
        ls="--",
        lw=1.2,
        label="75th percentile",
    )
    ax.set_xticks(range(len(species_ranked)))
    ax.set_xticklabels(species_ranked["species"], rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("Composite Novelty Score (95th percentile per label)")
    ax.set_title("Novelty Ranking (within-dataset normalized)")
    from matplotlib.patches import Patch
    ax.legend(
        handles=[
            Patch(color="#e74c3c", label="Known hyperaccumulator genus"),
            Patch(color="#3498db", label="Other species"),
            plt.Line2D([0], [0], color="black", ls="--", label="75th percentile"),
        ]
    )
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "novelty_species_ranking.png", dpi=150)
    plt.close(fig)
    print("  Saved: novelty_species_ranking.png")

    print(f"\n  Top-10 candidate labels (95th percentile novelty):")
    print(f"  {'Rank':>4}  {entity_label:<30} {'Genus':<15} {'Dataset':<16} {'Novelty':>8}  {'N scans':>7}  {'HyperAcc?':>9}")
    print("  " + "-" * 95)
    for _, row in species_ranked.head(10).iterrows():
        ha = "YES" if row["known_hyperacc"] else "-"
        print(
            f"  {int(row['rank']):>4}  {str(row['species']):<30} "
            f"{str(row['genus']):<15} {str(row['dataset']):<16} "
            f"{row['novelty_score']:>8.4f}  {int(row['n_spectra']):>7}  {ha:>9}"
        )


def main():
    print("\n" + "="*65)
    print("  Stage 3 - Integrated Discovery Pipeline")
    print("="*65)
    clear_stage_outputs(OUT_DIR)
    print(f"  Cleared previous Stage 3 outputs in: {OUT_DIR.resolve()}")

    print("\n[1/5] Loading encoder ...")
    model, mu, std = load_encoder()

    print("\n[2/5] Loading datasets ...")
    df = load_all()

    print("\n[3/5] Preprocessing ...")
    X_raw, meta = preprocess_all(df)
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
    compare_trait_regression_v2(X_raw, Z, meta)
    run_discovery_v2(X_raw, Z, meta)

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
