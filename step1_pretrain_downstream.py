"""
step1_pretrain_downstream.py
Stage 1: Pretraining + Downstream Prediction

Datasets:
  - AQGRAD               (EcoSIS, water-affinity gradient, Mantua lakes, Italy)
  - Talladega 2018       (EcoSIS, leaf spectra + foliar traits, Alabama)
  - CABO v2              (EcoSIS, broad multispecies leaf-level traits)
  - Droughted crops      (EcoSIS, crop leaf physiology + biochemistry under watering treatments)
  - LOPEX93              (EcoSIS spectra + biochemical traits via metadata=true)
  - Sierra Nevada 2023   (EcoSIS, field, California - includes measured traits)

Pipeline:
  1. Download all five datasets.
  2. Preprocess: Savitzky-Golay smooth -> SNV normalise -> align to 400-2400nm grid.
  3. Train deep autoencoder.
  4. Extract 192-dim embeddings; visualise with PCA.
  5. Trait regression on LOPEX93, CABO, and Sierra Nevada traits.

Run:
    python step1_pretrain_downstream.py
"""

import io
import json
import shutil
import urllib.request
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
try:
    import torch_directml
except ImportError:
    torch_directml = None
from scipy.signal import savgol_filter
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import classification_report, mean_squared_error, r2_score
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    KFold,
    ShuffleSplit,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from ecosis_loaders import (
    AQGRAD_PACKAGE_ID,
    CABO_V2_PACKAGE_ID,
    DROUGHTED_CROPS_PACKAGE_ID,
    LOPEX93_PACKAGE_IDS,
    SIERRA_NEVADA_2023_PACKAGE_ID,
    TALL2018_PACKAGE_ID,
    filter_lopex_spectral_outliers,
    load_aqgrad,
    load_cabo_v2,
    load_droughted_crops,
    load_lopex93,
    load_sierra_nevada_2023,
    load_tall2018,
)

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
OUT_DIR = Path("outputs/stage1")
OUT_DIR.mkdir(parents=True, exist_ok=True)
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

TARGET_WL = np.arange(400, 2401, 10)  # 201 bands
LATENT_DIM = 192
HIDDEN = [1024, 512, 256, 128]
DROPOUT = 0.05
BATCH_SIZE = 64
N_EPOCHS = 140
LR = 7.5e-4
WEIGHT_DECAY = 1e-5
NOISE_STD = 0.03
RIDGE_ALPHAS = np.logspace(-3, 3, 13)

LOPEX93_TRAIT_COLS = [
    "Chlorophyll_a+b (µg/cm²)",
    "Equivalent Water Thickness (g/cm²)",
    "Leaf mass per area (g/cm²)",
    "Protein_1 (% DW)",
    "Cellulose_1 (% DW)",
    "Carotenoid (µg/cm²)",
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

DROUGHTED_CROPS_TRAIT_COLS = [
    "ABA",
    "Amino_Acids",
    "Asat",
    "CO2s",
    "Ci",
    "Elemental_C",
    "Elemental_N",
    "Fructose",
    "Glucose",
    "H2Os",
    "HDP_Fructan",
    "LDP_Fructan",
    "Leaf mass per area",
    "Proline",
    "Protein",
    "Qin",
    "RHs",
    "RWC",
    "Starch",
    "Sucrose",
    "Tleaf",
    "Tr",
    "VPDleaf",
    "flow",
    "gs",
]

SIERRA_TRAIT_COLS = [
    "LMA_g_m2",
    "LWC_percent",
    "Chlorophyll_mg_m2",
]

ECOSIS_API = "https://ecosis.org/api/package/{id}/export?format=csv"
DATASETS = {
    "aqgrad": [AQGRAD_PACKAGE_ID],
    "talladega2018": [TALL2018_PACKAGE_ID],
    "cabo_v2": [CABO_V2_PACKAGE_ID],
    "droughted_crops": [DROUGHTED_CROPS_PACKAGE_ID],
    "lopex93": LOPEX93_PACKAGE_IDS,
    "sierranevada2023": [SIERRA_NEVADA_2023_PACKAGE_ID],
}
LABEL_MAP = {
    "aqgrad": 0,
    "talladega2018": 1,
    "cabo_v2": 2,
    "droughted_crops": 3,
    "lopex93": 4,
    "sierranevada2023": 5,
}
LABEL_NAMES = {
    0: "AQGRAD",
    1: "Talladega 2018",
    2: "CABO v2",
    3: "Droughted Crops",
    4: "LOPEX93",
    5: "Sierra Nevada 2023",
}
AE_CONFIG = {"hidden": HIDDEN, "latent_dim": LATENT_DIM, "dropout": DROPOUT}


def clear_stage_outputs(out_dir):
    """Remove only this stage's previously generated artifacts."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in out_dir.iterdir():
        if path.is_file() or path.is_symlink():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)


# ── I/O ───────────────────────────────────────────────────────────────────────
def fetch_csv(did, timeout=120):
    url = ECOSIS_API.format(id=did)
    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 Leaf2Traits/1.0",
                "Accept": "text/csv,text/plain,*/*",
            },
        )
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
    if name == "cabo_v2":
        df = load_cabo_v2()
        if not df.empty:
            print(f"    OK {CABO_V2_PACKAGE_ID}: {df.shape}")
            df["_dataset"] = name
            return df
        print(f"    WARN: CABO v2 download failed -> synthetic fallback")
        return pd.DataFrame()
    if name == "droughted_crops":
        df = load_droughted_crops()
        if not df.empty:
            print(f"    OK {DROUGHTED_CROPS_PACKAGE_ID}: {df.shape}")
            df["_dataset"] = name
            return df
        print(f"    WARN: Droughted crops download failed -> synthetic fallback")
        return pd.DataFrame()
    if name == "lopex93":
        df = load_lopex93()
        if not df.empty:
            print(f"    OK {LOPEX93_PACKAGE_IDS[0]}: {df.shape}")
            df["_dataset"] = name
            return df
        print(f"    WARN: LOPEX93 download failed -> synthetic fallback")
        return pd.DataFrame()
    if name == "sierranevada2023":
        df = load_sierra_nevada_2023()
        if not df.empty:
            print(f"    OK {SIERRA_NEVADA_2023_PACKAGE_ID}: {df.shape}")
            df["_dataset"] = name
            return df
        print(f"    WARN: Sierra Nevada 2023 download failed -> synthetic fallback")
        return pd.DataFrame()
    for did in ids:
        df = fetch_csv(did)
        if not df.empty:
            print(f"    OK {did}: {df.shape}")
            df["_dataset"] = name
            return df
    print(f"    WARN: download failed -> synthetic fallback")
    return pd.DataFrame()

def make_synthetic(name, n=120):
    rng = np.random.default_rng(abs(hash(name)) % (2**31))
    wl = np.arange(400, 2401, 10)
    base = 0.05 + 0.30 * np.exp(-((wl - 800) ** 2) / (2 * 200**2))
    spec = np.clip(base + rng.normal(0, 0.025, (n, len(wl))), 0, 1)
    meta = {"_dataset": name, "Species": [f"{name}_sp{j % 6}" for j in range(n)]}
    if name == "lopex93":
        meta.update({c: rng.uniform(1, 50, n) for c in LOPEX93_TRAIT_COLS})
    if name == "sierranevada2023":
        meta.update(
            {
                "LMA": rng.uniform(50, 300, n),
                "Nmass": rng.uniform(1, 4, n),
                "EWT": rng.uniform(0.005, 0.04, n),
            }
        )
    return pd.DataFrame({**meta, **{str(w): spec[:, i] for i, w in enumerate(wl)}})


# ── Preprocessing ─────────────────────────────────────────────────────────────
def is_wl(col):
    try:
        v = float(col)
        return 300.0 <= v <= 2600.0
    except (ValueError, TypeError):
        return False


def preprocess_dataset(df):
    wl_cols = sorted([c for c in df.columns if is_wl(c)], key=float)
    wl = np.array([float(c) for c in wl_cols])
    mat = df[wl_cols].values.astype(np.float32)
    bad = np.isnan(mat).mean(1) > 0.5
    mat = mat[~bad]
    df = df.loc[df.index[~bad]].reset_index(drop=True)
    mat = np.nan_to_num(mat, nan=0.0)
    mat = savgol_filter(mat, window_length=11, polyorder=3, axis=1)
    mu = mat.mean(1, keepdims=True)
    sd = mat.std(1, keepdims=True) + 1e-8
    mat = (mat - mu) / sd
    out = np.zeros((mat.shape[0], len(TARGET_WL)), dtype=np.float32)
    for i, row in enumerate(mat):
        out[i] = np.interp(TARGET_WL, wl, row)
    return out, df


def preprocess_all(datasets):
    all_X, all_y, all_meta = [], [], []
    for name, df in datasets.items():
        Xp, dfp = preprocess_dataset(df)
        if name == "lopex93" and Xp.shape[0] > 0:
            Xp, dfp = filter_lopex_spectral_outliers(Xp, dfp)
        if Xp.shape[0] == 0:
            continue
        all_X.append(Xp)
        all_y.extend([LABEL_MAP.get(name, -1)] * Xp.shape[0])
        mc = [c for c in dfp.columns if not is_wl(c)]
        all_meta.append(dfp[mc])
    return (
        np.vstack(all_X).astype(np.float32),
        np.array(all_y, dtype=np.int64),
        pd.concat(all_meta, ignore_index=True),
    )


# ── Autoencoder ───────────────────────────────────────────────────────────────
class Autoencoder(nn.Module):
    def __init__(self, inp=201, hidden=None, latent=64, dropout=0.0):
        super().__init__()
        if hidden is None:
            hidden = [256, 128]
        enc = []
        d = inp
        for h in hidden:
            enc += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.GELU()]
            if dropout > 0:
                enc.append(nn.Dropout(dropout))
            d = h
        enc.append(nn.Linear(d, latent))
        self.encoder = nn.Sequential(*enc)
        dec = []
        d = latent
        for h in reversed(hidden):
            dec += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.GELU()]
            if dropout > 0:
                dec.append(nn.Dropout(dropout))
            d = h
        dec.append(nn.Linear(d, inp))
        self.decoder = nn.Sequential(*dec)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

    def encode(self, x):
        return self.encoder(x)


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


def train_ae(X):
    device, device_name = resolve_device()
    print(f"  Device: {device_name} ({device})")
    mu = X.mean(0)
    std = X.std(0) + 1e-8
    Xn = (X - mu) / std
    dl = DataLoader(
        TensorDataset(torch.tensor(Xn, dtype=torch.float32)),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    model = Autoencoder(
        X.shape[1], HIDDEN, LATENT_DIM, dropout=DROPOUT
    ).to(device)
    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_EPOCHS)
    fn = nn.MSELoss()
    hist = []
    for ep in range(1, N_EPOCHS + 1):
        model.train()
        el = 0.0
        for (b,) in dl:
            b = b.to(device)
            noisy = b + NOISE_STD * torch.randn_like(b)
            r, _ = model(noisy)
            loss = fn(r, b)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            el += loss.item() * len(b)
        el /= len(Xn)
        hist.append(el)
        sched.step()
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
    ax.plot(range(1, len(hist) + 1), hist, lw=2, color="#2563eb")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Autoencoder Training Loss")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "autoencoder_loss.png", dpi=150)
    plt.close(fig)
    print("  Saved: autoencoder_loss.png")


def plot_pca(Z, labels):
    pca = PCA(n_components=2, random_state=SEED)
    Z2 = pca.fit_transform(Z)
    colours = ["#e74c3c", "#2ecc71", "#3498db", "#f39c12"]
    fig, ax = plt.subplots(figsize=(9, 6))
    for i, u in enumerate(np.unique(labels)):
        m = labels == u
        ax.scatter(
            Z2[m, 0],
            Z2[m, 1],
            label=LABEL_NAMES.get(u, str(u)),
            alpha=0.6,
            s=25,
            color=colours[i % len(colours)],
        )
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
    ax.set_title("PCA of Spectral Embeddings (by dataset)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "embedding_pca.png", dpi=150)
    plt.close(fig)
    print("  Saved: embedding_pca.png")


# ── Downstream ────────────────────────────────────────────────────────────────
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


def select_regressor(X, y):
    n_samples, n_features = X.shape
    cv_splits = min(5, max(3, n_samples // 10))
    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=SEED)
    best_name, best_model, best_score = None, None, -np.inf
    for name, model in build_regressor_candidates(n_samples, n_features).items():
        score = cross_val_score(model, X, y, cv=cv, scoring="r2").mean()
        if score > best_score:
            best_name, best_model, best_score = name, model, score
    return best_name, clone(best_model), best_score


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


def build_cv(y, groups=None):
    if groups is not None:
        n_groups = pd.Series(groups).nunique()
        if n_groups >= 3:
            return GroupKFold(n_splits=min(5, n_groups)), groups
    cv_splits = min(5, max(3, len(y) // 10))
    return KFold(n_splits=cv_splits, shuffle=True, random_state=SEED), None


def select_regressor_with_groups(X, y, groups=None):
    n_samples, n_features = X.shape
    cv, cv_groups = build_cv(y, groups)
    best_name, best_model, best_score = None, None, -np.inf
    for name, model in build_regressor_candidates(n_samples, n_features).items():
        score = cross_val_score(
            model, X, y, cv=cv, groups=cv_groups, scoring="r2"
        ).mean()
        if score > best_score:
            best_name, best_model, best_score = name, model, score
    return best_name, clone(best_model), best_score


def sanitize_trait_name(trait):
    return (
        trait.replace("/", "_")
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("%", "pct")
    )


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


def plot_trait_comparison(
    trait, tag, yte, raw_pred, emb_pred, raw_name, emb_name, raw_r2, emb_r2
):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
    mn = min(yte.min(), raw_pred.min(), emb_pred.min())
    mx = max(yte.max(), raw_pred.max(), emb_pred.max())
    specs = [
        (axes[0], raw_pred, f"Raw ({raw_name})  R2={raw_r2:.3f}", "#64748b"),
        (axes[1], emb_pred, f"Embedding ({emb_name})  R2={emb_r2:.3f}", "#2563eb"),
    ]
    for ax, pred, title, color in specs:
        ax.scatter(yte, pred, alpha=0.7, s=30, color=color)
        ax.plot([mn, mx], [mn, mx], "r--", lw=1.5)
        ax.set_xlabel(f"True {trait}")
        ax.set_title(title)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Predicted")
    fig.suptitle(trait)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"trait_compare_{tag}_{sanitize_trait_name(trait)}.png", dpi=150)
    plt.close(fig)


def run_trait_regression(X_raw, Z, meta_df, trait_cols, tag=""):
    if not trait_cols:
        return False
    print(f"  Traits ({tag}): {trait_cols}")
    X_eval = X_raw
    Z_eval = Z
    meta_eval = meta_df.reset_index(drop=True).copy()
    aggregated = False
    if tag == "LOPEX93":
        X_eval, Z_eval, meta_eval, aggregated = aggregate_sample_level(
            X_eval, Z_eval, meta_eval
        )
        if aggregated:
            print(
                f"    Sample-aggregated LOPEX93 regression: "
                f"{len(meta_df)} spectra -> {len(meta_eval)} samples"
            )

    group_col = choose_group_column(meta_eval, prefer_species=aggregated)
    if group_col is not None:
        print(f"    Group-aware split/CV via '{group_col}'")
    summary_rows = []
    for trait in trait_cols:
        if trait not in meta_eval.columns:
            continue
        y = meta_eval[trait].values.astype(float)
        ok = ~np.isnan(y)
        if ok.sum() < 15:
            continue
        yr = y[ok]
        Xr = X_eval[ok]
        Xz = Z_eval[ok]
        groups = None
        if group_col is not None:
            group_series = meta_eval.loc[ok, group_col].copy()
            group_series = group_series.where(group_series.notna(), "__missing_group__")
            groups = group_series.astype(str).values

        if groups is not None:
            splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
            train_idx, test_idx = next(splitter.split(Xr, yr, groups=groups))
            train_groups = groups[train_idx]
        else:
            splitter = ShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
            train_idx, test_idx = next(splitter.split(Xr, yr))
            train_groups = None

        Xr_tr, Xr_te = Xr[train_idx], Xr[test_idx]
        Xz_tr, Xz_te = Xz[train_idx], Xz[test_idx]
        ytr, yte = yr[train_idx], yr[test_idx]

        raw_name, raw_reg, raw_cv_r2 = select_regressor_with_groups(Xr_tr, ytr, train_groups)
        emb_name, emb_reg, emb_cv_r2 = select_regressor_with_groups(Xz_tr, ytr, train_groups)
        raw_reg.fit(Xr_tr, ytr)
        emb_reg.fit(Xz_tr, ytr)
        raw_pred = np.asarray(raw_reg.predict(Xr_te)).reshape(-1)
        emb_pred = np.asarray(emb_reg.predict(Xz_te)).reshape(-1)
        raw_r2 = r2_score(yte, raw_pred)
        emb_r2 = r2_score(yte, emb_pred)
        raw_rmse = mean_squared_error(yte, raw_pred) ** 0.5
        emb_rmse = mean_squared_error(yte, emb_pred) ** 0.5
        preferred = "embedding" if emb_r2 > raw_r2 else "raw"
        print(
            f"    {trait:<40}  raw={raw_name}:{raw_r2:.3f} (CV {raw_cv_r2:.3f})  "
            f"emb={emb_name}:{emb_r2:.3f} (CV {emb_cv_r2:.3f})  "
            f"gain={emb_r2-raw_r2:+.3f}  winner={preferred}"
        )
        plot_trait_comparison(
            trait, tag, yte, raw_pred, emb_pred, raw_name, emb_name, raw_r2, emb_r2
        )
        summary_rows.append(
            {
                "dataset": tag,
                "trait": trait,
                "aggregation": "sample_mean" if aggregated else "none",
                "group_col": group_col or "",
                "n_rows": int(len(yr)),
                "n_groups": int(pd.Series(groups).nunique()) if groups is not None else 0,
                "raw_model": raw_name,
                "raw_cv_r2": raw_cv_r2,
                "raw_test_r2": raw_r2,
                "raw_rmse": raw_rmse,
                "emb_model": emb_name,
                "emb_cv_r2": emb_cv_r2,
                "emb_test_r2": emb_r2,
                "emb_rmse": emb_rmse,
                "gain_emb_minus_raw": emb_r2 - raw_r2,
                "winner": preferred,
            }
        )
    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(
            OUT_DIR / f"trait_regression_summary_{tag}.csv", index=False
        )
        print(f"    Saved: trait_regression_summary_{tag}.csv")
        return True
    return False


def run_classification_fallback(Z, labels):
    print("  Dataset classification fallback:")
    Xtr, Xte, ytr, yte = train_test_split(
        Z, labels, test_size=0.25, stratify=labels, random_state=SEED
    )
    clf = LogisticRegression(max_iter=500, random_state=SEED)
    clf.fit(Xtr, ytr)
    names = [LABEL_NAMES[i] for i in sorted(LABEL_NAMES)]
    print(classification_report(yte, clf.predict(Xte), target_names=names))

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "=" * 65)
    print("  Stage 1 - Pretraining + Downstream Prediction")
    print("=" * 65)
    clear_stage_outputs(OUT_DIR)
    print(f"  Cleared previous Stage 1 outputs in: {OUT_DIR.resolve()}")

    print("\n[1/6] Loading datasets ...")
    raws = {}
    for name, ids in DATASETS.items():
        df = load_dataset(name, ids)
        if df.empty:
            df = make_synthetic(name)
            print(f"    Using synthetic {name}: {df.shape}")
        raws[name] = df

    print("\n[2/6] Preprocessing ...")
    X, labels, meta = preprocess_all(raws)
    print(f"  Pooled: {X.shape}")

    print("\n[3/6] Training autoencoder ...")
    model, hist, mu, std = train_ae(X)
    torch.save(model.state_dict(), OUT_DIR / "encoder.pt")
    with open(OUT_DIR / "ae_config.json", "w", encoding="utf-8") as f:
        json.dump(AE_CONFIG, f, indent=2)
    np.save(OUT_DIR / "scaler_mean.npy", mu)
    np.save(OUT_DIR / "scaler_std.npy", std)
    np.save(OUT_DIR / "target_wl.npy", TARGET_WL)
    print(f"  Final loss: {hist[-1]:.5f}")

    print("\n[4/6] Extracting embeddings ...")
    Z = get_embeddings(model, X, mu, std)
    np.savez(OUT_DIR / "embeddings_all.npz", Z=Z, labels=labels, X=X)
    print(f"  Embedding shape: {Z.shape}")

    print("\n[5/6] Plots ...")
    plot_loss(hist)
    plot_pca(Z, labels)

    print("\n[6/6] Downstream tasks ...")
    trait_found = False

    # LOPEX93 traits (joined at load; row-aligned with filtered spectra)
    lmask = labels == LABEL_MAP["lopex93"]
    if lmask.sum() > 0:
        lm = meta[lmask].reset_index(drop=True)
        trait_cols = [c for c in LOPEX93_TRAIT_COLS if c in lm.columns]
        if trait_cols:
            ok = run_trait_regression(X[lmask], Z[lmask], lm, trait_cols, tag="LOPEX93")
            if ok:
                trait_found = True

    # Sierra Nevada traits (inline in CSV)
    smask = labels == LABEL_MAP["sierranevada2023"]
    if smask.sum() > 0:
        sm = meta[smask].reset_index(drop=True)
        # Use renamed trait cols first, then any numeric column
        renamed = ["LMA_g_m2", "LWC_percent", "Chlorophyll_mg_m2"]
        sierra_traits = [
            c for c in renamed if c in sm.columns and sm[c].notna().sum() >= 10
        ]
        if not sierra_traits:
            sierra_traits = [
                c
                for c in sm.columns
                if pd.api.types.is_numeric_dtype(sm[c])
                and sm[c].notna().sum() >= 10
                and c not in ("_dataset",)
            ]
        if sierra_traits:
            ok = run_trait_regression(
                X[smask], Z[smask], sm, sierra_traits, tag="SierraNevada"
            )
            if ok:
                trait_found = True

    # CABO v2 traits (inline in EcoSIS export)
    cmask = labels == LABEL_MAP["cabo_v2"]
    if cmask.sum() > 0:
        cm = meta[cmask].reset_index(drop=True)
        cabo_traits = [
            c for c in CABO_V2_TRAIT_COLS if c in cm.columns and cm[c].notna().sum() >= 10
        ]
        if cabo_traits:
            ok = run_trait_regression(X[cmask], Z[cmask], cm, cabo_traits, tag="CABOv2")
            if ok:
                trait_found = True

    # Droughted crops traits (inline in EcoSIS export)
    nmask = labels == LABEL_MAP["droughted_crops"]
    if nmask.sum() > 0:
        nm = meta[nmask].reset_index(drop=True)
        crop_traits = [
            c
            for c in DROUGHTED_CROPS_TRAIT_COLS
            if c in nm.columns
            and nm[c].notna().sum() >= 10
            and pd.api.types.is_numeric_dtype(nm[c])
        ]
        if crop_traits:
            ok = run_trait_regression(
                X[nmask], Z[nmask], nm, crop_traits, tag="DroughtedCrops"
            )
            if ok:
                trait_found = True

    if not trait_found:
        print("  No trait columns found. Running dataset classification fallback.")
        run_classification_fallback(Z, labels)

    print(f"\n{'=' * 65}")
    print(f"  Stage 1 complete. Outputs in: {OUT_DIR.resolve()}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
