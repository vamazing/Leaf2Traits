"""
step2_transfer_learning.py
Stage 2: Cross-Domain Transfer Learning

For every ordered pair of the four datasets, trains a binary classifier to
distinguish source from target dataset spectra, using both raw spectra and
autoencoder embeddings. Accuracy > 0.5 means datasets are spectrally separable.

Datasets:
  - AQGRAD              (water-affinity gradient, Mantua lakes, Italy)
  - Talladega 2018      (Talladega National Forest, Alabama)
  - LOPEX93             (European lab)
  - Sierra Nevada 2023  (field, California)

Run:
    python step2_transfer_learning.py
"""

import io, json, shutil, urllib.request, warnings
from itertools import permutations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
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
OUT_DIR    = Path("outputs/stage2")
OUT_DIR.mkdir(parents=True, exist_ok=True)
SEED = 42; np.random.seed(SEED)
TARGET_WL = np.arange(400, 2401, 10)
MIN_SAMPLES = 5
DEFAULT_AE_CONFIG = {"hidden": [1024, 512, 256, 128], "latent_dim": 192, "dropout": 0.05}

ECOSIS_API = "https://ecosis.org/api/package/{id}/export?format=csv"
DATASETS = {
    "aqgrad":           [AQGRAD_PACKAGE_ID],
    "talladega2018":    [TALL2018_PACKAGE_ID],
    "lopex93":          ["leaf-optical-properties-experiment-database--lopex93-", "3849_2"],
    "sierranevada2023": ["fresh-leaf-spectra-and-measured-traits-from-the-sierra-nevada--ca--in-july-2023"],
}
SHORT = {"aqgrad": "aqgr", "talladega2018": "tall",
         "lopex93": "lope", "sierranevada2023": "sier"}


def clear_stage_outputs(out_dir):
    """Remove only this stage's previously generated artifacts."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in out_dir.iterdir():
        if path.is_file() or path.is_symlink():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)


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
    print(f"  Loaded encoder from {pt} on {device_name}")
    return m, np.load(mn), np.load(ms)


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


def load_dataset(name, ids):
    print(f"  {name} ...")
    if name == "aqgrad":
        df = load_aqgrad()
        if not df.empty:
            df["_dataset"] = name
            print(f"    OK: {df.shape}")
            return df
        print(f"    WARN: failed -> synthetic")
        return make_synthetic(name)
    if name == "talladega2018":
        df = load_tall2018()
        if not df.empty:
            df["_dataset"] = name
            print(f"    OK: {df.shape}")
            return df
        print(f"    WARN: failed -> synthetic")
        return make_synthetic(name)
    for did in ids:
        df = fetch_csv(did)
        if not df.empty:
            df["_dataset"] = name; print(f"    OK: {df.shape}"); return df
    print(f"    WARN: failed -> synthetic")
    return make_synthetic(name)


def make_synthetic(name, n=120):
    rng = np.random.default_rng(abs(hash(name)) % (2**31))
    wl  = np.arange(400, 2401, 10)
    base = 0.05 + 0.30 * np.exp(-((wl-800)**2)/(2*200**2))
    spec = np.clip(base + rng.normal(0, 0.025, (n, len(wl))), 0, 1)
    return pd.DataFrame({"_dataset": name, **{str(w): spec[:, i] for i, w in enumerate(wl)}})


def preprocess(df):
    wl_cols = sorted([c for c in df.columns if is_wl(c)], key=float)
    wl = np.array([float(c) for c in wl_cols])
    mat = df[wl_cols].values.astype(np.float32)
    bad = np.isnan(mat).mean(1) > 0.5
    mat = mat[~bad]; df = df.loc[df.index[~bad]].reset_index(drop=True)
    mat = np.nan_to_num(mat, nan=0.0)
    mat = savgol_filter(mat, window_length=11, polyorder=3, axis=1)
    mu  = mat.mean(1, keepdims=True); sd = mat.std(1, keepdims=True) + 1e-8
    mat = (mat - mu) / sd
    out = np.zeros((mat.shape[0], len(TARGET_WL)), dtype=np.float32)
    for i, row in enumerate(mat): out[i] = np.interp(TARGET_WL, wl, row)
    return out, df


def embed(X, model, mu, std):
    if model is None: return X
    device = next(model.parameters()).device
    Xn = (X - mu) / std
    with torch.no_grad(): z = model.encode(torch.tensor(Xn, dtype=torch.float32).to(device))
    return z.cpu().numpy()


def run_transfer(src_X, tgt_X, model, mu, std):
    if len(src_X) < MIN_SAMPLES or len(tgt_X) < MIN_SAMPLES: return None, None
    rng = np.random.default_rng(SEED)
    src_y = np.zeros(len(src_X), dtype=int); tgt_y = np.ones(len(tgt_X), dtype=int)
    X_all = np.vstack([src_X, tgt_X]); y_all = np.concatenate([src_y, tgt_y])
    n_src = len(src_X); n_tgt = len(tgt_X)
    si = rng.permutation(n_src); ti = rng.permutation(n_tgt) + n_src
    str_, ste = si[:int(.8*n_src)], si[int(.8*n_src):]
    ttr_, tte = ti[:int(.8*n_tgt)], ti[int(.8*n_tgt):]
    tri = np.concatenate([str_, ttr_]); tei = np.concatenate([ste, tte])
    clf_raw = LogisticRegression(max_iter=500, random_state=SEED, C=1.0)
    clf_raw.fit(X_all[tri], y_all[tri])
    raw_acc = accuracy_score(y_all[tei], clf_raw.predict(X_all[tei]))
    X_emb = embed(X_all, model, mu, std)
    clf_emb = LogisticRegression(max_iter=500, random_state=SEED, C=1.0)
    clf_emb.fit(X_emb[tri], y_all[tri])
    emb_acc = accuracy_score(y_all[tei], clf_emb.predict(X_emb[tei]))
    return raw_acc, emb_acc


def plot_comparison(results):
    dirs = [k for k, v in results.items() if v[0] is not None]
    if not dirs: return
    raw_vals = [results[d][0] for d in dirs]
    emb_vals = [results[d][1] for d in dirs]
    x = np.arange(len(dirs)); w = 0.35
    fig, ax = plt.subplots(figsize=(max(9, 1.8*len(dirs)), 5))
    bars1 = ax.bar(x - w/2, raw_vals, w, label="Raw Spectra",  color="#94a3b8")
    bars2 = ax.bar(x + w/2, emb_vals, w, label="Embeddings",   color="#2563eb")
    ax.axhline(0.5, color="red", ls="--", lw=1.2, label="Chance")
    ax.set_ylabel("Dataset Discriminability Accuracy (chance = 0.5)")
    ax.set_title("Cross-Domain Transfer: Raw vs Embedding (4 datasets)")
    ax.set_xticks(x); ax.set_xticklabels(dirs, rotation=35, ha="right")
    ax.set_ylim(0, 1.05); ax.legend(); ax.grid(axis="y", alpha=0.3)
    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT_DIR/"baseline_comparison.png", dpi=150); plt.close(fig)
    print("  Saved: baseline_comparison.png")


def main():
    print("\n" + "="*65)
    print("  Stage 2 - Cross-Domain Transfer Learning")
    print("="*65)
    clear_stage_outputs(OUT_DIR)
    print(f"  Cleared previous Stage 2 outputs in: {OUT_DIR.resolve()}")

    print("\n[1/4] Loading encoder ...")
    model, mu, std = load_encoder()

    print("\n[2/4] Loading datasets ...")
    proc_data = {}
    for name, ids in DATASETS.items():
        df = load_dataset(name, ids)
        Xp, dfp = preprocess(df)
        if name == "lopex93" and Xp.shape[0] > 0:
            Xp, _ = filter_lopex_spectral_outliers(Xp, dfp)
        proc_data[name] = Xp

    print("\n[3/4] Running transfer experiments ...")
    print("  (Binary task: can a model distinguish dataset A from dataset B?)\n")
    results = {}
    for src, tgt in permutations(DATASETS.keys(), 2):
        direction = f"{SHORT[src]}→{SHORT[tgt]}"
        raw_acc, emb_acc = run_transfer(proc_data[src], proc_data[tgt], model, mu, std)
        results[direction] = (raw_acc, emb_acc)
        if raw_acc is not None:
            print(f"  {direction}:  raw={raw_acc:.3f}  emb={emb_acc:.3f}  gain={emb_acc-raw_acc:+.3f}")
        else:
            print(f"  SKIP {direction}: too few samples")

    print("\n[4/4] Plotting ...")
    plot_comparison(results)
    pd.DataFrame([(d, v[0], v[1]) for d, v in results.items()],
                 columns=["direction","raw_accuracy","emb_accuracy"]
                 ).to_csv(OUT_DIR/"transfer_summary.csv", index=False)
    print("  Saved: transfer_summary.csv")

    print(f"\n{'='*65}")
    print(f"  Stage 2 complete. Outputs in: {OUT_DIR.resolve()}")
    print("="*65 + "\n")


if __name__ == "__main__":
    main()
