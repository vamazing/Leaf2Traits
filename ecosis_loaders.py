"""
EcoSIS helpers for Leaf2Traits.
"""

import io
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

ECOSIS_API_EXPORT = "https://ecosis.org/api/package/{id}/export?format=csv"
AQGRAD_PACKAGE_ID = (
    "leaf-reflectance-and-traits-of-plants-sampled-along-a-water-affinity-gradient--aqgrad-"
)
TALL2018_PACKAGE_ID = (
    "2018-talladega-national-forest--leaf-level-reflectance-spectra-and-foliar-traits"
)
TALL2018_METADATA_CSV = "tall_2018_ecosis_spectra_metadata_v2.csv"
LOPEX93_METADATA = "lopex1993metadata.xlsx"
# Fraction of LOPEX spectra removed as embedding-space outliers (IsolationForest on preprocessed spectra).
LOPEX_OUTLIER_CONTAMINATION = 0.03


def fetch_package_csv(package_id, metadata=False, timeout=120):
    """Download a package CSV from the EcoSIS export API."""
    url = ECOSIS_API_EXPORT.format(id=package_id)
    if metadata:
        url += "&metadata=true"
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


def attach_species_column(df):
    """Build a unified Species column from EcoSIS metadata fields."""
    if "Latin Genus" in df.columns and "Latin Species" in df.columns:
        genus = df["Latin Genus"].astype(str).str.strip()
        species = df["Latin Species"].astype(str).str.strip().str.replace("_", " ")
        df["Species"] = (genus + " " + species).str.strip()
    elif "Common Name" in df.columns:
        df["Species"] = df["Common Name"].astype(str).str.strip()
    return df


def load_aqgrad(timeout=120):
    """
    Load AQGRAD spectra and species names via EcoSIS export API (metadata=true).

    The default CSV export is spectra-only; metadata=true adds Latin Genus,
    Latin Species, Common Name, and measured traits per spectrum.
    """
    df = fetch_package_csv(AQGRAD_PACKAGE_ID, metadata=True, timeout=timeout)
    if df.empty:
        return df
    return attach_species_column(df)


def _tall2018_metadata_path():
    root = Path(__file__).resolve().parent
    for path in (root / TALL2018_METADATA_CSV, Path.home() / "Downloads" / TALL2018_METADATA_CSV):
        if path.exists():
            return path
    return None


def _join_tall2018_metadata(spectra_df, meta_df):
    """Merge spectra rows with uploaded metadata on Sample_ID."""
    key = "Sample_ID"
    if key not in spectra_df.columns or key not in meta_df.columns:
        return spectra_df
    meta = meta_df.drop_duplicates(subset=[key])
    merged = spectra_df.merge(meta, on=key, how="left", suffixes=("", "_meta"))
    if "Genus" in merged.columns and "Species" in merged.columns and "Latin Genus" not in merged.columns:
        genus = merged["Genus"].astype(str).str.strip()
        species = merged["Species"].astype(str).str.strip()
        merged["Latin Genus"] = genus
        merged["Latin Species"] = species
    return merged


def load_tall2018(timeout=120):
    """
    Load Talladega 2018 spectra and metadata via EcoSIS (metadata=true).

    Falls back to spectra-only API export joined with the uploaded metadata CSV
    if the metadata export is unavailable.
    """
    df = fetch_package_csv(TALL2018_PACKAGE_ID, metadata=True, timeout=timeout)
    if df.empty:
        spectra = fetch_package_csv(TALL2018_PACKAGE_ID, metadata=False, timeout=timeout)
        meta_path = _tall2018_metadata_path()
        if spectra.empty or meta_path is None:
            return pd.DataFrame()
        meta = pd.read_csv(meta_path)
        df = _join_tall2018_metadata(spectra, meta)
    if df.empty:
        return df
    if "Leaf mass per area" in df.columns and "LMA" not in df.columns:
        df = df.rename(columns={"Leaf mass per area": "LMA"})
    if "Leaf nitrogen content per leaf area" in df.columns and "N" not in df.columns:
        df = df.rename(columns={"Leaf nitrogen content per leaf area": "N"})
    return attach_species_column(df)


def join_lopex93_metadata(df, metadata_file=LOPEX93_METADATA):
    """Attach LOPEX93 trait metadata (two spectra per sample) to an EcoSIS spectra frame."""
    p = Path(metadata_file)
    if not p.exists() or df.empty:
        return df
    meta = pd.read_excel(p).replace(-999, np.nan)
    sample_ids = np.arange(len(df)) // 2 + 1
    joined = pd.DataFrame({"Sample_#": sample_ids}).merge(meta, on="Sample_#", how="left")
    for col in joined.columns:
        if col != "Sample_#":
            df[col] = joined[col].values
    if "Latin Name" in df.columns and "Species" not in df.columns:
        df = df.rename(columns={"Latin Name": "Species"})
    return df


def filter_lopex_spectral_outliers(
    X,
    df,
    contamination=LOPEX_OUTLIER_CONTAMINATION,
    random_state=42,
):
    """
    Drop LOPEX93 spectra far from the main cluster in preprocessed space.

    IsolationForest on Savitzky–Golay + SNV + grid-aligned spectra; applied before
    autoencoder training so embedding PCA is not pulled by bad LOPEX scans.
    """
    if X.shape[0] < 10:
        return X, df
    iso = IsolationForest(contamination=contamination, random_state=random_state)
    inlier = iso.fit_predict(X) == 1
    n_out = int((~inlier).sum())
    if n_out:
        print(
            f"    LOPEX93: removed {n_out} spectral outliers "
            f"(IsolationForest, contamination={contamination})"
        )
    return X[inlier], df.loc[inlier].reset_index(drop=True)
