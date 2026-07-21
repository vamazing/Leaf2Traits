"""
EcoSIS helpers for Leaf2Traits.
"""

import io
import urllib.request
from pathlib import Path
import ssl
import certifi
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
ANGERS_2003_PACKAGE_ID = "angers-leaf-optical-properties-database--2003-"
BLACKHAWK_ISLAND_WI_PACKAGE_ID = "seasonal-fresh-leaf-spectra-and-traits--blackhawk-island--wi"
CABO_V2_PACKAGE_ID = "a12286ac-929f-4e3e-9b9c-517e10fe4087"
NGEE_PANAMA_PACKAGE_ID = (
    "ngee-tropics-leaf-spectral-reflectance-measured-in-panama-collected-february-to-april-2016"
)
LOPEX93_PACKAGE_IDS = [
    "leaf-optical-properties-experiment-database--lopex93-",
    "3849_2",
    "3849",
]
SIERRA_NEVADA_2023_PACKAGE_ID = (
    "fresh-leaf-spectra-and-measured-traits-from-the-sierra-nevada--ca--in-july-2023"
)
TALL2018_METADATA_CSV = "tall_2018_ecosis_spectra_metadata_v2.csv"
# Fraction of LOPEX spectra removed as embedding-space outliers (IsolationForest on preprocessed spectra).
LOPEX_OUTLIER_CONTAMINATION = 0.03

CABO_V2_RENAME_MAP = {
    "sample_id": "Sample_ID",
    "species": "Species",
    "genus": "Genus",
    "family": "Family",
    "functionalgroup": "Functional_Group",
    "growthform": "Growth_Form",
    "Measurement Date": "Date",
    "LMA": "LMA_g_m2",
    "EWT": "EWT_g_cm2",
    "LDMC": "LDMC_mg_g",
    "Cmass": "C_mass_g_g",
    "Nmass": "N_mass_g_g",
    "Carea": "C_area_g_m2",
    "Narea": "N_area_g_m2",
    "car_mass": "Carotenoid_mass_g_g",
    "car_area": "Carotenoid_area_g_m2",
    "chlA_mass": "Chlorophyll_A_mass_g_g",
    "chlA_area": "Chlorophyll_A_area_g_m2",
    "chlB_mass": "Chlorophyll_B_mass_g_g",
    "chlB_area": "Chlorophyll_B_area_g_m2",
    "cellulose_mass": "Cellulose_mass_g_g",
    "cellulose_area": "Cellulose_area_g_m2",
    "hemicellulose_mass": "Hemicellulose_mass_g_g",
    "hemicellulose_area": "Hemicellulose_area_g_m2",
    "lignin_mass": "Lignin_mass_g_g",
    "lignin_area": "Lignin_area_g_m2",
    "solubles_mass": "Solubles_mass_g_g",
    "solubles_area": "Solubles_area_g_m2",
    "Al_mass": "Al_mass_mg_g",
    "Al_area": "Al_area_g_m2",
    "B2089_mass": "B2089_mass_mg_g",
    "B2498_mass": "B2498_mass_mg_g",
    "Ca_mass": "Ca_mass_mg_g",
    "Ca_area": "Ca_area_g_m2",
    "Cu_mass": "Cu_mass_mg_g",
    "Cu_area": "Cu_area_g_m2",
    "Fe_mass": "Fe_mass_mg_g",
    "Fe_area": "Fe_area_g_m2",
    "K_mass": "K_mass_mg_g",
    "K_area": "K_area_g_m2",
    "Mg_mass": "Mg_mass_mg_g",
    "Mg_area": "Mg_area_g_m2",
    "Mn_mass": "Mn_mass_mg_g",
    "Mn_area": "Mn_area_g_m2",
    "Na_mass": "Na_mass_mg_g",
    "Na_area": "Na_area_g_m2",
    "P_mass": "P_mass_mg_g",
    "P_area": "P_area_g_m2",
    "Zn_mass": "Zn_mass_mg_g",
    "Zn_area": "Zn_area_g_m2",
}
SIERRA_V2_RENAME_MAP = {
    "LMA_value_g/m2": "LMA_g_m2",
    "leaf_water_content_percent": "LWC_percent",
    "chlorophyll _mg/m2": "Chlorophyll_mg_m2",
    "Sample Collection Date": "Sample_Collection_Date",
    "Foreoptic Type": "Foreoptic_Type",
    "Vegetation Type": "Vegetation_Type",
}
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
        context = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(req, timeout=timeout, context=context) as r:
            raw = r.read().decode("utf-8", errors="replace")
        if raw.strip().startswith("{") or raw.strip().startswith("["):
            return pd.DataFrame()
        df = pd.read_csv(io.StringIO(raw))
        return df if not df.empty else pd.DataFrame()
    except Exception as exc:
        print(f"    ERROR: EcoSIS download failed for {package_id}: {exc}")
        raise


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
    for path in (
        root / TALL2018_METADATA_CSV,
        Path.home() / "Downloads" / TALL2018_METADATA_CSV,
    ):
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
    if (
        "Genus" in merged.columns
        and "Species" in merged.columns
        and "Latin Genus" not in merged.columns
    ):
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


def load_cabo_v2(timeout=120):
    """
    Load CABO 2018-2019 leaf-level spectra via EcoSIS export API.

    CABO v2 is a broad multispecies trait-linked leaf spectroscopy dataset that
    is useful as additional background diversity.
    """
    df = fetch_package_csv(CABO_V2_PACKAGE_ID, metadata=True, timeout=timeout)
    if df.empty:
        df = fetch_package_csv(CABO_V2_PACKAGE_ID, metadata=False, timeout=timeout)
    if df.empty:
        return df
    df = df.rename(columns={k: v for k, v in CABO_V2_RENAME_MAP.items() if k in df.columns})
    df = attach_species_column(df)
    if "Latin Name" in df.columns and "Species" not in df.columns:
        df["Species"] = df["Latin Name"].astype(str).str.strip()
    if "Species" not in df.columns and "USDA_Species_Code" in df.columns:
        df["Species"] = df["USDA_Species_Code"].astype(str).str.strip()
    if "C_mass_g_g" in df.columns and "N_mass_g_g" in df.columns and "CN_ratio" not in df.columns:
        n_mass = pd.to_numeric(df["N_mass_g_g"], errors="coerce")
        c_mass = pd.to_numeric(df["C_mass_g_g"], errors="coerce")
        df["CN_ratio"] = np.where(n_mass > 0, c_mass / n_mass, np.nan)
    return df


def load_angers_2003(timeout=120):
    """
    Load the ANGERS 2003 leaf optical properties package from EcoSIS.

    This package includes leaf-level optical and biochemical measurements.
    When both reflectance and transmittance records are present, keep the
    reflectance rows for consistency with the rest of the pipeline.
    """
    df = fetch_package_csv(ANGERS_2003_PACKAGE_ID, metadata=True, timeout=timeout)
    if df.empty:
        df = fetch_package_csv(ANGERS_2003_PACKAGE_ID, metadata=False, timeout=timeout)
    if df.empty:
        return df
    df = df.replace(-999, np.nan)
    if "Measurement_type" in df.columns:
        mt = df["Measurement_type"].astype(str).str.lower()
        refl = mt.str.contains("refl", na=False)
        if refl.any():
            df = df.loc[refl].reset_index(drop=True)
    if "Latin Name" in df.columns:
        df["Species"] = df["Latin Name"].astype(str).str.replace("_", " ", regex=False).str.strip()
        genus = df["Species"].str.split().str[0].fillna("")
        df["Genus"] = genus.replace({"nan": ""}).str.strip()
    elif "English Name" in df.columns:
        df["Species"] = df["English Name"].astype(str).str.strip()
    return df


def load_blackhawk_island_wi(timeout=120):
    """
    Load the Blackhawk Island fresh leaf spectra package from EcoSIS.

    This is a leaf-level contact dataset with species labels and a small but
    useful trait set centered on chlorophyll-a and LMA.
    """
    df = fetch_package_csv(BLACKHAWK_ISLAND_WI_PACKAGE_ID, metadata=True, timeout=timeout)
    if df.empty:
        df = fetch_package_csv(BLACKHAWK_ISLAND_WI_PACKAGE_ID, metadata=False, timeout=timeout)
    if df.empty:
        return df
    df = df.replace(-999, np.nan)
    df = attach_species_column(df)
    if "Species" not in df.columns and "Common Name" in df.columns:
        df["Species"] = df["Common Name"].astype(str).str.strip()
    if "Genus" not in df.columns and "Latin Genus" in df.columns:
        df["Genus"] = df["Latin Genus"].astype(str).str.strip()
    return df


def load_ngee_panama(timeout=120):
    """
    Load the NGEE Tropics Panama leaf spectra package from EcoSIS.

    This package includes leaf-level reflectance plus species labels and
    structural/chemical traits, making it a better fit for species-level
    discovery than canopy-labeled packages.
    """
    df = fetch_package_csv(NGEE_PANAMA_PACKAGE_ID, metadata=True, timeout=timeout)
    if df.empty:
        df = fetch_package_csv(NGEE_PANAMA_PACKAGE_ID, metadata=False, timeout=timeout)
    if df.empty:
        return df
    df = attach_species_column(df)
    if "Species" not in df.columns and "Genus_species" in df.columns:
        df["Species"] = df["Genus_species"].astype(str).str.replace("_", " ", regex=False).str.strip()
    if "Species" in df.columns:
        species = (
            df["Species"]
            .astype(str)
            .str.replace("_", " ", regex=False)
            .str.replace(r"\bnan\b", "", regex=True)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )
        species = species.replace({"": np.nan, "nan": np.nan})
        df["Species"] = species
    if "Species" not in df.columns or df["Species"].isna().all():
        if "Species_Code" in df.columns:
            df["Species"] = df["Species_Code"].astype(str).str.strip()
        elif "Spectra_Name" in df.columns:
            df["Species"] = df["Spectra_Name"].astype(str).str.strip()
    if "Genus" not in df.columns and "Latin Genus" in df.columns:
        df["Genus"] = df["Latin Genus"].astype(str).str.strip()
    return df


def load_lopex93(timeout=120):
    """
    Load LOPEX93 directly from EcoSIS with metadata=true.

    Current EcoSIS exports include the biochemical trait columns and Sample_#
    directly, so no local Excel join is needed during normal runs.
    """
    for package_id in LOPEX93_PACKAGE_IDS:
        df = fetch_package_csv(package_id, metadata=True, timeout=timeout)
        if not df.empty:
            break
    else:
        return pd.DataFrame()

    df = df.replace(-999, np.nan)

    if "Latin Name" in df.columns:
        latin = df["Latin Name"].astype(str).str.strip()
        if "Species" in df.columns:
            df["Species"] = df["Species"].where(df["Species"].notna(), latin)
        else:
            df["Species"] = latin
    return df


def load_sierra_nevada_2023(timeout=120):
    """
    Load Sierra Nevada 2023 directly from EcoSIS with metadata=true.

    Current EcoSIS exports include the measured trait columns and taxonomy, so
    the local supplemental CSV is no longer required for normal runs.
    """
    df = fetch_package_csv(SIERRA_NEVADA_2023_PACKAGE_ID, metadata=True, timeout=timeout)
    if df.empty:
        return df
    df = df.rename(columns={k: v for k, v in SIERRA_V2_RENAME_MAP.items() if k in df.columns})
    df = attach_species_column(df)
    if "genus" not in df.columns and "Latin Genus" in df.columns:
        df["genus"] = df["Latin Genus"].astype(str).str.strip()
    if "family" in df.columns:
        df["family"] = df["family"].astype(str).str.strip()
    return df


def filter_lopex_spectral_outliers(
    X,
    df,
    contamination=LOPEX_OUTLIER_CONTAMINATION,
    random_state=42,
):
    """
    Drop LOPEX93 spectra far from the main cluster in preprocessed space.

    IsolationForest on Savitzky-Golay + SNV + grid-aligned spectra; applied before
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
