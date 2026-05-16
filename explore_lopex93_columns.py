"""
explore_lopex93_columns.py
Utility: LOPEX93 column inspector

Downloads the LOPEX93 dataset from EcoSIS and prints a summary of all
non-spectral columns so you can verify the download and identify trait columns.

Run before any other stage:
    python explore_lopex93_columns.py
"""

import io, urllib.request, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

LOPEX93_IDS = [
    "leaf-optical-properties-experiment-database--lopex93-",
    "3849_2", "3849",
]
ECOSIS_API = "https://ecosis.org/api/package/{id}/export?format=csv"

def fetch_csv(did, timeout=120):
    url = ECOSIS_API.format(id=did)
    print(f"    -> {url}")
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
    except Exception as exc:
        print(f"    x Failed: {exc}")
        return pd.DataFrame()

def is_wl(col):
    try:
        v = float(col); return 300.0 <= v <= 2600.0
    except (ValueError, TypeError):
        return False

def summarise(df):
    spectral_cols = [c for c in df.columns if is_wl(c)]
    meta_cols     = [c for c in df.columns if not is_wl(c)]
    n = len(df)
    print(f"\n{'='*65}")
    print("  LOPEX93 Dataset Summary")
    print(f"{'='*65}")
    print(f"  Rows            : {n}")
    print(f"  Spectral bands  : {len(spectral_cols)}")
    if spectral_cols:
        wl = sorted(float(c) for c in spectral_cols)
        print(f"  Wavelength range: {wl[0]:.0f} - {wl[-1]:.0f} nm")
    print(f"  Metadata cols   : {len(meta_cols)}")
    if meta_cols:
        print(f"\n{'-'*65}")
        print(f"  {'Column':<36} {'Dtype':<10} {'Non-Null':>8}  {'Fill %':>7}")
        print(f"{'-'*65}")
        for col in meta_cols:
            nn  = int(df[col].notna().sum())
            pct = 100.0 * nn / n if n else 0.0
            dt  = str(df[col].dtype)
            print(f"  {col:<36} {dt:<10} {nn:>8}  {pct:>6.1f}%")
    else:
        print("\n  No metadata columns found.")
        print("  The EcoSIS LOPEX93 export contains only spectral bands.")
        print("  Biochemical traits must be supplied via lopex1993metadata.xlsx")

def main():
    print("\n" + "="*65)
    print("  explore_lopex93_columns.py  -  LOPEX93 Column Inspector")
    print("="*65)
    df = pd.DataFrame()
    for did in LOPEX93_IDS:
        print(f"  Trying id='{did}' ...")
        df = fetch_csv(did)
        if not df.empty:
            break
    if df.empty:
        print("  Download failed. Check internet connection.")
        return
    summarise(df)

if __name__ == "__main__":
    main()
    print("\nDone.\n")