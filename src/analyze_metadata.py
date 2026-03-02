"""
Analyze metadata: distribution of pathology stage, ptau, and Thal (CERAD) across Astrocytes and Microglia.
"""

import os
import argparse
import pandas as pd
import numpy as np
from config import REGION_TO_TAB, DATA_DIR
from utils import read_excel_columns

# Paths
ASTROCYTES_METADATA_XLSX = os.path.join(DATA_DIR, "2025-11-16_Astrocytes_metadata.xlsx")
MICROGLIA_METADATA_CSV = os.path.join(DATA_DIR, "2025-12-27_Microglia_Metadata.csv")
PTAU_COL = "Ptau.Total.Tau..A.U.."
THAL_COL = "Thal"


def _load_astrocytes_metadata():
    """Load Astrocytes metadata from Excel (all region sheets). Returns one DataFrame with Thal, ptau, region."""
    if not os.path.exists(ASTROCYTES_METADATA_XLSX):
        return None
    dfs = []
    for region, tab_idx in REGION_TO_TAB.items():
        try:
            df = read_excel_columns(
                ASTROCYTES_METADATA_XLSX,
                columns=["SampleName", THAL_COL, PTAU_COL],
                sheet_name=tab_idx,
            )
            df["region"] = region
            dfs.append(df)
        except Exception as e:
            print(f"  Skip {region}: {e}")
    if not dfs:
        return None
    return pd.concat(dfs, ignore_index=True)


def _load_microglia_metadata():
    """Load Microglia metadata from CSV. Tries common column names for Thal and ptau."""
    if not os.path.exists(MICROGLIA_METADATA_CSV):
        return None
    df = pd.read_csv(MICROGLIA_METADATA_CSV)
    # Normalize column names: some CSVs use "Thal" / "Ptau..." or "Braak" / "CERAD"
    if PTAU_COL not in df.columns:
        for c in ["Ptau.Total.Tau..A.U..", "Braak", "ptau", "Ptau"]:
            if c in df.columns:
                df = df.rename(columns={c: PTAU_COL})
                break
    if THAL_COL not in df.columns:
        for c in ["Thal", "CERAD", "thal"]:
            if c in df.columns:
                df = df.rename(columns={c: THAL_COL})
                break
    return df


def _print_distribution(name, series, drop_unk=True):
    """Print value counts and (for numeric) summary stats."""
    s = series.dropna()
    if drop_unk and s.astype(str).str.strip().str.lower().eq("unk").any():
        s = s[s.astype(str).str.strip().str.lower() != "unk"]
    print(f"\n--- {name} ---")
    if s.empty:
        print("  (no valid values)")
        return
    if pd.api.types.is_numeric_dtype(s):
        print(s.describe().to_string())
        print("\n  Value counts (binned/raw):")
        vc = s.value_counts().sort_index()
        for val, count in vc.items():
            print(f"    {val}: {count}")
    else:
        vc = s.astype(str).str.strip().value_counts().sort_index()
        for val, count in vc.items():
            print(f"    {val}: {count}")
        print(f"  Total: {vc.sum()}")


def _plot_distributions(df, thal_col, ptau_col, title_prefix, region_col=None, save_path=None):
    """Pie chart for Thal, histogram for ptau. Requires matplotlib."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    # Thal – pie chart
    if thal_col in df.columns:
        s = df[thal_col].dropna().astype(str).str.strip()
        s = s[s.str.lower() != "unk"]
        if not s.empty:
            vc = s.value_counts().sort_index()
            # Bright colors for pie slices
            bright_palette = ["#FF6B6B", "#4ECDC4", "#FFE66D", "#95E1D3", "#F38181", "#AA96DA", "#FCBAD3", "#A8D8EA"]
            colors = [bright_palette[i % len(bright_palette)] for i in range(len(vc))]
            axes[0].pie(vc.values, labels=vc.index.astype(str), autopct="%1.1f%%", colors=colors, startangle=90)
            axes[0].set_title(f"{title_prefix} – Thal (CERAD)")
    # ptau – box plot
    if ptau_col in df.columns:
        s = df[ptau_col].dropna()
        if pd.api.types.is_numeric_dtype(s):
            bp = axes[1].boxplot(s, vert=True, patch_artist=True)
            for patch in bp["boxes"]:
                patch.set_facecolor("#FFB347")  # bright orange
            axes[1].set_ylabel("Ptau.Total.Tau (A.U.)")
            axes[1].set_title(f"{title_prefix} – p-tau")
            axes[1].set_xticklabels(["ptau"])
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.show()


def run_astrocytes(plot=False, save_plot=None):
    """Load Astrocytes metadata and print/plot Thal and ptau distributions."""
    print("=" * 60)
    print("ASTROCYTES metadata")
    print("=" * 60)
    df = _load_astrocytes_metadata()
    if df is None:
        print(f"Could not load Astrocytes metadata from {ASTROCYTES_METADATA_XLSX}")
        return
    if THAL_COL not in df.columns or PTAU_COL not in df.columns:
        print(f"Expected columns {THAL_COL}, {PTAU_COL}. Found: {list(df.columns)}")
        return
    print(f"Loaded {len(df)} rows (cells). Unique patients (SampleName): {df['SampleName'].nunique()}")

    # Per-patient: one row per patient (take first Thal/ptau per SampleName for distribution of patients)
    patient_df = df.groupby("SampleName", as_index=False).agg({THAL_COL: "first", PTAU_COL: "first", "region": "first"})
    print(f"Unique patients: {len(patient_df)}")

    _print_distribution("Thal (patient-level)", patient_df[THAL_COL])
    _print_distribution("Ptau (patient-level)", patient_df[PTAU_COL])

    if "region" in patient_df.columns:
        print("\n--- Thal by region ---")
        for region in patient_df["region"].dropna().unique():
            sub = patient_df[patient_df["region"] == region]
            thal_vc = sub[THAL_COL].dropna().astype(str).str.strip()
            thal_vc = thal_vc[thal_vc.str.lower() != "unk"]
            if thal_vc.empty:
                continue
            print(f"  {region}: {thal_vc.value_counts().sort_index().to_dict()}")

        print("\n--- Ptau by region (mean, count) ---")
        for region in patient_df["region"].dropna().unique():
            sub = patient_df[patient_df["region"] == region][PTAU_COL].dropna()
            if pd.api.types.is_numeric_dtype(sub):
                print(f"  {region}: n={len(sub)}, mean={sub.mean():.4f}, min={sub.min()}, max={sub.max()}")
            else:
                print(f"  {region}: n={len(sub)}")

    if plot or save_plot:
        _plot_distributions(patient_df, THAL_COL, PTAU_COL, "Astrocytes", save_path=save_plot)


def run_microglia(plot=False, save_plot=None):
    """Load Microglia metadata and print/plot Thal (CERAD) and ptau (Braak) distributions."""
    print("\n" + "=" * 60)
    print("MICROGLIA metadata")
    print("=" * 60)
    df = _load_microglia_metadata()
    if df is None:
        print(f"Could not load Microglia metadata from {MICROGLIA_METADATA_CSV}")
        return
    print(f"Loaded {len(df)} rows. Columns: {list(df.columns)}")
    # If CSV has Donor ID, aggregate to one row per patient for distributions
    if "Donor ID" in df.columns:
        agg_cols = {}
        if THAL_COL in df.columns:
            agg_cols[THAL_COL] = "first"
        if PTAU_COL in df.columns:
            agg_cols[PTAU_COL] = "first"
        if "region" in df.columns:
            agg_cols["region"] = "first"
        if agg_cols:
            df = df.groupby("Donor ID", as_index=False).agg(agg_cols)
        print(f"Unique donors (patient-level): {len(df)}")

    if THAL_COL not in df.columns:
        print(f"  No '{THAL_COL}' column found. Skipping Thal distribution.")
    else:
        _print_distribution("Thal (CERAD)", df[THAL_COL])

    if PTAU_COL not in df.columns:
        print(f"  No '{PTAU_COL}' column found. Skipping ptau distribution.")
    else:
        # Try numeric
        ptau = pd.to_numeric(df[PTAU_COL], errors="coerce")
        if ptau.notna().any():
            _print_distribution("Ptau (Braak raw)", ptau)
            # Braak stage distribution (rounded)
            braak = ptau.dropna()
            braak_rounded = braak.round().astype(int)
            print("\n--- Braak stage (rounded ptau) ---")
            vc = braak_rounded.value_counts().sort_index()
            for val, count in vc.items():
                print(f"    {val}: {count}")
        else:
            _print_distribution("Ptau", df[PTAU_COL])

    if "region" in df.columns:
        print("\n--- Thal by region ---")
        for region in df["region"].dropna().unique():
            sub = df[df["region"] == region][THAL_COL].dropna().astype(str).str.strip()
            sub = sub[sub.str.lower() != "unk"]
            if sub.empty:
                continue
            print(f"  {region}: {sub.value_counts().sort_index().to_dict()}")

    if plot or save_plot:
        _plot_distributions(df, THAL_COL, PTAU_COL, "Microglia", save_path=save_plot)


def main():
    parser = argparse.ArgumentParser(description="Distribution of pathology stage, ptau, and Thal")
    parser.add_argument("--cell_type", type=str, default="both", choices=["Astrocytes", "Microglia", "both"],
                        help="Which metadata to analyze (default: both)")
    parser.add_argument("--plot", action="store_true", help="Show matplotlib plots")
    parser.add_argument("--save_plot", type=str, default=None,
                        help="Save plot to this path (e.g. metadata_distributions.png)")
    args = parser.parse_args()

    if args.cell_type in ("Astrocytes", "both"):
        run_astrocytes(plot=args.plot, save_plot=args.save_plot)
    if args.cell_type in ("Microglia", "both"):
        run_microglia(plot=args.plot, save_plot=args.save_plot)


if __name__ == "__main__":
    main()
