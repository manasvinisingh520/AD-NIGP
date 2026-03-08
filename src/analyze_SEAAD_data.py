"""Analyze MTG/SEA AD: plot Braak/Thal bar charts, ADNC pie chart, and Thal x Braak heatmap.
Also: plot original atlas Astrocytes cell metadata - pie chart of overall pathology stage (Path..Group.).
"""
import os
import sys

import anndata as ad
import numpy as np
import pandas as pd


def _find_mtg_h5ad():
    for path in [
        "data/model_data/MTG_AnnData_perCell.h5ad",
        "data/SEA_AD_Data/MTG/MTG_AnnData_perCell.h5ad",
        "data/SEA_AD_Data/MTG/MTG_AnnData.h5ad",
    ]:
        if os.path.exists(path):
            return path
    return None


def _load_mtg_patient_labels(include_adnc=False):
    """Load MTG AnnData, return (patient_ids, thal_per_patient, braak_per_patient[, adnc_per_patient]) after dropping Reference."""
    path = _find_mtg_h5ad()
    if path is None:
        out = (None, None, None)
        if include_adnc:
            out = (None, None, None, None)
        return out
    adata = ad.read_h5ad(path)
    if "SampleName" not in adata.obs.columns:
        if "Donor_ID" in adata.obs.columns:
            adata.obs["SampleName"] = adata.obs["Donor_ID"].astype(str)
        elif "Donor ID" in adata.obs.columns:
            adata.obs["SampleName"] = adata.obs["Donor ID"].astype(str)
        else:
            raise ValueError("MTG AnnData must have SampleName or Donor_ID / Donor ID.")
    if "Thal" not in adata.obs.columns or "Braak" not in adata.obs.columns:
        raise ValueError("MTG AnnData must have Thal and Braak columns.")
    thal_s = adata.obs["Thal"].astype(str).str.strip()
    braak_s = adata.obs["Braak"].astype(str).str.strip()
    valid = (
        adata.obs["Thal"].notna()
        & (thal_s != "")
        & (thal_s != "Reference")
        & adata.obs["Braak"].notna()
        & (braak_s != "")
        & (braak_s != "Reference")
    )
    adata = adata[valid].copy()
    unique_patients = adata.obs["SampleName"].unique()
    thal_per_patient = []
    braak_per_patient = []
    adnc_per_patient = [] if include_adnc else None
    adnc_col = "Overall AD neuropathological Change" if include_adnc else None
    if include_adnc and adnc_col in adata.obs.columns:
        adnc_s = adata.obs[adnc_col].astype(str).str.strip()
    else:
        adnc_col = None
    for pid in unique_patients:
        mask = adata.obs["SampleName"] == pid
        thal_per_patient.append(adata.obs["Thal"].astype(str).str.strip().loc[mask].iloc[0])
        braak_per_patient.append(adata.obs["Braak"].astype(str).str.strip().loc[mask].iloc[0])
        if adnc_col is not None:
            adnc_per_patient.append(adnc_s.loc[mask].iloc[0])
    result = (
        np.array(unique_patients),
        np.array(thal_per_patient),
        np.array(braak_per_patient),
    )
    if include_adnc:
        result = result + (np.array(adnc_per_patient) if adnc_col else np.array([], dtype=object),)
    return result


def _braak_sort_key(s):
    """Order Braak stages: 0, I, II, III, IV, V, VI."""
    s = str(s).strip().upper()
    order = {"BRAAK 0": 0, "0": 0, "BRAAK I": 1, "I": 1, "BRAAK II": 2, "II": 2,
             "BRAAK III": 3, "III": 3, "BRAAK IV": 4, "IV": 4, "BRAAK V": 5, "V": 5, "BRAAK VI": 6, "VI": 6}
    for k, v in order.items():
        if k in s or s == k:
            return (v, s)
    return (99, s)


def run_plots(outdir=None):
    """Plot bar charts (Braak, Thal), pie chart (ADNC), and heatmap (Thal x Braak)."""
    import matplotlib
    do_save = outdir is not None
    if do_save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    out = _load_mtg_patient_labels(include_adnc=True)
    if len(out) == 4:
        patient_ids, thal, braak, adnc = out
    else:
        patient_ids, thal, braak = out
        adnc = np.array([], dtype=object)
    if patient_ids is None:
        print("MTG AnnData not found. Cannot plot.")
        return
    if do_save:
        os.makedirs(outdir, exist_ok=True)
    thal = np.asarray([str(t).strip() for t in thal])
    braak = np.asarray([str(b).strip() for b in braak])

    # Bar chart: Braak stage distribution
    braak_counts = pd.Series(braak).value_counts()
    braak_ordered = sorted(braak_counts.index, key=_braak_sort_key)
    braak_counts = braak_counts.reindex(braak_ordered).fillna(0).astype(int)
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(range(len(braak_counts)), braak_counts.values, color="steelblue", edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(braak_counts)))
    ax.set_xticklabels(braak_counts.index, rotation=45, ha="right")
    ax.set_ylabel("Number of patients")
    ax.set_title("Braak stage distribution (patient-level)")
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5, int(b.get_height()), ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    if do_save:
        fig.savefig(os.path.join(outdir, "braak_stage_bar.png"), dpi=150, bbox_inches="tight")
        print(f"Saved {os.path.join(outdir, 'braak_stage_bar.png')}")
        plt.close()
    # Bar chart: Thal stage distribution
    thal_counts = pd.Series(thal).value_counts()
    thal_ordered = sorted(thal_counts.index, key=lambda x: (x.replace("thal", "").strip(), x))
    thal_counts = thal_counts.reindex(thal_ordered).fillna(0).astype(int)
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(range(len(thal_counts)), thal_counts.values, color="darkorange", edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(thal_counts)))
    ax.set_xticklabels(thal_counts.index, rotation=45, ha="right")
    ax.set_ylabel("Number of patients")
    ax.set_title("Thal stage distribution (patient-level)")
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5, int(b.get_height()), ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    if do_save:
        fig.savefig(os.path.join(outdir, "thal_stage_bar.png"), dpi=150, bbox_inches="tight")
        print(f"Saved {os.path.join(outdir, 'thal_stage_bar.png')}")
        plt.close()

    # Pie chart: Overall ADNC stage
    if adnc is not None and len(adnc) > 0:
        adnc_counts = pd.Series(adnc).value_counts()
        adnc_counts = adnc_counts[adnc_counts.index.astype(str).str.strip() != ""]
        if len(adnc_counts) > 0:
            fig, ax = plt.subplots(figsize=(7, 7))
            colors = plt.cm.Set3(np.linspace(0, 1, len(adnc_counts)))
            wedges, labels, pct = ax.pie(adnc_counts.values, labels=adnc_counts.index, autopct="%1.1f%%", colors=colors, startangle=90)
            for w in wedges:
                w.set_edgecolor("white")
                w.set_linewidth(1)
            ax.set_title("Overall AD neuropathological change (ADNC) – patient-level")
            plt.tight_layout()
            if do_save:
                fig.savefig(os.path.join(outdir, "adnc_stage_pie.png"), dpi=150, bbox_inches="tight")
                print(f"Saved {os.path.join(outdir, 'adnc_stage_pie.png')}")
                plt.close()
        else:
            print("No ADNC values to plot; skipping pie chart.")
    else:
        print("ADNC column not found or empty; skipping pie chart.")

    # Heatmap: Thal x Braak combinations
    df = pd.DataFrame({"Thal": thal, "Braak": braak})
    cross = pd.crosstab(df["Thal"], df["Braak"])
    thal_order = sorted(cross.index, key=lambda x: (x.replace("thal", "").strip(), x))
    braak_order = sorted(cross.columns, key=_braak_sort_key)
    cross = cross.reindex(index=thal_order, columns=braak_order).fillna(0).astype(int)
    fig, ax = plt.subplots(figsize=(max(6, cross.shape[1] * 1.2), max(4, cross.shape[0] * 0.6)))
    sns.heatmap(cross, annot=True, fmt="d", cmap="YlOrRd", ax=ax, cbar_kws={"label": "Number of patients"})
    ax.set_title("Thal x Braak stage combination (patient count)")
    ax.set_xlabel("Braak stage")
    ax.set_ylabel("Thal stage")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    if do_save:
        fig.savefig(os.path.join(outdir, "thal_braak_heatmap.png"), dpi=150, bbox_inches="tight")
        print(f"Saved {os.path.join(outdir, 'thal_braak_heatmap.png')}")
        plt.close()
    if not do_save:
        plt.show()


def _load_atlas_astrocytes_metadata():
    """Load original atlas Astrocytes cell metadata from Excel (all region sheets). Returns DataFrame with Path..Group. or None."""
    try:
        from config import REGION_TO_TAB, DATA_DIR
        from utils import read_excel_columns
    except ImportError:
        return None
    path = os.path.join(DATA_DIR, "2025-11-16_Astrocytes_metadata.xlsx")
    if not os.path.exists(path):
        return None
    PATHOLOGY_COL = "Path..Group."
    dfs = []
    for region, tab_idx in REGION_TO_TAB.items():
        try:
            df = read_excel_columns(
                path,
                columns=["cell_annotation", PATHOLOGY_COL],
                sheet_name=tab_idx,
            )
            df["region"] = region
            dfs.append(df)
        except Exception:
            continue
    if not dfs:
        return None
    return pd.concat(dfs, ignore_index=True)


def run_atlas_astrocytes_plot(outdir=None):
    """Load atlas Astrocytes cell metadata and plot pie chart of overall pathology stage (Path..Group.)."""
    import matplotlib
    do_save = outdir is not None
    if do_save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = _load_atlas_astrocytes_metadata()
    if df is None:
        print("Atlas Astrocytes metadata not found (data/AtlasData/2025-11-16_Astrocytes_metadata.xlsx).")
        return
    PATHOLOGY_COL = "Path..Group."
    if PATHOLOGY_COL not in df.columns:
        print("Column 'Path..Group.' not in metadata. Available: %s" % list(df.columns))
        return
    s = df[PATHOLOGY_COL].dropna().astype(str).str.strip()
    s = s[s != ""]
    if s.empty:
        print("No pathology stage values to plot.")
        return
    vc = s.value_counts()

    fig, ax = plt.subplots(figsize=(7, 7))
    colors = plt.cm.Set3(np.linspace(0, 1, len(vc)))
    wedges, labels, pct = ax.pie(vc.values, labels=vc.index, autopct="%1.1f%%", colors=colors, startangle=90)
    for w in wedges:
        w.set_edgecolor("white")
        w.set_linewidth(1)
    ax.set_title("Atlas Astrocytes - overall pathology stage (Path..Group., cell-level)")
    plt.tight_layout()
    if do_save:
        os.makedirs(outdir, exist_ok=True)
        fig.savefig(os.path.join(outdir, "atlas_astrocytes_pathology_stage_pie.png"), dpi=150, bbox_inches="tight")
        print("Saved %s" % os.path.join(outdir, "atlas_astrocytes_pathology_stage_pie.png"))
        plt.close()
    else:
        plt.show()


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Plot stage distributions: MTG/SEA AD (Braak/Thal/ADNC/heatmap) or atlas Astrocytes (pathology stage pie)"
    )
    parser.add_argument("--outdir", type=str, default=None, help="Output directory for figures (if not set, figures are not saved)")
    parser.add_argument("--atlas_astrocytes", action="store_true", help="Plot atlas Astrocytes cell metadata: pie chart of overall pathology stage (Path..Group.)")
    args = parser.parse_args()

    if args.atlas_astrocytes:
        run_atlas_astrocytes_plot(outdir=args.outdir)
    else:
        run_plots(outdir=args.outdir)


if __name__ == "__main__":
    main()
