# EDA over snRNA-seq cohorts (ROSMAP, SEA-AD, Atlas): per-donor sequencing stats
# and Table 1. Column names differ per dataset, so they live in DATASET_CONFIG below.
import os
import sys
import anndata as ad
import matplotlib.pyplot as plt

# Avoid cp1252 UnicodeEncodeError when printing tables with e.g. Greek 'ε' (Windows).
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency
from tableone import TableOne
from utils import _coerce_numeric_series


# ---------------------------------------------------------------------------
# Dataset configuration: the "respective column names" for each cohort.
# canonical concept -> that dataset's actual obs column / value coding.
# ---------------------------------------------------------------------------
DATASET_CONFIG = {
    "ROSMAP": {
        "loader": "h5ad",
        "path": "data/ROSMAP_Data/{cell_type}.h5ad",
        "cell_types": ["Astrocytes", "Microglia"],
        "counts_layer": "counts",
        "id_col": "individualID",
        "age_col": "age_death",
        "pmi_col": "pmi",
        "braak_col": "braaksc",
        "cerad_col": "ceradsc",
        "stratifiers": ["braaksc", "ceradsc"],
        "table1_continuous": ["age_death", "educ", "pmi", "cts_mmse30_lv"],
        "table1_categorical": ["msex", "race", "apoe_genotype", "cogdx", "Study", "braaksc", "ceradsc"],
        "row_labels": {
            "age_death": "Age at death (yrs)", "educ": "Education (yrs)",
            "pmi": "Post-mortem interval (h)", "cts_mmse30_lv": "MMSE (last visit)",
            "msex": "Sex", "race": "Race", "apoe_genotype": "APOE genotype",
            "cogdx": "Clinical diagnosis", "Study": "Study cohort",
            "braaksc": "Braak stage", "ceradsc": "CERAD score",
        },
        "value_labels": {
            "msex": {0: "Female", 1: "Male"},
            "race": {1: "White", 2: "Black", 3: "Native American", 6: "Asian", 7: "Other"},
            "apoe_genotype": {22: "ε2/ε2", 23: "ε2/ε3", 24: "ε2/ε4",
                              33: "ε3/ε3", 34: "ε3/ε4", 44: "ε4/ε4"},
            "cogdx": {1: "NCI", 2: "MCI", 3: "MCI + other", 4: "AD",
                      5: "AD + other", 6: "Other dementia"},
            "ceradsc": {1: "Definite AD", 2: "Probable AD", 3: "Possible AD", 4: "No AD"},
            "braaksc": {0: "0", 1: "I", 2: "II", 3: "III", 4: "IV", 5: "V", 6: "VI"},
        },
        "group_caption": {
            "braaksc": "Braak stage (0-VI)",
            "ceradsc": "CERAD score (1=Definite AD, 2=Probable, 3=Possible, 4=No AD)",
        },
        # Derived AD-vs-control grouping (in addition to the stage stratifiers above).
        # cogdx: 1=NCI (control), 4=AD, 5=AD+other -> AD; MCI/other-dementia excluded.
        "case_control": {
            "source": "cogdx",
            "label": "ad_vs_control",
            "mapping": {1: "Control", 4: "AD", 5: "AD"},
            "caption": "AD vs control (cogdx: 1=NCI control; 4/5=AD)",
        },
        "corr_cols": ["braaksc", "ceradsc", "age_death", "pmi", "educ", "cts_mmse30_lv"],
        "ordinal_maps": {},   # ROSMAP braaksc/ceradsc are already numeric
        "dist_continuous": ["age_death", "educ", "pmi", "cts_mmse30_lv"],
        "dist_categorical": ["msex", "race", "apoe_genotype", "cogdx", "Study", "braaksc", "ceradsc"],
    },
    "SEAAD": {
        # New CellxGene-format release: cell-type split, de-identified (Age/education/PMI
        # are BINNED categoricals), CellxGene column names, Supertype subtypes.
        "loader": "h5ad",
        "path": "data/SEA_AD_Data/{cell_type}/{cell_type}.h5ad",
        "cell_types": ["Astrocytes", "Microglia"],
        "counts_layer": None,           # no raw-count layer; X used if seq stats requested
        "id_col": "donor_id",
        "age_col": "Age at death",
        "pmi_col": "PMI",
        "braak_col": "Braak stage",
        "cerad_col": "CERAD score",
        "stratifiers": ["Braak stage", "CERAD score"],
        "table1_continuous": [],   # no donor-level continuous vars shown
        "table1_categorical": ["sex", "APOE4 status", "ADNC",
                               "Age at death", "PMI",
                               "Thal phase", "Braak stage", "CERAD score"],
        "row_labels": {
            "Continuous Pseudo-progression Score": "Pseudo-progression score",
            "Age at death": "Age at death", "Years of education": "Education",
            "PMI": "Post-mortem interval", "sex": "Sex", "APOE4 status": "APOE4 status",
            "Cognitive status": "Cognitive status", "ADNC": "ADNC",
            "Braak stage": "Braak stage", "CERAD score": "CERAD score",
            "Thal phase": "Thal phase", "self_reported_ethnicity": "Ethnicity",
            "Supertype": "Subtype (Supertype)",
        },
        "value_labels": {},             # values are already human-readable text
        "group_caption": {"Braak stage": "Braak stage", "CERAD score": "CERAD score"},
        # Derived dementia-vs-no-dementia grouping (SEA-AD analog of AD vs control).
        # 'Cognitive status' is already ~binary; map explicitly and drop anything else.
        "case_control": {
            "source": "Cognitive status",
            "label": "dementia_vs_no_dementia",
            "mapping": {"Dementia": "Dementia", "No dementia": "No dementia"},
            "caption": "Dementia vs no dementia (Cognitive status)",
        },
        "corr_cols": ["Braak stage", "CERAD score", "Thal phase", "ADNC",
                      "Continuous Pseudo-progression Score"],
        "ordinal_maps": {
            "Braak stage": {"Braak 0": 0, "Braak I": 1, "Braak II": 2, "Braak III": 3,
                            "Braak IV": 4, "Braak V": 5, "Braak VI": 6},
            "Thal phase": {"Thal 0": 0, "Thal 1": 1, "Thal 2": 2, "Thal 3": 3,
                           "Thal 4": 4, "Thal 5": 5},
            "CERAD score": {"Absent": 0, "Sparse": 1, "Moderate": 2, "Frequent": 3},
            "ADNC": {"Not AD": 0, "Low": 1, "Intermediate": 2, "High": 3},
            "Age at death": {"Less than 65 years old": 0, "65 to 77 years old": 1,
                             "78 to 89 years old": 2, "90+ years old": 3},
            "Years of education": {"0 to 8 years": 0, "9 to 11 years": 1, "12 to 15 years": 2,
                                   "16 to 18 years": 3, "19 to 21 years": 4},
            "PMI": {"3.2 to 5.9 hours": 0, "5.9 to 8.7 hours": 1, "8.7 to 11.4 hours": 2},
        },
        "dist_continuous": ["Continuous Pseudo-progression Score"],
        "dist_categorical": ["sex", "APOE4 status", "Cognitive status", "ADNC",
                             "Braak stage", "CERAD score", "Thal phase", "Age at death",
                             "Years of education", "PMI", "self_reported_ethnicity", "Supertype"],
    },
    "Atlas": {
        "loader": "atlas_xlsx",         # obs from the astro metadata xlsx (all region sheets)
        "meta_xlsx": "data/AtlasData/2025-11-16_Astrocytes_metadata.xlsx",
        "cell_types": ["Astrocytes", "Microglia"],
        "counts_layer": None,           # raw counts live in per-region MTX, not a single h5ad
        "id_col": "Donor.ID.y",         # true donor id (32 donors); 'SampleName' is per donor-region
        "age_col": "Age",
        "pmi_col": "PMI",
        "braak_col": "Braak",
        "cerad_col": None,              # Atlas has no CERAD; use Thal as the 2nd stratifier
        "stratifiers": ["Braak", "Thal"],
        "table1_continuous": ["Age", "PMI", "RIN"],
        "table1_categorical": ["Sex", "apoe", "Braak", "Thal", "Path..Group."],
        "row_labels": {
            "Age": "Age (yrs)", "PMI": "Post-mortem interval (h)", "RIN": "RIN",
            "Sex": "Sex", "apoe": "APOE", "Braak": "Braak stage",
            "Thal": "Thal phase", "Path..Group.": "Pathology group",
        },
        "value_labels": {},
        "group_caption": {"Braak": "Braak stage", "Thal": "Thal phase",
                          "Path..Group.": "Pathology group"},
        "corr_cols": ["Braak", "Thal", "Age", "PMI", "RIN"],
        "ordinal_maps": {
            "Braak": {"0": 0, "I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6},
        },  # Thal is already numeric ('unk' -> NaN)
        "dist_continuous": ["Age", "PMI", "RIN", "percent.mito", "percent.ribo",
                            "Total.Genes.Detected", "Median.UMI.Counts.per.Cell"],
        "dist_categorical": ["Sex", "apoe", "Braak", "Thal", "Path..Group."],
        # Atlas Microglia uses a different, sparser metadata source (per-region CSVs):
        # it has Braak/CERAD/Pathology Stage/Sex/APOE but no Age/PMI/RIN/Thal.
        "cell_type_overrides": {
            "Microglia": {
                "loader": "atlas_micro_csv",
                "meta_glob": "data/AtlasData/2025-12-27_Microglia_*_metadata.csv",
                "id_col": "Donor ID",
                "stratifiers": ["Braak", "CERAD"],
                # Atlas has no clinical dementia field -> derive a NEUROPATHOLOGY-based
                # AD vs control from Pathology Stage (1-2 = control, 3-4 = AD). This is
                # AD pathology, NOT clinical dementia status.
                "case_control": {
                    "source": "Pathology Stage",
                    "label": "ad_vs_control",
                    "mapping": {1: "Control", 2: "Control", 3: "AD", 4: "AD"},
                    "caption": "AD vs control (Pathology Stage 1-2 vs 3-4; neuropathology, not clinical dementia)",
                },
                "table1_continuous": [],
                "table1_categorical": ["Sex", "APOE", "Braak", "CERAD", "Pathology Stage"],
                "corr_cols": ["Braak", "CERAD", "Pathology Stage"],
                "ordinal_maps": {
                    "Braak": {"0": 0, "I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6},
                    "CERAD": {"none": 0, "sparse": 1, "moderate": 2, "frequent": 3},
                },
                "row_labels": {"Braak": "Braak stage", "CERAD": "CERAD score",
                               "Pathology Stage": "Pathology stage", "Sex": "Sex",
                               "APOE": "APOE genotype"},
                "value_labels": {},
                "group_caption": {"Braak": "Braak stage", "CERAD": "CERAD score"},
                "dist_continuous": [],
                "dist_categorical": ["Sex", "APOE", "Braak", "CERAD", "Pathology Stage"],
            },
        },
    },
}


def _cfg(dataset, cell_type=None):
    """Config for a dataset, with any per-cell-type overrides merged in."""
    cfg = dict(DATASET_CONFIG[dataset])
    overrides = cfg.get("cell_type_overrides", {})
    if cell_type in overrides:
        cfg = {**cfg, **overrides[cell_type]}
    return cfg


def load_dataset(dataset, cell_type=None, backed=False):
    """Load a dataset as an AnnData.

    ROSMAP/SEAAD -> read the h5ad (backed='r' for obs-only work like Table 1).
    Atlas -> build an obs-only AnnData from the metadata xlsx (all region sheets);
             it has no count matrix, so sequencing stats aren't available for Atlas here.
    """
    cfg = _cfg(dataset, cell_type)
    if cfg["loader"] == "h5ad":
        ct = cell_type or cfg["cell_types"][0]
        path = cfg["path"].format(cell_type=ct)
        if not os.path.exists(path):
            raise FileNotFoundError(f"{dataset} file not found: {path}")
        return ad.read_h5ad(path, backed="r" if backed else None)
    if cfg["loader"] == "atlas_micro_csv":
        import glob
        frames = [pd.read_csv(f) for f in sorted(glob.glob(cfg["meta_glob"]))]
        if not frames:
            raise FileNotFoundError(f"No Atlas microglia metadata matched {cfg['meta_glob']}")
        obs = pd.concat(frames, ignore_index=True).drop_duplicates(cfg["id_col"])
        obs.index = obs.index.astype(str)
        return ad.AnnData(obs=obs)
    if cfg["loader"] == "atlas_xlsx":
        import pickle
        xlsx = cfg["meta_xlsx"]
        cache_pkl = os.path.join("data", "AtlasData", ".atlas_obs_cache.pkl")
        mtime = os.path.getmtime(xlsx)
        # reuse cached obs unless the xlsx changed (the multi-sheet read is very slow)
        if os.path.exists(cache_pkl):
            try:
                with open(cache_pkl, "rb") as f:
                    c = pickle.load(f)
                if c.get("mtime") == mtime:
                    return ad.AnnData(obs=c["obs"])
            except Exception:
                pass
        print(f"Reading {os.path.basename(xlsx)} (all sheets; caching)...")
        sheets = pd.read_excel(xlsx, sheet_name=None)
        obs = pd.concat(sheets.values(), ignore_index=True)
        obs.index = obs.index.astype(str)
        try:
            with open(cache_pkl, "wb") as f:
                pickle.dump({"mtime": mtime, "obs": obs}, f)
        except Exception:
            pass
        return ad.AnnData(obs=obs)
    raise ValueError(f"Unknown loader for dataset {dataset}")


def donor_level_counts(adata, column, id_col="individualID"):
    per_donor = adata.obs.drop_duplicates(id_col)
    return per_donor[column].value_counts()


def general_counts(adatas, columns, id_col="individualID"):
    """Print per-donor value counts of `columns` for each (label, adata) in `adatas`."""
    for label, adata in adatas.items():
        print("=" * 100, f"\n{label}")
        for column in columns:
            if column in adata.obs.columns:
                print(f"\n{column} (per {id_col}):")
                print(donor_level_counts(adata, column, id_col))


def donor_sequencing_stats(adata, id_col="individualID", layer="counts"):
    """Per-individual sequencing statistics derived from the count matrix.

    Aggregates the raw counts (layers[layer] if present, else .X) to one row per
    donor, summarizing sequencing depth and library complexity.

    Returns a DataFrame indexed by donor with:
      n_cells, total_counts,
      mean/median counts_per_cell, mean/median genes_per_cell,
      n_genes_detected (genes with >0 counts anywhere in that donor),
      mean_pct_mito (avg fraction of counts in MT- genes per cell).
    """
    import scipy.sparse as sp

    X = adata.layers[layer] if (layer and layer in getattr(adata, "layers", {})) else adata.X
    if not sp.issparse(X):
        X = sp.csr_matrix(X)
    X = X.tocsr()

    counts_per_cell = np.asarray(X.sum(axis=1)).ravel()
    genes_per_cell = np.asarray((X > 0).sum(axis=1)).ravel()

    # mitochondrial fraction per cell (gene symbols starting with 'MT-')
    mt_mask = np.asarray(adata.var_names.astype(str).str.upper().str.startswith("MT-"))
    if mt_mask.any():
        mt_counts = np.asarray(X[:, mt_mask].sum(axis=1)).ravel()
        with np.errstate(divide="ignore", invalid="ignore"):
            pct_mito = np.where(counts_per_cell > 0, mt_counts / counts_per_cell * 100, np.nan)
    else:
        pct_mito = np.full(counts_per_cell.shape, np.nan)

    df = pd.DataFrame({
        id_col: np.asarray(adata.obs[id_col].values),
        "counts": counts_per_cell,
        "genes": genes_per_cell,
        "pct_mito": pct_mito,
    })
    g = df.groupby(id_col, observed=True)
    stats = pd.DataFrame({
        "n_cells": g.size(),
        "total_counts": g["counts"].sum(),
        "mean_counts_per_cell": g["counts"].mean(),
        "median_counts_per_cell": g["counts"].median(),
        "mean_genes_per_cell": g["genes"].mean(),
        "median_genes_per_cell": g["genes"].median(),
        "mean_pct_mito": g["pct_mito"].mean(),
    })

    # genes detected per donor (>0 counts in any of that donor's cells)
    codes, uniques = pd.factorize(df[id_col].values)
    n_genes_detected = np.zeros(len(uniques), dtype=int)
    for i in range(len(uniques)):
        rows = np.flatnonzero(codes == i)
        n_genes_detected[i] = int((X[rows].sum(axis=0) > 0).sum())
    stats["n_genes_detected"] = pd.Series(n_genes_detected, index=uniques)

    return stats.sort_values("n_cells", ascending=False)


# ---- Donor-level sequencing-stat plots ----
# (column, human-readable label) for the metrics produced by donor_sequencing_stats
_DONOR_SEQ_METRICS = [
    ("n_cells", "cells per donor"),
    ("total_counts", "total counts per donor"),
    ("mean_counts_per_cell", "mean counts / cell"),
    ("median_counts_per_cell", "median counts / cell"),
    ("mean_genes_per_cell", "mean genes / cell"),
    ("median_genes_per_cell", "median genes / cell"),
    ("mean_pct_mito", "mean % mito"),
    ("n_genes_detected", "genes detected per donor"),
]


def _finish_fig(fig, save_path):
    """Save the figure to save_path (making dirs) or show it interactively."""
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    else:
        plt.show()


def plot_seq_stats_histograms(stats_df, title="donor sequencing stats", bins=30, save_path=None):
    """Grid of histograms (one per metric) for a per-donor sequencing-stats DataFrame.

    A red dashed line marks each metric's median. Saves to save_path if given,
    otherwise shows the figure. Returns the figure.
    """
    df = stats_df
    metrics = [(c, l) for c, l in _DONOR_SEQ_METRICS if c in df.columns]

    ncols = 4
    nrows = int(np.ceil(len(metrics) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.2 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, (col, label) in zip(axes, metrics):
        vals = pd.to_numeric(df[col], errors="coerce").dropna()
        ax.hist(vals, bins=bins, edgecolor="black", linewidth=0.5)
        med = vals.median()
        ax.axvline(med, color="red", linestyle="--", linewidth=1, label=f"median={med:,.1f}")
        ax.set_xlabel(label)
        ax.set_ylabel("donors")
        ax.legend(fontsize=8)

    for ax in axes[len(metrics):]:
        ax.set_visible(False)

    fig.suptitle(f"{title} ({len(df)} donors)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    _finish_fig(fig, save_path)
    return fig


def compare_seq_stats(named_stats, bins=40, save_path=None):
    """Overlay per-metric histograms (normalized density) for several stats DataFrames.

    named_stats: dict {label -> stats DataFrame}.
    """
    common = None
    for df in named_stats.values():
        cols = {c for c, _ in _DONOR_SEQ_METRICS if c in df.columns}
        common = cols if common is None else (common & cols)
    metrics = [(c, l) for c, l in _DONOR_SEQ_METRICS if c in common]

    ncols = 4
    nrows = int(np.ceil(len(metrics) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.2 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, (col, label) in zip(axes, metrics):
        for name, df in named_stats.items():
            v = pd.to_numeric(df[col], errors="coerce").dropna()
            ax.hist(v, bins=bins, alpha=0.5, density=True, label=name)
        ax.set_xlabel(label)
        ax.set_ylabel("density")
        ax.legend(fontsize=8)

    for ax in axes[len(metrics):]:
        ax.set_visible(False)

    fig.suptitle("Donor sequencing stats: " + " vs ".join(named_stats), fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    _finish_fig(fig, save_path)
    return fig


def patient_cells(adata, id_col="individualID", bin_width=10, title=None):
    counts = adata.obs.groupby(id_col).size()
    max_count = int(counts.max())
    bins = np.arange(0, max_count + bin_width, bin_width)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(counts.values, bins=bins, edgecolor="black", linewidth=0.5)
    ax.set_xlabel(f"cells per donor (bins of {bin_width})")
    ax.set_ylabel("number of donors")
    ax.set_title(title or f"Donor cell counts ({len(counts)} donors)")
    fig.tight_layout()
    plt.show()


def _age_per_donor(adata, id_col="individualID", col="age_death"):
    per_donor = adata.obs.drop_duplicates(id_col)
    raw = per_donor[col].astype(str).str.strip().replace("90+", "90")
    return pd.to_numeric(raw, errors="coerce")


def patient_age_distribution(adata, id_col="individualID", col="age_death", bin_width=5, title=None):
    ages = _age_per_donor(adata, id_col, col).dropna()
    max_age = int(ages.max())
    bins = np.arange(0, max_age + bin_width, bin_width)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(ages.values, bins=bins, edgecolor="black", linewidth=0.5)
    ax.set_xlabel(f"{col} (bins of {bin_width}; 90+ → 90)")
    ax.set_ylabel("number of donors")
    ax.set_title(title or f"Age distribution ({len(ages)} donors)")
    fig.tight_layout()
    plt.show()


def pmi_distribution(adata, col="pmi", bin_width=1, title=None):
    """Cell-level PMI histogram (clinical PMI broadcast to each cell from its donor)."""
    pmi = pd.to_numeric(adata.obs[col], errors="coerce")
    n_total = len(pmi)
    pmi = pmi.dropna()
    print(f"cells with PMI: {len(pmi):,} / {n_total:,} total")

    max_pmi = float(pmi.max())
    bins = np.arange(0, max_pmi + bin_width, bin_width)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(pmi.values, bins=bins, edgecolor="black", linewidth=0.5)
    ax.set_xlabel(f"{col} (bins of {bin_width})")
    ax.set_ylabel("number of cells")
    ax.set_title(title or f"PMI distribution ({len(pmi):,} cells)")
    fig.tight_layout()
    plt.show()


def create_correlation_plots(adata, col_names, age_col="age_death"):
    df = adata.obs[col_names].copy()
    if age_col in df.columns:
        df[age_col] = pd.to_numeric(df[age_col].astype(str).replace("90+", "90"), errors="coerce")
    corr = df.apply(lambda s: _coerce_numeric_series(s)).corr(method="spearman")
    sns.heatmap(corr, annot=True, cmap="vlag", center=0, square=True, linewidths=0.5)
    plt.tight_layout()
    plt.show()


def _prepare_batch_vs_numeric(adata, col1, col2, n_bins=10):
    """Cell-level; coerce and bin numeric col2 for categorical tests."""
    df = adata.obs[[col1, col2]].copy()
    numeric = _coerce_numeric_series(df[col2])
    if numeric.notna().sum() > 0:
        df[col2] = numeric
        df = df.dropna(subset=[col1, col2])
        df[col2] = pd.cut(df[col2], bins=n_bins, precision=1)
    else:
        df = df.dropna(subset=[col1, col2])
    return df


def batch_effects_test(adata, col1, col2, n_bins=10):
    df_cat = _prepare_batch_vs_numeric(adata, col1, col2, n_bins)
    ct_norm = pd.crosstab(df_cat[col1], df_cat[col2], normalize="index")
    plt.figure(figsize=(10, 6))
    sns.heatmap(ct_norm, annot=False, fmt=".2f", cmap="Blues",
                cbar_kws={"label": "Proportion within row"})
    plt.xlabel(col2)
    plt.ylabel(col1)
    plt.xticks(rotation=45, ha="right")
    plt.title(f"Association between {col1} and {col2}")
    plt.tight_layout()
    plt.show()


def cramer_v_test(adata, col1, col2, n_bins=10):
    df_cat = _prepare_batch_vs_numeric(adata, col1, col2, n_bins)
    confusion_matrix = pd.crosstab(df_cat[col1], df_cat[col2])
    chi2, p, dof, expected = chi2_contingency(confusion_matrix)
    n = confusion_matrix.to_numpy().sum()
    r, k = confusion_matrix.shape
    phi2 = chi2 / n
    cramer_v = np.sqrt(phi2 / min(r - 1, k - 1))
    print(f"Cramer's V: {cramer_v:.4f}, p-value: {p:.4f}")


def _relabel_values(series, vmap):
    """Map numeric codes to readable labels; keep unknown codes, preserve NaN."""
    s = _coerce_numeric_series(series)
    return [vmap.get(int(v), v) if pd.notna(v) else v for v in s]


def _save_table_image(table, title, save_path):
    """Render a TableOne as a monospaced-table PNG."""
    txt = table.tabulate(tablefmt="grid")
    body = f"{title}\n\n{txt}"
    lines = body.split("\n")
    w = max(len(l) for l in lines)
    fig = plt.figure(figsize=(max(6, w * 0.085), max(2, len(lines) * 0.2)))
    fig.text(0.01, 0.99, body, family="monospace", fontsize=8.5, va="top", ha="left")
    plt.axis("off")
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def donor_table1(adata, groupby, id_col="individualID", continuous=None, categorical=None,
                 row_labels=None, value_labels=None, group_caption=None,
                 pval=True, save_path=None):
    """Donor-level Table 1 stratified by `groupby` (e.g. Braak or CERAD).

    One row per donor (deduplicated on id_col): covariates are rows, groups are columns.
    Continuous vars -> median [IQR] (nonnormal); categorical vars -> n (%).
    Column names / labels are dataset-specific (see DATASET_CONFIG); pass them in via
    continuous/categorical/row_labels/value_labels/group_caption.
    """
    continuous = list(continuous or [])
    categorical = list(categorical or [])
    row_labels = row_labels or {}
    value_labels = value_labels or {}
    group_caption = group_caption or {}

    # one row per donor
    df = adata.obs.drop_duplicates(id_col).copy()

    # groupby: use integer codes when the column is (mostly) numeric (ROSMAP braaksc/
    # ceradsc), else keep the string categories (SEA-AD 'Braak', 'CERAD score').
    gb_numeric = _coerce_numeric_series(df[groupby])
    df = df[gb_numeric.notna() | df[groupby].notna()].copy()
    if gb_numeric.notna().mean() > 0.5:
        df = df[gb_numeric.reindex(df.index).notna()].copy()
        df[groupby] = gb_numeric.reindex(df.index).astype(int).astype(str)
    else:
        df = df[df[groupby].notna()].copy()
        df[groupby] = df[groupby].astype(str)

    # never place the stratifier in the rows; keep only present columns
    continuous = [c for c in continuous if c != groupby and c in df.columns]
    categorical = [c for c in categorical if c != groupby and c in df.columns]

    for c in continuous:
        df[c] = _coerce_numeric_series(df[c])
    for c in categorical:
        if c in value_labels:
            df[c] = _relabel_values(df[c], value_labels[c])

    table = TableOne(
        df,
        columns=continuous + categorical,
        categorical=categorical,
        nonnormal=continuous,
        groupby=groupby,
        rename=row_labels,
        pval=pval,
    )
    caption = group_caption.get(groupby, groupby)
    title = f"Table 1 (donor-level, n={len(df)}) by {caption}"
    print(f"\n=== {title} ===")
    print(table.tabulate(tablefmt="github"))
    if save_path:
        _save_table_image(table, title, save_path)
    return table


def _add_case_control_col(adata, cc):
    """Derive a two-level AD-vs-control column onto adata.obs per a case_control config.

    `cc["source"]` is the diagnosis column; `cc["mapping"]` maps its values to group
    labels (unmapped values -> NaN, i.e. donors excluded from that Table 1). Numeric
    sources (e.g. ROSMAP cogdx) are coerced before lookup. Returns the new column name,
    or None if the source column is absent.
    """
    src = cc["source"]
    if src not in adata.obs.columns:
        return None
    mapping = cc["mapping"]
    col = cc["label"]
    numeric_keys = all(isinstance(k, (int, float)) and not isinstance(k, bool) for k in mapping)
    if numeric_keys:
        vals = _coerce_numeric_series(adata.obs[src])
        mapped = [mapping.get(int(v)) if pd.notna(v) else None for v in vals]
    else:
        mapped = [mapping.get(str(v).strip()) if pd.notna(v) else None for v in adata.obs[src]]
    adata.obs[col] = mapped
    return col


def dataset_table1(dataset, cell_type=None, groupby=None, save_dir="images"):
    """Build Table 1(s) for a dataset using its configured columns/labels.

    groupby defaults to the dataset's Braak (and CERAD, if any) columns, plus a derived
    AD-vs-control grouping when the dataset config defines `case_control`.
    Loads obs only (backed), so it's cheap even for the large SEA-AD file.
    """
    cfg = _cfg(dataset, cell_type)
    adata = load_dataset(dataset, cell_type, backed=True)
    default_gbs = cfg.get("stratifiers") or [c for c in (cfg.get("braak_col"), cfg.get("cerad_col")) if c]
    default_gbs = list(default_gbs)
    group_caption = dict(cfg.get("group_caption", {}))

    # Derive the AD-vs-control column (if configured) and fold it into the groupings.
    cc = cfg.get("case_control")
    if cc:
        cc_col = _add_case_control_col(adata, cc)
        if cc_col:
            group_caption[cc_col] = cc.get("caption", cc_col)
            if cc_col not in default_gbs:
                default_gbs.append(cc_col)
        else:
            print(f"  skip case/control: source '{cc['source']}' not in {dataset} obs")

    groupbys = [groupby] if groupby else default_gbs
    tag = f"{dataset}_{cell_type or cfg['cell_types'][0]}".lower()
    tables = {}
    for gb in groupbys:
        if gb not in adata.obs.columns:
            print(f"  skip {gb}: not in {dataset} obs")
            continue
        out = os.path.join(save_dir, f"{tag}_table1_{gb.replace(' ', '_')}.png")
        tables[gb] = donor_table1(
            adata, gb, id_col=cfg["id_col"],
            continuous=cfg["table1_continuous"], categorical=cfg["table1_categorical"],
            row_labels=cfg["row_labels"], value_labels=cfg["value_labels"],
            group_caption=group_caption, save_path=out,
        )
    return tables


def _to_numeric_ordinal(series, ordinal_map=None):
    """Convert a column to numbers for correlation.

    Uses an ordinal map for string stages (e.g. Braak/CERAD); otherwise coerces
    numerically ('90+' -> 90, 'unk'/'Reference' -> NaN).
    """
    if ordinal_map:
        mapped = series.astype(str).str.strip().map(ordinal_map)
        return mapped.where(mapped.notna(), _coerce_numeric_series(series))
    return _coerce_numeric_series(series)


def dataset_correlation(dataset, cell_type=None, method="spearman", save_dir="images"):
    """Correlation heatmap of a dataset's key clinical/QC variables (donor-level).

    Columns come from cfg['corr_cols']; string stages are ordinalized via
    cfg['ordinal_maps'] so Braak/CERAD/Thal correlate sensibly.
    """
    cfg = _cfg(dataset, cell_type)
    adata = load_dataset(dataset, cell_type, backed=True)
    df = adata.obs.drop_duplicates(cfg["id_col"]).copy()

    omaps = cfg.get("ordinal_maps", {})
    cols = [c for c in cfg.get("corr_cols", []) if c in df.columns]
    num = pd.DataFrame({c: _to_numeric_ordinal(df[c], omaps.get(c)) for c in cols})
    num = num.dropna(axis=1, how="all").rename(columns=cfg.get("row_labels", {}))

    corr = num.corr(method=method)
    fig, ax = plt.subplots(figsize=(1.15 * len(corr.columns) + 2, 1.0 * len(corr.columns) + 2))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0, vmin=-1, vmax=1,
                square=True, linewidths=0.5, cbar_kws={"label": f"{method} r"}, ax=ax)
    ct = cell_type or cfg["cell_types"][0]
    ax.set_title(f"{dataset} {ct}: key-variable {method} correlation (n={len(df)})")
    fig.tight_layout()
    _finish_fig(fig, os.path.join(save_dir, f"{dataset}_{ct}_correlation.png".lower()))
    return corr


def _derive_race(df):
    """Collapse SEA-AD's split 'Race (choice=X)' columns into a single 'Race' column."""
    choice = [c for c in df.columns if c.startswith("Race (choice=")]
    if not choice or "Race" in df.columns:
        return df
    df = df.copy()
    neg = {"unchecked", "no", "0", "false", "nan", "", "0.0"}

    def pick(row):
        hits = [c.split("choice=")[1].rstrip(")").strip() for c in choice
                if str(row[c]).strip().lower() not in neg]
        return hits[0] if len(hits) == 1 else ("Multiple" if hits else "Unknown")

    df["Race"] = df[choice].apply(pick, axis=1)
    return df


def dataset_distributions(dataset, cell_type=None, bins=20, save_dir="images"):
    """Donor-level distribution panel: histograms for continuous vars, bar charts for
    categorical vars (from cfg dist_continuous / dist_categorical)."""
    cfg = _cfg(dataset, cell_type)
    df = _derive_race(load_dataset(dataset, cell_type, backed=True).obs.drop_duplicates(cfg["id_col"]).copy())
    labels, omaps = cfg.get("row_labels", {}), cfg.get("ordinal_maps", {})
    cont = [c for c in cfg.get("dist_continuous", []) if c in df.columns]
    cats = [c for c in cfg.get("dist_categorical", []) if c in df.columns]
    panels = [("c", c) for c in cont] + [("k", c) for c in cats]

    ncols = 4
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, (kind, c) in zip(axes, panels):
        if kind == "c":
            v = _coerce_numeric_series(df[c]).dropna()
            ax.hist(v, bins=bins, edgecolor="black", linewidth=0.5)
            if len(v):
                ax.axvline(v.median(), color="red", ls="--", lw=1)
        else:
            vc = df[c].astype(str).replace({"nan": "NA"}).value_counts()
            omap = omaps.get(c)
            if omap:
                vc = vc.reindex(sorted(vc.index, key=lambda x: omap.get(x, 999)))
            ax.bar(range(len(vc)), vc.values, edgecolor="black", linewidth=0.5)
            ax.set_xticks(range(len(vc)))
            ax.set_xticklabels(vc.index, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("donors")
        ax.set_title(labels.get(c, c), fontsize=10)

    for ax in axes[len(panels):]:
        ax.set_visible(False)

    ct = cell_type or cfg["cell_types"][0]
    fig.suptitle(f"{dataset} {ct}: donor distributions (n={len(df)})", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    _finish_fig(fig, os.path.join(save_dir, f"{dataset}_{ct}_distributions.png".lower()))
    return df


def dataset_seq_stats(dataset, cell_type=None, out_dir="data", plot=True, save_dir="images"):
    """Compute + save per-donor sequencing stats for a dataset (needs a count matrix).

    Not available for Atlas (counts_layer is None -> raw counts are in per-region MTX).
    """
    cfg = _cfg(dataset, cell_type)
    if cfg["loader"] != "h5ad":
        raise ValueError(f"{dataset}: sequencing stats need a count matrix; "
                         f"not available via this loader ({cfg['loader']}).")
    ct = cell_type or cfg["cell_types"][0]
    adata = load_dataset(dataset, ct, backed=False)
    stats = donor_sequencing_stats(adata, id_col=cfg["id_col"], layer=cfg["counts_layer"])
    tag = f"{dataset}_{ct}".lower()
    csv = os.path.join(out_dir, f"{tag}_donor_seq_stats.csv")
    os.makedirs(out_dir, exist_ok=True)
    stats.to_csv(csv)
    print(f"Saved: {csv}  ({len(stats)} donors)")
    if plot:
        plot_seq_stats_histograms(stats, title=f"{dataset} {ct} donor seq stats",
                                  save_path=os.path.join(save_dir, f"{tag}_donor_seq_stats_hist.png"))
    return stats


def run_dataset(dataset, cell_types=None, seq_stats=True, out_dir="images"):
    """Run the standard EDA (Table 1 + optional sequencing stats) for a dataset."""
    cell_types = cell_types or DATASET_CONFIG[dataset]["cell_types"]
    print(f"\n########## {dataset} ##########")
    for ct in cell_types:
        dataset_table1(dataset, ct, save_dir=out_dir)
        dataset_correlation(dataset, ct, save_dir=out_dir)
        if seq_stats and _cfg(dataset, ct)["loader"] == "h5ad":
            dataset_seq_stats(dataset, ct, save_dir=out_dir)


def main():
    import argparse
    p = argparse.ArgumentParser(description="Cohort EDA: donor sequencing stats + Table 1.")
    p.add_argument("--dataset", default="ROSMAP", choices=list(DATASET_CONFIG))
    p.add_argument("--cell-types", nargs="*", default=None,
                   help="Subset of cell types (default: all for the dataset).")
    p.add_argument("--skip-seq-stats", action="store_true",
                   help="Only build Table 1 (skip the heavy count-matrix pass).")
    p.add_argument("--out-dir", default="images")
    args = p.parse_args()
    run_dataset(args.dataset, cell_types=args.cell_types,
                seq_stats=not args.skip_seq_stats, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
