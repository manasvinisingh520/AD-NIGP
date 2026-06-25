# load atlas anndata object and print sample of adata.obs
import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency

microglia_adata = ad.read_h5ad('data/ROSMAP_Data/Microglia.h5ad')
astrocytes_adata = ad.read_h5ad('data/ROSMAP_Data/Astrocytes.h5ad')

def donor_level_counts(adata, column, id_col="individualID"):
    per_donor = adata.obs.drop_duplicates(id_col)
    return per_donor[column].value_counts()

def general_counts(microglia_adata, astrocytes_adata, id_col="individualID"):
    all_columns = input("Enter all columns to print: ")
    all_columns = [c.strip() for c in all_columns.split(',') if c.strip()]

    for column in all_columns:
        print(f"{column} (Microglia, per {id_col}):\n")
        print(donor_level_counts(microglia_adata, column))
        print()

    print("=" * 100)

    for column in all_columns:
        print(f"{column} (Astrocytes, per {id_col}):\n")
        print(donor_level_counts(astrocytes_adata, column))
        print()


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


def _age_per_donor(adata, id_col="individualID", col="age_at_visit_max"):
    per_donor = adata.obs.drop_duplicates(id_col)
    raw = per_donor[col].astype(str).str.strip()
    raw = raw.replace("90+", "90")
    return pd.to_numeric(raw, errors="coerce")


def patient_age_distribution(adata, id_col="individualID", bin_width=5, title=None):
    ages = _age_per_donor(adata, id_col).dropna()
    max_age = int(ages.max())
    bins = np.arange(0, max_age + bin_width, bin_width)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(ages.values, bins=bins, edgecolor="black", linewidth=0.5)
    ax.set_xlabel(f"age at first AD diagnosis (bins of {bin_width}; 90+ → 90)")
    ax.set_ylabel("number of donors")
    ax.set_title(title or f"Age distribution ({len(ages)} donors)")
    fig.tight_layout()
    plt.show()

def pmi_distribution(adata, bin_width=1, title=None):
    """Cell-level PMI histogram (clinical PMI broadcast to each cell from its donor)."""
    pmi = pd.to_numeric(adata.obs["pmi"], errors="coerce")
    n_total = len(pmi)
    pmi = pmi.dropna()
    n_with_pmi = len(pmi)
    print(f"cells with PMI: {n_with_pmi:,} / {n_total:,} total")

    max_pmi = float(pmi.max())
    bins = np.arange(0, max_pmi + bin_width, bin_width)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(pmi.values, bins=bins, edgecolor="black", linewidth=0.5)
    ax.set_xlabel(f"PMI (bins of {bin_width})")
    ax.set_ylabel("number of cells")
    ax.set_title(title or f"PMI distribution ({n_with_pmi:,} cells)")
    fig.tight_layout()
    plt.show()

def create_correlation_plots(adata, col_names):
    #put age_death in numbers (90+ -> 90)
    adata.obs['age_death'] = adata.obs['age_death'].replace('90+', '90')
    adata.obs['age_death'] = pd.to_numeric(adata.obs['age_death'])
    corr = adata.obs[col_names].corr(method='spearman')
    sns.heatmap(corr,
                annot=True,
                cmap="vlag",
                center=0,
                square=True,
                linewidths=0.5)
    plt.tight_layout()
    plt.show()


def _coerce_numeric_series(s):
    return pd.to_numeric(
        s.astype(str).str.strip().replace("90+", "90"),
        errors="coerce",
    )


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
    sns.heatmap(
        ct_norm,
        annot=False,
        fmt=".2f",
        cmap="Blues",
        cbar_kws={"label": "Proportion within row"},
    )
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

batch_effects_test(microglia_adata, 'batch', 'age_death')
cramer_v_test(microglia_adata, 'batch', 'age_death')
batch_effects_test(astrocytes_adata, 'batch', 'age_death')
cramer_v_test(astrocytes_adata, 'batch', 'age_death')