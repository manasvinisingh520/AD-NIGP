"""Which covariates belong in the ATAC DESeq2 design? PCA + variance explained per variable.

Data: donor-level microglia pseudobulk, ATAC-only (ATX) nuclei (src/atac_pseudobulk.py --method ATX),
Reference donors excluded.

Steps
  0. Donor QC: drop donors with < MIN_TOTAL_COUNTS total microglia counts or FRiP < MIN_FRAC_IN_PEAKS.
  1. Keep microglia peaks (>= 10 counts in >= N donors, N = smallest ADNC group) - PCA on 200k
     mostly-empty peaks is noise.
  2. Normalise: log2 counts-per-million (removes sequencing-depth differences between donors).
  3. PCA on the 5,000 most variable peaks -> the main axes along which donors differ.
  4. For each candidate variable, R^2 = how much of each PC it explains. Summarised over the top
     10 PCs weighted by each PC's share of variance (PCR, as in src/batch_effect_metric.py).
  5. Permutation baseline: shuffle the variable 1,000 times and recompute. A variable with many
     categories (34 batches, 84 donors) explains a lot by chance alone, so only R^2 above the
     shuffled values counts. Reported as excess R^2 = observed - mean shuffled, and a p-value.
  6. Confounding: is each variable associated with ADNC? (if so, adjusting for it also removes
     part of the AD signal)

Run (Windows sf2026 env):  python src/atac_batch_check.py
                           python src/atac_batch_check.py --batch-vs-binthal   (association only, prints)
Outputs: results/atac_da/batch_check_pca.png, batch_check_variance_explained.csv
"""
import os

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

PSEUDOBULK = "data/SEAAD_ATAC/microglia_pseudobulk_counts_ATX.h5ad"
OUT_DIR = "results/atac_da"
ADNC_COL = "Overall AD neuropathological Change"
ADNC_ORDER = ["Not AD", "Low", "Intermediate", "High"]
N_TOP_PEAKS, N_PCS, N_PERM = 5000, 10, 1000
# donor QC: below this many total microglia counts a donor's profile is too sparse to use
# (the first PCA showed such donors drive PC1/PC2, and most are High ADNC)
MIN_TOTAL_COUNTS = 250_000
# donor QC: fraction of fragments in peaks (FRiP) below 0.2 = poor signal-to-noise
# (ENCODE ATAC-seq minimum acceptable FRiP); mean over the donor's microglia nuclei
MIN_FRAC_IN_PEAKS = 0.2
RNG = np.random.default_rng(0)

# candidate variables: (column, kind, technical or biological)
VARIABLES = {
    "batch_vendor_name":   ("categorical", "technical"),
    "log10_total_counts":  ("continuous",  "technical"),
    "n_microglia":         ("continuous",  "technical"),
    "TSS_enrichment":      ("continuous",  "technical"),
    "frac_in_peaks":       ("continuous",  "technical"),
    "PMI":                 ("continuous",  "technical"),
    "ADNC":                ("categorical", "biological (the question)"),
    "Sex":                 ("categorical", "biological"),
    "Age":                 ("continuous",  "biological"),
}


def load():
    a = ad.read_h5ad(PSEUDOBULK)
    a = a[a.obs[ADNC_COL].isin(ADNC_ORDER)].copy()
    totals = np.asarray(a.X.sum(1)).ravel()
    frip = a.obs.mean_ATAC_Fraction_of_high_quality_fragments_overlapping_peaks.astype(float).to_numpy()
    for label, bad in [(f"< {MIN_TOTAL_COUNTS:,} total counts", totals < MIN_TOTAL_COUNTS),
                       (f"FRiP < {MIN_FRAC_IN_PEAKS} (and enough counts)",
                        (frip < MIN_FRAC_IN_PEAKS) & (totals >= MIN_TOTAL_COUNTS))]:
        print(f"donor QC: removing {bad.sum()} donors with {label}: "
              + ", ".join(f"{d} ({a.obs.loc[d, ADNC_COL]})" for d in a.obs_names[bad]))
    a = a[(totals >= MIN_TOTAL_COUNTS) & (frip >= MIN_FRAC_IN_PEAKS)].copy()
    counts = a.X.toarray().astype(float)
    o = a.obs
    meta = pd.DataFrame(index=a.obs_names)
    meta["batch_vendor_name"] = o.batch_vendor_name.astype(str)
    meta["log10_total_counts"] = np.log10(counts.sum(1))
    meta["n_microglia"] = o.n_microglia.astype(float)
    meta["TSS_enrichment"] = o.mean_ATAC_TSS_enrichment_score.astype(float)
    meta["frac_in_peaks"] = o.mean_ATAC_Fraction_of_high_quality_fragments_overlapping_peaks.astype(float)
    meta["PMI"] = pd.to_numeric(o.PMI, errors="coerce")
    meta["ADNC"] = pd.Categorical(o[ADNC_COL].astype(str), ADNC_ORDER, ordered=True)
    # binThal as in the RNA DGE (dge2.py, threshold 3): low = Thal 0-2, high = Thal 3-5
    thal = o["Thal"].astype(str).str.extract(r"(\d)")[0].astype(int)
    meta["binThal"] = np.where(thal >= 3, "high (Thal 3-5)", "low (Thal 0-2)")
    meta["Sex"] = o.Sex.astype(str)
    # SEA-AD codes age 90+ as "90+"; treat as 90
    meta["Age"] = pd.to_numeric(o["Age at Death"].astype(str).str.replace("+", "", regex=False),
                                errors="coerce")
    return counts, meta


def pca_scores(counts, n_min):
    """log2-CPM of microglia peaks -> PCA scores + variance share of each PC.

    n_min: a peak is kept if >= 10 counts in >= n_min donors (n_min = smallest ADNC group).
    """
    keep = (counts >= 10).sum(0) >= n_min
    x = counts[:, keep]
    logcpm = np.log2(x / x.sum(1, keepdims=True) * 1e6 + 1)
    top = np.argsort(logcpm.var(0))[::-1][:N_TOP_PEAKS]
    pca = PCA(n_components=N_PCS).fit(logcpm[:, top])
    print(f"PCA on {len(top):,} most variable of {keep.sum():,} microglia peaks; "
          f"variance explained by PC1-{N_PCS}: {np.round(pca.explained_variance_ratio_ * 100, 1)}")
    return pca.transform(logcpm[:, top]), pca.explained_variance_ratio_


def design_matrix(values, kind):
    """Variable -> regression inputs (one-hot for categories; 1 column for numbers)."""
    if kind == "categorical":
        return pd.get_dummies(pd.Series(values).astype(str), drop_first=True).to_numpy(float)
    return np.asarray(values, float).reshape(-1, 1)


def weighted_r2(scores, var_ratio, X):
    """R^2 of each PC on X, averaged with PC variance shares as weights."""
    r2 = np.array([LinearRegression().fit(X, scores[:, j]).score(X, scores[:, j])
                   for j in range(scores.shape[1])])
    return float((r2 * var_ratio).sum() / var_ratio.sum()), r2


def adnc_association(meta, col, kind):
    """p-value for association of a variable with ADNC (confounding check)."""
    m = meta[[col, "ADNC"]].dropna()
    if kind == "categorical":
        return stats.chi2_contingency(pd.crosstab(m[col], m.ADNC))[1]
    return stats.kruskal(*[g[col] for _, g in m.groupby("ADNC", observed=True)])[1]


def batch_vs_condition(meta, col, n_perm=10_000):
    """Association between library batch and a condition (e.g. binThal), both categorical.

    Cramer's V: 0 = batch says nothing about the condition, 1 = batch fully determines it.
    The p-value is by permutation (shuffle the condition across donors), because most batches
    hold only 1-4 donors and the chi-square approximation is unreliable for such sparse tables.
    """
    tab = pd.crosstab(meta.batch_vendor_name, meta[col])

    def cramers_v(t):
        chi2 = stats.chi2_contingency(t, correction=False)[0]
        return np.sqrt(chi2 / (t.to_numpy().sum() * (min(t.shape) - 1)))

    v = cramers_v(tab)
    perm = np.array([cramers_v(pd.crosstab(meta.batch_vendor_name.to_numpy(),
                                           RNG.permutation(meta[col].to_numpy())))
                     for _ in range(n_perm)])
    p = (1 + (perm >= v).sum()) / (1 + n_perm)
    pure = (tab > 0).sum(1) == 1
    print(f"\n=== batch_vendor_name vs {col} ({len(meta)} donors, {len(tab)} batches)")
    print(tab.assign(total=tab.sum(1)).sort_values("total", ascending=False).to_string())
    print(f"\nbatches containing only one {col} group: {pure.sum()} of {len(tab)} "
          f"({tab[pure].to_numpy().sum()} donors); single-donor batches: {(tab.sum(1) == 1).sum()}")
    print(f"Cramer's V = {v:.3f} (shuffled labels: mean {perm.mean():.3f}, 95th pct {np.percentile(perm, 95):.3f}) "
          f"| permutation p = {p:.4f}")
    return v, p


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    counts, meta = load()
    if args.batch_vs_binthal:
        batch_vs_condition(meta, "binThal")
        return
    print(f"{len(meta)} donors; ADNC: {meta.ADNC.value_counts().reindex(ADNC_ORDER).to_dict()}")
    scores, vr = pca_scores(counts, n_min=meta.ADNC.value_counts().min())

    rows, per_pc = [], {}
    for col, (kind, role) in VARIABLES.items():
        ok = meta[col].notna().to_numpy()       # a few donors may lack PMI/age
        X = design_matrix(meta.loc[ok, col], kind)
        obs_r2, r2_pc = weighted_r2(scores[ok], vr, X)
        perm = np.array([weighted_r2(scores[ok], vr, X[RNG.permutation(len(X))])[0]
                         for _ in range(N_PERM)])
        per_pc[col] = r2_pc
        rows.append(dict(variable=col, role=role, n_params=X.shape[1], R2=obs_r2,
                         R2_shuffled_mean=perm.mean(), excess_R2=obs_r2 - perm.mean(),
                         perm_p=(1 + (perm >= obs_r2).sum()) / (1 + N_PERM),
                         ADNC_assoc_p=np.nan if col == "ADNC" else adnc_association(meta, col, kind)))
    res = pd.DataFrame(rows).sort_values("excess_R2", ascending=False)
    res.to_csv(f"{OUT_DIR}/batch_check_variance_explained.csv", index=False)
    with pd.option_context("display.width", 200):
        print("\nVariance in top-10 PCs explained by each variable "
              "(excess_R2 = beyond what shuffled labels give; perm_p from 1,000 shuffles):")
        print(res.round(3).to_string(index=False))
        print("\nR^2 per PC:")
        print(pd.DataFrame(per_pc, index=[f"PC{i + 1}" for i in range(N_PCS)]).T.round(2).to_string())

    # plots: PC1 vs PC2 coloured by the main variables
    panels = ["ADNC", "batch_vendor_name", "log10_total_counts", "TSS_enrichment",
              "frac_in_peaks", "n_microglia"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5))
    for ax, col in zip(axes.ravel(), panels):
        v = meta[col]
        if VARIABLES[col][0] == "categorical":
            cats = list(v.cat.categories) if hasattr(v, "cat") else sorted(v.unique())
            cmap = plt.get_cmap("viridis" if col == "ADNC" else "tab20", len(cats))
            for i, c in enumerate(cats):
                m = (v == c).to_numpy()
                ax.scatter(scores[m, 0], scores[m, 1], s=28, color=cmap(i), label=c, edgecolor="k", lw=0.3)
            if len(cats) <= 6:
                ax.legend(fontsize=8, frameon=False)
        else:
            sc_ = ax.scatter(scores[:, 0], scores[:, 1], c=v, s=28, cmap="viridis", edgecolor="k", lw=0.3)
            fig.colorbar(sc_, ax=ax, shrink=0.8)
        r = res.set_index("variable").loc[col]
        ax.set_title(f"{col}\nR²={r.R2:.2f} (shuffled {r.R2_shuffled_mean:.2f}), perm p={r.perm_p:.3f}",
                     fontsize=10)
        ax.set_xlabel(f"PC1 ({vr[0] * 100:.1f}%)")
        ax.set_ylabel(f"PC2 ({vr[1] * 100:.1f}%)")
    fig.suptitle("SEA-AD MTG microglia ATAC (ATX nuclei), donor pseudobulk PCA", fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/batch_check_pca.png", dpi=130)
    print(f"\nsaved {OUT_DIR}/batch_check_pca.png and batch_check_variance_explained.csv")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--batch-vs-binthal", action="store_true",
                    help="only check the association between library batch and binThal (no PCA)")
    args = ap.parse_args()
    main()
