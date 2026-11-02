"""Differential accessibility (DA) of microglia ATAC peaks across ADNC stages with PyDESeq2.

ATAC counts per peak are treated like RNA-seq counts per gene: one pseudobulk sample per donor
from ATAC-only (ATX) nuclei (src/atac_pseudobulk.py --method ATX), then DESeq2 as in the RNA DGE
(src/dge2.py):
    ADNC levels 1=Not AD, 2=Low, 3=Intermediate, 4=High  (Reference donors excluded)
    contrasts   1v2, 1v3, 1v4, 2v3, 2v4, 3v4;  log2FC = first group relative to second
                (so in "1_vs_4", positive = more open in Not AD, negative = more open in High)
    significant padj < 0.05 and |log2FC| > 0.5

Donor QC (thresholds shared with src/atac_batch_check.py): drop donors with < 250k total
microglia counts or fraction of fragments in peaks (FRiP) < 0.2.

Designs (covariates chosen from the PCA check in src/atac_batch_check.py):
    --design nobatch : ~ frac_in_peaks + Sex + condition
    --design batch   : ~ batch_vendor_name + frac_in_peaks + Sex + condition   (closer to the RNA DGE)
frac_in_peaks is standardised (mean 0, sd 1) so the GLM fit is numerically stable.

--grouping picks the condition (as the RNA DGE): ADNC (4 levels, default), binADNC (Not AD + Low
vs Intermediate + High) or binThal (Thal 0-2 vs Thal 3-5). Binary groupings have one contrast,
1_vs_2 (low vs high); positive log2FC = more open in the low group.

--prefilter switches on the ATAC-specific peak filter: keep peaks with >= 10 counts in >= N
donors, N = smallest ADNC group (DESeq2 vignette rule). Off by default, matching the RNA DGE.

Run (Windows sf2026 env):
    python src/atac_deseq2.py --design nobatch
    python src/atac_deseq2.py --design batch --prefilter --grouping binThal

Output: results/atac_da/da_results_SEAAD_Microglia_MTG_ATAC_ATX_<design>_<filter>_<g1>_vs_<g2>.csv
        DESeq2 columns + nearest gene annotation (see src/peak_annotation.py)
"""
import argparse
import os

import anndata as ad
import numpy as np
import pandas as pd
from pydeseq2.dds import DeseqDataSet
from pydeseq2.default_inference import DefaultInference
from pydeseq2.ds import DeseqStats

from atac_batch_check import MIN_FRAC_IN_PEAKS, MIN_TOTAL_COUNTS
from peak_annotation import annotate

PSEUDOBULK = "data/SEAAD_ATAC/microglia_pseudobulk_counts_ATX.h5ad"
OUT_DIR = "results/atac_da"
ADNC_COL = "Overall AD neuropathological Change"
FRIP_COL = "mean_ATAC_Fraction_of_high_quality_fragments_overlapping_peaks"
AD_LEVEL_MAP = {"Not AD": "1", "Low": "2", "Intermediate": "3", "High": "4"}  # as in dge2.py
CONTRASTS = [("1", "2"), ("1", "3"), ("1", "4"), ("2", "3"), ("2", "4"), ("3", "4")]
BINARY_CONTRASTS = [("1", "2")]
# --grouping: how donors are split into conditions (all as in dge2.py)
#   ADNC    : 4 levels, 6 pairwise contrasts
#   binADNC : 1 = low (Not AD/Low), 2 = high (Intermediate/High)
#   binThal : 1 = low (Thal 0-2), 2 = high (Thal 3-5)   (dge2.py default threshold 3)
GROUPINGS = {
    "ADNC":    (ADNC_COL, AD_LEVEL_MAP, CONTRASTS, ""),
    "binADNC": (ADNC_COL, {"Not AD": "1", "Low": "1", "Intermediate": "2", "High": "2"},
                BINARY_CONTRASTS, "_binADNC"),
    "binThal": ("Thal", {f"Thal {i}": ("1" if i < 3 else "2") for i in range(6)},
                BINARY_CONTRASTS, "_binThal"),
}
DESIGNS = {"nobatch": "~frac_in_peaks + Sex + condition",
           "batch":   "~batch_vendor_name + frac_in_peaks + Sex + condition"}  # condition last
PADJ, LFC = 0.05, 0.5


def load(prefilter, grouping="ADNC"):
    """Donor x peak counts + metadata after donor QC; optionally peak-filtered.

    Reference donors are always excluded (they have no ADNC/Thal stage), so every grouping
    uses the same QC-passing donors.
    """
    source_col, level_map, _, _ = GROUPINGS[grouping]
    a = ad.read_h5ad(PSEUDOBULK)
    a = a[a.obs[ADNC_COL].isin(AD_LEVEL_MAP)].copy()
    totals = np.asarray(a.X.sum(1)).ravel()
    frip = a.obs[FRIP_COL].astype(float).to_numpy()
    ok = (totals >= MIN_TOTAL_COUNTS) & (frip >= MIN_FRAC_IN_PEAKS)
    print(f"donor QC: removed {(~ok).sum()} donors (< {MIN_TOTAL_COUNTS:,} counts or FRiP < "
          f"{MIN_FRAC_IN_PEAKS}): {', '.join(a.obs_names[~ok])}")
    a = a[ok].copy()

    counts = pd.DataFrame(a.X.toarray().astype(int), index=a.obs_names, columns=a.var_names)
    frip = a.obs[FRIP_COL].astype(float)
    # formula-friendly column names (no spaces)
    meta = pd.DataFrame({"condition": a.obs[source_col].astype(str).map(level_map),
                         "batch_vendor_name": a.obs["batch_vendor_name"].astype(str),
                         "Sex": a.obs["Sex"].astype(str),
                         "frac_in_peaks": (frip - frip.mean()) / frip.std()},
                        index=a.obs_names)
    if meta.condition.isna().any():
        raise ValueError(f"unmapped {source_col} values: {a.obs.loc[meta.condition.isna(), source_col].unique()}")
    print(f"{len(meta)} donors x {counts.shape[1]:,} peaks")
    print(f"donors per {grouping} level:", meta.condition.value_counts().sort_index().to_dict())

    # a peak with zero counts in every donor carries no information and breaks the fit
    counts = counts.loc[:, counts.sum(axis=0) > 0]
    if prefilter:
        n_min = meta.condition.value_counts().min()
        counts = counts.loc[:, (counts >= 10).sum(axis=0) >= n_min]
        print(f"prefilter: >= 10 counts in >= {n_min} donors -> {counts.shape[1]:,} peaks kept")
    else:
        print(f"no prefilter (only all-zero peaks dropped): {counts.shape[1]:,} peaks tested")
    return counts, meta


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--design", choices=list(DESIGNS), required=True)
    ap.add_argument("--prefilter", action="store_true", help="apply the >=10 counts in >=N donors filter")
    ap.add_argument("--grouping", choices=list(GROUPINGS), default="ADNC",
                    help="ADNC (4 levels), binADNC or binThal (low vs high), as the RNA DGE")
    ap.add_argument("--n-cpus", type=int, default=8)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    _, _, contrasts, group_tag = GROUPINGS[args.grouping]
    tag = f"ATX_{args.design}_{'prefilter10' if args.prefilter else 'nofilter'}{group_tag}"

    counts, meta = load(args.prefilter, args.grouping)

    # 1. fit: size factors (depth normalisation), dispersions (noise per peak), GLM per peak
    design = DESIGNS[args.design]
    print(f"DESeq2 design: {design}")
    inference = DefaultInference(n_cpus=args.n_cpus)
    dds = DeseqDataSet(counts=counts, metadata=meta, design=design,
                       inference=inference, n_cpus=args.n_cpus)
    dds.deseq2()

    # 2. one Wald test per contrast; padj uses DESeq2's independent filtering (default on)
    #    each peak is annotated with its nearest gene (src/peak_annotation.py)
    annot = annotate(counts.columns)
    lines = []
    for g1, g2 in contrasts:
        st = DeseqStats(dds, contrast=["condition", g1, g2], inference=inference, n_cpus=args.n_cpus)
        st.summary()
        res = st.results_df.join(annot).sort_values("padj")
        res.to_csv(f"{OUT_DIR}/da_results_SEAAD_Microglia_MTG_ATAC_{tag}_{g1}_vs_{g2}.csv")
        sig = res[(res.padj < PADJ) & (res.log2FoldChange.abs() > LFC)]
        lines.append(f"{g1}_vs_{g2}: tested {res.padj.notna().sum():,} | significant {len(sig):,} "
                     f"(more open in {g1}: {(sig.log2FoldChange > 0).sum()}, "
                     f"more open in {g2}: {(sig.log2FoldChange < 0).sum()})")

    print(f"\n==== summary ({tag}; design {design}; significant = padj < {PADJ} and |log2FC| > {LFC}) ====")
    print("\n".join(lines))
    print(f"\nsaved per-contrast CSVs in {OUT_DIR}/")


if __name__ == "__main__":
    main()
