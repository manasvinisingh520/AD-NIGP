"""Gene-level differential accessibility: gene activity scores + PyDESeq2.

Peak-level DA (src/atac_deseq2.py) tests each peak on its own. Here peaks are pooled per gene:
  1. Gene region   : gene body + 2 kb upstream of the TSS (Signac/ArchR convention), protein-coding
                     genes from GENCODE v44. --region promoter uses TSS +/- 2 kb instead.
  2. Activity score: for each donor, the sum of the counts of all peaks overlapping the region.
                     A peak overlapping two gene regions counts toward both. Approximation: only
                     reads inside SEA-AD peaks are available (Signac/ArchR count all fragments).
  3. DESeq2        : donors x genes, same donor QC, covariates and contrasts as the peak-level
                     analysis, so results line up with the RNA DGE gene by gene.
Gene filter: >= 10 counts in >= N donors (N = smallest ADNC group), the DESeq2 vignette rule.

Run (Windows sf2026 env):
    python src/atac_gene_activity.py --design nobatch --grouping binThal

Output: results/atac_da/gene_activity_results_SEAAD_Microglia_MTG_ATAC_ATX_<region>_<design>[_binADNC|_binThal]_<g1>_vs_<g2>.csv
"""
import argparse
import os

import numpy as np
import pandas as pd
import scipy.sparse as sp
from pydeseq2.dds import DeseqDataSet
from pydeseq2.default_inference import DefaultInference
from pydeseq2.ds import DeseqStats

from atac_deseq2 import DESIGNS, GROUPINGS, LFC, OUT_DIR, PADJ, load
from peak_annotation import load_genes

UPSTREAM_BP = 2000


def gene_regions(region):
    """One region per protein-coding gene: gene body + 2 kb upstream, or TSS +/- 2 kb."""
    g = load_genes().drop_duplicates("ensembl")
    if region == "genebody":
        plus = g.strand == "+"
        g["r_start"] = np.where(plus, g.start - UPSTREAM_BP, g.start)
        g["r_end"] = np.where(plus, g.end, g.end + UPSTREAM_BP)
    else:  # promoter
        g["r_start"], g["r_end"] = g.tss - UPSTREAM_BP, g.tss + UPSTREAM_BP
    g["r_start"] = g.r_start.clip(lower=0)
    return g.reset_index(drop=True)


def peak_to_gene_matrix(peak_ids, genes):
    """Sparse peaks x genes 0/1 matrix: 1 where the peak overlaps the gene's region."""
    p = pd.Series(peak_ids).str.extract(r"^(chr\w+):(\d+)-(\d+)$")
    p.columns = ["chrom", "start", "end"]
    p[["start", "end"]] = p[["start", "end"]].astype(int)
    rows, cols = [], []
    for chrom, gp in genes.groupby("chrom"):
        pk = p[p.chrom == chrom].sort_values("start")
        if pk.empty:
            continue
        starts, ends, idx = pk.start.to_numpy(), pk.end.to_numpy(), pk.index.to_numpy()
        for gi, rs, re_ in zip(gp.index, gp.r_start, gp.r_end):
            # SEA-AD peaks do not overlap each other, so sorted starts imply sorted ends:
            # overlapping peaks are those ending after the region starts and starting before it ends
            lo, hi = np.searchsorted(ends, rs, side="right"), np.searchsorted(starts, re_, side="left")
            rows.extend(idx[lo:hi])
            cols.extend([gi] * (hi - lo))
    return sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(p), len(genes)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--design", choices=list(DESIGNS), required=True)
    ap.add_argument("--region", choices=["genebody", "promoter"], default="genebody")
    ap.add_argument("--grouping", choices=list(GROUPINGS), default="ADNC",
                    help="ADNC (4 levels), binADNC or binThal (low vs high), as the RNA DGE")
    ap.add_argument("--n-cpus", type=int, default=8)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    _, _, contrasts, group_tag = GROUPINGS[args.grouping]
    tag = f"ATX_{args.region}_{args.design}{group_tag}"

    # donor QC + metadata exactly as the peak-level analysis; all non-empty peaks
    peak_counts, meta = load(prefilter=False, grouping=args.grouping)

    # 1-2. pool peak counts into gene activity scores
    genes = gene_regions(args.region)
    M = peak_to_gene_matrix(peak_counts.columns, genes)
    # donors x peaks  @  peaks x genes  =  donors x genes (sum of each gene's peak counts)
    gene_counts = pd.DataFrame(np.asarray(M.T @ peak_counts.to_numpy().T).T.astype(int),
                               index=peak_counts.index, columns=genes.ensembl)
    n_peaks = pd.Series(np.asarray(M.sum(0)).ravel(), index=genes.ensembl, name="n_peaks")
    print(f"{args.region} regions: {(n_peaks > 0).sum():,} of {len(genes):,} protein-coding genes "
          f"contain >= 1 peak (median {n_peaks[n_peaks > 0].median():.0f} peaks per gene)")

    n_min = meta.condition.value_counts().min()
    gene_counts = gene_counts.loc[:, (gene_counts >= 10).sum(axis=0) >= n_min]
    print(f"gene filter: >= 10 counts in >= {n_min} donors -> {gene_counts.shape[1]:,} genes tested")

    # 3. DESeq2, same design as the peak-level analysis
    design = DESIGNS[args.design]
    print(f"DESeq2 design: {design}")
    inference = DefaultInference(n_cpus=args.n_cpus)
    dds = DeseqDataSet(counts=gene_counts, metadata=meta, design=design,
                       inference=inference, n_cpus=args.n_cpus)
    dds.deseq2()

    annot = genes.set_index("ensembl")[["gene", "chrom", "r_start", "r_end"]].join(n_peaks)
    lines = []
    for g1, g2 in contrasts:
        st = DeseqStats(dds, contrast=["condition", g1, g2], inference=inference, n_cpus=args.n_cpus)
        st.summary()
        res = annot.join(st.results_df, how="inner").sort_values("padj")
        res.index.name = "ensembl"
        res.to_csv(f"{OUT_DIR}/gene_activity_results_SEAAD_Microglia_MTG_ATAC_{tag}_{g1}_vs_{g2}.csv")
        sig = res[(res.padj < PADJ) & (res.log2FoldChange.abs() > LFC)]
        p = res.pvalue.dropna()
        lines.append(f"{g1}_vs_{g2}: tested {res.padj.notna().sum():,} | %p<0.05 {100 * (p < 0.05).mean():.1f}% "
                     f"| significant {len(sig)} (more open in {g1}: {(sig.log2FoldChange > 0).sum()}, "
                     f"more open in {g2}: {(sig.log2FoldChange < 0).sum()})"
                     + (f"\n    {', '.join(sig.gene)}" if 0 < len(sig) <= 40 else ""))

    print(f"\n==== summary ({tag}; significant = padj < {PADJ} and |log2FC| > {LFC}) ====")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
