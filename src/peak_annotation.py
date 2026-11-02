"""Annotate ATAC peaks with their nearest gene (GENCODE v44, protein-coding genes).

Peaks are DNA regions, not genes; this links each peak to the gene whose start (TSS) is closest
to the peak centre. The nearest gene is a best guess for which gene a peak regulates - promoter
peaks are reliable, distal peaks (enhancers) may control a different, farther gene.

Columns added
  nearest_gene, nearest_gene_ensembl : closest protein-coding TSS to the peak centre
  dist_to_TSS        : peak centre - TSS in bp, strand-aware: negative = upstream of the gene,
                       positive = downstream (into the gene body)
  location           : "promoter" (|dist| <= 2 kb), "inside a gene" or "between genes"
  overlapping_genes  : protein-coding genes whose body contains the peak centre (';'-joined)

Used by src/atac_deseq2.py. Run directly to annotate existing DA result CSVs in place:
    python src/peak_annotation.py results/atac_da/da_results_*ATX*.csv
"""
import glob
import gzip
import re
import sys

import numpy as np
import pandas as pd

GTF = "data/annotation/gencode.v44.basic.annotation.gtf.gz"
PROMOTER_BP = 2000
ANNOT_COLS = ["nearest_gene", "nearest_gene_ensembl", "dist_to_TSS", "location", "overlapping_genes"]


def load_genes(gtf=GTF):
    """Protein-coding genes: chrom, start, end (0-based), strand, TSS, symbol, Ensembl ID."""
    rows = []
    with gzip.open(gtf, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.split("\t")
            if f[2] != "gene" or 'gene_type "protein_coding"' not in f[8]:
                continue
            start, end = int(f[3]) - 1, int(f[4])
            rows.append((f[0], start, end, f[6], start if f[6] == "+" else end - 1,
                         re.search(r'gene_name "([^"]+)"', f[8]).group(1),
                         re.search(r'gene_id "([^".]+)', f[8]).group(1)))
    return pd.DataFrame(rows, columns=["chrom", "start", "end", "strand", "tss", "gene", "ensembl"])


def annotate(peak_ids, genes=None):
    """Annotation table for peak IDs like 'chr1:100-200', indexed by peak ID."""
    genes = load_genes() if genes is None else genes
    p = pd.Series(peak_ids, index=peak_ids).str.extract(r"^(chr\w+):(\d+)-(\d+)$")
    p.columns = ["chrom", "start", "end"]
    p[["start", "end"]] = p[["start", "end"]].astype(int)
    p["mid"] = (p.start + p.end) // 2

    parts = []
    for chrom, d in p.groupby("chrom"):
        g = genes[genes.chrom == chrom].sort_values("tss").reset_index(drop=True)
        if g.empty:
            parts.append(pd.DataFrame(index=d.index, columns=ANNOT_COLS))
            continue
        mid, tss = d.mid.to_numpy(), g.tss.to_numpy()
        # nearest TSS: compare the TSS just left and just right of each peak centre
        right = np.clip(np.searchsorted(tss, mid), 0, len(g) - 1)
        left = np.clip(right - 1, 0, len(g) - 1)
        near = np.where(np.abs(tss[left] - mid) <= np.abs(tss[right] - mid), left, right)
        ng = g.iloc[near]
        dist = (mid - ng.tss.to_numpy()) * np.where(ng.strand.to_numpy() == "+", 1, -1)
        # genes whose body contains the peak centre
        inside = (g.start.to_numpy()[None, :] <= mid[:, None]) & (g.end.to_numpy()[None, :] > mid[:, None])
        overlapping = [";".join(g.gene.to_numpy()[row]) for row in inside]
        location = np.where(np.abs(dist) <= PROMOTER_BP, "promoter",
                            np.where(inside.any(1), "inside a gene", "between genes"))
        parts.append(pd.DataFrame({"nearest_gene": ng.gene.to_numpy(),
                                   "nearest_gene_ensembl": ng.ensembl.to_numpy(),
                                   "dist_to_TSS": dist, "location": location,
                                   "overlapping_genes": overlapping}, index=d.index))
    return pd.concat(parts).reindex(peak_ids)


def annotate_csv(path, genes):
    """Add (or refresh) the annotation columns of a DA result CSV, in place."""
    res = pd.read_csv(path, index_col=0).drop(columns=ANNOT_COLS, errors="ignore")
    res.join(annotate(res.index, genes)).to_csv(path)


if __name__ == "__main__":
    paths = [p for arg in sys.argv[1:] for p in glob.glob(arg)]
    genes = load_genes()
    for p in paths:
        annotate_csv(p, genes)
        print("annotated", p)
