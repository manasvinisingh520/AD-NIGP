"""ATAC gene activity changes for specific gene sets (model genes, DGE consensus genes).

For each gene set, take the gene-level ATAC results (src/atac_gene_activity.py, binary low vs high,
--grouping binADNC or binThal) and line them up with the RNA DGE for the same grouping:
  - ATAC log2FC, p-value, and padj re-adjusted WITHIN the set (Benjamini-Hochberg). Testing ~50-170
    genes instead of 15,578 makes the correction far lighter - legitimate because the set was
    chosen before looking at the ATAC results.
  - Concordance: does accessibility move the same way as RNA expression? (binomial test vs 50%)
Sign convention for both: log2FC = low group relative to high; negative = higher in high group.

Gene-level ATAC covers protein-coding genes only, so lncRNAs/unnamed genes in a set are reported
as not measured.

Run (Windows sf2026 env):  python src/atac_gene_sets.py --grouping binThal --sets min7 --show-all
Prints results only; nothing is saved.
"""
import argparse

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

ATAC = "results/atac_da/gene_activity_results_SEAAD_Microglia_MTG_ATAC_ATX_genebody_{design}_{grouping}_1_vs_2.csv"
RNA = ("results/dge_final/dge_results_SEA_AD_Microglia_MTG_pseudobulk100_seed100_"
       "{grouping}_covbatch_vendor_name_1_vs_2.csv")
GENE_SETS = {
    "min6": ("ranked_lists_v2/SEAAD_Microglia_filtered_flatmlp_trial26/combined_ranked_genes_min6.csv",
             lambda p: pd.read_csv(p).set_index("gene")["gene_symbol"]),
    "min7": ("ranked_lists_v2/SEAAD_Microglia_filtered_flatmlp_trial26/combined_ranked_genes_min7.csv",
             lambda p: pd.read_csv(p).set_index("gene")["gene_symbol"]),
    "dge_5of6": ("results/dge_sig_atleast5of6_contrasts.txt",
                 lambda p: pd.read_csv(p, sep="\t", header=None, index_col=0)[1]),
}


def summarize(name, symbols, design, grouping, show_all=False):
    atac = pd.read_csv(ATAC.format(design=design, grouping=grouping), index_col=0)
    rna = pd.read_csv(RNA.format(grouping=grouping), index_col=0)
    t = pd.DataFrame({"symbol_in_list": symbols})
    t = t.join(atac[["gene", "log2FoldChange", "pvalue", "n_peaks"]].rename(
        columns={"log2FoldChange": "atac_log2FC", "pvalue": "atac_p"}))
    # the project's symbol table is outdated (e.g. ENSG00000143995 listed as AC092669.1 is MEIS1);
    # prefer the current GENCODE v44 name where the gene was measured
    t["symbol"] = t.pop("gene").fillna(t.symbol_in_list)
    t = t.join(rna[["log2FoldChange", "padj"]].rename(
        columns={"log2FoldChange": "rna_log2FC", "padj": "rna_padj"}))
    m = t.atac_p.notna()
    t.loc[m, "atac_padj_in_set"] = multipletests(t.loc[m, "atac_p"], method="fdr_bh")[1]
    has_both = m & t.rna_log2FC.notna()
    t["same_direction"] = pd.Series(np.sign(t.atac_log2FC) == np.sign(t.rna_log2FC),
                                    dtype="boolean").where(has_both)
    t = t.sort_values("atac_p")
    t.index.name = "ensembl"

    both = t[has_both.reindex(t.index)]
    agree = int(both.same_direction.sum())
    bt = stats.binomtest(agree, len(both), 0.5) if len(both) else None
    rho = stats.spearmanr(both.atac_log2FC, both.rna_log2FC) if len(both) > 2 else (np.nan, np.nan)
    print(f"\n=== {name} ({len(t)} genes), {grouping}, ATAC design: {design}")
    print(f"measured in ATAC: {m.sum()} (rest are not protein-coding or lack microglia peaks)")
    print(f"ATAC nominal p<0.05: {(t.atac_p < 0.05).sum()} | padj within set < 0.05: "
          f"{(t.atac_padj_in_set < 0.05).sum()} | < 0.1: {(t.atac_padj_in_set < 0.1).sum()}")
    if bt:
        print(f"same direction as RNA: {agree}/{len(both)} ({100 * agree / len(both):.0f}%, binomial p = "
              f"{bt.pvalue:.2g}); Spearman rho of log2FC = {rho[0]:.2f} (p = {rho[1]:.2g})")
    with pd.option_context("display.width", 200):
        shown = t[m.reindex(t.index)]
        print((shown if show_all else shown.head(12))[
            ["symbol", "n_peaks", "atac_log2FC", "atac_p", "atac_padj_in_set",
             "rna_log2FC", "rna_padj", "same_direction"]].round(3).to_string())
    return t


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--grouping", choices=["binADNC", "binThal"], default="binADNC",
                    help="ATAC and RNA results to compare (same low-vs-high split for both)")
    ap.add_argument("--sets", nargs="+", choices=list(GENE_SETS), default=list(GENE_SETS))
    ap.add_argument("--show-all", action="store_true", help="print every measured gene, not the top 12")
    args = ap.parse_args()
    for name in args.sets:
        path, reader = GENE_SETS[name]
        symbols = reader(path)
        for design in ["nobatch", "batch"]:
            summarize(name, symbols, design, args.grouping, args.show_all)


if __name__ == "__main__":
    main()
