# load the 3 genes that were DE (binThal, padj<0.05 & |lfc|>=0.5) from the 75 no_threshold genes
import pandas as pd

BINTHAL = ("results/dge_final/dge_results_SEA_AD_Microglia_MTG_"
           "pseudobulk100_seed100_binThal_covbatch_vendor_name_1_vs_2.csv")


def binthal_de(padj_thresh=0.05, lfc_thresh=None):
    d = pd.read_csv(BINTHAL, index_col=0).dropna(subset=["padj"])
    m = d["padj"] < padj_thresh
    if lfc_thresh:
        m &= d["log2FoldChange"].abs() >= lfc_thresh
    return set(d.index[m])


genes = pd.read_csv("results/grn_analysis/no_threshold/"
                    "GRN_analysis_f1_top200_min5_SEAAD_Microglia_MTG_top75_summary.csv")
gene_names = genes["ranked_gene"].tolist()

DEG_set = binthal_de(lfc_thresh=0.5)
for g in gene_names:
    if g in DEG_set:
        print(g)
