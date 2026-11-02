"""Overlap of NIGP and DGE gene lists with AD GWAS genes (GWAS Catalog, all AD associations).

GWAS genes   : union of `mapped_genes` over data/annotation/ad_gwas_catalog.json["all_ad"].
Gene IDs     : Ensembl IDs mapped to symbols with GENCODE v44.
Backgrounds  : NIGP lists -> the 2,703 genes fed to the model (filtered GRN);
               DGE lists  -> every gene with a non-NA padj in the binThal DGE.
Test         : one-sided hypergeometric (over-representation).
Output       : results/gwas_overlap.csv
"""
import json, gzip, re
from math import comb
import pandas as pd

d = json.load(open("data/annotation/ad_gwas_catalog.json"))
gwas = {g.strip() for a in d["all_ad"] for m in a["mapped_genes"] for g in re.split(r"[,;]", m) if g.strip()}

e2s = {}
with gzip.open("data/annotation/gencode.v44.basic.annotation.gtf.gz", "rt") as f:
    for l in f:
        if l[0] == "#": continue
        p = l.split("\t")
        if p[2] != "gene": continue
        e2s[re.search(r'gene_id "([^".]+)', p[8]).group(1)] = re.search(r'gene_name "([^"]+)', p[8]).group(1)
sym = lambda ids: [e2s.get(i, i) for i in ids]

def hyper_p(k, n, K, N):
    return sum(comb(K, i) * comb(N - K, n - i) for i in range(k, min(n, K) + 1)) / comb(N, n)

rna = pd.read_csv("results/dge_final/dge_results_SEA_AD_Microglia_MTG_pseudobulk100_seed100_binThal_covbatch_vendor_name_1_vs_2.csv", index_col=0)
bg_dge = set(sym(rna[rna.padj.notna()].index))
bg_grn = set(sym(open("results/gprofiler_background_grn2703.txt").read().split()))
rl = "ranked_lists_v2/SEAAD_Microglia_filtered_flatmlp_trial26/combined_ranked_genes_{}.csv"
sets = {
    "NIGP min6": (sym(pd.read_csv(rl.format("min6")).gene), bg_grn),
    "NIGP min7": (sym(pd.read_csv(rl.format("min7")).gene), bg_grn),
    "DGE binThal (|lfc|>=1, paper list)": (sym(rna[(rna.padj < .05) & (rna.log2FoldChange.abs() >= 1)].index), bg_dge),
    "DGE binThal (|lfc|>=0.5)": (sym(rna[(rna.padj < .05) & (rna.log2FoldChange.abs() >= .5)].index), bg_dge),
}
rows = []
for name, (L, B) in sets.items():
    L = sorted({g for g in L if g in B}); K = len(B & gwas); N = len(B); k = len(set(L) & gwas); exp = len(L) * K / N
    rows.append(dict(list=name, size=len(L), gwas_in_list=k, pct=round(100 * k / len(L), 1), expected=round(exp, 1),
                     fold=round(k / exp, 2), p=hyper_p(k, len(L), K, N), background=N, gwas_in_background=K,
                     genes=";".join(sorted(set(L) & gwas))))
out = pd.DataFrame(rows); out.to_csv("results/gwas_overlap.csv", index=False)
print(out.drop(columns="genes").to_string(index=False))
