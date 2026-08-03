"""Do the DGE-derived genes' GRN targets circle back to the model's ranked genes?

For each DGE-derived GRN set (lrt / binThal / >=5-of-6 contrasts) built by
GRN_analysis_rankedGenes.py (results/grn_analysis/dge_derived/), take the downstream targets
of those DGE genes and check which land in the model's min5 ranked list, restricted to genes
with consensus_mean_f1_drop > 0. Prints, per set, the overlap count and the genes (symbol +
f1-drop + rank).
"""
import os
import glob
import pandas as pd

DGE_DERIVED = "results/grn_analysis/dge_derived"
RANKED = ("ranked_lists_v2/SEAAD_Microglia_flatten_mlp_thalonly_trial25/"
          "combined_ranked_genes_top200_min5.csv")


def main():
    r = pd.read_csv(RANKED)
    ranked = r[r["consensus_mean_f1_drop"] > 0]
    ranked_set = set(ranked["gene"])
    sym = dict(zip(r["gene"], r["gene_symbol"]))
    f1 = dict(zip(ranked["gene"], ranked["consensus_mean_f1_drop"]))
    rank = dict(zip(ranked["gene"], ranked["combined_rank"]))
    print(f"min5 ranked genes with mean_f1_drop>0: {len(ranked_set)}\n")

    for f in sorted(glob.glob(os.path.join(DGE_DERIVED, "*edges.csv"))):
        label = os.path.basename(f).split("_")[2]          # lrt / bin / contrasts5of6
        e = pd.read_csv(f)
        n_dge = e["ranked_gene"].nunique()
        targets = set(e.loc[e["relationship"] == "target", "connected_gene"])
        regs = set(e.loc[e["relationship"] == "regulator", "connected_gene"])
        t_hit = sorted(targets & ranked_set, key=lambda g: rank.get(g, 1e9))
        r_hit = sorted(regs & ranked_set, key=lambda g: rank.get(g, 1e9))

        print(f"=== {label}: {n_dge} DGE genes, {len(targets)} distinct targets ===")
        print(f"  targets that are ranked genes (f1_drop>0): {len(t_hit)}")
        for g in t_hit:
            print(f"    rank {int(rank[g]):>3}  {g}  {sym.get(g,'')!s:<12} f1_drop={f1[g]:.4f}")
        if r_hit:
            print(f"  regulators that are ranked genes: {len(r_hit)} "
                  f"-> {[sym.get(g, g) for g in r_hit]}")
        print()


if __name__ == "__main__":
    main()
