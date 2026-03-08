"""Load MTG DGE union (top 10) and ranked_genes.csv top 10."""

import pandas as pd
from pathlib import Path


DGE_DIR = Path("results/dge_final")
# MTG DGE files: dge_results_SEA_AD_MTG_pseudobulk100_seed100_1_vs_2.csv etc.
MTG_DGE_PATTERN = "dge_results_SEA_AD_MTG_*.csv"
RANKED_GENES_PATH = Path("ranked_lists/ge_all_3L_shrinking_patience60_64_MTG_ThalOnly_run2/ranked_genes.csv")


def get_top_mtg_dge_union(top_k=10, dge_dir=None):
    """Top K genes from union of MTG DGE results (by best padj)."""
    dge_dir = Path(dge_dir or DGE_DIR)
    if not dge_dir.exists():
        raise FileNotFoundError(f"DGE results directory not found: {dge_dir}")
    dge_files = sorted(dge_dir.glob(MTG_DGE_PATTERN))
    if not dge_files:
        raise FileNotFoundError(f"No MTG DGE files in {dge_dir}. Pattern: {MTG_DGE_PATTERN}")
    gene_to_best_padj = {}
    for path in dge_files:
        df = pd.read_csv(path, index_col=0)
        if "padj" not in df.columns:
            continue
        for gene in df.index[df["padj"].notna()]:
            p = float(df.loc[gene, "padj"])
            if gene not in gene_to_best_padj or p < gene_to_best_padj[gene]:
                gene_to_best_padj[gene] = p
    if not gene_to_best_padj:
        raise ValueError("No genes with valid padj in DGE results.")
    sorted_genes = sorted(gene_to_best_padj.keys(), key=lambda g: gene_to_best_padj[g])
    return sorted_genes[:top_k]


def main():
    # MTG DGE: union across results/dge_final, top 10 by best padj
    print("MTG DGE (results/dge_final) – union, top 10 genes:")
    try:
        top10_dge = get_top_mtg_dge_union(top_k=10, dge_dir=DGE_DIR)
        for i, g in enumerate(top10_dge, 1):
            print(f"  {i}. {g}")
    except FileNotFoundError as e:
        print(f"  Error: {e}")
    except Exception as e:
        print(f"  Error: {e}")

    # Ranked genes: top 10 from CSV
    print("\nRanked genes (ranked_lists/ge_all_3L_shrinking_patience60_64_MTG_ThalOnly_run2/ranked_genes.csv) – top 10:")
    if not RANKED_GENES_PATH.exists():
        print(f"  Error: File not found: {RANKED_GENES_PATH}")
    else:
        df = pd.read_csv(RANKED_GENES_PATH)
        gene_col = "gene" if "gene" in df.columns else df.columns[1]
        top10_ranked = df[gene_col].head(10).astype(str).str.strip().tolist()
        for i, g in enumerate(top10_ranked, 1):
            print(f"  {i}. {g}")


if __name__ == "__main__":
    main()
