"""Remap gene_N indices to real gene symbols in ranked list CSVs for ITG, PFC, V1, V2."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.dirname(__file__))

import pandas as pd
from create_data import create_anndata, load_gene_embeddings

REGIONS = ["ITG", "PFC", "V1", "V2"]
RANKED_LISTS = {
    "ITG": "ranked_lists/ge_all_3L_shrinking_patience60_32_ITG/ranked_genes.csv",
    "PFC": "ranked_lists/ge_all_3L_shrinking_patience60_32_PFC/ranked_genes.csv",
    "V1":  "ranked_lists/ge_all_3L_shrinking_patience60_32_V1/ranked_genes.csv",
    "V2":  "ranked_lists/ge_all_3L_shrinking_patience60_32_V2/ranked_genes.csv",
}

for region in REGIONS:
    print(f"\n--- {region} ---")
    csv_path = RANKED_LISTS[region]
    df = pd.read_csv(csv_path)

    first_gene = str(df.iloc[0]["gene"])
    if not first_gene.startswith("gene_"):
        print(f"  Already has real gene names, skipping.")
        continue

    # Reproduce gene ordering from model training
    adata = create_anndata(region, "Astrocytes")
    emb_path = os.path.join("GREmLN", "embeddings", f"{region}_gene_embeddings.tsv")
    emb_dict = load_gene_embeddings(emb_path, gene_list=list(adata.var_names))
    gene_order = [g for g in adata.var_names if g in emb_dict]
    print(f"  Gene mapping: {len(gene_order)} genes")

    # Remap
    remapped = 0
    for i, row in df.iterrows():
        gene_name = str(row["gene"])
        if gene_name.startswith("gene_"):
            idx = int(row["gene_index"])
            if idx < len(gene_order):
                df.at[i, "gene"] = gene_order[idx]
                remapped += 1

    df.to_csv(csv_path, index=False)
    print(f"  Remapped {remapped}/{len(df)} genes -> {csv_path}")
    print(f"  Top 5: {list(df['gene'].head())}")
