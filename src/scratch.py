# print top 10 genes from DGE results for SEAAD Microglia
from utils import get_top_dge_union_genes
import pickle
import pandas as pd

n_genes = 91

if 0:
    genes = pd.read_csv("ranked_lists_v2/SEAAD_Microglia_flatten_mlp_thalonly_trial25/ranked_genes.csv")
    genes = genes["gene"].tolist()
    genes = genes[:n_genes]

if 1:
    genes = get_top_dge_union_genes(
        data="SEA_AD",
        cell_type="Microglia",
        region="MTG",
        top_k=n_genes,
        dge_results_dir="results/dge_final",
    )
with open("data/genes_2_ensembl_ids.pkl", "rb") as f:
    genes_2_ensembl_id = pickle.load(f)

ensembl_to_symbol = {v: k for k, v in genes_2_ensembl_id.items()}

list = []
for gene in genes:
    if gene in ensembl_to_symbol:
        g = ensembl_to_symbol[gene]
    else:
        g = gene
    list.append(g)
with open("results/scratch.txt", "w") as f:
    for gene in list:
        f.write(f"{gene}\n")