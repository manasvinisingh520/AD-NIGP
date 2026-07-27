# print top 10 genes from DGE results for SEAAD Microglia
from utils import get_top_dge_union_genes
import pickle
import pandas as pd

n_genes = 91

if 0:
    genes = pd.read_csv("ranked_lists_v2/SEAAD_Microglia_flatten_mlp_thalonly_trial25/ranked_genes.csv")
    genes = genes["gene"].tolist()
    genes = genes[:n_genes]

if 0:
    genes = get_top_dge_union_genes(
        data="SEA_AD",
        cell_type="Microglia",
        region="MTG",
        top_k=n_genes,
        dge_results_dir="results/dge_final",
    )

"""with open("data/genes_2_ensembl_ids.pkl", "rb") as f:
    genes_2_ensembl_id = pickle.load(f)

ensembl_to_symbol = {v: k for k, v in genes_2_ensembl_id.items()}

list = []
for gene in genes:
    if gene in ensembl_to_symbol:
        g = ensembl_to_symbol[gene]
    else:
        g = gene
    list.append(g)"""

def get_dge_genes_by_contrast_count(
    data="SEA_AD",
    cell_type="Microglia",
    region="MTG",
    n_contrasts=5,
    mode="exact",              # "exact" -> in exactly n_contrasts; "atleast" -> in >= n_contrasts
    padj_thresh=0.05,
    dge_results_dir="results/dge_final",
):
    """Genes significant (padj < padj_thresh) in a given number of pairwise DGE contrasts.

    Returns a list of Ensembl IDs. With the defaults (n_contrasts=5, mode="exact"),
    this yields the genes significant in exactly 5 of the 6 SEA-AD Microglia contrasts.
    """
    import glob
    from collections import Counter

    files = sorted(glob.glob(
        f"{dge_results_dir}/dge_results_{data}_{cell_type}_{region}_*.csv"))
    counts = Counter()
    for f in files:
        df = pd.read_csv(f, index_col=0)
        if "padj" not in df.columns:
            continue
        df = df.dropna(subset=["padj"])
        counts.update(df.index[df["padj"] < padj_thresh])

    if mode == "atleast":
        genes = [g for g, c in counts.items() if c >= n_contrasts]
    else:  # exact
        genes = [g for g, c in counts.items() if c == n_contrasts]
    return sorted(genes)


genes = get_dge_genes_by_contrast_count(n_contrasts=5, mode="atleast")
with open("results/scratch.txt", "w") as f:
    for gene in genes:
        f.write(f"{gene}\n")