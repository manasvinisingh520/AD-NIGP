"""
Plot expression of top biomarker genes in a specific sample.

Uses:
- Top 5 genes from ranked_genes.csv (embedding importance)
- Top 5 genes from the union of EC DGE results
Plots their expression in one chosen sample (e.g. one patient) from EC AnnData
to compare levels between the two gene sets.
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import get_top_ranked_genes_protein_coding, get_top_dge_union_genes

# Default paths
DEFAULT_ANNDATA = "E:/sf2026/perCell_AnnData/EC_AnnData_perCell.h5ad"
DEFAULT_DGE_DIR = "E:/sf2026/dge_results/dge_final"
SAMPLE_COL = "SampleName"
  

def plot_biomarker_expression(
    adata_path,
    ranked_genes,
    dge_genes,
    sample_id=None,
    sample_col=SAMPLE_COL,
    output_path=None,
):
    """
    Subset adata to one sample, then plot expression of ranked_genes and dge_genes.
    """
    import anndata

    adata = anndata.read_h5ad(adata_path)
    if sample_col not in adata.obs.columns:
        raise ValueError(f"Sample column '{sample_col}' not in adata.obs. Available: {list(adata.obs.columns)}")

    if sample_id is None:
        sample_id = str(adata.obs[sample_col].dropna().iloc[0])
    adata_sample = adata[adata.obs[sample_col].astype(str) == str(sample_id)].copy()

    if adata_sample.n_obs == 0:
        raise ValueError(f"No cells found for sample '{sample_id}'. Available: {adata.obs[sample_col].dropna().unique()[:20].tolist()}")

    # Use raw counts from UMIs layer if present, else .X
    if "UMIs" in getattr(adata_sample, "layers", {}):
        X = adata_sample.layers["UMIs"]
    else:
        X = adata_sample.X
    if hasattr(X, "toarray"):
        X = X.toarray()

    var_names = set(adata_sample.var_names)
    ranked_present = [g for g in ranked_genes if g in var_names]
    dge_present = [g for g in dge_genes if g in var_names]
    if not ranked_present and not dge_present:
        raise ValueError("None of the requested genes found in adata.var_names.")
    missing_r = set(ranked_genes) - set(ranked_present)
    missing_d = set(dge_genes) - set(dge_present)
    if missing_r:
        print(f"  Ranked genes not in AnnData (skipped): {missing_r}")
    if missing_d:
        print(f"  DGE genes not in AnnData (skipped): {missing_d}")

    def get_expr(genes, source_label):
        if not genes:
            return pd.DataFrame(columns=["gene", "source", "log1p_expression"])
        idx = [adata_sample.var_names.get_loc(g) for g in genes]
        expr = np.array(X[:, idx])
        log_expr = np.log1p(expr)
        rows = []
        for i, g in enumerate(genes):
            rows.extend([{"gene": g, "source": source_label, "log1p_expression": v} for v in log_expr[:, i]])
        return pd.DataFrame(rows)

    df_ranked = get_expr(ranked_present, "Ranked (embedding)")
    df_dge = get_expr(dge_present, "DGE union (EC)")
    df = pd.concat([df_ranked, df_dge], ignore_index=True)
    if df.empty:
        raise ValueError("No expression data to plot.")

    # Build list (gene, source_label, values) for each gene
    gene_plots = []
    for (gene, source), grp in df.groupby(["gene", "source"], sort=False):
        vals = np.sort(grp["log1p_expression"].values)  # sorted for smooth line
        gene_plots.append((gene, source, vals))

    n_genes = len(gene_plots)
    if n_genes == 0:
        raise ValueError("No expression data to plot.")
    n_cols = min(3, n_genes)
    n_rows = (n_genes + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), squeeze=False)
    axes = axes.flat

    for idx, (gene, source, vals) in enumerate(gene_plots):
        ax = axes[idx]
        x = np.arange(len(vals))
        ax.plot(x, vals, color="steelblue", linewidth=1)
        ax.set_title(f"{gene} ({source})")
        ax.set_ylabel("log1p(expression)")
        ax.set_xlabel("Cell (sorted)")
        ax.grid(True, alpha=0.3)
    for idx in range(n_genes, len(axes)):
        axes[idx].set_visible(False)
    plt.suptitle(f"Biomarker expression in one sample (n={adata_sample.n_obs} cells)", y=1.02)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    plt.show()
    return fig


def main():
    parser = argparse.ArgumentParser(description="Plot top 5 ranked + top 5 EC DGE genes expression in one sample")
    parser.add_argument("--ranked_genes", type=str, required=False, help="Path to ranked_genes.csv")
    parser.add_argument("--anndata", type=str, default=DEFAULT_ANNDATA, help="AnnData with expression")
    parser.add_argument("--dge_results_dir", type=str, default=DEFAULT_DGE_DIR, help="DGE results directory for EC union")
    parser.add_argument("--sample", type=str, default=None, help="Sample/patient ID to plot (default: first sample)")
    parser.add_argument("--top_k", type=int, default=5, help="Top K genes per list (default: 5)")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output figure path")
    parser.add_argument("--region", type=str, default="EC", help="Region for DGE union (default: EC)")
    parser.add_argument("--cell_type", type=str, default="Astrocytes", help="Cell type for DGE pattern (default: Astrocytes)")
    args = parser.parse_args()

    ranked_genes = args.ranked_genes or "ranked_lists/EC_Astrocytes_embedding_highest_lambda_entropy/ranked_genes.csv"
    if not os.path.isfile(ranked_genes):
        raise FileNotFoundError(f"ranked_genes not found: {ranked_genes}")
    if not os.path.isfile(args.anndata):
        raise FileNotFoundError(f"AnnData not found: {args.anndata}")

    ranked = get_top_ranked_genes_protein_coding(ranked_genes, top_k=args.top_k)
    dge = get_top_dge_union_genes(region=args.region, cell_type=args.cell_type, top_k=args.top_k, dge_results_dir=args.dge_results_dir)
    print(f"Top {args.top_k} ranked (embedding, protein-coding): {ranked}")
    print(f"Top {args.top_k} DGE union ({args.region}): {dge}")

    plot_biomarker_expression(
        adata_path=args.anndata,
        ranked_genes=ranked,
        dge_genes=dge,
        sample_id=args.sample,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
