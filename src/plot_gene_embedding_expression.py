"""UMAP of the (mi3.04) filtered-GRN gene embeddings, colored by expression level.

Each gene is one point in the 2D UMAP of its GREmLN gene embedding; the color is a
continuous "heatmap" of that gene's mean expression (the model-input per-donor log1p
mean, same measure as plot_importance_vs_expression). Genes with no expression value
are drawn grey.
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Reuse the exact per-gene expression measure used elsewhere (model-input log1p mean).
from plot_importance_vs_expression import per_donor_expr_stats

EMB = "GREmLN/embeddings/SEAAD_Microglia_filtered_gene_embeddings.tsv"
OUT = "images/gene_embeddings/SEAAD_Microglia_filtered_gene_embeddings_umap_expression.png"


def load_embeddings(path):
    """Filtered embedding tsv: comma-separated, header row of dim indices, col0 = Ensembl gene."""
    df = pd.read_csv(path, sep=",", index_col=0)
    genes = df.index.astype(str).tolist()
    X = np.nan_to_num(df.to_numpy(dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    return genes, X


def main(args):
    genes, X = load_embeddings(args.embeddings)
    print(f"embeddings: {X.shape[0]} genes x {X.shape[1]} dims")

    # Per-gene expression (mean log1p across donors); align to embedding genes.
    expr = per_donor_expr_stats().set_index("gene")["expr_mean"]
    color_vals = np.array([expr.get(g, np.nan) for g in genes], dtype=float)
    has_expr = np.isfinite(color_vals)
    print(f"genes with an expression value: {int(has_expr.sum())} / {len(genes)}")

    import umap
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    xy = reducer.fit_transform(X)

    fig, ax = plt.subplots(figsize=(10.5, 9))
    # genes lacking expression: grey background
    ax.scatter(xy[~has_expr, 0], xy[~has_expr, 1], s=10, c="#dddddd",
               edgecolors="none", label=f"no expr value ({int((~has_expr).sum())})")
    # expressed genes: continuous expression heatmap
    sc = ax.scatter(xy[has_expr, 0], xy[has_expr, 1], s=22, c=color_vals[has_expr],
                    cmap=args.cmap, alpha=0.9, edgecolors="none")
    cbar = fig.colorbar(sc, ax=ax, shrink=0.85)
    cbar.set_label("mean expression (log1p, model input)", fontsize=11)

    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.set_title("SEAAD Microglia filtered-GRN gene embeddings — colored by expression",
                 fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    if (~has_expr).any():
        ax.legend(loc="best", fontsize=9)
    fig.tight_layout()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"saved -> {args.output}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="UMAP of filtered-GRN gene embeddings colored by expression.")
    p.add_argument("--embeddings", default=EMB)
    p.add_argument("--cmap", default="viridis")
    p.add_argument("--output", default=OUT)
    main(p.parse_args())
