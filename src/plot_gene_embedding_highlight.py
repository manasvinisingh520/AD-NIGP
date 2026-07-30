# UMAP of gene embeddings, highlighting a chosen gene list (e.g. the min5 ranked genes).
# All genes plotted grey; genes in the highlight CSV drawn on top in color.

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_embeddings(path):
    """Comma-separated, no header: col0 = Ensembl gene, remaining cols = embedding dims."""
    df = pd.read_csv(path, sep=",", header=None)
    genes = df.iloc[:, 0].astype(str).tolist()
    X = df.iloc[:, 1:].astype(float).values
    return genes, X


def main(args):
    genes, X = load_embeddings(args.embeddings)
    print(f"embeddings: {X.shape[0]} genes x {X.shape[1]} dims")
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    hdf = pd.read_csv(args.highlight_csv)
    drop = dict(zip(hdf["gene"].astype(str), hdf[args.value_col])) if args.value_col in hdf else {}
    hl_set = set(hdf["gene"].astype(str))
    is_hl = np.array([g in hl_set for g in genes])
    # split highlight into genuine (drop > 0) vs tie-break (drop == 0 / missing)
    is_nz = np.array([g in hl_set and float(drop.get(g, 0) or 0) > 0 for g in genes])
    is_zero = is_hl & ~is_nz
    print(f"highlight genes present: {int(is_hl.sum())} / {len(hl_set)}  "
          f"(nonzero drop: {int(is_nz.sum())}, zero/tie-break: {int(is_zero.sum())})")

    import umap
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    xy = reducer.fit_transform(X)

    fig, ax = plt.subplots(figsize=(10, 9))
    ax.scatter(xy[~is_hl, 0], xy[~is_hl, 1], s=8, c="#cccccc", alpha=0.6,
               edgecolors="none", label=f"all genes ({(~is_hl).sum()})")
    ax.scatter(xy[is_zero, 0], xy[is_zero, 1], s=38, c="#1f77b4", alpha=0.85,
               edgecolors="black", linewidths=0.3, label=f"zero drop / tie-break ({int(is_zero.sum())})", zorder=2)
    ax.scatter(xy[is_nz, 0], xy[is_nz, 1], s=48, c="#c62828", alpha=0.95,
               edgecolors="black", linewidths=0.4, label=f"nonzero F1 drop ({int(is_nz.sum())})", zorder=3)
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.set_title(f"SEAAD Microglia gene embeddings — {args.label} highlighted",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="best", fontsize=10); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"saved -> {args.output}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--embeddings", default="GREmLN/embeddings/SEAAD_Microglia_gene_embeddings.tsv")
    p.add_argument("--highlight_csv",
                   default="ranked_lists_v2/SEAAD_Microglia_flatten_mlp_thalonly_trial25/combined_ranked_genes_top200_min5.csv")
    p.add_argument("--label", default="min5 (132 genes)")
    p.add_argument("--value_col", default="consensus_mean_f1_drop",
                   help="Column used to split highlight genes into nonzero (>0) vs tie-break (0).")
    p.add_argument("--output", default="images/gene_embeddings/SEAAD_Microglia_gene_embeddings_umap_min5.png")
    main(p.parse_args())
