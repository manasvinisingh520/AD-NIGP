"""
Basic Leiden clustering of GREmLN cell embeddings via a KNN graph in embedding space.

Loads GREmLN/embeddings/<region>_cell_embeddings.tsv (bins x embedding dims), builds a
k-nearest-neighbour graph directly on the embeddings (scanpy.pp.neighbors, use_rep='X'),
runs Leiden on that graph (igraph backend -> no leidenalg dependency), and writes the
cluster assignments to CSV. Optionally saves a UMAP coloured by cluster.

Usage:
  python src/leiden_cluster_embeddings.py --region ROSMAP_Astrocytes
  python src/leiden_cluster_embeddings.py --region MTG --n_neighbors 15 --resolution 1.0 --plot
  python src/leiden_cluster_embeddings.py --embeddings path/to/foo_cell_embeddings.tsv -o out.csv
"""
import argparse
import os

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc


def load_embeddings(path):
    """Load a *_cell_embeddings.tsv/csv: first column = bin name, rest = embedding dims.

    The first row is a header/index row (matches plot_cell_embeddings.load_cell_embeddings).
    """
    try:
        df = pd.read_csv(path, sep=",", header=None)
    except Exception:
        df = pd.read_csv(path, sep="\t", header=None)
    df = df.iloc[1:]  # skip header row
    names = df.iloc[:, 0].astype(str).str.strip().tolist()
    X = df.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
    X = X.dropna(axis=1, how="all")           # drop trailing empty column(s)
    return X.to_numpy(dtype=float), names


def leiden_cluster(X, names, n_neighbors=15, resolution=1.0, seed=0):
    """KNN graph on the embeddings + Leiden. Returns an AnnData with obs['leiden']."""
    adata = ad.AnnData(X=X.astype(np.float32))
    adata.obs_names = names
    # KNN graph in the embedding space (no PCA: use the embeddings as-is)
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep="X", random_state=seed)
    # Leiden via igraph backend (leidenalg not required)
    sc.tl.leiden(adata, resolution=resolution, flavor="igraph", n_iterations=2,
                 directed=False, random_state=seed)
    return adata


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--region", help="Embedding file stem, e.g. ROSMAP_Astrocytes, MTG, EC.")
    ap.add_argument("--embeddings", help="Direct path to a *_cell_embeddings.tsv (overrides --region).")
    ap.add_argument("--n_neighbors", type=int, default=15, help="k for the KNN graph (default 15).")
    ap.add_argument("--resolution", type=float, default=1.0, help="Leiden resolution (default 1.0).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plot", action="store_true", help="Also save a UMAP coloured by cluster.")
    ap.add_argument("-o", "--output", default=None,
                    help="Output CSV (default: results/leiden/<stem>_leiden.csv).")
    args = ap.parse_args()

    if args.embeddings:
        path = args.embeddings
        stem = os.path.basename(path).replace("_cell_embeddings.tsv", "").replace("_cell_embeddings.csv", "")
    elif args.region:
        stem = args.region
        path = os.path.join("GREmLN", "embeddings", f"{stem}_cell_embeddings.tsv")
    else:
        ap.error("Provide --region or --embeddings.")
    if not os.path.exists(path):
        ap.error(f"Embeddings not found: {path}")

    X, names = load_embeddings(path)
    print(f"Loaded {X.shape[0]} cells x {X.shape[1]} embedding dims from {path}")

    adata = leiden_cluster(X, names, args.n_neighbors, args.resolution, args.seed)
    labels = adata.obs["leiden"].astype(int)
    vc = labels.value_counts().sort_index()
    print(f"Leiden: {vc.size} clusters (n_neighbors={args.n_neighbors}, resolution={args.resolution})")
    for c, n in vc.items():
        print(f"  cluster {c}: {n} cells")

    out = args.output or os.path.join("results", "leiden", f"{stem}_leiden.csv")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    pd.DataFrame({"bin_name": names, "leiden": labels.values}).to_csv(out, index=False)
    print(f"Saved cluster assignments -> {out}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        sc.tl.umap(adata, random_state=args.seed)
        fig, ax = plt.subplots(figsize=(9, 8))
        xy = adata.obsm["X_umap"]
        for c in sorted(labels.unique()):
            m = labels.values == c
            ax.scatter(xy[m, 0], xy[m, 1], s=16, alpha=0.7, label=f"{c} (n={m.sum()})")
        ax.legend(loc="best", fontsize=8, title="Leiden")
        ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
        ax.set_title(f"{stem} Leiden (res={args.resolution}, k={args.n_neighbors}, {vc.size} clusters)")
        fig.tight_layout()
        png = os.path.splitext(out)[0] + ".png"
        fig.savefig(png, dpi=150, bbox_inches="tight")
        print(f"Saved UMAP -> {png}")


if __name__ == "__main__":
    main()
