"""
Build UMAP embeddings from the raw pseudobulk AnnData objects (all datasets).

Pipeline per object: counts -> CPM (normalize_total, target_sum=1e6) -> log1p -> PCA -> UMAP.
The 2-D UMAP coordinates (per bin) are written to
    <outdir>/<stem>_cell_embeddings.tsv
in the format plot_cell_embeddings expects (row 0 = header, then bin_name, UMAP1, UMAP2).

Because these are already 2-D, view/colour them with plot_cell_embeddings using
--precomputed_umap (so it does NOT re-run UMAP):
    python src/plot_cell_embeddings.py --region ROSMAP_Astrocytes \
        --embeddings_dir pseudobulk_embeddings --precomputed_umap --color_by condition --save

Region stems match the GREmLN embeddings (ROSMAP_Astrocytes, MTG, EC, EC_Microglia, ...)
so all metadata colourings keep working.

Usage:
  python src/umap_pseudobulk.py                 # all datasets
  python src/umap_pseudobulk.py --only MTG
"""
import argparse
import os

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import umap


def pseudobulk_registry():
    """(stem, pseudobulk_h5ad_path) for every dataset/cell-type/region that exists."""
    reg = []
    for ct in ["Astrocytes", "Microglia"]:                       # ROSMAP
        reg.append((f"ROSMAP_{ct}", f"data/GRN/ROSMAP/{ct}/{ct}_pseudobulk_AnnData.h5ad"))
    for ct in ["Astrocytes", "Microglia"]:                       # SEA-AD
        reg.append((f"SEAAD_{ct}", f"data/GRN/SEAAD/{ct}/{ct}_pseudobulk_AnnData.h5ad"))
    for r in ["EC", "ITG", "PFC", "V1", "V2"]:                   # Atlas astrocytes
        reg.append((r, f"data/GRN/Astrocytes/{r}_AnnData.h5ad"))
    for r in ["ITG", "PFC", "CrossRegion"]:    # Atlas microglia
        reg.append((f"{r}_Microglia", f"data/GRN/Microglia/{r}_Microglia_AnnData.h5ad"))
    return [(s, p) for s, p in reg if os.path.exists(p)]


def build_umap(path, n_pcs=50, target_sum=1e6, do_log=True, n_neighbors=15, min_dist=0.1, seed=42):
    """CPM -> log1p -> PCA -> UMAP(2D). Returns (bin_names, umap_2d, n_pcs_used)."""
    a = ad.read_h5ad(path)
    if "counts" in getattr(a, "layers", {}):
        a.X = a.layers["counts"]
    a.X = a.X.astype(np.float32)
    sc.pp.normalize_total(a, target_sum=target_sum)   # counts-per-million
    if do_log:
        sc.pp.log1p(a)
    npcs = int(min(n_pcs, a.n_obs - 1, a.n_vars - 1))
    sc.pp.pca(a, n_comps=npcs)
    reducer = umap.UMAP(n_components=2, random_state=seed, n_neighbors=n_neighbors, min_dist=min_dist)
    umap_2d = reducer.fit_transform(a.obsm["X_pca"])
    return a.obs_names.astype(str).tolist(), umap_2d, npcs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default="pseudobulk_embeddings", help="Output dir (default pseudobulk_embeddings).")
    ap.add_argument("--n_pcs", type=int, default=50, help="PCs kept before UMAP (default 50).")
    ap.add_argument("--n_neighbors", type=int, default=15, help="UMAP n_neighbors (default 15).")
    ap.add_argument("--min_dist", type=float, default=0.1, help="UMAP min_dist (default 0.1).")
    ap.add_argument("--target_sum", type=float, default=1e6, help="normalize_total target (default 1e6 = CPM).")
    ap.add_argument("--no_log", action="store_true", help="Skip log1p after CPM.")
    ap.add_argument("--only", default=None, help="Only build this stem (e.g. MTG, ROSMAP_Astrocytes).")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    reg = pseudobulk_registry()
    if args.only:
        reg = [(s, p) for s, p in reg if s == args.only]
        if not reg:
            ap.error(f"stem '{args.only}' not found among: {[s for s, _ in pseudobulk_registry()]}")

    print(f"Building UMAP embeddings for {len(reg)} pseudobulk objects "
          f"(CPM{'+log1p' if not args.no_log else ''} -> PCA{args.n_pcs} -> UMAP) -> {args.outdir}/\n")
    for stem, path in reg:
        names, umap_2d, npcs = build_umap(path, args.n_pcs, args.target_sum, not args.no_log,
                                          args.n_neighbors, args.min_dist)
        out = os.path.join(args.outdir, f"{stem}_cell_embeddings.tsv")
        pd.DataFrame(umap_2d, index=names, columns=["UMAP1", "UMAP2"]).to_csv(out)
        print(f"  {stem:22s} {len(names):>5} bins  (PCA {npcs}) -> {out}")

    print("\nDone. View/colour with e.g.:")
    print("  python src/plot_cell_embeddings.py --region ROSMAP_Astrocytes "
          "--embeddings_dir pseudobulk_embeddings --precomputed_umap --color_by condition --save")


if __name__ == "__main__":
    main()
