"""Build PER-CELL UMAP embeddings from the perCell AnnData and colour by metadata.

Same pipeline as umap_pseudobulk.py (counts -> CPM -> log1p -> PCA -> UMAP) but on the
per-cell AnnData (individual cells, not pseudobulk bins). Colouring is read straight from
the perCell obs -- per-cell barcodes don't map to the pseudobulk `_sample_id` lookups that
plot_cell_embeddings uses, so we colour from the cell's own metadata instead.

color_by -> perCell obs column:
  batch -> batch_vendor_name | individual -> SampleName | sex -> sex
  condition -> ADNC          | subtype -> Supertype (fine SEA-AD subtype)

Outputs:
  perCell_embeddings/<data>_<ct>_cell_embeddings.tsv               (barcode, UMAP1, UMAP2)
  images/cell_embeddings/perCell_embeddings/<data>_<ct>_<color_by>.png

Usage:
  python src/umap_percell.py --data SEAAD --cell_type Astrocytes --save
"""
import argparse
import os

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import umap
import matplotlib.pyplot as plt

OBS_COL = {"batch": "batch_vendor_name", "individual": "SampleName", "sex": "sex",
           "condition": "ADNC", "subtype": "Supertype"}
NICE = {"batch": "batch (vendor)", "individual": "individual (donor)", "sex": "sex",
        "condition": "ADNC", "subtype": "Supertype"}
# Severity ramp for the SEA-AD ADNC condition (matches plot_cell_embeddings).
ADNC_COLORS = {"Not AD": "green", "Low": "lightgreen", "Intermediate": "orange",
               "High": "red", "Reference": "lightgray"}
# Distinct pair for sex (avoids the default blue/orange).
SEX_COLORS = {"female": "#8e44ad", "male": "#16a085"}  # purple / teal


def percell_path(data, ct):
    return os.path.join("data", "model_data", "perCell_AnnData", f"{data}_{ct}_AnnData_perCell.h5ad")


def build_umap(a, n_pcs=50, target_sum=1e6, n_neighbors=15, min_dist=0.1, seed=42):
    """counts -> CPM -> log1p -> PCA -> UMAP(2D). Returns (umap_2d, n_pcs_used)."""
    a.X = a.X.astype(np.float32)
    sc.pp.normalize_total(a, target_sum=target_sum)   # counts-per-million
    sc.pp.log1p(a)
    npcs = int(min(n_pcs, a.n_obs - 1, a.n_vars - 1))
    sc.pp.pca(a, n_comps=npcs)
    reducer = umap.UMAP(n_components=2, random_state=seed, n_neighbors=n_neighbors, min_dist=min_dist)
    return reducer.fit_transform(a.obsm["X_pca"]), npcs


def _scatter(ax, xy, labels, title, custom_colors=None):
    uniq = sorted(set(labels))
    if custom_colors and all(u in custom_colors for u in uniq):
        lut = {u: custom_colors[u] for u in uniq}
    else:
        cmap = plt.get_cmap("tab10" if len(uniq) <= 10 else ("tab20" if len(uniq) <= 20 else "gist_ncar"))
        denom = max(1, len(uniq) - 1)
        lut = {u: cmap(i / denom) for i, u in enumerate(uniq)}
    ax.scatter(xy[:, 0], xy[:, 1], c=[lut[l] for l in labels], s=3, alpha=0.5, linewidths=0)
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2"); ax.grid(True, alpha=0.3)
    if len(uniq) <= 25:
        handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=lut[u],
                              markersize=8, label=u) for u in uniq]
        ax.legend(handles=handles, title=title, loc="best", fontsize=7, framealpha=0.9)
        ax.set_title(f"colored by {title}", fontsize=12, fontweight="bold")
    else:
        ax.set_title(f"colored by {title}  [{len(uniq)} groups]", fontsize=12, fontweight="bold")


def main(args):
    data, ct = args.data, args.cell_type
    stem = f"{data}_{ct}"
    emb_dir = "perCell_embeddings"
    tsv = os.path.join(emb_dir, f"{stem}_cell_embeddings.tsv")
    if args.precomputed:
        # Reuse saved UMAP coords (skip recompute); only need obs for colouring.
        a = ad.read_h5ad(percell_path(data, ct), backed="r")
        obs = a.obs.copy()
        coords = pd.read_csv(tsv, index_col=0).reindex(obs.index.astype(str))
        xy = coords[["UMAP1", "UMAP2"]].to_numpy()
        npcs = "precomputed"
        print(f"{stem}: {a.shape}  (loaded UMAP from {tsv})")
    else:
        a = ad.read_h5ad(percell_path(data, ct))
        obs = a.obs.copy()
        print(f"{stem}: {a.shape}")
        xy, npcs = build_umap(a, args.n_pcs, args.target_sum, args.n_neighbors, args.min_dist, args.seed)
        print(f"  UMAP done (PCA {npcs})")
        os.makedirs(emb_dir, exist_ok=True)
        pd.DataFrame(xy, index=obs.index.astype(str), columns=["UMAP1", "UMAP2"]).to_csv(tsv)

    outdir = os.path.join("images", "cell_embeddings", "perCell_embeddings")
    os.makedirs(outdir, exist_ok=True)
    for cb in args.color_by:
        col = OBS_COL[cb]
        if col not in obs.columns:
            print(f"  skip {cb}: obs has no column {col}")
            continue
        labels = obs[col].astype(str).values
        fig, ax = plt.subplots(figsize=(11, 9))
        cmap_override = {"condition": ADNC_COLORS, "sex": SEX_COLORS}.get(cb)
        _scatter(ax, xy, labels, NICE[cb], custom_colors=cmap_override)
        pipe = "UMAP (precomputed)" if npcs == "precomputed" else f"CPM->log1p->PCA{npcs}->UMAP"
        fig.suptitle(f"{data} {ct} perCell  |  {a.n_obs} cells  |  {pipe}",
                     fontsize=12, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        if args.save:
            out = os.path.join(outdir, f"{stem}_{cb}.png".lower())
            fig.savefig(out, dpi=150, bbox_inches="tight")
            print(f"  Saved -> {out}")
        else:
            plt.show()
        plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="SEAAD", choices=["SEAAD", "ROSMAP"])
    p.add_argument("--cell_type", default="Astrocytes", choices=["Astrocytes", "Microglia"])
    p.add_argument("--color_by", nargs="+", default=list(OBS_COL), choices=list(OBS_COL),
                   help="Which metadata to colour by (default: all five).")
    p.add_argument("--n_pcs", type=int, default=50)
    p.add_argument("--n_neighbors", type=int, default=15)
    p.add_argument("--min_dist", type=float, default=0.1)
    p.add_argument("--target_sum", type=float, default=1e6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--precomputed", action="store_true",
                   help="Reuse saved perCell_embeddings/<stem>_cell_embeddings.tsv coords "
                        "instead of recomputing UMAP (fast recolour).")
    p.add_argument("--save", action="store_true", help="Save PNGs instead of showing.")
    main(p.parse_args())
