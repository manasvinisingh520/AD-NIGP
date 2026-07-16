"""Visualize the EXACT per-gene model input for one patient, as a UMAP of genes.

The gene embeddings are first PCA-reduced 512 -> 16 (Reduction A, PCA(random_state=42),
exactly as src/utils.scale_embeddings does), then scaled by expression:

  scaled[gene, :] = log1p(count) * emb16[gene, :]

Produces four UMAP panels (3972 GRN genes as points, colored by log1p expression) that
walk the whole featurization pipeline:

  (1) Raw 512-d:          the GREmLN gene embeddings before any reduction (unscaled)
  (2) PCA-16 unscaled:    after Reduction A (512 -> 16), pure gene-embedding geometry
  (3) Single cell:        emb16 scaled by ONE cell's expression
  (4) Patient aggregated: emb16 scaled by the patient's MEAN log1p expression over its cells

Usage:
  python src/plot_model_input_umap.py --data SEAAD --cell_type Astrocytes --patient H20.33.030 --save
"""
import argparse
import os

import anndata as ad
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import umap

from create_data import load_gene_embeddings

TARGET_DIMS = 16  # Reduction A target, matching create_data_v2 / utils.scale_embeddings


def _percell_path(data, cell_type):
    return os.path.join("data", "model_data", "perCell_AnnData",
                        f"{data}_{cell_type}_AnnData_perCell.h5ad")


def _embeddings_path(data, cell_type):
    return os.path.join("GREmLN", "embeddings", f"{data}_{cell_type}_gene_embeddings.tsv")


def _umap2d(X, seed=42):
    """UMAP a (n_genes, D) matrix to 2-D. Rows (genes) are the points."""
    reducer = umap.UMAP(n_components=2, random_state=seed, n_neighbors=15, min_dist=0.1)
    return reducer.fit_transform(np.asarray(X, dtype=np.float64))


def main(args):
    data, ct, patient = args.data, args.cell_type, args.patient

    # --- Load per-cell expression, restrict to the patient's cells ---
    adata = ad.read_h5ad(_percell_path(data, ct))
    sn = adata.obs["SampleName"].astype(str).values
    if patient not in set(sn):
        raise SystemExit(f"Patient {patient} not in {data} {ct} perCell "
                         f"({adata.obs['SampleName'].nunique()} donors).")
    cell_mask = sn == patient
    patient_cells = np.where(cell_mask)[0]
    print(f"{patient}: {len(patient_cells)} cells in {data} {ct}")

    # --- Gene embeddings aligned to AnnData genes (same set the model uses) ---
    emb_dict = load_gene_embeddings(_embeddings_path(data, ct), gene_list=list(adata.var_names))
    gene_names = [g for g in adata.var_names if g in emb_dict]
    emb512 = np.array([emb_dict[g] for g in gene_names], dtype=np.float64)  # (n_genes, 512)
    print(f"Gene embeddings: {emb512.shape} ({len(gene_names)} GRN genes)")

    # --- Reduction A: PCA 512 -> 16 over the genes (model-exact, random_state=42) ---
    emb16 = PCA(n_components=TARGET_DIMS, random_state=42).fit_transform(emb512)  # (n_genes, 16)

    # --- Expression for this patient's cells (raw counts -> log1p), only the GRN genes ---
    gi = [adata.var_names.get_loc(g) for g in gene_names]
    Xp = adata.X[patient_cells][:, gi]
    Xp = Xp.toarray() if hasattr(Xp, "toarray") else np.asarray(Xp)
    log1p_expr = np.log1p(Xp.astype(np.float64))          # (n_cells, n_genes)

    # --- Pick the single cell to show, and the patient-mean expression ---
    if args.cell is not None and args.cell in adata.obs_names[patient_cells]:
        local = list(adata.obs_names[patient_cells]).index(args.cell)
    else:
        local = 0  # default: first cell of the patient
    cell_id = str(adata.obs_names[patient_cells][local])
    cell_expr = log1p_expr[local]                         # (n_genes,)
    patient_expr = log1p_expr.mean(axis=0)                # (n_genes,) mean log1p over the patient's cells
    print(f"Single cell: {cell_id}  |  patient mean over {len(patient_cells)} cells")

    # --- Scaled gene tensors: scaled[gene,:] = expr[gene] * emb[gene,:] (model's rule) ---
    # PCA-16 gene embeddings: unscaled, then scaled (scaled[gene,:] = expr[gene] * emb16[gene,:]).
    # The unscaled panel is patient/cell-independent (pure embedding geometry); colored by the
    # patient-mean expression only so the three panels share one color scale.
    panels = [
        ("Raw gene embeddings (512-d)\nNOT scaled  [before PCA]",
         emb512, patient_expr),
        ("Gene embeddings only (PCA-16)\nNOT scaled  [after PCA]",
         emb16, patient_expr),
        (f"Single cell (PCA-16)\nscaled by cell expression\n{cell_id}",
         cell_expr[:, None] * emb16, cell_expr),
        (f"Patient aggregated (PCA-16)\nmean over {len(patient_cells)} cells",
         patient_expr[:, None] * emb16, patient_expr),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(26, 7))
    for ax, (title, scaled, color) in zip(axes, panels):
        print(f"  UMAP: {title.splitlines()[0]} {scaled.shape} ...")
        xy = _umap2d(scaled, seed=args.seed)
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=color, cmap="viridis", s=14, alpha=0.7)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
        ax.grid(True, alpha=0.3)
        fig.colorbar(sc, ax=ax, label="log1p(count)")

    fig.suptitle(f"{data} {ct}  |  {patient}  |  featurization pipeline (3972 GRN genes as points):  "
                 f"raw 512-d  ->  PCA-16  ->  x cell expr  ->  patient mean",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if args.save:
        os.makedirs(args.outdir, exist_ok=True)
        out = os.path.join(args.outdir, f"{data}_{ct}_{patient}_model_input_umap.png".replace(" ", ""))
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved -> {out}")
    else:
        plt.show()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="SEAAD", choices=["SEAAD", "ROSMAP", "Atlas"])
    p.add_argument("--cell_type", default="Astrocytes", choices=["Astrocytes", "Microglia"])
    p.add_argument("--patient", default="H20.33.030", help="Donor SampleName (default H20.33.030).")
    p.add_argument("--cell", default=None, help="Specific cell obs_name to show (default: first of patient).")
    p.add_argument("--outdir", default=os.path.join("images", "model_input_umap"))
    p.add_argument("--seed", type=int, default=42, help="UMAP random_state.")
    p.add_argument("--save", action="store_true", help="Save PNG instead of showing.")
    main(p.parse_args())
