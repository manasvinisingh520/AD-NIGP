"""
Build GRN inputs from a ROSMAP cell-type AnnData (e.g. data/ROSMAP_Data/Astrocytes.h5ad).

Pseudobulks (100 cells/bin per individual) and writes to data/GRN/ROSMAP/<CellType>/:
  <CellType>_pseudobulk_AnnData.h5ad
  <CellType>_matrix.csv           (genes x bins, no header, gene name in col 1)
  <CellType>_patient_mapping.csv  (bin_name, patient_name, condition)
  <CellType>_TFs.txt              (Toronto TFs_utoronto.csv HGNC symbols present in matrix)

This mirrors create_GRN_data_SEAAD.py but uses ROSMAP conventions:
  - patient column   = individualID (config.ROSMAP_PATIENT_COL)
  - condition        = braaksc mapped to 1-4 via map_rosmap_condition (config.ROSMAP_CONDITION_COL)
  - raw counts come from layers['counts'] (handled inside aggregate_cells_into_pseudobulk)

Usage:
  python src/create_GRN_data_ROSMAP.py
  python src/create_GRN_data_ROSMAP.py --cell-type Microglia
  python src/create_GRN_data_ROSMAP.py --cells-per-bin 200 --min-cells-per-patient 0
  python src/create_GRN_data_ROSMAP.py path/to/Astrocytes.h5ad
"""

import argparse
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

_script_dir = Path(__file__).resolve().parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))
from utils import aggregate_cells_into_pseudobulk, map_rosmap_condition
from config import ROSMAP_DATA_DIR, ROSMAP_PATIENT_COL, ROSMAP_CONDITION_COL

GRN_DIR = Path("data/GRN/ROSMAP")
TARGET_CELLS_PER_BIN = 100
# GRN builders (Atlas/SEA-AD) use 80; ROSMAP DGE used 0. Default to 80 for GRN consistency.
MIN_CELLS_PER_PATIENT = 80
SEED = 100
TFS_CSV_CANDIDATES = [
    Path("data/GRN/Priors/TFs_utoronto.csv"),
    Path("data/Priors/TFs_utoronto.csv"),
    Path("data/TFs_utoronto.csv"),
]


def main():
    parser = argparse.ArgumentParser(
        description="Pseudobulk a ROSMAP cell-type AnnData and build GRN inputs.",
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=None,
        help="Path to .h5ad file (default: data/ROSMAP_Data/<CellType>.h5ad)",
    )
    parser.add_argument("--cell-type", default="Astrocytes", help="Cell type (default: Astrocytes)")
    parser.add_argument(
        "--cells-per-bin", type=int, default=TARGET_CELLS_PER_BIN,
        help=f"Cells per pseudobulk bin per individual (default: {TARGET_CELLS_PER_BIN})",
    )
    parser.add_argument(
        "--min-cells-per-patient", type=int, default=MIN_CELLS_PER_PATIENT,
        help=f"Min cells per individual to include (default: {MIN_CELLS_PER_PATIENT}; ROSMAP DGE used 0)",
    )
    parser.add_argument("--seed", type=int, default=SEED, help=f"Random seed (default: {SEED})")
    parser.add_argument(
        "--min-expr-frac", type=float, default=0.0,
        help="Keep only genes detected (count>0) in at least this fraction of pseudobulk bins "
             "(default: 0.0 = keep all). Use e.g. 0.10 to cut non-expressed genes and GPU memory.",
    )
    parser.add_argument(
        "--patient-col", default=ROSMAP_PATIENT_COL,
        help=f"Individual/patient column (default: {ROSMAP_PATIENT_COL})",
    )
    parser.add_argument(
        "--condition-col", default=ROSMAP_CONDITION_COL,
        help=f"Raw condition column to map to 1-4 (default: {ROSMAP_CONDITION_COL})",
    )
    args = parser.parse_args()

    cell_type = args.cell_type
    path = Path(args.path) if args.path else Path(ROSMAP_DATA_DIR) / f"{cell_type}.h5ad"
    if not path.exists():
        print(f"Not found: {path}")
        sys.exit(1)

    out_dir = GRN_DIR / cell_type
    out_dir.mkdir(parents=True, exist_ok=True)
    out_anndata = out_dir / f"{cell_type}_pseudobulk_AnnData.h5ad"
    out_matrix = out_dir / f"{cell_type}_matrix.csv"
    out_mapping = out_dir / f"{cell_type}_patient_mapping.csv"
    out_tfs = out_dir / f"{cell_type}_TFs.txt"

    print(f"Reading: {path}")
    adata = ad.read_h5ad(path)

    if args.patient_col not in adata.obs.columns:
        print(f"Patient column '{args.patient_col}' not found. Available: {list(adata.obs.columns)}")
        sys.exit(1)
    print(f"Cells: {adata.n_obs:,}, genes: {adata.n_vars:,}, patient_col={args.patient_col}")

    # braaksc (or chosen col) -> dge_condition in {1,2,3,4}; drops cells with unmapped values
    print(f"Mapping condition '{args.condition_col}' -> dge_condition (1-4) ...")
    adata = map_rosmap_condition(adata, args.condition_col)
    print(f"  After condition mapping: {adata.shape} (cells x genes)")

    print(f"Pseudobulking: {args.cells_per_bin} cells/bin per individual "
          f"(min {args.min_cells_per_patient} cells/individual) ...")
    adata_pb = aggregate_cells_into_pseudobulk(
        adata,
        target_cells_per_bin=args.cells_per_bin,
        filter_patients_cell_threshold=args.min_cells_per_patient,
        seed=args.seed,
        patient_col=args.patient_col,
    )

    # Gene filter: always drop genes with zero counts across all bins (no signal for
    # ARACNE), plus optionally genes detected in fewer than --min-expr-frac of bins.
    # Reduces GPU-ARACNE memory (footprint ~ genes x samples) and removes noise.
    Xpb = adata_pb.X
    Xpb = Xpb.toarray() if hasattr(Xpb, "toarray") else np.asarray(Xpb)
    detected_frac = np.asarray((Xpb > 0).mean(axis=0)).ravel()
    keep = detected_frac > 0.0
    if args.min_expr_frac > 0.0:
        keep &= detected_frac >= args.min_expr_frac
    n_before = adata_pb.n_vars
    adata_pb = adata_pb[:, keep].copy()
    thr_pct = max(0.0, args.min_expr_frac) * 100
    print(f"Gene filter (drop zero-expressed; keep >= {thr_pct:.0f}% of bins): "
          f"{n_before} -> {adata_pb.n_vars} genes (dropped {n_before - adata_pb.n_vars})")

    print(f"Writing {out_anndata} ...")
    adata_pb.write_h5ad(out_anndata)
    print(f"  Shape: {adata_pb.shape} (pseudobulk samples x genes)")

    # genes x bins matrix (no header, gene name in first column)
    X = adata_pb.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    expression = X.T
    gene_names = list(adata_pb.var_names)
    bin_names = list(adata_pb.obs_names)
    print(f"Writing {out_matrix} ...")
    with open(out_matrix, "w") as f:
        for gene_name, row in zip(gene_names, expression):
            f.write(f"{gene_name},{','.join(np.asarray(row).astype(str))}\n")
    print(f"  {len(gene_names)} genes x {len(bin_names)} bins")

    # patient mapping (+ condition)
    patient_names = (
        adata_pb.obs[args.patient_col].values.tolist()
        if args.patient_col in adata_pb.obs.columns
        else [bn.rsplit("_bin", 1)[0] if "_bin" in bn else bn for bn in bin_names]
    )
    mapping = pd.DataFrame({"bin_name": bin_names, "patient_name": patient_names})
    if "dge_condition" in adata_pb.obs.columns:
        mapping["condition"] = adata_pb.obs["dge_condition"].values
    mapping.to_csv(out_mapping, index=False)
    print(f"Writing {out_mapping} ... ({len(set(patient_names))} individuals, {len(bin_names)} bins)")

    # TFs = Toronto list (HGNC symbol) intersect genes in matrix
    project_root = _script_dir.parent
    tfs_csv = next(
        (project_root / c if not c.is_absolute() else c
         for c in TFS_CSV_CANDIDATES
         if (project_root / c if not c.is_absolute() else c).exists()),
        None,
    )
    if tfs_csv is None:
        print(f"Skip {out_tfs.name}: TF list not found. Tried: {[str(c) for c in TFS_CSV_CANDIDATES]}")
    else:
        df = pd.read_csv(tfs_csv)
        if "HGNC symbol" not in df.columns:
            print(f"Skip {out_tfs.name}: 'HGNC symbol' not in {tfs_csv}. Columns: {list(df.columns)}")
        else:
            matrix_genes = set(map(str, gene_names))
            hgnc = df["HGNC symbol"].dropna().astype(str).str.strip().unique()
            tfs_in_matrix = sorted(g for g in hgnc if g in matrix_genes)
            with open(out_tfs, "w") as f:
                f.write("\n".join(tfs_in_matrix) + "\n")
            print(f"Writing {out_tfs} ... ({len(tfs_in_matrix)} TFs from Toronto list in matrix)")

    print("Done.")


if __name__ == "__main__":
    main()
