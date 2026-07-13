"""
Load MTG_AnnData.h5ad from SEA_AD_Data/MTG/, pseudobulk (100 cells/bin per patient), save to data/GRN:
  MTG_pseudobulk_AnnData.h5ad, MTG_matrix.csv, MTG_patient_mapping.csv, MTG_TFs.txt

MTG_TFs.txt = genes that are in both TFs_utoronto.csv (Toronto TF list, HGNC symbol) and MTG_matrix.csv.

Usage:
  python src/create_GRN_data_SEAAD.py
  python src/create_GRN_data_SEAAD.py data/SEA_AD_Data/MTG/MTG_AnnData.h5ad
  python src/create_GRN_data_SEAAD.py --cells-per-bin 200
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
from utils import aggregate_cells_into_pseudobulk

MTG_DIR = Path("data/SEA_AD_Data/Astrocytes")
INPUT_H5AD = MTG_DIR / "Astrocytes.h5ad"
GRN_DIR = Path("data/GRN")
PATIENT_COL = "donor_id"  # fallback to SampleName if missing
TARGET_CELLS_PER_BIN = 100
#MIN_CELLS_PER_PATIENT = 80
SEED = 100
# Toronto TF list (HGNC symbol); try these paths
TFS_CSV_CANDIDATES = [
    Path("data/GRN/Priors/TFs_utoronto.csv"),
    Path("data/Priors/TFs_utoronto.csv"),
    Path("data/TFs_utoronto.csv"),
]


def main():
    parser = argparse.ArgumentParser(
        description="Load MTG_AnnData.h5ad, pseudobulk (100 cells/bin per patient), save to data/GRN.",
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=None,
        help=f"Path to .h5ad file (default: {INPUT_H5AD})",
    )
    parser.add_argument(
        "--cells-per-bin",
        type=int,
        default=TARGET_CELLS_PER_BIN,
        help=f"Cells per pseudobulk bin per patient (default: {TARGET_CELLS_PER_BIN})",
    )
    args = parser.parse_args()

    path = Path(args.path) if args.path else INPUT_H5AD
    if not path.exists():
        print(f"Not found: {path}")
        sys.exit(1)

    GRN_DIR.mkdir(parents=True, exist_ok=True)
    out_anndata = GRN_DIR / "Astrocytes_pseudobulk_AnnData.h5ad"
    out_matrix = GRN_DIR / "Astrocytes_matrix.csv"
    out_mapping = GRN_DIR / "MTG_patient_mapping.csv"
    out_tfs = GRN_DIR / "MTG_TFs.txt"

    print(f"Reading: {path}")
    adata = ad.read_h5ad(path, backed="r")
    try:
        patient_col = PATIENT_COL if PATIENT_COL in adata.obs.columns else "SampleName"
        if patient_col not in adata.obs.columns:
            print(f"Patient column '{patient_col}' not found. Available: {list(adata.obs.columns)}")
            sys.exit(1)
        print(f"Cells: {adata.n_obs:,}, genes: {adata.n_vars:,}, patient_col={patient_col}")

        print(f"Pseudobulking: {args.cells_per_bin} cells/bin per patient ...")
        adata_pb = aggregate_cells_into_pseudobulk(
            adata,
            target_cells_per_bin=args.cells_per_bin,
            filter_patients_cell_threshold=MIN_CELLS_PER_PATIENT,
            seed=SEED,
            patient_col=patient_col,
        )
    finally:
        if getattr(adata, "file", None) is not None:
            adata.file.close()

    print(f"Writing {out_anndata} ...")
    adata_pb.write_h5ad(out_anndata)
    print(f"  Shape: {adata_pb.shape} (pseudobulk samples × genes)")

    print(f"Writing {out_matrix} ...")
    X = adata_pb.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    expression = X.T
    gene_names = list(adata_pb.var_names)
    bin_names = list(adata_pb.obs_names)
    with open(out_matrix, "w") as f:
        for gene_name, row in zip(gene_names, expression):
            f.write(f"{gene_name},{','.join(np.asarray(row).astype(str))}\n")
    print(f"  {len(gene_names)} genes × {len(bin_names)} bins")

    patient_names = (
        adata_pb.obs[patient_col].values.tolist()
        if patient_col in adata_pb.obs.columns
        else [bn.rsplit("_bin", 1)[0] if "_bin" in bn else bn for bn in bin_names]
    )
    pd.DataFrame({"bin_name": bin_names, "patient_name": patient_names}).to_csv(
        out_mapping, index=False
    )
    print(f"Writing {out_mapping} ...")

    # TFs = Toronto list (TFs_utoronto.csv, HGNC symbol) ∩ genes in matrix
    project_root = _script_dir.parent
    tfs_csv = None
    for c in TFS_CSV_CANDIDATES:
        p = project_root / c if not c.is_absolute() else c
        if p.exists():
            tfs_csv = p
            break
    if tfs_csv is None:
        print(f"Skip MTG_TFs.txt: TF list not found. Tried: {[str(c) for c in TFS_CSV_CANDIDATES]}")
    else:
        matrix_genes = set(pd.read_csv(out_matrix, header=None).iloc[:, 0].astype(str).str.strip())
        df = pd.read_csv(tfs_csv)
        if "HGNC symbol" not in df.columns:
            print(f"Skip MTG_TFs.txt: 'HGNC symbol' not in {tfs_csv}. Columns: {list(df.columns)}")
        else:
            hgnc = df["HGNC symbol"].dropna().astype(str).str.strip().unique()
            tfs_in_matrix = sorted(g for g in hgnc if g in matrix_genes)
            with open(out_tfs, "w") as f:
                f.write("\n".join(tfs_in_matrix) + "\n")
            print(f"Writing {out_tfs} ... ({len(tfs_in_matrix)} TFs from Toronto list in matrix)")

    print("Done.")


if __name__ == "__main__":
    main()
