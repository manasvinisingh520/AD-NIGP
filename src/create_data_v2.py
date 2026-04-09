"""Generate v2 data pickles containing BOTH expression-only and embedding-scaled
representations per fold, with identical patient splits.

For each region the output is a single dict-pickle:
    data/model_data/{region}_v2_data.pkl   (Astrocytes)
    data/model_data/{region}_Microglia_v2_data.pkl
    data/model_data/MTG_v2_data.pkl

Keys:
    expression_folds   – list[np.ndarray (n_cells, n_genes)] float16
    embedding_folds    – list[np.ndarray (n_cells, n_genes, D)] float16
    patient_info_folds – same structure as create_data.py
    ptau_folds / thal_folds / braak_folds
    gene_names         – list[str]
    embedding_dim      – int (D)

Usage:
    python src/create_data_v2.py --cell_type Astrocytes
    python src/create_data_v2.py --cell_type Microglia
    python src/create_data_v2.py --cell_type MTG
    python src/create_data_v2.py --cell_type Astrocytes --region EC
"""

from __future__ import annotations

import argparse
import gc
import os
import pickle
from collections import defaultdict

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold

from create_data import (
    create_anndata,
    load_gene_embeddings,
)
from utils import scale_embeddings

N_FOLDS = 8
TARGET_DIMS = 16


def _build_folds(
    cell_type: str,
    expression_matrix: np.ndarray,
    scaled_emb: np.ndarray,
    adata_obs,
    unique_patients: np.ndarray,
    base_ptau: list,
    base_thal: list,
    base_braak: list,
    base_patient_cell_idx: list,
    base_patient_cell_names: list,
):
    """Split patients into N_FOLDS folds and return per-fold data."""
    n_patients = len(unique_patients)

    if cell_type == "MTG":
        strat_thal = [str(base_thal[i]).strip() for i in range(n_patients)]
        strat_braak = [str(base_braak[i]).strip() for i in range(n_patients)]
        fold_splits = None
        for label_name, strat_labels in [
            ("Thal only", strat_thal),
            ("Braak only", strat_braak),
        ]:
            try:
                kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
                fold_splits = list(kf.split(range(n_patients), strat_labels))
                print(f"  Using StratifiedKFold ({label_name}) for MTG folds.")
                break
            except ValueError:
                continue
        if fold_splits is None:
            kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
            fold_splits = list(kf.split(range(n_patients)))
    else:
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
        fold_splits = list(kf.split(range(n_patients)))

    expr_folds, emb_folds = [], []
    patient_info_folds, ptau_folds, thal_folds, braak_folds = [], [], [], []

    for fold_idx, (_, fold_patient_indices) in enumerate(fold_splits):
        fold_cell_idx = np.concatenate(
            [base_patient_cell_idx[i] for i in fold_patient_indices]
        )
        fold_expr = expression_matrix[fold_cell_idx].astype(np.float16)
        fold_emb = scaled_emb[fold_cell_idx].astype(np.float16)

        fold_pi = [
            (str(unique_patients[i]), base_patient_cell_names[i])
            for i in fold_patient_indices
        ]

        if cell_type == "MTG":
            fold_thal = [base_thal[i] for i in fold_patient_indices]
            fold_braak = [base_braak[i] for i in fold_patient_indices]
            fold_ptau = []
        else:
            fold_ptau = [base_ptau[i] for i in fold_patient_indices]
            fold_thal = [base_thal[i] for i in fold_patient_indices]
            fold_braak = []

        expr_folds.append(fold_expr)
        emb_folds.append(fold_emb)
        patient_info_folds.append(fold_pi)
        ptau_folds.append(fold_ptau)
        thal_folds.append(fold_thal)
        braak_folds.append(fold_braak)

        n_base = len(fold_patient_indices)
        print(
            f"  Fold {fold_idx + 1}: patients={n_base}, cells={fold_expr.shape[0]}"
        )

    return (
        expr_folds,
        emb_folds,
        patient_info_folds,
        ptau_folds,
        thal_folds,
        braak_folds,
    )


def main(args):
    cell_type = args.cell_type

    if cell_type == "Astrocytes":
        all_regions = ["EC", "ITG", "PFC", "V2", "V1"]
    elif cell_type == "Microglia":
        all_regions = ["CrossRegion", "ITG", "PFC"]
    elif cell_type == "MTG":
        all_regions = ["MTG"]
    else:
        raise ValueError(f"Invalid cell type: {cell_type}")

    regions = [args.region] if args.region else all_regions
    embeddings_dir = os.path.join("GREmLN", "embeddings")

    for region in regions:
        print(f"\n{'='*60}")
        print(f"Region: {region}, cell_type: {cell_type}")
        print(f"{'='*60}")

        # --- AnnData ---
        adata = create_anndata(region, cell_type)

        # --- Gene embeddings ---
        if cell_type == "Astrocytes":
            emb_path = os.path.join(embeddings_dir, f"{region}_gene_embeddings.tsv")
        elif cell_type == "Microglia":
            emb_path = os.path.join(
                embeddings_dir, f"{region}_Microglia_gene_embeddings.tsv"
            )
        else:
            emb_path = os.path.join(embeddings_dir, "MTG_gene_embeddings.tsv")
            if not os.path.exists(emb_path):
                emb_path = os.path.join(embeddings_dir, "EC_gene_embeddings.tsv")

        embeddings_dict = load_gene_embeddings(emb_path, gene_list=adata.var_names)
        adata = adata[:, adata.var_names.isin(embeddings_dict.keys())].copy()
        print(f"Filtered AnnData shape: {adata.shape}")

        gene_names = adata.var_names.tolist()

        # --- Raw expression matrix (n_cells, n_genes) ---
        X = adata.X
        raw_expr = np.array(
            X.toarray() if hasattr(X, "toarray") else np.asarray(X), dtype=np.float32
        )

        # --- Apply expression transform ---
        expr_transform = getattr(args, "expr_transform", "log1p")
        if expr_transform == "log1p":
            expression_matrix = np.log1p(raw_expr).astype(np.float16)
        elif expr_transform == "sqrt":
            expression_matrix = np.sqrt(raw_expr + 1e-8).astype(np.float16)
        elif expr_transform == "linear":
            expression_matrix = raw_expr.astype(np.float16)
        else:
            raise ValueError(f"Unknown --expr_transform: {expr_transform}")
        print(f"Expression transform: {expr_transform}")

        # --- Gene embedding matrix (n_genes, D) ---
        embeddings_matrix = np.array(
            [embeddings_dict[g] for g in gene_names], dtype=np.float16
        )
        embedding_dim = int(embeddings_matrix.shape[1])
        print(
            f"Expression: {expression_matrix.shape}, "
            f"Gene embeddings: {embeddings_matrix.shape}"
        )

        # --- Embedding-scaled tensor (n_cells, n_genes, D) ---
        # Uses the same transform on expression as the scaling factor f(x),
        # so embedding = f(x) * gene_emb, where f matches expr_transform.
        scaled_emb = scale_embeddings(
            embeddings_matrix,
            raw_expr.astype(np.float16),
            n_components=TARGET_DIMS,
            method=expr_transform,
        )
        del raw_expr
        print(f"Scaled embedding tensor: {scaled_emb.shape}")

        # --- Per-patient bookkeeping ---
        unique_patients = adata.obs["SampleName"].unique()
        base_patient_cell_idx = []
        base_patient_cell_names = []
        base_ptau, base_thal, base_braak = [], [], []

        # Microglia Thal column contains CERAD string labels, not numeric Thal stages
        CERAD_STR_TO_INT = {"none": 0, "sparse": 1, "moderate": 2, "frequent": 3}

        if cell_type == "MTG":
            thal_info = adata.obs["Thal"].astype(str).str.strip()
            braak_info = adata.obs["Braak"].astype(str).str.strip()
        elif cell_type == "Microglia":
            ptau_info = adata.obs["Ptau.Total.Tau..A.U.."]
            raw_thal = adata.obs["Thal"].astype(str).str.strip().str.lower()
            thal_info = raw_thal.map(CERAD_STR_TO_INT)
        else:
            ptau_info = adata.obs["Ptau.Total.Tau..A.U.."]
            thal_info = adata.obs["Thal"]

        for patient_name in unique_patients:
            mask = adata.obs["SampleName"] == patient_name
            cell_idx = np.where(mask.values if hasattr(mask, "values") else mask)[0]
            base_patient_cell_idx.append(cell_idx)
            base_patient_cell_names.append(adata.obs.index[mask].tolist())
            if cell_type == "MTG":
                base_thal.append(thal_info[mask].iloc[0])
                base_braak.append(braak_info[mask].iloc[0])
                base_ptau.append(None)
            else:
                base_ptau.append(ptau_info[mask].iloc[0])
                base_thal.append(thal_info[mask].iloc[0])
                base_braak.append(None)

        # Drop invalid patients (MTG Reference, Microglia unmapped, etc.)
        if cell_type == "MTG":
            valid_idx = [
                i
                for i in range(len(unique_patients))
                if str(base_thal[i]).strip() != "Reference"
                and str(base_braak[i]).strip() != "Reference"
            ]
        elif cell_type == "Microglia":
            CERAD_VALID = {0, 1, 2, 3}

            def _valid(ptau_val, thal_val):
                try:
                    return int(thal_val) in CERAD_VALID
                except (ValueError, TypeError):
                    return False

            valid_idx = [
                i
                for i in range(len(unique_patients))
                if _valid(base_ptau[i], base_thal[i])
            ]
        else:
            valid_idx = list(range(len(unique_patients)))

        if len(valid_idx) < len(unique_patients):
            n_dropped = len(unique_patients) - len(valid_idx)
            print(f"  Dropped {n_dropped} invalid patients")
            unique_patients = np.asarray(unique_patients)[valid_idx]
            base_patient_cell_idx = [base_patient_cell_idx[i] for i in valid_idx]
            base_patient_cell_names = [base_patient_cell_names[i] for i in valid_idx]
            base_ptau = [base_ptau[i] for i in valid_idx]
            base_thal = [base_thal[i] for i in valid_idx]
            base_braak = [base_braak[i] for i in valid_idx]

        print(f"  {len(unique_patients)} patients after filtering")

        # --- Build folds ---
        (
            expr_folds,
            emb_folds,
            patient_info_folds,
            ptau_folds,
            thal_folds,
            braak_folds,
        ) = _build_folds(
            cell_type=cell_type,
            expression_matrix=expression_matrix,
            scaled_emb=scaled_emb,
            adata_obs=adata.obs,
            unique_patients=unique_patients,
            base_ptau=base_ptau,
            base_thal=base_thal,
            base_braak=base_braak,
            base_patient_cell_idx=base_patient_cell_idx,
            base_patient_cell_names=base_patient_cell_names,
        )

        # --- Save ---
        out_dir = os.path.join("data", "model_data")
        os.makedirs(out_dir, exist_ok=True)
        if cell_type == "Microglia":
            out_name = f"{region}_Microglia_v2_data_{expr_transform}.pkl"
        elif cell_type == "MTG":
            out_name = "MTG_v2_data_{expr_transform}.pkl"
        else:
            out_name = f"{region}_v2_data_{expr_transform}.pkl"
        out_path = os.path.join(out_dir, out_name)

        payload = {
            "expression_folds": expr_folds,
            "embedding_folds": emb_folds,
            "patient_info_folds": patient_info_folds,
            "ptau_folds": ptau_folds,
            "thal_folds": thal_folds,
            "braak_folds": braak_folds,
            "gene_names": gene_names,
            "embedding_dim": embedding_dim,
            "expr_transform": expr_transform,
        }
        with open(out_path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved: {out_path}")

        # Cleanup
        del (
            expression_matrix,
            scaled_emb,
            embeddings_matrix,
            expr_folds,
            emb_folds,
            adata,
        )
        gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate v2 data (expression + embedding) for model training"
    )
    parser.add_argument(
        "--cell_type",
        type=str,
        default="Astrocytes",
        choices=["Astrocytes", "Microglia", "MTG"],
    )
    parser.add_argument(
        "--region",
        type=str,
        default=None,
        help="Process only this region (e.g. EC). If omitted, process all.",
    )
    parser.add_argument(
        "--expr_transform",
        type=str,
        default="log1p",
        choices=["log1p", "sqrt", "linear"],
        help="Transform applied to raw UMI counts before saving expression folds "
             "and before scaling gene embeddings. Default: log1p",
    )
    args = parser.parse_args()
    main(args)
