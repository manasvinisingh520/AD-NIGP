"""Generate v2 data pickles containing BOTH expression-only and embedding-scaled
representations per fold, with identical patient splits.

Output dict-pickles:
    data/model_data/{region}_v2_data.pkl            (Atlas Astrocytes)
    data/model_data/{region}_Microglia_v2_data.pkl  (Atlas Microglia)
    data/model_data/{SEAAD,ROSMAP}_{ct}_v2_data.pkl (staging datasets)

Keys:
    expression_folds   – list[np.ndarray (n_cells, n_genes)] float16
    embedding_folds    – list[np.ndarray (n_cells, n_genes, D)] float16
    patient_info_folds – same structure as create_data.py
    ptau_folds / thal_folds / braak_folds
    gene_names         – list[str]
    embedding_dim      – int (D)

Usage:
    python src/create_data_v2.py --cell_type Astrocytes --region EC
    python src/create_data_v2.py --data SEAAD  --cell_type Microglia
    python src/create_data_v2.py --data ROSMAP --cell_type Astrocytes
"""

from __future__ import annotations

import argparse
import gc
import os
import pickle
from collections import defaultdict

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold

from create_data import load_gene_embeddings
from save_perCell_AnnData import load_percell_anndata
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

    if cell_type == "staging":
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
                print(f"  Using StratifiedKFold ({label_name}) for staging folds.")
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

        if cell_type == "staging":
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
    data = getattr(args, "data", "Atlas")
    filt_tag = "_filtered" if getattr(args, "filtered", False) else ""
    if filt_tag:
        print("Using the FILTERED gene-embedding variant.")

    if data in ("SEAAD", "ROSMAP"):
        if cell_type not in ("Astrocytes", "Microglia"):
            raise ValueError(f"--data {data} requires --cell_type Astrocytes or Microglia")
        regions = [data]
        label_mode = "staging"      # Thal + Braak staging labels (loaded from perCell)
    else:
        if cell_type == "Astrocytes":
            all_regions = ["EC", "ITG", "PFC", "V2", "V1"]
        elif cell_type == "Microglia":
            all_regions = ["CrossRegion", "ITG", "PFC"]
        else:
            raise ValueError(f"Invalid cell type: {cell_type}")
        regions = [args.region] if args.region else all_regions
        label_mode = cell_type

    embeddings_dir = os.path.join("GREmLN", "embeddings")

    for region in regions:
        print(f"\n{'='*60}")
        print(f"Region: {region}, cell_type: {cell_type}")
        print(f"{'='*60}")

        # --- AnnData (all datasets load their pre-built perCell object) ---
        adata = load_percell_anndata(data, cell_type, region)

        # --- Gene embeddings (optionally the _filtered variant) ---
        if data in ("SEAAD", "ROSMAP"):
            emb_path = os.path.join(embeddings_dir, f"{data}_{cell_type}{filt_tag}_gene_embeddings.tsv")
        elif cell_type == "Astrocytes":
            emb_path = os.path.join(embeddings_dir, f"{region}{filt_tag}_gene_embeddings.tsv")
        else:  # Microglia
            emb_path = os.path.join(embeddings_dir, f"{region}_Microglia{filt_tag}_gene_embeddings.tsv")
        if not os.path.exists(emb_path):
            raise FileNotFoundError(f"Gene embedding file not found: {emb_path}")

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
        # --- Optional control: shuffle gene embeddings across genes ---
        # Each gene keeps its own expression but is paired with another gene's GREmLN vector,
        # so any GRN-specific information tied to a gene is destroyed while the set of vectors,
        # the expression, cells, folds and labels stay identical.
        shuffle_seed = getattr(args, "shuffle_embeddings", None)
        if shuffle_seed is not None:
            perm = np.random.default_rng(shuffle_seed).permutation(len(gene_names))
            embeddings_matrix = embeddings_matrix[perm]
            print(f"Gene embeddings SHUFFLED across genes (seed {shuffle_seed}); "
                  f"{(perm == np.arange(len(perm))).sum()} genes kept their own vector by chance")
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

        if label_mode == "staging":
            thal_info = adata.obs["Thal"].astype(str).str.strip()
            braak_info = adata.obs["Braak"].astype(str).str.strip()
        elif label_mode == "Microglia":
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
            if label_mode == "staging":
                base_thal.append(thal_info[mask].iloc[0])
                base_braak.append(braak_info[mask].iloc[0])
                base_ptau.append(None)
            else:
                base_ptau.append(ptau_info[mask].iloc[0])
                base_thal.append(thal_info[mask].iloc[0])
                base_braak.append(None)

        # Drop invalid patients. Staging (SEAAD/ROSMAP) labels are already cleaned when
        # the perCell AnnData is built (save_perCell_AnnData drops missing/Reference Thal
        # & Braak), so no extra filtering is needed here.
        if label_mode == "staging":
            valid_idx = list(range(len(unique_patients)))
        elif label_mode == "Microglia":
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
            cell_type=label_mode,
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
        out_tag = filt_tag + (f"_embshuffle_seed{shuffle_seed}" if shuffle_seed is not None else "")
        if data in ("SEAAD", "ROSMAP"):
            out_name = f"{data}_{cell_type}_v2_data_{expr_transform}{out_tag}.pkl"
        elif cell_type == "Microglia":
            out_name = f"{region}_Microglia_v2_data_{expr_transform}{out_tag}.pkl"
        else:
            out_name = f"{region}_v2_data_{expr_transform}{out_tag}.pkl"
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
        "--data",
        type=str,
        default="Atlas",
        choices=["Atlas", "SEAAD", "ROSMAP"],
        help="Dataset. Atlas (default) uses the per-region perCell AnnData; "
             "SEAAD/ROSMAP load <DATA>_<ct> perCell AnnData with Thal+Braak staging labels.",
    )
    parser.add_argument(
        "--cell_type",
        type=str,
        default="Astrocytes",
        choices=["Astrocytes", "Microglia"],
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
    parser.add_argument(
        "--filtered",
        action="store_true",
        help="Load the _filtered gene-embedding variant (e.g. "
             "GREmLN/embeddings/SEAAD_Microglia_filtered_gene_embeddings.tsv) and tag the "
             "output pickle with _filtered.",
    )
    parser.add_argument(
        "--shuffle_embeddings", type=int, default=None, metavar="SEED",
        help="Control: randomly reassign gene embeddings across genes (seeded) before scaling; "
             "output pickle tagged _embshuffle_seed<SEED>. Train with "
             "train_model_v2.py --data_tag _embshuffle_seed<SEED>.",
    )
    args = parser.parse_args()
    main(args)
