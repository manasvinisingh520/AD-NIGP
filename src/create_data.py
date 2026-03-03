"""Scale all gene embeddings according to cell expression and save in pickle file.
Supports Astrocytes, Microglia, and MTG (SEA AD). 8-fold cross-validation split with target variables.
For MTG, targets are Thal and Braak (both categorical).
"""

import argparse
from collections import defaultdict
import pandas as pd
import numpy as np
import os
import pickle
import gc
import anndata as ad
import glob
from tqdm import tqdm
from sklearn.model_selection import KFold
from utils import get_region_file_paths, read_excel_columns, read_mtx_file, scale_embeddings, create_anndata_object
from config import REGION_TO_TAB, DATA_DIR


def load_gene_embeddings(embeddings_path, gene_list: list[str]):
    """Load gene embeddings (Ensembl IDs in file), convert to symbols, key by AnnData var_names.
    When multiple symbols map to the same Ensembl ID, use the one that appears in AnnData."""
    df = pd.read_csv(embeddings_path, sep=',', header=0)
    gene_ids = df.iloc[:, 0].astype(str).values
    emb_values = df.iloc[:, 1:].values.astype(float)
    # Embedding file: Ensembl ID -> vector
    gene_embedding_dict = {gid: emb for gid, emb in zip(gene_ids, emb_values)}

    with open("data/genes_2_ensembl_ids.pkl", "rb") as f:
        symbol_to_ensembl = pickle.load(f)

    # Reverse: Ensembl -> list of symbols (multiple symbols can map to same Ensembl)
    ensembl_to_symbols = defaultdict(list)
    for sym, ens in symbol_to_ensembl.items():
        ensembl_to_symbols[ens].append(sym)

    adata_var_set = set(gene_list)
    # For each Ensembl in the embedding file, pick the symbol that is in AnnData (if any)
    embedding_data = {}
    for ensembl_id, vector in gene_embedding_dict.items():
        candidates = ensembl_to_symbols.get(ensembl_id, [ensembl_id])
        for sym in candidates:
            if sym in adata_var_set:
                embedding_data[sym] = vector
                break
        else:
            # No symbol in AnnData; keep Ensembl as key only if AnnData uses Ensembl
            if ensembl_id in adata_var_set:
                embedding_data[ensembl_id] = vector

    return embedding_data

def find_anndata_file(region, cell_type='Astrocytes'):
    """Find AnnData file for a given region and cell type (perCell only, or MTG)."""
    if cell_type == "Astrocytes":
        path = os.path.join('data', 'model_data', f'{region}_AnnData_perCell.h5ad')
    elif cell_type == "Microglia":
        path = os.path.join('data', 'model_data', f'{region}_Microglia_AnnData_perCell.h5ad')
    elif cell_type == "MTG":
        # Prefer model_data, then SEA_AD_Data
        model_data_per_cell = os.path.join('data', 'model_data', 'MTG_AnnData_perCell.h5ad')
        if os.path.exists(model_data_per_cell):
            path = model_data_per_cell
        else:
            print("HERE")
            exit()
    else:
        raise ValueError(f"Invalid cell type: {cell_type}")
    
    if os.path.exists(path):
        return path
    
    return None

def create_anndata_astrocytes(region):
    """Build AnnData from MTX + Excel metadata (Astrocytes only)."""
    base_prefix = f"2025-11-16_Astrocytes_{region}"

    mtx_path, row_annotation_path, col_annotation_path = get_region_file_paths(
        region,
        data_dir=DATA_DIR,
        base_prefix=base_prefix
    )
    metadata_path = os.path.join(DATA_DIR, "2025-11-16_Astrocytes_metadata.xlsx")
    tab_index = REGION_TO_TAB.get(region)

    metadata = read_excel_columns(
        metadata_path,
        columns=['cell_annotation', 'Path..Group.', 'SampleName', 'percent.mito', 'RIN', 'Total.Genes.Detected', 'Thal', 'Ptau.Total.Tau..A.U..'],
        sheet_name=tab_index
    )

    matrix, gene_names, cell_names = read_mtx_file(
        mtx_path=str(mtx_path),
        row_annotation_path=str(row_annotation_path),
        col_annotation_path=str(col_annotation_path),
        transpose=False
    )

    metadata = metadata.set_index('cell_annotation')
    metadata = metadata.reindex(cell_names)

    valid_cell_mask = (metadata['Thal'] != 'unk').fillna(False)
    valid_cell_names = metadata[valid_cell_mask].index.tolist()
    valid_metadata = metadata[valid_cell_mask].copy()

    cell_indices_to_keep = [i for i, name in enumerate(cell_names) if name in valid_cell_names]
    filtered_cell_names = [cell_names[i] for i in cell_indices_to_keep]
    filtered_matrix = matrix[:, cell_indices_to_keep]

    adata = create_anndata_object(
        matrix=filtered_matrix,
        gene_names=gene_names,
        cell_names=filtered_cell_names,
        obs=valid_metadata,
        transpose=True
    )
    return adata


def load_anndata_microglia(region):
    """Load Microglia per-cell AnnData from pre-built h5ad (metadata is not Excel/sheets)."""
    path = find_anndata_file(region, cell_type="Microglia")
    if path is None:
        raise FileNotFoundError(
            f"Microglia AnnData not found: data/model_data/{region}_Microglia_AnnData_perCell.h5ad. "
            "Create it first (e.g. from your Microglia pipeline); metadata for Microglia is CSV/per-region, not Excel."
        )
    adata = ad.read_h5ad(path)
    # Ensure patient column exists (Microglia often uses Donor ID / Donor_ID)
    if "SampleName" not in adata.obs.columns:
        if "Donor_ID" in adata.obs.columns:
            adata.obs["SampleName"] = adata.obs["Donor_ID"].astype(str)
        elif "Donor ID" in adata.obs.columns:
            adata.obs["SampleName"] = adata.obs["Donor ID"].astype(str)
        else:
            raise ValueError(f"Microglia AnnData must have obs column 'SampleName' or 'Donor_ID'/'Donor ID'. Found: {list(adata.obs.columns)}")
    if "Thal" not in adata.obs.columns or "Ptau.Total.Tau..A.U.." not in adata.obs.columns:
        raise ValueError(
            f"Microglia AnnData must have obs columns 'Thal' and 'Ptau.Total.Tau..A.U..'. "
            f"Found: {list(adata.obs.columns)}"
        )
    valid_cell_mask = (adata.obs["Thal"] != "unk").fillna(False)
    adata = adata[valid_cell_mask].copy()
    return adata


def load_anndata_mtg(region):
    """Load MTG (SEA AD) per-cell AnnData. Targets: Thal and Braak (categorical)."""
    path = find_anndata_file(region, cell_type="MTG")
    if path is None:
        raise FileNotFoundError(
            f"MTG AnnData not found: data/SEA_AD_Data/MTG/MTG_AnnData.h5ad"
        )
    adata = ad.read_h5ad(path)
    # Patient column: Donor ID or SampleName
    if "SampleName" not in adata.obs.columns:
        if "Donor_ID" in adata.obs.columns:
            adata.obs["SampleName"] = adata.obs["Donor_ID"].astype(str)
        elif "Donor ID" in adata.obs.columns:
            adata.obs["SampleName"] = adata.obs["Donor ID"].astype(str)
        else:
            raise ValueError(
                f"MTG AnnData must have obs column 'SampleName' or 'Donor_ID'/'Donor ID'. Found: {list(adata.obs.columns)}"
            )
    for col in ["Thal", "Braak"]:
        if col not in adata.obs.columns:
            raise ValueError(
                f"MTG AnnData must have obs column '{col}'. Found: {list(adata.obs.columns)}"
            )
    print(len(adata.obs))
    print(len(adata.var_names))
    # Keep only cells with valid Thal and Braak: exclude NaN, empty, and "Reference"
    thal_s = adata.obs["Thal"].astype(str).str.strip()
    braak_s = adata.obs["Braak"].astype(str).str.strip()
    valid = (
        adata.obs["Thal"].notna()
        & (thal_s != "")
        & (thal_s != "Reference")
        & adata.obs["Braak"].notna()
        & (braak_s != "")
        & (braak_s != "Reference")
    )
    print(valid.sum())
    adata = adata[valid].copy()
    return adata


def create_anndata(region, cell_type='Astrocytes'):
    """Load or build AnnData for a region. Astrocytes: MTX + Excel. Microglia/MTG: pre-built h5ad."""
    if cell_type == "Astrocytes":
        return create_anndata_astrocytes(region)
    elif cell_type == "Microglia":
        return load_anndata_microglia(region)
    elif cell_type == "MTG":
        return load_anndata_mtg(region)
    raise ValueError(f"Invalid cell type: {cell_type}")

def main(args=None):
    args = args or argparse.Namespace(cell_type="Astrocytes", region=None)
    cell_type = args.cell_type
    n_folds = 8
    target_dims = 16

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
        print("Loading data")
        if cell_type == "Astrocytes":
            emb_path = os.path.join(embeddings_dir, f"{region}_gene_embeddings.tsv")
        elif cell_type == "Microglia":
            emb_path = os.path.join(embeddings_dir, f"{region}_Microglia_gene_embeddings.tsv")
        else:
            # MTG: try MTG_gene_embeddings.tsv, fallback to EC
            emb_path = os.path.join(embeddings_dir, "MTG_gene_embeddings.tsv")
            if not os.path.exists(emb_path):
                emb_path = os.path.join(embeddings_dir, "EC_gene_embeddings.tsv")

        adata = create_anndata(region, cell_type)
        # Load gene embeddings
        embeddings_dict = load_gene_embeddings(emb_path, gene_list=adata.var_names)

        # Filter the adata for which not found genes are not in the adata.var_names
        adata = adata[:, adata.var_names.isin(embeddings_dict.keys())].copy()
        print(f"Size of percell data: {adata.shape}")

         # Get expression matrix with cells as rows and genes as columns
        X = adata.X
        expression_matrix = np.array(X.toarray() if hasattr(X, "toarray") else np.asarray(X), dtype=np.float16)

        # create embeddings matrix from embeddings_dict
        embeddings_matrix = np.array([embeddings_dict[gene] for gene in adata.var_names], dtype=np.float16)
        print(f"Size of embeddings before scaling: {embeddings_matrix.shape}")

        # Verify shapes match (should always match since we filtered to common genes)
        if embeddings_matrix.shape[0] != expression_matrix.shape[1]:
            print(f"ERROR: Shape mismatch! Embeddings: {embeddings_matrix.shape[0]} genes, Expression: {expression_matrix.shape[1]} genes.")
            quit()
        
        # Target variables: MTG uses Thal and Braak (categorical); others use ptau and thal
        if cell_type == "MTG":
            thal_info = adata.obs["Thal"].astype(str).str.strip()
            braak_info = adata.obs["Braak"].astype(str).str.strip()
            ptau_info = None  # unused for MTG
        else:
            ptau_info = adata.obs['Ptau.Total.Tau..A.U..']
            thal_info = adata.obs['Thal']
            braak_info = None  # unused for Astrocytes/Microglia
        
        # expression_matrix = d, embedding = A
        region_scaled_embeddings = scale_embeddings(embeddings_matrix, expression_matrix, n_components=target_dims)
        del embeddings_matrix, expression_matrix
        print(f"Size of embeddings after scaling: {region_scaled_embeddings.shape}")

        # Collect all patient embeddings into lists for KFold splitting
        all_patient_embeddings = []
        all_cell_names = []
        all_ptau = []
        all_thal = []
        all_braak = []
        unique_patients = adata.obs['SampleName'].unique()

        # verify that the number of cells in the embeddings is the same as the number of cells in the adata.obs
        if region_scaled_embeddings.shape[0] != len(adata.obs):
            print(f"ERROR: Mismatch! Embeddings: {region_scaled_embeddings.shape[0]} cells, AnnData obs: {len(adata.obs)} cells")
            quit()

        for patient_name in unique_patients:
            patient_mask = adata.obs['SampleName'] == patient_name
            patient_embeddings = region_scaled_embeddings[patient_mask].astype(np.float16)
            patient_cell_names = adata.obs.index[patient_mask].tolist()
            all_patient_embeddings.append(patient_embeddings)
            all_cell_names.append(patient_cell_names)
            if cell_type == "MTG":
                all_thal.append(thal_info[patient_mask].iloc[0])
                all_braak.append(braak_info[patient_mask].iloc[0])
                all_ptau.append(None)
            else:
                all_ptau.append(ptau_info[patient_mask].iloc[0])
                all_thal.append(thal_info[patient_mask].iloc[0])
                all_braak.append(None)

        # MTG: drop any patient with Thal or Braak "Reference" so model_data has no reference
        if cell_type == "MTG":
            ref_val = "Reference"
            valid_idx = [
                i for i in range(len(unique_patients))
                if (str(all_thal[i]).strip() != ref_val and str(all_braak[i]).strip() != ref_val)
            ]
            n_dropped = len(unique_patients) - len(valid_idx)
            unique_patients = np.asarray(unique_patients)[valid_idx]
            all_patient_embeddings = [all_patient_embeddings[i] for i in valid_idx]
            all_cell_names = [all_cell_names[i] for i in valid_idx]
            all_ptau = [all_ptau[i] for i in valid_idx]
            all_thal = [all_thal[i] for i in valid_idx]
            all_braak = [all_braak[i] for i in valid_idx]

        # Microglia: drop patients with NaN or invalid Braak/CERAD so downstream code never sees them
        if cell_type == "Microglia":
            BRAAK_VALID = {2, 3, 5, 6}
            CERAD_VALID = {"none", "sparse", "moderate", "frequent"}
            def valid_microglia(ptau_val, thal_val):
                try:
                    if pd.isna(ptau_val) or pd.isna(thal_val):
                        return False
                    braak = int(round(float(ptau_val)))
                    if braak not in BRAAK_VALID:
                        return False
                    cerad = str(thal_val).strip().lower()
                    return cerad in CERAD_VALID
                except (ValueError, TypeError):
                    return False
            valid_idx = [i for i in range(len(unique_patients)) if valid_microglia(all_ptau[i], all_thal[i])]
            unique_patients = np.asarray(unique_patients)[valid_idx]
            all_patient_embeddings = [all_patient_embeddings[i] for i in valid_idx]
            all_cell_names = [all_cell_names[i] for i in valid_idx]
            all_ptau = [all_ptau[i] for i in valid_idx]
            all_thal = [all_thal[i] for i in valid_idx]
            all_braak = [all_braak[i] for i in valid_idx]

        # Split patients into 8 folds (not cells, but patients)
        n_patients = len(unique_patients)
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        embeddings_folds = []
        patient_info_folds = []
        ptau_folds = []
        thal_folds = []
        braak_folds = []
        
        for fold_idx, (_, fold_patient_indices) in enumerate(kf.split(range(n_patients))):
            # Concatenate embeddings from patients in this fold
            fold_embeddings_list = [all_patient_embeddings[i] for i in fold_patient_indices]
            fold_embeddings = np.concatenate(fold_embeddings_list, axis=0).astype(np.float16)
            
            if cell_type == "MTG":
                fold_thal = [all_thal[i] for i in fold_patient_indices]
                fold_braak = [all_braak[i] for i in fold_patient_indices]
                fold_ptau = []
            else:
                fold_ptau = [all_ptau[i] for i in fold_patient_indices]
                fold_thal = [all_thal[i] for i in fold_patient_indices]
                fold_braak = []
            
            fold_patient_info = {unique_patients[i]: all_cell_names[i] for i in fold_patient_indices}
            
            embeddings_folds.append(fold_embeddings)
            patient_info_folds.append(fold_patient_info)
            ptau_folds.append(fold_ptau)
            thal_folds.append(fold_thal)
            braak_folds.append(fold_braak)
            
            n_fold_cells = fold_embeddings.shape[0]
            n_fold_patients = len(fold_patient_indices)

        # Size of all model inputs and outputs before dumping
        for i in range(n_folds):
            ef = embeddings_folds[i]
            pf = patient_info_folds[i]
            tf = thal_folds[i]
            bf = braak_folds[i] if cell_type == "MTG" else None
            if cell_type == "MTG":
                print(f"Size of model inputs/outputs (fold {i+1}): embeddings {ef.shape}, patient_info {len(pf)} patients, thal {len(tf)}, braak {len(bf)}")
            else:
                print(f"Size of model inputs/outputs (fold {i+1}): embeddings {ef.shape}, patient_info {len(pf)} patients, thal {len(tf)}")

        # Save folds
        if cell_type == "Microglia":
            out_name = f"{region}_Microglia_data_for_model_training.pkl"
        elif cell_type == "MTG":
            out_name = "MTG_data_for_model_training.pkl"
        else:
            out_name = f"{region}_data_for_model_training.pkl"
        emb_output_dir = os.path.join("data", "model_data", out_name)
        with open(emb_output_dir, "wb") as f:
            if cell_type == "MTG":
                pickle.dump((embeddings_folds, patient_info_folds, thal_folds, braak_folds), f, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                pickle.dump((embeddings_folds, patient_info_folds, ptau_folds, thal_folds), f, protocol=pickle.HIGHEST_PROTOCOL)
        del embeddings_folds, patient_info_folds, ptau_folds, thal_folds, braak_folds, region_scaled_embeddings
        del adata, fold_embeddings, fold_patient_info, fold_ptau, fold_thal, fold_braak
        del all_patient_embeddings, all_cell_names, all_ptau, all_thal, all_braak
        del unique_patients
        gc.collect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scale gene embeddings by expression and save for model training")
    parser.add_argument("--cell_type", type=str, default="Astrocytes", choices=["Astrocytes", "Microglia", "MTG"],
                        help="Cell type (default: Astrocytes). MTG uses Thal and Braak as categorical targets.")
    parser.add_argument("--region", type=str, default=None,
                        help="Process only this region (e.g., EC). If omitted, process all regions for the cell type.")
    args = parser.parse_args()
    main(args)
