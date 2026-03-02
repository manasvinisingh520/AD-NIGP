"""Save region-specific raw gene expression data with Thal and ptau, separated by patient.
Similar format to create_data.py but saves raw expression instead of scaled embeddings.
Astrocytes: MTX + Excel. Microglia: pre-built h5ad (data/model_data/{region}_Microglia_AnnData_perCell.h5ad).
MTG: per-cell AnnData from data/SEA_AD_Data/MTG/MTG_AnnData.h5ad (Thal + Braak, no ptau).
"""

import pandas as pd
import numpy as np
import os
import pickle
import gc
import argparse
from sklearn.model_selection import KFold
from pathlib import Path
import anndata as ad
from utils import (
    get_region_file_paths,
    read_excel_columns,
    read_mtx_file,
    create_anndata_object,
    get_top_k_genes,
)
from config import REGION_TO_TAB, DATA_DIR


def _find_anndata_microglia(region):
    path = os.path.join("data", "model_data", f"{region}_Microglia_AnnData_perCell.h5ad")
    return path if os.path.exists(path) else None


def _find_anndata_mtg():
    path = os.path.join("data", "SEA_AD_Data", "MTG", "MTG_AnnData.h5ad")
    return path if os.path.exists(path) else None


def _load_anndata_mtg():
    """Load MTG (SEA AD) per-cell AnnData. Targets: Thal and Braak (categorical)."""
    path = _find_anndata_mtg()
    if path is None:
        raise FileNotFoundError(
            f"MTG AnnData not found: data/SEA_AD_Data/MTG/MTG_AnnData.h5ad"
        )
    adata = ad.read_h5ad(path)
    if "SampleName" not in adata.obs.columns:
        if "Donor_ID" in adata.obs.columns:
            adata.obs["SampleName"] = adata.obs["Donor_ID"].astype(str)
        elif "Donor ID" in adata.obs.columns:
            adata.obs["SampleName"] = adata.obs["Donor ID"].astype(str)
        else:
            raise ValueError(
                f"MTG AnnData must have obs 'SampleName' or 'Donor_ID'/'Donor ID'. Found: {list(adata.obs.columns)}"
            )
    for col in ["Thal", "Braak"]:
        if col not in adata.obs.columns:
            raise ValueError(
                f"MTG AnnData must have obs column '{col}'. Found: {list(adata.obs.columns)}"
            )
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
    adata = adata[valid].copy()
    print(f"Loaded MTG AnnData: {adata.shape} from {path}")
    return adata


def _load_anndata_microglia(region):
    """Load Microglia per-cell AnnData from pre-built h5ad."""
    path = _find_anndata_microglia(region)
    if path is None:
        raise FileNotFoundError(
            f"Microglia AnnData not found: data/model_data/{region}_Microglia_AnnData_perCell.h5ad. "
            "Create it first (e.g. save_perCell_AnnData.py -cell_type Microglia -region <region>)."
        )
    adata = ad.read_h5ad(path)
    if "SampleName" not in adata.obs.columns:
        if "Donor_ID" in adata.obs.columns:
            adata.obs["SampleName"] = adata.obs["Donor_ID"].astype(str)
        elif "Donor ID" in adata.obs.columns:
            adata.obs["SampleName"] = adata.obs["Donor ID"].astype(str)
        else:
            raise ValueError(
                f"Microglia AnnData must have obs 'SampleName' or 'Donor_ID'/'Donor ID'. Found: {list(adata.obs.columns)}"
            )
    if "Thal" not in adata.obs.columns or "Ptau.Total.Tau..A.U.." not in adata.obs.columns:
        raise ValueError(
            f"Microglia AnnData must have obs 'Thal' and 'Ptau.Total.Tau..A.U..'. Found: {list(adata.obs.columns)}"
        )
    valid = (adata.obs["Thal"] != "unk").fillna(False)
    adata = adata[valid].copy()
    print(f"Loaded Microglia AnnData: {adata.shape} from {path}")
    return adata


def load_top_genes_from_dge_union(
    region: str,
    cell_type: str,
    top_k: int,
    dge_results_dir: str = "E:/sf2026/dge_results/dge_final",
) -> list[str]:
    """
    Load an ordered union of top-k DGE genes for a region from DGE result CSVs.

    We iterate DGE result files for the region, take the (filtered) top-k genes per file
    using the same criteria as utils.get_top_k_genes, and build an ordered union until
    we collect `top_k` unique genes.
    """
    dge_dir = Path(dge_results_dir)
    if not dge_dir.exists():
        raise FileNotFoundError(f"DGE results directory not found: {dge_results_dir}")

    # Match utils.get_DEGs() filename conventions
    if cell_type == "Microglia":
        pattern = f"dge_results_microglia_{region}_*.csv"
    else:
        pattern = f"dge_results_{region}_*.csv"

    dge_files = sorted(list(dge_dir.glob(pattern)))
    if not dge_files:
        raise FileNotFoundError(f"No DGE files found for region={region}, cell_type={cell_type} in {dge_results_dir}")

    # First, get union of all genes from all files (with their padj values)
    gene_to_min_padj = {}  # Track minimum padj for each gene across all files
    
    for dge_file in dge_files:
        df = pd.read_csv(dge_file, index_col=0)
        top_list, _top_set = get_top_k_genes(df, k=top_k, sort_by="padj")
        
        # Update padj values (keep minimum padj for each gene)
        for gene in top_list:
            padj_val = df.loc[gene, 'padj']
            if gene not in gene_to_min_padj or padj_val < gene_to_min_padj[gene]:
                gene_to_min_padj[gene] = padj_val
    
    # Sort union by padj (ascending = most significant first) and take top_k
    sorted_genes = sorted(gene_to_min_padj.items(), key=lambda x: x[1])
    return [gene for gene, _ in sorted_genes[:top_k]]


def create_anndata(region, cell_type='Astrocytes'):
    """Create AnnData for a region. Astrocytes: MTX + Excel. Microglia: pre-built h5ad. MTG: SEA_AD_Data/MTG."""
    if cell_type == "MTG":
        return _load_anndata_mtg()
    if cell_type == "Microglia":
        return _load_anndata_microglia(region)

    base_prefix = f"2025-11-16_{cell_type}_{region}"
    mtx_path, row_annotation_path, col_annotation_path = get_region_file_paths(
        region,
        data_dir=DATA_DIR,
        base_prefix=base_prefix
    )
    metadata_path = os.path.join(DATA_DIR, f"2025-11-16_{cell_type}_metadata.xlsx")
    tab_index = REGION_TO_TAB.get(region)
    
    # Load metadata
    metadata = read_excel_columns(
        metadata_path,
        columns=['cell_annotation', 'Path..Group.', 'SampleName', 'percent.mito', 'RIN', 
                'Total.Genes.Detected', 'Thal', 'Ptau.Total.Tau..A.U..'],
        sheet_name=tab_index
    )
    print(f"Loaded metadata: {metadata.shape}")
    
    # Read the MTX file (genes × cells format)
    matrix, gene_names, cell_names = read_mtx_file(
        mtx_path=str(mtx_path),
        row_annotation_path=str(row_annotation_path),
        col_annotation_path=str(col_annotation_path),
        transpose=False  # Keep as genes × cells
    )
    print(f"Loaded matrix shape: {matrix.shape} (genes × cells)")
    
    # Align metadata with cell_names (set cell_annotation as index)
    metadata = metadata.set_index('cell_annotation')
    metadata = metadata.reindex(cell_names)  # Align with cell_names order
    
    # Filter cells with unknown Thal category BEFORE creating AnnData
    valid_cell_mask = (metadata['Thal'] != 'unk').fillna(False)
    valid_cell_names = metadata[valid_cell_mask].index.tolist()
    valid_metadata = metadata[valid_cell_mask].copy()
    
    # Filter cell_names and matrix to match valid cells
    cell_indices_to_keep = [i for i, name in enumerate(cell_names) if name in valid_cell_names]
    filtered_cell_names = [cell_names[i] for i in cell_indices_to_keep]
    filtered_matrix = matrix[:, cell_indices_to_keep]  # genes × cells
    
    # Create AnnData object (transpose=True because matrix is genes × cells)
    adata = create_anndata_object(
        matrix=filtered_matrix,
        gene_names=gene_names,
        cell_names=filtered_cell_names,
        obs=valid_metadata,
        transpose=True  # Matrix is genes × cells, transpose for AnnData
    )
    
    return adata


def main(args):
    # Define regions based on cell type
    if args.cell_type == "Astrocytes":
        regions = ["EC", "ITG", "PFC", "V2", "V1"]
    elif args.cell_type == "Microglia":
        regions = ["CrossRegion", "EC", "ITG", "PFC", "V2", "V1"]
    elif args.cell_type == "MTG":
        regions = ["MTG"]
    else:
        raise ValueError(f"Invalid cell type: {args.cell_type}")
    
    print("="*80)
    print(f"Saving Raw Gene Expression Data for All Regions ({args.cell_type})")
    print(f"{args.n_folds}-Fold Cross-Validation Split with Target Variables")
    print(f"Regions to process: {regions}")
    print("="*80)
    
    # Iterate through all regions
    for region in regions:
        print(f"\n{'='*80}")
        print(f"Processing {region} ({args.cell_type})")
        print(f"{'='*80}")
        
        try:
            # Load AnnData for the region
            print(f"\nLoading data for {region}...")
            adata = create_anndata(region, cell_type=args.cell_type)
            print(f"AnnData shape: {adata.shape} (cells × genes)")
    
            # Get raw expression matrix (cells × genes)
            expression_matrix = np.array(adata.X.toarray(), dtype=np.float32)
            print(f"Expression matrix shape: {expression_matrix.shape}")
            
            # Select genes: DGE union for Astrocytes/Microglia; top-by-variance for MTG
            n_top_genes = args.top_k_genes
            if args.cell_type == "MTG":
                print(f"MTG: Selecting top {n_top_genes} genes by variance...")
                gene_var = np.var(expression_matrix, axis=0)
                top_idx = np.argsort(gene_var)[::-1][:n_top_genes]
                ordered_present = adata.var_names[top_idx].tolist()
                adata = adata[:, ordered_present].copy()
                expression_matrix = np.array(adata.X.toarray(), dtype=np.float32)
                print(f"  Using {expression_matrix.shape[1]} genes. Expression matrix shape: {expression_matrix.shape}")
            else:
                print(f"Selecting up to top {n_top_genes} genes from DGE union for region {region}...")
                dge_genes = load_top_genes_from_dge_union(
                    region=region,
                    cell_type=args.cell_type,
                    top_k=n_top_genes,
                    dge_results_dir=args.dge_results_dir,
                )
                ordered_present = [g for g in dge_genes if g in adata.var_names]
                if len(ordered_present) == 0:
                    raise ValueError(f"No DGE genes found in AnnData var_names for region={region}")
                adata = adata[:, ordered_present].copy()
                expression_matrix = np.array(adata.X.toarray(), dtype=np.float32)
                print(f"  Filtered to {expression_matrix.shape[1]} DGE-union genes. Expression matrix shape: {expression_matrix.shape}")
            
            # Extract target variables from obs
            thal_info = adata.obs['Thal']
            unique_patients = adata.obs['SampleName'].unique()
            if args.cell_type != "MTG":
                ptau_info = adata.obs['Ptau.Total.Tau..A.U..']
            else:
                braak_info = adata.obs['Braak']
            print(f"Number of unique patients: {len(unique_patients)}")
            
            # Verify that the number of cells in the expression matrix matches AnnData
            if expression_matrix.shape[0] != len(adata.obs):
                print(f"ERROR: Mismatch! Expression: {expression_matrix.shape[0]} cells, AnnData obs: {len(adata.obs)} cells")
                continue
            
            # Group by patient and aggregate cells (mean across cells per patient)
            all_patient_expressions = []
            all_patient_info = {}  # Dictionary: patient_name -> cell_names
            all_ptau = [] if args.cell_type != "MTG" else None
            all_thal = []
            all_braak = [] if args.cell_type == "MTG" else None
            
            for patient_name in unique_patients:
                patient_mask = adata.obs['SampleName'] == patient_name
                patient_expression = expression_matrix[patient_mask].astype(np.float32)  # (n_cells_patient, n_genes)
                patient_cell_names = adata.obs.index[patient_mask].tolist()
                
                # Aggregate cells per patient (mean across cells) to get (n_genes,) per patient
                patient_expression_aggregated = np.mean(patient_expression, axis=0).astype(np.float32)  # (n_genes,)
                
                all_patient_expressions.append(patient_expression_aggregated)
                all_patient_info[patient_name] = patient_cell_names
                all_thal.append(thal_info[patient_mask].iloc[0])
                if args.cell_type == "MTG":
                    all_braak.append(braak_info[patient_mask].iloc[0])
                else:
                    all_ptau.append(ptau_info[patient_mask].iloc[0])

            # Microglia: drop patients with NaN or invalid Braak/CERAD (same as create_data.py)
            if args.cell_type == "Microglia":
                import pandas as pd
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
                n_dropped = len(unique_patients) - len(valid_idx)
                if n_dropped > 0:
                    print(f"  Microglia: dropping {n_dropped} patients with NaN or invalid Braak/CERAD (keeping {len(valid_idx)})")
                unique_patients = np.asarray(unique_patients)[valid_idx]
                all_patient_expressions = [all_patient_expressions[i] for i in valid_idx]
                all_ptau = [all_ptau[i] for i in valid_idx]
                all_thal = [all_thal[i] for i in valid_idx]
                filtered_names = unique_patients.tolist() if hasattr(unique_patients, 'tolist') else list(unique_patients)
                all_patient_info = {n: all_patient_info[n] for n in filtered_names}

            # Split patients into folds
            n_patients = len(unique_patients)
            kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=42)
            
            expression_folds = []
            patient_info_folds = []
            ptau_folds = [] if args.cell_type != "MTG" else None
            thal_folds = []
            braak_folds = [] if args.cell_type == "MTG" else None
            
            print(f"\nSplitting {n_patients} patients into {args.n_folds} folds...")
            for fold_idx, (_, fold_patient_indices) in enumerate(kf.split(range(n_patients))):
                fold_expression_list = [all_patient_expressions[i] for i in fold_patient_indices]
                fold_expression = np.vstack(fold_expression_list).astype(np.float32)
                fold_thal = [all_thal[i] for i in fold_patient_indices]
                fold_patient_info = {unique_patients[i]: all_patient_info[unique_patients[i]] for i in fold_patient_indices}
                
                expression_folds.append(fold_expression)
                patient_info_folds.append(fold_patient_info)
                thal_folds.append(fold_thal)
                if args.cell_type == "MTG":
                    braak_folds.append([all_braak[i] for i in fold_patient_indices])
                else:
                    ptau_folds.append([all_ptau[i] for i in fold_patient_indices])
                
                n_fold_patients = len(fold_patient_indices)
                n_genes = fold_expression.shape[1]
                print(f"  Fold {fold_idx + 1}: {n_fold_patients} patients, {n_genes} genes per patient")
            
            # Save folds
            output_dir = os.path.join("data", "model_data")
            os.makedirs(output_dir, exist_ok=True)
            if args.cell_type == "MTG":
                output_path = os.path.join(output_dir, "MTG_raw_expression_data.pkl")
                with open(output_path, "wb") as f:
                    pickle.dump((expression_folds, patient_info_folds, thal_folds, braak_folds), f,
                                protocol=pickle.HIGHEST_PROTOCOL)
                print(f"\nSaved {args.n_folds} folds of MTG raw expression data to {output_path}")
                print(f"  Format: (expression_folds, patient_info_folds, thal_folds, braak_folds)")
            else:
                out_name = f"{region}_Microglia_raw_expression_data.pkl" if args.cell_type == "Microglia" else f"{region}_raw_expression_data.pkl"
                output_path = os.path.join(output_dir, out_name)
                with open(output_path, "wb") as f:
                    pickle.dump((expression_folds, patient_info_folds, ptau_folds, thal_folds), f,
                                protocol=pickle.HIGHEST_PROTOCOL)
                print(f"\nSaved {args.n_folds} folds of raw expression data to {output_path}")
                print(f"  Each fold contains: expression_folds, patient_info_folds, ptau_folds, thal_folds")
    
            # Clean up
            del expression_folds, patient_info_folds, thal_folds
            if args.cell_type == "MTG":
                del braak_folds, all_braak
            else:
                del ptau_folds, all_ptau
            del all_patient_expressions, all_patient_info, all_thal
            del adata, expression_matrix
            gc.collect()
            
            # Try to get memory info if psutil is available
            try:
                import psutil
                process = psutil.Process()
                memory_mb = process.memory_info().rss / (1024 * 1024)
                print(f"Memory after saving: {memory_mb:.2f} MB")
            except ImportError:
                print("Memory info not available (psutil not installed)")
        
        except Exception as e:
            print(f"ERROR: Failed to process {region}: {e}")
            print(f"Skipping {region} and continuing with next region...")
            continue
    
    print(f"\n{'='*80}")
    print(f"Completed processing all regions for {args.cell_type}")
    print(f"{'='*80}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Save raw gene expression data with Thal and ptau by patient')
    parser.add_argument('--cell_type', type=str, default='Astrocytes',
                       choices=['Astrocytes', 'Microglia', 'MTG'],
                       help='Cell type to process (Astrocytes, Microglia, or MTG). MTG uses data/SEA_AD_Data/MTG/MTG_AnnData.h5ad')
    parser.add_argument('--n_folds', type=int, default=8,
                       help='Number of folds for cross-validation split (default: 8)')
    parser.add_argument('--top_k_genes', type=int, default=3000,
                       help='Number of genes to keep from the region DGE union (default: 3000)')
    parser.add_argument('--dge_results_dir', type=str, default='E:/sf2026/dge_results/dge_final',
                       help='Directory containing DGE results CSVs (default: results/dge_final)')
    
    args = parser.parse_args()
    main(args)
