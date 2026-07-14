"""
Utility functions for reading and processing single-cell data.

This module provides helper functions for:
- Reading .mtx files (Matrix Market format)
- Creating AnnData objects from matrices
- Reading Excel files into pandas DataFrames
- Filtering cells based on metadata criteria
- Getting file paths for region-specific data
"""

import scipy.io
from scipy.sparse import csr_matrix
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Union, Tuple
import anndata
from collections import defaultdict
import re
import hashlib
import pickle
import os
import time
import config as cfg
from sklearn.decomposition import PCA

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

def read_mtx_file(mtx_path, row_annotation_path=None, col_annotation_path=None, 
                  transpose=True, make_gene_names_unique=True):
    """
    Read a .mtx file and optionally load row/column annotations.
    
    Parameters:
    -----------
    mtx_path : str
        Path to the .mtx file
    row_annotation_path : str, optional
        Path to file with row names (genes/features). 
        Each line should contain one name.
    col_annotation_path : str, optional
        Path to file with column names (cell barcodes).
        Each line should contain one name.
    transpose : bool, default=True
        If True, transpose the matrix so rows=genes, cols=cells.
        MTX format typically stores cells as rows and genes as columns.
    make_gene_names_unique : bool, default=True
        If True, ensure gene names are unique by appending _1, _2, etc.
    
    Returns:
    --------
    matrix : scipy.sparse.csr_matrix or scipy.sparse.csc_matrix
        Sparse matrix representation
    gene_names : array-like, optional
        Gene/feature names if row_annotation_path provided
    cell_names : array-like, optional
        Cell barcode names if col_annotation_path provided
    """
    
    # Read the .mtx file using scipy
    # This returns a COO (Coordinate) format sparse matrix
    matrix = scipy.io.mmread(mtx_path)
    print(f"Loaded matrix shape: {matrix.shape}")
    print(f"Number of non-zero entries: {matrix.nnz:,}")
    print(f"Sparsity: {(1 - matrix.nnz / (matrix.shape[0] * matrix.shape[1])) * 100:.2f}%")
    
    # Transpose if needed (MTX format often has cells as rows, genes as columns)
    if transpose:
        matrix = matrix.T
        print(f"Transposed matrix shape: {matrix.shape}")
    
    # Convert to CSR format for efficient row operations (or CSC for column operations)
    # CSR is typically preferred for most operations
    matrix = csr_matrix(matrix)
    
    # Load row annotations (genes/features) if provided
    gene_names = None
    if row_annotation_path:
        # Read gene names from file
        # Handle different formats (quoted strings, CSV, TSV, etc.)
        try:
            with open(row_annotation_path, 'r') as f:
                gene_names = [line.strip().strip('"') for line in f]
            
            # Handle unique gene names (common issue in scRNA-seq)
            if make_gene_names_unique:
                unique_names = []
                name_counts = {}
                for name in gene_names:
                    if name in name_counts:
                        name_counts[name] += 1
                        unique_names.append(f"{name}_{name_counts[name]}")
                    else:
                        name_counts[name] = 0
                        unique_names.append(name)
                gene_names = unique_names
            
            print(f"Loaded {len(gene_names)} gene names")
            
        except Exception as e:
            print(f"Warning: Could not load row annotations: {e}")
    
    # Load column annotations (cell barcodes) if provided
    cell_names = None
    if col_annotation_path:
        try:
            with open(col_annotation_path, 'r') as f:
                cell_names = [line.strip().strip('"') for line in f]
            print(f"Loaded {len(cell_names)} cell barcodes")
        except Exception as e:
            print(f"Warning: Could not load column annotations: {e}")
    
    return matrix, gene_names, cell_names

def read_excel_columns(
    file_path: str,
    columns: Optional[List[str]] = None,
    sheet_name: Optional[Union[str, int]] = 0,
    dtype: Optional[Dict[str, str]] = None,
    engine: Optional[str] = None,
    use_progress: bool = True
) -> pd.DataFrame:
    """
    Read a Microsoft Excel file and return a pandas DataFrame with selected columns.
    Results are cached based on all function arguments to speed up repeated calls.

    Parameters:
    -----------
    file_path : str
        Path to the Excel file (.xlsx, .xls, .xlsm).
    columns : list of str, optional
        Column names to select. If None, returns all columns.
    sheet_name : str or int, optional (default=0)
        Name or index of the sheet to read.
    dtype : dict, optional
        Optional mapping of column name -> dtype to enforce after reading.
    engine : str, optional
        Excel engine to use (e.g., 'openpyxl'). If None, pandas will auto-select.
    use_progress : bool, optional (default=True)
        If True, show a progress bar while reading using openpyxl+tqdm when available.

    Returns:
    --------
    df : pandas.DataFrame
        DataFrame containing the selected columns.
    """
    # Create cache directory if it doesn't exist
    cache_dir = Path(".cache")
    cache_dir.mkdir(exist_ok=True)
    
    # Generate cache key from all function arguments
    cache_key_parts = [
        str(file_path),
        str(sorted(columns) if columns else None),
        str(sheet_name),
        str(sorted(dtype.items()) if dtype else None),
        str(engine),
        str(use_progress)
    ]
    cache_key_str = "|".join(cache_key_parts)
    cache_key_hash = hashlib.md5(cache_key_str.encode()).hexdigest()
    cache_file = cache_dir / f"excel_cache_{cache_key_hash}.pkl"
    
    # Check if source file exists and get its modification time
    source_mtime = None
    if os.path.exists(file_path):
        source_mtime = os.path.getmtime(file_path)
    
    # Try to load from cache
    if cache_file.exists():
        try:
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                cached_mtime = cached_data.get('source_mtime')
                cached_df = cached_data.get('df')
                
                # Check if source file hasn't been modified since cache was created
                if cached_mtime is not None and source_mtime is not None:
                    if cached_mtime >= source_mtime:
                        return cached_df.copy()
                elif cached_mtime == source_mtime:  # Both None or same
                    return cached_df.copy()
        except Exception as e:
            # If cache loading fails, continue to read from file
            print(f"Warning: Failed to load cache ({e}), reading from file...")
    
    # Cache miss or invalid - read from file
    # If requested, try progress-enabled path using openpyxl + tqdm
    if use_progress:
        try:
            import openpyxl  # type: ignore
            try:
                from tqdm import tqdm  # type: ignore
            except Exception:
                tqdm = None  # fallback without visible progress

            wb = openpyxl.load_workbook(file_path, read_only=True, data_only=True)
            ws = wb.worksheets[sheet_name] if isinstance(sheet_name, int) else wb[sheet_name]

            header_row = next(ws.iter_rows(min_row=1, max_row=1, values_only=True))
            headers = list(header_row)

            # Determine which columns to keep
            if columns is not None:
                name_to_idx = {name: idx for idx, name in enumerate(headers)}
                missing = [c for c in columns if c not in name_to_idx]
                if missing:
                    print(f"Warning: Missing columns not found in Excel: {missing}")
                selected = [c for c in columns if c in name_to_idx]
                selected_indices = [name_to_idx[c] for c in selected]
                out_headers = selected
            else:
                selected_indices = list(range(len(headers)))
                out_headers = headers

            total_rows = max(ws.max_row - 1, 0)
            progress_iter = tqdm(total=total_rows, desc="Reading Excel", unit="row") if tqdm else None

            records = []
            # Iterate data rows
            for row in ws.iter_rows(min_row=2, values_only=True):
                if row is None:
                    if progress_iter:
                        progress_iter.update(1)
                    continue
                # Ensure row has enough length
                row_values = list(row)
                if len(row_values) < len(headers):
                    row_values += [None] * (len(headers) - len(row_values))
                # Select indices
                record = [row_values[i] for i in selected_indices]
                records.append(record)
                if progress_iter:
                    progress_iter.update(1)

            if progress_iter:
                progress_iter.close()
            wb.close()

            df = pd.DataFrame.from_records(records, columns=out_headers)
        except Exception as e:
            print(f"Note: Falling back to pandas.read_excel (reason: {e})")
            df = pd.read_excel(file_path, sheet_name=sheet_name, engine=engine)
    else:
        # Read the sheet via pandas directly
        df = pd.read_excel(file_path, sheet_name=sheet_name, engine=engine)

    # Select requested columns if provided
    if columns is not None:
        missing = [c for c in columns if c not in df.columns]
        if missing:
            print(f"Warning: Missing columns not found in Excel: {missing}")
        present = [c for c in columns if c in df.columns]
        df = df[present]

    # Apply dtype conversions if requested
    if dtype:
        for col, target_dtype in dtype.items():
            if col in df.columns:
                try:
                    df[col] = df[col].astype(target_dtype)
                except Exception as e:
                    print(f"Warning: Could not cast column '{col}' to {target_dtype}: {e}")

    # Save to cache
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump({
                'df': df,
                'source_mtime': source_mtime
            }, f)
    except Exception as e:
        # If caching fails, continue without cache
        print(f"Warning: Failed to save cache ({e})")

    return df

def create_anndata_object(matrix, gene_names=None, cell_names=None, obs=None, transpose=False):
    """
    Create an AnnData object (standard format for single-cell analysis).
    Requires scanpy/anndata package.
    
    Parameters:
    -----------
    matrix : scipy.sparse matrix
        Expression matrix (assumed to be genes × cells if transpose=False)
    gene_names : array-like, optional
        Gene names
    cell_names : array-like, optional
        Cell barcodes
    obs : pandas.DataFrame, optional
        Cell metadata (must have same number of rows as cells)
    transpose : bool, default=False
        If True, transpose the matrix before creating AnnData
        (AnnData expects cells as rows, genes as columns)
    
    Returns:
    --------
    adata : anndata.AnnData
        AnnData object ready for analysis
    """
    try:
        import anndata as ad
        
        # Create AnnData object
        # AnnData expects cells as rows (obs), genes as columns (var)
        if transpose:
            adata = ad.AnnData(matrix.T)
        else:
            # Matrix is already in cells × genes format
            adata = ad.AnnData(matrix)
        
        if gene_names is not None:
            adata.var_names = gene_names
        if cell_names is not None:
            adata.obs_names = cell_names
        if obs is not None:
            adata.obs = obs

        # Read gene annotations from gene_annotations.pkl
        gene_annotations_file = os.path.join("data", "gene_annotations.pkl")
        with open(gene_annotations_file, 'rb') as f:
            gene_annotations = pickle.load(f)
        # Add Gene type to var (gene-level metadata), not obs (cell-level metadata)
        gene_type_dict = gene_annotations.set_index('Gene name')['Gene type'].to_dict()
        adata.var['Gene type'] = [gene_type_dict.get(gene, 'unknown') for gene in adata.var_names]
        
        return adata
    
    except ImportError:
        print("anndata package not installed. Install with: pip install anndata scanpy")
        return None

def filter_anndata_object(adata, min_genes=200, min_cells=3, min_counts=None, 
                          max_counts=None, mito_max=None, RIN_threshold=None, gene_type=None):
    """
    Filter an AnnData object based on the number of genes and cells.
    
    Parameters:
    -----------
    adata : anndata.AnnData
        AnnData object to filter
    min_genes : int, default=200
        Minimum number of genes detected per cell
    min_cells : int, default=3
        Minimum number of cells expressing each gene (0 = no gene filtering)
    min_counts : int, optional
        Minimum total counts (UMIs) per cell
    max_counts : int, optional
        Maximum total counts (UMIs) per cell    
    Returns:
    --------
    adata : anndata.AnnData
        Filtered AnnData object
    """
    import scanpy as sc
    sc.pp.filter_cells(adata, min_genes=min_genes, min_counts=min_counts, max_counts=max_counts)
    # Only filter genes if min_cells > 0
    if min_cells > 0:
        sc.pp.filter_genes(adata, min_cells=min_cells)
    if mito_max is not None:
        mask = (adata.obs['percent.mito'] < mito_max).fillna(False)
        adata = adata[mask].copy()
    if RIN_threshold is not None:
        # Exclude cells where RIN == 'unk', then convert to numeric for comparison
        mask_unk = (adata.obs['RIN'] != 'unk')
        rin_numeric = pd.to_numeric(adata.obs['RIN'], errors='coerce')
        mask = mask_unk & (rin_numeric > RIN_threshold).fillna(False)
        adata = adata[mask].copy()

    if gene_type is not None:
        # Filter genes (columns) by gene type, not cells (rows)
        mask = (adata.var['Gene type'] == gene_type).fillna(False)
        adata = adata[:, mask].copy()

    return adata

# Shuffle an AnnData object based on the seed
def shuffle_data(adata: anndata.AnnData, seed: int) -> anndata.AnnData:
    rng = np.random.RandomState(seed)
    n_cells = adata.n_obs
    shuffled_indices = rng.permutation(n_cells)
    # Create a new AnnData object with shuffled data by reordering underlying arrays
    # This avoids issues with integer vs string observation names
    shuffled_adata = anndata.AnnData(
        X=adata.X[shuffled_indices] if adata.X is not None else None,
        obs=adata.obs.iloc[shuffled_indices].copy(),
        var=adata.var.copy(),
        obsm={k: v[shuffled_indices] for k, v in adata.obsm.items()} if adata.obsm else {},
        varm=adata.varm.copy() if adata.varm else {},
        uns=adata.uns.copy() if adata.uns else {},
        layers={k: v[shuffled_indices] for k, v in adata.layers.items()} if adata.layers else {}
    )
    return shuffled_adata

def aggregate_cells_into_pseudobulk(adata, target_cells_per_bin, filter_patients_cell_threshold=80, seed=100, patient_col='SampleName'):
    """
    Aggregate cells into pseudobulk samples.
    
    CRITICAL: Aggregate cells into pseudobulk samples if dataset is too large
    DESeq2 is designed for bulk RNA-seq with few samples (3-10 per condition)
    Single-cell data with thousands of cells causes memory errors
    Pseudobulk aggregation is standard practice for single-cell DGE analysis
    
    Parameters:
    -----------
    adata : AnnData
        AnnData object with cells to aggregate
    target_cells_per_bin : int
        Target number of cells per bin
    filter_patients_cell_threshold : int, default=80
        Minimum number of cells per patient to include
    seed : int, default=100
        Random seed for shuffling cells
    patient_col : str, default='SampleName'
        Column name in adata.obs containing patient/donor identifiers
        
    Returns:
    --------
    adata_pseudobulk : AnnData
        AnnData object with pseudobulk samples
    """
    from scipy.sparse import csr_matrix
    from anndata import AnnData
        
    pseudobulk_data = []
    pseudobulk_metadata = []
    
    if patient_col not in adata.obs.columns:
        raise ValueError(f"Patient column '{patient_col}' not found in adata.obs. Available columns: {list(adata.obs.columns)}")
    
    unique_patients = adata.obs[patient_col].dropna().unique()
    total_bins = 0
                
    for patient_name in unique_patients:
        patient_mask = adata.obs[patient_col] == patient_name
        patient_cells = adata[patient_mask]
        num_cells = patient_cells.n_obs
            
        if num_cells < filter_patients_cell_threshold:
            print(f"  {patient_name}: {num_cells} cells (skipped, < {filter_patients_cell_threshold} cells)")
            continue
            
        # Shuffle cells for this patient using the seed
        if seed != 100:
            patient_cells = shuffle_data(patient_cells, seed)
            
        num_bins = max(1, int(np.ceil(num_cells / target_cells_per_bin)))
        total_bins += num_bins
        print(f"  {patient_name}: {num_cells} cells {num_bins} bins")
        
        # Get patient metadata (same across all cells for this patient)
        # Use first cell's values since they're consistent across all cells
        patient_metadata = patient_cells.obs.iloc[0].to_dict()
        
        # Distribute cells evenly across bins
        base_cells_per_bin = num_cells // num_bins
        remainder = num_cells % num_bins
        
        # Work directly with underlying arrays to avoid AnnData indexing issues
        # Use raw UMI counts from layer if present (e.g. SEA_AD: UMIs, ROSMAP: counts), else .X
        layers = getattr(patient_cells, "layers", {})
        if "UMIs" in layers:
            count_matrix = patient_cells.layers["UMIs"]
        elif "counts" in layers:
            count_matrix = patient_cells.layers["counts"]
        else:
            count_matrix = patient_cells.X
        
        cell_idx = 0
        for bin_idx in range(num_bins):
            bin_size = base_cells_per_bin + (1 if bin_idx < remainder else 0)
            start_idx = cell_idx
            end_idx = cell_idx + bin_size
            cell_idx = end_idx
            
            X_subset = count_matrix[start_idx:end_idx]
            
            # Calculate bin counts (sum across cells, keep gene dimension)
            if hasattr(X_subset, 'toarray'):
                # Sparse matrix
                bin_counts = X_subset.sum(axis=0).A1
            else:
                # Dense matrix
                bin_counts = X_subset.sum(axis=0)
            
            # Build metadata dictionary with all patient-level metadata
            metadata_dict = patient_metadata.copy()
            metadata_dict['_sample_id'] = f"{patient_name}_bin{bin_idx+1}"
            
            pseudobulk_data.append(bin_counts)
            pseudobulk_metadata.append(metadata_dict)
    
    print(f"  Total bins: {total_bins}")
    print(f"\nTotal pseudobulk samples: {len(pseudobulk_data)}")
    
    pseudobulk_matrix = np.array(pseudobulk_data)
    pseudobulk_obs = pd.DataFrame(pseudobulk_metadata)
    pseudobulk_obs.index = pseudobulk_obs['_sample_id']
    
    adata_pseudobulk = AnnData(
        X=csr_matrix(pseudobulk_matrix),
        obs=pseudobulk_obs,
        var=adata.var.copy()
    )
    
    return adata_pseudobulk

def get_region_file_paths(region, data_dir="data", base_prefix="2025-10-22_Astrocytes_{region}"):
    """
    Get file paths for region-specific matrix, row, and column annotation files.
    
    Parameters:
    -----------
    region : str
        Region name (e.g., 'EC', 'ITG', 'PFC', 'V1', 'V2')
    data_dir : str or Path, default="data"
        Base directory containing region files
    base_prefix : str, default="2025-10-22_Astrocytes_{region}"
        File naming pattern with {region} placeholder
    
    Returns:
    --------
    mtx_path : Path
        Path to matrix.mtx file
    row_path : Path
        Path to row_annotation.txt file
    col_path : Path
        Path to cell_annotation.txt file
    """
    data_dir = Path(data_dir)
    prefix = base_prefix.format(region=region)
    mtx_path = (data_dir / f"{prefix}_matrix.mtx").resolve()
    row_path = (data_dir / f"{prefix}_row_annotation.txt").resolve()
    col_path = (data_dir / f"{prefix}_cell_annotation.txt").resolve()
    return mtx_path, row_path, col_path

def filter_cells(matrix, cell_names, metadata, mito_max=0.15, extra_filter = False):
    """
    Filter cells based on mitochondrial percentage threshold.
    
    Parameters:
    -----------
    matrix : scipy.sparse matrix
        Expression matrix (genes × cells)
    cell_names : list
        List of cell names/barcodes
    metadata : pandas.DataFrame
        Metadata DataFrame with 'cell_annotation' and 'percent.mito' columns
    mito_max : float, default=15.0
        Maximum mitochondrial percentage threshold
    
    Returns:
    --------
    filtered_matrix : scipy.sparse matrix
        Filtered expression matrix
    filtered_cell_names : list
        List of cell names that passed filters
    filtered_metadata : pandas.DataFrame
        Filtered metadata DataFrame
    """
    # Ensure cell_names matches matrix columns
    n_cols = matrix.shape[1]
    if len(cell_names) != n_cols:
        print(f"  Warning: cell_names length ({len(cell_names)}) != matrix columns ({n_cols})")
        cell_names = cell_names[:n_cols]  # Truncate to match matrix
    
    filtered_metadata = metadata[metadata['cell_annotation'].isin(cell_names)].copy()
    filtered_metadata = filtered_metadata.set_index('cell_annotation').reindex(cell_names)
    n_before_filter = len(filtered_metadata)

    # Debug: Check percent.mito values
    mito_values = filtered_metadata['percent.mito']
    print(f"  Mitochondrial percentage stats:")
    print(f"    Min: {mito_values.min():.2f}, Max: {mito_values.max():.2f}, Mean: {mito_values.mean():.2f}")
    print(f"    NaN count: {mito_values.isna().sum()}")
    print(f"    Threshold: {mito_max}")

    # Filter by mitochondrial percentage
    mito_mask = (filtered_metadata['percent.mito'] < mito_max).fillna(False)
    if extra_filter:
        # Convert to numeric, coercing errors to NaN
        RIN_numeric = pd.to_numeric(filtered_metadata['RIN'], errors='coerce')
        TotalGenes_numeric = pd.to_numeric(filtered_metadata['Total.Genes.Detected'], errors='coerce')
        RIN_genes_mask = ((RIN_numeric > 7.5) | (TotalGenes_numeric > 30000)).fillna(False)
        combined_mask = mito_mask & RIN_genes_mask
    else:
        combined_mask = mito_mask
    filtered_metadata = filtered_metadata[combined_mask].copy()
    
    print(f"  Before: {n_before_filter:,} cells")
    print(f"  After: {len(filtered_metadata):,} cells")
    print(f"  Filtered out: {n_before_filter - len(filtered_metadata):,} cells")
    
    # Get cell indices to keep (cells that passed filters)
    cells_to_keep = set(filtered_metadata.index.dropna().tolist())
    cell_indices_to_keep = [i for i, name in enumerate(cell_names) if name in cells_to_keep]
    
    # Ensure indices are within matrix bounds
    cell_indices_to_keep = [i for i in cell_indices_to_keep if 0 <= i < n_cols]
    
    # Filter matrix and cell names
    filtered_matrix = matrix[:, cell_indices_to_keep]
    filtered_cell_names = [cell_names[i] for i in cell_indices_to_keep]
    
    return filtered_matrix, filtered_cell_names, filtered_metadata

def min_max_normalize(matrix):
    mat = matrix.tocoo(copy=True).astype(float) #coordinate format
    # Convert to CSR for efficient row operations, then get max/min as dense arrays
    matrix_csr = matrix.tocsr()
    row_max = np.array(matrix_csr.max(axis=1).toarray()).ravel()
    row_min = np.array(matrix_csr.min(axis=1).toarray()).ravel()
    denom = row_max - row_min
    denom[denom == 0] = 1.0  # Avoid divide by zero
    mat.data = (mat.data - row_min[mat.row]) / denom[mat.row]
    return mat.tocsr()

def get_top_k_genes(df: pd.DataFrame, k: int, sort_by: str = 'padj', padj_threshold: float = 0.05, log2fc_threshold: float = 1) -> set:
    df_clean = df[df[sort_by].notna()].copy()
    if len(df_clean) == 0:
        return set()
    df_clean = df_clean[(df_clean['padj'] < padj_threshold) & (abs(df_clean['log2FoldChange']) > log2fc_threshold)]
    ascending = (sort_by == 'padj')
    df_sorted = df_clean.sort_values(sort_by, ascending=ascending)
    return list(df_sorted.head(k).index), set(df_sorted.head(k).index)

def get_DEGs(region, top_k=500, cell_type="Astrocytes", disable_intersection=False, dge_results_dir=None):
    """
    Get the intersection of top K genes (by padj) across all 6 comparisons for a region.
    
    Parameters:
    -----------
    region : str
        Region name (EC, ITG, PFC, V1, V2, CrossRegion)
    top_k : int
        Number of top genes to select from each file (sorted by padj)
    cell_type : str
        Cell type (Astrocytes or Microglia)
    disable_intersection : bool
        Whether to disable intersection of gene sets
    dge_results_dir : str, optional
        Directory containing DGE results files. If None, defaults to "results/dge_final"
    
    Returns:
    --------
    deg_genes : list
        List of gene names that are in the top K genes of all comparisons (intersection)
    """
    if dge_results_dir is None:
        dge_results_dir = "results/dge_final"
    dge_dir = Path(dge_results_dir)
    
    # Find all DGE results files for this region
    # Check if using protein coding directory (format: dge_results_PC_{region}_*.csv)
    is_protein_coding = "protein_coding" in dge_results_dir
    
    if cell_type == "Microglia":
        pattern = f"dge_results_microglia_{region}_*.csv"
    elif is_protein_coding:
        pattern = f"dge_results_PC_{region}_*.csv"
    else:
        pattern = f"dge_results_{region}_*.csv"
    #pattern = f"dge_results_{region}_*_2_vs_4.csv"
    dge_files = sorted(list(dge_dir.glob(pattern)))
    
    # Collect top K genes from each comparison
    gene_sets = []
    for dge_file in dge_files:
        try:
            df = pd.read_csv(dge_file, index_col=0)
            # Get top K genes sorted by padj
            top_genes_list, top_genes_set = get_top_k_genes(df, top_k, sort_by='padj')
            gene_sets.append(top_genes_set)
            print(f"  {dge_file.name}: {len(top_genes_set)} top genes")
            top_10_genes = top_genes_list[:min(len(top_genes_list), 10)]
            print(f"  First 10 genes: {top_10_genes}")
            print(f"  Gene types: {[cfg.gene_annotations_dict[gene] for gene in top_10_genes]}")

        except Exception as e:
            print(f"Warning: Could not load {dge_file}: {e}")
            continue
    
    if not gene_sets:
        print(f"No gene sets collected for region {region}")
        return []
 
    print(f"Total comparisons: {len(gene_sets)}")
    print(f"Gene set sizes: {[len(gs) for gs in gene_sets]}")
    
    # Return intersection of all top K gene sets
    # gene_sets is a list of sets, so we unpack with * to pass them as separate arguments
    if disable_intersection:
        common_genes = set.union(*gene_sets)
        print(f"Union size: {len(common_genes)}")

    else:
        common_genes = set.intersection(*gene_sets) if len(gene_sets) > 1 else gene_sets[0]
        print(f"Intersection size: {len(common_genes)}")

    ## Check intersection with genes from paper (if available for this region)
    if region in cfg.GENES_FROM_PAPER:
        genes_from_paper = cfg.GENES_FROM_PAPER[region]
        common_genes_with_paper = common_genes.intersection(genes_from_paper)
        print(f"\033[92mIntersection with genes from paper: {len(common_genes_with_paper)} out of {len(genes_from_paper)}\033[0m")
        for gene in common_genes_with_paper:
            print(f"\033[92m{gene}\033[0m")
    else:
        print(f"Note: No genes from paper defined for region '{region}'")
    
    return list(common_genes)

def get_grn_file_paths(region, cell_type="Astrocytes", data_dir="data"):
    """
    Construct file paths for GRN files (GRN.tsv, matrix.csv, AnnData.h5ad) based on region and cell type.
    For region "MTG" (SEA AD), uses data/GRN/MTG/ for GRN and data/GRN/ for matrix and AnnData.

    Parameters:
    -----------
    region : str
        Region name (e.g., "V1", "EC", "ITG", "PFC", "V2", "MTG")
    cell_type : str
        Cell type (e.g., "Astrocytes", "Microglia"). Default: "Astrocytes"
    data_dir : str
        Base data directory. Default: "data"
        
    Returns:
    --------
    tuple
        Tuple of (grn_file, matrix_file, anndata_file, grn_output, matrix_output, anndata_output)
        Output files are the same as input files but with "_ensembl" suffix before extension
    """
    # SEA AD MTG: GRN in data/GRN/MTG/, matrix and AnnData in data/GRN/
    if region == "MTG":
        grn_dir = os.path.join(data_dir, "GRN", "MTG")
        grn_file = os.path.join(grn_dir, "MTG_GRN.tsv")
        matrix_file = os.path.join(grn_dir, "MTG_matrix.csv")
        anndata_file = os.path.join(grn_dir, "MTG_AnnData.h5ad")
        grn_output = os.path.join(grn_dir, "MTG_GRN_ensembl.tsv")
        matrix_output = os.path.join(grn_dir, "MTG_matrix_ensembl.csv")
        anndata_output = os.path.join(grn_dir, "MTG_AnnData_ensembl.h5ad")
        return grn_file, matrix_file, anndata_file, grn_output, matrix_output, anndata_output

    grn_dir = os.path.join(data_dir, 'GRN', cell_type)
    
    # Construct filenames based on cell_type (Microglia has "_Microglia_" in filename)
    if cell_type == "Microglia":
        grn_file = os.path.join(grn_dir, f'{region}_Microglia_GRN.tsv')
        matrix_file = os.path.join(grn_dir, f'{region}_Microglia_matrix.csv')
        anndata_file = os.path.join(grn_dir, f'{region}_Microglia_AnnData.h5ad')
        # Output files
        grn_output = os.path.join(grn_dir, f'{region}_Microglia_GRN_ensembl.tsv')
        matrix_output = os.path.join(grn_dir, f'{region}_Microglia_matrix_ensembl.csv')
        anndata_output = os.path.join(grn_dir, f'{region}_Microglia_AnnData_ensembl.h5ad')
    else:
        grn_file = os.path.join(grn_dir, f'{region}_GRN.tsv')
        matrix_file = os.path.join(grn_dir, f'{region}_matrix.csv')
        anndata_file = os.path.join(grn_dir, f'{region}_AnnData.h5ad')
        # Output files
        grn_output = os.path.join(grn_dir, f'{region}_GRN_ensembl.tsv')
        matrix_output = os.path.join(grn_dir, f'{region}_matrix_ensembl.csv')
        anndata_output = os.path.join(grn_dir, f'{region}_AnnData_ensembl.h5ad')
    
    return grn_file, matrix_file, anndata_file, grn_output, matrix_output, anndata_output


def get_rosmap_file_paths(cell_type="Astrocytes", data_dir="data"):
    """
    Construct file paths for ROSMAP GRN inputs and raw cell-type AnnData.

    GRN pipeline (create_GRN_data_ROSMAP.py) writes under data/GRN/ROSMAP/<CellType>/.
    Raw processed AnnData lives in data/ROSMAP_Data/<CellType>.h5ad.

    Returns
    -------
    tuple
        (grn_file, matrix_file, anndata_file, grn_output, matrix_output, anndata_output,
         raw_anndata_file, raw_anndata_output)
    """
    grn_dir = os.path.join(data_dir, "GRN", "ROSMAP", cell_type)
    grn_file = os.path.join(grn_dir, f"ROSMAP_{cell_type}_GRN.tsv")
    matrix_file = os.path.join(grn_dir, f"{cell_type}_matrix.csv")
    anndata_file = os.path.join(grn_dir, f"{cell_type}_pseudobulk_AnnData.h5ad")
    grn_output = os.path.join(grn_dir, f"ROSMAP_{cell_type}_GRN_ensembl.tsv")
    matrix_output = os.path.join(grn_dir, f"{cell_type}_matrix_ensembl.csv")
    anndata_output = os.path.join(grn_dir, f"{cell_type}_pseudobulk_AnnData_ensembl.h5ad")
    raw_anndata_file = os.path.join(data_dir, "ROSMAP_Data", f"{cell_type}.h5ad")
    raw_anndata_output = os.path.join(data_dir, "ROSMAP_Data", f"{cell_type}_ensembl.h5ad")
    return (
        grn_file, matrix_file, anndata_file,
        grn_output, matrix_output, anndata_output,
        raw_anndata_file, raw_anndata_output,
    )


def create_color_mapping(attribute_values, attribute_type='auto', custom_colors=None):
    """
    Create color mapping for cell embeddings based on different attributes.
    
    Parameters:
    -----------
    attribute_values : list
        List of attribute values to color by. Can be:
        - List of cell labels (format: "region_celltype" or just "region")
        - List of conditions (e.g., ['1', '2', '3', '4'])
        - List of regions
        - List of any categorical values
    attribute_type : str, default='auto'
        Type of attribute: 'cell_type', 'region', 'condition', or 'auto'
        If 'auto', tries to infer from attribute_values
    custom_colors : dict, optional
        Custom color mapping dictionary. If provided, uses these colors instead of defaults.
        Example: {'Astrocytes': '#1f77b4', 'Microglia': '#ff7f0e'}
    
    Returns:
    --------
    point_colors : list
        List of colors for each cell
    color_map : dict
        Dictionary mapping attribute value to color
    unique_values : list
        List of unique attribute values found
    """
    import matplotlib.pyplot as plt
    
    # Auto-detect attribute type if needed
    if attribute_type == 'auto':
        # Check if it looks like cell labels (contains underscores and cell type keywords)
        sample = attribute_values[0] if attribute_values else ''
        if isinstance(sample, str) and ('Astrocyte' in sample or 'Microglia' in sample or '_' in sample):
            # Try to determine if it's cell_type or region
            if any('_' in str(v) and any(ct in str(v) for ct in ['Astrocytes', 'Microglia']) for v in attribute_values[:10]):
                attribute_type = 'cell_type'
            else:
                attribute_type = 'region'
        elif isinstance(sample, str) and sample.isdigit():
            attribute_type = 'condition'
        else:
            attribute_type = 'generic'
    
    # Extract values based on attribute type
    if attribute_type == 'cell_type':
        # Parse cell_labels to extract cell_type
        extracted_values = []
        for label in attribute_values:
            if isinstance(label, str):
                if '_' in label:
                    parts = label.rsplit('_', 1)
                    if parts[-1] in ['Astrocytes', 'Microglia']:
                        extracted_values.append(parts[-1])
                    elif 'Microglia' in label:
                        extracted_values.append('Microglia')
                    elif 'Astrocyte' in label:
                        extracted_values.append('Astrocytes')
                    else:
                        extracted_values.append('Astrocytes')  # Default
                else:
                    extracted_values.append('Astrocytes')  # Default
            else:
                extracted_values.append(str(label))
        attribute_values = extracted_values
        
        # Default colors for cell types
        if custom_colors is None:
            custom_colors = {
                'Astrocytes': '#1f77b4',      # Blue
                'Microglia': '#ff7f0e',       # Orange
                'Neuron': '#2ca02c',          # Green
                'Oligodendrocyte': '#d62728',  # Red
                'Endothelial': '#9467bd',     # Purple
            }
    
    elif attribute_type == 'region':
        # Parse cell_labels to extract region
        extracted_values = []
        for label in attribute_values:
            if isinstance(label, str):
                if '_' in label:
                    parts = label.rsplit('_', 1)
                    if parts[-1] in ['Astrocytes', 'Microglia']:
                        extracted_values.append(parts[0])
                    else:
                        extracted_values.append(label)
                else:
                    extracted_values.append(label)
            else:
                extracted_values.append(str(label))
        attribute_values = extracted_values
    
    elif attribute_type == 'condition':
        # Conditions are usually already in the right format
        # Default colors for conditions
        if custom_colors is None:
            custom_colors = {
                '1': 'green',
                '2': 'lightgreen',
                '3': 'orange',
                '4': 'red'
            }
    
    # Convert all values to strings for consistency
    attribute_values = [str(v) if v is not None else 'None' for v in attribute_values]
    
    # Get unique values
    unique_values = sorted(set(attribute_values))
    print(f"  Found {len(unique_values)} unique {attribute_type} values: {unique_values[:10]}{'...' if len(unique_values) > 10 else ''}")
    
    # Create color map
    cmap = plt.cm.get_cmap('tab20')
    color_map = {}
    
    for idx, value in enumerate(unique_values):
        if custom_colors and value in custom_colors:
            color_map[value] = custom_colors[value]
        else:
            # Use colormap for values not in custom colors
            color_map[value] = cmap(idx / max(len(unique_values), 1))
    
    # Map each cell to its color
    point_colors = [color_map.get(val, '#808080') for val in attribute_values]
    
    # Return extracted values for counting purposes
    return point_colors, color_map, unique_values, attribute_values

def extract_best_per_attn_loss_dim(csv_path: str, embedding: bool = None, microglia: bool = False):
    """Read experiments CSV and return the best model (by F1) per group.
    If embedding=True or CSV has 'gene_encoder', group by (gene_encoder, attention, loss_ptau, pca_components).
    Otherwise (expression) group by (attention, loss_ptau, pca_components).
    """
    from read_tensorboard_files import count_overfitts
    df = pd.read_csv(csv_path)
    if "avg_f1" not in df.columns:
        raise ValueError(f"CSV must have column 'avg_f1'. Found: {list(df.columns)}")
    df = df.dropna(subset=["avg_f1"])
    use_embedding = embedding if embedding is not None else ("gene_encoder" in df.columns)
    if use_embedding:
        if not microglia:
            group_cols = ["gene_encoder", "attention", "loss_ptau", "pca_components"]
        else:
            group_cols = ["gene_encoder", "attention", "pca_components"]
    else:
        if not microglia:
            group_cols = ["attention", "loss_ptau", "pca_components"]
        else:
            group_cols = ["attention", "pca_components"]
    for c in group_cols:
        if c not in df.columns:
            raise ValueError(f"CSV must have column '{c}'. Found: {list(df.columns)}")
    df = df.copy()
    # Normalize no-PCA for expression (embedding uses 2,4,8,16 only)
    if not use_embedding:
        no_pca = (df["pca_components"].isna()) | (df["pca_components"].astype(str).str.lower().isin(("none", "nan", "")))
        df = df.astype({"pca_components": object})
        df.loc[no_pca, "pca_components"] = "None"
    # Best row per group by F1
    idx = df.groupby(group_cols)["avg_f1"].idxmax()
    overfits = []
    for i in idx:
        row = df.loc[i]
        logdir = tensorboard_dir_for_row(row, microglia=microglia)
        try:
            overfit = count_overfitts(logdir)
        except Exception as e:
            print(f"Warning: Could not check overfitting for {logdir}: {e}")
            overfit = None
        overfits.append(overfit)
    df.loc[idx, "overfit"] = overfits
    best = df.loc[idx].sort_values(group_cols).reset_index(drop=True)
    return best


def tensorboard_dir_for_row(row, log_base=None, microglia=False):
    """Build TensorBoard directory path for a best-model row.
    Example: I:\\sf2026\\Astrocytes_V2_run2\\V2_None_mse_geneenc_drop0.5_wd1.0_geTrue_attnFalse_mse_pca16_all_strong_testfold8
    """
    region = row["region"]
    if log_base is None:
        log_base = os.path.join("I:\\", "sf2026", f"Microglia_{region}")
    if region != "EC":
        pooling = row.get("pooling", "None")
        if not microglia:
            loss_ptau = row["loss_ptau"]
        else:
            loss_ptau = ""
        gene_encoder = row.get("gene_encoder")
        pooling = "mean" if not gene_encoder else pooling
        attention = row.get("attention")
        dropout = row["dropout"]
        wd = row["weight_decay"]
        exp_id = row.get("exp_id", row.get("run", ""))
        if not microglia:
            mname = "_geneenc" if gene_encoder else ("_attn" if attention else "")
            name = f"{region}_{pooling}_{loss_ptau}{mname}_drop{dropout}_wd{wd}_{exp_id}"
        else:
            mname = "_ge" if gene_encoder else ("_attn" if attention else "")
            name = f"{region}_Microglia{mname}_drop{dropout}_wd{wd}_{exp_id}"
        return os.path.join(log_base, name)
    else:
        exp_id = row.get("exp_id", row.get("run", ""))
        return os.path.join(log_base, exp_id)


def scale_embeddings(embeddings_matrix, expression_matrix, n_components=None, method: str = "linear"):
    # expression_matrix: (n_cells, n_genes) - no need to transpose
    # embedding_matrix: (n_genes, n_dims)
    # scaled_embeddings_matrix: (n_cells, n_genes, n_dims)

    # PCA reduction on embeddings matrix
    pca = PCA(n_components=n_components, random_state=42)
    embeddings_matrix = pca.fit_transform(embeddings_matrix)

    m = (method or "linear").strip().lower()
    if m == "linear":
        expr = expression_matrix
    elif m in ("sqrt", "sqrt1p"):
        expr = np.sqrt(expression_matrix + 1e-8)
    elif m in ("log", "log1p"):
        expr = np.log1p(expression_matrix)
    else:
        raise ValueError(f"Unknown scaling method: {method}. Use linear, sqrt, or log1p.")

    # Using similar function to example (expression = d, embedding = A)
    scaled_embeddings_matrix = expr[:, :, None] * embeddings_matrix[None, :, :]
    scaled_embeddings_matrix = scaled_embeddings_matrix.astype('float16')
    return scaled_embeddings_matrix


def get_top_ranked_genes(csv_path, top_k=5):
    """Return top K gene names from ranked_genes.csv (by rank)."""
    df = pd.read_csv(csv_path)
    if "gene" not in df.columns:
        raise ValueError(f"ranked_genes CSV must have column 'gene'. Found: {list(df.columns)}")
    if "rank" in df.columns:
        df = df.sort_values("rank")
    return df["gene"].head(top_k).astype(str).str.strip().tolist()


def get_top_dge_union_genes(region="EC", cell_type="Astrocytes", top_k=5, dge_results_dir=None):
    """Return top K gene names from union of region DGE results (by best padj)."""
    default_dge_dir = Path("results/dge_final")
    dge_dir = Path(dge_results_dir) if dge_results_dir else default_dge_dir
    if not dge_dir.exists():
        raise FileNotFoundError(f"DGE results directory not found: {dge_dir}")
    if cell_type == "Microglia":
        pattern = f"dge_results_microglia_{region}_*.csv"
    else:
        pattern = f"dge_results_{region}_*.csv"
    dge_files = sorted(dge_dir.glob(pattern))
    if not dge_files:
        raise FileNotFoundError(f"No DGE files for region={region} in {dge_dir}")
    gene_to_best_padj = {}
    for path in dge_files:
        df = pd.read_csv(path, index_col=0)
        if "padj" not in df.columns:
            continue
        for gene in df.index[df["padj"].notna()]:
            p = float(df.loc[gene, "padj"])
            if gene not in gene_to_best_padj or p < gene_to_best_padj[gene]:
                gene_to_best_padj[gene] = p
    if not gene_to_best_padj:
        raise ValueError("No genes with valid padj in DGE results.")
    sorted_genes = sorted(gene_to_best_padj.keys(), key=lambda g: gene_to_best_padj[g])
    return sorted_genes[:top_k]


def coerce_numeric_series(s):
    """Coerce a Series to numeric: strip, map '90+' -> 90, invalid -> NaN."""
    return pd.to_numeric(
        s.astype(str).str.strip().replace("90+", "90"),
        errors="coerce",
    )


# Back-compat alias (ROSMAP_EDA and older call sites)
_coerce_numeric_series = coerce_numeric_series


def map_rosmap_condition(adata, condition_col):
    """Map ROSMAP condition column to Atlas-compatible condition column."""
    if condition_col not in adata.obs.columns:
        raise ValueError(f"Condition column '{condition_col}' not in adata.obs. Available: {list(adata.obs.columns)}")
    # coerce condition_col to numeric
    adata.obs[condition_col] = coerce_numeric_series(adata.obs[condition_col])
    # braaksc: 0: 1, 1,2: 2, 3,4: 3, 5,6: 4; drop cells with nan
    adata.obs["dge_condition"] = adata.obs[condition_col].map({0: 1, 1: 2, 2: 2, 3: 3, 4: 3, 5: 4, 6: 4})
    adata = adata[adata.obs["dge_condition"].notna()].copy()
    return adata
