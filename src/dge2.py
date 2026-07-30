"""
Differential Gene Expression (DGE) Analysis Class

This module provides a class-based interface for performing DGE analysis
using PyDESeq2 (DESeq2 implementation in Python).
"""

import os
import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, List, Any
from anndata import AnnData
from utils import filter_anndata_object, aggregate_cells_into_pseudobulk, map_rosmap_condition

try:
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.default_inference import DefaultInference
    from pydeseq2.ds import DeseqStats
    PYDESEQ2_AVAILABLE = True
except ImportError:
    PYDESEQ2_AVAILABLE = False
    print("Warning: pydeseq2 not installed. Install with: pip install pydeseq2")
from config import ROSMAP_DATA_DIR, ROSMAP_CELL_TYPES, ROSMAP_CONDITION_COL, ROSMAP_PATIENT_COL

class DGEAnalyzer:
    
    def __init__(
        self,
        data_file: AnnData,
        condition: str,
        groups: List[Tuple[str, str]],
        method: str = "deseq2",
        threshold: Optional[Dict[str, float]] = None,
        topK: int = 10,
        output_file: str = "dge_results.csv",
        min_counts: int = 10,
        n_cpus: int = 4,
        control_genes: Optional[List[str]] = None,
        covariates: Optional[List[str]] = None
    ):
        """
        Initialize DGEAnalyzer.
        
        Parameters:
        -----------
        data_file : AnnData
            AnnData object with gene expression data (cells x genes format)
            Must have raw counts in .X and metadata in .obs
        condition : str
            Column name in adata.obs specifying the condition/group variable
        groups : list of tuples
            List of (group1, group2) tuples for multiple comparisons
            Example: [("treated", "control"), ("disease", "healthy")]
        method : str, optional
            DGE method to use (default: "deseq2")
        threshold : dict, optional
            Dictionary with 'padj' and 'log2fc' thresholds for significance
            Default: {'padj': 0.05, 'log2fc': 1.0}
        topK : int, optional
            Number of top genes to return (default: 10)
        output_file : str, optional
            Output CSV file name (default: "dge_results.csv")
        min_counts : int, optional
            Minimum total counts across samples to keep a gene (default: 10)
        n_cpus : int, optional
            Number of CPUs for parallel processing (default: 4)
        control_genes : list, optional
            List of control gene names for normalization (default: None)
        covariates : list of str, optional
            Column names in adata.obs to include as covariates in the DESeq2 design
            formula. The design becomes ~cov1 + cov2 + ... + condition, so the
            condition effect is estimated adjusting for these covariates (e.g. batch).
            Default: None (design is ~condition).
        """
        # Validate pydeseq2 availability
        if method == "deseq2" and not PYDESEQ2_AVAILABLE:
            raise ImportError(
                "pydeseq2 is required for DESeq2 method. "
                "Install with: pip install pydeseq2"
            )
        
        # Validate condition column exists
        if condition not in data_file.obs.columns:
            raise ValueError(
                f"Condition column '{condition}' not found in adata.obs. "
                f"Available columns: {list(data_file.obs.columns)}"
            )
        
        # Validate and process groups
        if not groups:
            raise ValueError("'groups' list must contain at least one (group1, group2) tuple")
        
        unique_conditions = data_file.obs[condition].unique()
        
        # Validate all groups exist
        for g1, g2 in groups:
            if g1 not in unique_conditions:
                raise ValueError(
                    f"Group '{g1}' not found in condition '{condition}'. "
                    f"Available groups: {list(unique_conditions)}"
                )
            if g2 not in unique_conditions:
                raise ValueError(
                    f"Group '{g2}' not found in condition '{condition}'. "
                    f"Available groups: {list(unique_conditions)}"
                )
        
        # Validate covariate columns exist and differ from the condition
        covariates = list(covariates) if covariates else []
        for cov in covariates:
            if cov == condition:
                raise ValueError(f"Covariate '{cov}' is the same as the condition column.")
            if cov not in data_file.obs.columns:
                raise ValueError(
                    f"Covariate column '{cov}' not found in adata.obs. "
                    f"Available columns: {list(data_file.obs.columns)}"
                )

        # Store groups
        self.groups = groups        # Store parameters
        self.data_file = data_file
        self.condition = condition
        self.method = method
        self.topK = topK
        self.output_file = output_file
        self.min_counts = min_counts
        self.n_cpus = n_cpus
        self.control_genes = control_genes
        self.covariates = covariates
        
        # Set default thresholds if not provided
        if threshold is None:
            threshold = {'padj': 0.05, 'log2fc': 1.0}
        self.threshold = threshold
        
        # Initialize results attributes
        self.dds = None  # DeseqDataSet object
        self.stat_res_dict = {}  # Dictionary of DeseqStats objects for each group pair
        self.results_dict = {}  # Dictionary of results DataFrames for each group pair
        self.significant_genes_dict = {}  # Dictionary of significant genes for each group pair
        
        # Store extracted counts and metadata
        self.counts_df = None
        self.metadata_df = None
    
    def _prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare count matrix and metadata from AnnData object.
        
        Returns:
        --------
        counts_df : pd.DataFrame
            Count matrix with samples (cells) as rows, genes as columns
            Index must match metadata_df index (cell names)
        metadata_df : pd.DataFrame
            Sample metadata with condition information
        """

        if self.data_file is None:
            raise ValueError("Failed to filter AnnData object. Check your filter thresholds.")

        # Extract counts matrix (cells x genes from AnnData)
        # Prefer raw count layers (SEA_AD: UMIs; ROSMAP: counts), else .X
        layers = getattr(self.data_file, "layers", {})
        if "UMIs" in layers:
            X = self.data_file.layers["UMIs"]
        elif "counts" in layers:
            X = self.data_file.layers["counts"]
        else:
            X = self.data_file.X
        if hasattr(X, 'toarray'):
            counts_matrix = X.toarray()
        else:
            counts_matrix = np.asarray(X)
        counts_matrix = np.round(counts_matrix).astype(np.int64)

        # Create counts DataFrame with cell names as index (rows), gene names as columns
        self.counts_df = pd.DataFrame(
            counts_matrix,
            index=self.data_file.obs_names,  # Cells as rows - matches metadata_df index
            columns=self.data_file.var_names  # Genes as columns
        )
        
        # Extract metadata
        self.metadata_df = self.data_file.obs.copy()
        
        # Ensure indices match (important for PyDESeq2)
        # Filter metadata to only include cells present in counts
        # self.metadata_df = self.metadata_df.loc[self.counts_df.index] (CHECK)
        
        # Filter lowly expressed genes
        if 0:
            if self.min_counts > 0:
                gene_sums = self.counts_df.sum(axis=0)
                self.counts_df = self.counts_df.loc[:, gene_sums >= self.min_counts]
            
            # Verify indices match
            if not self.counts_df.index.equals(self.metadata_df.index):
                raise ValueError("Index mismatch between counts and metadata")
        
        return self.counts_df, self.metadata_df
    
    def run_dge(self) -> pd.DataFrame:
        """
        Run differential gene expression analysis.
        
        Returns:
        --------
        results : pd.DataFrame
            DataFrame with DGE results containing columns:
            - baseMean: Average normalized count
            - log2FoldChange: Log2 fold change (group1 vs group2)
            - lfcSE: Standard error of log2FoldChange
            - stat: Wald statistic
            - pvalue: Raw p-value
            - padj: Adjusted p-value (FDR)
        """
        if self.method != "deseq2":
            raise NotImplementedError(f"Method '{self.method}' not yet implemented. Only 'deseq2' is supported.")
        
        # Prepare data
        counts_df, metadata_df = self._prepare_data()
        
        # Create design formula. Covariates come first so the condition is the last
        # term; DeseqStats contrasts on `condition` then adjust for the covariates.
        if self.covariates:
            design = "~" + " + ".join(self.covariates + [self.condition])
        else:
            design = f"~{self.condition}"
        print(f"DESeq2 design formula: {design}")

        # Initialize inference
        inference = DefaultInference()
        
        # Create DeseqDataSet
        self.dds = DeseqDataSet(
            counts=counts_df,
            metadata=metadata_df,
            design=design,
            inference=inference,
            n_cpus=self.n_cpus,
            control_genes=self.control_genes
        )
        
        # Run DESeq2 pipeline
        self.dds.deseq2()
        
        return self.dds
    
    def get_results(
        self,
        groups: Optional[List[Tuple[str, str]]] = None,
        save: bool = True,
        filename_prefix: Optional[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Extract results, find significant genes, get top genes, and save results
        for all group pairs.
        
        This method combines:
        - Extracting results using DeseqStats (lines 251-258 equivalent)
        - Finding significant genes
        - Getting top genes
        - Saving results to files
        
        Parameters:
        -----------
        groups : list of tuples, optional
            List of (group1, group2) tuples for comparisons.
            If None, uses self.groups from initialization (default: None)
        save : bool, optional
            Whether to save results to files (default: True)
        filename_prefix : str, optional
            Prefix for output filenames. If None, uses self.output_file as base (default: None)
            
        Returns:
        --------
        results_dict : dict
            Dictionary with keys as f"{group1}_vs_{group2}" containing:
            - 'stat_res': DeseqStats object
            - 'results': All results DataFrame
            - 'significant_genes': Significant genes DataFrame
            - 'top_genes': Top K genes DataFrame
        """
        if self.dds is None:
            raise ValueError("DeseqDataSet not available. Run run_dge() first.")
        
        # Use provided groups or self.groups
        if groups is None:
            groups = self.groups
        
        if not groups:
            raise ValueError("No groups provided.")
        
        # Initialize inference
        inference = DefaultInference()
        
        # Dictionary to store all results
        all_results = {}
        
        # Iterate over all group pairs
        for group1, group2 in groups:
            stat_res = DeseqStats(
                self.dds,
                contrast=[self.condition, group1, group2],
                inference=inference,
                n_cpus=self.n_cpus
            )
            
            stat_res.summary()
            results = stat_res.results_df
            
            # Store stat_res and results
            pair_key = f"{group1}_vs_{group2}"
            self.stat_res_dict[pair_key] = stat_res
            self.results_dict[pair_key] = results
            
            # Diagnostic: Report NaN padj statistics
            nan_padj_count = results['padj'].isna().sum()
            valid_padj_count = results['padj'].notna().sum()
            total_genes = len(results)
            
            if nan_padj_count > 0:
                print(f"\n[Diagnostic] padj statistics for {pair_key}:")
                print(f"  Total genes: {total_genes}")
                print(f"  Genes with valid padj: {valid_padj_count} ({valid_padj_count/total_genes*100:.1f}%)")
                print(f"  Genes with NaN padj: {nan_padj_count} ({nan_padj_count/total_genes*100:.1f}%)")
                
                # Check if NaN padj is expected (low baseMean)
                nan_padj_genes = results[results['padj'].isna()]
                if len(nan_padj_genes) > 0:
                    max_baseMean_nan = nan_padj_genes['baseMean'].max()
                    mean_baseMean_nan = nan_padj_genes['baseMean'].mean()
                    print(f"  NaN padj genes - max baseMean: {max_baseMean_nan:.3f}, mean: {mean_baseMean_nan:.3f}")
                    
                    if max_baseMean_nan < 1.0:
                        print(f"Expected: All NaN padj genes have low expression (baseMean < 1)")
                    else:
                        print(f"Warning: Some NaN padj genes have high expression (max baseMean = {max_baseMean_nan:.3f})")
                        print(f"    This may indicate an issue - check DESeq2 convergence warnings")
            
            # Find significant genes (only use genes with valid padj)
            padj_threshold = self.threshold.get('padj', 0.05)
            log2fc_threshold = self.threshold.get('log2fc', 1.0)
            
            mask = (
                (results['padj'] < padj_threshold) &
                (results['padj'].notna()) &
                (abs(results['log2FoldChange']) > log2fc_threshold)
            )
            
            significant_genes = results[mask].copy()
            self.significant_genes_dict[pair_key] = significant_genes
            
            # Get genes with padj < 0.05
            padj_mask = (results['padj'] < 0.05) & (results['padj'].notna())
            genes_padj_005 = results[padj_mask].copy()
            if len(genes_padj_005) > 0:
                genes_padj_005 = genes_padj_005.sort_values('padj', ascending=True)
            
            # Keep top_genes
            top_genes = genes_padj_005.head(self.topK) if len(genes_padj_005) > 0 else pd.DataFrame()
            
            # Save results if requested
            if save:
                if filename_prefix is None:
                    base = self.output_file.rsplit('.', 1)[0] if '.' in self.output_file else self.output_file
                    if pair_key in base:
                        filename = f"{base}.csv"
                    else:
                        filename = f"{base}_{pair_key}.csv"
                else:
                    filename = f"{filename_prefix}_{pair_key}.csv"
                
                results.to_csv(filename)
            
            # Store all results for this pair
            all_results[pair_key] = {
                'stat_res': stat_res,
                'results': results,
                'significant_genes': significant_genes,
                'genes_padj_005': genes_padj_005,
                'top_genes': top_genes
            }
        
        return all_results
    
    def run_full_analysis(self) -> Dict[str, Any]:
        """
        Run complete DGE analysis pipeline.
        
        This method runs all steps:
        1. Run DGE analysis
        2. Get results for all group pairs (extracts, filters, gets top genes, saves)
        
        Returns:
        --------
        results_dict : dict
            Dictionary with keys as f"{group1}_vs_{group2}" containing:
            - 'stat_res': DeseqStats object
            - 'results': All DGE results
            - 'significant_genes': Significant DEGs
            - 'top_genes': Top K genes
        """
        # Step 1: Run DGE
        self.run_dge()
        
        # Step 2: Get results for all group pairs
        all_results = self.get_results()
        
        return all_results
    
    def __repr__(self):
        """String representation of the DGEAnalyzer object."""
        status = "Analysis run" if len(self.results_dict) > 0 else "Not run"
        if len(self.groups) > 1:
            comparisons = f"{len(self.groups)} comparisons: {self.groups}"
        else:
            comparisons = f"{self.group1} vs {self.group2}"
        return (
            f"DGEAnalyzer(\n"
            f"  method='{self.method}',\n"
            f"  condition='{self.condition}',\n"
            f"  comparisons={comparisons},\n"
            f"  threshold={self.threshold},\n"
            f"  topK={self.topK},\n"
            f"  output_file='{self.output_file}',\n"
            f"  status='{status}'\n"
            f")"
        )


def _ensure_rosmap_counts(adata: AnnData) -> AnnData:
    """Use raw counts for DGE; processed ROSMAP h5ad may store them in layers['counts']."""
    if "counts" in getattr(adata, "layers", {}):
        adata = adata.copy()
        adata.X = adata.layers["counts"]
    return adata


def _load_rosmap_anndata(
    cell_type: str,
    condition_col: Optional[str] = None,
    patient_col: Optional[str] = None,
) -> Tuple[AnnData, str, str]:
    import anndata

    h5ad_path = os.path.join(ROSMAP_DATA_DIR, f"{cell_type}.h5ad")
    if not os.path.isfile(h5ad_path):
        raise FileNotFoundError(
            f"ROSMAP AnnData not found: {h5ad_path}. "
            f"Run processing_ROSMAP_data.py (or equivalent) to build it."
        )
    condition_col = condition_col or ROSMAP_CONDITION_COL
    patient_col = patient_col or ROSMAP_PATIENT_COL

    adata = anndata.read_h5ad(h5ad_path)
    adata = _ensure_rosmap_counts(adata)
    if patient_col not in adata.obs.columns:
        raise ValueError(
            f"Patient column '{patient_col}' not in adata.obs. "
            f"Available: {list(adata.obs.columns)}"
        )
    print(f"Loaded ROSMAP AnnData from {h5ad_path}: {adata.shape} (cells × genes)")
    adata = map_rosmap_condition(adata, condition_col)
    return adata, "dge_condition", patient_col


# Example usage
if __name__ == "__main__":
    """
    Example usage of DGEAnalyzer class.
    
    This example demonstrates how to use the class with an AnnData object
    loaded from MTX files (same as read_mtx_example.py).
    
    Usage:
        python dge.py --group1 GROUP1 --group2 GROUP2 --region REGION
        
    Example:
        python dge.py --group1 "Control" --group2 "Treatment" --region "EC"
    """
    import argparse
    
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='Run Differential Gene Expression (DGE) analysis using PyDESeq2',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        '-data',
        type=str,
        default='Atlas',
        choices=['Atlas', 'SEA_AD', 'ROSMAP'],
        help='Data source: Atlas (MTX+metadata) , SEA_AD (MTX+metadata) or ROSMAP (preprocessed h5ad under data/ROSMAP_Data/)'
    )

    parser.add_argument(
        '-cell_type',
        type=str,
        required=True,
        help='Cell type (Atlas/SEA_AD/ROSMAP: Astrocytes or Microglia; with -adata: label for output only)'
    )
    
    parser.add_argument(
        '-region',
        type=str,
        required=False,
        help='Region identifier to locate data files (e.g., EC, ITG, PFC, V1, V2), if SEA_AD: (MTG, PFC)'
    )
    parser.add_argument(
        '-seed',
        type=int,
        default=100,
        help='Random seed for cell shuffling before pseudobulk binning (default: 100)'
    )
    parser.add_argument(
        '-topK',
        type=int,
        default=10,
        help='Number of top genes to display (default: 50)'
    )
    parser.add_argument(
        '-bins',
        type=int,
        default=800,
        help='Target number of cells per bin for pseudobulk aggregation (default: 800). Ignored if -pseudobulk is set.'
    )
    parser.add_argument(
        '-adata',
        type=str,
        default=None,
        help='Path to AnnData .h5ad file or to MTG directory (loads all .h5ad under it, keeps only conditions 1–4). Requires -condition_col and -patient_col.'
    )
    parser.add_argument(
        '-condition_col',
        type=str,
        default=None,
        help='Obs column for DGE design. For SEA_AD use "Overall AD neuropathological Change" (mapped to 1=Not AD, 2=Low, 3=Intermediate, 4=High). Required when using -adata.'
    )
    parser.add_argument(
        '-patient_col',
        type=str,
        default=None,
        help='Obs column for patient/donor ID (for pseudobulking). For SEA_AD use "Donor ID". Required when using -adata.'
    )
    parser.add_argument(
        '-pseudobulk',
        type=str,
        default=None,
        help='Pseudobulk: "none" (use default -bins) or integer cells per block (e.g. 100, 200). If unset, use -bins.'
    )
    parser.add_argument(
        '--groups',
        type=str,
        nargs='*',
        default=None,
        help='Contrasts as group1,group2 (e.g. 1,2 1,3 1,4 2,3 2,4 3,4). Default: all pairwise 1 vs 2, 1 vs 3, 1 vs 4, 2 vs 3, 2 vs 4, 3 vs 4.'
    )
    parser.add_argument(
        '-covariates',
        type=str,
        nargs='*',
        default=None,
        help='Obs column(s) to include as covariates in the DESeq2 design, so the design becomes '
             '~cov1 + ... + condition (condition effect adjusted for the covariates). '
             'For Atlas Astrocytes batch use: -covariates Flowcell.'
    )

    parser.add_argument(
        '-binarize',
        action='store_true',
        help='Binarize the condition into low(1) vs high(2) and run the single low-vs-high contrast. '
             'ADNC: Not-AD/Low vs Intermediate/High. Thal phase: <thr vs >=thr (default thr=3). '
             'Output tagged _binADNC / _binThal.'
    )
    parser.add_argument(
        '-binarize_threshold',
        type=int,
        default=None,
        help='Threshold for -binarize on numeric staging (Thal phase). Default 3 (Thal>=3 = high).'
    )

    # Parse arguments
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    
    from utils import read_mtx_file, create_anndata_object, read_excel_columns, get_region_file_paths
    from config import REGION_TO_TAB, DATA_DIR
    
    base_dir = DATA_DIR
    region = args.region or "MTG"
    sea_ad_already_pseudobulked = False
    pseudobulk_label = str(args.bins)

    # --- Load AnnData: either from h5ad (SEA_AD_Data) or from MTX+metadata (Atlas) ---
    if args.adata:
        # SEA_AD_Data: load .h5ad (single file or all .h5ad under a directory)
        if not args.condition_col or not args.patient_col:
            raise ValueError("When using -adata you must provide -condition_col and -patient_col.")
        import anndata
        adata_path = args.adata
        if os.path.isdir(adata_path):
            # MTG directory: load each .h5ad, filter to conditions, pseudobulk per file, then concat (avoids 50GB+ concat of raw cells)
            from pathlib import Path
            h5ad_files = sorted(Path(adata_path).rglob("*.h5ad"))
            if not h5ad_files:
                raise FileNotFoundError(f"No .h5ad files found under {adata_path}")
            print(f"Found {len(h5ad_files)} AnnData file(s) under {adata_path}")
            # Resolve target cells per bin (same as later pseudobulk section)
            n_cells = args.bins
            if getattr(args, "pseudobulk", None) is not None and str(args.pseudobulk).strip().lower() != "none":
                try:
                    n_cells = int(args.pseudobulk)
                except ValueError:
                    pass
            pseudobulk_label = str(n_cells)
            AD_LEVEL_MAP = {"Not AD": "1", "Low": "2", "Intermediate": "3", "High": "4"}
            cond_raw = args.condition_col
            patient_col = args.patient_col
            adatas = []
            for f in h5ad_files:
                ad = anndata.read_h5ad(str(f))
                if cond_raw not in ad.obs.columns:
                    print(f"  Skip {f.name}: missing column '{cond_raw}'")
                    continue
                raw = ad.obs[cond_raw].astype(str).str.strip()
                ad.obs["_dge_condition"] = raw.map(AD_LEVEL_MAP)
                ad_filtered = ad[ad.obs["_dge_condition"].notna()].copy()
                if ad_filtered.n_obs == 0:
                    print(f"  Skip {f.name}: no cells in Not AD / Low / Intermediate / High")
                    print(f"    Actual condition values: {raw.value_counts().to_dict()}")
                    continue
                ad = ad_filtered
                # Pseudobulk this file before appending to avoid huge concat (saves ~50GB RAM)
                if args.data == "ROSMAP":
                    filter_patients_cell_threshold = 0
                else:
                    filter_patients_cell_threshold = 80

                ad = aggregate_cells_into_pseudobulk(
                    ad, target_cells_per_bin=n_cells,
                    filter_patients_cell_threshold=filter_patients_cell_threshold, seed=args.seed, patient_col=patient_col
                )
                adatas.append(ad)
                print(f"  {f.name}: {ad.n_obs} pseudobulk samples")
            if not adatas:
                raise ValueError("No cells in conditions 1–4 across any file.")
            adata = anndata.concat(adatas, join="inner", label="batch", keys=[str(i) for i in range(len(adatas))])
            adata.obs.drop(columns=["batch"], inplace=True, errors="ignore")
            condition_col = "_dge_condition"
            sea_ad_already_pseudobulked = True
            print(f"Combined: {adata.n_obs} pseudobulk samples × {adata.n_vars} genes. Condition levels 1–4. Patients: {patient_col}.")
            if getattr(args, "binarize", False):
                _bmap = {"1": "1", "2": "1", "3": "2", "4": "2"}
                adata.obs["_dge_condition"] = adata.obs["_dge_condition"].map(_bmap)
                print("Binarized ADNC: 1=low (Not AD/Low), 2=high (Intermediate/High).")
        else:
            # Single file
            adata = anndata.read_h5ad(adata_path)
            condition_col = args.condition_col
            patient_col = args.patient_col
            if condition_col not in adata.obs.columns:
                raise ValueError(f"Condition column '{condition_col}' not in adata.obs. Available: {list(adata.obs.columns)}")
            if patient_col not in adata.obs.columns:
                raise ValueError(f"Patient column '{patient_col}' not in adata.obs. Available: {list(adata.obs.columns)}")
            print(f"Loaded AnnData from {adata_path}: {adata.shape} (cells × genes)")
            # SEA-AD pathology columns (ADNC or the legacy "Overall AD neuropathological Change")
            # map to numeric groups 1–4; "Reference"/unmapped values are dropped.
            if condition_col in ("ADNC", "Overall AD neuropathological Change"):
                AD_LEVEL_MAP = {"Not AD": "1", "Low": "2", "Intermediate": "3", "High": "4"}
                raw = adata.obs[condition_col].astype(str).str.strip()
                adata.obs["_dge_condition"] = raw.map(AD_LEVEL_MAP)
                n_before = adata.n_obs
                adata = adata[adata.obs["_dge_condition"].notna()].copy()
                n_dropped = n_before - adata.n_obs
                if n_dropped > 0:
                    print(f"Mapped condition to levels 1–4; dropped {n_dropped} cells with unmapped values (e.g. Reference).")
                condition_col = "_dge_condition"
                print(f"Condition levels for DGE: 1=Not AD, 2=Low, 3=Intermediate, 4=High. Patients: {patient_col}.")
                if getattr(args, "binarize", False):
                    _bmap = {"1": "1", "2": "1", "3": "2", "4": "2"}
                    adata.obs["_dge_condition"] = adata.obs["_dge_condition"].map(_bmap)
                    print("Binarized ADNC: 1=low (Not AD/Low), 2=high (Intermediate/High).")
            elif condition_col == "Thal phase":
                # "Thal 0".."Thal 5" -> 0..5; "Reference"/unmapped -> NaN (dropped).
                raw = adata.obs[condition_col].astype(str).str.strip()
                thal_map = {f"Thal {i}": str(i) for i in range(6)}
                adata.obs["_dge_condition"] = raw.map(thal_map)
                n_before = adata.n_obs
                adata = adata[adata.obs["_dge_condition"].notna()].copy()
                print(f"Mapped Thal phase to 0-5; dropped {n_before - adata.n_obs} cells (Reference/unmapped).")
                condition_col = "_dge_condition"
                if getattr(args, "binarize", False):
                    thr = int(getattr(args, "binarize_threshold", None) or 3)
                    adata.obs["_dge_condition"] = adata.obs["_dge_condition"].map(
                        lambda x: "2" if int(x) >= thr else "1")
                    print(f"Binarized Thal: 1=low (Thal<{thr}), 2=high (Thal>={thr}).")
                else:
                    print("Thal levels 0-5 for DGE.")
    elif args.data == "ROSMAP":
        adata, condition_col, patient_col = _load_rosmap_anndata(
            args.cell_type,
            condition_col=args.condition_col,
            patient_col=args.patient_col,
        )
        if not args.region:
            region = "PFC"
            print(f"ROSMAP: no -region given; using '{region}' for output naming only.")
    else:
        # Atlas data: MTX + metadata
        if not args.cell_type or not args.region:
            raise ValueError("When using Atlas data you must provide -cell_type and -region.")
        if args.cell_type == "Astrocytes":
            base_prefix = f"2025-11-16_{args.cell_type}_{region}"
        elif args.cell_type == "Microglia":
            base_prefix = f"2025-12-27_{args.cell_type}_{region}"
        else:
            raise ValueError(f"Cell type '{args.cell_type}' not supported. Use Astrocytes, Microglia, or -adata for h5ad.")
        
        mtx_path, row_annotation_path, col_annotation_path = get_region_file_paths(
            region,
            data_dir=base_dir,
            base_prefix=f"{base_prefix}"
        )
        if args.cell_type == "Astrocytes":
            metadata_path = os.path.join(base_dir, f"2025-11-16_Astrocytes_metadata.xlsx")
        else:
            metadata_path = os.path.join(base_dir, f"{base_prefix}_metadata.csv")
        
        if args.cell_type == "Astrocytes":
            tab_index = REGION_TO_TAB.get(region)
            if tab_index is None:
                raise ValueError(f"Region '{region}' not in REGION_TO_TAB. Available: {list(REGION_TO_TAB.keys())}")
            astro_cols = ['cell_annotation', "RIN", "Path..Group.", "SampleName",
                          "Median.UMI.Counts.per.Cell", "percent.mito"]
            # Load any requested covariate columns (e.g. Flowcell for batch) from the xlsx too.
            for cov in (args.covariates or []):
                if cov not in astro_cols:
                    astro_cols.append(cov)
            metadata = read_excel_columns(metadata_path, columns=astro_cols,
                                        sheet_name=tab_index)
        else:
            metadata = pd.read_csv(metadata_path)
            rename_dict = {}
            if 'Pathology Stage' in metadata.columns:
                rename_dict['Pathology Stage'] = 'Pathology_Stage'
            if 'Donor ID' in metadata.columns:
                rename_dict['Donor ID'] = 'Donor_ID'
            if rename_dict:
                metadata = metadata.rename(columns=rename_dict)
            if 'Pathology_Stage' in metadata.columns:
                metadata['Pathology_Stage'] = metadata['Pathology_Stage'].astype(str)
        
        matrix, gene_names, cell_names = read_mtx_file(
            mtx_path=str(mtx_path),
            row_annotation_path=str(row_annotation_path),
            col_annotation_path=str(col_annotation_path),
            transpose=False
        )
        adata = create_anndata_object(
            matrix=matrix,
            gene_names=gene_names,
            cell_names=cell_names,
            transpose=True,
            obs=metadata,
        )
        condition_col = 'Path..Group.' if args.cell_type == "Astrocytes" else 'Pathology_Stage'
        patient_col = 'SampleName' if args.cell_type == "Astrocytes" else 'Donor_ID'
        print(f"Loaded AnnData from MTX: {adata.shape} (cells × genes)")

    print(f"Before filtering: {adata.shape} (cells × genes)")

    # Apply quality filters (only for Atlas Astrocytes; no gene-type filter). Skip for h5ad / ROSMAP / Microglia.
    if not args.adata and args.data != "ROSMAP" and args.cell_type == "Astrocytes":
        adata = filter_anndata_object(adata, min_genes=0, min_cells=0, min_counts=None, max_counts=None, mito_max=None, gene_type=None)
    
    print(f"After filtering: {adata.shape} (cells × genes)")

    if adata is None:
        raise ImportError("Failed to create AnnData object. Make sure anndata is installed.")
    if args.data == "ROSMAP":
        filter_patients_cell_threshold = 0
    else:
        filter_patients_cell_threshold = 80
    # --- Pseudobulk ---
    if not sea_ad_already_pseudobulked:
        if getattr(args, 'pseudobulk', None) is not None:
            if str(args.pseudobulk).strip().lower() == "none":
                adata = aggregate_cells_into_pseudobulk(
                    adata, target_cells_per_bin=args.bins,
                    filter_patients_cell_threshold=filter_patients_cell_threshold, seed=args.seed, patient_col=patient_col
                )
                pseudobulk_label = "none"
            else:
                try:
                    n_cells = int(args.pseudobulk)
                    adata = aggregate_cells_into_pseudobulk(
                        adata, target_cells_per_bin=n_cells,
                        filter_patients_cell_threshold=filter_patients_cell_threshold, seed=args.seed, patient_col=patient_col
                    )
                    pseudobulk_label = str(n_cells)
                except ValueError:
                    raise ValueError(f"-pseudobulk must be 'none' or an integer (e.g. 100, 200), got: {args.pseudobulk}")
        else:
            adata = aggregate_cells_into_pseudobulk(
                adata, target_cells_per_bin=args.bins,
                filter_patients_cell_threshold=filter_patients_cell_threshold, seed=args.seed, patient_col=patient_col
            )
            pseudobulk_label = str(args.bins)
    
    # Ensure condition column is strings
    if condition_col in adata.obs.columns:
        adata.obs[condition_col] = adata.obs[condition_col].astype(str)

    # Ensure covariate columns are present and treated as categorical factors (strings)
    covariates = list(args.covariates or [])
    for cov in covariates:
        if cov not in adata.obs.columns:
            raise ValueError(
                f"Covariate '{cov}' not found in adata.obs after loading. "
                f"Available columns: {list(adata.obs.columns)}"
            )
        adata.obs[cov] = adata.obs[cov].astype(str)
    if covariates:
        print(f"Using covariate(s) in design: {covariates}")

    # Create output directory if it doesn't exist
    output_dir = "results/dge_final"
    os.makedirs(output_dir, exist_ok=True)
    
    if getattr(args, 'groups', None) is not None and len(args.groups) > 0:
        group_tuples = []
        for g in args.groups:
            parts = g.split(',')
            if len(parts) != 2:
                raise ValueError(f"Each --groups value must be group1,group2 (e.g. 1,2), got: {g}")
            group_tuples.append((str(parts[0].strip()), str(parts[1].strip())))
    elif getattr(args, "binarize", False):
        group_tuples = [('1', '2')]   # binary: low (Not AD/Low) vs high (Intermediate/High)
    else:
        group_tuples = [('1', '2'), ('1', '3'), ('1', '4'), ('2', '3'), ('2', '4'), ('3', '4')]
    print(f"Condition comparisons: {group_tuples}")

    cov_tag = ("_cov" + "-".join(covariates)) if covariates else ""
    if getattr(args, "binarize", False):
        _cc = str(args.condition_col)
        bin_tag = "_binThal" if _cc == "Thal phase" else ("_binBraak" if _cc == "Braak stage" else "_binADNC")
    else:
        bin_tag = ""
    out_suffix = f"{args.data}_{args.cell_type}_{region}_pseudobulk{pseudobulk_label}_seed{args.seed}{bin_tag}{cov_tag}"
    output_file = os.path.join(output_dir, f"dge_results_{out_suffix}.csv")
    dge = DGEAnalyzer(
        data_file=adata,
        condition=condition_col,
        groups=group_tuples,
        method="deseq2",
        threshold={'padj': 0.05, 'log2fc': 0.5},
        topK=args.topK,
        output_file=output_file,
        n_cpus=16,
        covariates=covariates
    )
    
    # Run full analysis
    results = dge.run_full_analysis()