"""Scratch: patient counts, extract best F1, or top DGE union genes from experiments CSV."""

import argparse
from collections import Counter
import os
import pickle
import sys
from pathlib import Path
import pandas as pd
try:
    import anndata as ad
except ImportError:
    ad = None
from config import REGION_TO_TAB, DATA_DIR, GENE_ANNOTATIONS_FILE
from utils import read_excel_columns, extract_best_per_attn_loss_dim

# Microglia combined metadata (Donor ID = patient identifier)
MICROGLIA_METADATA_CSV = os.path.join("data", "AtlasData", "2025-12-27_Microglia_Metadata.csv")
DEFAULT_DGE_RESULTS_DIR = "E:/sf2026/dge_results/dge_final"
DEFAULT_TFS_CSV = os.path.join("data", "GRN", "Priors", "TFs_utoronto.csv")
# SEA_AD metadata (MTG RNA-seq)
SEA_AD_MTG_METADATA = os.path.join("data", "SEA_AD_Data", "SEAAD_MTG_RNAseq_metadata.csv")


def _astrocytes_patients_per_region(regions):
    """Return dict region -> set of patient IDs (SampleName as str), and total set."""
    result = {}
    all_astro = set()
    for region in regions:
        anndata_path = os.path.join("data", "model_data", f"{region}_AnnData_perCell.h5ad")
        if os.path.exists(anndata_path):
            try:
                adata = ad.read_h5ad(anndata_path)
                patients = set(adata.obs["SampleName"].astype(str).dropna().unique())
                result[region] = patients
                all_astro |= patients
            except Exception as e:
                print(f"  {region}: Error loading AnnData - {e}")
                result[region] = set()
        else:
            metadata_path = os.path.join(DATA_DIR, "2025-11-16_Astrocytes_metadata.xlsx")
            tab_index = REGION_TO_TAB.get(region)
            if tab_index is not None and os.path.exists(metadata_path):
                try:
                    metadata = read_excel_columns(
                        metadata_path,
                        columns=["cell_annotation", "SampleName"],
                        sheet_name=tab_index,
                    )
                    patients = set(metadata["SampleName"].astype(str).dropna().unique())
                    result[region] = patients
                    all_astro |= patients
                except Exception as e:
                    print(f"  {region}: Error loading metadata - {e}")
                    result[region] = set()
            else:
                print(f"  {region}: No data file found")
                result[region] = set()
    return result, all_astro


def _microglia_patients_per_region(regions):
    """Return dict region -> set of Donor IDs (str), and total set. Uses combined Microglia CSV."""
    if not os.path.exists(MICROGLIA_METADATA_CSV):
        return {r: set() for r in regions}, set()
    try:
        df = pd.read_csv(MICROGLIA_METADATA_CSV)
        if "Donor ID" not in df.columns:
            return {r: set() for r in regions}, set()
        df["Donor ID"] = df["Donor ID"].astype(str)
        all_micro = set(df["Donor ID"].dropna().unique())
        result = {}
        if "region" in df.columns:
            for region in regions:
                result[region] = set(df.loc[df["region"] == region, "Donor ID"].astype(str).dropna().unique())
        else:
            for region in regions:
                result[region] = set()
        return result, all_micro
    except Exception as e:
        print(f"  Microglia: Error loading metadata - {e}")
        return {r: set() for r in regions}, set()


def print_patient_counts():
    """Print unique patient counts for Astrocytes and Microglia (per region and combined)."""
    astrocyte_regions = ["EC", "ITG", "PFC", "V2", "V1"]
    microglia_regions = ["CrossRegion", "EC", "ITG", "PFC", "V2", "V1"]

    print("=" * 60)
    print("Unique patients: Astrocytes and Microglia")
    print("=" * 60)

    # Astrocytes
    astro_per_region, astro_all = _astrocytes_patients_per_region(astrocyte_regions)
    print("\nAstrocytes (per region):")
    for region in astrocyte_regions:
        n = len(astro_per_region.get(region, set()))
        print(f"  {region}: {n} patients")
    print(f"  Total unique (Astrocytes): {len(astro_all)}")

    # Microglia
    micro_per_region, micro_all = _microglia_patients_per_region(microglia_regions)
    print("\nMicroglia (per region):")
    for region in microglia_regions:
        n = len(micro_per_region.get(region, set()))
        print(f"  {region}: {n} patients")
    print(f"  Total unique (Microglia): {len(micro_all)}")

    # Both: union and overlap (match on string ID)
    union = astro_all | micro_all
    overlap = astro_all & micro_all
    print("\nBoth cell types (same patient ID scheme):")
    print(f"  Union (in either Astrocytes or Microglia): {len(union)} unique patients")
    print(f"  Overlap (in both): {len(overlap)} unique patients")
    print("=" * 60)


def extract_best_f1(csv_path):
    """Load experiments CSV and return the row with highest avg_f1. Prints run, avg_f1, and key params."""
    df = pd.read_csv(csv_path)
    if "avg_f1" not in df.columns:
        raise ValueError(f"CSV must have column 'avg_f1'. Found: {list(df.columns)}")
    df = df.dropna(subset=["avg_f1"])
    if df.empty:
        raise ValueError("No rows with valid avg_f1.")
    idx = df["avg_f1"].idxmax()
    best = df.loc[idx]
    run = best.get("run", "")
    f1 = float(best["avg_f1"])
    print(f"Best F1: {f1:.6f}")
    print(f"Run: {run}")
    key_cols = ["experiment", "gene_encoder", "attention", "pca_components", "dropout", "weight_decay", "lr"]
    for c in key_cols:
        if c in best.index:
            print(f"  {c}: {best[c]}")
    if "avg_mse" in best.index and pd.notna(best.get("avg_mse")):
        print(f"  avg_mse: {best['avg_mse']:.6f}")
    if "avg_mae" in best.index and pd.notna(best.get("avg_mae")):
        print(f"  avg_mae: {best['avg_mae']:.6f}")
    return best


def print_top_dge_union_genes(region="EC", cell_type="Astrocytes", top_k=10, dge_results_dir=None):
    """Print top-k genes from the union of region DGE results (genes with valid padj), sorted by padj ascending."""
    dge_dir = Path(dge_results_dir or DEFAULT_DGE_RESULTS_DIR)
    if not dge_dir.exists():
        print(f"DGE results directory not found: {dge_dir}")
        return
    if cell_type == "Microglia":
        pattern = f"dge_results_microglia_{region}_*.csv"
    else:
        pattern = f"dge_results_{region}_*.csv"
    dge_files = sorted(dge_dir.glob(pattern))
    if not dge_files:
        print(f"No DGE files found for region={region}, cell_type={cell_type} in {dge_dir}")
        return
    gene_to_best_row = {}
    for path in dge_files:
        df = pd.read_csv(path, index_col=0)
        if "padj" not in df.columns:
            continue
        valid = df["padj"].notna()
        for gene in df.index[valid]:
            row = df.loc[gene]
            p = float(row["padj"])
            if gene not in gene_to_best_row or p < gene_to_best_row[gene]["padj"]:
                gene_to_best_row[gene] = {"padj": p, "row": row}
    if not gene_to_best_row:
        print("No genes with valid padj found.")
        return
    sorted_genes = sorted(gene_to_best_row.items(), key=lambda x: x[1]["padj"])
    top = sorted_genes[:top_k]
    print(f"Top {len(top)} genes (union of {region} {cell_type} DGE results with padj), sorted by padj:")
    print(f"  DGE dir: {dge_dir}, files: {len(dge_files)}")
    all_columns = top[0][1]["row"].index.tolist() if top else []
    print(f"  Columns: {all_columns}")
    for i, (gene, data) in enumerate(top, 1):
        row = data["row"]
        print(f"\n  --- {i}. {gene} ---")
        for col in all_columns:
            val = row[col]
            if pd.isna(val):
                print(f"    {col}: NaN")
            elif isinstance(val, (int, float)) and not isinstance(val, bool):
                if abs(val) < 1e-4 or abs(val) >= 1e4:
                    print(f"    {col}: {val:.2e}")
                else:
                    print(f"    {col}: {val}")
            else:
                print(f"    {col}: {val}")
    return [(g, data["row"]) for g, data in top]


def print_top_ranked_genes(csv_path, top_n=50, gene_annotations_path=None):
    """Print the top N gene names from a ranked_genes.csv (only gene names), restricted to protein-coding genes using gene_annotations.pkl."""
    df = pd.read_csv(csv_path)
    if "gene" not in df.columns:
        raise ValueError(f"CSV must have column 'gene'. Found: {list(df.columns)}")

    # Load protein-coding set from gene_annotations.pkl
    ann_path = gene_annotations_path or GENE_ANNOTATIONS_FILE
    if not os.path.isfile(ann_path):
        raise FileNotFoundError(f"Gene annotations not found: {ann_path}")
    with open(ann_path, "rb") as f:
        gene_annotations = pickle.load(f)
    protein_coding = set(
        gene_annotations[gene_annotations["Gene type"] == "protein_coding"]["Gene name"].values
    )

    # Take genes in rank order until we have top_n protein-coding
    top_genes = []
    for g in df["gene"]:
        if g in protein_coding:
            top_genes.append(g)
            if len(top_genes) >= top_n:
                break
    for g in top_genes:
        print(g)
    if len(top_genes) < top_n:
        print(f"  (only {len(top_genes)} protein-coding in top ranks; requested {top_n})", file=sys.stderr)
    return top_genes


def genes_from_dge_csv(csv_path, top_n=None, padj_threshold=0.05):
    """Load a DGE results CSV (from dge2.py) and return gene names.
    If top_n is set, return top N genes by padj (ascending); otherwise all genes with padj < padj_threshold, or all if none.
    CSV is expected to have gene names as index (first column) and a 'padj' column."""
    df = pd.read_csv(csv_path, index_col=0)
    if "padj" not in df.columns:
        genes = df.index.astype(str).str.strip().tolist()
        if top_n:
            genes = genes[: top_n]
        return genes
    df_valid = df[df["padj"].notna()].copy()
    if top_n is not None:
        df_valid = df_valid.sort_values("padj", ascending=True).head(top_n)
    else:
        df_valid = df_valid[df_valid["padj"] < padj_threshold]
        if len(df_valid) == 0:
            df_valid = df.sort_values("padj", ascending=True, na_position="last").head(500)
    return df_valid.index.astype(str).str.strip().tolist()


def find_TFs(top_genes, tfs_csv_path=None):
    """Map top_genes to the TF list in TFs_utoronto.csv and print distribution: X% of genes were linked to this TF.
    A gene is 'linked to' a TF if the gene is in the TF list (i.e. the gene is that TF).
    Matching is case-insensitive so that different symbol casings still match."""
    path = tfs_csv_path or DEFAULT_TFS_CSV
    if not os.path.isfile(path):
        alt = os.path.join("data", "TFs_utoronto.csv")
        path = alt if os.path.isfile(alt) else path
    if not os.path.isfile(path):
        raise FileNotFoundError(f"TFs list not found: {tfs_csv_path or DEFAULT_TFS_CSV} (also tried data/TFs_utoronto.csv)")
    if path.endswith(".txt"):
        df = pd.read_csv(path, header=None)
        tf_set = set(df.iloc[:, 0].dropna().astype(str).str.strip())
    else:
        df = pd.read_csv(path)
        if "HGNC symbol" in df.columns:
            tf_set = set(df["HGNC symbol"].dropna().astype(str).str.strip())
        else:
            tf_set = set(df.iloc[:, 0].dropna().astype(str).str.strip())
    # Case-insensitive lookup: map upper-case symbol -> original symbol for display
    tf_upper_to_orig = {s.upper(): s for s in tf_set if s}
    n_total = len(top_genes)
    if n_total == 0:
        print("No genes to map.")
        return {}
    # Count how many times each TF appears in top_genes (case-insensitive)
    tf_counts = Counter()
    for g in top_genes:
        g_clean = str(g).strip() if g else ""
        if not g_clean:
            continue
        key = g_clean.upper()
        if key in tf_upper_to_orig:
            tf_counts[tf_upper_to_orig[key]] += 1
    n_tfs = sum(tf_counts.values())
    print(f"TF distribution (genes in top list that are TFs): {n_tfs}/{n_total} ({100 * n_tfs / n_total:.1f}% are TFs)")
    print("-" * 50)
    if n_tfs == 0:
        print("  (No TFs found in top list. Check that gene symbols match TF list format, e.g. HGNC.)")
        if top_genes:
            print(f"  First 5 top genes: {list(top_genes[:5])}")
        return {}
    for tf, count in tf_counts.most_common():
        pct = 100 * count / n_total
        print(f"  {pct:.1f}% of genes were linked to {tf}")
    return dict(tf_counts)


def print_sea_ad_sample_names(path, column="SampleName", n=10):
    """Load SEA_AD metadata from .h5ad (adata.obs), .csv, or .xlsx and print first n values of column (default SampleName)."""
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")
    if path.endswith(".h5ad"):
        import anndata as ad
        adata = ad.read_h5ad(path)
        if column not in adata.obs.columns:
            raise ValueError(f"Column '{column}' not in adata.obs. Available: {list(adata.obs.columns)}")
        vals = adata.obs[column].head(n)
    else:
        if path.endswith(".xlsx") or path.endswith(".xls"):
            df = pd.read_excel(path, nrows=n)
        else:
            df = pd.read_csv(path, nrows=n)
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not in file. Available: {list(df.columns)}")
        vals = df[column].head(n)
    print(f"First {len(vals)} values of '{column}' from {path}:")
    for i, v in enumerate(vals, 1):
        print(f"  {i}. {v}")
    return vals.tolist()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Patient counts, extract best F1, or top DGE union genes")
    parser.add_argument("--best_f1", type=str, default=None,
                        help="Path to overfitting_experiments CSV; print row with highest avg_f1")
    parser.add_argument("--top_dge_genes", type=str, nargs="*", default=None,
                        help="Print top N genes from union of region DGE (e.g. EC 10). Default: EC 10")
    parser.add_argument("--dge_results_dir", type=str, default=None,
                        help="DGE results directory (default: E:/sf2026/dge_results/dge_final)")
    parser.add_argument("--top_ranked_genes", type=str, nargs="*", default=None,
                        help="Path to ranked_genes.csv and optional N (default 50). Prints top N gene names only.")
    parser.add_argument("--tfs", action="store_true", help="With --top_ranked_genes or --top_dge_genes: also print TF distribution (find_TFs).")
    parser.add_argument("--tfs_list", type=str, default=None,
                        help="Path to TF list CSV or txt (e.g. data/GRN/Priors/TFs_aertsLab.txt). Default: TFs_utoronto.csv.")
    parser.add_argument("--sea_ad_sample_names", type=str, nargs="*", default=None,
                        help="Path to SEA_AD metadata CSV (e.g. data/SEA_AD_Data/SEAAD_MTG_RNAseq_metadata.csv) or .h5ad; optional column (default SampleName). Print first 10.")
    parser.add_argument("--dge_csv", type=str, nargs="*", default=None,
                        help="Path to one DGE results CSV (e.g. from dge2.py); optional --top N (default: sig genes padj<0.05). Use with --tfs to print TF breakdown.")
    args = parser.parse_args()
    if args.best_f1:
        extract_best_f1(args.best_f1)
    elif args.sea_ad_sample_names is not None:
        # Default path: SEA_AD_Data/SEAAD_MTG_RNAseq_metadata.xlsx
        p = args.sea_ad_sample_names[0] if len(args.sea_ad_sample_names) >= 1 else SEA_AD_MTG_METADATA
        col = args.sea_ad_sample_names[1] if len(args.sea_ad_sample_names) >= 2 else "SampleName"
        print_sea_ad_sample_names(p, column=col, n=10)
    elif args.top_ranked_genes is not None:
        path = args.top_ranked_genes[0]
        top_n = int(args.top_ranked_genes[1]) if len(args.top_ranked_genes) >= 2 else 50
        top_genes = print_top_ranked_genes(path, top_n=top_n)
        if getattr(args, "tfs", False):
            print()
            find_TFs(top_genes, tfs_csv_path=getattr(args, "tfs_list", None))
    elif args.top_dge_genes is not None:
        region = args.top_dge_genes[0] if len(args.top_dge_genes) >= 1 else "EC"
        top_k = int(args.top_dge_genes[1]) if len(args.top_dge_genes) >= 2 else 10
        result = print_top_dge_union_genes(region=region, cell_type="Astrocytes", top_k=top_k, dge_results_dir=args.dge_results_dir)
        if getattr(args, "tfs", False) and result:
            genes = [g for g, r in result]
            print()
            find_TFs(genes, tfs_csv_path=getattr(args, "tfs_list", None))
    elif args.dge_csv is not None:
        csv_path = args.dge_csv[0]
        top_n = int(args.dge_csv[1]) if len(args.dge_csv) >= 2 else None
        genes = genes_from_dge_csv(csv_path, top_n=top_n)
        print(f"Genes from DGE CSV: {len(genes)} (top_n={top_n})")
        if getattr(args, "tfs", False):
            find_TFs(genes, tfs_csv_path=getattr(args, "tfs_list", None))
    else:
        print_patient_counts()
