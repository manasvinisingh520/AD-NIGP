"""
Plot cell embeddings using UMAP visualization.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import umap
import argparse
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from collections import Counter
import pickle
import re
from utils import create_color_mapping, get_grn_file_paths

# Dataset -> regions / default region. Embeddings live in GREmLN/embeddings/, keyed by region.
# DATA_REGIONS values are the *parsed* region names (used by --combine_all's filter);
# concatenate parses 'ROSMAP_Microglia_cell_embeddings.tsv' -> region 'ROSMAP'.
# DATA_DEFAULT_REGION values are the single-region *file stems*.
DATA_REGIONS = {
    "Atlas": ["EC", "ITG", "PFC", "V1", "V2", "CrossRegion"],
    "SEAAD": ["MTG"],
    "ROSMAP": ["ROSMAP"],
}
DATA_DEFAULT_REGION = {"Atlas": "V2", "SEAAD": "MTG", "ROSMAP": "ROSMAP_Microglia"}

parser = argparse.ArgumentParser(description='Plot cell embeddings using UMAP visualization.')
parser.add_argument('--data', type=str, default='Atlas', choices=['Atlas', 'SEAAD', 'ROSMAP'],
                    help='Dataset to plot: Atlas (default), SEAAD (=MTG), or ROSMAP. Sets the default '
                         'region and, with --combine_all, restricts to that dataset\'s regions.')
parser.add_argument('--region', type=str, default=None,
                    help='Region to process. Default depends on --data (Atlas: V2, SEAAD: MTG).')
parser.add_argument('--n_clusters', type=int, default=4, help='Maximum number of clusters to test (tests from 2 to n_clusters, default: 4)')
parser.add_argument('--combine_all', action='store_true', help='Combine all embeddings from all regions (both Astrocytes and Microglia)')
parser.add_argument('--color_by', type=str, default='cluster', choices=['cluster', 'cell_type', 'region', 'condition', 'batch'],
                    help='Color embeddings by: cluster (default), cell_type, region, condition, or batch')
parser.add_argument('--batch_col', type=str, default=None,
                    help="Metadata column to use as batch (default: auto per dataset, e.g. "
                         "'batch_vendor_name' for SEA-AD/MTG, 'Flowcell' for Atlas Astrocytes).")
parser.add_argument('--cells_to_keep', type=str, default=None,
                    choices=['Astrocytes', 'Microglia'],
                    help='Cell type to filter by: Astrocytes or Microglia. If not specified, loads all cell types.')
parser.add_argument('--save_clusters', action='store_true',
                    help='Save each KMeans cluster to data/Astrocyte_cluster_<id>_cells.pkl (default: off).')
args = parser.parse_args()

# Resolve default region from --data when not given explicitly
if args.region is None:
    args.region = DATA_DEFAULT_REGION.get(args.data)
    if args.region is None:
        parser.error(f"No embeddings available for --data {args.data} yet; pass --region explicitly.")


def load_cell_embeddings(embeddings_path):
    """Load cell embeddings from TSV/CSV file.
    
    Returns:
    --------
    embeddings_array : np.ndarray
        Array of embeddings (cells × dimensions)
    cell_names : list
        List of cell names from first column
    """
    try:
        df = pd.read_csv(embeddings_path, sep=',', header=None)
    except:
        df = pd.read_csv(embeddings_path, sep='\t', header=None)
    
    embeddings_list = []
    cell_names = []
    for idx in range(1, len(df)):
        row = df.iloc[idx]
        cell_name = str(row.iloc[0]).strip()  # First column is cell name (patient_bin format)
        embedding_values = row.iloc[1:].values
        embedding_values = [float(x) for x in embedding_values if pd.notna(x)]
        if len(embedding_values) > 0:
            embeddings_list.append(embedding_values)
            cell_names.append(cell_name)
        
    return np.array(embeddings_list), cell_names


def load_conditions_from_mapping(mapping_path):
    """Load conditions from patient mapping CSV file."""
    if not os.path.exists(mapping_path):
        print(f"Warning: Mapping file not found: {mapping_path}")
        return None
    
    try:
        mapping_df = pd.read_csv(mapping_path)
        if 'condition' in mapping_df.columns:
            cell_conditions = mapping_df['condition'].values.tolist()
            cell_conditions = [str(c) if pd.notna(c) else None for c in cell_conditions]
            return cell_conditions
        else:
            print(f"Warning: 'condition' column not found in {mapping_path}")
            return None
    except Exception as e:
        print(f"Warning: Could not load conditions from {mapping_path}: {e}")
        return None


def _mapping_path(region, cell_type):
    """Path to the patient_mapping.csv (holds 'condition') for a region + cell type."""
    if region == "ROSMAP":
        ct = cell_type or "Microglia"
        return os.path.join("data", "GRN", "ROSMAP", ct, f"{ct}_patient_mapping.csv")
    if region == "MTG":  # SEA-AD
        return os.path.join("data", "GRN", "MTG", "MTG_patient_mapping.csv")
    sub = "Microglia" if cell_type == "Microglia" else "Astrocytes"
    stem = f"{region}_Microglia" if cell_type == "Microglia" else f"{region}_Astrocytes"
    for cand in (os.path.join("data", "GRN", sub, f"{stem}_patient_mapping.csv"),
                 os.path.join("data", "GRN", sub, f"{region}_patient_mapping.csv")):
        if os.path.exists(cand):
            return cand
    return None


# Default batch (technical) column per dataset. Atlas Astrocytes falls back to the
# metadata xlsx 'Flowcell' (batch is not kept in the astro pseudobulk obs).
BATCH_COL_DEFAULTS = {"MTG": "batch_vendor_name", "ROSMAP": "batch"}   # per-dataset batch column
ATLAS_ASTRO_META_XLSX = os.path.join("data", "AtlasData", "2025-11-16_Astrocytes_metadata.xlsx")
# Cache for the (slow) Atlas astro metadata xlsx: in-process + on-disk, keyed by column.
_ATLAS_BATCH_MEM = {}
_ATLAS_BATCH_CACHE_PKL = os.path.join("data", "AtlasData", ".batch_map_cache.pkl")


def _default_batch_col(region, cell_type):
    """Preferred batch column when --batch_col isn't given.

    SEA-AD/MTG -> 'batch_vendor_name'; Atlas Astrocytes -> 'Flowcell' (from the
    metadata xlsx); otherwise None (auto-detect a batch-like column from obs).
    """
    if region in BATCH_COL_DEFAULTS:
        return BATCH_COL_DEFAULTS[region]
    if cell_type in (None, "Astrocytes"):
        return "Flowcell"
    return None


def _parse_region_celltype(cell_label):
    """Parse a cell_label like 'PFC_Microglia' or 'EC' -> (region, cell_type)."""
    if isinstance(cell_label, str) and cell_label.endswith("_Microglia"):
        return cell_label[: -len("_Microglia")], "Microglia"
    if isinstance(cell_label, str) and cell_label.endswith("_Astrocytes"):
        return cell_label[: -len("_Astrocytes")], "Astrocytes"
    return cell_label, "Astrocytes"  # astro files carry no suffix


def _atlas_astro_batch_map(col):
    """{SampleName -> batch value} from the Atlas astro metadata xlsx, for one column.

    Reads all sheets once and caches the result both in-process and on disk
    (data/AtlasData/.batch_map_cache.pkl), re-reading only if the xlsx changes.
    """
    if col in _ATLAS_BATCH_MEM:
        return _ATLAS_BATCH_MEM[col]
    if not os.path.exists(ATLAS_ASTRO_META_XLSX):
        return {}
    src_mtime = os.path.getmtime(ATLAS_ASTRO_META_XLSX)

    # on-disk cache
    cache = {}
    if os.path.exists(_ATLAS_BATCH_CACHE_PKL):
        try:
            with open(_ATLAS_BATCH_CACHE_PKL, "rb") as f:
                cache = pickle.load(f)
        except Exception:
            cache = {}
    entry = cache.get(col)
    if entry and entry.get("mtime") == src_mtime:
        _ATLAS_BATCH_MEM[col] = entry["map"]
        return entry["map"]

    # (re)build from the xlsx (slow) and persist
    print(f"  [batch] reading {os.path.basename(ATLAS_ASTRO_META_XLSX)} for '{col}' (caching)...")
    sheets = pd.read_excel(ATLAS_ASTRO_META_XLSX, sheet_name=None)
    frames = [d[["SampleName", col]] for d in sheets.values()
              if col in d.columns and "SampleName" in d.columns]
    mapping = {}
    if frames:
        m = pd.concat(frames).dropna().drop_duplicates("SampleName")
        mapping = dict(zip(m["SampleName"].astype(str), m[col].astype(str)))
    cache[col] = {"mtime": src_mtime, "map": mapping}
    try:
        with open(_ATLAS_BATCH_CACHE_PKL, "wb") as f:
            pickle.dump(cache, f)
    except Exception:
        pass
    _ATLAS_BATCH_MEM[col] = mapping
    return mapping


def _batch_lut(region, cell_type, batch_col=None):
    """Build {bin_or_patient -> batch} for one region/cell_type, plus the column used.

    Primary source: the region's pseudobulk AnnData obs (keyed by _sample_id) --
    works for SEA-AD/MTG ('batch_vendor_name') and ROSMAP ('batch').
    Fallback for Atlas Astrocytes: the metadata xlsx 'Flowcell' keyed by SampleName.
    """
    import anndata as ad
    lut, col = {}, batch_col or _default_batch_col(region, cell_type)

    if region == "ROSMAP":
        ct = cell_type or "Microglia"
        anndata_file = os.path.join("data", "GRN", "ROSMAP", ct, f"{ct}_pseudobulk_AnnData.h5ad")
    else:
        try:
            _, _, anndata_file, *_ = get_grn_file_paths(region, cell_type=cell_type or "Astrocytes", data_dir="data")
        except Exception:
            anndata_file = None

    if anndata_file and os.path.exists(anndata_file):
        obs = ad.read_h5ad(anndata_file, backed="r").obs
        if col is None:  # no preference -> auto-detect a batch-like column from obs
            for cand in ["batch_vendor_name", "batch", "load_name", "Flowcell", "orig.ident", "plate"]:
                if cand in obs.columns:
                    col = cand
                    break
        if col and col in obs.columns and "_sample_id" in obs.columns:
            lut = dict(zip(obs["_sample_id"].astype(str), obs[col].astype(str)))
            return lut, col

    # Atlas Astrocytes fallback: xlsx column (default Flowcell) keyed by SampleName (cached)
    if cell_type in (None, "Astrocytes"):
        fcol = batch_col or col or "Flowcell"
        lut = _atlas_astro_batch_map(fcol)
        if lut:
            return lut, fcol

    return lut, col


def load_batch_labels(cell_names, cell_labels, batch_col=None):
    """Return a batch label per cell (aligned to cell_names).

    Looks up each bin by its _sample_id, falling back to the patient id
    (the part of '<patient>_bin<n>' before '_bin'). Missing -> None.
    """
    lut_cache = {}
    labels, used_cols = [], set()
    for name, lab in zip(cell_names, cell_labels):
        region, cell_type = _parse_region_celltype(lab)
        key = (region, cell_type)
        if key not in lut_cache:
            lut_cache[key] = _batch_lut(region, cell_type, batch_col)
            used_cols.add(lut_cache[key][1])
        lut, _ = lut_cache[key]
        patient = name.rsplit("_bin", 1)[0]
        labels.append(lut.get(name, lut.get(patient)))
    n_found = sum(l is not None for l in labels)
    print(f"  Batch labels: matched {n_found}/{len(labels)} cells; column(s) used: {sorted(c for c in used_cols if c)}")
    return labels


def concatenate_all_embeddings(cell_type_filter=None, regions_filter=None):
    """
    Automatically find and combine all cell embeddings from all regions.
    
    Parameters:
    -----------
    cell_type_filter : str, optional
        Cell type to filter by: 'Astrocytes' or 'Microglia'. If None, loads all cell types.
    
    Returns:
    --------
    concatenated_embeddings : np.ndarray
        Concatenated embedding matrix (cells × dimensions)
    all_conditions : list
        List of conditions for each cell
    cell_labels : list
        List of labels for each cell (format: "region" or "region_celltype")
    all_cell_names : list
        List of cell names
    """
    embeddings_dir = os.path.join("GREmLN", "embeddings")
    if not os.path.exists(embeddings_dir):
        raise ValueError(f"Embeddings directory not found: {embeddings_dir}")
    
    # Find all cell embedding files
    import glob
    embedding_files = glob.glob(os.path.join(embeddings_dir, "*_cell_embeddings.tsv"))
    embedding_files.extend(glob.glob(os.path.join(embeddings_dir, "*_cell_embeddings.csv")))
    
    print(f"\nFound {len(embedding_files)} embedding file(s)...")
    all_embeddings = []
    all_conditions = []
    all_cell_labels = []
    all_cell_names = []
    
    for emb_path in sorted(embedding_files):
        filename = os.path.basename(emb_path)
        # Extract region and cell type from filename (e.g., "EC_cell_embeddings.tsv" or "EC_Astrocytes_cell_embeddings.tsv")
        parts = filename.replace('_cell_embeddings.tsv', '').replace('_cell_embeddings.csv', '').split('_')
        
        if len(parts) >= 2 and parts[-1] in ['Astrocytes', 'Microglia']:
            region = '_'.join(parts[:-1])
            cell_type = parts[-1]
        else:
            # Assume it's just region name, check if it's Astrocytes
            region = '_'.join(parts)
            # Check if filename contains Microglia
            if 'Microglia' in filename:
                cell_type = 'Microglia'
            else:
                cell_type = None  # Assume Astrocytes for labels without explicit type
        
        # Filter by dataset (region allowlist) if specified
        if regions_filter is not None and region not in regions_filter:
            print(f"  Skipping {filename} (region: {region} not in {regions_filter})")
            continue

        # Filter by cell type if specified
        actual_cell_type = cell_type or 'Astrocytes'
        if cell_type_filter is not None and actual_cell_type != cell_type_filter:
            print(f"  Skipping {filename} (cell_type: {actual_cell_type}, filter: {cell_type_filter})")
            continue
        
        print(f"  Loading: {filename} (region: {region}, cell_type: {actual_cell_type})")
        embeddings, cell_names = load_cell_embeddings(emb_path)
        all_embeddings.append(embeddings)
        n_cells = embeddings.shape[0]
        all_cell_names.extend(cell_names)
        
        # Create cell label
        if cell_type:
            cell_label = f"{region}_{cell_type}"
        else:
            cell_label = region  # Assume Astrocytes
        all_cell_labels.extend([cell_label] * n_cells)
        
        # Try to load conditions from appropriate directory
        conditions = None
        mapping_path = _mapping_path(region, cell_type)
        if mapping_path and os.path.exists(mapping_path):
            conditions = load_conditions_from_mapping(mapping_path)
        
        if conditions is not None:
            if len(conditions) != n_cells:
                print(f"    Warning: {len(conditions)} conditions but {n_cells} cells - using None for conditions")
                conditions = [None] * n_cells
            all_conditions.extend(conditions)
        else:
            all_conditions.extend([None] * n_cells)
    
    # Concatenate embeddings
    if not all_embeddings:
        raise ValueError(f"No embedding files found matching cell_type_filter={cell_type_filter}")
    
    concatenated_embeddings = np.vstack(all_embeddings)
    print(f"\nConcatenated embeddings shape: {concatenated_embeddings.shape}")
    print(f"  Total cells: {concatenated_embeddings.shape[0]}")
    print(f"  Embedding dimensions: {concatenated_embeddings.shape[1]}")
    if cell_type_filter:
        print(f"  Filtered by cell type: {cell_type_filter}")
    
    # Count cells per label
    from collections import Counter
    label_counts = Counter(all_cell_labels)
    print(f"  Cell groups: {dict(label_counts)}")
    
    # Count conditions
    condition_counts = {}
    for cond in all_conditions:
        if cond:
            condition_counts[cond] = condition_counts.get(cond, 0) + 1
    if condition_counts:
        print(f"  Conditions: {condition_counts}")
    
    return concatenated_embeddings, all_conditions, all_cell_labels, all_cell_names



# Determine if we're loading multiple embeddings
point_colors = None
color_map = None
unique_regions = None
cell_conditions = None
cell_labels = None

if args.combine_all:
    # Combine all embeddings, optionally filtered by cell type
    embeddings_matrix, cell_conditions, cell_labels, cell_names = concatenate_all_embeddings(
        cell_type_filter=args.cells_to_keep, regions_filter=DATA_REGIONS.get(args.data))
    has_unmapped = any(c is None for c in cell_conditions)
    if args.cells_to_keep:
        region_label = f"All Regions ({args.cells_to_keep} only)"
    else:
        region_label = "All Regions"
else:
    # Load single region
    region = args.region
    embeddings_path = os.path.join("GREmLN", "embeddings", f"{region}_cell_embeddings.tsv")
    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(
            f"Embeddings not found: {embeddings_path} (data={args.data}, region={region})")
    embeddings_matrix, cell_names = load_cell_embeddings(embeddings_path)
    # Parse the file stem into (region, cell_type); avoid doubling an existing suffix
    _region, _ct = _parse_region_celltype(region)
    is_micro = 'Microglia' in embeddings_path
    if is_micro and not region.endswith("Microglia"):
        cell_labels = [f"{region}_Microglia"] * len(cell_names)
    else:
        cell_labels = [region] * len(cell_names)
    # Load conditions from the region's patient mapping (dataset-aware path)
    mapping_path = _mapping_path(_region, _ct)
    cell_conditions = load_conditions_from_mapping(mapping_path) if mapping_path else None
    if cell_conditions is not None and len(cell_conditions) != len(cell_names):
        print(f"  Warning: {len(cell_conditions)} conditions but {len(cell_names)} cells - ignoring conditions")
        cell_conditions = None
    region_label = region

print(f"Embedding matrix shape: {embeddings_matrix.shape}")
print(f"  - Number of cells: {embeddings_matrix.shape[0]}")
print(f"  - Embedding dimensions: {embeddings_matrix.shape[1]}")

# Test different UMAP dimensions to find optimal (using n_clusters=2 for testing)
umap_dimensions_to_test = [2] #[2, 4, 8, 16]
max_umap = min(50, embeddings_matrix.shape[1], embeddings_matrix.shape[0] - 1)
umap_dimensions_to_test = [d for d in umap_dimensions_to_test if d <= max_umap]

print(f"\nTesting UMAP dimensions: {umap_dimensions_to_test} (using n_clusters=2 for testing)...")
umap_results = {}
best_umap_dim = None
best_silhouette = -1
test_n_clusters = 2  # Use 2 for dimension testing

for n_comp in umap_dimensions_to_test:
    print(f"  Computing UMAP {n_comp}D...")
    reducer_test = umap.UMAP(n_components=n_comp, random_state=42, n_neighbors=15, min_dist=0.1)
    embeddings_test = reducer_test.fit_transform(embeddings_matrix)
    
    # Test clustering
    clusterer_test = KMeans(n_clusters=test_n_clusters, random_state=42, n_init=10)
    cluster_labels_test = clusterer_test.fit_predict(embeddings_test)
    silhouette_avg = silhouette_score(embeddings_test, cluster_labels_test)
    
    umap_results[n_comp] = silhouette_avg
    print(f"  UMAP {n_comp}D: silhouette_score={silhouette_avg:.4f}")
    
    if silhouette_avg > best_silhouette:
        best_silhouette = silhouette_avg
        best_umap_dim = n_comp

print(f"\nOptimal UMAP dimension: {best_umap_dim}D (silhouette_score: {best_silhouette:.4f})")

# Plot UMAP dimension comparison
if len(umap_results) > 1:
    fig, ax = plt.subplots(figsize=(10, 6))
    umap_dims = list(umap_results.keys())
    silhouettes = list(umap_results.values())
    
    ax.plot(umap_dims, silhouettes, marker='o', linewidth=2, markersize=8, color='purple')
    ax.axvline(x=best_umap_dim, color='r', linestyle='--', linewidth=2, label=f'Optimal: {best_umap_dim}D')
    ax.set_xlabel('UMAP Dimensions', fontsize=12)
    ax.set_ylabel('Silhouette Score', fontsize=12)
    ax.set_title('Silhouette Score vs UMAP Dimensions', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(umap_dims)
    plt.tight_layout()
    plt.show()

# Use optimal UMAP dimension for clustering
print(f"\nUsing {best_umap_dim}D UMAP for clustering...")
reducer_optimal = umap.UMAP(n_components=best_umap_dim, random_state=42, n_neighbors=15, min_dist=0.1)
embeddings_optimal = reducer_optimal.fit_transform(embeddings_matrix)

# Use 4 clusters directly
n_clusters = 7

# Cluster embeddings with 4 clusters
print(f"\nClustering with n_clusters={n_clusters} on {best_umap_dim}D UMAP...")
clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
cluster_labels = clusterer.fit_predict(embeddings_optimal)
silhouette_avg = silhouette_score(embeddings_optimal, cluster_labels)
print(f"Silhouette score: {silhouette_avg:.4f}")

# Count cells per cluster
from collections import Counter
cluster_counts = Counter(cluster_labels)
print(f"\nCells per cluster:")
for cluster_id in sorted(cluster_counts.keys()):
    print(f"  Cluster {cluster_id}: {cluster_counts[cluster_id]} cells")

# Save each cluster to a separate file (only when --save_clusters is set)
all_cluster_ids = sorted(cluster_counts.keys())
if args.save_clusters:
    print(f"\nSaving clusters to separate files:")
    for cluster_id in all_cluster_ids:
        cluster_cell_indices = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
        cluster_cell_names = [cell_names[i] for i in cluster_cell_indices]

        # Save this cluster to a separate file
        cluster_pickle = f"data/Astrocyte_cluster_{cluster_id}_cells.pkl"
        print(f"  Saving cluster {cluster_id} ({len(cluster_cell_names)} cells) to: {cluster_pickle}")
        with open(cluster_pickle, 'wb') as f:
            pickle.dump({
                'cluster_id': cluster_id,
                'cell_names': cluster_cell_names,
                'n_cells': len(cluster_cell_names)
            }, f)
        print(f"  ✓ Saved {len(cluster_cell_names)} cells from cluster {cluster_id}")
    print(f"\n✓ All {len(all_cluster_ids)} clusters saved successfully")
else:
    print(f"\nSkipping cluster pickle export ({len(all_cluster_ids)} clusters); pass --save_clusters to write them.")

# Compute 2D UMAP for visualization
print("\nComputing 2D UMAP for visualization...")
reducer_2d = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
embeddings_2d = reducer_2d.fit_transform(embeddings_matrix)

# Plot embedding
fig, ax = plt.subplots(figsize=(12, 10))

# Determine coloring scheme
if args.color_by == 'cell_type' and cell_labels is not None:
    plot_colors, color_map, unique_values, extracted_values = create_color_mapping(cell_labels, attribute_type='cell_type')
    title_suffix = 'Cell Type'
    label_prefix = ''
elif args.color_by == 'region' and cell_labels is not None:
    plot_colors, color_map, unique_values, extracted_values = create_color_mapping(cell_labels, attribute_type='region')
    title_suffix = 'Region'
    label_prefix = ''
elif args.color_by == 'condition' and cell_conditions is not None:
    plot_colors, color_map, unique_values, extracted_values = create_color_mapping(cell_conditions, attribute_type='condition')
    title_suffix = 'Condition'
    label_prefix = 'Condition '
elif args.color_by == 'batch' and cell_labels is not None:
    batch_labels = load_batch_labels(cell_names, cell_labels, batch_col=args.batch_col)
    plot_colors, color_map, unique_values, extracted_values = create_color_mapping(batch_labels, attribute_type='generic')
    title_suffix = f"Batch ({args.batch_col or 'auto'})"
    label_prefix = ''
else:
    plot_colors = None

if plot_colors is not None:
    # Plot with colors
    scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=plot_colors, alpha=0.6, s=20)
    title = f'{region_label} Cell Embeddings - UMAP (Colored by {title_suffix})'
    
    # Create legend
    legend_elements = []
    value_counts = Counter(extracted_values)
    for value in unique_values:
        color = color_map[value]
        count = value_counts.get(value, 0)
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                           markerfacecolor=color, markersize=10, 
                                           label=f'{label_prefix}{value} (n={count})'))
    ax.legend(handles=legend_elements, loc='best', fontsize=10)
    
else:
    # Default: Color by cluster
    scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=cluster_labels, alpha=0.6, s=20, cmap='tab10')
    title = f'{region_label} Cell Embeddings - UMAP (Clustered on {best_umap_dim}D, {n_clusters} clusters, Silhouette: {silhouette_avg:.4f})'
    
    # Add legend for clusters
    unique_clusters = sorted(set(cluster_labels))
    legend_elements = []
    for cluster_id in unique_clusters:
        color = plt.cm.tab10(cluster_id / max(1, max(unique_clusters)))
        count = cluster_counts[cluster_id]
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                           markerfacecolor=color, markersize=10, 
                                           label=f'Cluster {cluster_id} (n={count})'))
    ax.legend(handles=legend_elements, loc='best', fontsize=10)

ax.set_xlabel('UMAP Dimension 1', fontsize=12)
ax.set_ylabel('UMAP Dimension 2', fontsize=12)
ax.set_title(title, fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
