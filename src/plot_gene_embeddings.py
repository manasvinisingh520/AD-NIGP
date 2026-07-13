"""
Plot gene embeddings in 2D using UMAP and/or PCA.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from sklearn.decomposition import PCA

from utils import create_color_mapping

# Same convention as plot_cell_embeddings: <stem>_gene_embeddings.tsv.
DATA_DEFAULT_CELL_TYPE = {"ROSMAP": "Microglia", "SEAAD": "Astrocytes", "Atlas": "Astrocytes"}


def _gene_embedding_stem(data, cell_type, region):
    """Resolve the <stem> in <stem>_gene_embeddings.tsv from data + cell_type (+ Atlas region).

    ROSMAP -> ROSMAP_<ct>; SEAAD -> SEAAD_<ct>; Atlas -> <region> (astro) or <region>_Microglia.
    """
    ct = cell_type or DATA_DEFAULT_CELL_TYPE.get(data)
    if data == "ROSMAP":
        return f"ROSMAP_{ct}"
    if data == "SEAAD":
        return f"SEAAD_{ct}"
    if data == "Atlas":
        reg = region or "V2"
        return f"{reg}_Microglia" if ct == "Microglia" else reg
    return region


def load_gene_embeddings(embeddings_path):
    """Load gene embeddings from TSV/CSV file."""
    try:
        df = pd.read_csv(embeddings_path, sep=',', header=None)
    except:
        df = pd.read_csv(embeddings_path, sep='\t', header=None)
    
    embeddings_list = []
    gene_ids = []
    
    # Skip the first row (header)
    for idx in range(1, len(df)):
        row = df.iloc[idx]
        gene_id = str(row.iloc[0])
        embedding_values = row.iloc[1:].values
        embedding_values = [float(x) for x in embedding_values if pd.notna(x)]
        if len(embedding_values) > 0:
            embeddings_list.append(embedding_values)
            gene_ids.append(gene_id)
    
    return np.array(embeddings_list), gene_ids


def concatenate_all_gene_embeddings():
    """
    Automatically find and combine all gene embeddings from all regions for both Astrocytes and Microglia.
    
    Returns:
    --------
    concatenated_embeddings : np.ndarray
        Concatenated embedding matrix (genes × dimensions)
    all_gene_ids : list
        List of gene IDs for each gene
    gene_labels : list
        List of region labels for each gene
    cell_types : list
        List of cell types (Astrocytes/Microglia) for each gene
    """
    embeddings_dir = os.path.join("GREmLN", "embeddings")
    if not os.path.exists(embeddings_dir):
        raise ValueError(f"Embeddings directory not found: {embeddings_dir}")
    
    # Find all gene embedding files
    import glob
    embedding_files = glob.glob(os.path.join(embeddings_dir, "*_gene_embeddings.tsv"))
    embedding_files.extend(glob.glob(os.path.join(embeddings_dir, "*_gene_embeddings.csv")))
    
    print(f"\nFound {len(embedding_files)} gene embedding file(s)...")
    all_embeddings = []
    all_gene_ids = []
    all_gene_labels = []
    all_cell_types = []
    
    for emb_path in sorted(embedding_files):
        filename = os.path.basename(emb_path)
        # Extract region and cell type from filename
        # e.g., "EC_gene_embeddings.tsv" (Astrocytes) or "EC_Microglia_gene_embeddings.tsv" (Microglia)
        base_name = filename.replace('_gene_embeddings.tsv', '').replace('_gene_embeddings.csv', '')
        
        # Determine cell type
        if 'Microglia' in filename:
            cell_type = 'Microglia'
            # Extract region (remove _Microglia from base_name)
            region = base_name.replace('_Microglia', '').replace('Microglia', '').strip('_')
        else:
            cell_type = 'Astrocytes'
            region = base_name.replace('_TF', '')  # Remove _TF if present
        
        print(f"  Loading: {filename} (region: {region}, cell_type: {cell_type})")
        embeddings, gene_ids = load_gene_embeddings(emb_path)
        
        # Skip empty embeddings
        if embeddings.shape[0] == 0:
            print(f"    Warning: Skipping empty embedding file")
            continue
        
        # Check embedding dimensions
        if len(all_embeddings) > 0:
            expected_dim = all_embeddings[0].shape[1]
            if embeddings.shape[1] != expected_dim:
                print(f"    Warning: Skipping {filename} - dimension mismatch ({embeddings.shape[1]} vs {expected_dim})")
                continue
        
        all_embeddings.append(embeddings)
        n_genes = embeddings.shape[0]
        
        # Create gene labels (region_celltype for each gene)
        gene_labels = [f"{region}_{cell_type}"] * n_genes
        all_gene_labels.extend(gene_labels)
        all_cell_types.extend([cell_type] * n_genes)
        all_gene_ids.extend(gene_ids)
    
    if len(all_embeddings) == 0:
        raise ValueError("No valid gene embeddings found to concatenate")
    
    # Concatenate embeddings
    concatenated_embeddings = np.vstack(all_embeddings)
    print(f"\nConcatenated embeddings shape: {concatenated_embeddings.shape}")
    print(f"  Total genes: {concatenated_embeddings.shape[0]}")
    print(f"  Embedding dimensions: {concatenated_embeddings.shape[1]}")
    
    # Count genes per region and cell type
    from collections import Counter
    label_counts = Counter(all_gene_labels)
    print(f"  Genes per region_celltype: {dict(label_counts)}")
    
    cell_type_counts = Counter(all_cell_types)
    print(f"  Genes per cell type: {dict(cell_type_counts)}")
    
    return concatenated_embeddings, all_gene_ids, all_gene_labels, all_cell_types


def resolve_colors(color_by: str, cell_types, gene_labels):
    """Return scatter color array / mapping and a short note for plot titles."""
    if color_by == "none":
        return "gray", None, None, None
    if color_by == "cell_type" and cell_types is not None:
        if cell_types and not any("_" in str(ct) for ct in cell_types[:10]):
            custom_colors = {
                "Astrocytes": "#1f77b4",
                "Microglia": "#ff7f0e",
            }
            point_colors, color_map, unique_values, _ = create_color_mapping(
                cell_types, attribute_type="generic", custom_colors=custom_colors
            )
        else:
            point_colors, color_map, unique_values, _ = create_color_mapping(
                cell_types, attribute_type="cell_type"
            )
        return point_colors, color_map, unique_values, "Colored by Cell Type"
    if color_by == "region" and gene_labels is not None:
        point_colors, color_map, unique_values, _ = create_color_mapping(
            gene_labels, attribute_type="region"
        )
        return point_colors, color_map, unique_values, "Colored by Region"
    if color_by == "auto":
        if cell_types is not None:
            if cell_types and not any("_" in str(ct) for ct in cell_types[:10]):
                custom_colors = {
                    "Astrocytes": "#1f77b4",
                    "Microglia": "#ff7f0e",
                }
                point_colors, color_map, unique_values, _ = create_color_mapping(
                    cell_types, attribute_type="generic", custom_colors=custom_colors
                )
            else:
                point_colors, color_map, unique_values, _ = create_color_mapping(
                    cell_types, attribute_type="cell_type"
                )
            return point_colors, color_map, unique_values, "Colored by Cell Type"
        if gene_labels is not None:
            point_colors, color_map, unique_values, _ = create_color_mapping(
                gene_labels, attribute_type="region"
            )
            return point_colors, color_map, unique_values, "Colored by Region"
        return "gray", None, None, None
    return "gray", None, None, None


def compute_embeddings_2d(embeddings_matrix: np.ndarray, method: str, args):
    """Return list of (label, xy ndarray). method: umap, pca, or both."""
    X = np.asarray(embeddings_matrix, dtype=np.float64)
    if not np.isfinite(X).all():
        n_bad = int(np.sum(~np.isfinite(X)))
        print(f"Warning: {n_bad} non-finite values in embedding matrix; replacing with 0 for reduction.")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    out = []
    if method in ("umap", "both"):
        import umap

        print("\nComputing UMAP (2D)...")
        reducer = umap.UMAP(
            n_components=2,
            random_state=42,
            n_neighbors=int(getattr(args, "umap_n_neighbors", 15)),
            min_dist=float(getattr(args, "umap_min_dist", 0.1)),
        )
        xy = reducer.fit_transform(X)
        out.append(("UMAP", np.asarray(xy, dtype=np.float64)))
    if method in ("pca", "both"):
        n_comp = min(2, X.shape[0], X.shape[1])
        print(f"\nComputing PCA (2D, n_comp={n_comp})...")
        pca = PCA(n_components=n_comp, random_state=42)
        xy = pca.fit_transform(X)
        print(f"  Explained variance ratio: {pca.explained_variance_ratio_}")
        print(f"  Cumulative explained variance: {float(np.sum(pca.explained_variance_ratio_)):.4f}")
        # Always return (n, 2) for plotting (pad if rank-deficient)
        if xy.ndim == 1:
            xy = xy.reshape(-1, 1)
        if xy.shape[1] == 1:
            xy = np.concatenate([xy, np.zeros((xy.shape[0], 1), dtype=xy.dtype)], axis=1)
            print("  Note: only one PCA component available; second axis padded with zeros.")
        out.append(("PCA", np.asarray(xy, dtype=np.float64)))
    return out


parser = argparse.ArgumentParser(
    description="Plot gene embeddings in 2D using UMAP and/or PCA.",
)
parser.add_argument('--data', type=str, default=None, choices=['Atlas', 'SEAAD', 'ROSMAP'],
                    help='Dataset to resolve the embedding file stem, mirroring plot_cell_embeddings '
                         '(ROSMAP -> ROSMAP_<ct>, SEAAD -> SEAAD_<ct>, Atlas -> <region>[_Microglia]). '
                         'Uses --cell_type (and --region for Atlas). Overridden by --embeddings_path.')
parser.add_argument('--region', type=str, default='EC', help='Region to process (default: EC). For Atlas this is the brain region (EC/ITG/...); for SEAAD/ROSMAP it is ignored when --data is given.')
parser.add_argument('--embeddings_path', type=str, default=None, help='Path to embeddings file (default: auto-detect)')
parser.add_argument('--combine_all', action='store_true', help='Combine all gene embeddings from all regions (Astrocytes and Microglia)')
parser.add_argument('--cell_type', type=str, default=None, choices=['Astrocytes', 'Microglia'],
                    help='Cell type. With --data it selects the file stem (SEAAD/ROSMAP <ct>, Atlas Microglia suffix); '
                         'with --combine_all it filters the combined set. If not specified, dataset default is used.')
parser.add_argument('--save', action='store_true', help='Save the figure to --outdir instead of only showing it.')
parser.add_argument('--outdir', type=str, default=os.path.join('images', 'gene_embeddings'),
                    help='Directory for --save (default: images/gene_embeddings).')
parser.add_argument('--color_by', type=str, default='auto', choices=['cell_type', 'region', 'none', 'auto'],
                    help='What to color by: cell_type, region, none (single color), or auto (detect from data)')
parser.add_argument(
    "--method",
    type=str,
    default="umap",
    choices=["umap", "pca", "both"],
    help="2D reduction: umap (default), pca, or both (side-by-side).",
)
parser.add_argument("--umap_n_neighbors", type=int, default=15, help="UMAP n_neighbors (default: 15).")
parser.add_argument("--umap_min_dist", type=float, default=0.1, help="UMAP min_dist (default: 0.1).")
args = parser.parse_args()

# Determine if we're loading multiple embeddings
gene_labels = None
cell_types = None

if args.combine_all:
    # Combine all gene embeddings from all regions (Astrocytes and Microglia)
    embeddings_matrix, gene_ids, gene_labels, cell_types = concatenate_all_gene_embeddings()
    region_label = "All Regions (Astrocytes and Microglia)"
else:
    # Load a single dataset/region embedding file.
    if args.embeddings_path is not None:
        embeddings_path = args.embeddings_path
        region_label = args.region
    else:
        # --data resolves the stem (SEAAD_<ct>, ROSMAP_<ct>, Atlas <region>[_Microglia]);
        # otherwise fall back to the literal --region stem (back-compat).
        stem = _gene_embedding_stem(args.data, args.cell_type, args.region) if args.data else args.region
        embeddings_path = os.path.join("GREmLN", "embeddings", f"{stem}_gene_embeddings.tsv")
        region_label = stem
    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(
            f"Gene embeddings not found: {embeddings_path} "
            f"(data={args.data}, region={args.region}, cell_type={args.cell_type})")

    embeddings_matrix, gene_ids = load_gene_embeddings(embeddings_path)
    gene_labels = None
    cell_types = None

# Filter by cell_type if specified
if args.cell_type is not None:
    if cell_types is None:
        print(f"Warning: cell_type filter specified but no cell type information available. Ignoring filter.")
    else:
        # Filter embeddings, gene_ids, gene_labels, and cell_types
        mask = np.array([ct == args.cell_type for ct in cell_types])
        embeddings_matrix = embeddings_matrix[mask]
        gene_ids = [gene_ids[i] for i in range(len(gene_ids)) if mask[i]]
        if gene_labels is not None:
            gene_labels = [gene_labels[i] for i in range(len(gene_labels)) if mask[i]]
        cell_types = [cell_types[i] for i in range(len(cell_types)) if mask[i]]
        print(f"Filtered to {args.cell_type}: {len(gene_ids)} genes")

print(f"Embedding matrix shape: {embeddings_matrix.shape}")
print(f"  - Number of genes: {embeddings_matrix.shape[0]}")
print(f"  - Embedding dimensions: {embeddings_matrix.shape[1]}")

plots_2d = compute_embeddings_2d(embeddings_matrix, args.method, args)

if args.color_by == "cell_type" and cell_types is None:
    print(
        "Warning: color_by=cell_type but no cell type metadata (use --combine_all). "
        "Falling back to single color."
    )
elif args.color_by == "region" and gene_labels is None:
    print(
        "Warning: color_by=region but no region labels (use --combine_all). "
        "Falling back to single color."
    )

point_colors, color_map, unique_values, color_note = resolve_colors(
    args.color_by, cell_types, gene_labels
)

n_plots = len(plots_2d)
fig_w = 12 * n_plots
fig, axes = plt.subplots(1, n_plots, figsize=(fig_w, 10))
if n_plots == 1:
    axes = [axes]

title_mid = f" ({color_note})" if color_note else ""

for ax_idx, (ax, (reducer_name, xy)) in enumerate(zip(axes, plots_2d)):
    ax.scatter(
        xy[:, 0],
        xy[:, 1],
        c=point_colors,
        alpha=0.6,
        s=20,
        edgecolors="none",
    )
    ax.set_xlabel(f"{reducer_name} component 1", fontsize=12)
    ax.set_ylabel(f"{reducer_name} component 2", fontsize=12)
    ax.set_title(
        f"{region_label} Gene Embeddings — {reducer_name}{title_mid}",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)

    if reducer_name == "UMAP":
        ax.relim()
        ax.autoscale_view()
        try:
            if np.nanmax(np.abs(xy)) < 30:
                ax.set_xlim(-5, 25)
                ax.set_ylim(-15, 15)
        except Exception:
            pass
    else:
        # PCA: explicit limits from data (avoid invisible panels when autoscale/legend interact)
        x0, x1 = float(np.nanmin(xy[:, 0])), float(np.nanmax(xy[:, 0]))
        y0, y1 = float(np.nanmin(xy[:, 1])), float(np.nanmax(xy[:, 1]))
        rx = max(x1 - x0, 1e-12)
        ry = max(y1 - y0, 1e-12)
        pad_x, pad_y = 0.05 * rx, 0.05 * ry
        ax.set_xlim(x0 - pad_x, x1 + pad_x)
        ax.set_ylim(y0 - pad_y, y1 + pad_y)

    # A large categorical legend on every panel (especially the second) often covers all points.
    show_legend_here = color_map is not None and unique_values is not None
    if show_legend_here:
        legend_elements = [
            Patch(facecolor=color_map[val], label=val) for val in unique_values
        ]
        if n_plots > 1 and ax_idx > 0:
            pass
        else:
            ax.legend(
                handles=legend_elements,
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                fontsize=9,
                borderaxespad=0.0,
            )

plt.tight_layout()
# Legend outside the first axes: give the row room so UMAP/PCA are not clipped.
if color_map is not None and unique_values is not None:
    if n_plots > 1:
        fig.subplots_adjust(wspace=0.22, right=0.88)
    else:
        fig.subplots_adjust(right=0.88)

if args.save:
    os.makedirs(args.outdir, exist_ok=True)
    stem = str(region_label).replace(" ", "_").replace("(", "").replace(")", "")
    fname = f"{stem}_gene_embeddings_{args.method}_{args.color_by}.png"
    out_path = os.path.join(args.outdir, fname)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to: {out_path}")
else:
    plt.show()
