# AnnData Object Guide

## What is an AnnData Object?

**AnnData** (Annotated Data) is a Python package designed for handling annotated matrices in computational biology, particularly **single-cell RNA sequencing** (scRNA-seq) data. It's the standard data structure used by packages like Scanpy, CellRanger, and many other single-cell analysis tools.

### Key Concepts

An AnnData object stores:
- **`.X`**: The main data matrix (cells × genes) - can be sparse or dense
- **`.obs`**: Cell-level metadata (annotations for each cell)
- **`.var`**: Gene-level metadata (annotations for each gene)
- **`.obsm`**: Multi-dimensional cell-level annotations (e.g., PCA, UMAP coordinates)
- **`.varm`**: Multi-dimensional gene-level annotations
- **`.obsp`**: Pairwise cell-level annotations (e.g., distance matrices)
- **`.varp`**: Pairwise gene-level annotations
- **`.uns`**: Unstructured metadata (e.g., analysis parameters, color schemes)

### Why Use AnnData?

1. **Standardized format**: Works seamlessly with Scanpy, scvelo, and other tools
2. **Memory efficient**: Handles sparse matrices natively
3. **Organized**: Keeps data, metadata, and analysis results together
4. **Versatile**: Supports multiple data types and annotations

## Core AnnData Methods and Attributes

### Creating AnnData Objects

```python
import anndata as ad
import numpy as np
from scipy.sparse import csr_matrix

# From a numpy array or sparse matrix
adata = ad.AnnData(X)  # X is cells × genes matrix

# With gene and cell names
adata = ad.AnnData(
    X,
    obs=cell_metadata,      # pandas DataFrame with cell info
    var=gene_metadata,       # pandas DataFrame with gene info
    obs_names=cell_barcodes, # list of cell IDs
    var_names=gene_names     # list of gene IDs
)
```

### Accessing Data

```python
# Main expression matrix
adata.X                    # Expression matrix (cells × genes)
adata.X.shape              # (n_cells, n_genes)

# Cell metadata (DataFrame)
adata.obs                  # Cell annotations
adata.obs['cell_type']     # Specific column
adata.obs_names            # Cell barcode names

# Gene metadata (DataFrame)
adata.var                  # Gene annotations
adata.var['gene_symbol']   # Specific column
adata.var_names            # Gene names/IDs

# Multi-dimensional annotations
adata.obsm['X_pca']        # PCA coordinates (cells × n_components)
adata.obsm['X_umap']       # UMAP coordinates (cells × 2)
adata.varm['PCs']          # Gene loadings for PCA

# Unstructured metadata
adata.uns['neighbors']     # Nearest neighbor information
adata.uns['pca']           # PCA parameters
```

### Basic Operations

```python
# Copy
adata_copy = adata.copy()

# Subset cells (rows)
adata_subset = adata[adata.obs['cell_type'] == 'astrocyte']

# Subset genes (columns)
adata_subset = adata[:, adata.var_names.isin(['GAPDH', 'ACTB'])]

# Subset by indices
adata_subset = adata[0:100, 0:50]  # First 100 cells, first 50 genes

# Filter
adata_filtered = adata[adata.obs['n_genes'] > 200]  # Cells with >200 genes
```

### Adding Metadata

```python
# Add cell-level annotations
adata.obs['n_genes'] = (adata.X > 0).sum(axis=1)  # Number of genes per cell
adata.obs['total_counts'] = adata.X.sum(axis=1)    # Total UMIs per cell

# Add gene-level annotations
adata.var['n_cells'] = (adata.X > 0).sum(axis=0)  # Number of cells expressing gene
adata.var['mean_expression'] = adata.X.mean(axis=0)  # Mean expression per gene

# Add multi-dimensional data
adata.obsm['X_pca'] = pca_coordinates  # Store PCA results
adata.obsm['X_umap'] = umap_coordinates  # Store UMAP results

# Add unstructured metadata
adata.uns['neighbors'] = {'connectivities': ...}
adata.uns['cell_type_colors'] = {'astrocyte': '#1f77b4', ...}
```

### File I/O

```python
# Save to HDF5 format (.h5ad) - recommended
adata.write('data.h5ad')
adata = ad.read_h5ad('data.h5ad')

# Save to CSV (WARNING: loses metadata structure)
adata.write_csvs('output', skip_data=False)

# Read from other formats
adata = ad.read_csv('expression.csv')      # CSV file
adata = ad.read_loom('data.loom')         # Loom format
adata = ad.read_mtx('matrix.mtx')          # MTX format
adata = ad.read_text('data.txt')           # Text file
```

### Useful Methods

```python
# View summary
print(adata)  # Prints object summary

# Shape and size
adata.shape           # (n_cells, n_genes)
adata.n_obs           # Number of cells
adata.n_vars          # Number of genes

# Check if sparse
adata.isbacked        # False if in memory, True if backed by file
adata.isview          # True if it's a view (subsets without copying)

# Memory management
adata.filename        # Path if backed by file
adata.file.close()    # Close backed file connection

# Convert to view (memory efficient)
adata_view = adata[0:100].copy()  # Creates copy
adata_view = adata[0:100]          # Creates view (lazy evaluation)

# Convert view to array
adata_view = adata_view.copy()     # Materialize view

# Chunked operations (for large files)
for chunk in adata.chunked(1000):  # Process 1000 cells at a time
    process(chunk)
```

### Integration with Scanpy

```python
import scanpy as sc

# Read 10X Genomics data
adata = sc.read_10x_mtx('path/to/directory')

# Basic analysis workflow
sc.pp.filter_cells(adata, min_genes=200)      # Filter cells
sc.pp.filter_genes(adata, min_cells=3)        # Filter genes
sc.pp.normalize_total(adata, target_sum=1e4)   # Normalize
sc.pp.log1p(adata)                             # Log transform
sc.pp.highly_variable_genes(adata)            # Find HVGs
sc.pp.scale(adata)                             # Scale
sc.tl.pca(adata)                                # PCA
sc.tl.umap(adata)                               # UMAP
sc.tl.leiden(adata)                             # Clustering

# Results stored in adata
adata.obsm['X_pca']     # PCA coordinates
adata.obsm['X_umap']    # UMAP coordinates
adata.obs['leiden']     # Cluster assignments
adata.var['highly_variable']  # HVG flags
```

### Common Patterns

```python
# Calculate QC metrics
adata.obs['n_genes_by_counts'] = (adata.X > 0).sum(axis=1)
adata.obs['total_counts'] = adata.X.sum(axis=1).A1 if issparse(adata.X) else adata.X.sum(axis=1)
adata.var['n_cells_by_counts'] = (adata.X > 0).sum(axis=0).A1 if issparse(adata.X) else (adata.X > 0).sum(axis=0)

# Calculate mitochondrial percentage
import pandas as pd
mito_genes = adata.var_names.str.startswith('MT-')
adata.obs['percent_mito'] = np.sum(adata[:, mito_genes].X, axis=1).A1 / np.sum(adata.X, axis=1).A1 * 100

# Add layer for normalized data
adata.layers['normalized'] = normalized_matrix
adata.layers['log_normalized'] = log_normalized_matrix

# Access layer
normalized_exp = adata.layers['normalized']
```

### Advanced Features

```python
# Raw data
adata.raw = adata  # Store unprocessed data
adata = adata[:, filtered_genes]  # Process filtered data
# adata.raw still has all genes

# Multiple layers
adata.layers['spliced'] = spliced_matrix
adata.layers['unspliced'] = unspliced_matrix

# Backed mode (for large files)
adata = ad.read_h5ad('large_file.h5ad', backed='r')  # Read-only, don't load into memory
adata_subset = adata[0:1000].copy()  # Load subset into memory
```

## Key Methods Reference

### Core Methods
- `adata.copy()` - Create a copy
- `adata.shape` - Dimensions (n_cells, n_genes)
- `adata.write(filename)` - Save to file
- `ad.read_h5ad(filename)` - Read from file

### Subsetting
- `adata[i, j]` - Index cells and genes
- `adata[:, gene_names]` - Select specific genes
- `adata[cell_mask, :]` - Select specific cells

### Metadata Access
- `adata.obs` - Cell metadata DataFrame
- `adata.var` - Gene metadata DataFrame
- `adata.obsm` - Multi-dimensional cell data
- `adata.varm` - Multi-dimensional gene data
- `adata.uns` - Unstructured metadata

### Attributes
- `adata.n_obs` - Number of cells
- `adata.n_vars` - Number of genes
- `adata.X` - Main expression matrix
- `adata.obs_names` - Cell barcode names
- `adata.var_names` - Gene names

## Example: Complete Workflow

```python
import anndata as ad
import scanpy as sc
import pandas as pd
import numpy as np

# 1. Read data
adata = sc.read_10x_mtx('data/')

# 2. Add cell metadata
metadata = pd.read_csv('cell_metadata.csv', index_col=0)
adata.obs = metadata

# 3. Calculate QC metrics
adata.obs['n_genes'] = (adata.X > 0).sum(axis=1)
adata.obs['total_counts'] = adata.X.sum(axis=1)

# 4. Filter
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)

# 5. Normalize and process
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata)

# 6. Analysis
sc.pp.scale(adata)
sc.tl.pca(adata)
sc.tl.umap(adata)
sc.tl.leiden(adata)

# 7. Save
adata.write('processed_data.h5ad')

# 8. Reload later
adata = ad.read_h5ad('processed_data.h5ad')
```

## Resources

- **Official Documentation**: https://anndata.readthedocs.io/
- **Scanpy Tutorial**: https://scanpy-tutorials.readthedocs.io/
- **GitHub**: https://github.com/scverse/anndata

