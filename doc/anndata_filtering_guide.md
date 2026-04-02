# AnnData Filtering Guide

This guide demonstrates how to filter AnnData objects by cells (rows) and genes (columns) using various methods.

## Table of Contents
1. [Basic Filtering Syntax](#basic-filtering-syntax)
2. [Filtering Cells (Rows)](#filtering-cells-rows)
3. [Filtering Genes (Columns)](#filtering-genes-columns)
4. [Combined Filtering](#combined-filtering)
5. [Using Scanpy Filtering Functions](#using-scanpy-filtering-functions)
6. [Common Filtering Patterns](#common-filtering-patterns)

---

## Basic Filtering Syntax

AnnData uses NumPy/Pandas-style indexing:

```python
import anndata as ad
import numpy as np
import pandas as pd

# Filter cells (rows) - first dimension
adata_filtered = adata[mask, :]  # or just adata[mask]

# Filter genes (columns) - second dimension
adata_filtered = adata[:, mask]

# Filter both
adata_filtered = adata[cell_mask, gene_mask]
```

**Important**: Filtering creates a **view** by default (memory efficient). Use `.copy()` to create an independent copy:

```python
adata_filtered = adata[mask].copy()  # Creates a copy
adata_view = adata[mask]              # Creates a view (lazy evaluation)
```

---

## Filtering Cells (Rows)

### 1. Filter by Metadata Column (Boolean Mask)

```python
# Filter cells by condition/group
mask = adata.obs['Path..Group.'] == 'Control'
adata_filtered = adata[mask].copy()

# Filter by multiple conditions
mask = adata.obs['Path..Group.'].isin(['Control', 'Disease'])
adata_filtered = adata[mask].copy()

# Filter by numeric threshold
mask = adata.obs['Median.UMI.Counts.per.Cell'] > 200
adata_filtered = adata[mask].copy()

# Filter by multiple criteria (AND)
mask = (adata.obs['Median.UMI.Counts.per.Cell'] > 200) & \
       (adata.obs['percent.mito'] < 0.15)
adata_filtered = adata[mask].copy()

# Filter by multiple criteria (OR)
mask = (adata.obs['RIN'] > 8) | (adata.obs['Total.Genes.Detected'] > 3000)
adata_filtered = adata[mask].copy()
```

### 2. Filter by Cell Names/Indices

```python
# Filter by specific cell barcodes
cells_to_keep = ['AAACCCACAGGTGTTT-1_6289-MW-0031', 'AAACCCACAGGTGTTT-2_6289-MW-0031']
mask = adata.obs_names.isin(cells_to_keep)
adata_filtered = adata[mask].copy()

# Filter by index range
adata_subset = adata[0:1000].copy()  # First 1000 cells

# Filter by specific indices
indices = [0, 5, 10, 15, 20]
adata_subset = adata[indices].copy()
```

### 3. Filter by Calculated Metrics

```python
# Calculate QC metrics first
from scipy.sparse import issparse

# Number of genes detected per cell
adata.obs['n_genes'] = (adata.X > 0).sum(axis=1).A1 if issparse(adata.X) else (adata.X > 0).sum(axis=1)

# Total UMIs per cell
adata.obs['total_counts'] = adata.X.sum(axis=1).A1 if issparse(adata.X) else adata.X.sum(axis=1)

# Filter based on calculated metrics
mask = (adata.obs['n_genes'] > 200) & (adata.obs['total_counts'] > 1000)
adata_filtered = adata[mask].copy()
```

### 4. Filter by Mitochondrial Content

```python
# Calculate mitochondrial percentage
mito_genes = adata.var_names.str.startswith('MT-')
if mito_genes.sum() > 0:
    adata.obs['percent_mito'] = (
        adata[:, mito_genes].X.sum(axis=1).A1 / 
        adata.X.sum(axis=1).A1 * 100
    )
    
    # Filter cells with low mitochondrial content
    mask = adata.obs['percent_mito'] < 15
    adata_filtered = adata[mask].copy()
```

---

## Filtering Genes (Columns)

### 1. Filter by Gene Names

```python
# Filter by specific gene names
genes_to_keep = ['GAPDH', 'ACTB', 'MYC']
mask = adata.var_names.isin(genes_to_keep)
adata_filtered = adata[:, mask].copy()

# Filter genes matching a pattern
mask = adata.var_names.str.startswith('MT-')  # Mitochondrial genes
adata_mito = adata[:, mask].copy()

# Filter genes NOT matching a pattern
mask = ~adata.var_names.str.startswith('MT-')  # Non-mitochondrial genes
adata_no_mito = adata[:, mask].copy()
```

### 2. Filter by Gene Metadata

```python
# If you have gene metadata
if 'highly_variable' in adata.var.columns:
    mask = adata.var['highly_variable']
    adata_hvg = adata[:, mask].copy()

# Filter by expression level
gene_means = adata.X.mean(axis=0).A1 if issparse(adata.X) else adata.X.mean(axis=0)
adata.var['mean_expression'] = gene_means
mask = adata.var['mean_expression'] > 0.1
adata_filtered = adata[:, mask].copy()
```

### 3. Filter by Number of Cells Expressing Gene

```python
# Calculate number of cells expressing each gene
n_cells = (adata.X > 0).sum(axis=0).A1 if issparse(adata.X) else (adata.X > 0).sum(axis=0)
adata.var['n_cells'] = n_cells

# Filter genes expressed in at least N cells
mask = adata.var['n_cells'] >= 3
adata_filtered = adata[:, mask].copy()
```

### 4. Filter by Index Range

```python
# Filter by gene index range
adata_subset = adata[:, 0:100].copy()  # First 100 genes

# Filter by specific indices
gene_indices = [0, 5, 10, 15, 20]
adata_subset = adata[:, gene_indices].copy()
```

---

## Combined Filtering

### Filter Both Cells and Genes

```python
# Filter cells by condition
cell_mask = adata.obs['Path..Group.'] == 'Control'

# Filter genes by expression
gene_mask = adata.var_names.str.startswith('MT-') == False  # Exclude mitochondrial

# Apply both filters
adata_filtered = adata[cell_mask, gene_mask].copy()
```

### Sequential Filtering

```python
# Step 1: Filter cells
adata = adata[adata.obs['n_genes'] > 200].copy()

# Step 2: Filter genes
adata = adata[:, adata.var['n_cells'] >= 3].copy()

# Step 3: Additional filtering
adata = adata[adata.obs['percent.mito'] < 0.15].copy()
```

---

## Using Scanpy Filtering Functions

Scanpy provides convenient filtering functions:

```python
import scanpy as sc

# Filter cells with fewer than min_genes genes detected
sc.pp.filter_cells(adata, min_genes=200)

# Filter genes expressed in fewer than min_cells cells
sc.pp.filter_genes(adata, min_cells=3)

# Filter by total counts (UMIs)
sc.pp.filter_cells(adata, min_counts=1000)

# Filter genes by total counts across all cells
sc.pp.filter_genes(adata, min_counts=10)

# Combined filtering
sc.pp.filter_cells(adata, min_genes=200, min_counts=1000)
sc.pp.filter_genes(adata, min_cells=3, min_counts=10)
```

**Note**: Scanpy functions modify the AnnData object **in place** by default.

---

## Common Filtering Patterns

### Pattern 1: Quality Control Filtering

```python
# Calculate QC metrics
from scipy.sparse import issparse

adata.obs['n_genes'] = (adata.X > 0).sum(axis=1).A1 if issparse(adata.X) else (adata.X > 0).sum(axis=1)
adata.obs['total_counts'] = adata.X.sum(axis=1).A1 if issparse(adata.X) else adata.X.sum(axis=1)

# Calculate mitochondrial percentage
mito_genes = adata.var_names.str.startswith('MT-')
if mito_genes.sum() > 0:
    adata.obs['percent_mito'] = (
        adata[:, mito_genes].X.sum(axis=1).A1 / 
        adata.X.sum(axis=1).A1 * 100
    )

# Apply QC filters
print(f"Before filtering: {adata.n_obs:,} cells, {adata.n_vars:,} genes")

# Filter cells
cell_mask = (
    (adata.obs['n_genes'] > 200) &
    (adata.obs['total_counts'] > 1000) &
    (adata.obs['percent_mito'] < 15)
)
adata = adata[cell_mask].copy()

# Filter genes
gene_mask = (adata.X > 0).sum(axis=0).A1 >= 3
adata = adata[:, gene_mask].copy()

print(f"After filtering: {adata.n_obs:,} cells, {adata.n_vars:,} genes")
```

### Pattern 2: Filter by Experimental Groups

```python
# Filter to specific groups for comparison
condition_col = 'Path..Group.'
group1 = 'Control'
group2 = 'Disease'

mask = adata.obs[condition_col].isin([group1, group2])
adata_filtered = adata[mask].copy()

print(f"Filtered to {group1} and {group2}: {adata_filtered.n_obs:,} cells")
print(f"Group distribution:\n{adata_filtered.obs[condition_col].value_counts()}")
```

### Pattern 3: Filter by Metadata Thresholds

```python
# Filter by RIN (RNA Integrity Number)
if 'RIN' in adata.obs.columns:
    mask = adata.obs['RIN'] > 7.0  # Keep high-quality samples
    adata = adata[mask].copy()

# Filter by UMI counts from metadata
if 'Median.UMI.Counts.per.Cell' in adata.obs.columns:
    mask = adata.obs['Median.UMI.Counts.per.Cell'] > 200
    adata = adata[mask].copy()

# Filter by mitochondrial content from metadata
if 'percent.mito' in adata.obs.columns:
    mask = adata.obs['percent.mito'] < 0.15
    adata = adata[mask].copy()
```

### Pattern 4: Filter Highly Variable Genes

```python
# First, identify highly variable genes (if not already done)
if 'highly_variable' not in adata.var.columns:
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)

# Filter to only highly variable genes
adata_hvg = adata[:, adata.var['highly_variable']].copy()
```

### Pattern 5: Filter by Cell Type

```python
# Filter to specific cell types
if 'cell_type' in adata.obs.columns:
    cell_types = ['Astrocyte', 'Neuron']
    mask = adata.obs['cell_type'].isin(cell_types)
    adata_filtered = adata[mask].copy()
```

---

## Best Practices

1. **Always use `.copy()`** when you want an independent filtered object:
   ```python
   adata_filtered = adata[mask].copy()  # Good
   adata_filtered = adata[mask]         # Creates view (may cause issues)
   ```

2. **Check filter results** before proceeding:
   ```python
   print(f"Before: {adata.n_obs:,} cells")
   mask = adata.obs['n_genes'] > 200
   print(f"After: {mask.sum():,} cells ({mask.sum()/len(mask)*100:.1f}%)")
   adata = adata[mask].copy()
   ```

3. **Filter in logical order**:
   - First: Filter cells by QC metrics
   - Second: Filter genes by expression
   - Third: Filter by experimental conditions

4. **Preserve original data**:
   ```python
   # Store original before filtering
   adata_raw = adata.copy()
   
   # Apply filters
   adata = adata[adata.obs['n_genes'] > 200].copy()
   
   # Original data still available in adata_raw
   ```

5. **Handle missing values**:
   ```python
   # Check for NaN values before filtering
   if adata.obs['RIN'].isna().any():
       print(f"Warning: {adata.obs['RIN'].isna().sum()} cells have missing RIN values")
       # Option 1: Exclude missing values
       mask = adata.obs['RIN'].notna() & (adata.obs['RIN'] > 7)
       # Option 2: Fill missing values
       adata.obs['RIN'] = adata.obs['RIN'].fillna(adata.obs['RIN'].median())
   ```

---

## Example: Complete Filtering Workflow

```python
import anndata as ad
import scanpy as sc
import numpy as np
from scipy.sparse import issparse

# Load your AnnData object
# adata = ...

# Step 1: Calculate QC metrics
print("Calculating QC metrics...")
adata.obs['n_genes'] = (adata.X > 0).sum(axis=1).A1 if issparse(adata.X) else (adata.X > 0).sum(axis=1)
adata.obs['total_counts'] = adata.X.sum(axis=1).A1 if issparse(adata.X) else adata.X.sum(axis=1)

# Calculate mitochondrial percentage
mito_genes = adata.var_names.str.startswith('MT-')
if mito_genes.sum() > 0:
    adata.obs['percent_mito'] = (
        adata[:, mito_genes].X.sum(axis=1).A1 / 
        adata.X.sum(axis=1).A1 * 100
    )

# Step 2: Filter cells
print(f"\nBefore filtering: {adata.n_obs:,} cells, {adata.n_vars:,} genes")

# Combine multiple cell filters
cell_mask = (
    (adata.obs['n_genes'] > 200) &
    (adata.obs['total_counts'] > 1000) &
    (adata.obs['percent_mito'] < 15)
)

if 'RIN' in adata.obs.columns:
    cell_mask = cell_mask & (adata.obs['RIN'] > 7.0)

if 'Median.UMI.Counts.per.Cell' in adata.obs.columns:
    cell_mask = cell_mask & (adata.obs['Median.UMI.Counts.per.Cell'] > 200)

adata = adata[cell_mask].copy()
print(f"After cell filtering: {adata.n_obs:,} cells, {adata.n_vars:,} genes")

# Step 3: Filter genes
gene_mask = (adata.X > 0).sum(axis=0).A1 >= 3
adata = adata[:, gene_mask].copy()
print(f"After gene filtering: {adata.n_obs:,} cells, {adata.n_vars:,} genes")

# Step 4: Filter by experimental groups (if needed)
if 'Path..Group.' in adata.obs.columns:
    groups_to_keep = ['Control', 'Disease']
    group_mask = adata.obs['Path..Group.'].isin(groups_to_keep)
    adata = adata[group_mask].copy()
    print(f"After group filtering: {adata.n_obs:,} cells, {adata.n_vars:,} genes")
    print(f"Group distribution:\n{adata.obs['Path..Group.'].value_counts()}")

print("\nFiltering complete!")
```

---

## References

- **AnnData Documentation**: https://anndata.readthedocs.io/
- **Scanpy Documentation**: https://scanpy.readthedocs.io/
- **AnnData Guide** (in this repo): `anndata_guide.md`

