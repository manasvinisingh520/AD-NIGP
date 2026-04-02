# MTX File Format Guide

## What is a .mtx file?

A `.mtx` file uses the **Matrix Market** format, a standard for storing sparse matrices. In single-cell RNA sequencing (scRNA-seq), it's commonly used to store gene expression count matrices because:

1. **Efficiency**: Most genes aren't expressed in most cells (sparse data)
2. **Standard format**: Widely supported format from 10X Genomics and other platforms
3. **Memory savings**: Only stores non-zero values

## MTX File Structure

The header of a `.mtx` file typically looks like:
```
%%MatrixMarket matrix coordinate integer general
<num_rows> <num_cols> <num_nonzero_entries>
<row_index> <col_index> <value>
<row_index> <col_index> <value>
...
```

**Important Note**: In MTX format, rows typically represent **genes** and columns represent **cells**, but in many scRNA-seq workflows, you need to **transpose** the matrix so that rows=genes and columns=cells.

## Reading MTX Files in Python

### Method 1: Using scipy (Simple)

```python
import scipy.io
from scipy.sparse import csr_matrix

# Read the MTX file
matrix = scipy.io.mmread('matrix.mtx')

# Transpose if needed (cells as rows, genes as columns)
matrix = matrix.T

# Convert to CSR format for efficient operations
matrix = csr_matrix(matrix)

print(f"Shape: {matrix.shape}")  # (genes, cells)
print(f"Non-zero entries: {matrix.nnz:,}")
```

### Method 2: With Gene and Cell Annotations

```python
import scipy.io
import pandas as pd

# Read matrix
matrix = scipy.io.mmread('matrix.mtx').T
matrix = csr_matrix(matrix)

# Read gene names (rows)
with open('genes.txt', 'r') as f:
    gene_names = [line.strip().strip('"') for line in f]

# Read cell barcodes (columns)
with open('barcodes.txt', 'r') as f:
    cell_names = [line.strip().strip('"') for line in f]

# Verify dimensions match
assert matrix.shape[0] == len(gene_names)
assert matrix.shape[1] == len(cell_names)
```

### Method 3: Using scanpy (Recommended for scRNA-seq)

```python
import scanpy as sc

# Scanpy can read 10X format directly
adata = sc.read_10x_mtx(
    'path/to/directory',
    var_names='gene_symbols',  # or 'gene_ids'
    cache=True
)

# Or read MTX with annotations manually
adata = sc.read_mtx('matrix.mtx')
adata = adata.T  # transpose

# Add gene and cell names
import pandas as pd
adata.var_names = pd.read_csv('genes.txt', header=None)[0]
adata.obs_names = pd.read_csv('barcodes.txt', header=None)[0]
```

## Common Operations

### Access specific genes or cells:
```python
# Get expression of first gene across all cells
gene_expression = matrix[0, :].toarray().flatten()

# Get expression profile of first cell
cell_expression = matrix[:, 0].toarray().flatten()

# Subset to specific genes
subset = matrix[[0, 5, 10], :]  # First, 6th, and 11th genes
```

### Convert to dense format (use with caution):
```python
# Convert entire matrix (WARNING: can be very memory intensive!)
dense_matrix = matrix.toarray()

# Better: convert only a subset
subset_dense = matrix[:100, :100].toarray()
```

### Convert to pandas DataFrame:
```python
import pandas as pd

# Only for small subsets!
df = pd.DataFrame(
    matrix[:100, :10].toarray(),
    index=gene_names[:100],
    columns=cell_names[:10]
)
```

## Your Specific Files

Based on your directory:
- **Matrix**: `2025-10-22_Astrocytes_EC_matrix.mtx` (~200MB+)
- **Genes/Rows**: `2025-10-22_Astrocytes_EC_row_annotation.txt` (36,601 genes)
- **Cells/Columns**: `2025-10-22_Astrocytes_EC_cell_annotation.txt` (101,489 cells)

**Matrix dimensions**: 36,601 genes × 101,489 cells

## Memory Considerations

With ~36K genes and ~101K cells:
- **Sparse format**: ~200MB (only stores non-zero values)
- **Dense format**: ~29GB (36,601 × 101,489 × 8 bytes = ~29.7 GB!)

**Recommendation**: Always work with sparse matrices unless you need to subset the data significantly.

## Quick Start Example

```python
import scipy.io
from scipy.sparse import csr_matrix

# Read MTX file
matrix = scipy.io.mmread('2025-10-22_Astrocytes_EC_matrix.mtx')
matrix = csr_matrix(matrix.T)  # Transpose: rows=genes, cols=cells

# Read annotations
with open('2025-10-22_Astrocytes_EC_row_annotation.txt') as f:
    genes = [line.strip().strip('"') for line in f]

with open('2025-10-22_Astrocytes_EC_cell_annotation.txt') as f:
    cells = [line.strip().strip('"') for line in f]

print(f"Loaded: {len(genes)} genes × {len(cells)} cells")
print(f"Matrix shape: {matrix.shape}")
```


Our data is UMI raw count

Quick summary
1. UMI Raw Counts
What: Actual number of unique mRNA molecules per gene per cell
Values: Integer counts (0, 1, 2, 3, ...)
Range: Typically 0-100+ per gene per cell
Use for: QC filtering, differential expression analysis, statistical modeling
2. CPM (Counts Per Million)
What: Normalized for sequencing depth (library size)
Formula: (count / total_counts_in_cell) × 1,000,000
Values: Continuous (0-1000+ per gene)
Use for: Single-cell visualization, comparing expression between cells
Note: Does NOT account for gene length
3. TPM (Transcripts Per Million)
What: Normalized for sequencing depth AND gene length
Formula: More complex (accounts for gene length in kb)
Values: Continuous (1-10,000+ per gene)
Use for: Bulk RNA-seq analysis, comparing different genes
Note: Less common in single-cell analysis