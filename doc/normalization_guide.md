# Normalization Methods: UMI Raw Counts, TPM, and CPM

## Overview

In single-cell RNA sequencing analysis, gene expression data can be represented in different units. Understanding these units is crucial for proper analysis and interpretation.

## 1. UMI Raw Counts

### What are UMI Raw Counts?
- **Definition**: The actual number of unique mRNA molecules (counted via UMIs) detected for each gene in each cell
- **Units**: Integer counts (0, 1, 2, 3, ...)
- **Range**: Typically 0-100+ per gene per cell (most genes are 0-10)
- **Scale**: Not normalized across cells or samples

### Characteristics:
- ✅ Preserves the true molecular count
- ✅ Suitable for most statistical models (negative binomial, Poisson)
- ✅ Used for differential expression analysis
- ✅ Used for QC metrics
- ❌ Cannot directly compare expression between cells with different sequencing depths
- ❌ Cannot directly compare expression between different samples/experiments

### Example:
```
Gene    Cell_A  Cell_B  Cell_C
GAPDH   150     300     75
ACTB    200     400     100
MYC     5       10      2
```

**Cell_A has 355 total UMIs, Cell_B has 710 total UMIs, Cell_C has 177 total UMIs**

## 2. CPM (Counts Per Million)

### What is CPM?
- **Definition**: Normalizes counts to account for sequencing depth (library size)
- **Formula**: `CPM = (count / total_counts_in_cell) × 1,000,000`
- **Units**: Counts per million (continuous values)
- **Range**: Typically 0-1000+ per gene
- **Purpose**: Makes expression comparable across cells with different sequencing depths

### Characteristics:
- ✅ Accounts for differences in sequencing depth (library size)
- ✅ Values are interpretable (e.g., "100 CPM" = 100 counts per million reads)
- ✅ Can compare expression between cells
- ❌ Does NOT account for gene length (unlike TPM)
- ❌ Still affected by highly expressed genes (composition bias)

### Calculation Example:

Using the same data:
```
Cell_A total: 355 UMIs
Cell_B total: 710 UMIs  
Cell_C total: 177 UMIs

GAPDH in Cell_A: (150 / 355) × 1,000,000 = 422,535 CPM
GAPDH in Cell_B: (300 / 710) × 1,000,000 = 422,535 CPM
GAPDH in Cell_C: (75 / 177) × 1,000,000 = 423,729 CPM

Now all cells show similar GAPDH expression! ✅
```

### In Python:
```python
import scanpy as sc

# Method 1: Using scanpy (standard in single-cell)
sc.pp.normalize_total(adata, target_sum=1e4)  # Normalizes to 10,000 (equivalent to CPM × 100)

# Method 2: Manual calculation
import numpy as np
from scipy.sparse import csr_matrix

def calculate_cpm(matrix):
    """
    Calculate CPM from raw count matrix.
    
    Parameters:
    -----------
    matrix : scipy.sparse or numpy array
        Count matrix (genes × cells)
    
    Returns:
    --------
    cpm_matrix : same format as input
        CPM normalized matrix
    """
    if hasattr(matrix, 'toarray'):  # Sparse matrix
        total_counts = matrix.sum(axis=0).A1  # Total UMIs per cell
    else:  # Dense matrix
        total_counts = matrix.sum(axis=0)
    
    # Normalize to per million
    cpm = matrix / total_counts * 1e6
    
    return cpm

# Example usage
cpm_matrix = calculate_cpm(raw_counts)
```

## 3. TPM (Transcripts Per Million)

### What is TPM?
- **Definition**: Normalizes for both sequencing depth AND gene length
- **Formula**: 
  1. RPK (Reads Per Kilobase) = count / (gene_length / 1000)
  2. RPK_total = sum(RPK) for all genes
  3. TPM = (RPK / RPK_total) × 1,000,000
- **Units**: Transcripts per million (continuous values)
- **Range**: Typically 1-10,000+ per gene
- **Purpose**: Best for comparing expression levels between different genes (accounts for gene length bias)

### Characteristics:
- ✅ Accounts for sequencing depth
- ✅ Accounts for gene length (longer genes get more reads by chance)
- ✅ Can compare expression levels between different genes
- ✅ Standard in bulk RNA-seq analysis
- ❌ Less commonly used in single-cell (gene length bias is smaller in scRNA-seq)
- ❌ Requires gene length information

### Calculation Example:

```
Gene    Count   Length  RPK         RPK_total  TPM
GAPDH   150     1000    150/1 = 150
ACTB    200     800     200/0.8 = 250
MYC     5       500     5/0.5 = 10
                        -------
                        Total = 410

GAPDH TPM = (150 / 410) × 1,000,000 = 365,854 TPM
ACTB TPM  = (250 / 410) × 1,000,000 = 609,756 TPM
MYC TPM   = (10 / 410) × 1,000,000 = 24,390 TPM

Total TPM always = 1,000,000 ✅
```

### In Python:
```python
def calculate_tpm(counts, gene_lengths):
    """
    Calculate TPM from raw counts.
    
    Parameters:
    -----------
    counts : array-like
        Gene expression counts (genes × cells)
    gene_lengths : array-like
        Gene lengths in base pairs (one per gene)
    
    Returns:
    --------
    tpm : array-like
        TPM normalized matrix
    """
    # Step 1: Calculate RPK (Reads Per Kilobase)
    rpk = counts / (gene_lengths / 1000)
    
    # Step 2: Calculate RPK total per cell
    rpk_total = rpk.sum(axis=0)
    
    # Step 3: Calculate TPM
    tpm = (rpk / rpk_total) * 1e6
    
    return tpm

# Example usage (requires gene lengths)
gene_lengths = np.array([1000, 800, 500])  # Lengths for GAPDH, ACTB, MYC
tpm_matrix = calculate_tpm(raw_counts, gene_lengths)
```

## Comparison Table

| Feature | UMI Raw Counts | CPM | TPM |
|---------|---------------|-----|-----|
| **Normalization** | None | Sequencing depth only | Sequencing depth + gene length |
| **Units** | Integer counts | Counts per million | Transcripts per million |
| **Typical Range** | 0-100+ | 0-1000+ | 1-10,000+ |
| **Gene length bias** | Present | Present | Accounted for |
| **Sequencing depth bias** | Present | Accounted for | Accounted for |
| **Used in scRNA-seq** | ✅ Primary | ✅ Common | ❌ Rare |
| **Used in bulk RNA-seq** | ❌ Rare | ✅ Common | ✅ Standard |
| **Differential expression** | ✅ Yes (with models) | ⚠️ Sometimes | ⚠️ Sometimes |
| **Visualization** | ❌ Limited | ✅ Good | ✅ Good |

## When to Use Each

### Use UMI Raw Counts For:
- ✅ **QC filtering** (detecting low-quality cells/genes)
- ✅ **Differential expression analysis** (using negative binomial models)
- ✅ **Downstream statistical modeling** (most scRNA-seq tools expect raw counts)
- ✅ **Cell type clustering** (before normalization)
- ✅ **Calculating QC metrics** (total UMIs, genes detected)

### Use CPM For:
- ✅ **Visualization** (heatmaps, violin plots)
- ✅ **Comparing expression between cells**
- ✅ **Exploratory data analysis**
- ✅ **Log transformation** (log(CPM+1) for visualization)
- ✅ **Single-cell analysis** (standard workflow)

### Use TPM For:
- ✅ **Bulk RNA-seq analysis** (standard)
- ✅ **Comparing expression between different genes**
- ✅ **When gene length differences are important**
- ⚠️ **Less common in single-cell** (gene length bias is minimal)

## Standard Single-Cell Workflow

```python
import scanpy as sc

# 1. Start with UMI raw counts
adata = sc.read_10x_mtx('data/')
print(adata.X[:5, :5])  # Raw counts

# 2. QC filtering (use raw counts)
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)

# 3. Normalize to CPM-like (10,000 UMIs per cell)
sc.pp.normalize_total(adata, target_sum=1e4)
print(adata.X[:5, :5])  # Now normalized (equivalent to CPM × 100)

# 4. Log transform for visualization
sc.pp.log1p(adata)
print(adata.X[:5, :5])  # Now log(CPM+1) values

# Note: Scanpy stores the normalized data in adata.X
# Raw counts are still available in adata.raw if saved
```

## Common Misconceptions

### ❌ "TPM is always better than CPM"
- **Reality**: In single-cell, CPM is often preferred. Gene length bias is minimal, and TPM requires gene length info that may be incomplete.

### ❌ "Normalized data should be used for differential expression"
- **Reality**: Most DE tools (DESeq2, edgeR, scran) expect raw counts and model the variance appropriately.

### ❌ "Higher CPM/TPM means higher absolute expression"
- **Reality**: Only within the same sample/cell. Between different cells, CPM accounts for library size but not other biases.

## Visual Comparison

### Raw Counts:
```
Gene    Cell_1  Cell_2  Cell_3
GAPDH   100     500     50    ← Looks like Cell_2 has much more, but it just has more total reads
Total   1000    5000    500
```

### CPM (normalized):
```
Gene    Cell_1    Cell_2    Cell_3
GAPDH   100,000   100,000   100,000  ← Now comparable! All cells show same relative expression
Total   1,000,000 1,000,000 1,000,000
```

### TPM (with gene length correction):
```
Gene    Length  Cell_1    Cell_2    Cell_3
GAPDH   1000    200,000   200,000   200,000  ← Accounts for gene length
ACTB    500     400,000   400,000   400,000  ← Shorter gene gets adjusted
Total          1,000,000 1,000,000 1,000,000
```

## Summary

- **UMI Raw Counts**: True molecular counts, use for statistical analysis
- **CPM**: Normalized for sequencing depth, use for single-cell visualization and comparisons
- **TPM**: Normalized for sequencing depth + gene length, standard in bulk RNA-seq

For single-cell analysis, the typical workflow is:
1. **Start with raw counts** → QC and filtering
2. **Normalize to CPM** (via `normalize_total`) → Visualization
3. **Log transform** → Downstream analysis

