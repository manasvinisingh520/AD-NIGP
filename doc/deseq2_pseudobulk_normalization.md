# DESeq2 Normalization and Pseudobulk: Why Summing Counts Works

## The Question

When creating pseudobulk samples by summing counts across cells in bins:
- **Bin A**: 100 cells → Summed counts = 100,000 total reads
- **Bin B**: 50 cells → Summed counts = 50,000 total reads

**Question**: Won't the difference in total counts (library size) create bias in the analysis?

**Answer**: No! DESeq2 handles this automatically through **size factor normalization**.

---

## How DESeq2 Normalization Works

### 1. Size Factor Calculation (Median-of-Ratios Method)

DESeq2 calculates a **size factor** for each sample (pseudobulk bin) that accounts for:
- **Library size** (total number of counts)
- **RNA composition** (proportional differences)

**Algorithm**:
1. For each gene, calculate the geometric mean across all samples
2. For each sample, divide each gene's count by the geometric mean
3. Take the median of these ratios → this is the size factor for that sample

**Example**:

```
Sample        Gene1  Gene2  Gene3  Gene4  Total    Size Factor
---------------------------------------------------------------
Bin A (100)   1000   500    200    300    2000     2.0
Bin B (50)    500    250    100    150    1000     1.0
Bin C (75)    750    375    150    225    1500     1.5

Geometric mean (across all samples):
Gene1: (1000 × 500 × 750)^(1/3) = 722
Gene2: (500 × 250 × 375)^(1/3) = 361
Gene3: (200 × 100 × 150)^(1/3) = 144
Gene4: (300 × 150 × 225)^(1/3) = 216

Ratios for Bin A:
Gene1: 1000 / 722 = 1.38
Gene2: 500 / 361 = 1.38
Gene3: 200 / 144 = 1.39
Gene4: 300 / 216 = 1.39
Median = 1.385 → Size Factor for Bin A ≈ 1.39

Ratios for Bin B:
Gene1: 500 / 722 = 0.69
Gene2: 250 / 361 = 0.69
Gene3: 100 / 144 = 0.69
Gene4: 150 / 216 = 0.69
Median = 0.69 → Size Factor for Bin B ≈ 0.69
```

### 2. Normalization

Once size factors are calculated, DESeq2 normalizes counts:

```
Normalized Count = Raw Count / Size Factor
```

**Example**:
- Bin A (100 cells): Raw = 1000, Size Factor = 1.39 → Normalized = 1000/1.39 = 719
- Bin B (50 cells): Raw = 500, Size Factor = 0.69 → Normalized = 500/0.69 = 725

**Notice**: After normalization, both bins have similar expression levels, despite having different numbers of cells!

### 3. Why This Works for Pseudobulk

When you sum counts from different numbers of cells:
- **Bin with 100 cells**: Total counts = 100 × (average counts per cell)
- **Bin with 50 cells**: Total counts = 50 × (average counts per cell)

The **proportional relationships between genes remain the same**:
- If Gene1 is expressed 2× more than Gene2 in each cell, it will be 2× more in the summed pseudobulk regardless of bin size
- Size factors capture this proportionality and normalize out the library size effect

---

## Mathematical Intuition

### Scenario: Two Bins, Different Cell Counts

**Bin 1** (100 cells):
```
Cell 1: [10, 5, 3, 2]  (Gene1, Gene2, Gene3, Gene4)
Cell 2: [12, 6, 4, 1]
...
Cell 100: [11, 5, 3, 2]
Sum: [1000, 500, 300, 200]  (Total: 2000)
```

**Bin 2** (50 cells):
```
Cell 1: [10, 5, 3, 2]
Cell 2: [12, 6, 4, 1]
...
Cell 50: [11, 5, 3, 2]
Sum: [500, 250, 150, 100]  (Total: 1000)
```

**Key observation**: The **proportions are identical**:
- Bin 1: Gene1/Gene2 = 1000/500 = 2.0
- Bin 2: Gene1/Gene2 = 500/250 = 2.0

DESeq2's size factor normalization:
1. Detects that Bin 1 has 2× the total counts of Bin 2
2. Assigns size factors: SF₁ = 2.0, SF₂ = 1.0
3. Normalizes: Bin 1 normalized = 1000/2.0 = 500 (same as Bin 2!)

---

## Why Averaging Would Be Wrong

**If you averaged instead of summed:**

**Bin 1** (100 cells): Average = [1000/100, 500/100, 300/100, 200/100] = [10, 5, 3, 2]
**Bin 2** (50 cells): Average = [500/50, 250/50, 150/50, 100/50] = [10, 5, 3, 2]

Averaging would make them identical, but this is **incorrect** because:
1. You lose information about sampling depth
2. DESeq2 expects **counts** (integer values), not normalized averages
3. The negative binomial model needs raw counts to properly estimate variance

---

## DESeq2's Negative Binomial Model

DESeq2 uses a **negative binomial GLM** that models:

```
E[count] = μ = Size Factor × Base Expression × Design Effects
```

The model accounts for:
1. **Library size** via size factors (handles different cell counts per bin)
2. **Biological differences** via design matrix (condition effects)
3. **Variance** via dispersion parameters (gene-specific variability)

**Important**: The negative binomial model expects **raw integer counts**, not normalized averages, because it models the count distribution directly.

---

## Practical Example from Your Code

In your `dge.py` (lines 746-752):

```python
# Get subset of cells for this bin
patient_cells_subset = patient_cells[start_idx:end_idx]

# Sum counts across cells in this bin
bin_counts = patient_cells_subset.X.sum(axis=0)

# Add to pseudobulk data
pseudobulk_data.append(bin_counts)
```

**What happens next**:
1. DESeq2 receives pseudobulk matrix with varying library sizes
2. DESeq2 calculates size factors for each bin (sample)
3. DESeq2 normalizes counts internally during model fitting
4. DESeq2 tests for differential expression using normalized values

**Result**: Different bin sizes don't matter because DESeq2 accounts for them automatically!

---

## Summary

| Aspect | Why It Works |
|--------|--------------|
| **Summing vs Averaging** | Summing preserves count distribution needed for negative binomial model |
| **Different bin sizes** | Size factors normalize out library size differences |
| **Statistical validity** | DESeq2's model accounts for variance in counts appropriately |
| **Biological interpretation** | Normalized counts represent proportional expression, not absolute counts |

**Key Takeaway**: As long as bins are created from the same cell type and have similar biological composition, summing counts is the correct approach. DESeq2 handles the rest through its normalization and modeling framework.

---

## References

- Love, M. I., Huber, W., & Anders, S. (2014). Moderated estimation of fold change and dispersion for RNA-seq data with DESeq2. Genome biology, 15(12), 1-21.
- DESeq2 documentation: https://bioconductor.org/packages/release/bioc/vignettes/DESeq2/inst/doc/DESeq2.html
- PyDESeq2: https://github.com/owkin/PyDESeq2

