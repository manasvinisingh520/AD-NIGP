# Understanding NaN padj Values in DESeq2 Results

## Summary

**NaN padj values are EXPECTED and NORMAL** in DESeq2/pydeseq2 results. They occur for genes that are filtered out during the multiple testing correction step, typically because they have very low expression levels.

## Why padj is NaN

### 1. **Automatic Filtering by DESeq2**
DESeq2 automatically filters genes with very low expression (low `baseMean`) **before** applying the Benjamini-Hochberg (BH) multiple testing correction. These genes:
- Still get p-values calculated (from the Wald test)
- Have their `padj` set to `NaN` because they're excluded from the FDR correction
- This prevents false discoveries from genes with unreliable expression estimates

### 2. **Common Characteristics of NaN padj Genes**
Based on your data analysis:
- **All have baseMean < 1** (very low average normalized counts)
- Mean baseMean: ~0.12 (extremely low)
- Most have high p-values (> 0.7), meaning they're not significantly different anyway
- They represent ~27% of all genes (8,092 out of 29,810)

### 3. **This is NOT an Error**
This is **expected behavior** and indicates:
- DESeq2 is working correctly
- Low-expression genes are being appropriately filtered
- Only genes with sufficient expression are included in FDR correction

## How to Debug/Verify

### Step 1: Check the Pattern
```python
import pandas as pd

df = pd.read_csv('your_results.csv', index_col=0)

# Count NaN padj
nan_count = df['padj'].isna().sum()
total = len(df)
print(f"NaN padj: {nan_count} ({nan_count/total*100:.1f}%)")

# Check baseMean distribution for NaN padj genes
nan_padj = df[df['padj'].isna()]
print(f"\nNaN padj genes - baseMean stats:")
print(nan_padj['baseMean'].describe())

# All should have baseMean < 1
print(f"\nAll NaN padj have baseMean < 1: {(nan_padj['baseMean'] < 1).all()}")
```

### Step 2: Verify It's Expected
If NaN padj genes have:
- ✅ baseMean < 1 (or very low)
- ✅ High p-values (> 0.5 typically)
- ✅ No extreme log2FoldChange values

Then this is **normal and expected**.

### Step 3: Check for Real Issues
**Red flags** (unusual patterns):
- ❌ Many genes with **high baseMean** (> 10) but NaN padj
- ❌ Many genes with **low p-values** (< 0.05) but NaN padj
- ❌ All genes have NaN padj (indicates a real problem)

## When to Worry

### Normal (No Action Needed):
- 20-40% of genes have NaN padj
- All NaN padj genes have low baseMean (< 1)
- Most NaN padj genes have high p-values

### Concerning (Investigate):
- > 50% of genes have NaN padj
- Many high-expression genes (baseMean > 10) have NaN padj
- Many low p-value genes (< 0.05) have NaN padj
- **All** genes have NaN padj (indicates a bug or data issue)

## Common Causes of Real Issues

### 1. **Insufficient Replicates**
- Need at least 3-5 samples per condition
- With pseudobulk: need multiple bins per condition
- **Check**: Count samples per group

### 2. **All Genes Filtered Out**
- If min_counts threshold is too high
- **Check**: Look at baseMean distribution

### 3. **Convergence Issues**
- DESeq2 model didn't converge for some genes
- **Check**: Look for warnings in DESeq2 output

### 4. **Design Matrix Issues**
- Problem with condition variable
- **Check**: Verify condition column has expected values

## How to Handle NaN padj in Analysis

### Option 1: Filter Out (Recommended)
```python
# Only use genes with valid padj
results_valid = results[results['padj'].notna()]

# Then filter by significance
significant = results_valid[
    (results_valid['padj'] < 0.05) & 
    (abs(results_valid['log2FoldChange']) > 0.5)
]
```

### Option 2: Use p-value as Fallback (Not Recommended)
```python
# Only for exploratory analysis - padj is preferred
# Filter NaN padj, then use pvalue for remaining
results_filtered = results[results['padj'].notna() | (results['pvalue'] < 0.05)]
```

### Option 3: Report Both
```python
# Report statistics for both
print(f"Genes with valid padj: {results['padj'].notna().sum()}")
print(f"Significant (padj < 0.05): {(results['padj'] < 0.05).sum()}")
print(f"Significant (pvalue < 0.05, padj=NaN): {((results['pvalue'] < 0.05) & results['padj'].isna()).sum()}")
```

## Your Current Situation

Based on your results:
- ✅ **8,092 genes (27%) have NaN padj** - This is normal
- ✅ **All have baseMean < 1** - Expected for filtered genes
- ✅ **21,718 genes have valid padj** - Good, most genes are included
- ✅ **8,868 genes are significant (padj < 0.05)** - Good number of findings

**Conclusion**: Your results are normal. The NaN padj values are expected for low-expression genes.

## References

- DESeq2 paper: Love et al. (2014) "Moderated estimation of fold change and dispersion for RNA-seq data"
- DESeq2 vignette: https://bioconductor.org/packages/release/bioc/vignettes/DESeq2/inst/doc/DESeq2.html
- Benjamini-Hochberg procedure: Controls false discovery rate by adjusting p-values

