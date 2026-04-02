# GLM (Generalized Linear Model) in DESeq2 Context

## What is GLM?

**GLM** stands for **Generalized Linear Model**. It's a statistical framework that extends ordinary linear regression to handle different types of data distributions and relationships.

### Basic Concept

A GLM has three components:

1. **Random Component**: The probability distribution of the response variable
2. **Systematic Component**: The linear predictor (combination of explanatory variables)
3. **Link Function**: Connects the mean of the response to the linear predictor

## GLM vs Ordinary Linear Regression

### Ordinary Linear Regression

```
Y = β₀ + β₁X₁ + β₂X₂ + ... + ε
```

**Assumptions:**
- Response (Y) is **continuous** and **normally distributed**
- Constant variance (homoscedasticity)
- Linear relationship

**Use case**: Continuous, normally distributed data

### Generalized Linear Model

```
g(μ) = β₀ + β₁X₁ + β₂X₂ + ...
```

**Key differences:**
- Response can follow **any distribution** from exponential family (Poisson, binomial, negative binomial, etc.)
- **Link function** g() connects mean (μ) to linear predictor
- Variance can depend on the mean

**Use case**: Count data, binary data, non-normal data

## GLM in DESeq2: Why It's Needed

### The Problem with RNA-seq Count Data

RNA-seq data has **counts** (non-negative integers) that are:
- ❌ Not normally distributed
- ❌ Not continuous
- ❌ Have variance that depends on the mean
- ❌ Contain many zeros (sparse)

**Ordinary linear regression doesn't work here!**

### The Solution: Negative Binomial GLM

DESeq2 uses a **Negative Binomial GLM** because:

1. **Negative Binomial distribution** models count data well
2. **Overdispersion**: Handles variance > mean (common in RNA-seq)
3. **Zero inflation**: Can model many zeros
4. **GLM framework**: Allows modeling multiple factors simultaneously

## DESeq2 GLM Structure

### The Model for Each Gene

For gene *i* in sample *j*:

```
log(μᵢⱼ) = βᵢ₀ + βᵢ₁ × conditionⱼ + βᵢ₂ × batchⱼ + ...
```

Or more generally:

```
g(μᵢⱼ) = Xⱼᵀ βᵢ
```

Where:
- **μᵢⱼ** = Expected count for gene *i* in sample *j*
- **g()** = Link function (log link in DESeq2)
- **Xⱼ** = Design matrix (covariates for sample *j*)
- **βᵢ** = Coefficients for gene *i* (what we want to estimate)

### Components Breakdown

#### 1. Random Component (Distribution)

**Negative Binomial** distribution:
```
Countᵢⱼ ~ NB(mean = μᵢⱼ, dispersion = αᵢ)
```

Properties:
- Handles overdispersion (variance > mean)
- Better fit than Poisson for RNA-seq
- Gene-specific dispersion (αᵢ)

#### 2. Systematic Component (Linear Predictor)

```
ηᵢⱼ = βᵢ₀ + βᵢ₁ × conditionⱼ + βᵢ₂ × batchⱼ
```

This is a **linear combination** of predictors:
- βᵢ₀ = Intercept (baseline expression)
- βᵢ₁ = Effect of condition (log2 fold change)
- βᵢ₂ = Effect of batch (batch correction)

#### 3. Link Function

**Log link** in DESeq2:
```
log(μᵢⱼ) = ηᵢⱼ
```

Or equivalently:
```
μᵢⱼ = exp(ηᵢⱼ)
```

**Why log link?**
- Ensures μ > 0 (counts are always positive)
- Makes fold changes additive (log scale)
- Standard for count data models

## Example: Simple Two-Group Comparison

### Your Design Formula: `~condition`

For a simple comparison with `condition` (A vs B):

```
log(μᵢⱼ) = βᵢ₀ + βᵢ₁ × I(conditionⱼ == "B")
```

Where:
- **βᵢ₀** = Intercept = log(mean expression in condition A)
- **βᵢ₁** = Log2 fold change (condition B vs A)
- **I(conditionⱼ == "B")** = Indicator variable (1 if B, 0 if A)

### Interpretation

**For samples in condition A:**
```
log(μᵢⱼ) = βᵢ₀
μᵢⱼ = exp(βᵢ₀)
```

**For samples in condition B:**
```
log(μᵢⱼ) = βᵢ₀ + βᵢ₁
μᵢⱼ = exp(βᵢ₀ + βᵢ₁) = exp(βᵢ₀) × exp(βᵢ₁)
```

**Fold change (B vs A):**
```
FC = μ_B / μ_A = exp(βᵢ₁)
log2(FC) = βᵢ₁ / log(2) ≈ βᵢ₁ / 0.693 ≈ 1.44 × βᵢ₁
```

In DESeq2, βᵢ₁ is actually in **log2 scale**, so:
```
log2(FC) = βᵢ₁
FC = 2^(βᵢ₁)
```

## More Complex Designs

### Design with Batch: `~batch + condition`

```
log(μᵢⱼ) = βᵢ₀ + βᵢ₁ × batch2ⱼ + βᵢ₂ × conditionBⱼ
```

Where:
- **βᵢ₀** = Baseline (batch1, conditionA)
- **βᵢ₁** = Batch effect
- **βᵢ₂** = Condition effect (adjusted for batch)

### Design with Interaction: `~condition * treatment`

```
log(μᵢⱼ) = βᵢ₀ + βᵢ₁ × conditionBⱼ + βᵢ₂ × treatmentⱼ + βᵢ₃ × (conditionBⱼ × treatmentⱼ)
```

Where:
- **βᵢ₁** = Main effect of condition
- **βᵢ₂** = Main effect of treatment
- **βᵢ₃** = Interaction effect (does treatment effect differ by condition?)

## GLM Estimation in DESeq2

### Maximum Likelihood Estimation (MLE)

DESeq2 uses **maximum likelihood** to estimate parameters:

1. **For each gene**, maximize the likelihood function
2. Find βᵢ values that make observed counts most probable
3. Account for:
   - Size factor normalization
   - Gene-specific dispersion
   - Design matrix

### Size Factor Normalization

Before fitting GLM, DESeq2 normalizes counts:

```
Normalized count = Raw count / Size factor
```

Size factors account for:
- Library size (sequencing depth)
- RNA composition differences

### Dispersion Estimation

For each gene, DESeq2 estimates:
- **Dispersion parameter** (αᵢ): Controls variance
- Methods:
  - Maximum likelihood per gene
  - Shrinkage toward mean dispersion
  - Outlier detection and replacement

### Wald Test for Coefficients

After estimation, test if βᵢ₁ ≠ 0:

```
W = β̂ᵢ₁ / SE(β̂ᵢ₁) ~ N(0,1)
```

This gives p-value for differential expression.

## Why GLM vs Other Methods?

### GLM vs Simple Ratios

**Simple ratio**: FC = mean(condition B) / mean(condition A)

**Problems:**
- ❌ Doesn't account for variance
- ❌ Can't handle multiple factors
- ❌ Doesn't model count distribution
- ❌ No statistical testing

**GLM advantages:**
- ✅ Accounts for variance properly
- ✅ Can model multiple factors simultaneously
- ✅ Models count distribution correctly
- ✅ Provides statistical tests

### GLM vs t-test

**t-test**: Compares means between groups

**Problems for RNA-seq:**
- ❌ Assumes normal distribution (counts aren't normal)
- ❌ Assumes equal variance (RNA-seq variance depends on mean)
- ❌ Can't handle multiple factors easily

**GLM advantages:**
- ✅ Uses appropriate distribution (negative binomial)
- ✅ Models mean-variance relationship
- ✅ Handles complex designs

### GLM vs Voom (limma)

**Voom**: Transforms counts, then uses linear models

**GLM advantages:**
- ✅ Works directly with counts (no transformation)
- ✅ Models count distribution explicitly
- ✅ More appropriate for count data

## Visual Understanding

### Linear vs Log-Link

**Ordinary regression:**
```
Y = β₀ + β₁X
```
- Direct relationship
- Can go negative
- Constant variance

**GLM with log link:**
```
log(μ) = β₀ + β₁X
μ = exp(β₀ + β₁X)
```
- Exponential relationship
- Always positive (good for counts!)
- Variance increases with mean

### Mean-Variance Relationship

**Poisson**: Variance = Mean
**Negative Binomial**: Variance = Mean + α × Mean²

RNA-seq data shows:
```
Variance > Mean (overdispersion)
```

This is why **Negative Binomial GLM** is used, not Poisson!

## Practical Example

### Your DGE Workflow

```python
from pydeseq2.dds import DeseqDataSet

# Create DeseqDataSet
dds = DeseqDataSet(
    counts=counts_df,        # Raw counts matrix
    metadata=metadata_df,     # Sample metadata
    design="~condition"      # GLM design formula
)

# This internally fits:
# log(μᵢⱼ) = βᵢ₀ + βᵢ₁ × conditionⱼ
# for each gene i and sample j

# Run DESeq2 (fits GLM for each gene)
dds.deseq2()

# Extract results
results = stat_res.results_df

# results['log2FoldChange'] = β̂ᵢ₁ (estimated coefficient)
# results['pvalue'] = Wald test p-value for H₀: βᵢ₁ = 0
```

## Key Takeaways

### What is GLM in DESeq2?

1. **Generalized Linear Model**: Extends regression to count data
2. **Negative Binomial**: Distribution for RNA-seq counts
3. **Log link**: Connects linear predictor to mean counts
4. **Design matrix**: Models experimental factors

### Why GLM?

- ✅ Handles count data properly (not normal)
- ✅ Models overdispersion (variance > mean)
- ✅ Allows complex experimental designs
- ✅ Provides statistical tests (Wald test)

### The Model

For gene *i*, sample *j*:

```
log(μᵢⱼ) = βᵢ₀ + βᵢ₁ × conditionⱼ + ...
Countᵢⱼ ~ NB(mean = μᵢⱼ, dispersion = αᵢ)
```

Where:
- **μᵢⱼ** = Expected count
- **βᵢ₁** = Log2 fold change (what we test)
- **αᵢ** = Dispersion (controls variance)

### In Your Results

When you get DGE results:
- **`log2FoldChange`** = β̂ᵢ₁ (coefficient from GLM)
- **`baseMean`** = Average normalized μᵢⱼ across samples
- **`pvalue`** = Test if βᵢ₁ ≠ 0 (Wald test on GLM coefficient)

## Summary

**GLM in DESeq2** is a statistical model that:
1. Uses **Negative Binomial distribution** for counts
2. Uses **log link function** to model mean counts
3. Fits a **linear predictor** based on experimental design
4. Estimates **coefficients** (log2 fold changes) for each gene
5. Tests significance using **Wald test**

This framework is what makes DESeq2 powerful for analyzing RNA-seq count data!

