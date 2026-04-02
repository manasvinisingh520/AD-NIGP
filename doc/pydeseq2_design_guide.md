# PyDESeq2 Design Formula Guide

## What is the Design Formula?

The **design formula** in PyDESeq2 (and DESeq2) specifies the statistical model used to analyze your data. It uses **Wilkinson notation** (also called "formula notation") borrowed from R.

## Understanding `~condition`

### The Tilde (`~`) Symbol

The `~` symbol means **"is modeled as a function of"** or **"depends on"**.

```
~condition
```

This reads as: **"Gene expression is modeled as a function of condition"**

### What it Does

When you specify `design="~condition"`, you're telling PyDESeq2:

1. **Response variable**: Gene expression counts (automatically used)
2. **Predictor variable**: The `condition` column from your metadata DataFrame
3. **Model**: Gene expression ~ condition

This creates a statistical model that estimates how gene expression differs between different levels of `condition`.

### Example Metadata

```python
import pandas as pd

# Your metadata DataFrame must have a 'condition' column
metadata = pd.DataFrame({
    'sample_id': ['S1', 'S2', 'S3', 'S4', 'S5', 'S6'],
    'condition': ['A', 'A', 'A', 'B', 'B', 'B'],  # Must match design formula
    'batch': ['1', '1', '2', '1', '2', '2']
})

print(metadata)
#   sample_id condition batch
# 0        S1         A     1
# 1        S2         A     1
# 2        S3         A     2
# 3        S4         B     1
# 4        S5         B     2
# 5        S6         B     2

# Now you can use: design="~condition"
```

## Simple Design Formulas

### 1. Single Factor: `~condition`

```python
design = "~condition"
```

**Meaning**: Compare expression between different conditions (e.g., treated vs control)

**Model**: `expression = β₀ + β₁ × condition_B`

- β₀: Intercept (baseline expression for condition A)
- β₁: Effect of condition B (log2 fold change B vs A)

**Use case**: Two-group comparison (control vs treated, disease vs healthy, etc.)

### 2. Multiple Factors: `~batch + condition`

```python
design = "~batch + condition"
```

**Meaning**: Model expression as a function of BOTH batch and condition, controlling for batch effects

**Model**: `expression = β₀ + β₁ × batch_2 + β₂ × condition_B`

- β₀: Intercept (baseline for batch_1, condition_A)
- β₁: Effect of batch_2 (batch correction term)
- β₂: Effect of condition B (log2 fold change, adjusted for batch)

**Use case**: When you have technical batches (different sequencing runs, days, etc.) that might confound your results

### 3. Interaction: `~condition + treatment + condition:treatment`

```python
design = "~condition + treatment + condition:treatment"
# or equivalently:
design = "~condition * treatment"  # * creates main effects + interaction
```

**Meaning**: Model main effects of condition and treatment, PLUS their interaction

**Model**: `expression = β₀ + β₁×condition_B + β₂×treatment_yes + β₃×(condition_B × treatment_yes)`

**Use case**: When you want to test if the effect of treatment differs between conditions

## Design Formula Syntax

### Basic Syntax Rules

1. **Tilde (`~`)**: Separates response (left) from predictors (right)
   - In DESeq2, response is always counts (implicit), so you only write `~predictors`

2. **Plus (`+`)**: Adds terms (main effects)
   - `~A + B` means both A and B are predictors

3. **Colon (`:`)**: Interaction term
   - `~A:B` means interaction between A and B

4. **Asterisk (`*`)**: Shorthand for main effects + interaction
   - `~A * B` = `~A + B + A:B`

5. **Zero (`0`)**: No intercept (rarely used)
   - `~0 + condition` (usually not recommended)

6. **Parentheses**: Grouping
   - `~(A + B):C` means interactions of C with both A and B

### Formula Examples

```python
# Simple two-group comparison
design = "~condition"

# Control for batch effects
design = "~batch + condition"

# Multiple covariates
design = "~age + sex + condition"

# Interaction model (does effect differ by group?)
design = "~condition * treatment"

# Complex model with multiple factors
design = "~batch + condition + age + sex + condition:age"

# Grouping with parentheses
design = "~(condition + batch):treatment"
```

## Reference Levels

### How Reference Levels Work

When you have a categorical variable like `condition` with levels `['A', 'B', 'C']`, DESeq2 needs to choose a **reference level** (baseline). 

- **Default**: Alphabetically first level (usually)
- **You set it**: Specify in the `contrast` when calling `DeseqStats`

### Example

```python
metadata = pd.DataFrame({
    'condition': ['control', 'treated', 'control', 'treated']
})

# If design = "~condition"
# Default reference might be 'control' (alphabetical)
# Then coefficients represent: log2(treated / control)

# When extracting results:
stat_res = DeseqStats(
    dds,
    contrast=["condition", "treated", "control"],  # treated vs control
    ...
)
```

## Common Design Patterns

### Pattern 1: Simple Comparison
```python
# Metadata
metadata = pd.DataFrame({
    'sample': ['S1', 'S2', 'S3', 'S4'],
    'group': ['control', 'control', 'treated', 'treated']
})

# Design
design = "~group"

# Compare
contrast = ["group", "treated", "control"]
```

### Pattern 2: Batch Correction
```python
# Metadata
metadata = pd.DataFrame({
    'sample': ['S1', 'S2', 'S3', 'S4'],
    'condition': ['A', 'A', 'B', 'B'],
    'batch': ['1', '2', '1', '2']
})

# Design (batch first, then condition)
design = "~batch + condition"

# Compare (condition effect, controlling for batch)
contrast = ["condition", "B", "A"]
```

### Pattern 3: Paired Design
```python
# Metadata (paired samples)
metadata = pd.DataFrame({
    'sample': ['S1', 'S2', 'S3', 'S4'],
    'patient': ['P1', 'P1', 'P2', 'P2'],
    'condition': ['before', 'after', 'before', 'after']
})

# Design (control for patient, then condition)
design = "~patient + condition"

# Compare
contrast = ["condition", "after", "before"]
```

### Pattern 4: Multiple Conditions (3+ Groups)
```python
# Metadata
metadata = pd.DataFrame({
    'sample': ['S1', 'S2', 'S3', 'S4', 'S5', 'S6'],
    'condition': ['control', 'low_dose', 'high_dose'],  # 3+ levels
    # or severity: ['1', '2', '3', '4']  # 4+ levels
})

# Design (works with 2, 3, 4, or any number of levels!)
design = "~condition"

# Multiple pairwise comparisons
# control vs low_dose
contrast1 = ["condition", "low_dose", "control"]

# control vs high_dose  
contrast2 = ["condition", "high_dose", "control"]

# low_dose vs high_dose
contrast3 = ["condition", "high_dose", "low_dose"]

# For severity 1-4, you could compare:
# 4 vs 1 (most vs least)
# 3 vs 1, 2 vs 1
# 4 vs 2, 4 vs 3
# Any pairwise comparison is possible!
```

**Note**: DESeq2 handles **any number of categorical levels**, not just binary comparisons. The design formula `~condition` works identically whether you have 2, 3, 4, or more groups. You specify which groups to compare in the `contrast` parameter.

### Pattern 5: Multi-Level Categorical (Ordinal/Non-Ordinal)
```python
# Metadata with multiple severity levels (1-4)
metadata = pd.DataFrame({
    'sample': ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8'],
    'severity': ['1', '1', '2', '2', '3', '3', '4', '4'],  # 4 levels as strings
    # or numeric (will be treated as categorical if design uses it as factor):
    # 'severity': [1, 1, 2, 2, 3, 3, 4, 4]
})

# Design (same as binary - works with any number of levels!)
design = "~severity"

# Compare any two levels using contrasts
# Severity 4 vs 1 (most severe vs least)
contrast1 = ["severity", "4", "1"]

# Adjacent levels: 2 vs 1
contrast2 = ["severity", "2", "1"]

# High vs Low: 4 vs 2
contrast3 = ["severity", "4", "2"]

# All possible pairwise comparisons can be done
```

**Notes**:
- DESeq2 supports **any number of levels** (2, 3, 4, 5+)
- You can compare **any two levels** using contrasts
- Ensure sufficient samples per level (≥3-5 recommended)
- For multiple comparisons, adjust p-values (DESeq2 does FDR correction per contrast)

### Pattern 6: Continuous Ordinal (Linear Trend)
```python
# Metadata with numeric severity (testing linear trend)
metadata = pd.DataFrame({
    'sample': ['S1', 'S2', 'S3', 'S4'],
    'severity': [1, 2, 3, 4],  # Numeric values (continuous)
})

# Design with continuous variable
design = "~severity"  # Tests linear relationship

# Model: expression = β₀ + β₁ × severity
# β₁ = log2 fold change per unit increase in severity
# Positive β₁ = increases with severity
# Negative β₁ = decreases with severity
```

**When to use**:
- **Categorical (Pattern 5)**: When levels are distinct categories, want flexible pairwise comparisons
- **Continuous (Pattern 6)**: When you believe there's a linear trend, want simpler model

**Example with disease severity 1-4**:
- If severity levels are **distinct disease stages** → Use **categorical** (Pattern 5)
- If severity represents **progressive worsening** and you expect linear changes → Use **continuous** (Pattern 6)

## Important Notes

### 1. Column Names Must Match

```python
# ✅ CORRECT: metadata has 'condition' column
metadata = pd.DataFrame({'condition': ['A', 'B']})
design = "~condition"

# ❌ WRONG: metadata has 'group' column but design says 'condition'
metadata = pd.DataFrame({'group': ['A', 'B']})
design = "~condition"  # Error: 'condition' not found!
```

### 2. Categorical vs Numeric Variables

```python
# Categorical (factors) - treated as groups
metadata['condition'] = ['A', 'B', 'A', 'B']  # Strings or categories
design = "~condition"

# Numeric - treated as continuous
metadata['dose'] = [0, 1, 2, 3]  # Numbers
design = "~dose"  # Linear relationship: expression ~ dose
```

### 3. Order Matters (for interactions)

```python
# These are different!
design1 = "~condition + batch"      # condition effect, controlling for batch
design2 = "~batch + condition"      # Same mathematically, but different interpretation

# Usually put the main factor of interest last
design = "~batch + condition"  # Main interest: condition
```

### 4. Avoid Overparameterization

```python
# ❌ BAD: Too many terms for small sample size
design = "~batch + condition + age + sex + treatment + batch:condition"

# ✅ GOOD: Simpler model
design = "~batch + condition"
```

## Your Code Example

Looking at your code:

```python
# Your metadata should have a 'condition' column
metadata = pd.DataFrame({
    'sample_id': [...],
    'condition': ['A', 'A', 'B', 'B', ...]  # Must match design!
})

# Design formula
design = "~condition"

# This creates a model:
# log(expression) = β₀ + β₁ × (condition == 'B')
# Where:
#   β₀ = baseline (condition A)
#   β₁ = log2 fold change (B vs A)

dds = DeseqDataSet(
    counts=counts_df,
    metadata=metadata,
    design=design,  # Uses 'condition' column
    ...
)

# Extract results: B vs A
stat_res = DeseqStats(
    dds,
    contrast=["condition", "B", "A"],  # numerator vs denominator
    ...
)
```

## Formula Cheat Sheet

| Formula | Meaning |
|---------|---------|
| `~condition` | Simple two-group comparison (or multi-level: 3+ groups) |
| `~batch + condition` | Control for batch effects |
| `~condition * treatment` | Main effects + interaction |
| `~condition + treatment + condition:treatment` | Same as above (explicit) |
| `~age + condition` | Control for continuous covariate |
| `~patient + condition` | Paired design (control for patient) |
| `~severity` | Multi-level categorical (3+ groups) or continuous ordinal |
| `~(A + B):C` | Interactions of C with A and B |
| `~0 + condition` | No intercept (rarely used) |

## Summary

- **`~condition`** means: "Gene expression depends on condition"
- The **tilde (`~`)** separates response from predictors
- **Column names** in design must match metadata column names
- **Reference level** is set in the contrast, not the design
- **Add terms with `+`**, interactions with `:`
- **Order matters** when you have multiple terms
- **Multi-level support**: Condition can have 2, 3, 4, or any number of levels (not just binary)
  - Compare any two levels using contrasts
  - For ordinal data, can treat as categorical (flexible) or continuous (linear trend)

For your use case:
- **Binary**: `design="~condition"` compares two groups (e.g., control vs treated)
- **Multi-level**: `design="~condition"` with 3+ groups allows comparing any pair via contrasts
- **Continuous**: `design="~severity"` with numeric values tests linear trend

