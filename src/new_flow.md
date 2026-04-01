## Goal

Compare models trained on:
- **Raw expression per cell**: `(n_cells, n_genes)` UMI-count–based features.
- **Gene-embedding per cell**: `(n_cells, n_genes, D)` features, where each gene has a D‑dim embedding (e.g. D=16), scaled by that gene’s expression in the cell.

Objective: understand whether gene embeddings add predictive power for cell category beyond expression alone.

---

## Data and representations

- **Input data**:
  - Single-cell expression matrix: `X_expr` of shape `(n_cells, n_genes)` with UMI counts.
  - Gene embeddings: `E_gene` of shape `(n_genes, D)` (e.g. D=16).

- **Embedding-based cell representation**:
  - Normalize expression (library-size normalization + log1p or similar).
  - Optionally reduce `E_gene` via PCA to D components.
  - For each cell \(i\) and gene \(j\):

    \[
    \text{CellEmbedding}[i, j, :] = f(X_{\text{expr}}[i, j]) \cdot E_{\text{gene}}[j, :]
    \]

    where \(f\) is a chosen transform (`linear`, `sqrt`, or `log1p`).

  - Result: `X_embed` of shape `(n_cells, n_genes, D)`.

- **Expression-only representation**:
  - Use the same normalized `X_expr` (after library-size norm + log1p / scaling).

---

## Model families

### 1. Expression-only models

**Input:** `(n_cells, n_genes)`.

Typical baselines:

- **Logistic regression / linear SVM**
  - Strong classical baselines for cell-type classification on normalized scRNA-seq.
  - Often preceded by PCA to ~50–100 components.

- **MLP (expression MLP)**
  - 1–3 hidden layers, ReLU, dropout, weight decay.
  - Optional PCA on `X_expr` before the network.

- **scVI / scANVI (optional)**
  - Specialised VAEs for scRNA-seq; scANVI adds supervised cell-type head.
  - Good to include as a “high-end” baseline if available.

### 2. Embedding-based models

**Input:** `(n_cells, n_genes, D)`.

Architectures:

- **Flatten + MLP**
  - Flatten each cell to `(n_genes * D)` and feed to an MLP.
  - Needs careful regularization due to high dimensionality.

- **Per-gene encoder + pooling**
  - Shared encoder \(g(\cdot)\) over each gene’s D‑dim embedding.
  - Pool across genes:

    \[
    h_i = \text{Pool}_j \, g(\text{CellEmbedding}[i, j, :])
    \]

    where Pool is mean, max, or attention.
  - Equivalent in spirit to the `GeneEncoderModel` pattern.

- **MIL-style model (if aggregating cells to patients)**
  - Treat each patient’s cells as a bag, with a cell encoder + attention pooling over cells.
  - Useful when labels are patient-level, not cell-level.

- **Transformer over genes (optional)**
  - Treat gene embeddings per cell as a sequence of tokens and use self-attention.
  - Higher capacity; requires stronger regularization and more data.

---

## Fair comparison protocol

To compare expression-only vs embedding-based models fairly:

1. **Use identical data splits**
   - Same train/validation/test (or the same CV folds) across both representations.
   - If using LOFO or k-fold CV, keep folds identical.

2. **Use comparable preprocessing**
   - Expression-only:
     - Library-size normalize → log1p → optional gene-wise scaling.
   - Embedding-based:
     - Use the *same* normalization transform on the expression factor \(f(\cdot)\) used to scale embeddings.
     - Apply similar scaling (e.g. StandardScaler) to flattened features if needed.

3. **Match training protocol**
   - Same number of epochs / early-stopping rule.
   - Same optimizer (e.g. Adam), batch size, and learning rate grid (appropriately tuned per model type).
   - Use multiple seeds (e.g. 5–10) to estimate variance.

4. **Control for capacity**
   - Embedding models have more features (`n_genes * D` vs `n_genes`).
   - Include expression-only baselines with:
     - PCA to similar effective dimensionality, or
     - MLPs with similar parameter counts.
   - Consider an ablation with **random embeddings** (same shape but random vectors) to test whether improvements come from embedding structure or just higher feature dimension.

5. **Evaluation metrics**
   - Primary: **weighted F1** (handles class imbalance).
   - Also report: accuracy, macro F1, confusion matrix.
   - For CV:
     - Report mean ± std F1 across folds and **pooled F1** (computed over all held-out cells across all folds).
   - For multi-seed:
     - Report mean ± std across seeds and folds, and optionally per-seed averages.

---

## Experimental grid

For each representation:

- **Expression-only**
  - Logistic regression + (no PCA / PCA-50 / PCA-100).
  - MLP with 1–3 hidden layers, varying width & dropout.
  - Optional scVI / scANVI (if desired).

- **Embedding-based**
  - Flatten + MLP.
  - GeneEncoder-style model (per-gene encoder + pooling).
  - Optional MIL model if labels are patient-level rather than cell-level.

### Summary table of model variants

| Model ID | Input shape per cell             | Representation           | Example architecture                                      | Notes                                                   |
|---------|-----------------------------------|--------------------------|-----------------------------------------------------------|---------------------------------------------------------|
| E-Base  | `(n_cells, n_genes)`             | Expression-only          | Logistic regression / linear SVM                          | Strong classical baseline                               |
| E-MLP   | `(n_cells, n_genes)` or PCA-k    | Expression-only          | 1–3 layer MLP with dropout, weight decay                 | Compare with embedding MLPs under similar capacity      |
| E-VAE   | `(n_cells, n_genes)`             | Expression-only (latent) | scVI / scANVI                                             | Optional, higher-complexity baseline                    |
| G-Flat  | `(n_cells, n_genes, D)` → flat   | Embedding-based          | Flatten to `(n_genes*D)` → MLP                            | Tests benefit of richer features without structure      |
| G-Pool  | `(n_cells, n_genes, D)`          | Embedding-based          | Per-gene encoder + mean/max/attention pooling over genes | Aligns with `GeneEncoderModel` style                    |
| G-MIL   | bags of `(cells, n_genes, D)`    | Embedding-based + MIL    | Cell encoder + attention over cells (bag-level output)   | For patient-level labels; can also be adapted to cells  |

Optional ablations:

- **Random embeddings**:
  - Replace learned gene embeddings with random vectors (same D).
  - If performance remains similar, the specific gene embedding content may not add much beyond expression.

- **Expression-only + random noise dims**:
  - Add random extra features to match `n_genes * D` dimensionality.
  - Tests whether gains are just from allowing more parameters.

---

## Interpreting results

- If embedding-based models **consistently outperform** expression-only baselines across:
  - multiple seeds,
  - multiple folds,
  - and with similar capacity / normalization,

  then gene embeddings are adding genuine structure (e.g. capturing gene–gene relationships, pathways, or shared function) useful for cell category prediction.

- If performance is similar, or random embeddings perform as well as learned embeddings:
  - Expression level alone may already contain almost all discriminative signal for this task.
  - The added dimensions may act mostly as generic capacity, not meaningful biology.

- For small datasets:
  - Focus on variance estimates and effect sizes (mean differences vs std).
  - Use paired tests (e.g. Wilcoxon signed-rank on per-fold F1 differences) to judge significance.