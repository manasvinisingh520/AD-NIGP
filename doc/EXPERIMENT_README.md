# Gene Embedding vs Expression Experiment Guide

Step-by-step instructions for running the v2 experiment framework that compares
gene-embedding-based models against expression-only baselines.

---

## Prerequisites

```
pip install torch numpy scikit-learn pandas anndata tqdm
pip install optuna optuna-dashboard   # for hyperparameter search
pip install openai python-dotenv      # for LLM-based sampler (optional)
pip install scvi-tools                # for scVI model (optional)
```

Make sure the raw data files exist:

- `data/model_data/{region}_AnnData_perCell.h5ad` for each Astrocyte region
- `data/model_data/{region}_Microglia_AnnData_perCell.h5ad` for Microglia
- `data/model_data/MTG_AnnData_perCell.h5ad` for MTG
- `GREmLN/embeddings/{region}_gene_embeddings.tsv` for gene embeddings
- `data/genes_2_ensembl_ids.pkl` for gene symbol mapping

---

## New files overview

| File | Purpose |
|------|---------|
| `src/create_data_v2.py` | Generate combined pickles with both expression and embedding data per fold |
| `src/train_utils.py` | Shared training utilities (datasets, training loops, evaluation) |
| `src/train_model_v2.py` | Unified training script for all 7 model types across 3 input types |
| `src/autosearch.py` | Generalized hyperparameter search (Optuna + random + LLM samplers) |
| `run_autosearch_v2.ps1` | PowerShell launcher for all three samplers |

---

## Step 1: Generate data

Create v2 pickles that contain **both** expression-only `(n_cells, n_genes)` and
embedding-scaled `(n_cells, n_genes, D)` folds with identical patient splits.

```bash
# All Astrocyte regions (EC, ITG, PFC, V2, V1)
python src/create_data_v2.py --cell_type Astrocytes

# All Microglia regions (CrossRegion, ITG, PFC)
python src/create_data_v2.py --cell_type Microglia

# MTG
python src/create_data_v2.py --cell_type MTG

# Single region only (faster for testing)
python src/create_data_v2.py --cell_type Astrocytes --region EC
```

Output: `data/model_data/{region}_v2_data.pkl` (one file per region).

---

## Step 2: Run individual models

`train_model_v2.py` has three required arguments:

- `--region` : e.g. EC, ITG, PFC, V2, V1, CrossRegion, MTG
- `--input_type` : `expression`, `embedding`, or `random_embedding`
- `--model_type` : depends on input_type (see table below)

### Valid input_type / model_type combinations

| input_type | Allowed model_types |
|---|---|
| `expression` | `logreg`, `svm`, `mlp`, `scvi` |
| `embedding` | `flatten_mlp`, `gene_encoder`, `mil` |
| `random_embedding` | `flatten_mlp`, `gene_encoder`, `mil` |

### Expression-only models

```bash
# Logistic regression (sklearn, fast, no GPU needed)
python src/train_model_v2.py --region EC --input_type expression --model_type logreg \
    --n_seeds 10 --verbose-cv

# Linear SVM
python src/train_model_v2.py --region EC --input_type expression --model_type svm \
    --C 1.0 --pca_components 50 --n_seeds 10 --verbose-cv

# MLP on expression
python src/train_model_v2.py --region EC --input_type expression --model_type mlp \
    --mlp_hidden_sizes 128,64,32 --epochs 200 --n_seeds 10 --verbose-cv

# scVI (requires scvi-tools)
python src/train_model_v2.py --region EC --input_type expression --model_type scvi \
    --scvi_n_latent 30 --epochs 100 --n_seeds 5 --verbose-cv
```

### Embedding-based models

```bash
# Flatten + MLP (embedding flattened to n_genes*D)
python src/train_model_v2.py --region EC --input_type embedding --model_type flatten_mlp \
    --mlp_hidden_sizes 256,128,64 --epochs 200 --n_seeds 10 --verbose-cv

# Cell-level variant (no patient pooling; each cell is a sample)
python src/train_model_v2.py --region EC --input_type embedding --model_type flatten_mlp \
    --cell_level --mlp_hidden_sizes 256,128,64 --epochs 200 --n_seeds 10 --verbose-cv

# Gene encoder (per-gene attention pooling)
python src/train_model_v2.py --region EC --input_type embedding --model_type gene_encoder \
    --lambda_entropy 0.01 --epochs 200 --n_seeds 10 --verbose-cv

# MIL (cell bags per patient, attention aggregation)
python src/train_model_v2.py --region EC --input_type embedding --model_type mil \
    --mil_cells_per_patient 128 --mil_hidden_dim 128 --epochs 200 --n_seeds 10 --verbose-cv
```

### Random embedding ablation

Uses random gene vectors instead of learned embeddings (same shape).
This tests whether any improvement comes from embedding structure or just
higher dimensionality.

```bash
python src/train_model_v2.py --region EC --input_type random_embedding --model_type flatten_mlp \
    --mlp_hidden_sizes 256,128,64 --epochs 200 --n_seeds 10 --verbose-cv

# Cell-level random embedding variant
python src/train_model_v2.py --region EC --input_type random_embedding --model_type flatten_mlp \
    --cell_level --mlp_hidden_sizes 256,128,64 --epochs 200 --n_seeds 10 --verbose-cv
```

---

## Step 3: Hyperparameter search with autosearch

`autosearch.py` searches over model types and hyperparams automatically.

### Search within one input type

```bash
# Optuna search over expression-only models
python src/autosearch.py --region EC --cell_type Astrocytes \
    --input_type expression --sampler optuna --n_trials 50

# Optuna search over embedding models
python src/autosearch.py --region EC --cell_type Astrocytes \
    --input_type embedding --sampler optuna --n_trials 50
```

### Search across all input types

```bash
# Let Optuna explore expression, embedding, and random_embedding
python src/autosearch.py --region EC --cell_type Astrocytes \
    --sampler optuna --n_trials 100
```

### Use random sampler

```bash
python src/autosearch.py --region EC --cell_type Astrocytes \
    --sampler random --n_trials 100
```

### Use LLM sampler (needs OPENAI_API_KEY or --api_key)

```bash
python src/autosearch.py --region EC --cell_type Astrocytes \
    --sampler llm --n_trials 20 --model gpt-4o-mini --proposals_per_round 3
```

### Run all three samplers (PowerShell)

```powershell
.\run_autosearch_v2.ps1 -Region EC -CellType Astrocytes -NTrials 50

# Only search expression models
.\run_autosearch_v2.ps1 -Region EC -CellType Astrocytes -InputType expression

# Microglia
.\run_autosearch_v2.ps1 -Region CrossRegion -CellType Microglia -NTrials 30
```

### View results with Optuna dashboard

```bash
optuna-dashboard sqlite:///ranking_model_runs/autosearch_v2_EC_Astrocytes/EC_Astrocytes_optuna.db
```

---

## Step 4: Full experiment protocol

### Phase 1 -- Data generation (one-time)

```bash
python src/create_data_v2.py --cell_type Astrocytes
python src/create_data_v2.py --cell_type Microglia
python src/create_data_v2.py --cell_type MTG
```

### Phase 2 -- Find good hyperparams per model family (fast, 1 seed)

```bash
# For each region of interest:
python src/autosearch.py --region EC --cell_type Astrocytes --input_type expression --sampler optuna --n_trials 50
python src/autosearch.py --region EC --cell_type Astrocytes --input_type embedding  --sampler optuna --n_trials 50
```

Check `ranking_model_runs/autosearch_v2/<study_name>_best_params.json` for the winning configs.

### Phase 3 -- Run best configs with 10 seeds for statistical power

Take the best hyperparams from Phase 2 and run each model with 10 seeds:

```bash
# Expression baselines
python src/train_model_v2.py --region EC --input_type expression --model_type logreg  --n_seeds 10 --verbose-cv
python src/train_model_v2.py --region EC --input_type expression --model_type mlp     --n_seeds 10 --verbose-cv

# Embedding models
python src/train_model_v2.py --region EC --input_type embedding  --model_type flatten_mlp   --n_seeds 10 --verbose-cv
python src/train_model_v2.py --region EC --input_type embedding  --model_type gene_encoder  --n_seeds 10 --verbose-cv
python src/train_model_v2.py --region EC --input_type embedding  --model_type mil           --n_seeds 10 --verbose-cv

# Random embedding ablation
python src/train_model_v2.py --region EC --input_type random_embedding --model_type flatten_mlp --n_seeds 10 --verbose-cv
```

### Phase 4 -- Compare results

Each run prints a summary like:

```
Grand Summary: 10 seeds x 8 folds = 80 runs
  Generalized F1: 0.5432 +/- 0.0891
  Pooled held-out F1: 0.5510
```

Compare:

1. **Expression logreg/mlp F1** vs **Embedding flatten_mlp/gene_encoder F1**
   - If embedding models consistently have higher F1, gene embeddings help.

2. **Embedding F1** vs **Random embedding F1**
   - If random embeddings perform similarly to real embeddings, the gain is from
     higher dimensionality, not embedding structure.

3. Use paired Wilcoxon signed-rank test on per-fold F1 values for significance.

---

## Key arguments reference

### train_model_v2.py

| Argument | Default | Description |
|---|---|---|
| `--region` | (required) | EC, ITG, PFC, V2, V1, CrossRegion, MTG |
| `--cell_type` | Astrocytes | Astrocytes, Microglia, MTG |
| `--input_type` | (required) | expression, embedding, random_embedding |
| `--model_type` | (required) | logreg, svm, mlp, scvi, flatten_mlp, gene_encoder, mil |
| `--n_seeds` | 10 | Number of random seeds for multi-seed CV |
| `--n_folds_eval` | 8 | Number of CV folds (can merge to 4) |
| `--epochs` | 200 | Max training epochs (PyTorch models) |
| `--patience` | 40 | Early stopping patience |
| `--min_epochs` | 30 | Min epochs before early stopping |
| `--pca_components` | None | Optional PCA dimensionality reduction |
| `--C` | 1.0 | Regularization for logreg/svm |
| `--lr` | 0.001 | Learning rate (PyTorch models) |
| `--dropout` | 0.2 | Dropout rate |
| `--verbose-cv` | off | Print per-fold progress |

### autosearch.py

| Argument | Default | Description |
|---|---|---|
| `--region` | (required) | Target region |
| `--cell_type` | Astrocytes | Cell type |
| `--input_type` | None | Fix to one input type, or omit to search all |
| `--sampler` | optuna | optuna, random, or llm |
| `--n_trials` | 20 | Number of trials |
| `--study_name` | autosearch_v2 | Optuna study name |
| `--log_base` | ranking_model_runs/autosearch_v2 | Output directory |
| `--model` | gpt-4o-mini | LLM model for --sampler llm |
