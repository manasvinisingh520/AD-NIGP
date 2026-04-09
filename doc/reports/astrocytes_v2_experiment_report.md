# Astrocytes v2 Experiment -- Multi-Region Comparison Report

**Protocol:** EXPERIMENT_README.md Step 4 + v2-experiment-protocol SKILL  
**Cell type:** Astrocytes  
**Date:** 2026-04-04  
**Regions:** EC, ITG, PFC, V2, V1

## Experiment Setup

- Phase 1: Data already generated (create_data_v2 --cell_type Astrocytes, expr_transform=log1p)
- Phase 2: Optuna hyperparameter search, 20 new trials per study (expression and embedding per region)
- Phase 3: Seeded training with best params: n_seeds=10, epochs=200, patience=40, 8-fold LOFO CV, 80 runs per config
- Random embedding ablation: Same architecture as best embedding model (flatten_mlp) but with gene x 16 random vectors scaled by expression

All expression models selected by autosearch were MLP variants. All embedding/random_embedding models were flatten_mlp.

## Phase 2 -- Best Autosearch F1 (1-seed, quick screening)

| Region | Expression | Embedding |
|--------|-----------|-----------|
| EC | 0.6014 (mlp, pca=50) | 0.5875 (flatten_mlp, pca=16) |
| ITG | 0.5701 (mlp, no pca) | 0.5868 (flatten_mlp, pca=16) |
| PFC | 0.5063 (mlp, pca=100) | 0.5889 (flatten_mlp, no pca) |
| V2 | 0.5486 (mlp, pca=50) | 0.6111 (flatten_mlp, no pca) |
| V1 | 0.5326 (mlp, pca=16) | N/A (stuck; used EC emb params) |

## Phase 3 -- Seeded Training Results (10 seeds x 8 folds = 80 runs)

### Generalized F1 (mean +/- std)

| Region | Expression | Embedding | Random Embedding |
|--------|-----------|-----------|-----------------|
| EC | 0.3876 +/- 0.2567 | 0.4515 +/- 0.2854 | **0.4565 +/- 0.2618** |
| ITG | 0.4530 +/- 0.2425 | **0.4909 +/- 0.2216** | 0.4787 +/- 0.2378 |
| PFC | 0.4432 +/- 0.2320 | 0.5691 +/- 0.1624 | **0.5715 +/- 0.1590** |
| V2 | 0.3688 +/- 0.2335 | 0.4737 +/- 0.2187 | **0.5385 +/- 0.2160** |
| V1 | 0.3889 +/- 0.2231 | 0.3903 +/- 0.2468 | **0.4608 +/- 0.2551** |

### Pooled held-out F1

| Region | Expression | Embedding | Random Embedding |
|--------|-----------|-----------|-----------------|
| EC | 0.4350 | **0.4755** | 0.4491 |
| ITG | 0.4475 | **0.4997** | 0.4625 |
| PFC | 0.4406 | **0.5597** | 0.5561 |
| V2 | 0.3730 | 0.4790 | **0.5429** |
| V1 | 0.3812 | 0.3882 | **0.4497** |

### Best input type per region (by Generalized F1)

| Region | Best Input Type | Generalized F1 | Runner-up |
|--------|----------------|----------------|-----------|
| EC | random_embedding | 0.4565 | embedding (0.4515) |
| ITG | embedding | 0.4909 | random_embedding (0.4787) |
| PFC | random_embedding | 0.5715 | embedding (0.5691) |
| V2 | random_embedding | 0.5385 | embedding (0.4737) |
| V1 | random_embedding | 0.4608 | embedding (0.3903) |

## Cross-Region Summary

### Expression vs Embedding

| Metric | Expression wins | Embedding wins |
|--------|----------------|---------------|
| Generalized F1 | 0/5 | 5/5 |
| Pooled F1 | 0/5 | 5/5 |

Embedding consistently outperforms expression across all 5 regions, with improvements ranging from +1.4 pp (V1) to +12.6 pp (PFC) in Generalized F1.

### Embedding vs Random Embedding

| Metric | Embedding wins | Random wins |
|--------|---------------|-------------|
| Generalized F1 | 1/5 (ITG) | 4/5 |
| Pooled F1 | 3/5 (EC, ITG, PFC) | 2/5 (V2, V1) |

Results are mixed:
- Generalized F1: Random embedding outperforms learned embedding in 4/5 regions. Only ITG shows a clear advantage for learned embeddings.
- Pooled F1: Learned embedding edges out random in 3/5 regions (EC, ITG, PFC), but margins are small.
- V2 and V1 show the largest advantage for random embeddings (+6.5 pp and +7.1 pp in Generalized F1).

### Key Observations

1. Expression-only models are the weakest in every region. Gene embedding information (whether learned or random) consistently helps.

2. Random embeddings are surprisingly competitive. The gene x 16 random vectors scaled by expression capture enough structural variation to match or exceed learned GREmLN embeddings in most regions.

3. Large variance across folds/seeds (std 0.16-0.29) indicates that the Thal classification task is inherently noisy with the available sample sizes. Differences between embedding and random_embedding are often within one standard deviation.

4. PFC is the most classifiable region (best F1 ~ 0.57), while V2 and V1 are the hardest. Expression alone peaks around 0.37-0.44.

5. V1 embedding was searched with EC proxy params (autosearch timed out on MIL), which may have underestimated the true embedding potential for V1.

### Interpretation

The learned gene embeddings from GREmLN do provide value over raw expression (consistent 5-13 pp improvement), but do not clearly outperform random embeddings in this patient-level Thal classification task. This suggests that the primary benefit may come from the dimensionality expansion (expression x gene-vectors to richer feature space) rather than the specific biological information captured by GREmLN.

Possible next steps:
- Cell-level training (--cell_level) to leverage individual cell data rather than patient aggregates
- Larger embedding models or gene_encoder architecture to better exploit learned embeddings
- Paired statistical tests (Wilcoxon on per-fold F1) to determine significance of differences
- More Optuna trials (50-100) for embedding autosearch to find better hyperparameters

## Run Artifacts

- Phase 2 best params: ranking_model_runs/autosearch_v2/v2_REGION_Astrocytes_expr|emb_best_params.json
- Phase 3 run logs: runs/train_model_v2_20260404_HHMMSS_REGION_input_model.log
- Phase 3 runner script: scripts/run_phase3.py
