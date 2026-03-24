File: Description
adni_blood_gep.py: reads ADNI_gene_expression_profile
                fixes file format (uses subjects in column 3 as first column)
                prints info for 27 genes (listed in file)

anndata_filtering_guide.md: how to filter anndata objects based on rows, columns, etc

anndata_guide.md: describes anndata objects

categorize_genes.py: go through all Astrocytes genes
                    use mygene library to categorize genes into protein_coding, etc

cells_in_patients.py: counts number of cells for each patient in each condition
                        used to create filtering patients with less than 80 cells

cellxgene_prepare.py: prepare data for cellxgene visualization

compute_best_region.py: uses DGE analysis results to find "best" region
                        (STOPPED because no true formula)

convert_to_ensembl_IDs.py: takes in GRN file, matrix file, and AnnData file
                            creates replicas with ensembl IDs in place of genes

debug_padj_guide.md: understanding NaN padj Values in DESeq2 Results

deseq2_pseudobulk_normalization.md: how pseudobulking words in DGE analysis

dge.py: does DGE analysis on single-cell level analysis (NOT USED)

dge2.py: does DGE analysis after pseudobulking cells by patient (USED)

export_matrices_to_csv.py: creates {region}_AnnData.h5ad objects saved in data/GRN/
                            saves {region}_matrix.csv file as well in data/GRN/

export_TFs_to_txt.py: saves TFs found in specific region matrix in file {region}_TFs.txt in data/GRN/

glm_deseq2_guide.md: general linearized model usage in deseq2 Description

GREmLN_int_DGE.py: finds intersection in the genes in GREmLN embeddings and DGE results

mtx_file_guide.md: all about matrix files

normalization_guide.md: different ways to normalize sn-RNA seq data (CPM, UMI raw counts, etc)

plot_cell_embeddings.py: plots cell embeddings from GREmLN
                        goes through multiple dimensions and clusters (best silhoutte score)

plot_expressionLevel.py: plots graph showing
                        X-axis: Cell index (cells grouped by condition)
                        Y-axis: Normalized gene expression (0–1)
                        Colors: Different genes
                        Vertical lines: Separating conditions

plot_gene_embeddings.py: plot gene embeddings from GREmLN

plot_genes.py: creates an image with genes on Y-axis and cell expression (normalized) by Colors

pydeseq2_design_guide.md: explains design formula of deseq2

read_mtx_example.py: creates a "fake" mtx file to read

scale_gene_embeddings.py: scale gene embeddings according to cell expression (not read over)

scratch.py: simple scratch work

test_binning.py: extracts top k genes per condition comparison for given region 
                calculates overlap between all dge results 
                tests reproducability of binning strategy

test_motifs_dataset.py: test overlap between motif file and TF file (NOT USED)

test_TF_dataset.py: test overlap between TF file and Astrocytes genes (USED)

train_model.py online augmentation (replaces deprecated create_data --augment_scaling / _augScaling pickles):
  Use base data/model_data/{region}_data_for_model_training.pkl. Train-time: --aug_scale_jitter,
  --aug_per_gene_scale_jitter, --aug_gene_dropout, --aug_emb_dim_dropout, --aug_feature_dropout,
  --aug_gene_block_cutout_max, --aug_noise_std; batch-level --mixup_alpha, --gene_cutmix_prob;
  --label_smoothing for Thal. --no_online_augment disables dataset aug. Offline _augScaling is deprecated.

autosearch_ec.py: Optuna or random hyperparameter search for EC, loss_ptau=none, maximize mean LOFO F1.
  Pass --llm for OpenAI-compatible trial proposals. See ../program_ec_autosearch.md; LLM deps: ../requirements-autosearch-llm.txt.