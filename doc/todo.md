# DGE Analysis Tasks

## Figure out optimal value for max_cells_for_deseq2
 - Find out max top 10 genes from one region all groups combination (1,2), (1,3), (1,4), (2,3) etc with this varibale changed from [500, 1000, .. MAX]

### old_filtering_dge_results dir: we were randomly binning combining cells from different patients and putting into same bin, so the results are mixed up beteween different biological conditions. [Discarded]

### dge1: Binning by patients. Each patient has same number of bins. Results are dge1_results directory. Problem here is that some bins have no cells because number of cells are different for each patient. 1 bin or minimum number of bins that resulted in no empty bins, should have meaningful result.

### dge2: Number of bins per patient proportional to number of cells collected for said patient. Focuses on same number of cells per bin rather that same number of bins. Should ideally extract best number of bins for each patient.

## 1. Run different group comparisons within EC region
- Run all comparisons (1 vs 3, 1 vs 4, etc.) with `max_cells_for_deseq2 = 500`
  - While running, figure out how to switch tabs in metadata file (from EC = default to BA 20, etc)
  - Learn about DESeq2 pipeline (dispersions, MAP dispersions, LFCs, outlier genes, etc)

## 2. Test different `max_cells_for_deseq2` values
- Run analyses with `max_cells_for_deseq2 = 250`
- Run analyses with `max_cells_for_deseq2 = 1000`
- Compare results

## 3. Test different `min_counts` values
- Why is `min_counts = 10`?
- Set different values and run analyses
- Compare results

Winter Break:
## 1. Finish DEG analysis
    - should finish 12/23
## 2. Code GRN
    - GRNBoost2 looks best to me
    - uses motif/TF dataset which is already downloaded
    - 1-1.5 weeks
## 3. Start GREmLN process