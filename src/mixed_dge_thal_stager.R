# Combined stage-aware Thal DGE for SEA-AD Microglia (DESeq2 + stageR).
#   design ~ sex + age + PMI + thal   (RIN unavailable in SEA-AD; age/PMI binned factors)
#
# Screen-then-confirm (stageR):
#   SCREEN  : LRT omnibus  full(~sex+age+PMI+thal) vs reduced(~sex+age+PMI)
#             -> one p/gene "does this gene respond to Thal at all?" (gene-level FDR gate)
#   CONFIRM : within each screened gene (Holm, alpha=0.05), a matrix of contrasts on a
#             single Wald fit -- polynomial trend .L (monotone) & .Q (curvature), plus
#             per-stage thal k vs 0 (each stage vs the healthy Thal-0 reference).
#   .L/.Q are orthogonal-polynomial contrast weights on the thal coefficients, so the whole
#   confirmation set comes from ONE Wald model (no separate ordered-factor refit). Holm is
#   valid under arbitrary dependence, so mixing the (correlated) poly & k-vs-0 contrasts is fine.
#
# Output: results/r_dge_out/stager_combined.csv -- padjScreen + confirmation padj columns
#         (trend.L, trend.Q, thal1_vs_0..thal5_vs_0).
.libPaths(c(file.path(Sys.getenv("USERPROFILE"), "Rlibs"), .libPaths()))
suppressMessages({library(DESeq2); library(stageR)})

OUT <- "results/r_dge_out"; dir.create(OUT, showWarnings = FALSE, recursive = TRUE)

counts <- as.matrix(read.csv("results/r_dge_input/counts.csv", row.names = 1, check.names = FALSE))
coldata <- read.csv("results/r_dge_input/coldata.csv", row.names = 1, check.names = FALSE,
                    stringsAsFactors = FALSE)
counts <- counts[, rownames(coldata)]                      # align samples
storage.mode(counts) <- "integer"
for (c in c("sex", "age", "PMI")) coldata[[c]] <- factor(coldata[[c]])
coldata$thal <- factor(as.character(coldata$thal), levels = as.character(0:5))
covars <- c("sex", "age", "PMI")
covars <- covars[sapply(covars, function(c) nlevels(coldata[[c]]) > 1)]   # drop single-level
cat("covariates in design:", paste(covars, collapse = ", "), "\n")
full <- as.formula(paste("~", paste(c(covars, "thal"), collapse = " + ")))
reduced <- as.formula(paste("~", paste(covars, collapse = " + ")))
cat("full:", deparse(full), "| reduced:", deparse(reduced), "\n")

# ---- SCREEN: LRT omnibus (full vs reduced) ----
ddsL <- DESeqDataSetFromMatrix(counts, coldata, design = full)
ddsL <- DESeq(ddsL, test = "LRT", reduced = reduced, quiet = TRUE)
resL <- results(ddsL)
cat(sprintf("LRT screen: tested=%d, raw p<0.05=%d\n",
            sum(!is.na(resL$pvalue)), sum(resL$pvalue < 0.05, na.rm = TRUE)))

# ---- CONFIRM: single Wald fit -> poly (.L/.Q) + OVR contrasts ----
ddsW <- DESeq(DESeqDataSetFromMatrix(counts, coldata, design = full), quiet = TRUE)
rn <- resultsNames(ddsW)
cat("resultsNames:", paste(rn, collapse = ", "), "\n")
thal_cols <- grep("^thal_", rn)                            # thal_1_vs_0 .. thal_5_vs_0
lev_of_col <- as.integer(sub("^thal_([0-9]+)_vs_0$", "\\1", rn[thal_cols]))  # 1..5

# Orthogonal-polynomial contrast vector over coefficients. For level k the group mean is
# Intercept + beta_k (beta_0 = 0), and poly weights sum to zero -> Intercept term drops, so
# the contrast is sum_k w[k] * beta_k with w[k] = contr.poly weight for that level.
P <- contr.poly(6)                                         # rows = levels 0..5, cols .L .Q .C ^4 ^5
polyvec <- function(cn) {
  v <- numeric(length(rn)); w <- P[, cn]
  v[thal_cols] <- w[lev_of_col + 1L]                       # level k -> row k+1
  v
}
vL <- polyvec(".L"); vQ <- polyvec(".Q")

pL <- results(ddsW, contrast = vL)$pvalue
pQ <- results(ddsW, contrast = vQ)$pvalue
# Per-stage vs the Thal-0 reference (pairwise): stage k vs 0, k = 1..5.
pK0 <- sapply(as.character(1:5), function(k) results(ddsW, contrast = c("thal", k, "0"))$pvalue)
pConf <- cbind(trend.L = pL, trend.Q = pQ, pK0)
colnames(pConf) <- c("trend.L", "trend.Q", paste0("thal", 1:5, "_vs_0"))
rownames(pConf) <- rownames(ddsW)

# ---- stageR: screen on LRT, Holm-confirm the contrast matrix within gene ----
pScreen <- resL$pvalue; names(pScreen) <- rownames(resL)
keep <- !is.na(pScreen)
pC <- pConf[keep, , drop = FALSE]; pC[is.na(pC)] <- 1
obj <- stageR(pScreen = pScreen[keep], pConfirmation = pC, pScreenAdjusted = FALSE)
obj <- stageWiseAdjustment(obj, method = "holm", alpha = 0.05)
sr <- getResults(obj)
cat(sprintf("stageR: genes passing screen=%d\n", sum(sr[, "padjScreen"] == 1)))
cat("genes significant per confirmation contrast (stage-wise, alpha=0.05):\n")
print(colSums(sr[, -1, drop = FALSE]))
padj <- getAdjustedPValues(obj, onlySignificantGenes = FALSE, order = FALSE)
write.csv(as.data.frame(padj), file.path(OUT, "stager_combined.csv"))
cat("Saved combined screen+confirm ->", file.path(OUT, "stager_combined.csv"), "\n")
