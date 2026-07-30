pkgs <- c("DESeq2", "stageR", "DEGreport", "ashr", "apeglm", "SummarizedExperiment", "data.table")
for (p in pkgs) {
  ok <- requireNamespace(p, quietly = TRUE)
  cat(sprintf("%-22s %s\n", p, if (ok) "INSTALLED" else "MISSING"))
}
cat("R:", R.version.string, "\n")
