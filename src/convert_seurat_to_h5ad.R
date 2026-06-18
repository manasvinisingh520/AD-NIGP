# =============================================================================
#  seurat_to_h5ad.R
#  Converts a Seurat object (.rds) to .h5ad format using anndataR
# =============================================================================

# ── 1. SET YOUR PATHS HERE ───────────────────────────────────────────────────

INPUT_RDS   <- "I:/sf2026_Extension/data/ROSMAP_Data/Immune_cells.rds"   # <-- change me
OUTPUT_H5AD <- "I:/sf2026_Extension/data/ROSMAP_Data/Immune_cells.h5ad"          # <-- change me

# ── 2. INSTALL MISSING PACKAGES ──────────────────────────────────────────────

CRAN_MIRROR <- "https://cloud.r-project.org"

install_if_missing_cran <- function(pkg) {
  if (requireNamespace(pkg, quietly = TRUE)) {
    message(sprintf("[OK]      '%s' is already installed.", pkg))
    return(invisible(NULL))
  }
  message(sprintf("[INSTALL] Installing '%s' from CRAN...", pkg))
  install.packages(pkg, repos = CRAN_MIRROR, quiet = TRUE)
  if (!requireNamespace(pkg, quietly = TRUE)) {
    stop(sprintf("Failed to install '%s'. Please install it manually.", pkg))
  }
  message(sprintf("[OK]      '%s' installed successfully.", pkg))
}

install_if_missing_bioc <- function(pkg) {
  if (requireNamespace(pkg, quietly = TRUE)) {
    message(sprintf("[OK]      '%s' is already installed.", pkg))
    return(invisible(NULL))
  }
  message(sprintf("[INSTALL] Installing '%s' from Bioconductor...", pkg))
  BiocManager::install(pkg, ask = FALSE, quiet = TRUE)
  if (!requireNamespace(pkg, quietly = TRUE)) {
    stop(sprintf("Failed to install '%s'. Please install it manually.", pkg))
  }
  message(sprintf("[OK]      '%s' installed successfully.", pkg))
}

# BiocManager must come first so we can use it to install Bioc packages
install_if_missing_cran("BiocManager")
install_if_missing_cran("SeuratObject")

# Both anndataR and rhdf5 (required for h5ad I/O) are Bioconductor packages
install_if_missing_bioc("rhdf5")
install_if_missing_bioc("anndataR")

library(SeuratObject)
library(anndataR)

# ── 3. VALIDATE PATHS ────────────────────────────────────────────────────────

if (!file.exists(INPUT_RDS)) {
  stop("Input file not found: ", INPUT_RDS)
}

out_dir <- dirname(OUTPUT_H5AD)
if (!dir.exists(out_dir)) {
  message("Creating output directory: ", out_dir)
  dir.create(out_dir, recursive = TRUE)
}

# ── 4. LOAD AND WRITE ────────────────────────────────────────────────────────

message("Reading RDS: ", INPUT_RDS)
seurat_obj <- readRDS(INPUT_RDS)

if (!inherits(seurat_obj, "Seurat")) {
  stop("Loaded object is not a Seurat object. Class: ", paste(class(seurat_obj), collapse = ", "))
}
message(sprintf("Loaded  |  %d cells  x  %d genes", ncol(seurat_obj), nrow(seurat_obj)))

message("Writing H5AD to: ", OUTPUT_H5AD)
write_h5ad(seurat_obj, OUTPUT_H5AD)

if (file.exists(OUTPUT_H5AD)) {
  size_mb <- round(file.info(OUTPUT_H5AD)$size / 1024^2, 2)
  message(sprintf("[DONE] %s written (%.2f MB)", OUTPUT_H5AD, size_mb))
} else {
  stop("Output file not found after write: ", OUTPUT_H5AD)
}