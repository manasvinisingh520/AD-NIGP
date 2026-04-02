# Run DGE2 for multiple pseudobulk sizes x 5 seeds, then print intersection % (reproducibility).
# Usage: .\run_dge2_pseudobulk_reproducibility.ps1
#        .\run_dge2_pseudobulk_reproducibility.ps1 -SkipRuns   # only recompute from existing results

param(
    [switch]$SkipRuns
)

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

if ($SkipRuns) { $env:SKIP_RUNS = "1" } else { $env:SKIP_RUNS = "" }

$pythonCode = @'
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

PSEUDOBULK_SIZES = [100, 200, 400, 600, 800, 1000]
N_SEEDS = 5
SEED_START = 100
OUTPUT_DIR = Path("results/dge_final")
PADJ_THRESHOLD = 0.05
LOG2FC_THRESHOLD = 0.5
COMPARISONS = ['1_vs_2', '1_vs_3', '1_vs_4', '2_vs_3', '2_vs_4', '3_vs_4']
DGE2_SCRIPT = Path("src/dge2.py")


def run_dge2(pseudobulk, seed):
    cmd = [sys.executable, str(DGE2_SCRIPT),
           "-cell_type", "SEA_AD", "-region", "MTG",
           "-adata", "data/SEA_AD_Data/MTG/MTG_AnnData.h5ad",
           "-condition_col", "Overall AD neuropathological Change",
           "-patient_col", "Donor ID",
           "-pseudobulk", str(pseudobulk), "-seed", str(seed)]
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"  [FAIL] pseudobulk={pseudobulk} seed={seed}: {e}")
        return False


def significant_genes_from_csv(csv_path, padj_thresh, log2fc_thresh):
    if not csv_path.exists():
        return set()
    df = pd.read_csv(csv_path, index_col=0)
    if "padj" not in df.columns or "log2FoldChange" not in df.columns:
        return set()
    mask = (df["padj"].notna() & (df["padj"] < padj_thresh) &
            (df["log2FoldChange"].abs() > log2fc_thresh))
    return set(df.index[mask].astype(str).str.strip())


def out_suffix(pseudobulk, seed):
    return f"SEA_AD_MTG_pseudobulk{pseudobulk}_seed{seed}"


skip_runs = os.environ.get("SKIP_RUNS") == "1"
output_dir = OUTPUT_DIR
seeds = [SEED_START + i for i in range(N_SEEDS)]

if not skip_runs:
    if not DGE2_SCRIPT.exists():
        raise SystemExit(f"dge2 script not found: {DGE2_SCRIPT}")
    print("Running dge2.py for each (pseudobulk, seed)...")
    for pseudobulk in PSEUDOBULK_SIZES:
        for seed in seeds:
            print(f"  pseudobulk={pseudobulk} seed={seed}")
            run_dge2(pseudobulk, seed)
    print("Done running dge2.\n")

print("Computing intersection of significant genes across runs...")
rows = []
for pseudobulk in PSEUDOBULK_SIZES:
    for comp in COMPARISONS:
        sets_per_run = []
        for seed in seeds:
            csv_path = output_dir / f"dge_results_{out_suffix(pseudobulk, seed)}_{comp}.csv"
            sets_per_run.append(significant_genes_from_csv(csv_path, PADJ_THRESHOLD, LOG2FC_THRESHOLD))
        if not sets_per_run:
            inter, union, avg_run_size = set(), set(), 0
        else:
            inter = set.intersection(*sets_per_run)
            union = set.union(*sets_per_run)
            run_sizes = [len(s) for s in sets_per_run]
            avg_run_size = sum(run_sizes) / len(run_sizes)
        jaccard = len(inter) / len(union) if union else 0.0
        pct_of_union = 100.0 * jaccard
        pct_of_avg_run = (100.0 * len(inter) / avg_run_size) if avg_run_size else 0.0
        rows.append({"pseudobulk": pseudobulk, "comparison": comp,
                    "n_intersection": len(inter), "n_union": len(union),
                    "pct_intersection": round(pct_of_union, 1), "pct_of_avg_run": round(pct_of_avg_run, 1),
                    "jaccard": round(jaccard, 4)})
df = pd.DataFrame(rows)
summary = df.groupby("pseudobulk").agg(
    total_reproducible=("n_intersection", "sum"),
    mean_jaccard=("jaccard", "mean"),
    mean_pct_intersection=("pct_intersection", "mean")).reset_index()
summary = summary.sort_values("total_reproducible", ascending=False)

print("\n--- Per comparison: intersection size and % (intersection/union x 100) ---")
print(df.to_string(index=False))
print("\n--- Summary by pseudobulk ---")
print(summary.to_string(index=False))
best = summary.iloc[0]
print(f"\nMost reproducible: pseudobulk = {best['pseudobulk']} (total reproducible = {int(best['total_reproducible'])}, mean % intersection = {best['mean_pct_intersection']:.1f}%, mean Jaccard = {best['mean_jaccard']:.4f})")
'@

$tempPy = Join-Path $env:TEMP "run_dge2_pseudobulk_reproducibility_temp.py"
$pythonCode | Set-Content -Path $tempPy -Encoding UTF8
try {
    & python $tempPy
} finally {
    Remove-Item -Path $tempPy -Force -ErrorAction SilentlyContinue
}
exit $LASTEXITCODE
