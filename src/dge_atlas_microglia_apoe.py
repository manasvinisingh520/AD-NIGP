"""Atlas microglia DGE per region: AD vs control (Pathology Stage 3-4 vs 1-2).

For each brain region (EC, ITG, PFC, V1, V2) run DESeq2 on donor-pseudobulked Atlas
microglia. Condition = AD (stage 3-4) vs control (1-2). NO covariates (design ~condition),
matching the rest of the pipeline (the model and the SEA-AD/astro DGEs never adjust for
APOE; Atlas microglia also has no batch column). APOE per-donor balance is printed for
context only.

Note: pseudobulk uses ~100-cell bins per donor (project convention), so bins are
pseudo-replicates within donor.
"""
import os
import numpy as np
import pandas as pd

from utils import (read_mtx_file, create_anndata_object,
                   aggregate_cells_into_pseudobulk, get_region_file_paths)
from dge2 import DGEAnalyzer

BASE = "data/AtlasData"
OUT = "results/dge_final"
REGIONS = ["EC", "ITG", "PFC", "V1", "V2"]
SIG = {"padj": 0.05, "lfc": 0.5}


def run_region(region):
    mtx, row, col = get_region_file_paths(region, data_dir=BASE,
                                          base_prefix="2025-12-27_Microglia_{region}")
    meta = pd.read_csv(f"{BASE}/2025-12-27_Microglia_{region}_metadata.csv")
    bc = meta.columns[0]
    meta["condition"] = np.where(meta["Pathology Stage"].astype(str).isin(["1", "2"]),
                                 "control", "AD")
    meta["apoe4"] = np.where(meta["APOE"].astype(str).str.contains("4"),
                             "carrier", "noncarrier")
    meta["Donor_ID"] = meta["Donor ID"].astype(str)

    dd = meta.drop_duplicates("Donor_ID")
    cond = dd["condition"].value_counts().to_dict()
    print(f"\n{'#'*60}\n# {region}: cells={len(meta)} donors={dd.shape[0]} condition={cond} "
          f"(APOE4 carriers={int((dd['apoe4']=='carrier').sum())}, context only)")
    if cond.get("AD", 0) < 2 or cond.get("control", 0) < 2:
        print(f"  SKIP {region}: need >=2 donors per group.")
        return region, None, "too few donors per group"

    matrix, genes, cells = read_mtx_file(str(mtx), str(row), str(col), transpose=False)
    meta = meta.set_index(bc).reindex(cells)
    adata = create_anndata_object(matrix=matrix, gene_names=genes, cell_names=cells,
                                  transpose=True, obs=meta)
    pb = aggregate_cells_into_pseudobulk(adata, target_cells_per_bin=100,
                                         filter_patients_cell_threshold=80, seed=100,
                                         patient_col="Donor_ID")
    pb.obs["condition"] = pb.obs["condition"].astype(str)

    out = f"{OUT}/dge_results_Atlas_Microglia_{region}_pseudobulk100_seed100_ADvsControl.csv"
    try:
        DGEAnalyzer(data_file=pb, condition="condition", groups=[("AD", "control")],
                    method="deseq2", threshold={"padj": SIG["padj"], "log2fc": SIG["lfc"]},
                    topK=50, output_file=out, n_cpus=16, covariates=None).run_full_analysis()
    except Exception as e:
        print(f"  ERROR {region}: {type(e).__name__}: {e}")
        return region, None, f"{type(e).__name__}: {e}"

    actual = out[:-4] + "_AD_vs_control.csv"      # DGEAnalyzer appends the contrast
    df = pd.read_csv(actual, index_col=0).dropna(subset=["padj"])
    n_sig = int(((df["padj"] < SIG["padj"]) & (df["log2FoldChange"].abs() >= SIG["lfc"])).sum())
    return region, (len(df), n_sig, actual), None


def main():
    os.makedirs(OUT, exist_ok=True)
    summary = [run_region(r) for r in REGIONS]
    print("\n" + "=" * 60)
    print("SUMMARY  (AD vs control, ~condition, no covariates)")
    for region, res, err in summary:
        if err:
            print(f"  {region:4s}: -- ({err})")
        else:
            tested, n_sig, _ = res
            print(f"  {region:4s}: {n_sig} significant (padj<{SIG['padj']}, |log2FC|>={SIG['lfc']}) of {tested} tested")
    print("=" * 60)


if __name__ == "__main__":
    main()
