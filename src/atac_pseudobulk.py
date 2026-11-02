"""Build a donor x peak pseudobulk count matrix of microglia ATAC counts (SEA-AD MTG).

Each SEA-AD donor object holds raw ATAC counts per nucleus per peak (218,882 peaks).
For every donor we keep only the microglia nuclei and SUM their counts per peak, giving one
sample per donor - the unit DESeq2 needs (AD stage is a donor property, so donors are the
replicates, not nuclei).

SEA-AD used two assays: 10x ATAC-only ("10xATAC_v1.1", batches ATX-*) and 10x Multiome
("10xMulti", batches MTX-*). Every donor has ATAC-only nuclei; 27 donors also have Multiome
nuclei and no donor is Multiome-only. --method ATX keeps only ATAC-only nuclei so that every
count comes from the same protocol while keeping all donors.

Input : data/SEAAD_ATAC/donor_objects/<donor>_SEAAD_MTG_ATACseq_final-nuclei.2024-12-06.h5ad
Output: data/SEAAD_ATAC/microglia_pseudobulk_counts[_<method>].h5ad
        obs = one row per donor (n microglia, ADNC, Thal, batch, sex, age, QC, ...)
        var = peaks;  X = summed integer counts

Run (Windows sf2026 env):
    python src/atac_pseudobulk.py --method ATX
"""
import argparse
import glob
import os

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

DONOR_DIR = "data/SEAAD_ATAC/donor_objects"
OUT = "data/SEAAD_ATAC/microglia_pseudobulk_counts{tag}.h5ad"
CELL_TYPE_COL, CELL_TYPE = "Subclass", "Microglia-PVM"
METHODS = {"ATX": "10xATAC_v1.1", "MTX": "10xMulti"}  # obs["method"] values

# donor-level metadata to carry along (same value for every nucleus of a donor)
DONOR_COLS = ["Donor ID", "Overall AD neuropathological Change", "Thal", "Braak", "CERAD score",
              "Cognitive Status", "APOE Genotype", "Sex", "Age at Death", "PMI",
              "Neurotypical reference", "Severely Affected Donor"]
# can differ between a donor's libraries; we record the most common value and how many there were
LIBRARY_COLS = ["batch_vendor_name", "method"]
# library QC (one value per library); averaged over the donor's microglia nuclei
QC_COLS = ["ATAC_TSS_enrichment_score", "ATAC_Fraction_of_high_quality_fragments_overlapping_peaks"]


def pseudobulk_donor(path, method=None):
    """Sum microglia counts per peak for one donor. Returns (counts vector, metadata dict, peaks).

    method: None = all nuclei, else keep only nuclei from that assay (key of METHODS).
    """
    a = ad.read_h5ad(path)
    keep = a.obs[CELL_TYPE_COL] == CELL_TYPE
    if method:
        keep &= a.obs["method"].astype(str) == METHODS[method]
    mg = a[keep]
    meta = {"file": os.path.basename(path), "n_microglia": mg.n_obs, "n_nuclei_total": a.n_obs}
    for c in QC_COLS:
        meta[f"mean_{c}"] = float(pd.to_numeric(mg.obs[c], errors="coerce").mean()) if mg.n_obs else np.nan
    for c in DONOR_COLS:
        vals = a.obs[c].astype(str).unique()
        if len(vals) > 1:
            raise ValueError(f"{path}: donor column {c!r} has several values {vals}")
        meta[c] = vals[0]
    for c in LIBRARY_COLS:
        src = mg.obs[c] if mg.n_obs else a.obs[c]
        meta[c] = src.astype(str).mode().iloc[0]
        meta[f"{c}_n_distinct"] = src.nunique()
    X = mg.X if sp.issparse(mg.X) else sp.csr_matrix(mg.X)
    counts = np.asarray(X.sum(axis=0)).ravel()
    # counts are stored as float32 but should be whole numbers; check before casting
    if not np.allclose(counts, np.round(counts)):
        raise ValueError(f"{path}: non-integer counts - not raw counts?")
    return counts.round().astype(np.int64), meta, a.var_names


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--method", choices=list(METHODS), default=None,
                    help="keep only nuclei from this assay (default: all)")
    args = ap.parse_args()
    out_path = OUT.format(tag=f"_{args.method}" if args.method else "")

    files = sorted(glob.glob(f"{DONOR_DIR}/*_SEAAD_MTG_ATACseq_final-nuclei.*.h5ad"))
    print(f"{len(files)} donor objects; nuclei kept: {METHODS.get(args.method, 'all assays')}")
    rows, metas, peaks = [], [], None
    for i, f in enumerate(files, 1):
        counts, meta, var_names = pseudobulk_donor(f, args.method)
        # every donor must use the same peak list in the same order, or summing across is wrong
        if peaks is None:
            peaks = var_names
        elif not peaks.equals(var_names):
            raise ValueError(f"{f}: peak list differs from the first donor")
        rows.append(counts)
        metas.append(meta)
        print(f"[{i}/{len(files)}] {meta['Donor ID']}: {meta['n_microglia']} microglia of "
              f"{meta['n_nuclei_total']} nuclei, ADNC={meta['Overall AD neuropathological Change']}, "
              f"total microglia counts={counts.sum():,}")

    obs = pd.DataFrame(metas).set_index("Donor ID")
    out = ad.AnnData(X=sp.csr_matrix(np.vstack(rows)), obs=obs, var=pd.DataFrame(index=peaks))
    out.write_h5ad(out_path)
    print(f"\nsaved {out_path}: {out.n_obs} donors x {out.n_vars} peaks")
    print("donors whose microglia span >1 library batch:", (obs.batch_vendor_name_n_distinct > 1).sum())
    print("\nmicroglia per donor:", obs.n_microglia.describe().round(0).to_dict())
    print("\nADNC counts:", obs["Overall AD neuropathological Change"].value_counts().to_dict())


if __name__ == "__main__":
    main()
