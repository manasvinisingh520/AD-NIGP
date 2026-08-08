# load ranked gene list, calculate log1p expression, mi threshold, etc and generate new list to put into GRN analysis
"""Build a thresholded ranked-gene list to feed into GRN_analysis_rankedGenes.py.

Loads a ranked-gene CSV (permutation-importance output) and filters it by:
  - mean_f1_drop  : keep genes with drop > --f1_min (default 0 -> importance-positive genes),
  - mean_log1p_expr : keep genes expressed at >= --expr_min (computed from the per-cell AnnData;
                      matches the log1p transform of the v2 pipeline).
Writes a new CSV (with a 'gene' column + the kept metrics), ordered by mean_f1_drop, for:
    python src/GRN_analysis_rankedGenes.py --ranked_list <out> --mi_threshold <mi> ...
(the MI/edge threshold is applied there, on the GRN edges.)
"""
import os
import argparse

import numpy as np
import pandas as pd

PERCELL = "data/model_data/perCell_AnnData/SEAAD_Microglia_AnnData_perCell.h5ad"
EXPR_CACHE = "data/model_data/SEAAD_Microglia_mean_log1p_expr.csv"


def gene_mean_log1p_expr(percell=PERCELL, cache=EXPR_CACHE):
    """Per-gene mean of log1p(raw counts) over all cells (Ensembl-indexed), cached to CSV."""
    if os.path.exists(cache):
        return pd.read_csv(cache, index_col=0).iloc[:, 0]
    import anndata
    import scipy.sparse as sp
    print(f"computing mean_log1p_expr from {percell} ...")
    ad = anndata.read_h5ad(percell)
    X = ad.X
    if sp.issparse(X):
        Xl = X.astype("float32").copy()
        Xl.data = np.log1p(Xl.data)
        m = np.asarray(Xl.mean(axis=0)).ravel()
    else:
        m = np.log1p(np.asarray(X, dtype="float32")).mean(axis=0)
    s = pd.Series(m, index=ad.var_names.astype(str), name="mean_log1p_expr")
    os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
    s.to_csv(cache)
    print(f"  cached {len(s)} genes -> {cache}")
    return s


def _drop_col(df):
    for c in ("mean_f1_drop", "consensus_mean_f1_drop"):
        if c in df.columns:
            return c
    raise ValueError("ranked list needs a 'mean_f1_drop' or 'consensus_mean_f1_drop' column.")


def main(args):
    df = pd.read_csv(args.ranked_list)
    df["gene"] = df["gene"].astype(str)
    dropcol = _drop_col(df)
    n0 = len(df)

    if args.f1_min is not None:
        df = df[df[dropcol] > args.f1_min]
    if args.expr_min is not None:
        expr = gene_mean_log1p_expr(args.percell)
        df["mean_log1p_expr"] = df["gene"].map(expr).astype(float).fillna(0.0)
        df = df[df["mean_log1p_expr"] >= args.expr_min]

    df = df.sort_values(dropcol, ascending=False).reset_index(drop=True)
    print(f"{n0} -> {len(df)} genes  (f1_min={args.f1_min}, expr_min={args.expr_min})")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"wrote thresholded ranked list -> {args.output}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ranked_list", required=True,
                   help="Input ranked-gene CSV (needs a 'gene' column and a *_f1_drop column).")
    p.add_argument("--f1_min", type=float, default=0.0,
                   help="Keep genes with mean_f1_drop > this (default 0). Pass a negative number to keep all.")
    p.add_argument("--expr_min", type=float, default=None,
                   help="Keep genes with mean_log1p_expr >= this (e.g. 0.1). Default: no expression filter.")
    p.add_argument("--percell", default=PERCELL,
                   help="Per-cell AnnData used to compute mean_log1p_expr (Ensembl var_names).")
    p.add_argument("--output", required=True, help="Output CSV path for the thresholded list.")
    main(p.parse_args())
