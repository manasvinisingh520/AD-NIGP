"""Create patient-mean-expression model data (experiment #3: AnnData concat).

For each donor, aggregate ALL their cells into ONE vector = mean of log1p(counts)
across the donor's cells (the same patient aggregation the default pipeline uses),
then PCA-reduce to 16 dims. So each donor is a single sample.

Input:
  data/model_data/perCell_AnnData/<DATA>_<ct>_AnnData_perCell.h5ad  (X = raw counts;
      obs: SampleName / Thal / Braak)

Output:
  data/model_data/<DATA>_<ct>_patientmean_data.pkl
    X_folds      list[np.ndarray (n_donors, 16) float32]  per fold
    thal_folds / braak_folds  list[np.ndarray (n_donors,) int]
    donor_folds / bin_folds   donor ids per fold
    n_folds, embedding_dim, data, cell_type

Usage:
  python src/create_patient_mean_data.py --data SEAAD --cell_type Astrocytes
"""
import argparse
import os
import pickle

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold

N_FOLDS = 8
TARGET_DIMS = 16


def _donor_mean_log1p(X, mask):
    """Mean of log1p(counts) over a donor's cells -> (n_genes,)."""
    sub = X[mask]
    if sp.issparse(sub):
        sub = sub.astype(np.float32).copy()
        sub.data = np.log1p(sub.data)          # log1p(0)=0, so sparsity preserved
        return np.asarray(sub.mean(axis=0)).ravel()
    return np.log1p(np.asarray(sub, dtype=np.float32)).mean(axis=0)


def main(args):
    data, ct = args.data, args.cell_type
    pc = os.path.join("data", "model_data", "perCell_AnnData", f"{data}_{ct}_AnnData_perCell.h5ad")
    adata = ad.read_h5ad(pc)
    X = adata.X
    obs = adata.obs
    donors = obs["SampleName"].astype(str).values
    uniq = list(pd.unique(donors))
    print(f"perCell: {adata.shape} | {len(uniq)} donors")

    # Per-donor mean log1p expression + per-donor labels (constant within a donor).
    Xd = np.vstack([_donor_mean_log1p(X, donors == d) for d in uniq]).astype(np.float32)
    y_thal = np.array([int(float(obs.loc[donors == d, "Thal"].iloc[0])) for d in uniq], np.int64)
    y_braak = np.array([int(float(obs.loc[donors == d, "Braak"].iloc[0])) for d in uniq], np.int64)
    print(f"Patient-mean expression: {Xd.shape} (donors x genes)")
    print(f"  Thal classes: {sorted(set(y_thal))}  Braak classes: {sorted(set(y_braak))}")

    # Optional global PCA. Pass --n_components 0 to store RAW mean expression so the
    # trainer can do leakage-free per-fold PCA instead (matches flatten_mlp).
    if args.n_components:
        n_comp = min(args.n_components, Xd.shape[0], Xd.shape[1])
        Xd = PCA(n_components=n_comp, random_state=42).fit_transform(Xd).astype(np.float32)
        print(f"  PCA-reduced -> {Xd.shape}")
    else:
        print(f"  Kept raw (no PCA) -> {Xd.shape}; trainer will do per-fold PCA")

    donors_arr = np.array(uniq)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    X_folds, thal_folds, braak_folds, donor_folds, bin_folds = [], [], [], [], []
    for fi, (_, te) in enumerate(skf.split(Xd, y_thal)):
        X_folds.append(Xd[te]); thal_folds.append(y_thal[te]); braak_folds.append(y_braak[te])
        donor_folds.append(donors_arr[te]); bin_folds.append(list(donors_arr[te]))
        print(f"  Fold {fi + 1}: {len(te)} donors")

    out_dir = os.path.join("data", "model_data")
    out_path = os.path.join(out_dir, f"{data}_{ct}_patientmean_data.pkl")
    with open(out_path, "wb") as f:
        pickle.dump({
            "X_folds": X_folds, "thal_folds": thal_folds, "braak_folds": braak_folds,
            "donor_folds": donor_folds, "bin_folds": bin_folds,
            "n_folds": N_FOLDS, "embedding_dim": int(Xd.shape[1]),
            "data": data, "cell_type": ct,
        }, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="SEAAD", choices=["SEAAD", "ROSMAP"])
    ap.add_argument("--cell_type", default="Astrocytes", choices=["Astrocytes", "Microglia"])
    ap.add_argument("--n_components", type=int, default=TARGET_DIMS)
    main(ap.parse_args())
