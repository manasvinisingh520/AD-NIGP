"""Create pseudobulk-bin model data (experiment #4: raw expression per pseudocell,
no GREmLN embedding, no patient aggregation).

This is the raw-expression twin of experiment #2 (create_cell_embedding_data.py):
each pseudobulk bin is one sample, we do NOT aggregate bins to the patient level, and
we build PATIENT-GROUPED folds (StratifiedGroupKFold on the donor) so every bin of a
donor lands in the same fold -- no patient leakage. The only difference from #2 is the
feature: here X is the bin's (library-normalized, log1p) gene expression instead of the
512-dim GREmLN cell embedding.

Bins are summed counts with varying bin sizes, so each bin is normalized to a common
library size (normalize_total, 1e4) then log1p'd -- the standard way to compare summed
pseudobulk profiles. Data is stored RAW (all genes, no PCA); the trainer does leakage-
free per-fold PCA (matching flatten_mlp / #2 / #3).

Input:
  data/GRN/<DATA>/<ct>/<ct>_pseudobulk_AnnData.h5ad        (bin x genes, per-bin obs)

Output:
  data/model_data/<DATA>_<ct>_pseudobulk_data.pkl          (same schema as #2)

Usage:
  python src/create_pseudobulk_bin_data.py --data SEAAD --cell_type Astrocytes
"""
import argparse
import os
import pickle

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedGroupKFold

from config import _SEAAD_THAL, _SEAAD_BRAAK

N_FOLDS = 8
TARGET_DIMS = 16
NORM_TARGET = 1e4   # per-bin library-size normalization target (counts-per-10k), then log1p

# Per-dataset pseudobulk obs columns for (thal, braak, donor). mapping=None -> numeric.
LABELS = {
    "SEAAD": {"thal": ("Thal phase", _SEAAD_THAL), "braak": ("Braak stage", _SEAAD_BRAAK),
              "donor": "donor_id"},
    "ROSMAP": {"thal": ("ceradsc", None), "braak": ("braaksc", None),
               "donor": "individualID"},
}


def _map(series, mapping):
    if mapping is not None:
        return series.astype(str).str.strip().map(mapping)
    return pd.to_numeric(series.astype(str).str.strip(), errors="coerce")


def _normalize_log1p(X):
    """Per-bin library-size normalize to NORM_TARGET, then log1p. Returns dense float32."""
    X = X.astype(np.float32)
    if sp.issparse(X):
        X = X.toarray()
    else:
        X = np.asarray(X, dtype=np.float32)
    lib = X.sum(axis=1, keepdims=True)
    lib[lib == 0] = 1.0
    X = X / lib * NORM_TARGET
    return np.log1p(X).astype(np.float32)


def main(args):
    data, ct = args.data, args.cell_type
    cfg = LABELS[data]
    pb_path = os.path.join("data", "GRN", data, ct, f"{ct}_pseudobulk_AnnData.h5ad")

    adata = ad.read_h5ad(pb_path)
    obs = adata.obs
    bin_names = obs["_sample_id"].astype(str).tolist()
    X = _normalize_log1p(adata.X)
    print(f"Pseudobulk bins: {X.shape} ({len(bin_names)} pseudocells), library-normalized + log1p")

    y_thal_all = _map(obs[cfg["thal"][0]], cfg["thal"][1]).to_numpy()
    y_braak_all = _map(obs[cfg["braak"][0]], cfg["braak"][1]).to_numpy()
    donors_all = obs[cfg["donor"]].astype(str).to_numpy()

    # Drop any bin with a missing Thal/Braak label.
    keep = [i for i in range(len(bin_names)) if pd.notna(y_thal_all[i]) and pd.notna(y_braak_all[i])]
    if len(keep) < len(bin_names):
        print(f"  dropped {len(bin_names) - len(keep)} bins with missing labels")
    X = X[keep]
    bin_names = [bin_names[i] for i in keep]
    y_thal = np.array([int(y_thal_all[i]) for i in keep], dtype=np.int64)
    y_braak = np.array([int(y_braak_all[i]) for i in keep], dtype=np.int64)
    donors = np.array([donors_all[i] for i in keep])
    print(f"Samples: {len(bin_names)} pseudocells across {len(set(donors))} donors "
          f"({len(bin_names) / len(set(donors)):.1f} bins/donor)")
    print(f"  Thal classes: {sorted(set(y_thal))}  Braak classes: {sorted(set(y_braak))}")

    # Optional global PCA (default off: store raw so the trainer does per-fold PCA).
    if args.n_components and args.n_components < X.shape[1]:
        X = PCA(n_components=args.n_components, random_state=42).fit_transform(X).astype(np.float32)
        print(f"  PCA-reduced -> {X.shape}")
    else:
        print(f"  Kept raw (no PCA) -> {X.shape}; trainer will do per-fold PCA")

    # Patient-grouped, Thal-stratified 8-fold split (all of a donor's bins stay together).
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    X_folds, thal_folds, braak_folds, donor_folds, bin_folds = [], [], [], [], []
    for fi, (_, test_idx) in enumerate(sgkf.split(X, y_thal, donors)):
        X_folds.append(X[test_idx])
        thal_folds.append(y_thal[test_idx])
        braak_folds.append(y_braak[test_idx])
        donor_folds.append(donors[test_idx])
        bin_folds.append([bin_names[i] for i in test_idx])
        print(f"  Fold {fi + 1}: {len(test_idx)} bins, {len(set(donors[test_idx]))} donors")

    # Sanity: no donor appears in more than one fold.
    seen = {}
    for fi, df_ in enumerate(donor_folds):
        for d in set(df_):
            if d in seen:
                raise RuntimeError(f"Leakage: donor {d} in folds {seen[d]} and {fi}")
            seen[d] = fi
    print("  OK: every donor's bins are confined to a single fold (no leakage).")

    out_dir = os.path.join("data", "model_data")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{data}_{ct}_pseudobulk_data.pkl")
    payload = {
        "X_folds": X_folds, "thal_folds": thal_folds, "braak_folds": braak_folds,
        "donor_folds": donor_folds, "bin_folds": bin_folds,
        "n_folds": N_FOLDS, "embedding_dim": int(X.shape[1]),
        "data": data, "cell_type": ct,
    }
    with open(out_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="SEAAD", choices=list(LABELS.keys()))
    ap.add_argument("--cell_type", default="Astrocytes", choices=["Astrocytes", "Microglia"])
    ap.add_argument("--n_components", type=int, default=0,
                    help="Global PCA dim (default 0 = keep raw; trainer does per-fold PCA).")
    main(ap.parse_args())
