"""Create cell-embedding model data (experiment: cell embeddings, no patient concat).

Each GREmLN cell embedding is one pseudocell (pseudobulk bin) = one sample. We do NOT
aggregate bins to the patient level (so there are ~n_bins samples, not n_patients), but
we build PATIENT-GROUPED folds (StratifiedGroupKFold on the donor) so every bin of a
donor lands in the same fold — no patient leakage across train/dev/test.

Input:
  GREmLN/embeddings/<DATA>_<ct>_cell_embeddings.tsv        (bin x 512)
  data/GRN/<DATA>/<ct>/<ct>_pseudobulk_AnnData.h5ad        (per-bin Thal/Braak/donor)

Output:
  data/model_data/<DATA>_<ct>_cellemb_data.pkl
    X_folds      list[np.ndarray (n_bins, D) float32]   per fold
    thal_folds   list[np.ndarray (n_bins,) int]         per-bin Thal (donor's label)
    braak_folds  list[np.ndarray (n_bins,) int]
    donor_folds  list[np.ndarray (n_bins,) str]         group id per bin
    bin_folds    list[list[str]]                        bin names
    n_folds, embedding_dim, data, cell_type

Usage:
  python src/create_cell_embedding_data.py --data SEAAD --cell_type Astrocytes
"""
import argparse
import os
import pickle

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedGroupKFold

from config import _SEAAD_THAL, _SEAAD_BRAAK

N_FOLDS = 8
TARGET_DIMS = 16   # PCA target, matching the gene-embedding pipeline (create_data_v2)

# Per-dataset pseudobulk obs columns for (thal, braak, donor). mapping=None -> numeric.
LABELS = {
    "SEAAD": {"thal": ("Thal phase", _SEAAD_THAL), "braak": ("Braak stage", _SEAAD_BRAAK),
              "donor": "donor_id"},
    "ROSMAP": {"thal": ("ceradsc", None), "braak": ("braaksc", None),
               "donor": "individualID"},
}


def load_cell_embeddings(path):
    """First row = header, first column = bin name, rest = embedding dims."""
    df = pd.read_csv(path, header=None)
    df = df.iloc[1:]
    names = df.iloc[:, 0].astype(str).str.strip().tolist()
    X = df.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all").to_numpy(np.float32)
    return X, names


def _map(series, mapping):
    if mapping is not None:
        return series.astype(str).str.strip().map(mapping)
    return pd.to_numeric(series.astype(str).str.strip(), errors="coerce")


def main(args):
    data, ct = args.data, args.cell_type
    cfg = LABELS[data]
    emb_path = os.path.join("GREmLN", "embeddings", f"{data}_{ct}_cell_embeddings.tsv")
    pb_path = os.path.join("data", "GRN", data, ct, f"{ct}_pseudobulk_AnnData.h5ad")

    X, bin_names = load_cell_embeddings(emb_path)
    print(f"Cell embeddings: {X.shape} ({len(bin_names)} pseudocells)")

    obs = ad.read_h5ad(pb_path, backed="r").obs
    sid = obs["_sample_id"].astype(str)
    lut_thal = dict(zip(sid, _map(obs[cfg["thal"][0]], cfg["thal"][1])))
    lut_braak = dict(zip(sid, _map(obs[cfg["braak"][0]], cfg["braak"][1])))
    lut_donor = dict(zip(sid, obs[cfg["donor"]].astype(str)))

    # Align bins to labels; drop any bin with a missing Thal/Braak.
    keep = [i for i, b in enumerate(bin_names)
            if b in lut_thal and pd.notna(lut_thal[b]) and pd.notna(lut_braak[b])]
    if len(keep) < len(bin_names):
        print(f"  dropped {len(bin_names) - len(keep)} bins with missing labels")
    X = X[keep]
    bin_names = [bin_names[i] for i in keep]
    y_thal = np.array([int(lut_thal[b]) for b in bin_names], dtype=np.int64)
    y_braak = np.array([int(lut_braak[b]) for b in bin_names], dtype=np.int64)
    donors = np.array([lut_donor[b] for b in bin_names])
    print(f"Samples: {len(bin_names)} pseudocells across {len(set(donors))} donors "
          f"({len(bin_names) / len(set(donors)):.1f} bins/donor)")
    print(f"  Thal classes: {sorted(set(y_thal))}  Braak classes: {sorted(set(y_braak))}")

    # PCA-reduce the cell embeddings (matches the gene-embedding pipeline's 512->16).
    # Global unsupervised PCA (no labels), like create_data_v2's PCA on the embedding matrix.
    if args.n_components and args.n_components < X.shape[1]:
        X = PCA(n_components=args.n_components, random_state=42).fit_transform(X).astype(np.float32)
        print(f"  PCA-reduced cell embeddings -> {X.shape}")

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
    out_path = os.path.join(out_dir, f"{data}_{ct}_cellemb_data.pkl")
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
    ap.add_argument("--n_components", type=int, default=TARGET_DIMS,
                    help=f"PCA-reduce cell embeddings to this dim (default {TARGET_DIMS}; 0 = keep full 512).")
    main(ap.parse_args())
