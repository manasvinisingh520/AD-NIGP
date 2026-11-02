"""Batch-effect metrics: raw pseudobulk expression vs GREmLN embedding (SEA-AD Microglia).

Per pseudobulk bin (429 bins, aligned by _sample_id), quantify how much each
representation clusters by technical batch (batch_vendor_name). We also report a
biology-preservation control (Thal phase), so we can distinguish genuine batch *mixing*
from just blurring signal away.

Representations (each reduced to PCA-50):
  raw     : pseudobulk expression, normalize_total + log1p
  gremln  : GREmLN 512-d bin embedding, standardized

Metrics (per representation):
  ASW_batch  silhouette by batch        lower  = less batch structure (better)
  iLISI      kNN inverse-Simpson batch  higher = better batch mixing (max ~#batches)
  PCR_batch  variance explained by batch  lower  = less batch effect
  ASW_bio    silhouette by Thal         higher = biology preserved
  cLISI_bio  kNN inverse-Simpson Thal   lower  = biology preserved (max ~#Thal levels)
"""
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression
from sklearn.metrics import silhouette_score

PB = "data/GRN/SEAAD/Microglia/Microglia_pseudobulk_AnnData.h5ad"
EMB = "GREmLN/embeddings/SEAAD_Microglia_cell_embeddings.tsv"
BATCH, BIO = "batch_vendor_name", "Thal phase"


def lisi(Z, labels, k=30):
    labels = np.asarray(labels)
    k = min(k, len(Z) - 1)
    _, idx = NearestNeighbors(n_neighbors=k + 1).fit(Z).kneighbors(Z)
    vals = []
    for i in range(len(Z)):
        neigh = labels[idx[i, 1:]]
        p = pd.Series(neigh).value_counts(normalize=True).to_numpy()
        vals.append(1.0 / np.sum(p ** 2))          # inverse Simpson
    return float(np.mean(vals))


def pcr(Z, var_ratio, labels):
    oh = OneHotEncoder(sparse_output=False, drop="first").fit_transform(
        np.asarray(labels).reshape(-1, 1))
    r2 = np.array([LinearRegression().fit(oh, Z[:, j]).score(oh, Z[:, j])
                   for j in range(Z.shape[1])])
    return float(np.sum(r2 * var_ratio) / np.sum(var_ratio))


def reduce_pca(X, n=50, standardize=False):
    X = np.asarray(X, dtype=np.float64)
    if standardize:
        X = StandardScaler().fit_transform(X)
    n = min(n, X.shape[1], X.shape[0] - 1)
    p = PCA(n_components=n, random_state=0).fit(X)
    return p.transform(X), p.explained_variance_ratio_


def metrics(Z, vr, batch, bio):
    return {
        "ASW_batch": silhouette_score(Z, batch),
        "iLISI_batch": lisi(Z, batch),
        "PCR_batch": pcr(Z, vr, batch),
        "ASW_bio": silhouette_score(Z, bio),
        "cLISI_bio": lisi(Z, bio),
    }


def main():
    a = ad.read_h5ad(PB)
    emb = pd.read_csv(EMB, sep=None, engine="python", index_col=0)
    emb.index = emb.index.astype(str)
    a.obs["_sample_id"] = a.obs["_sample_id"].astype(str)

    common = [s for s in a.obs["_sample_id"] if s in emb.index]
    a = a[a.obs["_sample_id"].isin(common)].copy()
    order = a.obs["_sample_id"].tolist()
    E = emb.loc[order].to_numpy(dtype=np.float64)
    batch = a.obs[BATCH].astype(str).to_numpy()
    bio = a.obs[BIO].astype(str).to_numpy()
    print(f"bins={a.n_obs}  batches={len(set(batch))}  Thal levels={len(set(bio))}")

    # raw expression: normalize + log1p
    raw = a.copy()
    sc.pp.normalize_total(raw, target_sum=1e4)
    sc.pp.log1p(raw)
    Zr, vr_r = reduce_pca(raw.X.toarray() if hasattr(raw.X, "toarray") else raw.X, standardize=True)
    Zg, vr_g = reduce_pca(E, standardize=True)

    res = {"raw_expression": metrics(Zr, vr_r, batch, bio),
           "gremln_embedding": metrics(Zg, vr_g, batch, bio)}
    df = pd.DataFrame(res).T

    print("\n" + "=" * 66)
    print(df.round(3).to_string())
    print("-" * 66)
    r, g = res["raw_expression"], res["gremln_embedding"]
    better = lambda k, lower: ("gremln" if (g[k] < r[k]) == lower else "raw")
    print("Batch effect (want gremln better):")
    print(f"  ASW_batch  raw={r['ASW_batch']:+.3f} gremln={g['ASW_batch']:+.3f}  -> less batch: {better('ASW_batch', True)}")
    print(f"  iLISI      raw={r['iLISI_batch']:.2f}  gremln={g['iLISI_batch']:.2f}   -> better mixing: {better('iLISI_batch', False)}")
    print(f"  PCR_batch  raw={r['PCR_batch']:.3f} gremln={g['PCR_batch']:.3f}  -> less batch var: {better('PCR_batch', True)}")
    print("Biology preservation (want maintained):")
    print(f"  ASW_bio    raw={r['ASW_bio']:+.3f} gremln={g['ASW_bio']:+.3f}")
    print(f"  cLISI_bio  raw={r['cLISI_bio']:.2f}  gremln={g['cLISI_bio']:.2f}  (lower=biology kept)")
    print("=" * 66)
    import os
    os.makedirs("results/atac", exist_ok=True)
    df.to_csv("results/batch_effect_SEAAD_Microglia_raw_vs_gremln.csv")
    print("saved -> results/batch_effect_SEAAD_Microglia_raw_vs_gremln.csv")


if __name__ == "__main__":
    main()
