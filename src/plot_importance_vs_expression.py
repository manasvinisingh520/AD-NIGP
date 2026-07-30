# Does gene "importance" just track expression level?
#
# For the model ranking and the DGE list, plot each gene's importance against its
# expression level and its expression variance, and report Spearman correlations.
# If importance rises with expression, the ranking is largely an expression artifact.
#
#   RANKED importance = mean_f1_drop (permutation ranking); expression = per-donor
#           log1p mean / variance from the model pickle (the model's actual input).
#   DGE    importance = -log10(best padj); expression = DESeq2 baseMean (its own basis).

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

DATA, CT, REGION = "SEAAD", "Microglia", "MTG"
RANKED_CSV = "ranked_lists_v2/SEAAD_Microglia_flatten_mlp_thalonly_trial25/ranked_genes.csv"
TOP_N = 200
OUT = f"images/importance_vs_basemean_{DATA}_{CT}.png"


def per_donor_expr_stats():
    """Per-gene mean and variance of expression across donors (pseudobulk)."""
    d = pickle.load(open(f"data/model_data/{DATA}_{CT}_v2_data_log1p.pkl", "rb"))
    genes = list(d["gene_names"])
    donor_means = []
    for fi, fold in enumerate(d["expression_folds"]):
        fe = np.asarray(fold, dtype=np.float32)
        ci = 0
        for _, cells in d["patient_info_folds"][fi]:
            n = len(cells)
            donor_means.append(np.nanmean(fe[ci:ci + n], axis=0))
            ci += n
    M = np.vstack(donor_means)                 # (donors, genes)
    return pd.DataFrame({"gene": genes,
                         "expr_mean": M.mean(0),
                         "expr_var": M.var(0)})


def dge_best():
    """Best padj and its baseMean per gene across the 6 pairwise DGE contrasts."""
    import glob
    files = glob.glob(f"results/dge_final/dge_results_SEA_AD_{CT}_{REGION}_*.csv")
    best = {}
    for f in files:
        df = pd.read_csv(f, index_col=0)
        if "padj" not in df:  # skip malformed
            continue
        sub = df.dropna(subset=["padj"])
        for g, row in sub.iterrows():
            p = float(row["padj"])
            if g not in best or p < best[g][0]:
                best[g] = (p, float(row["baseMean"]))
    return pd.DataFrame([(g, p, bm) for g, (p, bm) in best.items()],
                        columns=["gene", "padj", "baseMean"])


def scatter(ax, x, y, top_mask, xlabel, ylabel, title, logx=False):
    ax.scatter(x[~top_mask], y[~top_mask], s=6, c="#bbbbbb", alpha=0.5, label="all genes")
    ax.scatter(x[top_mask], y[top_mask], s=26, c="#c62828", edgecolor="k",
               linewidth=0.3, label=f"top {TOP_N}", zorder=3)
    rho, p = spearmanr(x, y)
    ax.set_title(f"{title}\nSpearman ρ={rho:.2f} (p={p:.1e})", fontsize=11, fontweight="bold")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    if logx:
        ax.set_xscale("log")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    return rho


def main():
    # DESeq2 baseMean per gene (its own expression measure) — used as the x-axis for BOTH panels.
    dg = dge_best()
    dg["neglog10padj"] = -np.log10(dg["padj"].clip(lower=1e-300))
    base = dg[["gene", "baseMean"]]

    # ---- RANKED: importance vs DESeq2 baseMean ----
    r = pd.read_csv(RANKED_CSV)[["gene", "rank", "mean_f1_drop"]]
    rk = r.merge(base, on="gene", how="inner")   # keep ranked genes that have a baseMean
    rk_top = rk["rank"] <= TOP_N

    # ---- DGE: importance vs DESeq2 baseMean ----
    dgs = dg.sort_values("padj").reset_index(drop=True)
    dg_top = np.arange(len(dgs)) < TOP_N

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    r1 = scatter(axes[0], rk["baseMean"].values, rk["mean_f1_drop"].values, rk_top.values,
                 "DESeq2 baseMean (expression level)", "importance (mean F1 drop)",
                 "RANKED: importance vs DESeq2 baseMean", logx=True)
    r3 = scatter(axes[1], dgs["baseMean"].values, dgs["neglog10padj"].values, dg_top,
                 "DESeq2 baseMean (expression level)", "importance (-log10 padj)",
                 "DGE: importance vs DESeq2 baseMean", logx=True)

    fig.suptitle(f"{DATA} {CT} — is gene importance just expression (DESeq2 baseMean)?\n"
                 f"RANKED ρ={r1:+.2f}   |   DGE ρ={r3:+.2f}   "
                 f"(|ρ|~0 → not expression; large → tracks expression)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    os.makedirs("images", exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"RANKED genes with a baseMean: {len(rk)} of {len(r)}")
    print(f"RANKED  Spearman: importance vs baseMean={r1:+.3f}")
    print(f"DGE     Spearman: importance vs baseMean={r3:+.3f}")
    print(f"Saved -> {OUT}")


if __name__ == "__main__":
    main()
