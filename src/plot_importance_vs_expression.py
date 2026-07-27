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
TOP_N = 50
OUT = f"images/importance_vs_expression_{DATA}_{CT}.png"


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
    stats = per_donor_expr_stats()

    # ---- RANKED ----
    r = pd.read_csv(RANKED_CSV)[["gene", "rank", "mean_f1_drop"]]
    rk = stats.merge(r, on="gene", how="inner")
    rk_top = rk["rank"] <= TOP_N

    # ---- DGE ----
    dg = dge_best()
    dg["neglog10padj"] = -np.log10(dg["padj"].clip(lower=1e-300))
    dg = dg.sort_values("padj").reset_index(drop=True)
    dg_top = np.arange(len(dg)) < TOP_N

    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    r1 = scatter(axes[0, 0], rk["expr_mean"].values, rk["mean_f1_drop"].values, rk_top.values,
                 "mean log1p expression (per-donor)", "importance (mean F1 drop)",
                 "RANKED: importance vs expression LEVEL")
    r2 = scatter(axes[0, 1], rk["expr_var"].values, rk["mean_f1_drop"].values, rk_top.values,
                 "expression variance across donors", "importance (mean F1 drop)",
                 "RANKED: importance vs expression VARIANCE")
    r3 = scatter(axes[1, 0], dg["baseMean"].values, dg["neglog10padj"].values, dg_top,
                 "DESeq2 baseMean (expression level)", "importance (-log10 padj)",
                 "DGE: importance vs expression LEVEL", logx=True)
    # DGE variance proxy: |log2FC| isn't variance; use baseMean again vs rank is covered.
    # Second DGE panel: importance vs |log2FC| to show it's not just fold-change-from-noise.
    axes[1, 1].axis("off")
    axes[1, 1].text(0.5, 0.5,
                    f"Spearman ρ (importance vs expression):\n\n"
                    f"RANKED vs expr level:     {r1:+.2f}\n"
                    f"RANKED vs expr variance:  {r2:+.2f}\n"
                    f"DGE   vs expr level:      {r3:+.2f}\n\n"
                    f"(|ρ| near 0  → importance is NOT just expression;\n"
                    f" |ρ| large   → importance largely tracks expression)",
                    ha="center", va="center", fontsize=12, family="monospace",
                    bbox=dict(boxstyle="round", facecolor="#f5f5f5"))

    fig.suptitle(f"{DATA} {CT} — is gene importance just expression level?",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs("images", exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"RANKED  Spearman: importance vs expr level={r1:+.3f}, vs expr variance={r2:+.3f}")
    print(f"DGE     Spearman: importance vs baseMean={r3:+.3f}")
    print(f"Saved -> {OUT}")


if __name__ == "__main__":
    main()
