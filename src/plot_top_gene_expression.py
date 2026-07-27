# Line-plot the expression of the top-N genes (from the model ranking or the DGE union)
# across pathology stages (Thal phase or Braak stage).
#
# For each gene, expression is aggregated pseudobulk-style: mean log1p expression per
# donor (over that donor's cells), then averaged across donors within each stage.
# One line per gene; shaded band = SEM across donors. x-axis = stage, y = mean expression.
#
# Expression source: data/model_data/<DATA>_<ct>_v2_data_log1p.pkl (cell-level, GRN gene set).
# Genes not in that matrix (common for DGE hits, which span the full transcriptome) are skipped.

import os
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt

STAGE_KEY = {"thal": "thal_folds", "braak": "braak_folds"}
STAGE_NICE = {"thal": "Thal phase", "braak": "Braak stage"}


def load_top_genes(args):
    if args.source == "dge":
        from utils import get_top_dge_union_genes
        dge_data = {"SEAAD": "SEA_AD"}.get(args.data, args.data)
        return get_top_dge_union_genes(data=dge_data, cell_type=args.cell_type,
                                       region=args.region, top_k=args.n_genes,
                                       dge_results_dir="results/dge_final")
    import pandas as pd
    return pd.read_csv(args.ranked_list)["gene"].tolist()[: args.n_genes]


def main(args):
    pkl = f"data/model_data/{args.data}_{args.cell_type}_v2_data_log1p.pkl"
    d = pickle.load(open(pkl, "rb"))
    gene_names = list(d["gene_names"])
    gidx = {g: i for i, g in enumerate(gene_names)}
    expr_folds = d["expression_folds"]
    info_folds = d["patient_info_folds"]
    stage_folds = d[STAGE_KEY[args.stage]]

    genes = load_top_genes(args)
    present = [g for g in genes if g in gidx]
    missing = [g for g in genes if g not in gidx]
    if missing:
        print(f"Skipping {len(missing)} gene(s) not in the expression matrix "
              f"({len(gene_names)}-gene set): {missing[:8]}{' ...' if len(missing) > 8 else ''}")
    if not present:
        raise SystemExit("None of the requested genes are in the expression matrix.")
    cols = [gidx[g] for g in present]

    # symbols for the legend
    try:
        g2e = pickle.load(open("data/genes_2_ensembl_ids.pkl", "rb"))
        e2s = {v: k for k, v in g2e.items()}
    except Exception:
        e2s = {}
    symbols = [e2s.get(g, g) for g in present]

    # Per-donor mean expression (over that donor's cells) for the selected genes.
    donor_vecs, donor_stage = [], []
    for fi in range(len(expr_folds)):
        fold_expr = np.asarray(expr_folds[fi], dtype=np.float32)  # (cells, n_genes)
        cell_idx = 0
        for di, (_, cell_names) in enumerate(info_folds[fi]):
            n = len(cell_names)
            block = fold_expr[cell_idx:cell_idx + n][:, cols]
            donor_vecs.append(np.nanmean(block, axis=0))
            donor_stage.append(int(float(stage_folds[fi][di])))
            cell_idx += n
    donor_vecs = np.vstack(donor_vecs)               # (n_donors, n_present_genes)
    donor_stage = np.asarray(donor_stage)

    stages = sorted(np.unique(donor_stage))
    means = np.full((len(present), len(stages)), np.nan)
    sems = np.full((len(present), len(stages)), np.nan)
    n_per_stage = []
    for si, s in enumerate(stages):
        m = donor_stage == s
        n_per_stage.append(int(m.sum()))
        means[:, si] = donor_vecs[m].mean(axis=0)
        sems[:, si] = donor_vecs[m].std(axis=0, ddof=1) / max(np.sqrt(m.sum()), 1)

    # Plot: one line per gene.
    fig, ax = plt.subplots(figsize=(max(7, 1.2 * len(stages)), 6))
    cmap = plt.get_cmap("tab20" if len(present) > 10 else "tab10")
    x = np.arange(len(stages))
    for i, sym in enumerate(symbols):
        ax.plot(x, means[i], marker="o", lw=1.8, color=cmap(i % cmap.N), label=sym)
        ax.fill_between(x, means[i] - sems[i], means[i] + sems[i], color=cmap(i % cmap.N), alpha=0.15)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}\n(n={n})" for s, n in zip(stages, n_per_stage)])
    ax.set_xlabel(f"{STAGE_NICE[args.stage]}  (donors per stage)")
    ax.set_ylabel("mean log1p expression (per-donor, averaged over donors)")
    ax.set_title(f"{args.data} {args.cell_type} — top {len(present)} {args.source} genes "
                 f"vs {STAGE_NICE[args.stage]}", fontweight="bold")
    ax.legend(fontsize=8, ncol=2 if len(present) > 12 else 1, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Plotted {len(present)} genes across {len(stages)} {args.stage} stages -> {args.output}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=["ranking", "dge"], default="ranking")
    p.add_argument("--n_genes", type=int, default=10)
    p.add_argument("--data", default="SEAAD")
    p.add_argument("--cell_type", default="Microglia")
    p.add_argument("--region", default="MTG")
    p.add_argument("--stage", choices=["thal", "braak"], default="thal")
    p.add_argument("--ranked_list",
                   default="ranked_lists_v2/SEAAD_Microglia_flatten_mlp_thalonly_trial25/ranked_genes.csv")
    p.add_argument("--output", default=None)
    args = p.parse_args()
    if args.output is None:
        args.output = f"images/top{args.n_genes}_{args.source}_{args.data}_{args.cell_type}_{args.stage}_expression.png"
    main(args)
