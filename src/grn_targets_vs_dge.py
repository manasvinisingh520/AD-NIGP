"""Cross-reference GRN targets against the stage-aware Thal DEGs (SEA-AD Microglia).

For each GRN set (no_threshold / mi3.0 / expr0.1 ranked-gene edge lists), take the targets
of that set's ranked genes and ask which of them are trend-DEGs in the combined stageR table
(results/r_dge_out/stager_combined.csv, produced by src/mixed_dge_thal_stager.R). Prints the
overlap + Fisher enrichment per set and writes the overlapping target genes to
results/grn_analysis/target_dge_overlap/.
"""

import os
import argparse
import pandas as pd


# GRN sets to cross-reference: (label, ranked-gene edge list). Each edge list gives the
# targets of that set's ranked genes; we ask which of those targets are trend-DEGs.
GRN_SETS = [
    ("no_threshold_top75", "results/grn_analysis/no_threshold/"
        "GRN_analysis_f1_top200_min5_SEAAD_Microglia_MTG_top75_edges.csv"),
    ("mi3.0_top52", "results/grn_analysis/mi3.0/"
        "GRN_analysis_f1_top200_min5_SEAAD_Microglia_MTG_top52_mi3_edges.csv"),
    ("expr0.1_top27", "results/grn_analysis/expr_mean0.1/"
        "GRN_analysis_f1_expr0.1_min5_SEAAD_Microglia_MTG_top27_edges.csv"),
]
STAGER_TABLE = "results/r_dge_out/stager_combined.csv"
OVERLAP_DIR = "results/grn_analysis/target_dge_overlap"


def check_targets_vs_dge(trend="both", padj=0.05):
    """Which GRN targets (of each set's ranked genes) are trend-DEGs in the stageR table?

    DGE reference = genes significant in the combined stageR trend test(s) selected by
    `trend` (".L" monotone, ".Q" curvature, or "both" = .L or .Q), read from STAGER_TABLE
    via utils.get_top_DEGs. For each GRN set, prints how many of its targets are DE and the
    enrichment over the tested background (Fisher, one-sided), and writes the overlapping
    target genes (with their trend padj + how many ranked genes hit them) to OVERLAP_DIR.
    """
    from scipy.stats import fisher_exact
    from utils import get_top_DEGs
    stager = pd.read_csv(STAGER_TABLE, index_col=0)
    universe = set(stager.index[stager["padjScreen"].notna()])   # genes actually tested
    deg = set(get_top_DEGs(trend=trend, padj_thresh=padj))
    print(f"DGE reference: stageR trend '{trend}' padj<{padj} -> {len(deg)} DEGs "
          f"(tested universe {len(universe)})\n")
    os.makedirs(OVERLAP_DIR, exist_ok=True)
    hdr = (f"{'GRN set':<20} {'targets':>8} {'tested':>7} {'DE':>5} "
           f"{'bg%':>6} {'fold':>6} {'fisher_p':>9}  overlap_file")
    print(hdr); print("-" * len(hdr))
    for label, path in GRN_SETS:
        if not os.path.exists(path):
            print(f"{label:<20} (missing: {path})"); continue
        e = pd.read_csv(path)
        tsub = e[e["relationship"] == "target"]
        targets = set(tsub["connected_gene"])
        tt = targets & universe              # targets that were tested in the DGE
        td = tt & deg                        # targets that are trend-DEGs
        bg = len(deg) / max(len(universe), 1)
        frac = len(td) / max(len(tt), 1)
        de_t, nde_t = len(td), len(tt - deg)
        de_nt, nde_nt = len(deg - tt), len(universe - tt - deg)
        OR, p = fisher_exact([[de_t, de_nt], [nde_t, nde_nt]], alternative="greater")
        # overlapping targets, with trend stats and how many ranked genes target each
        n_ranked = tsub.groupby("connected_gene")["ranked_gene"].nunique()
        ov = (stager.loc[sorted(td), ["padjScreen", "trend.L", "trend.Q"]]
              .assign(n_ranked_genes=[int(n_ranked[g]) for g in sorted(td)])
              .sort_values(["trend.L", "trend.Q"]))
        ofile = os.path.join(OVERLAP_DIR, f"{label}_{trend.strip('.')}_targets_DE.csv")
        ov.to_csv(ofile)
        print(f"{label:<20} {len(targets):>8} {len(tt):>7} {len(td):>5} "
              f"{bg:>5.1%} {frac/max(bg,1e-9):>5.2f}x {p:>9.1e}  {ofile}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--trend", default="both", choices=[".L", ".Q", "both"],
                   help="Which stageR trend defines the DEG set (default: both = .L or .Q).")
    p.add_argument("--padj", type=float, default=0.05, help="padj cutoff for DEGs (default 0.05).")
    args = p.parse_args()
    check_targets_vs_dge(trend=args.trend, padj=args.padj)
