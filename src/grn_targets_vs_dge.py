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
# All genes that appear as a node (source or target) in the full Microglia GRN -- the
# network's gene vocabulary. Used as the enrichment background when universe="grn".
GRN_GENES_FILE = "data/GRN/SEAAD/Microglia/Microglia_GRN_genes.txt"


def check_targets_vs_dge(trend="both", padj=0.05, universe="grn"):
    """Which GRN targets (of each set's ranked genes) are trend-DEGs in the stageR table?

    DGE reference = genes significant in the combined stageR trend test(s) selected by
    `trend` (".L" monotone, ".Q" curvature, or "both" = .L or .Q), read from STAGER_TABLE
    via utils.get_top_DEGs.

    universe : {"grn", "transcriptome"}
        Background genes for the enrichment test. "grn" restricts to genes that appear in
        the full Microglia GRN (GRN_GENES_FILE) -- the fair background, since only those
        genes can ever be targets. "transcriptome" uses every gene DESeq2 tested (this
        conflates "is the gene in the GRN at all" with "is it a target", biasing the fold).

    For each GRN set, prints how many of its targets are DE plus the enrichment over the
    chosen background (Fisher, two-sided), and writes the overlapping target genes (with
    their trend padj + how many ranked genes hit them) to OVERLAP_DIR.
    """
    from scipy.stats import fisher_exact
    from utils import get_top_DEGs
    stager = pd.read_csv(STAGER_TABLE, index_col=0)
    uni = set(stager.index[stager["padjScreen"].notna()])        # genes actually tested
    if universe == "grn":
        grn_genes = {ln.strip() for ln in open(GRN_GENES_FILE) if ln.strip()}
        uni &= grn_genes                                         # restrict to GRN vocabulary
    deg = set(get_top_DEGs(trend=trend, padj_thresh=padj)) & uni
    print(f"DGE reference: stageR trend '{trend}' padj<{padj}, universe='{universe}' "
          f"-> {len(deg)} DEGs (background {len(uni)} genes, rate {len(deg)/max(len(uni),1):.1%})\n")
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
        tt = targets & uni                   # targets that are in the background universe
        td = tt & deg                        # targets that are trend-DEGs
        bg = len(deg) / max(len(uni), 1)
        frac = len(td) / max(len(tt), 1)
        de_t, nde_t = len(td), len(tt - deg)
        de_nt, nde_nt = len(deg - tt), len(uni - tt - deg)
        OR, p = fisher_exact([[de_t, de_nt], [nde_t, nde_nt]], alternative="two-sided")
        # overlapping targets, with trend stats and how many ranked genes target each
        n_ranked = tsub.groupby("connected_gene")["ranked_gene"].nunique()
        ov = (stager.loc[sorted(td), ["padjScreen", "trend.L", "trend.Q"]]
              .assign(n_ranked_genes=[int(n_ranked[g]) for g in sorted(td)])
              .sort_values(["trend.L", "trend.Q"]))
        ofile = os.path.join(OVERLAP_DIR, f"{label}_{trend.strip('.')}_{universe}_targets_DE.csv")
        ov.to_csv(ofile)
        print(f"{label:<20} {len(targets):>8} {len(tt):>7} {len(td):>5} "
              f"{bg:>5.1%} {frac/max(bg,1e-9):>5.2f}x {p:>9.1e}  {ofile}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--trend", default="both", choices=[".L", ".Q", "both"],
                   help="Which stageR trend defines the DEG set (default: both = .L or .Q).")
    p.add_argument("--padj", type=float, default=0.05, help="padj cutoff for DEGs (default 0.05).")
    p.add_argument("--universe", default="grn", choices=["grn", "transcriptome"],
                   help="Enrichment background: 'grn' = genes in the GRN (fair), "
                        "'transcriptome' = all DESeq2-tested genes (default: grn).")
    args = p.parse_args()
    check_targets_vs_dge(trend=args.trend, padj=args.padj, universe=args.universe)
