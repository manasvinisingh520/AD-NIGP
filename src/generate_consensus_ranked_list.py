"""Regenerate cross-seed consensus ranked-gene lists (min<N>) for the filtered GRN (trial26).

Consensus is computed across the per-seed ranked files (ranked_genes.csv + ranked_genes_2..10):
  n_seeds_pos            = # seeds where the gene's mean_f1_drop > 0
  consensus_mean_f1_drop = mean of mean_f1_drop across seeds
  mean_rank_across_seeds = mean of rank across seeds
Genes with n_seeds_pos >= N are kept, sorted by consensus_mean_f1_drop (desc), re-ranked
as combined_rank. Columns match the existing combined_ranked_genes_min6/min7.csv.

The script first VALIDATES by regenerating min6 & min7 and comparing to the saved files;
only if they match does it write the requested new thresholds.
"""
import glob
import os
import re
import pickle
import pandas as pd

DIR = "ranked_lists_v2/SEAAD_Microglia_filtered_flatmlp_trial26"
NEW_THRESHOLDS = [8, 9, 10]


def load_seed_files():
    files = [f for f in glob.glob(f"{DIR}/ranked_genes*.csv")
             if re.search(r"ranked_genes(_\d+)?\.csv$", os.path.basename(f))]
    files.sort()
    print(f"per-seed files ({len(files)}): {[os.path.basename(f) for f in files]}")
    return files


def build_consensus(files):
    drops, ranks = {}, {}   # gene -> list across seeds
    for f in files:
        df = pd.read_csv(f)
        for _, r in df[["gene", "mean_f1_drop", "rank"]].iterrows():
            drops.setdefault(r["gene"], []).append(float(r["mean_f1_drop"]))
            ranks.setdefault(r["gene"], []).append(float(r["rank"]))
    rows = []
    for g in drops:
        d = drops[g]
        rows.append({"gene": g,
                     "n_seeds_pos": sum(1 for x in d if x > 0),
                     "consensus_mean_f1_drop": sum(d) / len(d),
                     "mean_rank_across_seeds": sum(ranks[g]) / len(ranks[g])})
    con = pd.DataFrame(rows)
    e2g = {v: k for k, v in pickle.load(open("data/genes_2_ensembl_ids.pkl", "rb")).items()}
    con["gene_symbol"] = con["gene"].map(lambda e: e2g.get(e, e))
    return con


def make_threshold(con, n):
    sub = con[con["n_seeds_pos"] >= n].sort_values(
        "consensus_mean_f1_drop", ascending=False).reset_index(drop=True)
    sub.insert(0, "combined_rank", range(1, len(sub) + 1))
    return sub[["combined_rank", "gene", "gene_symbol", "n_seeds_pos",
                "consensus_mean_f1_drop", "mean_rank_across_seeds"]]


def validate(con):
    ok = True
    for n in (6, 7):
        saved = pd.read_csv(f"{DIR}/combined_ranked_genes_min{n}.csv")
        gen = make_threshold(con, n)
        same_genes = set(saved["gene"]) == set(gen["gene"])
        same_order = list(saved["gene"]) == list(gen["gene"])
        print(f"  min{n}: saved={len(saved)} regenerated={len(gen)} | "
              f"same_genes={same_genes} same_order={same_order}")
        ok = ok and same_genes
    return ok


def main():
    con = build_consensus(load_seed_files())
    print(f"consensus over {con.shape[0]} genes; "
          f"n_seeds_pos range {con.n_seeds_pos.min()}-{con.n_seeds_pos.max()}")
    print("VALIDATION vs saved min6/min7:")
    if not validate(con):
        print("  !! validation FAILED - not writing new files (logic differs). Investigate.")
        return
    print("  validation PASSED.\n")
    for n in NEW_THRESHOLDS:
        gen = make_threshold(con, n)
        out = f"{DIR}/combined_ranked_genes_min{n}.csv"
        gen.to_csv(out, index=False)
        print(f"  wrote min{n}: {len(gen)} genes -> {out}")


if __name__ == "__main__":
    main()
