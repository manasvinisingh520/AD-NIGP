# load GRN for data/cell_type/region
# load n ranked genes
# for each gene, find genes it directly targets (as a TF) + the TFs that regulate it, with MI
# save a tidy edge list + a per-gene summary to csv
#
# GRN format: headerless, TAB-separated, 3 columns -> source(TF)  target(gene)  MI
# Gene IDs are Ensembl (ENSG...), matching ranked_lists_v2 and the DGE results.

import os
import argparse
import pandas as pd

GRN_COLS = ["source", "target", "mi"]

# Ranked-list / GRN use "SEAAD"; the DGE files on disk use "SEA_AD".
DGE_DATA_NAME = {"SEAAD": "SEA_AD", "ROSMAP": "ROSMAP", "Atlas": "Atlas"}


def resolve_grn_path(data, cell_type, region):
    """GRN lives under one of two layouts depending on the dataset."""
    candidates = [
        f"data/GRN/{data}/{cell_type}/{cell_type}_GRN.tsv",  # SEAAD/ROSMAP
        f"data/GRN/{cell_type}/{region}_GRN.tsv",            # Atlas (region-prefixed)
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"No GRN found for data={data}, cell_type={cell_type}, region={region}. "
        f"Tried: {candidates}"
    )


def load_ranked_genes(args):
    """Top-N ranked genes as Ensembl IDs (from the permutation ranking or the DGE union)."""
    if args.load_dge_results:
        from utils import get_top_dge_union_genes
        return get_top_dge_union_genes(
            data=DGE_DATA_NAME.get(args.data, args.data),
            cell_type=args.cell_type,
            region=args.region,
            top_k=args.n_genes,
            dge_results_dir="results/dge_final",
        )
    df = pd.read_csv(args.ranked_list)
    return df["gene"].tolist()[: args.n_genes]


def load_grn_edges_for_genes(grn_path, geneset, chunksize=2_000_000):
    """Stream the (large) GRN and keep only edges touching a gene in `geneset`.

    Avoids loading the full ~1GB file into memory and avoids re-scanning it once
    per gene. Returns a small DataFrame with columns source/target/mi.
    """
    geneset = set(geneset)
    kept = []
    for chunk in pd.read_csv(
        grn_path, sep="\t", header=None, names=GRN_COLS,
        dtype={"source": str, "target": str, "mi": float}, chunksize=chunksize,
    ):
        mask = chunk["source"].isin(geneset) | chunk["target"].isin(geneset)
        if mask.any():
            kept.append(chunk.loc[mask])
    if not kept:
        return pd.DataFrame(columns=GRN_COLS)
    return pd.concat(kept, ignore_index=True)


def build_connections(genes, edges):
    """For each ranked gene, list its targets (it as TF) and regulators (TFs on it)."""
    # Pre-group once instead of scanning per gene.
    by_source = {g: sub for g, sub in edges.groupby("source")}
    by_target = {g: sub for g, sub in edges.groupby("target")}

    long_rows = []      # tidy edge list, one connection per row
    summary_rows = []   # per-gene counts
    for rank, gene in enumerate(genes, start=1):
        tgt = by_source.get(gene)
        reg = by_target.get(gene)
        n_t = 0 if tgt is None else len(tgt)
        n_r = 0 if reg is None else len(reg)
        if tgt is not None:
            for t, mi in zip(tgt["target"], tgt["mi"]):
                long_rows.append({"ranked_gene": gene, "rank": rank,
                                  "relationship": "target", "connected_gene": t, "mi": mi})
        if reg is not None:
            for s, mi in zip(reg["source"], reg["mi"]):
                long_rows.append({"ranked_gene": gene, "rank": rank,
                                  "relationship": "regulator", "connected_gene": s, "mi": mi})
        summary_rows.append({"ranked_gene": gene, "rank": rank,
                             "n_targets": n_t, "n_regulators": n_r})

    long_df = pd.DataFrame(long_rows, columns=["ranked_gene", "rank", "relationship",
                                               "connected_gene", "mi"])
    if not long_df.empty:
        long_df = long_df.sort_values(["rank", "relationship", "mi"],
                                      ascending=[True, True, False], ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)
    return long_df, summary_df


def main(args):
    grn_path = resolve_grn_path(args.data, args.cell_type, args.region)
    genes = load_ranked_genes(args)
    print(f"GRN: {grn_path}")
    print(f"Ranked genes ({len(genes)}): {genes[:10]}{' ...' if len(genes) > 10 else ''}")

    edges = load_grn_edges_for_genes(grn_path, genes)
    print(f"Edges touching the ranked genes: {len(edges)}")

    long_df, summary_df = build_connections(genes, edges)

    # Drop genes absent from the GRN (no targets and no regulators).
    absent = summary_df[(summary_df["n_targets"] == 0) & (summary_df["n_regulators"] == 0)]
    if len(absent):
        print(f"Dropping {len(absent)} gene(s) not present in the GRN: {list(absent['ranked_gene'])}")
    keep = set(summary_df.loc[~summary_df["ranked_gene"].isin(absent["ranked_gene"]), "ranked_gene"])
    summary_df = summary_df[summary_df["ranked_gene"].isin(keep)].reset_index(drop=True)
    long_df = long_df[long_df["ranked_gene"].isin(keep)].reset_index(drop=True)

    os.makedirs("results", exist_ok=True)
    src = args.label if args.label else ("dge" if args.load_dge_results else "ranked")
    stem = f"GRN_analysis_{src}_{args.data}_{args.cell_type}_{args.region}_top{args.n_genes}"
    edges_out = f"results/{stem}_edges.csv"
    summary_out = f"results/{stem}_summary.csv"
    long_df.to_csv(edges_out, index=False)
    summary_df.to_csv(summary_out, index=False)
    print(f"Wrote {len(long_df)} connections -> {edges_out}")
    print(f"Wrote per-gene summary   -> {summary_out}")
    print("\nPer-gene connection counts:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_genes", type=int, default=10)
    parser.add_argument("--data", type=str, default="SEAAD")
    parser.add_argument("--cell_type", type=str, default="Microglia")
    parser.add_argument("--region", type=str, default="MTG")
    parser.add_argument(
        "--ranked_list", type=str,
        default="ranked_lists_v2/SEAAD_Microglia_flatten_mlp_thalonly_trial25/ranked_genes.csv",
        help="CSV with a 'gene' column of Ensembl IDs, ranked best-first.",
    )
    parser.add_argument("--load_dge_results", action="store_true",
                        help="Rank genes by the DGE union instead of --ranked_list.")
    parser.add_argument("--label", type=str, default=None,
                        help="Override the source tag in the output filename (e.g. combined_min5).")
    main(parser.parse_args())
