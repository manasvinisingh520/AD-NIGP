"""Per-gene GRN cascade x DE analysis (SEA-AD Microglia).

For each gene in a list, build its target cascade (root -> targets -> targets-of-targets ...
up to `levels`, keeping the top-k targets by MI per node) and count how many nodes at each
level are DE in binThal and in the >=5/6 ADNC contrasts (padj + optional |log2FC| filter).

The cascade level-mapping mirrors build_GRN_for_genes.build_cascade, but runs on a prebuilt
in-memory adjacency so it is fast to repeat over a whole list of genes (no per-gene GRN scan).
"""

import os
import glob
import time
import argparse
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

GRN = "data/GRN/SEAAD/Microglia/Microglia_GRN.tsv"
GRN_GENES = "data/GRN/SEAAD/Microglia/Microglia_GRN_genes.txt"
BINTHAL = ("results/dge_final/dge_results_SEA_AD_Microglia_MTG_"
           "pseudobulk100_seed100_binThal_covbatch_vendor_name_1_vs_2.csv")
ADNC_GLOB = "results/dge_final/dge_results_SEA_AD_Microglia_MTG_*.csv"
DEFAULT_GENES = ("results/grn_analysis/no_threshold/"
                 "GRN_analysis_f1_top200_min5_SEAAD_Microglia_MTG_top75_summary.csv")


def load_or_build_nodes(grn_path, cache=GRN_GENES, chunksize=3_000_000):
    """All genes appearing as source or target in the GRN (cached)."""
    if os.path.exists(cache):
        return [ln.strip() for ln in open(cache) if ln.strip()]
    print("node cache missing -> scanning GRN ...")
    genes = set()
    for ch in pd.read_csv(grn_path, sep="\t", header=None, names=["s", "t", "mi"],
                          usecols=["s", "t"], dtype=str, chunksize=chunksize):
        genes.update(ch["s"].values); genes.update(ch["t"].values)
    genes = sorted(g for g in genes if isinstance(g, str) and g.startswith("ENSG"))
    try:
        open(cache, "w").write("\n".join(genes) + "\n")
    except OSError:
        pass
    return genes


def build_adjacency(grn_path, gene2idx, mi_threshold=None, chunksize=3_000_000):
    """adjacency[src_idx] -> (targets int32 array, mi float32 array). Only sources are keys."""
    mt = mi_threshold or 0.0
    tparts, mparts = defaultdict(list), defaultdict(list)
    t0 = time.time()
    for ch in pd.read_csv(grn_path, sep="\t", header=None, names=["s", "t", "mi"],
                          dtype={"s": str, "t": str, "mi": float}, chunksize=chunksize):
        if mt > 0:
            ch = ch[ch["mi"] >= mt]
        s = ch["s"].map(gene2idx).to_numpy()
        t = ch["t"].map(gene2idx).to_numpy()
        mi = ch["mi"].to_numpy()
        keep = ~(np.isnan(s) | np.isnan(t))
        s, t, mi = s[keep].astype(np.int32), t[keep].astype(np.int32), mi[keep].astype(np.float32)
        order = np.argsort(s, kind="stable")
        s, t, mi = s[order], t[order], mi[order]
        uniq, starts = np.unique(s, return_index=True)
        bounds = np.append(starts, len(s))
        for k, src in enumerate(uniq):
            tparts[int(src)].append(t[bounds[k]:bounds[k + 1]])
            mparts[int(src)].append(mi[bounds[k]:bounds[k + 1]])
    adj = {src: (np.concatenate(tparts[src]), np.concatenate(mparts[src])) for src in tparts}
    print(f"adjacency built: {len(adj)} sources, {time.time()-t0:.0f}s")
    return adj


def load_gene_set(path):
    df = pd.read_csv(path)
    col = "ranked_gene" if "ranked_gene" in df.columns else ("gene" if "gene" in df.columns else df.columns[0])
    return df[col].astype(str).tolist()


def binthal_de(padj_thresh=0.05, lfc_thresh=None):
    d = pd.read_csv(BINTHAL, index_col=0).dropna(subset=["padj"])
    m = d["padj"] < padj_thresh
    if lfc_thresh:
        m &= d["log2FoldChange"].abs() >= lfc_thresh
    return set(d.index[m])


def contrasts5of6_de(padj_thresh=0.05, lfc_thresh=None, n_contrasts=5):
    files = [f for f in sorted(glob.glob(ADNC_GLOB))
             if "binADNC" not in f and "binThal" not in f]
    cnt = Counter()
    for f in files:
        d = pd.read_csv(f, index_col=0).dropna(subset=["padj"])
        m = d["padj"] < padj_thresh
        if lfc_thresh:
            m &= d["log2FoldChange"].abs() >= lfc_thresh
        cnt.update(d.index[m])
    return {g for g, c in cnt.items() if c >= n_contrasts}


def cascade_levels(start, adjacency, n_nodes, levels=4, top_k=None):
    """BFS from `start`; return an int level array (-1 = unreached). At each hop keep the
    top_k targets by MI per frontier node (top_k=None -> all targets)."""
    level = np.full(n_nodes, -1, dtype=np.int16)
    level[start] = 0
    frontier = np.array([start], dtype=np.int32)
    for lvl in range(1, levels + 1):
        arrs = []
        for s in frontier.tolist():
            e = adjacency.get(s)
            if e is None:
                continue
            t, mi = e
            if top_k is not None and t.size > top_k:
                t = t[np.argpartition(mi, -top_k)[-top_k:]]
            arrs.append(t)
        if not arrs:
            break
        cand = np.unique(np.concatenate(arrs))
        new = cand[level[cand] < 0]
        if new.size == 0:
            break
        level[new] = lvl
        frontier = new
    return level


def run(args):
    genes_list = load_or_build_nodes(args.grn)
    gene2idx = {g: i for i, g in enumerate(genes_list)}
    n_nodes = len(genes_list)
    print(f"GRN nodes: {n_nodes}")
    adjacency = build_adjacency(args.grn, gene2idx, mi_threshold=args.mi_threshold)

    de_sets = {"binThal": binthal_de(args.padj, args.lfc),
               "contrasts5of6": contrasts5of6_de(args.padj, args.lfc)}
    de_masks = {k: np.array([g in v for g in genes_list]) for k, v in de_sets.items()}
    print(f"DE sets (padj<{args.padj}"
          + (f", |lfc|>={args.lfc}" if args.lfc else ", no lfc filter") + "): "
          + ", ".join(f"{k}={len(v)}" for k, v in de_sets.items()))

    top_k = args.top_k if args.top_k and args.top_k > 0 else None
    seeds = load_gene_set(args.genes)

    if args.union:
        # union of all cascades: each gene keeps the MIN level at which any seed reaches it
        glevel = np.full(n_nodes, -1, dtype=np.int16)
        built = 0
        for g in seeds:
            gi = gene2idx.get(g)
            if gi is None:
                continue
            built += 1
            lv = cascade_levels(gi, adjacency, n_nodes, args.levels, top_k)
            r = lv >= 0
            prev = glevel[r]
            glevel[r] = np.where(prev < 0, lv[r], np.minimum(prev, lv[r]))
        rows = []
        for lv in range(args.levels + 1):
            idx = np.where(glevel == lv)[0]
            row = {"level": lv, "n_unique_genes": int(idx.size)}
            for name, mask in de_masks.items():
                row[f"{name}_DE"] = int(mask[idx].sum())
            rows.append(row)
        df = pd.DataFrame(rows)
        print(f"\nseeds: {len(seeds)} genes (cascades built for {built}); "
              f"UNION, each gene at its earliest level (levels={args.levels}, top_k={top_k})")
        print("\nunique genes per level:")
        print(df.to_string(index=False))
        out = args.output
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        df.to_csv(out, index=False)
        print(f"\nsaved union table -> {out}")
        return

    rows = []
    for g in seeds:
        gi = gene2idx.get(g)
        if gi is None:
            continue
        level = cascade_levels(gi, adjacency, n_nodes, args.levels, top_k)
        for lv in range(args.levels + 1):
            idx = np.where(level == lv)[0]
            if idx.size == 0:
                continue
            row = {"gene": g, "level": lv, "n_nodes": int(idx.size)}
            for name, mask in de_masks.items():
                row[f"{name}_DE"] = int(mask[idx].sum())
            rows.append(row)
    df = pd.DataFrame(rows)
    n_in = df["gene"].nunique() if not df.empty else 0
    print(f"\nseeds: {len(seeds)} genes (cascades built for {n_in} in the GRN, "
          f"levels={args.levels}, top_k={top_k})")

    agg = (df.groupby("level")
             .agg(genes=("gene", "nunique"), n_nodes=("n_nodes", "sum"),
                  binThal_DE=("binThal_DE", "sum"), contrasts5of6_DE=("contrasts5of6_DE", "sum"))
             .reset_index())
    print("\nsummed over the gene list, per level:")
    print(agg.to_string(index=False))

    out = args.output
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nsaved per-gene x level table -> {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--grn", default=GRN)
    p.add_argument("--genes", default=DEFAULT_GENES, help="CSV with ranked_gene/gene column (seed list).")
    p.add_argument("--levels", type=int, default=4, help="Cascade depth (TFs saturate ~4).")
    p.add_argument("--top_k", type=int, default=None,
                   help="Top-MI targets kept per node per level (<=0 / omitted = all).")
    p.add_argument("--padj", type=float, default=0.05, help="padj cutoff for DE sets.")
    p.add_argument("--lfc", type=float, default=None, help="|log2FC| cutoff for DE sets (optional).")
    p.add_argument("--mi_threshold", type=float, default=None, help="Optional GRN MI cutoff.")
    p.add_argument("--union", action="store_true",
                   help="De-duplicate across seeds: count UNIQUE genes per level (each gene at "
                        "its earliest level over all cascades) instead of summing per-gene.")
    p.add_argument("--output", default="results/grn_analysis/cascade_de_by_level.csv")
    main = run
    main(p.parse_args())
