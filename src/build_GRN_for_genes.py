# Build a target cascade for one gene in the SEA-AD Microglia GRN.
# root -> its top-MI targets (level 1) -> their top-MI targets (level 2) -> ... up to `levels`
# (TF cascades saturate the network in ~4 hops). Then every GRN edge *among* the reached nodes
# is collected (with MI), so convergent links (e.g. two regulators -> same gene) show up too.
# Output: a cascade dict {root, nodes:{name:{level,type}}, edges:[{source,target,MI}]} saved as JSON.
import os
import sys
import json
import pickle
import argparse
from collections import Counter

import networkx as nx
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from GRN_analysis_rankedGenes import resolve_grn_path, load_grn_edges_for_genes

LEVEL_COLOR = {0: "#c62828", 1: "#ef9a3d", 2: "#6ea8d8", 3: "#8bc34a", 4: "#9c7bd0"}


def symbol_maps():
    """(symbol->ensembl, ensembl->symbol); empty if the mapping pickle is missing."""
    p = "data/genes_2_ensembl_ids.pkl"
    if not os.path.exists(p):
        return {}, {}
    with open(p, "rb") as f:
        g2e = pickle.load(f)          # symbol -> ensembl
    return g2e, {v: k for k, v in g2e.items()}


def top_targets(edges, gene, k):
    """Ensembl targets `gene` regulates, highest MI first; k=None -> all."""
    sub = edges[edges["source"] == gene].sort_values("mi", ascending=False)
    if k is not None:
        sub = sub.head(k)
    return list(sub["target"])


def build_cascade(root, grn, levels=4, top_k=5, mi_threshold=None):
    """Cascade dict for `root`. mi_threshold is optional -> only filters edges when given."""
    mt = mi_threshold or 0.0                          # None / 0 -> no MI filter
    g2e, e2g = symbol_maps()
    root_e = g2e.get(root, root)                      # accept a symbol or an Ensembl ID
    level_of = {root_e: 0}
    frontier = [root_e]

    # BFS: expand each frontier gene's top-k targets into the next level
    for lvl in range(1, levels + 1):
        edges = load_grn_edges_for_genes(grn, set(frontier), mi_threshold=mt)
        nxt = []
        for g in frontier:
            for t in top_targets(edges, g, top_k):
                if t not in level_of:
                    level_of[t] = lvl
                    nxt.append(t)
        frontier = nxt
        if not frontier:
            break

    # edges: every GRN edge whose source and target are both in the reached node set
    nodes = set(level_of)
    ei = load_grn_edges_for_genes(grn, nodes, mi_threshold=mt)
    ind = ei[ei["source"].isin(nodes) & ei["target"].isin(nodes)]

    name = lambda e: e2g.get(e, e)
    return {
        "root": name(root_e),
        "nodes": {name(g): {"level": lv, "type": "regulator" if lv == 0 else "gene"}
                  for g, lv in level_of.items()},
        "edges": [{"source": name(s), "target": name(t), "MI": round(float(m), 4)}
                  for s, t, m in zip(ind["source"], ind["target"], ind["mi"])],
    }


def draw_cascade(cascade, out_png):
    """Static networkx plot of a cascade dict: columns by level, edge width ~ MI."""
    G = nx.DiGraph()
    for n, d in cascade["nodes"].items():
        G.add_node(n, level=d["level"])
    for e in cascade["edges"]:
        if e["source"] != e["target"]:                 # drop self-loops
            G.add_edge(e["source"], e["target"], mi=e["MI"])
    if G.number_of_nodes() <= 1:
        print("cascade too small to plot."); return
    levels = [G.nodes[n]["level"] for n in G.nodes]
    big = G.number_of_nodes() > 80
    colors = [LEVEL_COLOR.get(l % 5, "#6ea8d8") for l in levels]
    sizes = [1600 if l == 0 else (120 if big else 500) for l in levels]
    mis = [G[u][v]["mi"] for u, v in G.edges]
    mmin, mmax = (min(mis), max(mis)) if mis else (0.0, 1.0)
    widths = [(0.2 if big else 0.6) + (1.2 if big else 2.6) * (m - mmin) / (mmax - mmin + 1e-9)
              for m in mis]
    # concentric rings by level: root at the centre, each level on an outer circle
    shells = [[n for n in G.nodes if G.nodes[n]["level"] == lv] for lv in sorted(set(levels))]
    try:
        pos = nx.shell_layout(G, nlist=shells)
    except Exception:
        pos = nx.spring_layout(G, seed=42)
    fig, ax = plt.subplots(figsize=(12, 12))
    nx.draw_networkx_edges(G, pos, ax=ax, width=widths, edge_color="#999999",
                           alpha=0.5, arrows=not big, arrowsize=9, node_size=sizes)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors, node_size=sizes,
                           edgecolors="black", linewidths=0.3 if big else 0.5)
    lab = {cascade["root"]: cascade["root"]} if big else {n: n for n in G.nodes}
    nx.draw_networkx_labels(G, pos, labels=lab, ax=ax, font_size=8,
                            font_weight="bold" if big else "normal")
    ax.set_title(f"GRN cascade: {cascade['root']}  "
                 f"({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)", fontweight="bold")
    ax.axis("off")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"saved plot -> {out_png}")


def main(args):
    grn = resolve_grn_path(args.data, args.cell_type, args.region)
    print(f"GRN: {grn}")
    top_k = args.top_k if args.top_k and args.top_k > 0 else None   # <=0 -> all targets
    cascade = build_cascade(args.gene, grn, levels=args.levels, top_k=top_k,
                            mi_threshold=args.mi_threshold)          # None unless --mi_threshold given
    if len(cascade["nodes"]) == 1:
        print(f"{cascade['root']} has no targets in the GRN (not a TF?) -- empty cascade.")
    print(f"root={cascade['root']}  nodes={len(cascade['nodes'])}  edges={len(cascade['edges'])}")
    print("nodes per level:",
          dict(sorted(Counter(v["level"] for v in cascade["nodes"].values()).items())))
    out = args.output or f"images/grn_cascades/{cascade['root']}_cascade.json"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(cascade, f, indent=2)
    print(f"saved -> {out}")
    draw_cascade(cascade, out.replace(".json", ".png"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--gene", default="SPI1", help="Gene symbol or Ensembl ID (root of the cascade).")
    p.add_argument("--data", default="SEAAD")
    p.add_argument("--cell_type", default="Microglia")
    p.add_argument("--region", default="MTG")
    p.add_argument("--levels", type=int, default=4, help="Cascade depth (TFs saturate ~4).")
    p.add_argument("--top_k", type=int, default=5,
                   help="Top-MI targets kept per gene per level (<=0 = all; huge for hubs).")
    p.add_argument("--mi_threshold", type=float, default=None,
                   help="Optional MI cutoff; only applied if given.")
    p.add_argument("--output", default=None)
    main(p.parse_args())
