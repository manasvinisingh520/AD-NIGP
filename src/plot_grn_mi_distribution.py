"""Plot the MI (edge-confidence) distribution of a GRN.

Left panel:  histogram of edge MI (log y), with a single dashed line at the
             MI cutoff used to filter the GRN (default 3.04).
Right panel: fraction of edges retained (>= threshold, i.e. 1-CDF) and the CDF,
             as a function of the MI cutoff.

The GRN tsv is source<TAB>target<TAB>mi with no header; only the mi column is
read (in chunks) so the ~1GB file fits comfortably in memory.
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

GRN = "data/GRN/SEAAD/Microglia/Microglia_GRN.tsv"


def load_mi(grn_path, chunksize=5_000_000):
    """Read just the mi column of the GRN into a float32 array."""
    parts = []
    for ch in pd.read_csv(grn_path, sep="\t", header=None, usecols=[2],
                          names=["mi"], dtype={"mi": np.float32}, chunksize=chunksize):
        parts.append(ch["mi"].to_numpy())
    return np.concatenate(parts)


def plot_mi_distribution(mi, cutoff=3.04, title="SEAAD Microglia GRN", save_path=None,
                         bins=250):
    n = mi.size
    kept = float((mi >= cutoff).mean())

    fig, (ax_h, ax_c) = plt.subplots(1, 2, figsize=(15, 6))

    # --- left: histogram of MI (log count) with a single cutoff line ---
    ax_h.hist(mi, bins=bins, color="#4878a8")
    ax_h.set_yscale("log")
    ax_h.set_xlabel("mutual information (edge confidence)")
    ax_h.set_ylabel("# edges (log)")
    ax_h.set_title(f"{title} — MI distribution")
    ax_h.axvline(cutoff, color="red", ls="--", lw=1.2)
    ax_h.text(cutoff, ax_h.get_ylim()[1], f" MI={cutoff:g}", color="red",
              rotation=90, va="top", ha="left", fontsize=10)

    # --- right: fraction retained (1-CDF) and CDF vs threshold ---
    xs = np.linspace(mi.min(), mi.max(), 400)
    mi_sorted = np.sort(mi)
    # for each x: fraction with mi >= x  ==  1 - (#{mi < x} / n)
    frac_kept = 1.0 - np.searchsorted(mi_sorted, xs, side="left") / n
    ax_c.plot(xs, frac_kept, color="#8e2f8e", label="fraction kept if MI>=x")
    ax_c.plot(xs, 1.0 - frac_kept, color="#2e7d32", label="fraction dropped (CDF)")
    ax_c.axvline(cutoff, color="red", ls="--", lw=1.2)
    ax_c.text(cutoff, 1.0, f" MI={cutoff:g} → {kept:.1%} kept", color="red",
              rotation=90, va="top", ha="left", fontsize=10)
    ax_c.set_xlabel("MI threshold")
    ax_c.set_ylabel("fraction of edges kept (>= threshold: 1-CDF)")
    ax_c.set_title("Fraction of edges retained vs MI cutoff")
    ax_c.legend(loc="center left")

    fig.tight_layout()
    print(f"edges={n:,}  cutoff={cutoff}  kept={kept:.2%} ({int(round(kept*n)):,} edges)")
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Plot GRN MI distribution with a single MI-cutoff label.")
    p.add_argument("--grn", default=GRN)
    p.add_argument("--cutoff", type=float, default=3.04, help="MI cutoff to label (default 3.04).")
    p.add_argument("--title", default="SEAAD Microglia GRN")
    p.add_argument("--out", default="images/GRN_MI_distribution_SEAAD_Microglia.png")
    args = p.parse_args()

    mi = load_mi(args.grn)
    plot_mi_distribution(mi, cutoff=args.cutoff, title=args.title, save_path=args.out)


if __name__ == "__main__":
    main()
