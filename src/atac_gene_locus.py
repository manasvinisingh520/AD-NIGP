"""Look at chromatin accessibility (ATAC peaks) around one gene, across AD stages.

Data: SEA-AD MTG snATAC-seq, microglia pseudobulk signal (one bigwig per ADNC stage).
Question: which open-chromatin regions (peaks) sit in/around the gene, and does
their openness change from low (ADNC0/1) to high (ADNC3) AD pathology?

Runs in WSL (pyBigWig is Linux-only; data lives on WSL disk under ~/fp):
    wsl.exe -e bash -lc "source ~/fp-env/bin/activate && cd /mnt/i/sf2026 && python src/atac_gene_locus.py --gene TLR2"

Outputs (in results/atac_locus/):
    <GENE>_peaks.csv   one row per peak: location, type, signal per stage, log2 change
    <GENE>_locus.png   signal tracks for each stage, with peaks and the gene drawn underneath
"""
import argparse
import gzip
import os
import re

import matplotlib
matplotlib.use("Agg")  # no display in WSL; write straight to file
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
import pyBigWig

FP_DIR = os.path.expanduser("~/fp")
GTF = f"{FP_DIR}/annotation/gencode.v44.basic.annotation.gtf.gz"
PEAKS_BED = f"{FP_DIR}/peaks/SEAAD_MTG_ATAC_peaks.bed"

# ADNC0 and ADNC1 bigwigs from SEA-AD are byte-identical (same md5), so there are only
# 3 distinct tracks. We use the ADNC0 file and label it "ADNC0/1".
STAGES = {
    "ADNC0/1": f"{FP_DIR}/bw/ADNC0Micro-PVM.bw",
    "ADNC2":   f"{FP_DIR}/bw/ADNC2Micro-PVM.bw",
    "ADNC3":   f"{FP_DIR}/bw/ADNC3Micro-PVM.bw",
}
PROMOTER_BP = 2000  # a peak within +/-2 kb of the TSS is called a promoter peak


# ---------------------------------------------------------------- 1. find the gene
def find_gene(gene_name):
    """Return chrom, start, end, strand, tss for a gene from the GENCODE GTF.

    GTF coordinates are 1-based inclusive; we convert to 0-based half-open (BED style)
    so they line up with the peak file and the bigwigs.
    """
    pattern = f'gene_name "{gene_name}";'
    with gzip.open(GTF, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.split("\t")
            if f[2] == "gene" and pattern in f[8]:
                chrom, start, end, strand = f[0], int(f[3]) - 1, int(f[4]), f[6]
                # the TSS is where transcription starts: left end on +, right end on -
                tss = start if strand == "+" else end - 1
                gene_id = re.search(r'gene_id "([^"]+)"', f[8]).group(1)
                return dict(chrom=chrom, start=start, end=end, strand=strand,
                            tss=tss, gene_id=gene_id)
    raise ValueError(f"{gene_name} not found in {GTF}")


# ---------------------------------------------------------------- 2. peaks near the gene
def peaks_in_window(chrom, win_start, win_end):
    """All SEA-AD peaks that overlap [win_start, win_end) on chrom."""
    peaks = pd.read_csv(PEAKS_BED, sep="\t", header=None, names=["chrom", "start", "end"])
    hit = peaks[(peaks.chrom == chrom) & (peaks.end > win_start) & (peaks.start < win_end)]
    return hit.reset_index(drop=True)


def classify_peak(row, gene):
    """Label a peak by where it sits relative to the gene.

    promoter  : within +/-PROMOTER_BP of the TSS (the gene's on-switch)
    gene body : inside the gene but not at the promoter (often intronic enhancers)
    upstream / downstream : outside the gene (distal enhancers), relative to gene direction
    """
    center = (row.start + row.end) // 2
    if abs(center - gene["tss"]) <= PROMOTER_BP or (row.start <= gene["tss"] < row.end):
        return "promoter"
    if row.end > gene["start"] and row.start < gene["end"]:
        return "gene body"
    left_of_gene = center < gene["start"]
    # on the + strand, "upstream" is to the left; on the - strand it is to the right
    return "upstream" if left_of_gene == (gene["strand"] == "+") else "downstream"


# ---------------------------------------------------------------- 3. signal per peak per stage
def peak_signal(peaks, bws):
    """Mean bigwig signal over each peak, for each stage.

    Mean (not sum) so wide and narrow peaks are on the same scale.
    """
    for stage, bw in bws.items():
        vals = []
        for p in peaks.itertuples():
            v = bw.stats(p.chrom, int(p.start), int(p.end), type="mean")[0]
            vals.append(0.0 if v is None else v)  # None = no signal recorded there
        peaks[stage] = vals
    # log2 fold change high vs low pathology; +0.01 avoids dividing by zero
    peaks["log2FC_ADNC3_vs_0/1"] = np.log2((peaks["ADNC3"] + 0.01) / (peaks["ADNC0/1"] + 0.01))
    return peaks


# ---------------------------------------------------------------- 4. plot
def plot_locus(gene_name, gene, peaks, bws, win_start, win_end, out_png, n_bins=1500):
    """One signal track per stage (same y-axis so heights are comparable),
    peaks shaded, and the gene drawn as an arrow at the bottom."""
    fig, axes = plt.subplots(len(bws) + 1, 1, figsize=(13, 2 * len(bws) + 1.6), sharex=True,
                             gridspec_kw={"height_ratios": [3] * len(bws) + [1]})
    xs = np.linspace(win_start, win_end, n_bins)
    tracks = {}
    for stage, bw in bws.items():
        # averaging into n_bins keeps the plot light while showing the shape of the signal
        v = bw.stats(gene["chrom"], win_start, win_end, type="mean", nBins=n_bins)
        tracks[stage] = np.array([0.0 if x is None else x for x in v])
    ymax = max(t.max() for t in tracks.values()) * 1.05 or 1

    colors = {"ADNC0/1": "#4C72B0", "ADNC2": "#DD8452", "ADNC3": "#C44E52"}
    for ax, (stage, sig) in zip(axes, tracks.items()):
        ax.fill_between(xs, sig, color=colors[stage], lw=0)
        for p in peaks.itertuples():
            ax.axvspan(p.start, p.end, color="grey", alpha=0.15, lw=0)
        ax.set_ylim(0, ymax)
        ax.set_ylabel(stage, rotation=0, ha="right", va="center")
        ax.spines[["top", "right"]].set_visible(False)

    # gene track: line for the gene body, arrow for direction, tick at the TSS
    ag = axes[-1]
    ag.plot([gene["start"], gene["end"]], [0, 0], color="black", lw=6, solid_capstyle="butt")
    dx = (gene["end"] - gene["start"]) * 0.25 * (1 if gene["strand"] == "+" else -1)
    ag.annotate("", xy=(gene["tss"] + dx, 0.55), xytext=(gene["tss"], 0.55),
                arrowprops=dict(arrowstyle="->", lw=1.5))
    ag.plot([gene["tss"], gene["tss"]], [0, 0.55], color="black", lw=1.5)
    ag.text((gene["start"] + gene["end"]) / 2, -0.7, f"{gene_name} ({gene['strand']})",
            ha="center", va="top", fontsize=10, style="italic")
    ag.set_ylim(-1.5, 1.2)
    ag.set_yticks([])
    ag.spines[["top", "right", "left"]].set_visible(False)
    ag.set_xlim(win_start, win_end)
    ag.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x / 1e6:.3f}"))
    ag.set_xlabel(f"{gene['chrom']} position (Mb)")

    axes[0].set_title(f"{gene_name}: SEA-AD MTG microglia ATAC signal by ADNC stage "
                      f"(grey = peaks, n={len(peaks)})", loc="left")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--gene", required=True, help="gene symbol, e.g. TLR2")
    ap.add_argument("--flank", type=int, default=50_000,
                    help="bp to look either side of the gene (default 50 kb)")
    ap.add_argument("--out-dir", default="results/atac_locus")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    gene = find_gene(args.gene)
    win_start = max(0, gene["start"] - args.flank)
    win_end = gene["end"] + args.flank
    print(f"{args.gene} ({gene['gene_id']}): {gene['chrom']}:{gene['start']:,}-{gene['end']:,} "
          f"({gene['strand']}), TSS {gene['tss']:,}")
    print(f"window: {gene['chrom']}:{win_start:,}-{win_end:,} ({(win_end - win_start) / 1000:.0f} kb)")

    peaks = peaks_in_window(gene["chrom"], win_start, win_end)
    peaks["width"] = peaks.end - peaks.start
    peaks["type"] = [classify_peak(r, gene) for r in peaks.itertuples()]
    peaks["dist_to_TSS"] = (peaks.start + peaks.end) // 2 - gene["tss"]
    print(f"peaks in window: {len(peaks)}  ({peaks['type'].value_counts().to_dict()})")

    bws = {s: pyBigWig.open(p) for s, p in STAGES.items()}
    peaks = peak_signal(peaks, bws)

    out_csv = f"{args.out_dir}/{args.gene}_peaks.csv"
    peaks.round(4).to_csv(out_csv, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(peaks.round(3).to_string(index=False))

    out_png = f"{args.out_dir}/{args.gene}_locus.png"
    plot_locus(args.gene, gene, peaks, bws, win_start, win_end, out_png)
    print(f"\nsaved {out_csv}\nsaved {out_png}")


if __name__ == "__main__":
    main()
