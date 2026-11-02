"""What fraction of a TF's GRN targets have the TF's motif nearby?

"Nearby" = inside a peak that is open in microglia, within TSS +/- WINDOW bp of the target.

Steps
  1. GRN targets : genes the filtered SEA-AD microglia GRN links to the TF.
  2. Windows     : TSS +/- WINDOW bp around each target (GENCODE v44).
  3. Open peaks  : SEA-AD peaks with microglia bigwig signal >= threshold in at least one stage.
  4. Motif scan  : look for the TF's JASPAR motif in the DNA of those peaks (MOODS, p < 1e-4).
  5. Fraction    : targets with >= 1 motif match / all targets.

Runs in WSL:
    wsl.exe -e bash -lc "source ~/fp-env/bin/activate && cd /mnt/i/sf2026 && python src/grn_target_motifs.py --tf FOXO4 --motif ~/fp/motifs/MA0848.1_FOXO4.meme"
"""
import argparse
import gzip
import os

import MOODS.scan
import MOODS.tools
import numpy as np
import pandas as pd
import pyBigWig
import pysam

FP_DIR = os.path.expanduser("~/fp")
GTF = f"{FP_DIR}/annotation/gencode.v44.basic.annotation.gtf.gz"
GENOME = f"{FP_DIR}/genome/hg38.fa"
PEAKS_BED = f"{FP_DIR}/peaks/SEAAD_MTG_ATAC_peaks.bed"
PEAK_SIGNAL_CACHE = f"{FP_DIR}/peaks/SEAAD_MTG_ATAC_peaks_microglia_signal.tsv"
GRN = "data/GRN/SEAAD/Microglia/Microglia_GRN_filtered.tsv"  # regulator, target, MI (mi >= 3.04)

# ADNC0 and ADNC1 bigwigs are byte-identical, so ADNC0 stands for "ADNC0/1"
STAGES = {"ADNC0/1": f"{FP_DIR}/bw/ADNC0Micro-PVM.bw",
          "ADNC2":   f"{FP_DIR}/bw/ADNC2Micro-PVM.bw",
          "ADNC3":   f"{FP_DIR}/bw/ADNC3Micro-PVM.bw"}
BACKGROUND = [0.295, 0.205, 0.205, 0.295]  # hg38 base composition A, C, G, T


def load_genes():
    """symbol, chrom, TSS for every GENCODE gene, keyed by Ensembl ID (no version)."""
    rows = []
    with gzip.open(GTF, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.split("\t")
            if f[2] != "gene":
                continue
            gid = f[8].split('gene_id "')[1].split('"')[0].split(".")[0]
            name = f[8].split('gene_name "')[1].split('"')[0]
            start, end = int(f[3]) - 1, int(f[4])          # GTF is 1-based; convert to 0-based
            tss = start if f[6] == "+" else end - 1        # transcription starts at the 5' end
            rows.append((gid, name, f[0], tss))
    genes = pd.DataFrame(rows, columns=["ensembl", "symbol", "chrom", "tss"])
    return genes.drop_duplicates("ensembl").set_index("ensembl")


def open_peaks(threshold):
    """SEA-AD peaks open in microglia. Per-peak signal is computed once, then cached."""
    if os.path.exists(PEAK_SIGNAL_CACHE):
        peaks = pd.read_csv(PEAK_SIGNAL_CACHE, sep="\t")
    else:
        peaks = pd.read_csv(PEAKS_BED, sep="\t", header=None, names=["chrom", "start", "end"])
        for stage, path in STAGES.items():
            bw = pyBigWig.open(path)
            peaks[stage] = [bw.stats(c, int(s), int(e), type="mean")[0] or 0.0
                            for c, s, e in zip(peaks.chrom, peaks.start, peaks.end)]
        peaks.round(4).to_csv(PEAK_SIGNAL_CACHE, sep="\t", index=False)
    return peaks[peaks[list(STAGES)].max(axis=1) >= threshold]


def read_meme(path):
    """First motif in a MEME file -> 4 x width probability matrix (rows A, C, G, T)."""
    lines = open(path).read().splitlines()
    i = next(k for k, l in enumerate(lines) if l.startswith("letter-probability"))
    width = int(lines[i].split("w=")[1].split()[0])
    probs = np.array([[float(x) for x in lines[i + 1 + r].split()] for r in range(width)])
    return probs.T.tolist()


def has_motif(seq, motif, threshold):
    """True if the motif matches the sequence on either strand."""
    fwd, rev = MOODS.scan.scan_dna(seq, motif, BACKGROUND, [threshold, threshold], 7)
    return len(fwd) + len(rev) > 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tf", required=True, help="TF gene symbol, e.g. FOXO4")
    ap.add_argument("--motif", required=True, help="JASPAR motif in MEME format")
    ap.add_argument("--window", type=int, default=50_000, help="bp either side of the target TSS")
    ap.add_argument("--open-threshold", type=float, default=0.2,
                    help="min mean signal (any stage) for a peak to count as open in microglia")
    ap.add_argument("--pvalue", type=float, default=1e-4, help="motif match p-value cutoff")
    args = ap.parse_args()

    # 1. GRN targets of the TF
    genes = load_genes()
    tf_id = genes.index[genes.symbol == args.tf][0]
    grn = pd.read_csv(GRN, sep="\t", header=None, names=["regulator", "target", "mi"])
    targets = genes[genes.index.isin(grn.loc[grn.regulator == tf_id, "target"])]

    # 3./4. motif as a log-odds score matrix, plus its reverse complement for the other strand;
    # the threshold is the score random DNA reaches with probability `pvalue`
    lo = MOODS.tools.log_odds(read_meme(os.path.expanduser(args.motif)), BACKGROUND, 0.01)
    motif = [lo, MOODS.tools.reverse_complement(lo)]
    thr = MOODS.tools.threshold_from_p(lo, BACKGROUND, args.pvalue)
    peaks = open_peaks(args.open_threshold)
    fasta = pysam.FastaFile(GENOME)

    # 2./4. for each target: open peaks inside its window -> any motif match?
    n_near = 0
    for t in targets.itertuples():
        lo_bp, hi_bp = t.tss - args.window, t.tss + args.window
        pk = peaks[(peaks.chrom == t.chrom) & (peaks.end > lo_bp) & (peaks.start < hi_bp)]
        # clip peaks to the window so a match just outside it is not counted
        if any(has_motif(fasta.fetch(t.chrom, max(p.start, lo_bp), min(p.end, hi_bp)).upper(),
                         motif, thr) for p in pk.itertuples()):
            n_near += 1

    # 5. fraction
    print(f"{args.tf}: {n_near} of {len(targets)} GRN targets have a motif match within "
          f"+/-{args.window // 1000} kb of their TSS ({n_near / len(targets):.1%})")


if __name__ == "__main__":
    main()
