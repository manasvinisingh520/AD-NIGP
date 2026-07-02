"""
Compute the DOMINANT cell subtype (cell_type_high_resolution) per ROSMAP pseudobulk bin.

A pseudobulk bin mixes ~cells_per_bin cells, so its stored subtype is just the first
cell's. This replays the exact binning from aggregate_cells_into_pseudobulk (contiguous,
in-order chunks per patient; the number of bins per patient is read from the pseudobulk
_sample_ids) and majority-votes the subtype in each bin.

Writes a sidecar CSV: data/GRN/ROSMAP/<CellType>/<CellType>_bin_dominant_subtype.csv
with columns: _sample_id, dominant_subtype, dominant_frac, n_cells, n_subtypes.

Usage:
  python src/build_bin_dominant_subtype.py --cell-type Microglia
"""
import argparse
import math
import os

import anndata as ad
import pandas as pd

SUBTYPE_COL = "cell_type_high_resolution"
ID_COL = "individualID"


def build(cell_type):
    raw_path = f"data/ROSMAP_Data/{cell_type}.h5ad"
    pb_path = f"data/GRN/ROSMAP/{cell_type}/{cell_type}_pseudobulk_AnnData.h5ad"
    out = f"data/GRN/ROSMAP/{cell_type}/{cell_type}_bin_dominant_subtype.csv"

    # Bins per patient, in pseudobulk (== raw first-appearance) patient order
    pb = ad.read_h5ad(pb_path, backed="r").obs
    sid = pb["_sample_id"].astype(str)
    patient = sid.str.rsplit("_bin", n=1).str[0]
    binno = sid.str.rsplit("_bin", n=1).str[1].astype(int)
    num_bins = binno.groupby(patient).max()
    patient_order = list(dict.fromkeys(patient.tolist()))
    print(f"{cell_type}: {len(pb)} bins across {len(patient_order)} patients")

    # Raw per-cell subtypes, grouped by patient preserving original row order
    raw = ad.read_h5ad(raw_path, backed="r").obs[[ID_COL, SUBTYPE_COL]].copy()
    raw[ID_COL] = raw[ID_COL].astype(str)
    raw[SUBTYPE_COL] = raw[SUBTYPE_COL].astype(str)
    groups = {p: g[SUBTYPE_COL].tolist() for p, g in raw.groupby(ID_COL, sort=False)}

    rows, mismatches = [], 0
    for p in patient_order:
        nb = int(num_bins[p])
        cells = groups.get(p, [])
        n = len(cells)
        # sanity: replayed bin count should match ceil(n / cells_per_bin) for some cpb
        if nb and n and math.ceil(n / math.ceil(n / nb)) != nb:
            mismatches += 1
        base, rem, idx = (n // nb, n % nb, 0) if nb else (0, 0, 0)
        for b in range(nb):
            size = base + (1 if b < rem else 0)
            chunk = cells[idx:idx + size]
            idx += size
            vc = pd.Series(chunk).value_counts()
            rows.append({
                "_sample_id": f"{p}_bin{b + 1}",
                "dominant_subtype": vc.index[0] if len(vc) else "NA",
                "dominant_frac": round(vc.iloc[0] / len(chunk), 3) if chunk else float("nan"),
                "n_cells": len(chunk),
                "n_subtypes": int(vc.size),
            })

    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    covered = set(df["_sample_id"]) >= set(sid)
    print(f"Saved {out}: {len(df)} bins | subtypes={df['dominant_subtype'].nunique()} "
          f"| covers all pseudobulk bins={covered} | patient-binning mismatches={mismatches}")
    print(f"median dominant fraction: {df['dominant_frac'].median():.2f}")
    print(df['dominant_subtype'].value_counts().to_string())
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cell-type", default="Microglia")
    build(ap.parse_args().cell_type)


if __name__ == "__main__":
    main()
