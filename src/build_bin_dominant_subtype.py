"""
Compute the DOMINANT cell subtype per pseudobulk bin (ROSMAP or SEA-AD).

A pseudobulk bin mixes ~cells_per_bin cells, so its stored subtype is just the first
cell's. This replays the exact binning from aggregate_cells_into_pseudobulk (contiguous,
in-order chunks per patient; the number of bins per patient is read from the pseudobulk
_sample_ids) and majority-votes the subtype in each bin.

Datasets (config-driven, --dataset):
  ROSMAP: raw data/ROSMAP_Data/<ct>.h5ad, id=individualID, subtype=cell_type_high_resolution,
          pseudobulk data/GRN/ROSMAP/<ct>/... , no cell pre-filter.
  SEAAD:  raw data/SEA_AD_Data/<ct>/<ct>.h5ad, id=donor_id, subtype=Supertype,
          pseudobulk data/GRN/SEAAD/<ct>/... , keep only cells whose ADNC maps to 1-4
          (the same filter the GRN builder applied before pseudobulking).

Writes a sidecar CSV: data/GRN/<DATASET>/<CellType>/<CellType>_bin_dominant_subtype.csv
with columns: _sample_id, dominant_subtype, dominant_frac, n_cells, n_subtypes, <per-subtype fractions>.

Usage:
  python src/build_bin_dominant_subtype.py --dataset ROSMAP --cell-type Microglia
  python src/build_bin_dominant_subtype.py --dataset SEAAD  --cell-type Astrocytes
"""
import argparse
import math
import os

import anndata as ad
import pandas as pd

# ADNC categories the SEA-AD GRN builder keeps (Reference/Neurotypical dropped).
SEAAD_ADNC_KEEP = {"Not AD", "Low", "Intermediate", "High"}

DATASETS = {
    "ROSMAP": {
        "raw": "data/ROSMAP_Data/{ct}.h5ad",
        "grn": "data/GRN/ROSMAP/{ct}",
        "id_col": "individualID",
        "subtype_col": "cell_type_high_resolution",
        "adnc_filter": False,
    },
    "SEAAD": {
        "raw": "data/SEA_AD_Data/{ct}/{ct}.h5ad",
        "grn": "data/GRN/SEAAD/{ct}",
        "id_col": "donor_id",
        "subtype_col": "Supertype",
        "adnc_filter": True,
    },
}


def build(dataset, cell_type):
    cfg = DATASETS[dataset]
    ct = cell_type
    raw_path = cfg["raw"].format(ct=ct)
    grn_dir = cfg["grn"].format(ct=ct)
    pb_path = os.path.join(grn_dir, f"{ct}_pseudobulk_AnnData.h5ad")
    out = os.path.join(grn_dir, f"{ct}_bin_dominant_subtype.csv")
    id_col, subtype_col = cfg["id_col"], cfg["subtype_col"]

    # Bins per patient, in pseudobulk (== raw first-appearance) patient order
    pb = ad.read_h5ad(pb_path, backed="r").obs
    sid = pb["_sample_id"].astype(str)
    patient = sid.str.rsplit("_bin", n=1).str[0]
    binno = sid.str.rsplit("_bin", n=1).str[1].astype(int)
    num_bins = binno.groupby(patient).max()
    patient_order = list(dict.fromkeys(patient.tolist()))
    print(f"{dataset} {ct}: {len(pb)} bins across {len(patient_order)} patients")

    # Raw per-cell subtypes, grouped by patient preserving original row order.
    cols = [id_col, subtype_col] + (["ADNC"] if cfg["adnc_filter"] else [])
    raw = ad.read_h5ad(raw_path, backed="r").obs[cols].copy()
    if cfg["adnc_filter"]:
        keep = raw["ADNC"].astype(str).isin(SEAAD_ADNC_KEEP)
        dropped = len(raw) - int(keep.sum())
        raw = raw[keep]
        print(f"  ADNC filter: kept {len(raw)} cells, dropped {dropped} (Reference/unmapped)")
    raw[id_col] = raw[id_col].astype(str)
    raw[subtype_col] = raw[subtype_col].astype(str)
    groups = {p: g[subtype_col].tolist() for p, g in raw.groupby(id_col, sort=False)}

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
            row = {
                "_sample_id": f"{p}_bin{b + 1}",
                "dominant_subtype": vc.index[0] if len(vc) else "NA",
                "dominant_frac": round(vc.iloc[0] / len(chunk), 3) if chunk else float("nan"),
                "n_cells": len(chunk),
                "n_subtypes": int(vc.size),
            }
            # per-subtype fraction (composition), one column per subtype
            for s, c in vc.items():
                row[s] = round(c / len(chunk), 4)
            rows.append(row)

    meta = ["_sample_id", "dominant_subtype", "dominant_frac", "n_cells", "n_subtypes"]
    df = pd.DataFrame(rows)
    sub_cols = sorted(c for c in df.columns if c not in meta)
    df[sub_cols] = df[sub_cols].fillna(0.0)
    df = df[meta + sub_cols]
    df.to_csv(out, index=False)
    covered = set(df["_sample_id"]) >= set(sid)
    print(f"Saved {out}: {len(df)} bins | subtypes={len(sub_cols)} {sub_cols} "
          f"| covers all pseudobulk bins={covered} | patient-binning mismatches={mismatches}")
    print(f"median dominant fraction: {df['dominant_frac'].median():.2f}")
    print(df['dominant_subtype'].value_counts().to_string())
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="ROSMAP", choices=list(DATASETS.keys()),
                    help="Dataset (default ROSMAP).")
    ap.add_argument("--cell-type", default="Microglia")
    args = ap.parse_args()
    build(args.dataset, args.cell_type)


if __name__ == "__main__":
    main()
