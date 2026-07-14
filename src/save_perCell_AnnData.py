"""
Create, filter, and save a per-cell AnnData for the model pipeline, for any
dataset / cell type, with standardized obs columns (SampleName, Thal, Braak).

Supported:
  Atlas  Astrocytes  – MTX + Excel metadata (per region: EC/ITG/PFC/V1/V2)
  Atlas  Microglia   – MTX + per-region CSV  (CrossRegion/ITG/PFC)
  SEAAD  Astro/Micro – data/SEA_AD_Data/<ct>/<ct>.h5ad  (counts in .raw.X)
  ROSMAP Astro/Micro – data/ROSMAP_Data/<ct>.h5ad       (counts in layers['counts'])

Output: data/model_data/<stem>_AnnData_perCell.h5ad
  Atlas: <region>[_Microglia]_AnnData_perCell.h5ad   (unchanged naming)
  SEAAD/ROSMAP: <DATA>_<CellType>_AnnData_perCell.h5ad

Usage:
  python src/save_perCell_AnnData.py -data Atlas  -cell_type Astrocytes -region EC
  python src/save_perCell_AnnData.py -data SEAAD  -cell_type Microglia
  python src/save_perCell_AnnData.py -data ROSMAP -cell_type Astrocytes
"""

import argparse
import os

import anndata as ad
import numpy as np
import pandas as pd

from utils import (
    read_mtx_file,
    create_anndata_object,
    read_excel_columns,
    get_region_file_paths,
    coerce_numeric_series,
)
from config import (
    ASTRO_REGIONS, MICRO_REGIONS, DATA_DIR, REGION_TO_TAB,
    _SEAAD_THAL, _SEAAD_BRAAK, MICROGLIA_DATE_PREFIX, ASTRO_DATE_PREFIX,
)

# DATA_DIR is Atlas-specific (e.g. data/AtlasData); SEAAD/ROSMAP live under the base data dir.
BASE_DATA_DIR = os.path.dirname(DATA_DIR) or "data"

# Where built per-cell AnnData objects are written / loaded from.
PERCELL_DIR = os.path.join("data", "model_data", "perCell_AnnData")

# h5ad-based datasets: path, counts source, patient/target columns, drop rules.
# 'thal'/'braak' = (obs_column, mapping_or_None); None mapping -> coerce numeric.
H5AD_CONFIG = {
    "SEAAD": {
        "path": lambda ct: os.path.join(BASE_DATA_DIR, "SEA_AD_Data", ct, f"{ct}.h5ad"),
        "counts": "raw",                       # X is log-normalized; counts in .raw.X
        "patient_col": "donor_id",
        "thal": ("Thal phase", _SEAAD_THAL),
        "braak": ("Braak stage", _SEAAD_BRAAK),
        "drop": [("ADNC", {"Reference"})],     # drop neurotypical Reference cells
    },
    "ROSMAP": {
        "path": lambda ct: os.path.join(BASE_DATA_DIR, "ROSMAP_Data", f"{ct}.h5ad"),
        "counts": "layer:counts",
        "patient_col": "individualID",
        # CERAD (neuritic-plaque) ~ Thal (amyloid); braaksc (tau) ~ Braak.
        "thal": ("ceradsc", None),
        "braak": ("braaksc", None),
        "drop": [],
    },
}


def _map_target(series: pd.Series, mapping):
    """String-stage -> int via `mapping`, or coerce to numeric when mapping is None."""
    if mapping is not None:
        return series.astype(str).str.strip().map(mapping)
    return coerce_numeric_series(series)


def _stringify_obs(df: pd.DataFrame) -> pd.DataFrame:
    """Cast object/category columns to str so h5ad write does not choke on mixed types."""
    for col in df.columns:
        if df[col].dtype == object or isinstance(df[col].dtype, pd.CategoricalDtype):
            df[col] = df[col].astype(str)
    return df


# ---------------------------------------------------------------------------
# h5ad datasets (SEAAD, ROSMAP)
# ---------------------------------------------------------------------------
def _load_h5ad_perCell(data: str, cell_type: str) -> ad.AnnData:
    cfg = H5AD_CONFIG[data]
    path = cfg["path"](cell_type)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{data} {cell_type} AnnData not found: {path}")
    print(f"Reading {path} ...")
    a = ad.read_h5ad(path)

    # counts source
    src = cfg["counts"]
    if src == "raw":
        if a.raw is None:
            raise ValueError(f"{data}: expected raw counts in .raw.X but .raw is None")
        counts, var = a.raw.X, a.raw.var.copy()
    elif src.startswith("layer:"):
        layer = src.split(":", 1)[1]
        if layer not in a.layers:
            raise ValueError(f"{data}: layer '{layer}' not found. Have: {list(a.layers.keys())}")
        counts, var = a.layers[layer], a.var.copy()
    else:
        counts, var = a.X, a.var.copy()

    b = ad.AnnData(X=counts, obs=a.obs.copy(), var=var)

    # standardized target columns
    b.obs["SampleName"] = b.obs[cfg["patient_col"]].astype(str)
    b.obs["Thal"] = _map_target(b.obs[cfg["thal"][0]], cfg["thal"][1])
    if cfg["braak"]:
        b.obs["Braak"] = _map_target(b.obs[cfg["braak"][0]], cfg["braak"][1])

    # drop invalid / reference cells
    keep = b.obs["Thal"].notna()
    for col, bad in cfg["drop"]:
        keep &= ~b.obs[col].astype(str).isin(bad)
    n_drop = int((~keep).sum())
    b = b[keep.values].copy()
    b.obs["Thal"] = b.obs["Thal"].astype(int)
    if "Braak" in b.obs:
        b.obs["Braak"] = _map_target(b.obs["Braak"], None)  # ensure numeric; NaNs allowed
    b.obs = _stringify_obs(b.obs)
    print(f"  {b.n_obs} cells (dropped {n_drop}), {b.obs['SampleName'].nunique()} donors")
    return b


# ---------------------------------------------------------------------------
# Atlas (MTX + external metadata) — Astrocytes (Excel) / Microglia (CSV)
# ---------------------------------------------------------------------------
def _load_metadata_microglia(region, cell_names):
    """Load Atlas Microglia per-region CSV and align to MTX cell order."""
    metadata_path = os.path.join(DATA_DIR, f"{MICROGLIA_DATE_PREFIX}_Microglia_{region}_metadata.csv")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Microglia metadata not found: {metadata_path}")
    metadata = pd.read_csv(metadata_path)
    rename = {}
    if "Donor ID" in metadata.columns:
        rename["Donor ID"] = "Donor_ID"
    if "Pathology Stage" in metadata.columns:
        rename["Pathology Stage"] = "Pathology_Stage"
    if rename:
        metadata = metadata.rename(columns=rename)
    if "SampleName" not in metadata.columns and "Donor_ID" in metadata.columns:
        metadata["SampleName"] = metadata["Donor_ID"].astype(str)
    if "Thal" not in metadata.columns:
        for c in ["CERAD", "thal", "THAL", "Thal_phase"]:
            if c in metadata.columns:
                metadata["Thal"] = metadata[c].astype(str).str.strip()
                break
        if "Thal" not in metadata.columns:
            raise ValueError(f"Microglia metadata must have 'Thal' or 'CERAD'. Found: {list(metadata.columns)}")
    if "Ptau.Total.Tau..A.U.." not in metadata.columns:
        for c in ["Braak", "ptau", "Ptau", "Ptau.Total.Tau..A.U..", "Ptau_Total_Tau_A_U"]:
            if c not in metadata.columns:
                continue
            if c == "Braak":
                braak_map = {"I": 2, "II": 3, "V": 5, "VI": 6}
                s = metadata["Braak"].astype(str).str.strip().str.upper()
                metadata["Ptau.Total.Tau..A.U.."] = pd.to_numeric(s.replace(braak_map), errors="coerce")
                break
            metadata["Ptau.Total.Tau..A.U.."] = pd.to_numeric(metadata[c], errors="coerce")
            break
        if "Ptau.Total.Tau..A.U.." not in metadata.columns:
            raise ValueError(f"Microglia metadata must have 'Ptau.Total.Tau..A.U..' or 'Braak'. Found: {list(metadata.columns)}")
    cell_col = None
    for col in ["cell_annotation", "Cell", "Unnamed: 0", "cell"]:
        if col in metadata.columns and metadata[col].isin(cell_names).all():
            cell_col = col
            break
    if cell_col is not None:
        metadata = metadata.set_index(cell_col).reindex(cell_names)
    elif len(metadata) == len(cell_names):
        metadata = metadata.copy()
        metadata.index = cell_names
    else:
        raise ValueError(
            f"Microglia metadata rows ({len(metadata)}) must match cell count ({len(cell_names)})."
        )
    valid = metadata["Thal"].notna() & (metadata["Thal"].astype(str).str.strip().str.lower() != "unk")
    metadata = metadata[valid].copy()
    return _stringify_obs(metadata)


def _load_atlas_perCell(region, cell_type):
    """Build an Atlas per-cell AnnData from MTX + external metadata.

    This is the single source-of-truth builder for Atlas per-cell objects; the model
    pipeline (create_data.create_anndata) delegates here rather than duplicating it.
    """
    is_micro = cell_type == "Microglia"
    date_prefix = MICROGLIA_DATE_PREFIX if is_micro else ASTRO_DATE_PREFIX
    base_prefix = f"{date_prefix}_Microglia_{region}" if is_micro else f"{date_prefix}_Astrocytes_{region}"
    mtx_path, row_annotation_path, col_annotation_path = get_region_file_paths(
        region, data_dir=DATA_DIR, base_prefix=base_prefix
    )
    matrix, gene_names, cell_names = read_mtx_file(
        mtx_path=str(mtx_path), row_annotation_path=str(row_annotation_path),
        col_annotation_path=str(col_annotation_path), transpose=False,
    )
    print(f"Loaded matrix shape: {matrix.shape} (genes × cells)")

    if is_micro:
        filtered_metadata = _load_metadata_microglia(region, cell_names)
        valid = set(filtered_metadata.index.tolist())
        idx = [i for i, c in enumerate(cell_names) if c in valid]
    else:
        metadata = read_excel_columns(
            os.path.join(DATA_DIR, f"{date_prefix}_Astrocytes_metadata.xlsx"),
            columns=['cell_annotation', 'Path..Group.', 'SampleName', 'percent.mito',
                     'RIN', 'Total.Genes.Detected', 'Thal', 'Ptau.Total.Tau..A.U..'],
            sheet_name=REGION_TO_TAB.get(region),
        )
        valid_mask = (metadata["Thal"] != "unk").fillna(False)
        valid = set(metadata.loc[valid_mask, "cell_annotation"].tolist())
        idx = [i for i, c in enumerate(cell_names) if c in valid]
    cell_names = [cell_names[i] for i in idx]
    matrix = matrix[:, idx]
    if not is_micro:
        filtered_metadata = _stringify_obs(
            metadata[metadata["cell_annotation"].isin(cell_names)]
            .set_index("cell_annotation").reindex(cell_names)
        )

    adata = create_anndata_object(
        matrix=matrix, gene_names=gene_names, cell_names=cell_names,
        transpose=True, obs=filtered_metadata,
    )
    if adata is None:
        raise ImportError("Failed to create AnnData object.")
    return adata


# ---------------------------------------------------------------------------
def _percell_stem(data, cell_type, region):
    """Output/file stem for a dataset/cell-type (Atlas keeps legacy per-region naming)."""
    if data == "Atlas":
        return f"{region}_Microglia" if cell_type == "Microglia" else region
    return f"{data}_{cell_type}"


def build_anndata(data, cell_type, region=None):
    """Single source-of-truth builder: return a standardized per-cell AnnData
    (obs has SampleName / Thal / Braak) for any dataset & cell type.

    Consumed by create_data / create_data_v2 so the build logic lives in one place.
    """
    if data == "Atlas":
        return _load_atlas_perCell(region, cell_type)
    return _load_h5ad_perCell(data, cell_type)


def percell_path(data, cell_type, region=None):
    """Path to the pre-built per-cell AnnData for a dataset/cell type."""
    return os.path.join(PERCELL_DIR, f"{_percell_stem(data, cell_type, region)}_AnnData_perCell.h5ad")


def load_percell_anndata(data, cell_type, region=None):
    """Load a pre-built per-cell AnnData (produced by build/save). Fast path for the
    model pipeline — no rebuild from source."""
    path = percell_path(data, cell_type, region)
    if not os.path.exists(path):
        cmd = f"python src/save_perCell_AnnData.py -data {data} -cell_type {cell_type}"
        if data == "Atlas":
            cmd += f" -region {region}"
        raise FileNotFoundError(f"Per-cell AnnData not found: {path}\nBuild it with:\n  {cmd}")
    print(f"Loading per-cell AnnData: {path}")
    return ad.read_h5ad(path)


def create_and_save_anndata(data, cell_type, region=None, output_dir=PERCELL_DIR):
    print(f"\n{'='*60}\nProcessing: data={data} cell_type={cell_type} region={region}\n{'='*60}")

    adata = build_anndata(data, cell_type, region)
    stem = _percell_stem(data, cell_type, region)

    print(f"AnnData created: {adata.shape} (cells × genes)")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{stem}_AnnData_perCell.h5ad")
    adata.write_h5ad(output_path)
    print(f"Saved -> {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Create and save a per-cell AnnData for any dataset / cell type.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-data", type=str, default="Atlas", choices=["Atlas", "SEAAD", "ROSMAP"],
                        help="Dataset (default: Atlas).")
    parser.add_argument("-cell_type", type=str, default="Astrocytes",
                        choices=["Astrocytes", "Microglia"], help="Cell type (default: Astrocytes).")
    parser.add_argument("-region", type=str, default=None,
                        help="Region — required for Atlas (EC/ITG/PFC/V1/V2 or CrossRegion for Microglia); "
                             "ignored for SEAAD/ROSMAP.")
    parser.add_argument("-output_dir", type=str, default=PERCELL_DIR,
                        help=f"Output directory for the .h5ad file (default: {PERCELL_DIR}).")
    args = parser.parse_args()

    if args.data == "Atlas":
        if not args.region:
            parser.error("-region is required for -data Atlas")
        if args.cell_type == "Astrocytes" and args.region not in ASTRO_REGIONS:
            parser.error(f"Atlas Astrocytes region must be one of {ASTRO_REGIONS}")
        if args.cell_type == "Microglia" and args.region not in MICRO_REGIONS:
            parser.error(f"Atlas Microglia region must be one of {MICRO_REGIONS}")

    create_and_save_anndata(
        data=args.data, cell_type=args.cell_type, region=args.region, output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
