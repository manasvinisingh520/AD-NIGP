"""
Print the "Subclass" column from a SEA_AD AnnData file: unique values and counts.

Usage:
  python src/print_subclass_counts.py
  python src/print_subclass_counts.py path/to/file.h5ad
  python src/print_subclass_counts.py data/SEA_AD_Data/MTG
"""

import argparse
import sys
from pathlib import Path

import anndata as ad

# Default: first non-reference h5ad in MTG dir
DATA_DIR = Path("data/SEA_AD_Data/new_MTG")


def main():
    parser = argparse.ArgumentParser(
        description="Print Subclass column unique values and counts from an AnnData file."
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=None,
        help="Path to .h5ad file or directory containing .h5ad files (default: first file in data/SEA_AD_Data/MTG)",
    )
    args = parser.parse_args()

    path = Path(args.path) if args.path else DATA_DIR
    if path.is_dir():
        files = sorted(
            p for p in path.glob("*.h5ad")
            if "reference" not in p.name.lower()
        )
        if not files:
            print(f"No non-reference .h5ad files in {path}")
            sys.exit(1)
        path = files[0]
        print(f"Using first file: {path}\n")

    if not path.exists():
        print(f"Not found: {path}")
        sys.exit(1)

    # Use backed mode to avoid loading X/layers (can be tens of GB); we only need obs
    adata = ad.read_h5ad(path, backed="r")
    try:
        if "Subclass" not in adata.obs.columns:
            print(f"Column 'Subclass' not found in adata.obs.")
            print(f"Available columns: {list(adata.obs.columns)}")
            sys.exit(1)

        counts = adata.obs["Subclass"].value_counts()
        print("Subclass — unique values and counts:")
        print(counts.to_string())
        print(f"\nTotal cells: {adata.n_obs}")
        print(f"Unique Subclass values: {counts.shape[0]}")
    finally:
        if getattr(adata, "file", None) is not None:
            adata.file.close()


if __name__ == "__main__":
    main()
