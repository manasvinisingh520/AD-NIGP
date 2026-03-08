"""
Read the Astrocytes metadata Excel file and print column headers.
Used to confirm which column contains the cell ID (expected: cell_annotation).
Run from project root: python scripts/print_metadata_headers.py
"""

import os
import sys

# Allow importing config from src when run from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

try:
    import pandas as pd
except ImportError:
    print("Error: pandas is required. Install with: pip install pandas openpyxl")
    sys.exit(1)

from config import DATA_DIR

METADATA_FILENAME = "2025-11-16_Astrocytes_metadata.xlsx"
METADATA_PATH = os.path.join(os.path.dirname(__file__), "..", DATA_DIR, METADATA_FILENAME)


def main():
    if not os.path.exists(METADATA_PATH):
        print(f"Metadata file not found: {METADATA_PATH}")
        print("Please place the Astrocytes metadata xlsx at data/AtlasData/2025-11-16_Astrocytes_metadata.xlsx")
        sys.exit(1)

    print(f"Reading: {METADATA_PATH}")
    print("Sheet 0 (first sheet) columns:\n")

    df = pd.read_excel(METADATA_PATH, sheet_name=0, nrows=0, engine="openpyxl")
    headers = df.columns.tolist()

    for i, col in enumerate(headers):
        print(f"  {i}: {col!r}")

    if "cell_annotation" in headers:
        print("\n--> Cell ID column: 'cell_annotation' (index {})".format(headers.index("cell_annotation")))
    else:
        print("\n--> 'cell_annotation' not found. Check the list above for the cell ID column.")


if __name__ == "__main__":
    main()
