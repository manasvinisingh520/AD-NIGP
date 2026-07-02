"""
Script to map gene symbols to Ensembl IDs using genes_2_ensembl_ids.pkl and update GRN.tsv, matrix.csv, and AnnData.h5ad files.

Atlas: per-region files under data/GRN/<CellType>/ (requires -region).
ROSMAP: GRN inputs under data/GRN/ROSMAP/<CellType>/ plus raw data/ROSMAP_Data/<CellType>.h5ad.
"""

import pandas as pd
import os
import argparse
import anndata as ad
import pickle
from tqdm import tqdm
from utils import get_grn_file_paths, get_rosmap_file_paths
from config import ROSMAP_CELL_TYPES


def load_mapping(mapping_file=None):
    """Load gene symbol to Ensembl ID mapping from genes_2_ensembl_ids.pkl."""
    if mapping_file is None:
        mapping_file = os.path.join('data', 'genes_2_ensembl_ids.pkl')
    
    if not os.path.exists(mapping_file):
        raise FileNotFoundError(f"Mapping file not found: {mapping_file}")
    
    print(f"Loading Ensembl ID mapping from: {mapping_file}")
    with open(mapping_file, 'rb') as f:
        mapping = pickle.load(f)
    
    print(f"  Loaded {len(mapping)} total gene mappings from pickle file")
    
    return mapping


def process_grn_file(grn_file, output_file, mapping):
    """Process GRN file by mapping TF and Target symbols to Ensembl IDs in one pass.
    
    Assumes GRN file is tab-separated: TF\tTarget\tScore
    """
    print(f"Processing GRN file: {grn_file}")
    print(f"File size: {os.path.getsize(grn_file) / (1024 * 1024):.2f} MB")
    
    stats = {'tf_mapped': 0, 'target_mapped': 0, 'total': 0, 'tf_unmapped': set(), 'target_unmapped': set()}
    
    with open(grn_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line in tqdm(infile, desc="Processing GRN", unit="lines"):
            line = line.rstrip('\n\r')
            if not line:
                continue

            parts = line.split('\t')
            if len(parts) < 2:
                continue

            original_tf = parts[0].strip()
            second = parts[1].strip()
            score = parts[2].strip() if len(parts) > 2 else ''

            # If we only got 2 columns but the second contains a space, it may be "Target Score"
            # (some GRN files use space between target and score instead of tab)
            if not score and ' ' in second:
                tok = second.rsplit(None, 1)
                if len(tok) == 2:
                    original_target, score = tok[0], tok[1]
                else:
                    original_target = second
            else:
                original_target = second

            # Map to Ensembl IDs
            tf = mapping.get(original_tf, original_tf)
            target = mapping.get(original_target, original_target) if original_target else ''
            
            # Track statistics
            stats['tf_mapped'] += int(tf != original_tf)
            stats['target_mapped'] += int(target != original_target and bool(original_target))
            stats['total'] += 1
            if original_tf not in mapping:
                stats['tf_unmapped'].add(original_tf)
            if original_target and original_target not in mapping:
                stats['target_unmapped'].add(original_target)
            
            outfile.write(f"{tf}\t{target}\t{score}\n" if score else f"{tf}\t{target}\n")
    
    print(f"Processed {stats['total']} rows")
    print(f"TF mapping: {stats['tf_mapped']} instances mapped, {len(stats['tf_unmapped'])} unique unmapped symbols")
    print(f"Target mapping: {stats['target_mapped']} instances mapped, {len(stats['target_unmapped'])} unique unmapped symbols")
    print(f"Successfully saved to: {output_file}")


def process_matrix_file(matrix_file, output_file, mapping):
    """Process matrix file by mapping gene symbols to Ensembl IDs in one pass."""
    print(f"Processing matrix file: {matrix_file}")
    file_size = os.path.getsize(matrix_file) / (1024 * 1024)
    print(f"File size: {file_size:.2f} MB")
    
    chunk_size = 10000
    chunks = []
    mapped_count = 0
    unmapped_symbols = set()
    
    def map_symbol(symbol):
        nonlocal mapped_count, unmapped_symbols
        if symbol in mapping:
            mapped_count += 1
            return mapping[symbol]
        else:
            unmapped_symbols.add(symbol)
            return symbol  # Keep original if not in mapping
    
    chunk_iter = pd.read_csv(matrix_file, header=None, chunksize=chunk_size)
    with tqdm(desc="Processing matrix", unit="chunk") as pbar:
        for chunk in chunk_iter:
            # Replace first column with mapped values
            chunk.iloc[:, 0] = chunk.iloc[:, 0].astype(str).apply(map_symbol)
            chunks.append(chunk)
            pbar.update(1)
    
    # Concatenate all chunks
    df = pd.concat(chunks, ignore_index=True)
    
    print(f"  Mapped {mapped_count} gene instances to Ensembl IDs, {len(unmapped_symbols)} unique unmapped symbols")
    
    # Save to file
    print(f"Writing to: {output_file}")
    df.to_csv(output_file, header=False, index=False)
    
    print(f"Successfully saved to: {output_file}")


def process_anndata_file(anndata_file, output_file, mapping):
    """Process AnnData file by mapping gene symbols to Ensembl IDs."""
    print(f"Processing AnnData file: {anndata_file}")
    
    try:
        adata = ad.read_h5ad(anndata_file)
    except Exception as e:
        raise FileNotFoundError(f"Could not read AnnData file: {e}")
    
    print(f"Loaded AnnData: {adata.shape} (cells × genes)")
    
    # Map gene names (var_names) from symbols to Ensembl IDs
    mapped_count = 0
    unmapped_symbols = set()
    
    # Store original gene names in var metadata if not already there
    if 'gene_symbol' not in adata.var.columns:
        adata.var['gene_symbol'] = adata.var_names.copy()
    
    # Create new var_names with Ensembl IDs (with progress bar)
    new_var_names = []
    for symbol in tqdm(adata.var_names, desc="Mapping genes", unit="gene"):
        if symbol in mapping:
            new_var_names.append(mapping[symbol])
            mapped_count += 1
        else:
            new_var_names.append(symbol)  # Keep original if not in mapping
            unmapped_symbols.add(symbol)
    
    # Update var_names
    adata.var_names = new_var_names
    
    print(f"  Mapped {mapped_count} symbols to Ensembl IDs")
    print(f"  {len(unmapped_symbols)} symbols not found in mapping (keeping original)")
    
    # Save to file
    print(f"Writing to: {output_file}")
    adata.write_h5ad(output_file)
    
    print(f"Successfully saved to: {output_file}")


def _process_grn_matrix_anndata(grn_file, matrix_file, anndata_file, grn_output, matrix_output, anndata_output, mapping):
    """Convert GRN, matrix, and pseudobulk AnnData if each input file exists."""
    if os.path.exists(grn_file):
        process_grn_file(grn_file, grn_output, mapping)
    else:
        print(f"\nNote: GRN file not found at {grn_file}, skipping...")

    if os.path.exists(matrix_file):
        process_matrix_file(matrix_file, matrix_output, mapping)
    else:
        print(f"\nNote: Matrix file not found at {matrix_file}, skipping...")

    if os.path.exists(anndata_file):
        process_anndata_file(anndata_file, anndata_output, mapping)
    else:
        print(f"\nNote: AnnData file not found at {anndata_file}, skipping...")


def main():
    """Main function to update GRN, matrix, and AnnData files."""
    parser = argparse.ArgumentParser(
        description='Convert gene symbols to Ensembl IDs in GRN, matrix, and AnnData files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python src/convert_to_ensemble_IDs.py -data Atlas -region PFC -cell_type Astrocytes
  python src/convert_to_ensemble_IDs.py -data ROSMAP -cell_type Astrocytes
""",
    )
    parser.add_argument(
        '-data',
        type=str,
        default='Atlas',
        choices=['Atlas', 'ROSMAP'],
        help='Dataset: Atlas (per-region GRN files) or ROSMAP (data/GRN/ROSMAP/ and data/ROSMAP_Data/)'
    )
    parser.add_argument(
        '-region',
        type=str,
        default=None,
        help='Region for Atlas/SEA-AD (e.g., V1, EC, ITG, PFC, MTG). Not used for ROSMAP.'
    )
    parser.add_argument(
        '-cell_type',
        type=str,
        default='Astrocytes',
        help='Cell type (e.g., Astrocytes, Microglia). Default: Astrocytes'
    )
    parser.add_argument(
        '--skip-raw',
        action='store_true',
        help='ROSMAP only: skip converting data/ROSMAP_Data/<CellType>.h5ad'
    )

    args = parser.parse_args()

    if args.data == 'Atlas' and not args.region:
        parser.error('-region is required when -data Atlas')

    if args.data == 'ROSMAP' and args.cell_type not in ROSMAP_CELL_TYPES:
        parser.error(
            f"ROSMAP cell type must be one of {sorted(ROSMAP_CELL_TYPES)}, got '{args.cell_type}'"
        )

    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

    mapping = load_mapping()

    try:
        if args.data == 'ROSMAP':
            (
                grn_file, matrix_file, anndata_file,
                grn_output, matrix_output, anndata_output,
                raw_anndata_file, raw_anndata_output,
            ) = get_rosmap_file_paths(cell_type=args.cell_type, data_dir=data_dir)

            _process_grn_matrix_anndata(
                grn_file, matrix_file, anndata_file,
                grn_output, matrix_output, anndata_output, mapping,
            )

            if not args.skip_raw:
                if os.path.exists(raw_anndata_file):
                    process_anndata_file(raw_anndata_file, raw_anndata_output, mapping)
                else:
                    print(f"\nNote: Raw ROSMAP AnnData not found at {raw_anndata_file}, skipping...")
        else:
            grn_file, matrix_file, anndata_file, grn_output, matrix_output, anndata_output = get_grn_file_paths(
                args.region, cell_type=args.cell_type, data_dir=data_dir
            )
            _process_grn_matrix_anndata(
                grn_file, matrix_file, anndata_file,
                grn_output, matrix_output, anndata_output, mapping,
            )

    except Exception as e:
        print(f"\nError: {e}")
        raise

if __name__ == "__main__":
    main()
