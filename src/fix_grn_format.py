"""
Fix GRN file format to ensure each row is properly tab-separated.

The GRN files currently have TF on one line and Target/Score on the next line.
This script converts them to proper tab-separated format: TF\tTarget\tScore
"""

import os
import argparse
from pathlib import Path


def fix_grn_format(input_file, output_file):
    """
    Fix GRN file format to ensure tab-separated rows.
    Pattern: TF on one line, Target Score on next line (with leading space)
    Converts to: TF\tTarget\tScore
    
    Parameters:
    -----------
    input_file : str
        Path to input GRN file
    output_file : str
        Path to output GRN file
    """
    print(f"\nReading GRN file: {input_file}")
    file_size = os.path.getsize(input_file) / (1024 * 1024)
    print(f"File size: {file_size:.2f} MB")
    
    # Count TFs in original file
    tf_count = 0
    rows_written = 0
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        while True:
            # Read TF line
            tf_line = infile.readline()
            if not tf_line:  # End of file
                break
            
            tf_line = tf_line.rstrip('\n\r')
            if not tf_line:  # Empty line, skip
                continue
            
            # Check if already tab-separated
            if '\t' in tf_line:
                # Already in correct format, write as is
                outfile.write(tf_line + '\n')
                rows_written += 1
                continue
            
            # Read Target/Score line
            target_line = infile.readline()
            if not target_line:  # End of file, no Target/Score
                break
            
            target_line = target_line.rstrip('\n\r')
            if not target_line:  # Empty line, skip
                continue
            
            # Remove leading whitespace and split
            parts = target_line.strip().split(None, 1)
            if len(parts) == 2:
                target, score = parts
                # Write as tab-separated: TF\tTarget\tScore
                outfile.write(f"{tf_line}\t{target}\t{score}\n")
                rows_written += 1
                tf_count += 1
            else:
                print(f"  Warning: Target/Score line '{target_line}' at line {rows_written + 1} has no Score")
                quit()
    
    print(f"Processed {rows_written} rows")
    print(f"Successfully wrote fixed format to: {output_file}")
    
    # Validation check
    print(f"\nValidation:")
    print(f"  TFs found in original file: {tf_count}")
    print(f"  Rows written to new file: {rows_written}")
    if tf_count == rows_written:
        print(f"  ✓ Match! All TFs converted successfully.")
    else:
        print(f"  ⚠ Mismatch: {abs(tf_count - rows_written)} difference")


def main():
    """Main function to fix GRN file format."""
    parser = argparse.ArgumentParser(
        description='Fix GRN file format to ensure proper tab-separated rows',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        'input_file',
        type=str,
        help='Full path to the GRN file (e.g. data/GRN/Astrocytes/EC_GRN.tsv)',
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Output path. Default: input path with _fixed before extension',
    )

    args = parser.parse_args()

    input_file = Path(args.input_file).resolve()
    if not input_file.exists():
        print(f"Error: GRN file not found: {input_file}")
        return

    if args.output is not None:
        output_file = Path(args.output).resolve()
    else:
        # Add _fixed before extension
        output_file = input_file.parent / (input_file.stem + '_fixed' + input_file.suffix)

    try:
        fix_grn_format(str(input_file), str(output_file))
        print("\n" + "="*60)
        print("Complete!")
        print("="*60)
    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()
