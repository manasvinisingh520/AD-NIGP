"""
Diagnostic test: verify metadata joins to embeddings and Color by works.
Run from project root: python embedding_viz/test_color_by.py
"""
import os, sys, pickle, glob
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
os.chdir(_project_root)

from embedding_viz.load_data import (
    _embedding_row_to_cell_id,
    _get_astrocyte_cell_embedding_files,
    load_metadata_for_region,
    load_region_embeddings_and_metadata,
    METADATA_PATH, REGION_TO_TAB,
)
import pandas as pd


def test():
    errors = []

    # 1. Test one region end-to-end (EC is small)
    region = "EC"
    print(f"[1] Loading patient-level metadata for {region}...")
    meta_df = load_metadata_for_region(region)
    if meta_df is None:
        errors.append("load_metadata_for_region returned None for EC")
        print("    ERROR: returned None")
    else:
        print(f"    Shape: {meta_df.shape}")
        print(f"    Columns: {list(meta_df.columns)}")
        print(f"    Index (SampleName) samples: {meta_df.index[:5].tolist()}")
        print(f"    Path..Group. samples: {meta_df['Path..Group.'].head().tolist() if 'Path..Group.' in meta_df.columns else 'N/A'}")

    # 2. Check embedding cell IDs match metadata index
    files = _get_astrocyte_cell_embedding_files()
    emb_file = next((p for r, p in files if r == region), None)
    if emb_file:
        emb_df = pd.read_csv(emb_file, sep=",", index_col=0, nrows=20)
        emb_cell_ids = list(set(_embedding_row_to_cell_id(r) for r in emb_df.index.astype(str)))
        print(f"\n[2] Embedding cell IDs (unique, first 20 rows): {emb_cell_ids[:5]}")
        if meta_df is not None:
            matched = [c for c in emb_cell_ids if c in meta_df.index]
            print(f"    Matched: {len(matched)}/{len(emb_cell_ids)}")
            if not matched:
                errors.append("No cell ID matches for EC")
            else:
                sample_cid = matched[0]
                row = meta_df.loc[sample_cid]
                print(f"    Sample match '{sample_cid}': {dict(row)}")

    # 3. Full end-to-end: load_region_embeddings_and_metadata
    print(f"\n[3] Full pipeline for {region} (use_cache=False)...")
    data = load_region_embeddings_and_metadata(region, use_cache=False)
    pts = data.get("points", [])
    mc = data.get("meta_columns", [])
    print(f"    Points: {len(pts)}, meta_columns: {mc}")

    non_empty = {}
    for col in mc:
        cnt = sum(1 for p in pts if p.get(col) and str(p[col]).strip() not in ("", "(missing)"))
        non_empty[col] = cnt
    print(f"    Non-empty counts: {non_empty}")

    if pts:
        print(f"    Sample point: { {k: v for k, v in pts[0].items() if k not in ('x','y')} }")

    empty_cols = [c for c in mc if c != "region" and non_empty.get(c, 0) == 0]
    if empty_cols:
        errors.append(f"Columns with ALL empty values: {empty_cols}")

    return errors


if __name__ == "__main__":
    errors = test()
    print("\n" + "=" * 60)
    if errors:
        print(f"ISSUES ({len(errors)}):")
        for e in errors:
            print(f"  - {e}")
    else:
        print("ALL CHECKS PASSED - Color by should work!")
