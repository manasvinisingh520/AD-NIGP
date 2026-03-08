"""
Precompute embedding visualization cache for all Astrocytes regions.
Run from project root: python -m embedding_viz.precompute_cache
Or: python embedding_viz/precompute_cache.py

Uses existing cache when present (use_cache=True), so only missing regions are computed.
To force recompute for all regions, run with: python -m embedding_viz.precompute_cache --force
"""

import argparse
import os
import sys

# When run as script by path (e.g. python embedding_viz/precompute_cache.py), ensure project root is on path
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
os.chdir(_project_root)

from embedding_viz.load_data import (
    get_available_regions,
    load_region_embeddings_and_metadata,
    load_all_regions_embeddings,
    SUPPORTED_METHODS,
)


def main():
    parser = argparse.ArgumentParser(description="Precompute embedding viz cache for all regions")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute even if cache exists for a region",
    )
    parser.add_argument(
        "--method",
        nargs="*",
        default=list(SUPPORTED_METHODS),
        choices=SUPPORTED_METHODS,
        help="Reduction methods to precompute (default: all)",
    )
    args = parser.parse_args()
    use_cache = not args.force
    methods = args.method

    regions = get_available_regions()
    if not regions:
        print("No Astrocytes regions found under GREmLN/embeddings/")
        sys.exit(1)

    print(f"Regions to process: {regions}")
    print(f"Methods: {methods}")
    for method in methods:
        print(f"\n=== {method.upper()} ===")
        for region in regions:
            print(f"  {region} ({method})...", end=" ", flush=True)
            try:
                data = load_region_embeddings_and_metadata(region, method=method, use_cache=use_cache)
                n = len(data.get("points", []))
                print(f"done ({n} points)")
            except Exception as e:
                print(f"ERROR: {e}")
        print(f"  All Regions ({method})...", end=" ", flush=True)
        try:
            data = load_all_regions_embeddings(method=method, use_cache=use_cache)
            n = len(data.get("points", []))
            print(f"done ({n} points)")
        except Exception as e:
            print(f"ERROR: {e}")
    print("\nDone.")


if __name__ == "__main__":
    main()
