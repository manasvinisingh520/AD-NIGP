"""
One-time script to populate embedding_viz/data with all files needed by the app.
Run from project root: python -m embedding_viz.copy_data_to_app_data

Creates embedding_viz/data/ and copies from:
- GREmLN/embeddings (cell/gene embedding TSVs, excluding Microglia)
- data/AtlasData, data/*.pkl, data/GRN
- results/dge_final
- e:/sf2026/dge_results/dge_final (Atlas DGE - if present)
- ranked_lists (only sets used by pipeline + gene rank tab)
- ranking_model_runs (only the 12 CSV files needed)
- visualizer_results
"""
from pathlib import Path
import shutil

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(__file__).resolve().parent / "data"
ATLAS_DGE_EXTRA = Path("e:/sf2026/dge_results/dge_final")

# Ranked list sets used by pipeline (from _ranked_set_name) and any others for Gene Rankings tab
RANKED_SETS = [
    "EC_Astrocytes_embedding_highest_lambda_entropy",
    "ge_all_3L_shrinking_patience60_32_ITG",
    "ge_all_3L_shrinking_patience60_32_PFC",
    "ge_all_3L_shrinking_patience60_32_V1",
    "ge_all_3L_shrinking_patience60_32_V2",
    "ge_all_3L_shrinking_patience60_64_MTG_ThalOnly_run2",
]

# ranking_model_runs: subdir -> list of CSV filenames
MODEL_RUN_FILES = {
    "ge_all_3L_shrinking_patience60_32_EC": ["overfitting_experiments.csv"],
    "ge_all_3L_shrinking_patience60_32_EC_Expression": ["expression_experiments.csv"],
    "ge_all_3L_shrinking_patience60_32_ITG": ["overfitting_experiments.csv"],
    "ge_all_3L_shrinking_patience60_32_ITG_Expression": ["expression_experiments.csv"],
    "ge_all_3L_shrinking_patience60_32_PFC": ["overfitting_experiments.csv"],
    "ge_all_3L_shrinking_patience60_32_PFC_Expression": ["expression_experiments.csv"],
    "ge_all_3L_shrinking_patience60_32_V1": ["overfitting_experiments.csv"],
    "ge_all_3L_shrinking_patience60_32_V1_Expression": ["expression_experiments.csv"],
    "ge_all_3L_shrinking_patience60_32_V2": ["overfitting_experiments.csv"],
    "ge_all_3L_shrinking_patience60_32_V2_Expression": ["expression_experiments.csv"],
    "ge_all_3L_shrinking_patience60_64_MTG": ["overfitting_experiments.csv"],
    "ge_all_3L_shrinking_patience60_64_MTG_Expression": ["expression_experiments.csv"],
}


def main():
    print("Populating embedding_viz/data from project root:", PROJECT_ROOT)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Embeddings (exclude Microglia)
    emb_src = PROJECT_ROOT / "GREmLN" / "embeddings"
    emb_dst = DATA_DIR / "embeddings"
    if emb_src.exists():
        emb_dst.mkdir(parents=True, exist_ok=True)
        for p in list(emb_src.glob("*_cell_embeddings.tsv")) + list(emb_src.glob("*_gene_embeddings.tsv")):
            if "Microglia" in p.name:
                continue
            shutil.copy2(p, emb_dst / p.name)
            print(f"  Copied embeddings/{p.name}")
    else:
        print("  Skip GREmLN/embeddings (not found)")

    # 2. AtlasData
    atlas_src = PROJECT_ROOT / "data" / "AtlasData"
    atlas_dst = DATA_DIR / "AtlasData"
    if atlas_src.exists():
        atlas_dst.mkdir(parents=True, exist_ok=True)
        for f in atlas_src.glob("*.xlsx"):
            shutil.copy2(f, atlas_dst / f.name)
            print(f"  Copied AtlasData/{f.name}")
    else:
        print("  Skip data/AtlasData (not found)")

    # 3. Gene annotations
    for name in ["gene_annotations.pkl", "genes_2_ensembl_ids.pkl"]:
        src = PROJECT_ROOT / "data" / name
        if src.exists():
            shutil.copy2(src, DATA_DIR / name)
            print(f"  Copied {name}")
        else:
            print(f"  Skip data/{name} (not found)")

    # 4. GRN
    grn_src = PROJECT_ROOT / "data" / "GRN"
    grn_dst = DATA_DIR / "GRN"
    if grn_src.exists():
        for sub in ["Astrocytes", "MTG"]:
            s = grn_src / sub
            if s.exists():
                d = grn_dst / sub
                d.mkdir(parents=True, exist_ok=True)
                for f in s.iterdir():
                    if f.is_file():
                        shutil.copy2(f, d / f.name)
                        print(f"  Copied GRN/{sub}/{f.name}")
    else:
        print("  Skip data/GRN (not found)")

    # 5. DGE: results/dge_final + optional Atlas DGE from e:/
    dge_dst = DATA_DIR / "dge_final"
    dge_dst.mkdir(parents=True, exist_ok=True)
    for dge_src in [PROJECT_ROOT / "results" / "dge_final", ATLAS_DGE_EXTRA]:
        if not dge_src.exists():
            continue
        for f in dge_src.glob("dge_results_*.csv"):
            shutil.copy2(f, dge_dst / f.name)
            print(f"  Copied dge_final/{f.name}")

    # 6. Ranked lists (only used sets)
    rl_src = PROJECT_ROOT / "ranked_lists"
    rl_dst = DATA_DIR / "ranked_lists"
    if rl_src.exists():
        for set_name in RANKED_SETS:
            csv_path = rl_src / set_name / "ranked_genes.csv"
            if csv_path.exists():
                (rl_dst / set_name).mkdir(parents=True, exist_ok=True)
                shutil.copy2(csv_path, rl_dst / set_name / "ranked_genes.csv")
                print(f"  Copied ranked_lists/{set_name}/ranked_genes.csv")
        # Also copy any other ranked_lists that exist (for Gene Rankings tab dropdown)
        for d in sorted(rl_src.iterdir()):
            if d.is_dir() and (d / "ranked_genes.csv").exists():
                if not (rl_dst / d.name / "ranked_genes.csv").exists():
                    (rl_dst / d.name).mkdir(parents=True, exist_ok=True)
                    shutil.copy2(d / "ranked_genes.csv", rl_dst / d.name / "ranked_genes.csv")
                    print(f"  Copied ranked_lists/{d.name}/ranked_genes.csv")
    else:
        print("  Skip ranked_lists (not found)")

    # 7. Ranking model runs (only the 12 CSVs)
    mr_src = PROJECT_ROOT / "ranking_model_runs"
    mr_dst = DATA_DIR / "ranking_model_runs"
    if mr_src.exists():
        for subdir, files in MODEL_RUN_FILES.items():
            for fn in files:
                src_file = mr_src / subdir / fn
                if src_file.exists():
                    (mr_dst / subdir).mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src_file, mr_dst / subdir / fn)
                    print(f"  Copied ranking_model_runs/{subdir}/{fn}")
    else:
        print("  Skip ranking_model_runs (not found)")

    # 8. Visualizer results
    viz_src = PROJECT_ROOT / "visualizer_results"
    viz_dst = DATA_DIR / "visualizer_results"
    if viz_src.exists():
        viz_dst.mkdir(parents=True, exist_ok=True)
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.svg", "*.gif"):
            for f in viz_src.glob(ext):
                shutil.copy2(f, viz_dst / f.name)
                print(f"  Copied visualizer_results/{f.name}")
    else:
        print("  Skip visualizer_results (not found)")

    print("Done. Data is under:", DATA_DIR)


if __name__ == "__main__":
    main()
