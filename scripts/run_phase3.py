"""Phase 3: generate and run seeded training commands from Phase 2 best_params JSONs.

Usage:
    python scripts/run_phase3.py --cell_type Astrocytes [--dry-run] [--include_random]
    python scripts/run_phase3.py --cell_type Microglia  [--dry-run]
    python scripts/run_phase3.py --cell_type Astrocytes --cell_type Microglia [--dry-run]
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

PYTHON = r"C:\Users\Manish\.conda\envs\sf2026\python.exe"
REPO = Path(__file__).resolve().parent.parent
PARAMS_DIR = REPO / "ranking_model_runs" / "autosearch_v2"

REGION_MAP = {
    "Astrocytes": ["EC", "ITG", "PFC", "V2", "V1"],
    "Microglia": ["CrossRegion", "ITG", "PFC"],
    "MTG": ["MTG"],
}


def load_params(cell_type, region, input_tag, prefix="v2b"):
    fp = PARAMS_DIR / f"{prefix}_{region}_{cell_type}_{input_tag}_best_params.json"
    if not fp.exists():
        return None
    with open(fp) as f:
        return json.load(f)


def build_cmd(cell_type, region, tp, n_seeds=10):
    """Build train_model_v2.py CLI from trial_params dict."""
    cmd = [
        PYTHON, "src/train_model_v2.py",
        "--region", region,
        "--cell_type", cell_type,
        "--input_type", tp["input_type"],
        "--model_type", tp["model_type"],
        "--n_seeds", str(n_seeds),
        "--epochs", "200",
        "--patience", "40",
        "--batch_size", str(tp["batch_size"]),
        "--lr", str(tp["lr"]),
        "--dropout", str(tp["dropout"]),
        "--weight_decay", str(tp["weight_decay"]),
        "--label_smoothing", str(tp["label_smoothing"]),
        "--n_hidden_layers", str(tp.get("n_hidden_layers", 1)),
        "--hidden_dim_3layer", str(tp.get("hidden_dim_3layer", 32)),
    ]
    pca = tp.get("pca_components")
    if pca is not None:
        cmd += ["--pca_components", str(pca)]
    mlp_hs = tp.get("mlp_hidden_sizes", "")
    if mlp_hs:
        cmd += ["--mlp_hidden_sizes", mlp_hs]
    if tp.get("thal_ordinal"):
        cmd.append("--thal_ordinal")
    if tp.get("attention"):
        cmd.append("--attention")

    for aug_key in [
        "aug_scale_jitter", "aug_per_gene_scale_jitter", "aug_gene_dropout",
        "aug_emb_dim_dropout", "aug_feature_dropout", "aug_gene_block_cutout_max",
        "aug_noise_std", "mixup_alpha", "gene_cutmix_prob", "gene_cutmix_alpha",
    ]:
        v = tp.get(aug_key)
        if v is not None and v != 0 and v != 0.0:
            cmd += [f"--{aug_key}", str(v)]
    return cmd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell_type", action="append", required=True,
                        choices=list(REGION_MAP.keys()))
    parser.add_argument("--n_seeds", type=int, default=10)
    parser.add_argument("--include_random", action="store_true",
                        help="Also run random_embedding ablation.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    os.chdir(REPO)
    tasks = []

    for cell_type in args.cell_type:
        regions = REGION_MAP[cell_type]
        for region in regions:
            ep = load_params(cell_type, region, "expr")
            if ep and ep.get("best_value", float("-inf")) > 0:
                tasks.append((cell_type, "expr", region,
                              build_cmd(cell_type, region, ep["trial_params"], args.n_seeds)))

            emb_p = load_params(cell_type, region, "emb")
            if emb_p and emb_p.get("best_value", float("-inf")) > 0:
                tasks.append((cell_type, "emb", region,
                              build_cmd(cell_type, region, emb_p["trial_params"], args.n_seeds)))

            if args.include_random and emb_p and emb_p.get("best_value", float("-inf")) > 0:
                tp_rand = dict(emb_p["trial_params"])
                tp_rand["input_type"] = "random_embedding"
                tp_rand["model_type"] = "flatten_mlp"
                tasks.append((cell_type, "rand", region,
                              build_cmd(cell_type, region, tp_rand, args.n_seeds)))

    print(f"Phase 3: {len(tasks)} training runs queued")
    for ct, tag, region, cmd in tasks:
        print(f"  [{ct}/{region}/{tag}]: ...{' '.join(cmd[-6:])}")

    if args.dry_run:
        print("\nDry run -- not executing.")
        return

    for i, (ct, tag, region, cmd) in enumerate(tasks):
        label = f"[{i+1}/{len(tasks)}] {ct}/{region}/{tag}"
        print(f"\n{'='*60}\n{label}\n{'='*60}")
        result = subprocess.run(cmd, cwd=str(REPO))
        if result.returncode != 0:
            print(f"WARNING: {label} exited with code {result.returncode}")


if __name__ == "__main__":
    main()
