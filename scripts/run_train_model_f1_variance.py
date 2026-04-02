"""
Run train_model.main repeatedly with fixed hyperparameters to estimate variance of mean LOFO F1.

From repo root:
  python run_train_model_f1_variance.py

Uses EC + loss_ptau=none (same as autosearch_ec). Each run uses a different RNG seed so F1 varies
with stochastic training (shuffle, augmentation, mixup/CutMix). Defaults: 10 runs, epochs=200, patience=30.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_args(
    *,
    seed: int,
    region: str,
    epochs: int,
    patience: int,
    device: str | None,
) -> SimpleNamespace:
    import torch

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Fixed trial config (edit here). Empty mlp_hidden_sizes -> baseline uses n_hidden_layers +
    # hidden_dim_3layer (see baseline_model.BaselineRankingModel).
    return SimpleNamespace(
        region=region,
        cell_type="Astrocytes",
        pooling=None,
        loss_ptau="none",
        batch_size=4,
        epochs=epochs,
        lr=0.0026295688409911596,
        device=dev,
        attention=False,
        attention_heads=1,
        dropout=0.5257136289179769,
        weight_decay=1.0,
        pca_components=8,
        gene_encoder=False,
        lambda_entropy=0.007297814436145046,
        thal_ordinal=False,
        recompute=False,
        verbose=False,
        verbose_cv=False,
        patience=patience,
        n_hidden_layers=1,
        hidden_dim_3layer=128,
        mlp_hidden_sizes="",
        best_f1=True,
        top_genes=None,
        ranked_genes_path=None,
        save_best_model=False,
        save_dir="models",
        data_suffix="",
        no_online_augment=False,
        aug_scale_jitter=0.15,
        aug_per_gene_scale_jitter=0.05,
        aug_gene_dropout=0.05,
        aug_emb_dim_dropout=0.1,
        aug_feature_dropout=0.0,
        aug_gene_block_cutout_max=32,
        aug_gene_block_cutout_prob=0.5,
        aug_noise_std=0.01,
        mixup_alpha=0.2,
        gene_cutmix_prob=0.25,
        gene_cutmix_alpha=0.9856345250050196,
        label_smoothing=0.1,
        no_tensorboard=True,
        log_base="ranking_model_runs",
        exp_id=f"f1var_seed{seed}",
        q=True,
    )


def main() -> None:
    import os

    os.chdir(ROOT)

    parser = argparse.ArgumentParser(description="Repeated train_model runs for F1 variance")
    parser.add_argument("--n_runs", type=int, default=10, help="Number of training runs")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--region", type=str, default="EC")
    parser.add_argument("--device", type=str, default=None, help="cuda | cpu (default: auto)")
    parser.add_argument("--base_seed", type=int, default=0, help="Seeds base_seed, base_seed+1, ...")
    args_cli = parser.parse_args()

    from train_model import main as train_main

    f1s: list[float] = []
    for i in range(args_cli.n_runs):
        seed = args_cli.base_seed + i
        _seed_all(seed)
        ns = build_args(
            seed=seed,
            region=args_cli.region,
            epochs=args_cli.epochs,
            patience=args_cli.patience,
            device=args_cli.device,
        )
        print(f"=== Run {i + 1}/{args_cli.n_runs} (seed={seed}) ===", flush=True)
        avg_f1, avg_mse, avg_mae, _ = train_main(ns)
        print(f"  mean LOFO F1 = {avg_f1:.6f} | MSE={avg_mse:.6f} MAE={avg_mae:.6f}", flush=True)
        f1s.append(float(avg_f1))

    arr = np.array(f1s, dtype=np.float64)
    print()
    print(f"F1 scores ({args_cli.n_runs} runs): {np.array2string(arr, precision=6)}")
    print(f"mean(F1) = {arr.mean():.6f}")
    print(f"std(F1)  = {arr.std(ddof=1):.6f}" if len(arr) > 1 else "std(F1)  = nan (need n_runs>=2)")
    print(f"var(F1)  = {arr.var(ddof=1):.6f}" if len(arr) > 1 else "var(F1)  = nan (need n_runs>=2)")


if __name__ == "__main__":
    main()
