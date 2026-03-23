"""
Autoresearch-style hyperparameter search for EC, loss_ptau=none, objective mean LOFO F1.

Calls train_model.main (fixed entrypoint); does not edit model code.
Optuna (default) or --sampler random with JSONL log.

Usage:
  pip install -r requirements-autosearch.txt
  python src/autosearch_ec.py --n_trials 20 --log_base ranking_model_runs/autosearch_ec --epochs 150 --patience 25
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_model import main as train_main

REGION = "EC"
CELL_TYPE = "Astrocytes"
LOSS_PTAU = "none"

MLP_MENU = ["", "64,32,16", "128,64,32", "256,128,64,32"]
PCA_CHOICES = [2, 4, 8, 16]
POOLING_BASELINE = ["mean", "max", "none"]


@dataclass
class TrialParams:
    """Serializable trial configuration (for JSON/CSV)."""

    gene_encoder: bool
    attention: bool
    pooling: Optional[str]  # None for 3D / gene_encoder; "mean" or "max" for pooled baseline
    pca_components: int
    mlp_hidden_sizes: str
    n_hidden_layers: int
    hidden_dim_3layer: int
    dropout: float
    weight_decay: float
    lr: float
    lambda_entropy: float
    batch_size: int
    aug_scale_jitter: float
    aug_per_gene_scale_jitter: float
    aug_gene_dropout: float
    aug_emb_dim_dropout: float
    aug_feature_dropout: float
    aug_gene_block_cutout_max: int
    aug_noise_std: float
    mixup_alpha: float
    gene_cutmix_prob: float
    gene_cutmix_alpha: float
    label_smoothing: float
    thal_ordinal: bool


def _pooling_for_namespace(gene_encoder: bool, pooling_choice: str) -> Any:
    if gene_encoder:
        return None
    if pooling_choice == "none":
        return None
    return pooling_choice


def suggest_trial_optuna(trial) -> TrialParams:
    """Sample one configuration (Optuna trial). Always suggest same param names for storage compatibility."""
    gene_encoder = trial.suggest_categorical("gene_encoder", [False, True])
    attention_s = trial.suggest_categorical("attention", [False, True])
    pooling_choice_s = trial.suggest_categorical("pooling", POOLING_BASELINE)
    lambda_entropy_s = trial.suggest_float("lambda_entropy", 1e-4, 0.2, log=True)

    if gene_encoder:
        attention = False
        pooling_choice = "none"
        lambda_entropy = float(lambda_entropy_s)
    else:
        attention = bool(attention_s)
        pooling_choice = str(pooling_choice_s)
        lambda_entropy = 1e-3

    pca_components = trial.suggest_categorical("pca_components", PCA_CHOICES)
    mlp_hidden_sizes = trial.suggest_categorical("mlp_hidden_sizes", MLP_MENU)
    n_hidden_layers = trial.suggest_categorical("n_hidden_layers", [1, 3])
    hidden_dim_3layer = trial.suggest_categorical("hidden_dim_3layer", [32, 64, 128])

    dropout = trial.suggest_float("dropout", 0.2, 0.55)
    weight_decay = trial.suggest_categorical("weight_decay", [0.1, 0.5, 1.0])
    lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [4, 8, 16])

    aug_scale_jitter = trial.suggest_categorical("aug_scale_jitter", [0.0, 0.05, 0.1, 0.15])
    aug_per_gene_scale_jitter = trial.suggest_categorical("aug_per_gene_scale_jitter", [0.0, 0.05, 0.1])
    aug_emb_dim_dropout = trial.suggest_categorical("aug_emb_dim_dropout", [0.0, 0.05, 0.1])
    aug_noise_std = trial.suggest_categorical("aug_noise_std", [0.0, 0.01, 0.02, 0.03])
    aug_gene_block_cutout_max = trial.suggest_categorical("aug_gene_block_cutout_max", [0, 32, 128, 256])

    is_3d = gene_encoder or (pooling_choice == "none")
    if is_3d:
        aug_gene_dropout = trial.suggest_categorical("aug_gene_dropout", [0.0, 0.05, 0.1, 0.15])
        aug_feature_dropout = 0.0
        gene_cutmix_prob = trial.suggest_categorical("gene_cutmix_prob", [0.0, 0.25, 0.5])
    else:
        aug_gene_dropout = 0.0
        aug_feature_dropout = trial.suggest_categorical("aug_feature_dropout", [0.0, 0.05, 0.1, 0.15])
        gene_cutmix_prob = 0.0

    mixup_alpha = trial.suggest_categorical("mixup_alpha", [0.0, 0.2, 0.4])
    gene_cutmix_alpha = trial.suggest_float("gene_cutmix_alpha", 0.5, 2.0)
    label_smoothing = trial.suggest_categorical("label_smoothing", [0.0, 0.05, 0.1])
    thal_ordinal = trial.suggest_categorical("thal_ordinal", [False, True])

    if thal_ordinal:
        mixup_alpha = 0.0
        gene_cutmix_prob = 0.0

    return TrialParams(
        gene_encoder=gene_encoder,
        attention=attention,
        pooling=_pooling_for_namespace(gene_encoder, pooling_choice),
        pca_components=pca_components,
        mlp_hidden_sizes=mlp_hidden_sizes if not gene_encoder else "",
        n_hidden_layers=n_hidden_layers if not gene_encoder else 1,
        hidden_dim_3layer=hidden_dim_3layer if not gene_encoder else 32,
        dropout=dropout,
        weight_decay=weight_decay,
        lr=lr,
        lambda_entropy=lambda_entropy,
        batch_size=batch_size,
        aug_scale_jitter=aug_scale_jitter,
        aug_per_gene_scale_jitter=aug_per_gene_scale_jitter,
        aug_gene_dropout=aug_gene_dropout,
        aug_emb_dim_dropout=aug_emb_dim_dropout,
        aug_feature_dropout=aug_feature_dropout,
        aug_gene_block_cutout_max=aug_gene_block_cutout_max,
        aug_noise_std=aug_noise_std,
        mixup_alpha=mixup_alpha,
        gene_cutmix_prob=gene_cutmix_prob,
        gene_cutmix_alpha=gene_cutmix_alpha,
        label_smoothing=label_smoothing,
        thal_ordinal=thal_ordinal,
    )


def suggest_trial_random(rng) -> TrialParams:
    """Match Optuna search space approximately with stdlib random."""
    gene_encoder = rng.random() < 0.5
    if gene_encoder:
        attention = False
        pooling_choice = "none"
        lambda_entropy = 10 ** rng.uniform(-4, math.log10(0.2))
    else:
        attention = rng.random() < 0.5
        pooling_choice = rng.choice(POOLING_BASELINE)
        lambda_entropy = 1e-3

    pca_components = rng.choice(PCA_CHOICES)
    mlp_hidden_sizes = "" if gene_encoder else rng.choice(MLP_MENU)
    n_hidden_layers = 1 if gene_encoder else rng.choice([1, 3])
    hidden_dim_3layer = 32 if gene_encoder else rng.choice([32, 64, 128])

    dropout = rng.uniform(0.2, 0.55)
    weight_decay = rng.choice([0.1, 0.5, 1.0])
    lr = 10 ** rng.uniform(-4, math.log10(3e-3))
    batch_size = rng.choice([4, 8, 16])

    aug_scale_jitter = rng.choice([0.0, 0.05, 0.1, 0.15])
    aug_per_gene_scale_jitter = rng.choice([0.0, 0.05, 0.1])
    aug_emb_dim_dropout = rng.choice([0.0, 0.05, 0.1])
    aug_noise_std = rng.choice([0.0, 0.01, 0.02, 0.03])
    aug_gene_block_cutout_max = rng.choice([0, 32, 128, 256])

    is_3d = gene_encoder or (pooling_choice == "none")
    if is_3d:
        aug_gene_dropout = rng.choice([0.0, 0.05, 0.1, 0.15])
        aug_feature_dropout = 0.0
        gene_cutmix_prob = rng.choice([0.0, 0.25, 0.5])
    else:
        aug_gene_dropout = 0.0
        aug_feature_dropout = rng.choice([0.0, 0.05, 0.1, 0.15])
        gene_cutmix_prob = 0.0

    mixup_alpha = rng.choice([0.0, 0.2, 0.4])
    gene_cutmix_alpha = rng.uniform(0.5, 2.0)
    label_smoothing = rng.choice([0.0, 0.05, 0.1])
    thal_ordinal = rng.random() < 0.25
    if thal_ordinal:
        mixup_alpha = 0.0
        gene_cutmix_prob = 0.0

    return TrialParams(
        gene_encoder=gene_encoder,
        attention=attention,
        pooling=_pooling_for_namespace(gene_encoder, pooling_choice),
        pca_components=pca_components,
        mlp_hidden_sizes=mlp_hidden_sizes,
        n_hidden_layers=n_hidden_layers,
        hidden_dim_3layer=hidden_dim_3layer,
        dropout=dropout,
        weight_decay=weight_decay,
        lr=lr,
        lambda_entropy=lambda_entropy,
        batch_size=batch_size,
        aug_scale_jitter=aug_scale_jitter,
        aug_per_gene_scale_jitter=aug_per_gene_scale_jitter,
        aug_gene_dropout=aug_gene_dropout,
        aug_emb_dim_dropout=aug_emb_dim_dropout,
        aug_feature_dropout=aug_feature_dropout,
        aug_gene_block_cutout_max=aug_gene_block_cutout_max,
        aug_noise_std=aug_noise_std,
        mixup_alpha=mixup_alpha,
        gene_cutmix_prob=gene_cutmix_prob,
        gene_cutmix_alpha=gene_cutmix_alpha,
        label_smoothing=label_smoothing,
        thal_ordinal=thal_ordinal,
    )


def params_to_namespace(
    p: TrialParams,
    *,
    exp_id: str,
    log_base: str,
    epochs: int,
    patience: int,
    device: str,
    data_suffix: str,
    no_online_augment: bool,
    attention_heads: int,
) -> SimpleNamespace:
    """Build train_model args. Always best_f1=True (objective is F1)."""
    return SimpleNamespace(
        region=REGION,
        cell_type=CELL_TYPE,
        pooling=p.pooling,
        loss_ptau=LOSS_PTAU,
        batch_size=p.batch_size,
        epochs=epochs,
        lr=p.lr,
        device=device,
        dropout=p.dropout,
        weight_decay=p.weight_decay,
        attention=p.attention,
        attention_heads=attention_heads,
        pca_components=p.pca_components,
        gene_encoder=p.gene_encoder,
        lambda_entropy=p.lambda_entropy,
        log_base=log_base,
        exp_id=exp_id,
        verbose=False,
        n_hidden_layers=p.n_hidden_layers,
        hidden_dim_3layer=p.hidden_dim_3layer,
        mlp_hidden_sizes=p.mlp_hidden_sizes,
        patience=patience,
        best_f1=True,
        thal_ordinal=p.thal_ordinal,
        top_genes=None,
        ranked_genes_path=None,
        save_best_model=False,
        save_dir="models",
        data_suffix=data_suffix,
        recompute=False,
        no_online_augment=no_online_augment,
        aug_scale_jitter=0.0 if no_online_augment else p.aug_scale_jitter,
        aug_per_gene_scale_jitter=0.0 if no_online_augment else p.aug_per_gene_scale_jitter,
        aug_gene_dropout=0.0 if no_online_augment else p.aug_gene_dropout,
        aug_emb_dim_dropout=0.0 if no_online_augment else p.aug_emb_dim_dropout,
        aug_feature_dropout=0.0 if no_online_augment else p.aug_feature_dropout,
        aug_gene_block_cutout_max=0 if no_online_augment else p.aug_gene_block_cutout_max,
        aug_gene_block_cutout_prob=0.5,
        aug_noise_std=0.0 if no_online_augment else p.aug_noise_std,
        mixup_alpha=0.0 if no_online_augment else p.mixup_alpha,
        gene_cutmix_prob=0.0 if no_online_augment else p.gene_cutmix_prob,
        gene_cutmix_alpha=p.gene_cutmix_alpha,
        label_smoothing=p.label_smoothing,
    )


def _run_one_trial(ns: SimpleNamespace) -> Tuple[float, Optional[str]]:
    try:
        avg_f1, _, _, _ = train_main(ns)
        if avg_f1 is None or (isinstance(avg_f1, float) and math.isnan(avg_f1)):
            return float("-inf"), "invalid_f1"
        return float(avg_f1), None
    except Exception as e:
        return float("-inf"), f"{type(e).__name__}: {e}\n{traceback.format_exc()}"


def export_trials_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    keys = sorted(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="EC autosearch: maximize mean F1 (loss_ptau=none)")
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--study_name", type=str, default="ec_f1_none")
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Optuna storage URL; default sqlite under --log_base",
    )
    parser.add_argument("--sampler", type=str, choices=["optuna", "random"], default="optuna")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--log_base", type=str, default=os.path.join("ranking_model_runs", "autosearch_ec"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data_suffix", type=str, default="")
    parser.add_argument("--no_online_augment", action="store_true")
    parser.add_argument("--attention_heads", type=int, default=1)
    args = parser.parse_args()

    import random

    import numpy as np
    import torch

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.log_base, exist_ok=True)
    # Fresh JSONL per invocation (Optuna DB still holds history if load_if_exists).
    jsonl_path = os.path.join(args.log_base, f"{args.study_name}_trials.jsonl")
    open(jsonl_path, "w", encoding="utf-8").close()
    csv_path = os.path.join(args.log_base, f"{args.study_name}_trials.csv")
    best_json_path = os.path.join(args.log_base, f"{args.study_name}_best_params.json")

    if args.sampler == "optuna":
        try:
            import optuna
        except ImportError as e:
            raise SystemExit(
                "Optuna is required for --sampler optuna. Install: pip install -r requirements-autosearch.txt"
            ) from e

        storage = args.storage or f"sqlite:///{os.path.join(args.log_base, args.study_name + '_optuna.db')}"
        study = optuna.create_study(
            study_name=args.study_name,
            storage=storage,
            direction="maximize",
            load_if_exists=True,
        )

        trial_rows: List[Dict[str, Any]] = []

        def objective(trial: optuna.Trial) -> float:
            p = suggest_trial_optuna(trial)
            trial.set_user_attr("trial_params", asdict(p))
            exp_id = f"autosearch_{args.study_name}_t{trial.number}"
            ns = params_to_namespace(
                p,
                exp_id=exp_id,
                log_base=args.log_base,
                epochs=args.epochs,
                patience=args.patience,
                device=device,
                data_suffix=args.data_suffix,
                no_online_augment=args.no_online_augment,
                attention_heads=args.attention_heads,
            )
            f1, err = _run_one_trial(ns)
            row = {
                "trial": trial.number,
                "value": f1,
                "error": err or "",
                "exp_id": exp_id,
                **asdict(p),
            }
            trial_rows.append(row)
            with open(jsonl_path, "a", encoding="utf-8") as jf:
                jf.write(json.dumps(row, default=str) + "\n")
            if err:
                print(f"  Trial {trial.number} FAILED: {err[:200]}...")
            else:
                print(f"  Trial {trial.number} F1={f1:.4f}")
            return f1

        study.optimize(objective, n_trials=args.n_trials)

        export_trials_csv(csv_path, trial_rows)
        try:
            best = study.best_trial
            best_payload = {
                "study_name": args.study_name,
                "storage": storage,
                "best_value": best.value,
                "best_trial": best.number,
                "optuna_params": study.best_params,
                "trial_params": best.user_attrs.get("trial_params", {}),
                "exported_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            with open(best_json_path, "w", encoding="utf-8") as f:
                json.dump(best_payload, f, indent=2, default=str)
            print(f"\nBest F1: {best.value:.4f} (trial {best.number})")
        except ValueError:
            best_payload = {
                "study_name": args.study_name,
                "error": "no_completed_trials",
                "exported_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            with open(best_json_path, "w", encoding="utf-8") as f:
                json.dump(best_payload, f, indent=2, default=str)
            print("\nNo successful Optuna trials (all failed or -inf).")
        print(f"Wrote {best_json_path}, {csv_path}, {jsonl_path}")

    else:
        trial_rows = []
        best_f1 = float("-inf")
        best_payload = None
        for i in range(args.n_trials):
            p = suggest_trial_random(rng)
            exp_id = f"autosearch_{args.study_name}_r{i}"
            ns = params_to_namespace(
                p,
                exp_id=exp_id,
                log_base=args.log_base,
                epochs=args.epochs,
                patience=args.patience,
                device=device,
                data_suffix=args.data_suffix,
                no_online_augment=args.no_online_augment,
                attention_heads=args.attention_heads,
            )
            f1, err = _run_one_trial(ns)
            row = {
                "trial": i,
                "value": f1,
                "error": err or "",
                "exp_id": exp_id,
                **asdict(p),
            }
            trial_rows.append(row)
            with open(jsonl_path, "a", encoding="utf-8") as jf:
                jf.write(json.dumps(row, default=str) + "\n")
            if f1 > best_f1 and err is None:
                best_f1 = f1
                best_payload = {"best_value": f1, "trial": i, "params": asdict(p), "exp_id": exp_id}
            print(f"  Random trial {i} F1={f1:.4f}" + (f" err={err[:80]}" if err else ""))

        export_trials_csv(csv_path, trial_rows)
        if best_payload:
            with open(best_json_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "study_name": args.study_name,
                        "sampler": "random",
                        "seed": args.seed,
                        **best_payload,
                    },
                    f,
                    indent=2,
                    default=str,
                )
        print(f"\nWrote {best_json_path}, {csv_path}, {jsonl_path}")


if __name__ == "__main__":
    main()
