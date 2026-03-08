"""Run overfitting experiments for Microglia (CERAD + Braak classification).
Grid: experiments x gene_encoder x attention x pca. Saves to Microglia_{region}/overfitting_experiments.csv."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import csv
from types import SimpleNamespace

from train_model_Microglia import main as train_main

EXPERIMENTS = [
    {"name": "baseline", "dropout": 0.2, "weight_decay": 0.1, "lr": 0.001, "lambda_entropy": 1e-3},
    {"name": "higher_dropout", "dropout": 0.4, "weight_decay": 0.1, "lr": 0.001, "lambda_entropy": 1e-3},
    {"name": "highest_dropout", "dropout": 0.5, "weight_decay": 0.1, "lr": 0.001, "lambda_entropy": 1e-3},
    {"name": "higher_weight_decay", "dropout": 0.2, "weight_decay": 0.5, "lr": 0.001, "lambda_entropy": 1e-3},
    {"name": "highest_weight_decay", "dropout": 0.2, "weight_decay": 1.0, "lr": 0.001, "lambda_entropy": 1e-3},
    {"name": "lower_lr", "dropout": 0.2, "weight_decay": 0.1, "lr": 5e-4, "lambda_entropy": 1e-3},
    {"name": "lowest_lr", "dropout": 0.2, "weight_decay": 0.1, "lr": 1e-4, "lambda_entropy": 1e-3},
    {"name": "higher_lambda_entropy", "dropout": 0.2, "weight_decay": 0.1, "lr": 0.001, "lambda_entropy": 1e-2},
    {"name": "highest_lambda_entropy", "dropout": 0.2, "weight_decay": 0.1, "lr": 0.001, "lambda_entropy": 1e-1},
    {"name": "dropout_wd", "dropout": 0.4, "weight_decay": 0.5, "lr": 0.001, "lambda_entropy": 1e-3},
    {"name": "dropout_lr", "dropout": 0.4, "weight_decay": 0.1, "lr": 5e-4, "lambda_entropy": 1e-3},
    {"name": "wd_lr", "dropout": 0.2, "weight_decay": 0.5, "lr": 5e-4, "lambda_entropy": 1e-3},
    {"name": "all_moderate", "dropout": 0.4, "weight_decay": 0.5, "lr": 5e-4, "lambda_entropy": 1e-2},
    {"name": "all_strong", "dropout": 0.5, "weight_decay": 1.0, "lr": 1e-4, "lambda_entropy": 1e-1},
]
GENE_ENCODER_OPTS = [True, False]
ATTENTION_OPTS = [True, False]
PCA_OPTS = [2, 4, 8, 16]


def build_args(region, exp, gene_encoder, attention, pca_components, base_args):
    # Gene encoder always uses pooling=None; else use --pooling if set, default "mean"
    if gene_encoder:
        pooling = None
    else:
        pooling_opt = getattr(base_args, "pooling", None)
        pooling = None if (pooling_opt and str(pooling_opt).lower() == "none") else (pooling_opt or "mean")
    exp_id = f"ge{gene_encoder}_attn{attention}_pca{pca_components}_{exp['name']}"
    log_base = getattr(base_args, "log_base", None) or f"Microglia_{region}"
    return SimpleNamespace(
        region=region,
        cell_type="Microglia",
        pooling=pooling,
        loss_ptau=None,  # Microglia: Braak/CERAD are categorical (no ptau regression)
        batch_size=base_args.batch_size,
        epochs=base_args.epochs,
        lr=exp["lr"],
        device=base_args.device,
        dropout=exp["dropout"],
        weight_decay=exp["weight_decay"],
        attention=attention,
        attention_heads=getattr(base_args, "attention_heads", 1),
        pca_components=pca_components,
        gene_encoder=gene_encoder,
        lambda_entropy=exp["lambda_entropy"],
        log_base=log_base,
        exp_id=exp_id,
        patience=getattr(base_args, "patience", 20),
        n_hidden_layers=getattr(base_args, "n_hidden_layers", 1),
        hidden_dim_3layer=getattr(base_args, "hidden_dim_3layer", 32),
        best_f1=getattr(base_args, "best_f1", False),
    )


def run_experiments(args):
    region = args.region
    log_base = getattr(args, "log_base", None) or f"Microglia_{region}"
    output_csv = os.path.join(log_base, "overfitting_experiments.csv")
    os.makedirs(log_base, exist_ok=True)

    experiments = EXPERIMENTS
    gene_encoder_opts = GENE_ENCODER_OPTS
    attention_opts = ATTENTION_OPTS
    pca_opts = PCA_OPTS
    n_runs = len(EXPERIMENTS) * len(GENE_ENCODER_OPTS) * len(ATTENTION_OPTS) * len(PCA_OPTS)

    results = []
    best_f1 = -1.0

    print(f"\n{'='*70}")
    print(f"Microglia overfitting experiments: {n_runs} runs")
    print(f"Region={region}. Logs: {log_base}/")
    print(f"Results: {output_csv}")
    print(f"{'='*70}\n")

    run_idx = 0
    for gene_encoder in gene_encoder_opts:
        attn_opts = [False] if gene_encoder else attention_opts
        for attention in attn_opts:
            for pca_components in pca_opts:
                for exp in experiments:
                    run_idx += 1
                    exp_name = exp["name"]
                    run_name = f"ge{gene_encoder}_attn{attention}_pca{pca_components}_{exp_name}"
                    print(f"\n{'#'*70}")
                    print(f"Run {run_idx}/{n_runs}: {run_name}")
                    print(f"  dropout={exp['dropout']}, weight_decay={exp['weight_decay']}, lr={exp['lr']}")
                    print(f"{'#'*70}\n")

                    train_args = build_args(region, exp, gene_encoder, attention, pca_components, args)
                    try:
                        avg_f1, avg_mse, avg_mae, _ = train_main(train_args)
                    except Exception as e:
                        print(f"  ERROR: {e}")
                        results.append({
                            "region": region,
                            "cell_type": "Microglia",
                            "run": run_name,
                            "experiment": exp_name,
                            "gene_encoder": gene_encoder,
                            "attention": attention,
                            "pca_components": pca_components,
                            "dropout": exp["dropout"],
                            "weight_decay": exp["weight_decay"],
                            "lr": exp["lr"],
                            "lambda_entropy": exp["lambda_entropy"],
                            "avg_f1": None,
                            "avg_mse": None,
                            "avg_mae": None,
                            "error": str(e),
                        })
                        continue
                    results.append({
                        "region": region,
                        "cell_type": "Microglia",
                        "run": run_name,
                        "experiment": exp_name,
                        "gene_encoder": gene_encoder,
                        "attention": attention,
                        "pca_components": pca_components,
                        "dropout": exp["dropout"],
                        "weight_decay": exp["weight_decay"],
                        "lr": exp["lr"],
                        "lambda_entropy": exp["lambda_entropy"],
                        "avg_f1": avg_f1,
                        "avg_mse": avg_mse,
                        "avg_mae": avg_mae,
                        "error": "",
                    })
                    if avg_f1 > best_f1:
                        best_f1 = avg_f1
                        print(f"  New best F1: {best_f1:.4f}")

    fieldnames = [
        "region", "cell_type", "run", "experiment", "gene_encoder", "attention", "pca_components",
        "avg_f1", "avg_mse", "avg_mae", "dropout", "weight_decay", "lr", "lambda_entropy", "error",
    ]
    with open(output_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(results)
    print(f"\nWrote {len(results)} rows to {output_csv}")

    valid = [r for r in results if r.get("avg_f1") is not None]
    if not valid:
        print("No successful runs for 'Best model per combo'.")
    else:
        from collections import defaultdict
        groups = defaultdict(list)
        for r in valid:
            key = (r["gene_encoder"], r["attention"], r["pca_components"])
            groups[key].append(r)
        best_per_combo = [max(g, key=lambda x: x["avg_f1"]) for g in groups.values()]
        print(f"\n{'='*60}")
        print("Best model per (gene_encoder, attention, pca_components):")
        print(f"{'='*60}")
        for r in sorted(best_per_combo, key=lambda x: (-x["avg_f1"], str(x["run"]))):
            print(f"  {r['run'][:50]}  F1={r['avg_f1']:.4f}")
        print(f"{'='*60}")
    print(f"Results saved to: {output_csv}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Microglia overfitting experiments (CERAD + Braak)")
    parser.add_argument("--region", type=str, required=True, help="Region (e.g., EC, ITG, PFC, MTG). MTG uses Microglia-style two-head model with Thal/Braak.")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    parser.add_argument("--attention_heads", type=int, default=1)
    parser.add_argument("--log_base", type=str, default=None, help="Base dir for logs and CSV (default: Microglia_{region})")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience (default: 20). Set to 0 to disable.")
    parser.add_argument("--n_hidden_layers", type=int, default=1, choices=[1, 3], help="Baseline MLP: 1 or 3 hidden layers (default: 1).")
    parser.add_argument("--hidden_dim_3layer", type=int, default=32, help="Hidden size when n_hidden_layers=3 (default: 32).")
    parser.add_argument("--best_f1", action="store_true", help="Use best F1 across epochs; default is last-epoch F1.")
    parser.add_argument("--pooling", type=str, default=None,
                        help="Pooling when not using gene_encoder: mean, max, or none (no pooling, 3D input). Default: mean. Ignored when gene_encoder=True.")
    args = parser.parse_args()
    run_experiments(args)
