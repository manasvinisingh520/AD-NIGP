"""Run overfitting experiments for Microglia on expression data (CERAD + Braak classification).
Grid: experiments x attention x pca. Saves to Microglia_{region}_Expression/microglia_expression_experiments.csv."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import csv
from types import SimpleNamespace

from train_baseline_rankingModel_expression_Microglia import main as expression_main

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
ATTENTION_OPTS = [True, False]
PCA_OPTS = [2, 4, 8, None]


def build_args(region, exp, attention, pca_components, base_args):
    exp_id = f"attn{attention}_pca{pca_components}_{exp['name']}"
    log_base = getattr(base_args, "log_base", None) or f"Microglia_{region}_Expression"
    return SimpleNamespace(
        region=region,
        cell_type="Microglia",
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
        log_base=log_base,
        exp_id=exp_id,
        n_hidden_layers=getattr(base_args, "n_hidden_layers", 1),
        hidden_dim_3layer=getattr(base_args, "hidden_dim_3layer", 32),
        patience=getattr(base_args, "patience", 20),
        best_f1=getattr(base_args, "best_f1", False),
    )


def run_experiments(args):
    region = args.region
    log_base_dir = getattr(args, "log_base", None) or f"Microglia_{region}_Expression"
    output_csv = args.output or os.path.join(log_base_dir, "microglia_expression_experiments.csv")
    os.makedirs(log_base_dir, exist_ok=True)

    experiments = EXPERIMENTS
    attention_opts = ATTENTION_OPTS
    pca_opts = PCA_OPTS
    n_runs = len(experiments) * len(attention_opts) * len(pca_opts)

    results = []
    best_f1 = -1.0

    print(f"\n{'='*70}")
    print(f"Microglia expression experiments: {n_runs} runs")
    print(f"Region={region}. Logs: {log_base_dir}/")
    print(f"Results: {output_csv}")
    print(f"{'='*70}\n")

    run_idx = 0
    for exp in experiments:
        exp_name = exp["name"]
        for attention in attention_opts:
            for pca_components in pca_opts:
                run_idx += 1
                run_name = f"{exp_name}_attn{attention}_pca{pca_components}"
                print(f"\n{'#'*70}")
                print(f"Run {run_idx}/{n_runs}: {run_name}")
                print(f"  dropout={exp['dropout']}, weight_decay={exp['weight_decay']}, lr={exp['lr']}")
                print(f"{'#'*70}\n")

                train_args = build_args(region, exp, attention, pca_components, args)
                try:
                    avg_f1, avg_mse, avg_mae, _ = expression_main(train_args)
                except Exception as e:
                    print(f"  ERROR: {e}")
                    results.append({
                        "run": run_name,
                        "experiment": exp_name,
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
                    "run": run_name,
                    "experiment": exp_name,
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
        "run", "experiment", "attention", "pca_components",
        "dropout", "weight_decay", "lr", "lambda_entropy",
        "avg_f1", "avg_mse", "avg_mae", "error",
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
            key = (r["attention"], r["pca_components"])
            groups[key].append(r)
        best_per_combo = [max(g, key=lambda x: x["avg_f1"]) for g in groups.values()]
        print(f"\n{'='*60}")
        print("Best model per (attention, pca_components):")
        print(f"{'='*60}")
        for r in sorted(best_per_combo, key=lambda x: -x["avg_f1"]):
            print(f"  attn={r['attention']}, pca={r['pca_components']}  ->  {r['run']}  F1={r['avg_f1']:.4f}")
        print(f"{'='*60}")
    print(f"Results saved to: {output_csv}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Microglia overfitting experiments on expression data (CERAD + Braak)")
    parser.add_argument("--region", type=str, required=True, help="Region (e.g., CrossRegion, ITG, PFC)")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    parser.add_argument("--attention_heads", type=int, default=1)
    parser.add_argument("--log_base", type=str, default=None,
                        help="Base dir for logs and CSV (default: Microglia_{region}_Expression)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV path (default: Microglia_{region}_Expression/microglia_expression_experiments.csv)")
    parser.add_argument("--n_hidden_layers", type=int, default=1, choices=[1, 3],
                        help="Number of hidden layers in baseline MLP: 1 (minimal) or 3 (original). Default: 1.")
    parser.add_argument("--hidden_dim_3layer", type=int, default=32,
                        help="Hidden size when n_hidden_layers=3 (default: 32).")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience (default: 20). Set to 0 to disable.")
    parser.add_argument("--best_f1", action="store_true", help="Use best F1 across epochs; default is last-epoch F1.")
    args = parser.parse_args()
    run_experiments(args)
