"""Run overfitting experiments on embedding data: full grid of experiment x gene_encoder x attention x loss x pca.
Saves CSV to {cell_type}_{region}/overfitting_experiments.csv (or --log_base parent).
When region is MTG, delegates to Microglia two-head model (CERAD + Braak) using MTG_data_for_model_training.pkl."""
import os
import sys
import warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import csv
from types import SimpleNamespace

from train_model import main as train_main


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

"""EXPERIMENTS = [
    {"name": "baseline", "dropout": 0.2, "weight_decay": 0.1, "lr": 0.001, "lambda_entropy": 1e-3},
    {"name": "higher_weight_decay", "dropout": 0.2, "weight_decay": 0.5, "lr": 0.001, "lambda_entropy": 1e-3},
    {"name": "all_moderate", "dropout": 0.4, "weight_decay": 0.5, "lr": 5e-4, "lambda_entropy": 1e-2}
]"""

"""EXPERIMENTS = [
        {"name": "highest_weight_decay", "dropout": 0.2, "weight_decay": 1.0, "lr": 0.001, "lambda_entropy": 1e-3},
]"""

GENE_ENCODER_OPTS = [False, True]
ATTENTION_OPTS = [True, False]
LOSS_PTAU_OPTS = ["huber", "mse", "mae", "none"]
LOSS_PTAU_OPTS = ["none"]
PCA_OPTS = [2, 4, 8, 16]

def build_args(region, cell_type, exp, gene_encoder, attention, loss_ptau, pca_components, base_args):
    # Gene encoder always uses pooling=None; else use --pooling if set, default "mean"
    if gene_encoder:
        pooling = None
    else:
        pooling_opt = getattr(base_args, "pooling", None)
        pooling = None if (pooling_opt and str(pooling_opt).lower() == "none") else (pooling_opt or "mean")
    mlp_raw = getattr(base_args, "mlp_hidden_sizes", "") or ""
    mlp_tag = ""
    if str(mlp_raw).strip():
        mlp_tag = "_mlp" + "_".join(p.strip() for p in str(mlp_raw).split(",") if p.strip())
    exp_id = f"ge{gene_encoder}_attn{attention}_{loss_ptau}_pca{pca_components}{mlp_tag}_{exp['name']}"
    log_base = getattr(base_args, "log_base", None) or f"{cell_type}_{region}"
    return SimpleNamespace(
        region=region,
        cell_type=cell_type,
        pooling=pooling,
        loss_ptau=loss_ptau,
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
        verbose=False,
        n_hidden_layers=getattr(base_args, "n_hidden_layers", 1),
        hidden_dim_3layer=getattr(base_args, "hidden_dim_3layer", 32),
        mlp_hidden_sizes=getattr(base_args, "mlp_hidden_sizes", ""),
        patience=getattr(base_args, "patience", 20),
        best_f1=getattr(base_args, "best_f1", False),
        thal_ordinal=getattr(base_args, "thal_ordinal", False),
        top_genes=getattr(base_args, "top_genes", None),
        ranked_genes_path=getattr(base_args, "ranked_genes_path", None),
        save_best_model=getattr(base_args, "save_best_model", False),
        save_dir=getattr(base_args, "save_dir", "models"),
        data_suffix=getattr(base_args, "data_suffix", ""),
        no_online_augment=getattr(base_args, "no_online_augment", False),
        aug_scale_jitter=getattr(base_args, "aug_scale_jitter", 0.0),
        aug_per_gene_scale_jitter=getattr(base_args, "aug_per_gene_scale_jitter", 0.0),
        aug_gene_dropout=getattr(base_args, "aug_gene_dropout", 0.0),
        aug_emb_dim_dropout=getattr(base_args, "aug_emb_dim_dropout", 0.0),
        aug_feature_dropout=getattr(base_args, "aug_feature_dropout", 0.0),
        aug_gene_block_cutout_max=getattr(base_args, "aug_gene_block_cutout_max", 0),
        aug_gene_block_cutout_prob=getattr(base_args, "aug_gene_block_cutout_prob", 0.5),
        aug_noise_std=getattr(base_args, "aug_noise_std", 0.0),
        mixup_alpha=getattr(base_args, "mixup_alpha", 0.0),
        gene_cutmix_prob=getattr(base_args, "gene_cutmix_prob", 0.0),
        gene_cutmix_alpha=getattr(base_args, "gene_cutmix_alpha", 1.0),
        label_smoothing=getattr(base_args, "label_smoothing", 0.0),
    )


def run_experiments(args):
    region = args.region
    # MTG and CrossRegion use the Microglia two-head model and their respective pkls
    if region == "MTG":
        from run_overfitting_experiments_microglia import run_experiments as run_microglia_experiments
        run_microglia_experiments(args)
        return
    if region == "CrossRegion":
        from run_overfitting_experiments_microglia import run_experiments as run_microglia_experiments
        run_microglia_experiments(args)
        return

    cell_type = getattr(args, "cell_type", "Astrocytes")
    log_base = getattr(args, "log_base", None) or f"{cell_type}_{region}"
    output_csv = os.path.join(log_base, "overfitting_experiments.csv")

    os.makedirs(log_base, exist_ok=True)

    experiments = EXPERIMENTS
    gene_encoder_opts = GENE_ENCODER_OPTS
    loss_opts = LOSS_PTAU_OPTS
    pca_opts = PCA_OPTS
    n_runs = len(EXPERIMENTS) * (
        len(ATTENTION_OPTS) * len(LOSS_PTAU_OPTS) * len(PCA_OPTS)
        + len(LOSS_PTAU_OPTS) * len(PCA_OPTS)
    )
    results = []
    best_f1 = -1.0

    print(f"\n{'='*70}")
    print(f"Embedding experiments: {n_runs} runs")
    print(f"Region={region}, cell_type={cell_type}. Logs: {log_base}/")
    print(f"Results: {output_csv}")
    print(f"{'='*70}\n")

    run_idx = 0
    for gene_encoder in gene_encoder_opts:
        attention_opts = ATTENTION_OPTS if not gene_encoder else [False]
        for attention in attention_opts:
            for loss_ptau in loss_opts:
                for pca_components in pca_opts:
                    for exp in experiments:
                        run_idx += 1
                        exp_name = exp["name"]
                        run_name = f"ge{gene_encoder}_attn{attention}_{loss_ptau}_pca{pca_components}_{exp_name}"
                        print(f"\n{'#'*70}")
                        print(f"Run {run_idx}/{n_runs}: {run_name}")
                        print(f"  dropout={exp['dropout']}, weight_decay={exp['weight_decay']}, lr={exp['lr']}")
                        print(f"{'#'*70}\n")

                        train_args = build_args(
                            region, cell_type, exp, gene_encoder, attention, loss_ptau, pca_components, args
                        )
                        try:
                            avg_f1, avg_mse, avg_mae, _ = train_main(train_args)
                        except Exception as e:
                            print(f"  ERROR: {e}")
                            results.append({
                                "region": region,
                                "cell_type": cell_type,
                                "run": run_name,
                                "experiment": exp_name,
                                "gene_encoder": gene_encoder,
                                "attention": attention,
                                "loss_ptau": loss_ptau,
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
                            "cell_type": cell_type,
                            "run": run_name,
                            "experiment": exp_name,
                            "gene_encoder": gene_encoder,
                            "attention": attention,
                            "loss_ptau": loss_ptau,
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
        "region", "cell_type", "run", "experiment", "gene_encoder", "attention", "loss_ptau", "pca_components",
        "avg_f1", "avg_mse", "avg_mae", "dropout", "weight_decay", "lr", "lambda_entropy", "error",
    ]
    with open(output_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(results)
    print(f"\nWrote {len(results)} rows to {output_csv}")

    # Best per (gene_encoder, attention, loss_ptau, pca_components)
    valid = [r for r in results if r.get("avg_f1") is not None]
    if not valid:
        print("No successful runs for 'Best model per combo'.")
    else:
        from collections import defaultdict
        groups = defaultdict(list)
        for r in valid:
            key = (r["gene_encoder"], r["attention"], r["loss_ptau"], r["pca_components"])
            groups[key].append(r)
        best_per_combo = [max(g, key=lambda x: x["avg_f1"]) for g in groups.values()]
        print(f"\n{'='*60}")
        print("Best model per (gene_encoder, attention, loss_ptau, pca_components):")
        print(f"{'='*60}")
        for r in sorted(best_per_combo, key=lambda x: (-x["avg_f1"], str(x["run"]))):
            print(f"  {r['run'][:50]}  F1={r['avg_f1']:.4f}, MSE={r['avg_mse']:.4f}, MAE={r['avg_mae']:.4f}")
        print(f"{'='*60}")
    print(f"Results saved to: {output_csv}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run overfitting experiments on embedding data")
    parser.add_argument("--region", type=str, required=True, help="Region (e.g., EC, ITG, PFC, MTG). MTG runs Microglia two-head model.")
    parser.add_argument("--cell_type", type=str, default="Astrocytes", help="Cell type (use run_overfitting_experiments_microglia.py for Microglia)")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    parser.add_argument("--attention_heads", type=int, default=1)
    parser.add_argument("--log_base", type=str, default=None, help="Base dir for logs and CSV (default: {cell_type}_{region})")
    parser.add_argument("--n_hidden_layers", type=int, default=1, choices=[1, 3],
                        help="Number of hidden layers in baseline MLP: 1 (minimal) or 3 (original). Default: 1.")
    parser.add_argument("--hidden_dim_3layer", type=int, default=32,
                        help="Hidden size when n_hidden_layers=3 (default: 32).")
    parser.add_argument(
        "--mlp_hidden_sizes",
        type=str,
        default="",
        help="Baseline MLP widths, comma-separated, e.g. 64,32,16,8,4. Overrides --n_hidden_layers when set.",
    )
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience (default: 20).")
    parser.add_argument("--best_f1", action="store_true", help="Use best F1 across epochs; default is last-epoch F1.")
    parser.add_argument("--pooling", type=str, default=None,
                        help="Pooling when not using gene_encoder: mean, max, or none (no pooling, 3D input). Default: mean. Ignored when gene_encoder=True.")
    parser.add_argument("--thal_ordinal", action="store_true",
                        help="Treat Thal as ordinal (K-1 cumulative logits + BCE loss).")
    parser.add_argument("--top_genes", type=int, default=None,
                        help="Use only top N genes from ranked_genes.csv (filter at train time; default: all genes).")
    parser.add_argument("--ranked_genes_path", type=str, default=None,
                        help="Path to ranked_genes.csv. Default when --top_genes set: ranked_lists/EC_Astrocytes_embedding_highest_lambda_entropy/ranked_genes.csv")
    parser.add_argument("--save_best_model", action="store_true",
                        help="Save best model to --save_dir per test fold (default: models/).")
    parser.add_argument("--save_dir", type=str, default="models",
                        help="Directory for saved best models when --save_best_model (default: models).")
    parser.add_argument("--data_suffix", type=str, default="",
                        help="Suffix for model-data .pkl (avoid _augScaling; use --aug_* online augmentation).")
    parser.add_argument("--use_augScaling", action="store_true",
                        help="DEPRECATED: sets --data_suffix _augScaling. Prefer base .pkl + --aug_* flags.")
    parser.add_argument("--no_online_augment", action="store_true",
                        help="Disable EmbeddingDataset online augmentation.")
    parser.add_argument("--aug_scale_jitter", type=float, default=0.0, help="Train-only global scale U(1-a,1+a).")
    parser.add_argument("--aug_per_gene_scale_jitter", type=float, default=0.0, help="Train-only per-gene scale (3D).")
    parser.add_argument("--aug_gene_dropout", type=float, default=0.0, help="Gene-row dropout prob (3D).")
    parser.add_argument("--aug_emb_dim_dropout", type=float, default=0.0, help="Embedding-dim dropout (3D or 2D).")
    parser.add_argument("--aug_feature_dropout", type=float, default=0.0, help="Feature dropout (2D pooled).")
    parser.add_argument("--aug_gene_block_cutout_max", type=int, default=0, help="Max genes in contiguous cutout (3D).")
    parser.add_argument("--aug_gene_block_cutout_prob", type=float, default=0.5, help="Apply cutout prob when max>0.")
    parser.add_argument("--aug_noise_std", type=float, default=0.0, help="Additive Gaussian noise std (train).")
    parser.add_argument("--mixup_alpha", type=float, default=0.0, help="Mixup Beta(alpha,alpha); 0=off.")
    parser.add_argument("--gene_cutmix_prob", type=float, default=0.0, help="Per-batch gene CutMix prob (3D).")
    parser.add_argument("--gene_cutmix_alpha", type=float, default=1.0, help="Beta for CutMix cut length.")
    parser.add_argument("--label_smoothing", type=float, default=0.0, help="Thal CrossEntropy label smoothing.")
    args = parser.parse_args()
    if getattr(args, "use_augScaling", False):
        args.data_suffix = "_augScaling"
        warnings.warn(
            "--use_augScaling loads offline 3× augmented pickles; deprecated. "
            "Use base data_for_model_training.pkl with --aug_scale_jitter / --aug_noise_std / etc.",
            UserWarning,
            stacklevel=2,
        )
    run_experiments(args)
