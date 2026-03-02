"""
Run overfitting experiments on **combined** Astrocytes data: pool all regions (EC, ITG, PFC, V2, V1), then 8-fold CV.

- Combines regions from config REGIONS. Labels: Thal + ptau.
- 8-fold: Stratified split on Thal. Runs both embedding and expression grids.
- Outputs: overfitting_experiments_combined_embedding.csv and overfitting_experiments_combined_expression.csv in --log_base (default: combined_Astrocytes).

For Microglia combined (CrossRegion, ITG, PFC), use run_overfitting_experiments_combined_data_microglia.py.
"""
import os
import sys
import pickle
import argparse
import csv
import numpy as np
from types import SimpleNamespace
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import REGIONS
from create_raw_expression_data import load_top_genes_from_dge_union
from create_data import create_anndata, load_gene_embeddings

DATA_DIR = os.path.join("data", "model_data")
EMBEDDINGS_DIR = os.path.join("GREmLN", "embeddings")
DEFAULT_TOP_K_GENES = 3000
DEFAULT_DGE_RESULTS_DIR = "E:/sf2026/dge_results/dge_final"

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
GENE_ENCODER_OPTS = [False, True]
ATTENTION_OPTS = [True, False]
LOSS_PTAU_OPTS = ["none"]
PCA_OPTS = [2, 4, 8, 16]
ATTENTION_OPTS_EXPR = [True, False]
LOSS_PTAU_OPTS_EXPR = ["none"]
PCA_OPTS_EXPR = [None]


def _get_common_genes_embedding(regions, cell_type="Astrocytes"):
    """
    Get gene list per region in same order as in data_for_model_training.pkl (create_data pipeline).
    Returns (common_genes_ordered, region_to_gene_indices) where region_to_gene_indices[region]
    is the array of row indices (gene axis) for that region's (n_genes, n_dims) embedding.
    """
    region_genes = {}
    for region in regions:
        data_path = os.path.join(DATA_DIR, f"{region}_data_for_model_training.pkl")
        if not os.path.isfile(data_path):
            continue
        emb_path = os.path.join(EMBEDDINGS_DIR, f"{region}_gene_embeddings.tsv")
        if not os.path.isfile(emb_path):
            print(f"  Skip {region}: embedding file not found {emb_path}")
            continue
        try:
            adata = create_anndata(region, cell_type)
            embeddings_dict = load_gene_embeddings(emb_path, adata.var_names.tolist())
            ordered_genes = [g for g in adata.var_names if g in embeddings_dict]
            region_genes[region] = ordered_genes
        except Exception as e:
            print(f"  Skip {region}: could not get embedding gene list - {e}")
            continue

    if not region_genes:
        return None, {}

    common = set(region_genes[list(region_genes.keys())[0]])
    for genes in region_genes.values():
        common &= set(genes)
    common_ordered = sorted(common)

    region_to_gene_indices = {}
    for region, genes in region_genes.items():
        gene_to_row = {g: i for i, g in enumerate(genes)}
        region_to_gene_indices[region] = np.array([gene_to_row[g] for g in common_ordered], dtype=np.intp)
    return common_ordered, region_to_gene_indices


def _get_common_genes_and_column_indices(regions, cell_type="Astrocytes", top_k_genes=DEFAULT_TOP_K_GENES, dge_results_dir=DEFAULT_DGE_RESULTS_DIR):
    """Get common genes across regions for expression (DGE union order). Returns (common_genes_ordered, region_to_col_indices)."""
    region_genes = {}
    for region in regions:
        data_path = os.path.join(DATA_DIR, f"{region}_raw_expression_data.pkl")
        if not os.path.isfile(data_path):
            continue
        try:
            genes = load_top_genes_from_dge_union(
                region=region, cell_type=cell_type, top_k=top_k_genes, dge_results_dir=dge_results_dir
            )
            region_genes[region] = genes
        except Exception as e:
            print(f"  Skip {region}: could not load DGE gene list - {e}")
            continue

    if not region_genes:
        return None, {}

    common = set(region_genes[list(region_genes.keys())[0]])
    for genes in region_genes.values():
        common &= set(genes)
    common_ordered = sorted(common)
    region_to_col_indices = {}
    for region, genes in region_genes.items():
        gene_to_col = {g: i for i, g in enumerate(genes)}
        region_to_col_indices[region] = np.array([gene_to_col[g] for g in common_ordered], dtype=np.intp)
    return common_ordered, region_to_col_indices


def load_combined_embedding_data(regions, n_folds=8):
    """
    Load Astrocytes embedding data from all regions (patient-level 3D), restrict to genes common to all regions,
    split into n_folds stratified by Thal. Returns: all_fold_patient_data, thal_to_idx, idx_to_thal, n_thal_classes.
    """
    common_genes, region_to_gene_indices = _get_common_genes_embedding(regions, cell_type="Astrocytes")
    if not common_genes or not region_to_gene_indices:
        raise FileNotFoundError("No embedding data or gene lists available for any region.")

    n_common = len(common_genes)
    all_embeddings = []
    all_ptau = []
    all_thal_raw = []
    per_region_genes_before = {}

    for region in region_to_gene_indices:
        data_path = os.path.join(DATA_DIR, f"{region}_data_for_model_training.pkl")
        if not os.path.isfile(data_path):
            continue
        with open(data_path, "rb") as f:
            embeddings_folds, patient_info_folds, ptau_folds, thal_folds = pickle.load(f)
        gene_idx = region_to_gene_indices[region]
        if len(embeddings_folds) > 0 and embeddings_folds[0].size > 0:
            per_region_genes_before[region] = embeddings_folds[0].shape[1]

        for fold_idx in range(len(embeddings_folds)):
            fold_emb = embeddings_folds[fold_idx]
            fold_ptau = ptau_folds[fold_idx]
            fold_thal = thal_folds[fold_idx]
            fold_patient_info = patient_info_folds[fold_idx]
            cell_idx = 0
            for patient_idx, (_, cell_names) in enumerate(fold_patient_info.items()):
                n_c = len(cell_names)
                patient_emb = np.mean(fold_emb[cell_idx : cell_idx + n_c], axis=0)
                patient_emb_common = patient_emb[gene_idx, :]
                all_embeddings.append(patient_emb_common)
                all_ptau.append(fold_ptau[patient_idx])
                all_thal_raw.append(fold_thal[patient_idx])
                cell_idx += n_c

    if not all_embeddings:
        raise FileNotFoundError("No embedding data loaded from any region.")

    all_embeddings = np.array(all_embeddings)
    all_ptau = np.array(all_ptau)
    all_thal_int = np.array([int(t) for t in all_thal_raw])
    thal_classes = sorted(set(all_thal_int))
    thal_to_idx = {c: i for i, c in enumerate(thal_classes)}
    idx_to_thal = {i: c for c, i in thal_to_idx.items()}
    n_thal_classes = len(thal_classes)
    all_thal_encoded = np.array([thal_to_idx[t] for t in all_thal_int])
    all_ptau_log = np.log1p(all_ptau)

    before_str = ", ".join(f"{r} {n}" for r, n in sorted(per_region_genes_before.items()))
    print(f"Embedding size: per-region genes (before) = {before_str}; common genes = {n_common}; combined shape = ({len(all_embeddings)}, {n_common}, {all_embeddings.shape[2]})")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    all_fold_patient_data = []
    for _, test_indices in skf.split(np.zeros(len(all_thal_encoded)), all_thal_encoded):
        all_fold_patient_data.append({
            "embeddings": all_embeddings[test_indices],
            "ptau": all_ptau_log[test_indices],
            "thal": all_thal_encoded[test_indices],
        })
    print(f"Combined embedding: {len(all_embeddings)} patients from {list(region_to_gene_indices.keys())}, {n_folds} folds")
    return all_fold_patient_data, thal_to_idx, idx_to_thal, n_thal_classes


def load_combined_expression_data(regions, n_folds=8, top_k_genes=DEFAULT_TOP_K_GENES, dge_results_dir=DEFAULT_DGE_RESULTS_DIR):
    """Load Astrocytes raw expression from all regions; keep only genes common to all regions; stratified by Thal."""
    common_genes, region_to_col_indices = _get_common_genes_and_column_indices(
        regions, cell_type="Astrocytes", top_k_genes=top_k_genes, dge_results_dir=dge_results_dir
    )
    if not common_genes or not region_to_col_indices:
        raise FileNotFoundError("No expression data or DGE gene lists available for any region.")

    n_common = len(common_genes)
    all_X = []
    all_ptau = []
    all_thal_raw = []
    per_region_genes_before = {}

    for region in region_to_col_indices:
        data_path = os.path.join(DATA_DIR, f"{region}_raw_expression_data.pkl")
        if not os.path.isfile(data_path):
            continue
        with open(data_path, "rb") as f:
            expression_folds, _, ptau_folds, thal_folds = pickle.load(f)
        col_idx = region_to_col_indices[region]
        if len(expression_folds) > 0 and expression_folds[0].size > 0:
            per_region_genes_before[region] = expression_folds[0].shape[1]
        for fold_idx in range(len(expression_folds)):
            X = np.array(expression_folds[fold_idx], dtype=np.float32)
            X_common = X[:, col_idx]
            ptau = ptau_folds[fold_idx]
            thal = thal_folds[fold_idx]
            for i in range(len(ptau)):
                all_X.append(X_common[i])
                all_ptau.append(ptau[i])
                all_thal_raw.append(thal[i])

    if not all_X:
        raise FileNotFoundError("No expression data loaded from any region.")

    all_X = np.array(all_X)
    all_ptau = np.array(all_ptau, dtype=np.float32)
    all_thal_int = np.array([int(t) for t in all_thal_raw])
    thal_classes = sorted(set(all_thal_int))
    thal_to_idx = {c: i for i, c in enumerate(thal_classes)}
    idx_to_thal = {i: c for c, i in thal_to_idx.items()}
    n_thal_classes = len(thal_classes)
    all_thal_encoded = np.array([thal_to_idx[t] for t in all_thal_int])
    all_ptau_log = np.log1p(all_ptau)

    before_str = ", ".join(f"{r} {n}" for r, n in sorted(per_region_genes_before.items()))
    print(f"Expression size: per-region genes (before) = {before_str}; common genes = {n_common}; combined shape = ({len(all_X)}, {n_common})")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    all_fold_patient_data = []
    for _, test_indices in skf.split(np.zeros(len(all_thal_encoded)), all_thal_encoded):
        all_fold_patient_data.append({
            "embeddings": all_X[test_indices],
            "ptau": all_ptau_log[test_indices],
            "thal": all_thal_encoded[test_indices],
        })
    print(f"Combined expression: {len(all_X)} patients from {list(region_to_col_indices.keys())}, {n_folds} folds")
    return all_fold_patient_data, thal_to_idx, idx_to_thal, n_thal_classes


def run_embedding_cv_combined(all_fold_patient_data, thal_to_idx, idx_to_thal, n_thal_classes, args):
    """8-fold CV for Astrocytes embedding. Returns avg_f1, avg_mse, avg_mae, fold_results."""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torch.utils.tensorboard import SummaryWriter
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    from train_model import EmbeddingDataset, train_epoch, evaluate
    from baseline_model import BaselineRankingModel
    from gene_encoder_model import GeneEncoderModel, gene_entropy_loss

    device = torch.device(args.device)
    n_folds = len(all_fold_patient_data)
    criterion_ptau = None if getattr(args, "loss_ptau", "mse") in (None, "none") else \
        {"mse": nn.MSELoss(), "mae": nn.L1Loss(), "huber": nn.HuberLoss()}.get(args.loss_ptau, nn.MSELoss())
    criterion_thal = nn.CrossEntropyLoss()

    fold_results = []
    for test_fold_idx in range(n_folds):
        train_fold_indices = [i for i in range(n_folds) if i != test_fold_idx]
        train_emb = np.concatenate([all_fold_patient_data[i]["embeddings"] for i in train_fold_indices], axis=0)
        train_ptau = np.concatenate([all_fold_patient_data[i]["ptau"] for i in train_fold_indices], axis=0)
        train_thal = np.concatenate([all_fold_patient_data[i]["thal"] for i in train_fold_indices], axis=0)
        test_emb = all_fold_patient_data[test_fold_idx]["embeddings"]
        test_ptau = all_fold_patient_data[test_fold_idx]["ptau"]
        test_thal = all_fold_patient_data[test_fold_idx]["thal"]

        n_train, n_test = train_emb.shape[0], test_emb.shape[0]
        if len(train_emb.shape) == 3:
            if not getattr(args, "gene_encoder", False):
                train_emb = np.mean(train_emb, axis=1)
                test_emb = np.mean(test_emb, axis=1)
        if len(train_emb.shape) == 3:
            n_genes, n_dims = train_emb.shape[1], train_emb.shape[2]
            train_2d = train_emb.reshape(n_train, -1)
            test_2d = test_emb.reshape(n_test, -1)
            scaler = StandardScaler()
            train_emb = scaler.fit_transform(train_2d).reshape(n_train, n_genes, n_dims)
            test_emb = scaler.transform(test_2d).reshape(n_test, n_genes, n_dims)
            if getattr(args, "pca_components", None) is not None:
                pc = args.pca_components
                pca = PCA(n_components=pc)
                train_emb = pca.fit_transform(train_emb.reshape(-1, n_dims)).reshape(n_train, n_genes, pc)
                test_emb = pca.transform(test_emb.reshape(-1, n_dims)).reshape(n_test, n_genes, pc)
        else:
            scaler = StandardScaler()
            train_emb = scaler.fit_transform(train_emb)
            test_emb = scaler.transform(test_emb)
            if getattr(args, "pca_components", None) is not None:
                pca = PCA(n_components=args.pca_components)
                train_emb = pca.fit_transform(train_emb)
                test_emb = pca.transform(test_emb)

        train_ds = EmbeddingDataset(train_emb, train_ptau, train_thal)
        test_ds = EmbeddingDataset(test_emb, test_ptau, test_thal)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

        if getattr(args, "gene_encoder", False):
            n_genes, n_dims = train_emb.shape[1], train_emb.shape[2]
            model = GeneEncoderModel(
                n_dims=n_dims, n_thal_classes=n_thal_classes, dropout=args.dropout, n_genes=n_genes
            ).to(device)
        else:
            input_shape = (train_emb.shape[1], train_emb.shape[2]) if len(train_emb.shape) == 3 else None
            input_dim = None if input_shape else train_emb.shape[1]
            model = BaselineRankingModel(
                input_dim=input_dim, input_shape=input_shape, n_thal_classes=n_thal_classes,
                use_attention=getattr(args, "attention", False),
                num_heads=getattr(args, "attention_heads", 1) if getattr(args, "attention", False) else 1,
                dropout=args.dropout,
                n_hidden_layers=getattr(args, "n_hidden_layers", 1),
                hidden_dim_3layer=getattr(args, "hidden_dim_3layer", 32),
            ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        log_dir = os.path.join(args.log_base, f"combined_embedding_{args.exp_id}_testfold{test_fold_idx+1}")
        if os.path.exists(log_dir):
            try:
                import shutil
                shutil.rmtree(log_dir)
            except Exception:
                pass
        writer = SummaryWriter(log_dir)

        best_test_loss = float("inf")
        best_test_f1 = float("-inf")
        best_test_mse = best_test_mae = np.nan
        best_state = None
        patience = getattr(args, "patience", 20) or 999
        epochs_without_improvement = 0
        lambda_entropy = getattr(args, "lambda_entropy", 0.0) if getattr(args, "gene_encoder", False) else 0.0

        for epoch in range(args.epochs):
            train_loss, _, _ = train_epoch(
                model, train_loader, criterion_ptau, criterion_thal, optimizer, device, lambda_entropy=lambda_entropy
            )
            test_loss, test_mse, test_mae, test_f1 = evaluate(
                model, test_loader, criterion_ptau, criterion_thal, device, idx_to_thal
            )
            writer.add_scalar("Loss/Train", train_loss, epoch)
            writer.add_scalar("Loss/Test", test_loss, epoch)
            writer.add_scalar("Metrics/Test_F1", test_f1, epoch)
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                best_test_f1 = test_f1
                best_test_mse = test_mse
                best_test_mae = test_mae
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break
        writer.close()
        if best_state is not None:
            model.load_state_dict(best_state)
        fold_results.append({
            "best_f1": best_test_f1,
            "best_test_mse": best_test_mse,
            "best_test_mae": best_test_mae,
        })

    avg_f1 = float(np.mean([r["best_f1"] for r in fold_results]))
    avg_mse = float(np.nanmean([r["best_test_mse"] for r in fold_results]))
    avg_mae = float(np.nanmean([r["best_test_mae"] for r in fold_results]))
    return avg_f1, avg_mse, avg_mae, fold_results


def run_expression_cv_combined(all_fold_patient_data, thal_to_idx, idx_to_thal, n_thal_classes, args):
    """8-fold CV for Astrocytes expression. Returns avg_f1, avg_mse, avg_mae, fold_results."""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torch.utils.tensorboard import SummaryWriter
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    from train_model import EmbeddingDataset, train_epoch, evaluate
    from baseline_model import BaselineRankingModel

    device = torch.device(args.device)
    n_folds = len(all_fold_patient_data)
    criterion_ptau = None if getattr(args, "loss_ptau", "mse") in (None, "none") else \
        {"mse": nn.MSELoss(), "mae": nn.L1Loss(), "huber": nn.HuberLoss()}.get(args.loss_ptau, nn.MSELoss())
    criterion_thal = nn.CrossEntropyLoss()

    fold_results = []
    for test_fold_idx in range(n_folds):
        train_fold_indices = [i for i in range(n_folds) if i != test_fold_idx]
        train_X = np.concatenate([all_fold_patient_data[i]["embeddings"] for i in train_fold_indices], axis=0)
        train_ptau = np.concatenate([all_fold_patient_data[i]["ptau"] for i in train_fold_indices], axis=0)
        train_thal = np.concatenate([all_fold_patient_data[i]["thal"] for i in train_fold_indices], axis=0)
        test_X = all_fold_patient_data[test_fold_idx]["embeddings"]
        test_ptau = all_fold_patient_data[test_fold_idx]["ptau"]
        test_thal = all_fold_patient_data[test_fold_idx]["thal"]

        scaler = StandardScaler()
        train_X = scaler.fit_transform(train_X)
        test_X = scaler.transform(test_X)
        if getattr(args, "pca_components", None) is not None:
            pca = PCA(n_components=args.pca_components)
            train_X = pca.fit_transform(train_X)
            test_X = pca.transform(test_X)

        train_ds = EmbeddingDataset(train_X, train_ptau, train_thal)
        test_ds = EmbeddingDataset(test_X, test_ptau, test_thal)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

        model = BaselineRankingModel(
            input_dim=train_X.shape[1], input_shape=None, n_thal_classes=n_thal_classes,
            use_attention=getattr(args, "attention", False),
            num_heads=getattr(args, "attention_heads", 1) if getattr(args, "attention", False) else 1,
            dropout=args.dropout,
            n_hidden_layers=getattr(args, "n_hidden_layers", 1),
            hidden_dim_3layer=getattr(args, "hidden_dim_3layer", 32),
        ).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        log_dir = os.path.join(args.log_base, f"combined_expression_{args.exp_id}_testfold{test_fold_idx+1}")
        if os.path.exists(log_dir):
            try:
                import shutil
                shutil.rmtree(log_dir)
            except Exception:
                pass
        writer = SummaryWriter(log_dir)

        best_test_loss = float("inf")
        best_test_f1 = float("-inf")
        best_test_mse = best_test_mae = np.nan
        best_state = None
        patience = getattr(args, "patience", 20) or 999
        epochs_without_improvement = 0

        for epoch in range(args.epochs):
            train_loss, _, _ = train_epoch(model, train_loader, criterion_ptau, criterion_thal, optimizer, device)
            test_loss, test_mse, test_mae, test_f1 = evaluate(
                model, test_loader, criterion_ptau, criterion_thal, device, idx_to_thal
            )
            writer.add_scalar("Loss/Train", train_loss, epoch)
            writer.add_scalar("Loss/Test", test_loss, epoch)
            writer.add_scalar("Metrics/Test_F1", test_f1, epoch)
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                best_test_f1 = test_f1
                best_test_mse = test_mse
                best_test_mae = test_mae
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break
        writer.close()
        if best_state is not None:
            model.load_state_dict(best_state)
        fold_results.append({
            "best_f1": best_test_f1,
            "best_test_mse": best_test_mse,
            "best_test_mae": best_test_mae,
        })

    avg_f1 = float(np.mean([r["best_f1"] for r in fold_results]))
    avg_mse = float(np.nanmean([r["best_test_mse"] for r in fold_results]))
    avg_mae = float(np.nanmean([r["best_test_mae"] for r in fold_results]))
    return avg_f1, avg_mse, avg_mae, fold_results


def build_embedding_args(exp, gene_encoder, attention, loss_ptau, pca_components, base_args):
    exp_id = f"ge{gene_encoder}_attn{attention}_{loss_ptau}_pca{pca_components}_{exp['name']}"
    return SimpleNamespace(
        region="combined",
        cell_type="Astrocytes",
        pooling=None if gene_encoder else "mean",
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
        log_base=base_args.log_base,
        exp_id=exp_id,
        patience=getattr(base_args, "patience", 20),
        n_hidden_layers=getattr(base_args, "n_hidden_layers", 1),
        hidden_dim_3layer=getattr(base_args, "hidden_dim_3layer", 32),
    )


def build_expression_args(exp, attention, loss_ptau, pca_components, base_args):
    exp_id = f"attn{attention}_{loss_ptau}_pca{pca_components}_{exp['name']}"
    return SimpleNamespace(
        region="combined",
        cell_type="Astrocytes",
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
        log_base=base_args.log_base,
        exp_id=exp_id,
        patience=getattr(base_args, "patience", 20),
        n_hidden_layers=getattr(base_args, "n_hidden_layers", 1),
        hidden_dim_3layer=getattr(base_args, "hidden_dim_3layer", 32),
    )


def main():
    parser = argparse.ArgumentParser(description="Astrocytes combined (all regions) overfitting experiments, 8-fold CV")
    parser.add_argument("--regions", type=str, nargs="*", default=None,
                        help="Regions to combine (default: EC ITG PFC V2 V1 from config)")
    parser.add_argument("--n_folds", type=int, default=8, help="Number of CV folds (default: 8)")
    parser.add_argument("--top_k_genes", type=int, default=DEFAULT_TOP_K_GENES, help="Top-k genes per region for expression (match create_raw_expression_data)")
    parser.add_argument("--dge_results_dir", type=str, default=DEFAULT_DGE_RESULTS_DIR, help="DGE results dir for expression gene lists")
    parser.add_argument("--embedding", action="store_true", help="Run embedding experiments")
    parser.add_argument("--expression", action="store_true", help="Run expression experiments")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    parser.add_argument("--attention_heads", type=int, default=1)
    parser.add_argument("--log_base", type=str, default="combined_Astrocytes", help="Base dir for logs and CSVs")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--n_hidden_layers", type=int, default=1, choices=[1, 3])
    parser.add_argument("--hidden_dim_3layer", type=int, default=32)
    args = parser.parse_args()

    if not args.embedding and not args.expression:
        args.embedding = True
        args.expression = True
    os.makedirs(args.log_base, exist_ok=True)
    regions = args.regions if (args.regions is not None and len(args.regions) > 0) else REGIONS
    print(f"Astrocytes combined regions: {regions}")

    if args.embedding:
        print("\n" + "=" * 70)
        print("Combined EMBEDDING experiments (8-fold)")
        print("=" * 70)
        try:
            all_fold_data, thal_to_idx, idx_to_thal, n_thal_classes = load_combined_embedding_data(
                regions, n_folds=args.n_folds
            )
        except FileNotFoundError as e:
            print(f"Embedding load failed: {e}")
            all_fold_data = None
        if all_fold_data is not None:
            n_runs = len(EXPERIMENTS) * (
                len(ATTENTION_OPTS) * len(LOSS_PTAU_OPTS) * len(PCA_OPTS)
                + len(LOSS_PTAU_OPTS) * len(PCA_OPTS)
            )
            results = []
            run_idx = 0
            for gene_encoder in GENE_ENCODER_OPTS:
                attention_opts = ATTENTION_OPTS if not gene_encoder else [False]
                for attention in attention_opts:
                    for loss_ptau in LOSS_PTAU_OPTS:
                        for pca_components in PCA_OPTS:
                            for exp in EXPERIMENTS:
                                run_idx += 1
                                run_name = f"ge{gene_encoder}_attn{attention}_{loss_ptau}_pca{pca_components}_{exp['name']}"
                                print(f"\nRun {run_idx}/{n_runs}: {run_name}")
                                train_args = build_embedding_args(exp, gene_encoder, attention, loss_ptau, pca_components, args)
                                try:
                                    avg_f1, avg_mse, avg_mae, _ = run_embedding_cv_combined(
                                        all_fold_data, thal_to_idx, idx_to_thal, n_thal_classes, train_args
                                    )
                                except Exception as e:
                                    print(f"  ERROR: {e}")
                                    results.append({"region": "combined", "cell_type": "Astrocytes", "run": run_name,
                                        "experiment": exp["name"], "gene_encoder": gene_encoder, "attention": attention,
                                        "loss_ptau": loss_ptau, "pca_components": pca_components,
                                        "dropout": exp["dropout"], "weight_decay": exp["weight_decay"], "lr": exp["lr"],
                                        "lambda_entropy": exp["lambda_entropy"], "avg_f1": None, "avg_mse": None, "avg_mae": None, "error": str(e)})
                                    continue
                                results.append({
                                    "region": "combined", "cell_type": "Astrocytes", "run": run_name,
                                    "experiment": exp["name"], "gene_encoder": gene_encoder, "attention": attention,
                                    "loss_ptau": loss_ptau, "pca_components": pca_components,
                                    "dropout": exp["dropout"], "weight_decay": exp["weight_decay"], "lr": exp["lr"],
                                    "lambda_entropy": exp["lambda_entropy"],
                                    "avg_f1": avg_f1, "avg_mse": avg_mse, "avg_mae": avg_mae, "error": "",
                                })
            if results:
                out_csv = os.path.join(args.log_base, "overfitting_experiments_combined_embedding.csv")
                fieldnames = ["region", "cell_type", "run", "experiment", "gene_encoder", "attention", "loss_ptau", "pca_components",
                              "avg_f1", "avg_mse", "avg_mae", "dropout", "weight_decay", "lr", "lambda_entropy", "error"]
                with open(out_csv, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=fieldnames)
                    w.writeheader()
                    w.writerows(results)
                print(f"Wrote {len(results)} rows to {out_csv}")

    if args.expression:
        print("\n" + "=" * 70)
        print("Combined EXPRESSION experiments (8-fold)")
        print("=" * 70)
        try:
            all_fold_data, thal_to_idx, idx_to_thal, n_thal_classes = load_combined_expression_data(
                regions, n_folds=args.n_folds,
                top_k_genes=args.top_k_genes, dge_results_dir=args.dge_results_dir
            )
        except FileNotFoundError as e:
            print(f"Expression load failed: {e}")
            all_fold_data = None
        if all_fold_data is not None:
            n_runs = len(EXPERIMENTS) * len(ATTENTION_OPTS_EXPR) * len(LOSS_PTAU_OPTS_EXPR) * len(PCA_OPTS_EXPR)
            results = []
            run_idx = 0
            for exp in EXPERIMENTS:
                for attention in ATTENTION_OPTS_EXPR:
                    for loss_ptau in LOSS_PTAU_OPTS_EXPR:
                        for pca_components in PCA_OPTS_EXPR:
                            run_idx += 1
                            run_name = f"attn{attention}_{loss_ptau}_pca{pca_components}_{exp['name']}"
                            print(f"\nRun {run_idx}/{n_runs}: {run_name}")
                            train_args = build_expression_args(exp, attention, loss_ptau, pca_components, args)
                            try:
                                avg_f1, avg_mse, avg_mae, _ = run_expression_cv_combined(
                                    all_fold_data, thal_to_idx, idx_to_thal, n_thal_classes, train_args
                                )
                            except Exception as e:
                                print(f"  ERROR: {e}")
                                results.append({"region": "combined", "cell_type": "Astrocytes", "run": run_name,
                                    "experiment": exp["name"], "attention": attention, "loss_ptau": loss_ptau,
                                    "pca_components": pca_components, "dropout": exp["dropout"],
                                    "weight_decay": exp["weight_decay"], "lr": exp["lr"],
                                    "lambda_entropy": exp["lambda_entropy"], "avg_f1": None, "avg_mse": None, "avg_mae": None, "error": str(e)})
                                continue
                            results.append({
                                "region": "combined", "cell_type": "Astrocytes", "run": run_name,
                                "experiment": exp["name"], "attention": attention, "loss_ptau": loss_ptau,
                                "pca_components": pca_components, "dropout": exp["dropout"],
                                "weight_decay": exp["weight_decay"], "lr": exp["lr"],
                                "lambda_entropy": exp["lambda_entropy"],
                                "avg_f1": avg_f1, "avg_mse": avg_mse, "avg_mae": avg_mae, "error": "",
                            })
            if results:
                out_csv = os.path.join(args.log_base, "overfitting_experiments_combined_expression.csv")
                fieldnames = ["region", "cell_type", "run", "experiment", "attention", "loss_ptau", "pca_components",
                              "avg_f1", "avg_mse", "avg_mae", "dropout", "weight_decay", "lr", "lambda_entropy", "error"]
                with open(out_csv, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=fieldnames)
                    w.writeheader()
                    w.writerows(results)
                print(f"Wrote {len(results)} rows to {out_csv}")


if __name__ == "__main__":
    main()
