"""Generate ranked list of genes by permutation importance (F1 drop).

8-fold training: for each left-out fold, train on 7 folds, then permute one gene at a time
on the test fold and measure F1 drop. Aggregate across folds to produce one total gene ranking.

Supports both expression and embedding models.
"""

import argparse
import csv
import os
import pickle
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from baseline_model import BaselineRankingModel
from baseline_model_Microglia import BaselineRankingModelMicroglia, CERAD_CLASS_ORDER, BRAAK_TO_IDX
from train_model import EmbeddingDataset, train_epoch
from train_model_Microglia import EmbeddingDatasetMicroglia, train_epoch_microglia
from gene_encoder_model import GeneEncoderModel, GeneEncoderModelMicroglia, gene_entropy_loss

# Experiment configs (same as run_overfitting_experiments)
EXPERIMENTS = {
    "baseline": {"dropout": 0.2, "weight_decay": 0.1, "lr": 0.001, "lambda_entropy": 1e-3},
    "higher_dropout": {"dropout": 0.4, "weight_decay": 0.1, "lr": 0.001, "lambda_entropy": 1e-3},
    "highest_dropout": {"dropout": 0.5, "weight_decay": 0.1, "lr": 0.001, "lambda_entropy": 1e-3},
    "higher_weight_decay": {"dropout": 0.2, "weight_decay": 0.5, "lr": 0.001, "lambda_entropy": 1e-3},
    "highest_weight_decay": {"dropout": 0.2, "weight_decay": 1.0, "lr": 0.001, "lambda_entropy": 1e-3},
    "lower_lr": {"dropout": 0.2, "weight_decay": 0.1, "lr": 5e-4, "lambda_entropy": 1e-3},
    "lowest_lr": {"dropout": 0.2, "weight_decay": 0.1, "lr": 1e-4, "lambda_entropy": 1e-3},
    "higher_lambda_entropy": {"dropout": 0.2, "weight_decay": 0.1, "lr": 0.001, "lambda_entropy": 1e-2},
    "highest_lambda_entropy": {"dropout": 0.2, "weight_decay": 0.1, "lr": 0.01, "lambda_entropy": 1e-1},
    "dropout_wd": {"dropout": 0.4, "weight_decay": 0.5, "lr": 0.001, "lambda_entropy": 1e-3},
    "dropout_lr": {"dropout": 0.4, "weight_decay": 0.1, "lr": 5e-4, "lambda_entropy": 1e-3},
    "wd_lr": {"dropout": 0.2, "weight_decay": 0.5, "lr": 5e-4, "lambda_entropy": 1e-3},
    "all_moderate": {"dropout": 0.4, "weight_decay": 0.5, "lr": 5e-4, "lambda_entropy": 1e-2},
    "all_strong": {"dropout": 0.5, "weight_decay": 1.0, "lr": 1e-4, "lambda_entropy": 1e-1},
}


def load_gene_names(region, cell_type, top_k_genes=3000):
    """Load gene names in same order as expression matrix columns (from create_raw_expression_data)."""
    try:
        from create_raw_expression_data import load_top_genes_from_dge_union, create_anndata
        adata = create_anndata(region, cell_type)
        dge_genes = load_top_genes_from_dge_union(region=region, cell_type=cell_type, top_k=top_k_genes)
        gene_names = [g for g in dge_genes if g in adata.var_names]
        return gene_names
    except Exception as e:
        print(f"Warning: Could not load gene names ({e}). Using gene_0, gene_1, ...")
        return None


def load_gene_names_embedding(region, cell_type):
    """Load gene names in same order as embedding matrix (from create_data)."""
    try:
        from create_data import create_anndata, load_gene_embeddings
        adata = create_anndata(region, cell_type)
        emb_dir = os.path.join("GREmLN", "embeddings")
        if region == "MTG" or cell_type == "MTG":
            emb_path = os.path.join(emb_dir, "MTG_gene_embeddings.tsv")
        else:
            emb_path = os.path.join(emb_dir, f"{region}_Microglia_gene_embeddings.tsv" if cell_type == "Microglia" else f"{region}_gene_embeddings.tsv")
        emb_dict = load_gene_embeddings(emb_path, gene_list=list(adata.var_names))
        gene_names = [g for g in adata.var_names if g in emb_dict]
        return gene_names
    except Exception as e:
        print(f"Warning: Could not load embedding gene names ({e}). Using gene_0, gene_1, ...")
        return None


def _permute_gene_embedding(X, gene_idx, rng):
    """Permute gene_idx slice across patients. X: (n_patients, n_genes, n_dims). In-place."""
    slice_g = X[:, gene_idx, :].copy()  # (n_patients, n_dims)
    perm = rng.permutation(len(slice_g))
    X[:, gene_idx, :] = slice_g[perm]
    return X


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate ranked list of genes by permutation importance (F1 drop)"
    )
    parser.add_argument("--region", type=str, required=True, help="Region (e.g., EC, ITG, PFC, V1, MTG). MTG uses Microglia two-head model and MTG data.")
    parser.add_argument("--cell_type", type=str, default="Astrocytes", choices=["Astrocytes", "Microglia"],
                        help="Cell type (ignored when region=MTG; MTG always uses Microglia model)")
    parser.add_argument("--experiment", type=str, default="baseline", choices=list(EXPERIMENTS.keys()))
    parser.add_argument("--model", type=str, default="expression", choices=["embedding", "expression"])
    parser.add_argument("--gene_encoder", action="store_true", help="Use gene encoder (embedding only)")
    parser.add_argument("--pooling", type=str, default="mean", choices=["mean", "max"],
                        help="Pooling for embedding when not gene_encoder (default: mean)")
    parser.add_argument("--attention", action="store_true")
    parser.add_argument("--loss_ptau", type=str, default="huber", choices=["mse", "mae", "huber", "none"],
                        help="Loss for ptau (none to omit ptau loss)")
    parser.add_argument("--pca_components", type=int, default=None)
    parser.add_argument("--n_folds", type=int, default=8, help="Number of CV folds")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--top_k_genes", type=int, default=3000,
                        help="Must match value used in create_raw_expression_data")
    parser.add_argument("--n_permute", type=int, default=1,
                        help="Number of permutation repeats per gene (for stability); mean F1 drop is used")
    parser.add_argument("--log_base", type=str, default="run_new",
                        help="Base directory for TensorBoard logs (default: run_new)")
    parser.add_argument("--n_hidden_layers", type=int, default=1, choices=[1, 3],
                        help="Baseline MLP: 1 (minimal) or 3 (original) hidden layers (default: 1).")
    parser.add_argument("--hidden_dim_3layer", type=int, default=32,
                        help="Hidden size when n_hidden_layers=3 (default: 32).")
    return parser.parse_args()


def _compute_f1_astrocytes(model, test_loader, device, idx_to_thal):
    """Compute F1 on test set (Thal classification) for Astrocytes."""
    model.eval()
    thal_preds, thal_targets = [], []
    with torch.no_grad():
        for emb, _, thal in test_loader:
            emb = emb.to(device)
            out = model(emb)
            thal_pred = out[1].argmax(1)
            thal_preds.extend(thal_pred.cpu().numpy())
            thal_targets.extend(thal.cpu().numpy())
    pred_labels = [idx_to_thal[i] for i in thal_preds]
    true_labels = [idx_to_thal[i] for i in thal_targets]
    return f1_score(true_labels, pred_labels, average="weighted")


def _compute_f1_microglia(model, test_loader, device, thal_only=False):
    """Compute F1 on test set. If thal_only=True (MTG), return only CERAD/Thal F1."""
    model.eval()
    cerad_preds, cerad_targets = [], []
    braak_preds, braak_targets = [], []
    with torch.no_grad():
        for emb, cerad, braak in test_loader:
            emb = emb.to(device)
            out = model(emb)
            cerad_out, braak_out = out[0], out[1]
            cerad_preds.extend(cerad_out.argmax(1).cpu().numpy())
            braak_preds.extend(braak_out.argmax(1).cpu().numpy())
            cerad_targets.extend(cerad.cpu().numpy())
            braak_targets.extend(braak.cpu().numpy())
    f1_cerad = f1_score(cerad_targets, cerad_preds, average="weighted")
    if thal_only:
        return f1_cerad
    f1_braak = f1_score(braak_targets, braak_preds, average="weighted")
    return (f1_cerad + f1_braak) / 2.0


def _permute_gene_column(X, gene_idx, rng):
    """Permute values in column gene_idx across rows (patients). In-place for the copy we pass."""
    col = X[:, gene_idx].copy()
    rng.shuffle(col)
    X[:, gene_idx] = col
    return X


# --- Cell-type config: data paths, target encoding, model/dataset creation ---

def _get_expression_data_path(region, cell_type):
    if region == "MTG":
        return os.path.join("data", "model_data", "MTG_raw_expression_data.pkl")
    base = f"{region}_Microglia_raw_expression_data.pkl" if cell_type == "Microglia" else f"{region}_raw_expression_data.pkl"
    return os.path.join("data", "model_data", base)


def _get_embedding_data_path(region, cell_type):
    if region == "MTG":
        return os.path.join("data", "model_data", "MTG_data_for_model_training.pkl")
    base = f"{region}_Microglia_data_for_model_training.pkl" if cell_type == "Microglia" else f"{region}_data_for_model_training.pkl"
    return os.path.join("data", "model_data", base)


def _build_expression_fold_data(expression_folds, ptau_folds, thal_folds, fold_idx, cell_type, thal_to_idx=None,
                                braak_folds=None, cerad_to_idx=None, braak_to_idx=None, thal_only=False):
    """Build fold dict with X, and cell-type-specific targets."""
    X = np.array(expression_folds[fold_idx], dtype=np.float32)
    if cell_type == "Astrocytes":
        ptau = np.log1p(np.array(ptau_folds[fold_idx], dtype=np.float32))
        thal = np.array([thal_to_idx[int(t)] for t in thal_folds[fold_idx]], dtype=np.int64)
        return {"X": X, "ptau": ptau, "thal": thal}
    elif cerad_to_idx is not None and braak_to_idx is not None and braak_folds is not None:
        cerad = np.array([cerad_to_idx.get(str(t).strip().lower(), 0) for t in thal_folds[fold_idx]], dtype=np.int64)
        braak = np.array([braak_to_idx.get(str(b).strip(), 0) for b in braak_folds[fold_idx]], dtype=np.int64)
        return {"X": X, "cerad": cerad, "braak": braak}
    elif cerad_to_idx is not None and thal_only:
        cerad = np.array([cerad_to_idx.get(str(t).strip().lower(), 0) for t in thal_folds[fold_idx]], dtype=np.int64)
        n = len(cerad)
        return {"X": X, "cerad": cerad, "braak": np.zeros(n, dtype=np.int64)}
    else:
        cerad_to_idx = {s: i for i, s in enumerate(CERAD_CLASS_ORDER)}
        cerad = np.array([cerad_to_idx.get(str(t).strip().lower(), 0) for t in thal_folds[fold_idx]])
        braak = np.array([BRAAK_TO_IDX.get(int(round(p)), 0) for p in ptau_folds[fold_idx]])
        return {"X": X, "cerad": cerad, "braak": braak}


def _build_embedding_fold_data(embeddings_folds, patient_info_folds, ptau_folds, thal_folds, fold_idx, cell_type, thal_to_idx=None,
                               braak_folds=None, cerad_to_idx=None, braak_to_idx=None, thal_only=False):
    """Build fold dict with X (n_patients, n_genes, n_dims) and targets."""
    fold_emb = embeddings_folds[fold_idx]
    fold_patient_info = patient_info_folds[fold_idx]
    patient_embs = []
    patient_ptau_raw = []
    patient_thal_raw = []
    patient_braak_raw = [] if braak_folds is not None else None
    cell_idx = 0
    for patient_idx, (_, cell_names) in enumerate(fold_patient_info.items()):
        n_cells = len(cell_names)
        patient_embs.append(np.mean(fold_emb[cell_idx : cell_idx + n_cells], axis=0))
        if ptau_folds is not None:
            patient_ptau_raw.append(ptau_folds[fold_idx][patient_idx])
        patient_thal_raw.append(thal_folds[fold_idx][patient_idx])
        if braak_folds is not None:
            patient_braak_raw.append(braak_folds[fold_idx][patient_idx])
        cell_idx += n_cells
    X = np.array(patient_embs, dtype=np.float32)
    if cell_type == "Astrocytes":
        ptau = np.log1p(np.array(patient_ptau_raw, dtype=np.float32))
        thal = np.array([thal_to_idx[int(t)] for t in patient_thal_raw], dtype=np.int64)
        return {"X": X, "ptau": ptau, "thal": thal}
    elif cerad_to_idx is not None and braak_to_idx is not None and patient_braak_raw is not None:
        cerad = np.array([cerad_to_idx.get(str(t).strip().lower(), 0) for t in patient_thal_raw], dtype=np.int64)
        braak = np.array([braak_to_idx.get(str(b).strip(), 0) for b in patient_braak_raw], dtype=np.int64)
        return {"X": X, "cerad": cerad, "braak": braak}
    elif cerad_to_idx is not None and thal_only:
        cerad = np.array([cerad_to_idx.get(str(t).strip().lower(), 0) for t in patient_thal_raw], dtype=np.int64)
        n = len(cerad)
        return {"X": X, "cerad": cerad, "braak": np.zeros(n, dtype=np.int64)}
    else:
        cerad_to_idx = {s: i for i, s in enumerate(CERAD_CLASS_ORDER)}
        cerad = np.array([cerad_to_idx.get(str(t).strip().lower(), 0) for t in patient_thal_raw], dtype=np.int64)
        braak = np.array([BRAAK_TO_IDX.get(int(round(p)), 0) for p in patient_ptau_raw], dtype=np.int64)
        return {"X": X, "cerad": cerad, "braak": braak}


def run_permutation_importance_expression(args):
    """8-fold CV, permute one gene at a time, measure F1 drop. Expression model (Astrocytes, Microglia, or MTG)."""
    data_path = _get_expression_data_path(args.region, args.cell_type)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Expression data not found: {data_path}")

    with open(data_path, "rb") as f:
        raw = pickle.load(f)
    if args.region == "MTG":
        expression_folds, patient_info_folds, thal_folds, braak_folds = raw
        ptau_folds = None
    else:
        expression_folds, patient_info_folds, ptau_folds, thal_folds = raw
        braak_folds = None

    n_folds = len(expression_folds)
    if n_folds != args.n_folds:
        print(f"Warning: data has {n_folds} folds. Using {n_folds}.")

    gene_load_cell_type = "MTG" if args.region == "MTG" else args.cell_type
    gene_names = load_gene_names(args.region, gene_load_cell_type, args.top_k_genes)
    n_genes = expression_folds[0].shape[1]
    if gene_names is None or len(gene_names) != n_genes:
        gene_names = [f"gene_{i}" for i in range(n_genes)]

    thal_to_idx = None
    idx_to_thal = None
    cerad_to_idx = None
    braak_to_idx = None
    n_cerad_classes = len(CERAD_CLASS_ORDER)
    n_braak_classes = len(BRAAK_TO_IDX)
    if args.cell_type == "Astrocytes" and args.region != "MTG":
        all_thal = [t for fold in thal_folds for t in fold]
        thal_classes = sorted(set(int(x) for x in all_thal))
        thal_to_idx = {c: i for i, c in enumerate(thal_classes)}
        idx_to_thal = {i: c for c, i in thal_to_idx.items()}
    elif args.region == "MTG":
        # MTG: Thal (CERAD) only; no Braak head
        all_thal = [str(t).strip().lower() for fold in thal_folds for t in fold]
        unique_cerad = sorted(set(all_thal))
        cerad_to_idx = {s: i for i, s in enumerate(unique_cerad)}
        n_cerad_classes = len(unique_cerad)
        n_braak_classes = 1
        braak_to_idx = None  # unused for MTG Thal-only

    all_fold_data = [
        _build_expression_fold_data(expression_folds, ptau_folds, thal_folds, i,
                                    "Microglia" if args.region == "MTG" else args.cell_type,
                                    thal_to_idx, braak_folds=braak_folds, cerad_to_idx=cerad_to_idx, braak_to_idx=braak_to_idx,
                                    thal_only=(args.region == "MTG"))
        for i in range(n_folds)
    ]

    exp_cfg = EXPERIMENTS[args.experiment]
    device = torch.device(args.device)
    rng = np.random.default_rng(42)
    gene_f1_drops = defaultdict(list)
    baseline_f1_list = []

    for test_fold_idx in range(n_folds):
        print(f"\n{'='*60}\nTest fold {test_fold_idx + 1}/{n_folds} (expression {args.region if args.region == 'MTG' else args.cell_type})\n{'='*60}")

        train_idx = [i for i in range(n_folds) if i != test_fold_idx]
        train_X = np.concatenate([all_fold_data[i]["X"] for i in train_idx], axis=0)
        test_ptau = all_fold_data[test_fold_idx].get("ptau")
        test_thal = all_fold_data[test_fold_idx].get("thal")
        test_cerad = all_fold_data[test_fold_idx].get("cerad")
        test_braak = all_fold_data[test_fold_idx].get("braak")

        if args.cell_type == "Astrocytes" and args.region != "MTG":
            train_ptau = np.concatenate([all_fold_data[i]["ptau"] for i in train_idx], axis=0)
            train_thal = np.concatenate([all_fold_data[i]["thal"] for i in train_idx], axis=0)
        else:
            train_cerad = np.concatenate([all_fold_data[i]["cerad"] for i in train_idx], axis=0)
            train_braak = np.concatenate([all_fold_data[i]["braak"] for i in train_idx], axis=0)

        scaler = StandardScaler()
        train_X = scaler.fit_transform(train_X)
        test_X_orig = scaler.transform(all_fold_data[test_fold_idx]["X"].copy())
        if args.pca_components is not None:
            pca = PCA(n_components=args.pca_components)
            train_X = pca.fit_transform(train_X)
            test_X_orig = pca.transform(scaler.transform(all_fold_data[test_fold_idx]["X"].copy()))
        else:
            pca = None

        if args.cell_type == "Astrocytes" and args.region != "MTG":
            train_ds = EmbeddingDataset(train_X, train_ptau, train_thal)
            test_ds = EmbeddingDataset(test_X_orig, test_ptau, test_thal)
            model = BaselineRankingModel(
                input_dim=train_X.shape[1], input_shape=None,
                n_thal_classes=len(idx_to_thal), use_attention=args.attention, num_heads=1,
                dropout=exp_cfg["dropout"],
                n_hidden_layers=args.n_hidden_layers,
                hidden_dim_3layer=args.hidden_dim_3layer,
            ).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=exp_cfg["lr"], weight_decay=exp_cfg["weight_decay"])
            criterion_ptau = None if args.loss_ptau == "none" else {"mse": nn.MSELoss(), "mae": nn.L1Loss(), "huber": nn.HuberLoss()}[args.loss_ptau]
            criterion_thal = nn.CrossEntropyLoss()
            train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
            test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
            compute_f1 = lambda m, loader: _compute_f1_astrocytes(m, loader, device, idx_to_thal)
            log_dir = os.path.join(args.log_base, f"{args.region}_{gene_load_cell_type}_expression_{args.experiment}_testfold{test_fold_idx+1}")
            writer = SummaryWriter(log_dir)
            best_f1, best_state = -1.0, None
            for epoch in range(args.epochs):
                train_loss, train_ptau_loss, train_thal_loss = train_epoch(model, train_loader, criterion_ptau, criterion_thal, optimizer, device)
                writer.add_scalar("Loss/Train", train_loss, epoch)
                writer.add_scalar("Loss/Train_Ptau", train_ptau_loss, epoch)
                writer.add_scalar("Loss/Train_Thal", train_thal_loss, epoch)
                f1 = compute_f1(model, test_loader)
                writer.add_scalar("Metrics/Test_F1", f1, epoch)
                if f1 > best_f1:
                    best_f1 = f1
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if best_state is not None:
                model.load_state_dict(best_state)
            baseline_f1 = best_f1
        else:
            train_ds = EmbeddingDatasetMicroglia(train_X, train_cerad, train_braak)
            test_ds = EmbeddingDatasetMicroglia(test_X_orig, test_cerad, test_braak)
            model = BaselineRankingModelMicroglia(
                input_dim=train_X.shape[1], input_shape=None,
                n_cerad_classes=n_cerad_classes, n_braak_classes=n_braak_classes,
                use_attention=False, num_heads=1, dropout=exp_cfg["dropout"],
            ).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=exp_cfg["lr"], weight_decay=exp_cfg["weight_decay"])
            criterion_cerad = nn.CrossEntropyLoss()
            criterion_braak = nn.CrossEntropyLoss()
            train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
            test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
            compute_f1 = lambda m, loader: _compute_f1_microglia(m, loader, device, thal_only=(args.region == "MTG"))
            log_dir = os.path.join(args.log_base, f"{args.region}_{gene_load_cell_type}_expression_{args.experiment}_testfold{test_fold_idx+1}")
            writer = SummaryWriter(log_dir)
            best_f1, best_state = -1.0, None
            for epoch in range(args.epochs):
                train_loss = train_epoch_microglia(model, train_loader, criterion_cerad, criterion_braak, optimizer, device)
                writer.add_scalar("Loss/Train", train_loss, epoch)
                f1 = compute_f1(model, test_loader)
                writer.add_scalar("Metrics/Test_F1", f1, epoch)
                if f1 > best_f1:
                    best_f1 = f1
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if best_state is not None:
                model.load_state_dict(best_state)
            baseline_f1 = best_f1

        baseline_f1_list.append(baseline_f1)
        writer.add_scalar("Metrics/Best_F1", baseline_f1, 0)
        writer.close()
        print(f"  Best F1: {baseline_f1:.4f}")

        for g in range(n_genes):
            drops = []
            for _ in range(args.n_permute):
                test_X_perm = all_fold_data[test_fold_idx]["X"].copy()
                _permute_gene_column(test_X_perm, g, rng)
                test_X_perm = scaler.transform(test_X_perm)
                if pca is not None:
                    test_X_perm = pca.transform(test_X_perm)
                if args.cell_type == "Astrocytes" and args.region != "MTG":
                    test_ds_perm = EmbeddingDataset(test_X_perm, test_ptau, test_thal)
                else:
                    test_ds_perm = EmbeddingDatasetMicroglia(test_X_perm, test_cerad, test_braak)
                test_loader_perm = DataLoader(test_ds_perm, batch_size=args.batch_size, shuffle=False)
                perm_f1 = compute_f1(model, test_loader_perm)
                drops.append(baseline_f1 - perm_f1)
            gene_f1_drops[g].append(np.mean(drops))
            if (g + 1) % 200 == 0:
                print(f"  Gene {g + 1}/{n_genes} done")

    print(f"\nAverage best F1 across {n_folds} folds: {np.mean(baseline_f1_list):.4f}")
    return _aggregate_and_rank(gene_f1_drops, gene_names, n_folds)


def run_permutation_importance_embedding(args):
    """8-fold CV, permute one gene at a time on embedding, measure F1 drop. Embedding model (Astrocytes, Microglia, or MTG)."""
    data_path = _get_embedding_data_path(args.region, args.cell_type)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Embedding data not found: {data_path}")

    with open(data_path, "rb") as f:
        raw = pickle.load(f)
    if args.region == "MTG":
        embeddings_folds, patient_info_folds, thal_folds, braak_folds = raw
        ptau_folds = None
    else:
        embeddings_folds, patient_info_folds, ptau_folds, thal_folds = raw
        braak_folds = None

    n_folds = len(embeddings_folds)
    if n_folds != args.n_folds:
        print(f"Warning: data has {n_folds} folds. Using {n_folds}.")

    gene_load_cell_type = "MTG" if args.region == "MTG" else args.cell_type
    gene_names = load_gene_names_embedding(args.region, gene_load_cell_type)
    device = torch.device(args.device)
    rng = np.random.default_rng(42)
    exp_cfg = EXPERIMENTS[args.experiment]
    pooling = None if args.gene_encoder else args.pooling

    thal_to_idx = None
    idx_to_thal = None
    cerad_to_idx = None
    braak_to_idx = None
    n_cerad_classes = len(CERAD_CLASS_ORDER)
    n_braak_classes = len(BRAAK_TO_IDX)
    if args.cell_type == "Astrocytes" and args.region != "MTG":
        all_thal = [t for fold in thal_folds for t in fold]
        thal_classes = sorted(set(int(x) for x in all_thal))
        thal_to_idx = {c: i for i, c in enumerate(thal_classes)}
        idx_to_thal = {i: c for c, i in thal_to_idx.items()}
    elif args.region == "MTG":
        # MTG: Thal (CERAD) only; no Braak head
        all_thal = [str(t).strip().lower() for fold in thal_folds for t in fold]
        unique_cerad = sorted(set(all_thal))
        cerad_to_idx = {s: i for i, s in enumerate(unique_cerad)}
        n_cerad_classes = len(unique_cerad)
        n_braak_classes = 1
        braak_to_idx = None  # unused for MTG Thal-only

    all_fold_data = [
        _build_embedding_fold_data(embeddings_folds, patient_info_folds, ptau_folds, thal_folds, i,
                                  "Microglia" if args.region == "MTG" else args.cell_type,
                                  thal_to_idx, braak_folds=braak_folds, cerad_to_idx=cerad_to_idx, braak_to_idx=braak_to_idx,
                                  thal_only=(args.region == "MTG"))
        for i in range(n_folds)
    ]
    n_genes = all_fold_data[0]["X"].shape[1]
    if gene_names is None or len(gene_names) != n_genes:
        gene_names = [f"gene_{i}" for i in range(n_genes)]

    gene_f1_drops = defaultdict(list)
    baseline_f1_list = []

    for test_fold_idx in range(n_folds):
        print(f"\n{'='*60}\nTest fold {test_fold_idx + 1}/{n_folds} (embedding {args.region if args.region == 'MTG' else args.cell_type})\n{'='*60}")

        train_idx = [i for i in range(n_folds) if i != test_fold_idx]
        train_X = np.concatenate([all_fold_data[i]["X"] for i in train_idx], axis=0)
        n_train, n_test = train_X.shape[0], all_fold_data[test_fold_idx]["X"].shape[0]
        n_genes, n_dims = train_X.shape[1], train_X.shape[2]

        test_ptau = all_fold_data[test_fold_idx].get("ptau")
        test_thal = all_fold_data[test_fold_idx].get("thal")
        test_cerad = all_fold_data[test_fold_idx].get("cerad")
        test_braak = all_fold_data[test_fold_idx].get("braak")

        if args.cell_type == "Astrocytes" and args.region != "MTG":
            train_ptau = np.concatenate([all_fold_data[i]["ptau"] for i in train_idx], axis=0)
            train_thal = np.concatenate([all_fold_data[i]["thal"] for i in train_idx], axis=0)
        else:
            train_cerad = np.concatenate([all_fold_data[i]["cerad"] for i in train_idx], axis=0)
            train_braak = np.concatenate([all_fold_data[i]["braak"] for i in train_idx], axis=0)

        train_2d = train_X.reshape(n_train, -1)
        test_2d = all_fold_data[test_fold_idx]["X"].reshape(n_test, -1)
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_2d).reshape(n_train, n_genes, n_dims)
        test_scaled = scaler.transform(test_2d).reshape(n_test, n_genes, n_dims)

        if args.pca_components is not None:
            pca = PCA(n_components=args.pca_components)
            pca.fit(train_scaled.reshape(-1, n_dims))
            train_scaled = pca.transform(train_scaled.reshape(-1, n_dims)).reshape(n_train, n_genes, args.pca_components)
            test_scaled = pca.transform(test_scaled.reshape(-1, n_dims)).reshape(n_test, n_genes, args.pca_components)
            n_dims = args.pca_components
        else:
            pca = None

        if pooling:
            train_input = np.mean(train_scaled, axis=1)
            test_input = np.mean(test_scaled, axis=1)
            input_shape = None
            input_dim = train_input.shape[1]
        else:
            train_input = train_scaled
            test_input = test_scaled
            input_shape = (n_genes, n_dims)
            input_dim = None

        if args.cell_type == "Astrocytes" and args.region != "MTG":
            train_ds = EmbeddingDataset(train_input, train_ptau, train_thal)
            test_ds = EmbeddingDataset(test_input, test_ptau, test_thal)
            criterion_ptau = None if args.loss_ptau == "none" else {"mse": nn.MSELoss(), "mae": nn.L1Loss(), "huber": nn.HuberLoss()}[args.loss_ptau]
            criterion_thal = nn.CrossEntropyLoss()
            if args.gene_encoder:
                model = GeneEncoderModel(n_dims=n_dims, n_thal_classes=len(idx_to_thal), dropout=exp_cfg["dropout"], n_genes=n_genes).to(device)
                lambda_entropy = exp_cfg["lambda_entropy"]
            else:
                model = BaselineRankingModel(
                    input_dim=input_dim, input_shape=input_shape, n_thal_classes=len(idx_to_thal),
                    use_attention=args.attention, num_heads=1, dropout=exp_cfg["dropout"],
                    n_hidden_layers=args.n_hidden_layers, hidden_dim_3layer=args.hidden_dim_3layer,
                ).to(device)
                lambda_entropy = 0.0
            optimizer = torch.optim.Adam(model.parameters(), lr=exp_cfg["lr"], weight_decay=exp_cfg["weight_decay"])
            train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
            ge_str = "_ge" if args.gene_encoder else ""
            attn_str = "_attn" if args.attention else ""
            pca_str = f"_pca{args.pca_components}" if args.pca_components else ""
            test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
            compute_f1 = lambda m, loader: _compute_f1_astrocytes(m, loader, device, idx_to_thal)
            log_dir = os.path.join(args.log_base, f"{args.region}_{gene_load_cell_type}_embedding{ge_str}{attn_str}{pca_str}_{args.experiment}_testfold{test_fold_idx+1}")
            writer = SummaryWriter(log_dir)
            best_f1, best_state = -1.0, None
            for epoch in range(args.epochs):
                train_loss, train_ptau_loss, train_thal_loss = train_epoch(model, train_loader, criterion_ptau, criterion_thal, optimizer, device, lambda_entropy=lambda_entropy)
                writer.add_scalar("Loss/Train", train_loss, epoch)
                writer.add_scalar("Loss/Train_Ptau", train_ptau_loss, epoch)
                writer.add_scalar("Loss/Train_Thal", train_thal_loss, epoch)
                f1 = compute_f1(model, test_loader)
                writer.add_scalar("Metrics/Test_F1", f1, epoch)
                if f1 > best_f1:
                    best_f1 = f1
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if best_state is not None:
                model.load_state_dict(best_state)
            baseline_f1 = best_f1
        else:
            train_ds = EmbeddingDatasetMicroglia(train_input, train_cerad, train_braak)
            test_ds = EmbeddingDatasetMicroglia(test_input, test_cerad, test_braak)
            if args.gene_encoder:
                model = GeneEncoderModelMicroglia(n_dims=n_dims, n_cerad_classes=n_cerad_classes,
                    n_braak_classes=n_braak_classes, dropout=exp_cfg["dropout"]).to(device)
                lambda_entropy = exp_cfg["lambda_entropy"]
            else:
                model = BaselineRankingModelMicroglia(input_dim=input_dim, input_shape=input_shape,
                    n_cerad_classes=n_cerad_classes, n_braak_classes=n_braak_classes,
                    use_attention=args.attention, num_heads=1, dropout=exp_cfg["dropout"]).to(device)
                lambda_entropy = 0.0
            criterion_cerad = nn.CrossEntropyLoss()
            criterion_braak = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=exp_cfg["lr"], weight_decay=exp_cfg["weight_decay"])
            train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
            ge_str = "_ge" if args.gene_encoder else ""
            attn_str = "_attn" if args.attention else ""
            pca_str = f"_pca{args.pca_components}" if args.pca_components else ""
            test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
            compute_f1 = lambda m, loader: _compute_f1_microglia(m, loader, device, thal_only=(args.region == "MTG"))
            log_dir = os.path.join(args.log_base, f"{args.region}_{gene_load_cell_type}_embedding{ge_str}{attn_str}{pca_str}_{args.experiment}_testfold{test_fold_idx+1}")
            writer = SummaryWriter(log_dir)
            best_f1, best_state = -1.0, None
            for epoch in range(args.epochs):
                train_loss = train_epoch_microglia(model, train_loader, criterion_cerad, criterion_braak, optimizer, device, lambda_entropy=lambda_entropy)
                writer.add_scalar("Loss/Train", train_loss, epoch)
                f1 = compute_f1(model, test_loader)
                writer.add_scalar("Metrics/Test_F1", f1, epoch)
                if f1 > best_f1:
                    best_f1 = f1
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if best_state is not None:
                model.load_state_dict(best_state)
            baseline_f1 = best_f1

        baseline_f1_list.append(baseline_f1)
        writer.add_scalar("Metrics/Best_F1", baseline_f1, 0)
        writer.close()
        print(f"  Best F1: {baseline_f1:.4f}")

        for g in range(n_genes):
            drops = []
            for _ in range(args.n_permute):
                test_X_perm = all_fold_data[test_fold_idx]["X"].copy()
                _permute_gene_embedding(test_X_perm, g, rng)
                test_2d_perm = test_X_perm.reshape(n_test, -1)
                test_scaled_perm = scaler.transform(test_2d_perm).reshape(n_test, n_genes, all_fold_data[test_fold_idx]["X"].shape[2])
                if pca is not None:
                    test_scaled_perm = pca.transform(test_scaled_perm.reshape(-1, test_scaled_perm.shape[2])).reshape(n_test, n_genes, args.pca_components)
                if pooling:
                    test_input_perm = np.mean(test_scaled_perm, axis=1)
                else:
                    test_input_perm = test_scaled_perm
                if args.cell_type == "Astrocytes" and args.region != "MTG":
                    test_ds_perm = EmbeddingDataset(test_input_perm, test_ptau, test_thal)
                else:
                    test_ds_perm = EmbeddingDatasetMicroglia(test_input_perm, test_cerad, test_braak)
                test_loader_perm = DataLoader(test_ds_perm, batch_size=args.batch_size, shuffle=False)
                perm_f1 = compute_f1(model, test_loader_perm)
                drops.append(baseline_f1 - perm_f1)
            gene_f1_drops[g].append(np.mean(drops))
            if (g + 1) % 200 == 0:
                print(f"  Gene {g + 1}/{n_genes} done")

    print(f"\nAverage best F1 across {n_folds} folds: {np.mean(baseline_f1_list):.4f}")
    return _aggregate_and_rank(gene_f1_drops, gene_names, n_folds)


def _aggregate_and_rank(gene_f1_drops, gene_names, n_folds):
    """Aggregate F1 drops across folds: mean per gene, then rank by mean drop (higher = more important)."""
    rows = []
    fold_keys = [f"fold{i+1}_f1_drop" for i in range(n_folds)]
    for g, drops in gene_f1_drops.items():
        while len(drops) < n_folds:
            drops.append(np.nan)
        mean_drop = float(np.nanmean(drops))
        row = {"gene": gene_names[g], "gene_index": g, "mean_f1_drop": mean_drop}
        for i, k in enumerate(fold_keys):
            row[k] = drops[i] if i < len(drops) else np.nan
        rows.append(row)
    rows.sort(key=lambda r: -r["mean_f1_drop"])
    for rank, r in enumerate(rows, start=1):
        r["rank"] = rank
    return rows


def main():
    args = parse_args()
    print(f"Region={args.region}, cell_type={args.cell_type}, model={args.model}, n_folds={args.n_folds}")

    if args.model == "expression":
        rows = run_permutation_importance_expression(args)
    else:
        rows = run_permutation_importance_embedding(args)

    out_path = args.output
    if out_path is None:
        ge_str = "_ge" if getattr(args, "gene_encoder", False) else ""
        attn_str = "_attn" if args.attention else ""
        pca_str = f"_pca{args.pca_components}" if args.pca_components else ""
        cell_label = "MTG" if args.region == "MTG" else args.cell_type
        model_name = f"{args.region}_{cell_label}_{args.model}{ge_str}{attn_str}{pca_str}_{args.experiment}"
        out_path = os.path.join(
            "ranked_lists", model_name, "ranked_genes.csv",
        )
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fieldnames = ["rank", "gene", "gene_index", "mean_f1_drop"]
    if rows:
        fieldnames += [k for k in rows[0].keys() if k.startswith("fold")]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} genes to {out_path}")


if __name__ == "__main__":
    main()
