"""Generate v2 ranked genes by permutation importance.

This mirrors the v2 training pipeline in train_model_v2.py:
  - reads create_data_v2 pickles
  - predicts the v2 Thal label for Astrocytes, Microglia, and MTG
  - supports expression, embedding, and random_embedding inputs
  - can load Optuna/autosearch *_best_params.json files

For each held-out fold, the script trains the requested model, keeps the best
state by held-out F1, permutes one gene in the held-out fold, and ranks genes
by the mean F1 drop across folds.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from collections import defaultdict
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from torch.utils.data import DataLoader

from baseline_model import BaselineRankingModel
from gene_encoder_model import GeneEncoderModel
from mil_model import MILModel
from train_model_v2 import (
    ALL_MODEL_TYPES,
    INPUT_TYPES,
    RANDOM_EMBEDDING_DIM,
    _aggregate_patients,
    _build_mil_bags,
    _load_v2_pickle,
    _merge_folds,
    _select_folds,
)
from train_utils import (
    BagDataset,
    EmbeddingAugmentConfig,
    augment_config_from_args,
    SimpleDataset,
    collate_mil_eval,
    evaluate_mil_model,
    evaluate_standard,
    scaler_partial_fit_bag,
    scaler_transform_bag_inplace,
    train_epoch_mil,
    train_epoch_standard,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate v2 ranked genes by permutation F1 drop"
    )
    parser.add_argument("--region", type=str, required=True)
    parser.add_argument(
        "--cell_type",
        type=str,
        default="Astrocytes",
        choices=["Astrocytes", "Microglia"],
    )
    parser.add_argument("--data", type=str, default="Atlas", choices=["Atlas", "SEAAD", "ROSMAP"],
                        help="Dataset. SEAAD/ROSMAP load <data>_<cell_type>_v2_data (region is a label).")
    parser.add_argument("--multitask", action="store_true",
                        help="Rank by the combined 1:1 Thal+Braak F1 drop (matches a multitask model).")
    parser.add_argument("--braak_weight", type=float, default=1.0,
                        help="Braak loss weight when --multitask (default 1.0).")
    parser.add_argument(
        "--input_type",
        type=str,
        required=True,
        choices=sorted(INPUT_TYPES),
    )
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=sorted(ALL_MODEL_TYPES - {"scvi"}),
        help="scvi is not supported for permutation ranking in this script.",
    )
    parser.add_argument(
        "--best_params_json",
        type=str,
        default=None,
        help="Optional autosearch *_best_params.json; trial_params override matching CLI args.",
    )
    parser.add_argument("--expr_transform", type=str, default="log1p")
    parser.add_argument("--n_folds_eval", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--min_epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--n_permute", type=int, default=1)
    parser.add_argument(
        "--progress_every",
        type=int,
        default=100,
        help="Print permutation progress every N genes (0 disables per-gene progress).",
    )
    parser.add_argument(
        "--max_genes",
        type=int,
        default=None,
        help="Optional debug limit; ranks only the first N genes.",
    )

    # Common train_model_v2 hyperparameters.
    parser.add_argument("--loss_ptau", type=str, default="none", choices=["none", "mse", "mae", "huber"])
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--thal_ordinal", action="store_true")
    parser.add_argument("--rank_metric", type=str, default="f1", choices=["f1", "auroc"],
                        help="Metric whose per-gene drop defines importance. 'auroc' needs a "
                             "binary task (use --binarize); it ranks by the drop in P(positive) "
                             "AUROC, which is far more discriminative than hard-label F1.")
    parser.add_argument("--binarize", type=int, default=None, metavar="THR",
                        help="Binarize Thal to {0:<THR, 1:>=THR} before ranking (e.g. --binarize 3). "
                             "Required for a well-defined --rank_metric auroc.")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--pca_components", type=int, default=None)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--mlp_hidden_sizes", type=str, default="")
    parser.add_argument("--n_hidden_layers", type=int, default=1, choices=[1, 3])
    parser.add_argument("--hidden_dim_3layer", type=int, default=32)
    parser.add_argument("--attention", action="store_true")
    parser.add_argument("--attention_heads", type=int, default=1)
    parser.add_argument("--lambda_entropy", type=float, default=1e-3)
    parser.add_argument("--mil_cells_per_patient", type=int, default=128)
    parser.add_argument("--mil_hidden_dim", type=int, default=128)
    parser.add_argument("--mil_attn_dim", type=int, default=64)
    parser.add_argument("--mil_encoder_layers", type=int, default=2)
    # Online augmentation (train only) -- match train_model_v2 so a retrain reproduces a trial.
    parser.add_argument("--aug_scale_jitter", type=float, default=0.0)
    parser.add_argument("--aug_per_gene_scale_jitter", type=float, default=0.0)
    parser.add_argument("--aug_gene_dropout", type=float, default=0.0)
    parser.add_argument("--aug_emb_dim_dropout", type=float, default=0.0)
    parser.add_argument("--aug_feature_dropout", type=float, default=0.0)
    parser.add_argument("--aug_gene_block_cutout_max", type=int, default=0)
    parser.add_argument("--aug_gene_block_cutout_prob", type=float, default=0.5)
    parser.add_argument("--aug_noise_std", type=float, default=0.0)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def apply_best_params(args: argparse.Namespace) -> argparse.Namespace:
    if not args.best_params_json:
        return args
    with open(args.best_params_json, encoding="utf-8") as f:
        payload = json.load(f)
    params = payload.get("trial_params") or payload.get("params") or {}
    for key, value in params.items():
        if hasattr(args, key):
            setattr(args, key, value)
    return args


def _binary_auroc(true_labels, scores) -> float:
    """AUROC of P(positive) vs binary truth. NaN when a fold has only one class."""
    t = np.asarray([int(x) for x in true_labels])
    if len(np.unique(t)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(t, np.asarray(scores, dtype=np.float64)))
    except Exception:
        return float("nan")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_criteria(args: argparse.Namespace) -> Tuple[Optional[nn.Module], nn.Module]:
    loss_ptau = str(getattr(args, "loss_ptau", "none")).lower()
    if loss_ptau == "none":
        criterion_ptau = None
    elif loss_ptau == "mse":
        criterion_ptau = nn.MSELoss()
    elif loss_ptau == "mae":
        criterion_ptau = nn.L1Loss()
    elif loss_ptau == "huber":
        criterion_ptau = nn.HuberLoss()
    else:
        raise ValueError(f"Unknown loss_ptau: {loss_ptau}")

    smoothing = float(getattr(args, "label_smoothing", 0.0) or 0.0)
    criterion_thal = (
        nn.CrossEntropyLoss(label_smoothing=smoothing)
        if smoothing > 0
        else nn.CrossEntropyLoss()
    )
    criterion_braak = (
        nn.CrossEntropyLoss(label_smoothing=smoothing) if getattr(args, "multitask", False) else None
    )
    return criterion_ptau, criterion_thal, criterion_braak


def load_fold_data(args: argparse.Namespace):
    data = _load_v2_pickle(args.region, args.cell_type, expr_transform=args.expr_transform,
                           dataset=getattr(args, "data", "Atlas"))
    raw_folds = _select_folds(data, args.input_type)
    patient_info_folds = data["patient_info_folds"]
    ptau_folds = data["ptau_folds"]
    thal_folds = data["thal_folds"]
    multitask = bool(getattr(args, "multitask", False))
    braak_folds = data.get("braak_folds") if multitask else None
    gene_names = list(data["gene_names"])
    embedding_dim = int(data.get("embedding_dim", 0) or 0)
    if args.input_type == "random_embedding":
        embedding_dim = RANDOM_EMBEDDING_DIM
    if multitask and (braak_folds is None or not any(len(f) for f in braak_folds)):
        raise ValueError(f"--multitask requested but no Braak labels in the {args.data} "
                         f"{args.cell_type} v2 data.")

    n_folds_eval = int(args.n_folds_eval or len(raw_folds))
    if n_folds_eval != len(raw_folds):
        if len(raw_folds) % n_folds_eval != 0:
            raise ValueError(f"--n_folds_eval={n_folds_eval} must divide {len(raw_folds)}")
        group_size = len(raw_folds) // n_folds_eval
        raw_folds = _merge_folds(raw_folds, group_size)
        patient_info_folds = _merge_folds(patient_info_folds, group_size)
        ptau_folds = _merge_folds(ptau_folds, group_size)
        thal_folds = _merge_folds(thal_folds, group_size)
        if multitask:
            braak_folds = _merge_folds(braak_folds, group_size)

    # Labels can be stored as float-strings ('5.0'); normalize to clean ints so the
    # downstream int(...) in _build_mil_bags / _aggregate_patients doesn't choke.
    thal_folds = [[int(float(x)) for x in f] for f in thal_folds]
    if multitask:
        braak_folds = [[int(float(x)) for x in f] for f in braak_folds]

    # Optional binarization of Thal (0:<thr, 1:>=thr) -- makes AUROC well defined.
    binarize = getattr(args, "binarize", None)
    if binarize is not None:
        thr = int(binarize)
        thal_folds = [[1 if x >= thr else 0 for x in f] for f in thal_folds]

    all_thal = [int(x) for fold in thal_folds for x in fold]
    thal_classes = sorted(set(all_thal))
    thal_to_idx = {c: i for i, c in enumerate(thal_classes)}
    idx_to_thal = {i: c for c, i in thal_to_idx.items()}

    braak_to_idx = idx_to_braak = None
    n_braak_classes = None
    if multitask:
        all_braak = [int(float(x)) for fold in braak_folds for x in fold]
        braak_classes = sorted(set(all_braak))
        braak_to_idx = {c: i for i, c in enumerate(braak_classes)}
        idx_to_braak = {i: c for c, i in braak_to_idx.items()}
        n_braak_classes = len(braak_classes)

    use_mil = args.model_type == "mil"
    all_fold_data = []
    for fi in range(len(raw_folds)):
        bkw = dict(braak_fold=braak_folds[fi], braak_to_idx=braak_to_idx) if multitask else {}
        if use_mil:
            all_fold_data.append(
                _build_mil_bags(raw_folds[fi], patient_info_folds[fi], ptau_folds[fi],
                                thal_folds[fi], thal_to_idx, **bkw)
            )
        else:
            all_fold_data.append(
                _aggregate_patients(raw_folds[fi], patient_info_folds[fi], ptau_folds[fi],
                                    thal_folds[fi], thal_to_idx, **bkw)
            )

    return {
        "folds": all_fold_data,
        "gene_names": gene_names,
        "embedding_dim": embedding_dim,
        "idx_to_thal": idx_to_thal,
        "n_thal_classes": len(thal_classes),
        "idx_to_braak": idx_to_braak,
        "n_braak_classes": n_braak_classes,
        "multitask": multitask,
    }


def _infer_gene_width(args: argparse.Namespace, feature_dim: int, n_genes: int) -> int:
    if args.input_type == "expression":
        return 1
    if n_genes > 0 and feature_dim % n_genes == 0:
        return feature_dim // n_genes
    raise ValueError(f"Feature dim {feature_dim} is not divisible by n_genes={n_genes}")


def _permute_patient_gene_matrix(
    x: np.ndarray,
    gene_idx: int,
    gene_width: int,
    rng: np.random.Generator,
) -> np.ndarray:
    xp = np.asarray(x, dtype=np.float32).copy()
    start = gene_idx * gene_width
    end = start + gene_width
    perm = rng.permutation(xp.shape[0])
    xp[:, start:end] = xp[perm, start:end]
    return xp


def _permute_mil_gene_bags(
    bags: List[np.ndarray],
    gene_idx: int,
    gene_width: int,
    rng: np.random.Generator,
) -> List[np.ndarray]:
    out = [np.asarray(b, dtype=np.float32).copy() for b in bags]
    start = gene_idx * gene_width
    end = start + gene_width
    total_cells = sum(b.shape[0] for b in out)
    if total_cells <= 1:
        return out
    stacked = np.concatenate([b[:, start:end] for b in out], axis=0)
    stacked = stacked[rng.permutation(stacked.shape[0])]
    offset = 0
    for bag in out:
        n = bag.shape[0]
        bag[:, start:end] = stacked[offset : offset + n]
        offset += n
    return out


def _fit_mil_transform(
    train_bags: List[np.ndarray],
    pca_components: Optional[int],
    seed: int,
) -> Tuple[List[np.ndarray], StandardScaler, Optional[PCA], int, int]:
    """Fit MIL StandardScaler/PCA once per fold and transform train bags."""
    ref = train_bags[0]
    feat_dim = ref.shape[1] if ref.ndim == 2 else ref.shape[1] * ref.shape[2]

    scaler = StandardScaler()
    for bag in train_bags:
        scaler_partial_fit_bag(scaler, bag)

    train_scaled = [
        scaler_transform_bag_inplace(scaler, np.asarray(bag, dtype=np.float32).copy())
        for bag in train_bags
    ]

    pca = None
    cell_in = feat_dim
    if pca_components and pca_components < feat_dim:
        pool = []
        n_pool = 0
        max_fit = 100_000
        for bag in train_scaled:
            flat = bag.reshape(bag.shape[0], feat_dim).astype(np.float32, copy=False)
            pool.append(flat)
            n_pool += flat.shape[0]
            if n_pool >= max_fit * 2:
                break
        big = np.concatenate(pool, axis=0)
        if big.shape[0] > max_fit:
            rng = np.random.default_rng(seed)
            idx = rng.choice(big.shape[0], size=max_fit, replace=False)
            big = big[idx]
        pca = PCA(
            n_components=min(int(pca_components), feat_dim),
            svd_solver="randomized",
            random_state=seed,
        )
        pca.fit(big)
        cell_in = int(pca.n_components_)
        train_scaled = [
            pca.transform(bag.reshape(bag.shape[0], feat_dim)).astype(np.float32)
            for bag in train_scaled
        ]

    return train_scaled, scaler, pca, cell_in, feat_dim


def _apply_mil_transform(
    bags: List[np.ndarray],
    scaler: StandardScaler,
    pca: Optional[PCA],
    feat_dim: int,
) -> List[np.ndarray]:
    """Apply an already-fitted MIL StandardScaler/PCA to bag copies."""
    scaled = [
        scaler_transform_bag_inplace(scaler, np.asarray(bag, dtype=np.float32).copy())
        for bag in bags
    ]
    if pca is not None:
        scaled = [
            pca.transform(bag.reshape(bag.shape[0], feat_dim)).astype(np.float32)
            for bag in scaled
        ]
    return scaled


def _transform_matrix(
    train_x: np.ndarray,
    test_x: np.ndarray,
    args: argparse.Namespace,
    seed: int,
):
    train_x = np.nan_to_num(np.asarray(train_x, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    test_x = np.nan_to_num(np.asarray(test_x, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_x)
    test_scaled = scaler.transform(test_x)

    pca = None
    pca_comp = getattr(args, "pca_components", None)
    if args.model_type == "gene_encoder" and args.input_type in ("embedding", "random_embedding"):
        pca_comp = None
    if pca_comp and pca_comp < train_scaled.shape[1]:
        n_comp = min(int(pca_comp), train_scaled.shape[0], train_scaled.shape[1])
        pca = PCA(n_components=n_comp, random_state=seed)
        train_scaled = pca.fit_transform(train_scaled)
        test_scaled = pca.transform(test_scaled)
    return train_scaled.astype(np.float32), test_scaled.astype(np.float32), scaler, pca


def _transform_test_matrix(test_x: np.ndarray, scaler, pca) -> np.ndarray:
    test_x = np.nan_to_num(np.asarray(test_x, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if scaler is None:
        raise ValueError("Permutation transform requires a StandardScaler instance")
    test_scaled = scaler.transform(test_x)
    if pca is not None:
        test_scaled = pca.transform(test_scaled)
    return test_scaled.astype(np.float32)


def run_sklearn_fold(
    args: argparse.Namespace,
    fold_data: List[dict],
    test_fi: int,
    idx_to_thal: dict,
    n_genes: int,
    rng: np.random.Generator,
    n_permute_genes: Optional[int] = None,
) -> Tuple[float, Dict[int, float]]:
    train_indices = [i for i in range(len(fold_data)) if i != test_fi]
    train_x = np.concatenate([fold_data[i]["embeddings"] for i in train_indices])
    train_y = np.concatenate([fold_data[i]["thal"] for i in train_indices])
    test_x = fold_data[test_fi]["embeddings"]
    test_y = fold_data[test_fi]["thal"]

    train_scaled, test_scaled, scaler, pca = _transform_matrix(train_x, test_x, args, args.seed + test_fi)
    if args.model_type == "logreg":
        clf = LogisticRegression(
            C=float(args.C),
            max_iter=2000,
            random_state=args.seed + test_fi,
            multi_class="multinomial",
            solver="lbfgs",
        )
    elif args.model_type == "svm":
        clf = LinearSVC(C=float(args.C), max_iter=5000, random_state=args.seed + test_fi, dual="auto")
    else:
        raise ValueError(f"Unknown sklearn model: {args.model_type}")
    clf.fit(train_scaled, train_y)
    pred = clf.predict(test_scaled)
    baseline_f1 = f1_score(
        [idx_to_thal[int(y)] for y in test_y],
        [idx_to_thal[int(y)] for y in pred],
        average="weighted",
    )

    gene_width = _infer_gene_width(args, test_x.shape[1], n_genes)
    n_perm = n_permute_genes or n_genes
    drops = {}
    for g in range(n_perm):
        if args.progress_every and (g == 0 or (g + 1) % int(args.progress_every) == 0):
            print(f"    permuting gene {g + 1}/{n_perm}", flush=True)
        per_rep = []
        for _ in range(args.n_permute):
            perm_raw = _permute_patient_gene_matrix(test_x, g, gene_width, rng)
            perm_scaled = _transform_test_matrix(perm_raw, scaler, pca)
            perm_pred = clf.predict(perm_scaled)
            perm_f1 = f1_score(
                [idx_to_thal[int(y)] for y in test_y],
                [idx_to_thal[int(y)] for y in perm_pred],
                average="weighted",
            )
            per_rep.append(baseline_f1 - perm_f1)
        drops[g] = float(np.mean(per_rep))
    return baseline_f1, drops


def _reshape_for_gene_encoder(
    train_x: np.ndarray,
    test_x: np.ndarray,
    args: argparse.Namespace,
    n_genes: int,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    if args.input_type in ("embedding", "random_embedding"):
        if train_x.shape[1] % n_genes != 0:
            raise ValueError(f"Cannot reshape {train_x.shape[1]} features into n_genes={n_genes}")
        actual_g = n_genes
        actual_d = train_x.shape[1] // n_genes
    else:
        actual_g = 1
        actual_d = train_x.shape[1]
    return (
        train_x.reshape(-1, actual_g, actual_d),
        test_x.reshape(-1, actual_g, actual_d),
        actual_g,
        actual_d,
    )


def build_standard_model(
    args: argparse.Namespace,
    train_x: np.ndarray,
    n_thal_classes: int,
    n_genes: int,
    n_braak_classes: Optional[int] = None,
) -> nn.Module:
    dropout = float(args.dropout)
    if args.model_type == "gene_encoder":
        n_dims = train_x.shape[2]
        return GeneEncoderModel(
            n_dims=n_dims,
            n_thal_classes=n_thal_classes,
            dropout=dropout,
            n_genes=train_x.shape[1],
            thal_ordinal=bool(args.thal_ordinal),
        )

    input_shape = (train_x.shape[1], train_x.shape[2]) if train_x.ndim == 3 else None
    input_dim = int(np.prod(train_x.shape[1:])) if input_shape is not None else train_x.shape[1]
    mlp_sizes = None
    if str(args.mlp_hidden_sizes or "").strip():
        mlp_sizes = [int(x) for x in str(args.mlp_hidden_sizes).split(",") if x.strip()]
    return BaselineRankingModel(
        input_dim=input_dim,
        input_shape=input_shape,
        n_thal_classes=n_thal_classes,
        use_attention=bool(args.attention),
        num_heads=int(args.attention_heads),
        dropout=dropout,
        n_hidden_layers=int(args.n_hidden_layers),
        hidden_dim_3layer=int(args.hidden_dim_3layer),
        mlp_hidden_sizes=mlp_sizes,
        thal_ordinal=bool(args.thal_ordinal),
        n_braak_classes=n_braak_classes,
    )


def run_torch_fold(
    args: argparse.Namespace,
    fold_data: List[dict],
    test_fi: int,
    idx_to_thal: dict,
    n_thal_classes: int,
    n_genes: int,
    device: torch.device,
    rng: np.random.Generator,
    idx_to_braak: Optional[dict] = None,
    n_braak_classes: Optional[int] = None,
    multitask: bool = False,
    n_permute_genes: Optional[int] = None,
) -> Tuple[float, Dict[int, float]]:
    train_indices = [i for i in range(len(fold_data)) if i != test_fi]
    criterion_ptau, criterion_thal, criterion_braak = make_criteria(args)
    thal_ordinal = bool(args.thal_ordinal)
    braak_weight = float(getattr(args, "braak_weight", 1.0))
    # Extra kwargs so eval also reports/optimizes Braak when multitask.
    eval_kw = dict(thal_ordinal=thal_ordinal, n_thal_classes=n_thal_classes)
    train_braak_kw = dict(braak_weight=braak_weight)
    if multitask:
        eval_kw.update(criterion_braak=criterion_braak, idx_to_braak=idx_to_braak,
                       braak_weight=braak_weight)
        train_braak_kw.update(criterion_braak=criterion_braak)

    rank_metric = str(getattr(args, "rank_metric", "f1")).lower()

    def _sel(thal_f1, braak_f1):
        return (thal_f1 + braak_f1) / 2.0 if (multitask and braak_f1 is not None) else thal_f1

    def _score_loader(loader) -> Tuple[float, float]:
        """Return (ranking_score, loss). Score is higher-is-better: F1 (default) or
        binary AUROC on P(top class) when rank_metric == 'auroc'."""
        res = run_eval(
            model, loader, criterion_ptau, criterion_thal, device, idx_to_thal,
            return_preds=True, **eval_kw,
        )
        # return_preds tuple: (loss, mse, mae, thal_f1, thal_true, thal_pred,
        #                       braak_f1, braak_true, braak_pred, thal_probs, braak_probs)
        loss = float(res[0])
        if rank_metric == "auroc":
            thal_true = res[4]
            thal_scores = [p[-1] for p in res[9]]  # P(positive/top): ordinal->sigmoid, 2-class->softmax[1]
            return _binary_auroc(thal_true, thal_scores), loss
        return _sel(res[3], res[6]), loss

    if args.model_type == "mil":
        train_bags, train_ptau, train_thal, train_braak = [], [], [], []
        for i in train_indices:
            train_bags.extend(fold_data[i]["cell_bags"])
            train_ptau.extend(fold_data[i]["ptau"])
            train_thal.extend(fold_data[i]["thal"])
            if multitask:
                train_braak.extend(fold_data[i]["braak"])
        test_bags_raw = list(fold_data[test_fi]["cell_bags"])
        test_ptau = fold_data[test_fi]["ptau"]
        test_thal = fold_data[test_fi]["thal"]
        test_braak = fold_data[test_fi]["braak"] if multitask else None
        train_braak_arr = np.asarray(train_braak, dtype=np.int64) if multitask else None
        raw_gene_width = _infer_gene_width(args, test_bags_raw[0].shape[1], n_genes)

        train_bags_tf, mil_scaler, mil_pca, cell_in, mil_feat_dim = _fit_mil_transform(
            train_bags,
            getattr(args, "pca_components", None),
            seed=args.seed + test_fi,
        )
        test_bags_tf = _apply_mil_transform(test_bags_raw, mil_scaler, mil_pca, mil_feat_dim)
        train_ds = BagDataset(
            train_bags_tf,
            np.asarray(train_ptau, dtype=np.float32),
            np.asarray(train_thal, dtype=np.int64),
            train=True,
            mil_cells_per_patient=int(args.mil_cells_per_patient),
            augment_config=augment_config_from_args(args),
            seed=args.seed + test_fi,
            braak=train_braak_arr,
        )
        test_ds = BagDataset(
            test_bags_tf,
            test_ptau,
            test_thal,
            train=False,
            mil_cells_per_patient=int(args.mil_cells_per_patient),
            seed=args.seed + test_fi,
            braak=test_braak,
        )
        train_loader = DataLoader(train_ds, batch_size=int(args.batch_size), shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=collate_mil_eval)
        model = MILModel(
            cell_input_dim=cell_in,
            n_thal_classes=n_thal_classes,
            mil_hidden_dim=int(args.mil_hidden_dim),
            mil_attn_dim=int(args.mil_attn_dim),
            mil_encoder_layers=int(args.mil_encoder_layers),
            dropout=float(args.dropout),
            thal_ordinal=thal_ordinal,
            n_braak_classes=n_braak_classes,
        ).to(device)
        run_train = train_epoch_mil
        run_eval = evaluate_mil_model
        permute_eval_data = (
            "mil",
            test_bags_raw,
            raw_gene_width,
            mil_scaler,
            mil_pca,
            mil_feat_dim,
        )
    else:
        train_x_raw = np.concatenate([fold_data[i]["embeddings"] for i in train_indices])
        train_ptau = np.concatenate([fold_data[i]["ptau"] for i in train_indices])
        train_thal = np.concatenate([fold_data[i]["thal"] for i in train_indices])
        train_braak_arr = (np.concatenate([fold_data[i]["braak"] for i in train_indices])
                           if multitask else None)
        test_x_raw = fold_data[test_fi]["embeddings"]
        test_ptau = fold_data[test_fi]["ptau"]
        test_thal = fold_data[test_fi]["thal"]
        test_braak = fold_data[test_fi]["braak"] if multitask else None
        raw_gene_width = _infer_gene_width(args, test_x_raw.shape[1], n_genes)

        train_x, test_x, scaler, pca = _transform_matrix(train_x_raw, test_x_raw, args, args.seed + test_fi)
        if args.model_type == "gene_encoder":
            train_x, test_x, _, _ = _reshape_for_gene_encoder(train_x, test_x, args, n_genes)

        train_ds = SimpleDataset(train_x, train_ptau, train_thal, train=True,
                                 augment_config=augment_config_from_args(args), braak=train_braak_arr)
        test_ds = SimpleDataset(test_x, test_ptau, test_thal, train=False, braak=test_braak)
        train_loader = DataLoader(train_ds, batch_size=int(args.batch_size), shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=int(args.batch_size), shuffle=False)
        model = build_standard_model(args, train_x, n_thal_classes, n_genes,
                                     n_braak_classes=n_braak_classes).to(device)
        run_train = train_epoch_standard
        run_eval = evaluate_standard
        permute_eval_data = ("standard", test_x_raw, raw_gene_width, scaler, pca)

    optimizer = optim.Adam(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    best_score = float("-inf")
    best_loss = float("inf")
    best_state = None
    epochs_no_improve = 0
    last_epoch = -1
    for epoch in range(int(args.epochs)):
        last_epoch = epoch
        if args.model_type == "mil":
            run_train(
                model,
                train_loader,
                criterion_ptau,
                criterion_thal,
                optimizer,
                device,
                thal_ordinal=thal_ordinal,
                n_thal_classes=n_thal_classes,
                **train_braak_kw,
            )
        else:
            run_train(
                model,
                train_loader,
                criterion_ptau,
                criterion_thal,
                optimizer,
                device,
                lambda_entropy=float(args.lambda_entropy) if args.model_type == "gene_encoder" else 0.0,
                thal_ordinal=thal_ordinal,
                n_thal_classes=n_thal_classes,
                **train_braak_kw,
            )
        test_score, test_loss = _score_loader(test_loader)
        # NaN score (single-class fold under AUROC) never counts as an improvement.
        improved = np.isfinite(test_score) and (
            test_score > best_score or (test_score == best_score and test_loss < best_loss)
        )
        if improved:
            best_score = float(test_score)
            best_loss = float(test_loss)
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if (
            int(args.patience) > 0
            and epochs_no_improve >= int(args.patience)
            and (epoch + 1) >= int(args.min_epochs)
        ):
            break
    if best_state is not None:
        model.load_state_dict(best_state)

    # Baseline for the drop. If no epoch produced a finite score (e.g. single-class
    # fold under AUROC), re-score the final model; may still be NaN -> fold skipped.
    baseline = best_score
    if not np.isfinite(baseline):
        baseline, _ = _score_loader(test_loader)

    if args.verbose:
        print(f"    baseline {rank_metric}={baseline:.4f}, epochs={last_epoch + 1}")

    n_perm = n_permute_genes or n_genes
    drops = {}
    if not np.isfinite(baseline):
        # Undefined baseline -> every gene's drop is NaN for this fold.
        for g in range(n_perm):
            drops[g] = float("nan")
        return baseline, drops
    for g in range(n_perm):
        if args.progress_every and (g == 0 or (g + 1) % int(args.progress_every) == 0):
            print(f"    permuting gene {g + 1}/{n_perm}", flush=True)
        per_rep = []
        for _ in range(args.n_permute):
            mode = permute_eval_data[0]
            if mode == "mil":
                _, test_bags_raw, gene_width, mil_scaler, mil_pca, mil_feat_dim = permute_eval_data
                perm_raw = _permute_mil_gene_bags(test_bags_raw, g, gene_width, rng)
                perm_bags = _apply_mil_transform(perm_raw, mil_scaler, mil_pca, mil_feat_dim)
                perm_ds = BagDataset(
                    perm_bags,
                    test_ptau,
                    test_thal,
                    train=False,
                    mil_cells_per_patient=int(args.mil_cells_per_patient),
                    seed=args.seed + test_fi,
                    braak=test_braak,
                )
                perm_loader = DataLoader(perm_ds, batch_size=1, shuffle=False, collate_fn=collate_mil_eval)
            else:
                _, test_x_raw, gene_width, scaler, pca = permute_eval_data
                perm_raw = _permute_patient_gene_matrix(test_x_raw, g, gene_width, rng)
                perm_x = _transform_test_matrix(perm_raw, scaler, pca)
                if args.model_type == "gene_encoder":
                    _, perm_x, _, _ = _reshape_for_gene_encoder(perm_x, perm_x, args, n_genes)
                perm_ds = SimpleDataset(perm_x, test_ptau, test_thal, train=False, braak=test_braak)
                perm_loader = DataLoader(perm_ds, batch_size=int(args.batch_size), shuffle=False)
            perm_score, _ = _score_loader(perm_loader)
            per_rep.append(baseline - perm_score)
        drops[g] = float(np.mean(per_rep))
    return baseline, drops


def aggregate_rank(gene_drops: Dict[int, List[float]], gene_names: List[str], n_folds: int,
                   metric: str = "f1") -> List[dict]:
    mean_col = f"mean_{metric}_drop"
    rows = []
    for g, drops in gene_drops.items():
        while len(drops) < n_folds:
            drops.append(np.nan)
        row = {
            "gene": gene_names[g],
            "gene_index": g,
            mean_col: float(np.nanmean(drops)),
        }
        for i, value in enumerate(drops):
            row[f"fold{i + 1}_{metric}_drop"] = value
        rows.append(row)
    rows.sort(key=lambda r: (np.isnan(r[mean_col]), -np.nan_to_num(r[mean_col], nan=-np.inf)))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def default_output_path(args: argparse.Namespace) -> str:
    bits = [
        args.region,
        args.cell_type,
        args.input_type,
        args.model_type,
    ]
    if args.thal_ordinal:
        bits.append("ordinal")
    if args.pca_components:
        bits.append(f"pca{args.pca_components}")
    if getattr(args, "binarize", None) is not None:
        bits.append(f"bin{int(args.binarize)}")
    if str(getattr(args, "rank_metric", "f1")).lower() != "f1":
        bits.append(str(args.rank_metric).lower())
    return os.path.join("ranked_lists_v2", "_".join(bits), "ranked_genes.csv")


def main() -> None:
    args = apply_best_params(parse_args())
    if args.model_type == "scvi":
        raise ValueError("scvi is not supported by generate_ranked_list_v2.py")
    set_seed(int(args.seed))

    loaded = load_fold_data(args)
    fold_data = loaded["folds"]
    gene_names = loaded["gene_names"]
    idx_to_thal = loaded["idx_to_thal"]
    n_thal_classes = loaded["n_thal_classes"]
    idx_to_braak = loaded["idx_to_braak"]
    n_braak_classes = loaded["n_braak_classes"]
    multitask = loaded["multitask"]
    if multitask and args.model_type not in ("mil", "flatten_mlp", "mlp"):
        raise ValueError(f"--multitask supports mil/flatten_mlp/mlp, not {args.model_type}")
    rank_metric = str(getattr(args, "rank_metric", "f1")).lower()
    if rank_metric == "auroc":
        if args.model_type in ("logreg", "svm"):
            raise ValueError("--rank_metric auroc is not supported for logreg/svm (F1 path only).")
        if getattr(args, "binarize", None) is None and n_thal_classes > 2:
            print("WARNING: --rank_metric auroc without --binarize on a multiclass task uses "
                  "P(top class) as the score, which is a poorly defined ROC target. "
                  "Strongly recommend --binarize.")
    n_genes = len(gene_names)  # TRUE gene count -> used for gene-width inference
    n_permute_genes = n_genes  # how many genes to actually permute (debug limit)
    if args.max_genes is not None:
        n_permute_genes = min(n_genes, int(args.max_genes))

    device = torch.device(args.device)
    rng = np.random.default_rng(int(args.seed))
    gene_drops: Dict[int, List[float]] = defaultdict(list)
    baseline_f1s = []

    print(
        f"v2 ranked genes: region={args.region}, cell_type={args.cell_type}, "
        f"input_type={args.input_type}, model_type={args.model_type}, genes={n_genes}"
    )

    for test_fi in range(len(fold_data)):
        print(f"\nFold {test_fi + 1}/{len(fold_data)}")
        fold_args = deepcopy(args)
        fold_args.seed = int(args.seed) + test_fi
        set_seed(int(fold_args.seed))
        if args.model_type in ("logreg", "svm"):
            base_f1, drops = run_sklearn_fold(
                fold_args,
                fold_data,
                test_fi,
                idx_to_thal,
                n_genes,
                rng,
                n_permute_genes=n_permute_genes,
            )
        else:
            base_f1, drops = run_torch_fold(
                fold_args,
                fold_data,
                test_fi,
                idx_to_thal,
                n_thal_classes,
                n_genes,
                device,
                rng,
                idx_to_braak=idx_to_braak,
                n_braak_classes=n_braak_classes,
                multitask=multitask,
                n_permute_genes=n_permute_genes,
            )
        baseline_f1s.append(base_f1)
        print(f"  baseline {rank_metric}={base_f1:.4f}")
        for g, drop in drops.items():
            gene_drops[g].append(drop)

    rows = aggregate_rank(gene_drops, gene_names, len(fold_data), metric=rank_metric)
    out_path = args.output or default_output_path(args)
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fieldnames = ["rank", "gene", "gene_index", f"mean_{rank_metric}_drop"]
    if rows:
        fieldnames.extend([k for k in rows[0] if k.startswith("fold")])
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nAverage baseline {rank_metric}: {float(np.nanmean(baseline_f1s)):.4f}")
    print(f"Wrote {len(rows)} genes to {out_path}")


if __name__ == "__main__":
    main()
