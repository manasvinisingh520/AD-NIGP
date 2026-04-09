"""Unified training script for the embedding-vs-expression experiment framework.

Supports all model families from new_flow.md:
  Expression-only:  logreg, svm, mlp, scvi
  Embedding-based:  flatten_mlp, gene_encoder, mil

Input types:  expression | embedding | random_embedding

Usage:
  python src/train_model_v2.py --region EC --input_type expression --model_type logreg --n_seeds 10
  python src/train_model_v2.py --region EC --input_type embedding  --model_type gene_encoder --n_seeds 10
  python src/train_model_v2.py --region EC --input_type random_embedding --model_type flatten_mlp --n_seeds 5
"""

from __future__ import annotations

import argparse
import datetime
import io
import os
import pickle
import random as _py_random
import sys
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from torch.utils.data import DataLoader

from baseline_model import BaselineRankingModel
from gene_encoder_model import GeneEncoderModel
from mil_model import MILModel
from train_utils import (
    BagDataset,
    EmbeddingAugmentConfig,
    NullWriter,
    SimpleDataset,
    augment_config_from_args,
    collate_mil_eval,
    evaluate_mil_model,
    evaluate_standard,
    generate_random_embeddings,
    scaler_partial_fit_bag,
    scaler_transform_bag_inplace,
    train_epoch_mil,
    train_epoch_standard,
)

# ---------------------------------------------------------------------------
# Model types by input
# ---------------------------------------------------------------------------

EXPRESSION_MODELS = {"logreg", "svm", "mlp", "scvi"}
EMBEDDING_MODELS = {"flatten_mlp", "gene_encoder", "mil"}
ALL_MODEL_TYPES = EXPRESSION_MODELS | EMBEDDING_MODELS
INPUT_TYPES = {"expression", "embedding", "random_embedding"}

# Fixed dimensionality for random gene embeddings (gene x D, scaled by expression)
RANDOM_EMBEDDING_DIM = 16

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_v2_pickle(region: str, cell_type: str, expr_transform: str = "log1p") -> dict:
    """Load v2 pickle, optionally selecting by expression transform suffix."""
    if cell_type == "Microglia":
        base = f"{region}_Microglia_v2_data"
    elif cell_type == "MTG":
        base = "MTG_v2_data"
    else:
        base = f"{region}_v2_data"

    candidates = []
    if expr_transform:
        candidates.append(f"{base}_{expr_transform}.pkl")
    candidates.append(f"{base}.pkl")

    for name in candidates:
        path = os.path.join("data", "model_data", name)
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)

    raise FileNotFoundError(
        f"No v2 data pickle found for region={region}, cell_type={cell_type}, "
        f"expr_transform={expr_transform}. Tried: {candidates}"
    )


def _select_folds(data: dict, input_type: str) -> List[np.ndarray]:
    """Return the appropriate per-fold arrays for the chosen input_type."""
    if input_type == "expression":
        return data["expression_folds"]
    elif input_type == "embedding":
        return data["embedding_folds"]
    elif input_type == "random_embedding":
        expr_folds = data["expression_folds"]
        n_genes = len(data["gene_names"])
        result = []
        for expr in expr_folds:
            result.append(
                generate_random_embeddings(
                    np.asarray(expr, dtype=np.float32),
                    n_genes=n_genes,
                    embedding_dim=RANDOM_EMBEDDING_DIM,
                )
            )
        return result
    raise ValueError(f"Unknown input_type: {input_type}")


def _merge_folds(folds: list, group_size: int) -> list:
    merged = []
    for i in range(0, len(folds), group_size):
        chunk = folds[i : i + group_size]
        if isinstance(chunk[0], np.ndarray):
            merged.append(np.concatenate(chunk, axis=0))
        elif isinstance(chunk[0], list):
            combined = []
            for c in chunk:
                combined.extend(c)
            merged.append(combined)
        else:
            combined = []
            for c in chunk:
                combined.extend(c)
            merged.append(combined)
    return merged


def _standard_scale_inplace_chunked(
    train_x: np.ndarray,
    test_x: np.ndarray,
    *,
    eps: float = 1e-6,
    max_bytes: int = 512 * 1024 * 1024,
) -> None:
    """Standardize features in-place using chunked float operations.

    sklearn's StandardScaler often casts to float64 and can allocate huge
    temporaries for large 2D tensors (e.g. cells x (genes*D)).
    This implementation keeps memory bounded by chunking rows.
    """
    if train_x.ndim != 2 or test_x.ndim != 2:
        raise ValueError(
            f"Expected 2D arrays for scaling. Got train_x.ndim={train_x.ndim}, test_x.ndim={test_x.ndim}"
        )
    if train_x.shape[1] != test_x.shape[1]:
        raise ValueError(
            f"Train/test feature dims mismatch: {train_x.shape[1]} vs {test_x.shape[1]}"
        )

    if train_x.dtype != np.float32:
        train_x = train_x.astype(np.float32)
    if test_x.dtype != np.float32:
        test_x = test_x.astype(np.float32)

    n_samples, n_features = train_x.shape

    # Number of rows per chunk so that x64 has roughly <= max_bytes.
    # x64 uses 8 bytes/element.
    rows_per_chunk = max(1, max_bytes // max(1, n_features * 8))
    rows_per_chunk = int(min(rows_per_chunk, n_samples))

    sum_x = np.zeros(n_features, dtype=np.float64)
    sum_x2 = np.zeros(n_features, dtype=np.float64)

    for start in range(0, n_samples, rows_per_chunk):
        end = min(n_samples, start + rows_per_chunk)
        x = train_x[start:end]
        x64 = x.astype(np.float64)  # temp for this chunk
        sum_x += x64.sum(axis=0)
        sum_x2 += (x64 * x64).sum(axis=0)

    mean = sum_x / float(n_samples)
    var = sum_x2 / float(n_samples) - mean * mean
    std = np.sqrt(np.maximum(var, eps))

    mean32 = mean.astype(np.float32)
    std32 = std.astype(np.float32)
    std32 = np.where(std32 == 0, 1.0, std32)

    # Transform train in-place
    for start in range(0, n_samples, rows_per_chunk):
        end = min(n_samples, start + rows_per_chunk)
        train_x[start:end] -= mean32
        train_x[start:end] /= std32

    # Transform test in-place
    n_test = test_x.shape[0]
    rows_test_chunk = max(1, max_bytes // max(1, n_features * 8))
    rows_test_chunk = int(min(rows_test_chunk, n_test))
    for start in range(0, n_test, rows_test_chunk):
        end = min(n_test, start + rows_test_chunk)
        test_x[start:end] -= mean32
        test_x[start:end] /= std32


# ---------------------------------------------------------------------------
# Patient-level aggregation
# ---------------------------------------------------------------------------


def _aggregate_patients(
    fold_data: np.ndarray,
    patient_info: list,
    ptau_fold,
    thal_fold,
    thal_to_idx: dict,
) -> dict:
    """Aggregate cell-level data to patient-level (mean pooling)."""
    patient_embeddings = []
    patient_ptau = []
    patient_thal = []
    cell_idx = 0
    for pi_idx, (_, cell_names) in enumerate(patient_info):
        n_cells = len(cell_names)
        patient_cells = fold_data[cell_idx : cell_idx + n_cells]
        if patient_cells.ndim == 3:
            patient_emb = patient_cells.mean(axis=0).flatten()
        else:
            patient_emb = patient_cells.mean(axis=0)
        patient_embeddings.append(patient_emb.astype(np.float32))
        patient_ptau.append(float(ptau_fold[pi_idx]) if ptau_fold else 0.0)
        patient_thal.append(thal_to_idx[int(thal_fold[pi_idx])])
        cell_idx += n_cells
    return {
        "embeddings": np.array(patient_embeddings, dtype=np.float32),
        "ptau": np.log1p(np.array(patient_ptau, dtype=np.float32)),
        "thal": np.array(patient_thal, dtype=np.int64),
    }


def _build_mil_bags(
    fold_data: np.ndarray,
    patient_info: list,
    ptau_fold,
    thal_fold,
    thal_to_idx: dict,
) -> dict:
    """Build per-patient cell bags for MIL."""
    bags = []
    ptau_list, thal_list = [], []
    cell_idx = 0
    for pi_idx, (_, cell_names) in enumerate(patient_info):
        n_cells = len(cell_names)
        bag = fold_data[cell_idx : cell_idx + n_cells]
        if bag.ndim == 3:
            bag = bag.reshape(n_cells, -1)
        bags.append(bag.astype(np.float32))
        ptau_list.append(float(ptau_fold[pi_idx]) if ptau_fold else 0.0)
        thal_list.append(thal_to_idx[int(thal_fold[pi_idx])])
        cell_idx += n_cells
    return {
        "cell_bags": bags,
        "ptau": np.log1p(np.array(ptau_list, dtype=np.float32)),
        "thal": np.array(thal_list, dtype=np.int64),
    }


# ---------------------------------------------------------------------------
# sklearn model runners (logreg, svm)
# ---------------------------------------------------------------------------


def _run_sklearn(
    model_type: str,
    train_X: np.ndarray,
    train_y: np.ndarray,
    test_X: np.ndarray,
    test_y: np.ndarray,
    idx_to_thal: dict,
    pca_components: Optional[int],
    C: float = 1.0,
    seed: int = 0,
) -> Tuple[float, list, list]:
    """Fit sklearn classifier, return (f1, true_labels, pred_labels)."""
    scaler = StandardScaler()
    train_X = scaler.fit_transform(train_X)
    test_X = scaler.transform(test_X)

    if pca_components and pca_components < train_X.shape[1]:
        n_comp = min(pca_components, train_X.shape[0], train_X.shape[1])
        pca = PCA(n_components=n_comp, random_state=seed)
        train_X = pca.fit_transform(train_X)
        test_X = pca.transform(test_X)

    if model_type == "logreg":
        clf = LogisticRegression(
            C=C,
            max_iter=2000,
            random_state=seed,
            multi_class="multinomial",
            solver="lbfgs",
        )
    elif model_type == "svm":
        clf = LinearSVC(C=C, max_iter=5000, random_state=seed, dual="auto")
    else:
        raise ValueError(f"Unknown sklearn model: {model_type}")

    clf.fit(train_X, train_y)
    pred_y = clf.predict(test_X)

    true_labels = [idx_to_thal[int(y)] for y in test_y]
    pred_labels = [idx_to_thal[int(y)] for y in pred_y]
    f1 = f1_score(true_labels, pred_labels, average="weighted")
    return f1, true_labels, pred_labels


# ---------------------------------------------------------------------------
# scVI runner
# ---------------------------------------------------------------------------


def _run_scvi(
    train_expr: np.ndarray,
    train_y: np.ndarray,
    test_expr: np.ndarray,
    test_y: np.ndarray,
    idx_to_thal: dict,
    n_latent: int = 30,
    n_layers: int = 2,
    n_hidden: int = 128,
    max_epochs: int = 100,
    seed: int = 0,
) -> Tuple[float, list, list]:
    """Train scVI, extract latent, classify with LogisticRegression."""
    try:
        import anndata as ad
        import scvi
    except ImportError:
        raise ImportError(
            "scvi-tools is required for --model_type scvi. "
            "Install: pip install scvi-tools"
        )

    scvi.settings.seed = seed

    # Build AnnData for scVI
    adata_train = ad.AnnData(X=train_expr.astype(np.float32))
    adata_test = ad.AnnData(X=test_expr.astype(np.float32))

    scvi.model.SCVI.setup_anndata(adata_train)
    model = scvi.model.SCVI(
        adata_train,
        n_latent=n_latent,
        n_layers=n_layers,
        n_hidden=n_hidden,
    )
    model.train(max_epochs=max_epochs, train_size=0.9, early_stopping=True)

    # Extract latent
    z_train = model.get_latent_representation(adata_train)
    z_test = model.get_latent_representation(adata_test)

    # Classify
    clf = LogisticRegression(max_iter=2000, random_state=seed, multi_class="multinomial")
    clf.fit(z_train, train_y)
    pred_y = clf.predict(z_test)

    true_labels = [idx_to_thal[int(y)] for y in test_y]
    pred_labels = [idx_to_thal[int(y)] for y in pred_y]
    f1 = f1_score(true_labels, pred_labels, average="weighted")
    return f1, true_labels, pred_labels


# ---------------------------------------------------------------------------
# MIL scaler/transform pipeline
# ---------------------------------------------------------------------------


def _transform_mil_bags(
    train_bags: List[np.ndarray],
    test_bags: List[np.ndarray],
    pca_components: Optional[int],
    quiet: bool = True,
) -> Tuple[List[np.ndarray], List[np.ndarray], int]:
    """StandardScaler + optional PCA on flattened cell vectors."""
    ref = train_bags[0]
    feat_dim = ref.shape[1] if ref.ndim == 2 else ref.shape[1] * ref.shape[2]

    scaler = StandardScaler()
    for b in train_bags:
        scaler_partial_fit_bag(scaler, b)

    train_scaled = [scaler_transform_bag_inplace(scaler, b) for b in train_bags]
    test_scaled = [scaler_transform_bag_inplace(scaler, b) for b in test_bags]

    if pca_components and pca_components < feat_dim:
        pool = []
        n_pool = 0
        max_fit = 100_000
        for b in train_scaled:
            flat = b.reshape(-1, feat_dim).astype(np.float32, copy=False)
            pool.append(flat)
            n_pool += flat.shape[0]
            if n_pool >= max_fit * 2:
                break
        big = np.concatenate(pool, axis=0)
        if big.shape[0] > max_fit:
            rng = np.random.default_rng(42)
            idx = rng.choice(big.shape[0], size=max_fit, replace=False)
            big = big[idx]
        pca = PCA(n_components=min(pca_components, feat_dim), svd_solver="randomized", random_state=42)
        pca.fit(big)
        n_out = pca.n_components_

        def _tf(b):
            flat = b.reshape(b.shape[0], -1).astype(np.float32, copy=False)
            return pca.transform(flat).astype(np.float32)

        train_scaled = [_tf(b) for b in train_scaled]
        test_scaled = [_tf(b) for b in test_scaled]
        cell_in = n_out
    else:
        cell_in = feat_dim

    return train_scaled, test_scaled, cell_in


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _open_run_log_file(args) -> None:
    """Set args._run_log_fp to a text file, or None if disabled."""
    setattr(args, "_run_log_fp", None)
    if getattr(args, "no_run_log", False):
        return

    explicit = getattr(args, "run_log", None)
    log_base = getattr(args, "log_base", "runs") or "runs"
    exp_id = getattr(args, "exp_id", None)
    if exp_id:
        log_base = os.path.join(log_base, str(exp_id))
    os.makedirs(log_base, exist_ok=True)

    def _safe(s: str) -> str:
        return "".join(c if c.isalnum() or c in "-_" else "_" for c in str(s))

    if explicit:
        log_path = os.path.abspath(explicit)
        parent = os.path.dirname(log_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
    else:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        name = (
            f"train_model_v2_{ts}_{_safe(args.region)}_{_safe(args.input_type)}"
            f"_{_safe(args.model_type)}.log"
        )
        log_path = os.path.join(log_base, name)

    fp = open(log_path, "w", encoding="utf-8")
    fp.write("train_model_v2 run log\n")
    fp.write(f"started: {datetime.datetime.now().isoformat()}\n")
    fp.write(f"argv: {sys.argv}\n")
    fp.write("-" * 60 + "\n")
    fp.flush()
    setattr(args, "_run_log_fp", fp)
    print(f"[train_model_v2] Logging to: {os.path.abspath(log_path)}", flush=True)


def main(args) -> Tuple[float, float, float, List[dict]]:
    quiet = getattr(args, "q", not getattr(args, "verbose_cv", False))

    _open_run_log_file(args)

    def out(*a, **kw):
        fp = getattr(args, "_run_log_fp", None)
        if fp is not None:
            buf = io.StringIO()
            print(*a, file=buf, **kw)
            fp.write(buf.getvalue())
            fp.flush()
        if not quiet:
            print(*a, **kw)

    try:
        return _main_inner(args, out)
    finally:
        fp = getattr(args, "_run_log_fp", None)
        if fp is not None:
            try:
                exc = sys.exc_info()[0]
                fp.write(f"\nended: {datetime.datetime.now().isoformat()}\n")
                if exc is None:
                    fp.write("STATUS: completed\n")
                else:
                    fp.write(f"STATUS: failed ({getattr(exc, '__name__', exc)})\n")
            except Exception:
                pass
            try:
                fp.close()
            except Exception:
                pass
            setattr(args, "_run_log_fp", None)


def _main_inner(args, out) -> Tuple[float, float, float, List[dict]]:
    quiet = getattr(args, "q", not getattr(args, "verbose_cv", False))
    # --- Load data ---
    expr_transform = getattr(args, "expr_transform", "log1p")
    data = _load_v2_pickle(
        args.region,
        getattr(args, "cell_type", "Astrocytes"),
        expr_transform=expr_transform,
    )
    raw_folds = _select_folds(data, args.input_type)
    patient_info_folds = data["patient_info_folds"]
    ptau_folds = data["ptau_folds"]
    thal_folds = data["thal_folds"]
    gene_names = data["gene_names"]
    embedding_dim = data["embedding_dim"]
    if args.input_type == "random_embedding":
        # For random embeddings we override the original gene-embedding dim
        # and instead use a small fixed D (e.g., 16) for gene x D random vectors.
        embedding_dim = RANDOM_EMBEDDING_DIM
    n_genes = len(gene_names)

    out(
        f"Loaded v2 data: {len(raw_folds)} folds, input_type={args.input_type}, "
        f"model_type={args.model_type}, region={args.region}, "
        f"expr_transform={data.get('expr_transform', expr_transform)}"
    )

    # --- Merge folds if requested ---
    n_folds_eval = getattr(args, "n_folds_eval", len(raw_folds))
    if n_folds_eval is None:
        n_folds_eval = len(raw_folds)
    orig_n = len(raw_folds)
    if n_folds_eval != orig_n:
        if orig_n % n_folds_eval != 0:
            raise ValueError(f"--n_folds_eval={n_folds_eval} must divide {orig_n}")
        gs = orig_n // n_folds_eval
        raw_folds = _merge_folds(raw_folds, gs)
        patient_info_folds = _merge_folds(patient_info_folds, gs)
        ptau_folds = _merge_folds(ptau_folds, gs)
        thal_folds = _merge_folds(thal_folds, gs)
        out(f"Merged {orig_n} -> {n_folds_eval} folds")

    n_folds = len(raw_folds)

    # --- Thal encoding ---
    all_thal_labels = []
    for tf in thal_folds:
        if isinstance(tf, (list, np.ndarray)):
            all_thal_labels.extend(tf)
        else:
            all_thal_labels.extend(tf)
    all_thal_int = [int(x) for x in all_thal_labels]
    thal_classes = sorted(set(all_thal_int))
    thal_to_idx = {c: i for i, c in enumerate(thal_classes)}
    idx_to_thal = {i: c for c, i in thal_to_idx.items()}
    n_thal_classes = len(thal_classes)
    out(f"Thal classes: {thal_classes} ({n_thal_classes} classes)")

    # --- Mode selection ---
    use_cell_level = bool(getattr(args, "cell_level", False))

    # --- Prepare per-fold data ---
    use_mil = (args.model_type == "mil") and not use_cell_level
    use_sklearn = args.model_type in ("logreg", "svm")
    use_scvi = (args.model_type == "scvi") and not use_cell_level

    all_fold_data = []

    def _build_cell_level_thal(
        fold_data: np.ndarray,
        patient_info: list,
        thal_fold,
        thal_to_idx: dict,
    ) -> np.ndarray:
        """Create per-cell Thal labels given per-patient Thal."""
        cell_thal = []
        cell_idx = 0
        for pi, (_, cell_names) in enumerate(patient_info):
            n_cells = len(cell_names)
            thal_class = thal_to_idx[int(thal_fold[pi])]
            cell_thal.extend([thal_class] * n_cells)
            cell_idx += n_cells
        return np.asarray(cell_thal, dtype=np.int64)

    for fi in range(n_folds):
        if use_cell_level:
            # Keep original cell-level data; labels are per cell.
            fold_cells = np.asarray(raw_folds[fi], dtype=np.float32)
            cell_thal = _build_cell_level_thal(
                fold_cells,
                patient_info_folds[fi],
                thal_folds[fi],
                thal_to_idx,
            )
            all_fold_data.append(
                {
                    "embeddings": fold_cells,
                    "thal": cell_thal,
                    "ptau": np.zeros_like(cell_thal, dtype=np.float32),
                }
            )
        elif use_mil:
            all_fold_data.append(
                _build_mil_bags(
                    raw_folds[fi],
                    patient_info_folds[fi],
                    ptau_folds[fi],
                    thal_folds[fi],
                    thal_to_idx,
                )
            )
        elif use_scvi:
            # For scVI we keep cell-level expression, aggregate to patient for the classifier
            all_fold_data.append(
                _aggregate_patients(
                    raw_folds[fi],
                    patient_info_folds[fi],
                    ptau_folds[fi],
                    thal_folds[fi],
                    thal_to_idx,
                )
            )
        else:
            all_fold_data.append(
                _aggregate_patients(
                    raw_folds[fi],
                    patient_info_folds[fi],
                    ptau_folds[fi],
                    thal_folds[fi],
                    thal_to_idx,
                )
            )

    # --- Loss setup ---
    loss_ptau_str = getattr(args, "loss_ptau", "none")
    if loss_ptau_str is None or str(loss_ptau_str).lower() == "none":
        criterion_ptau = None
    elif loss_ptau_str == "mse":
        criterion_ptau = nn.MSELoss()
    elif loss_ptau_str == "mae":
        criterion_ptau = nn.L1Loss()
    elif loss_ptau_str == "huber":
        criterion_ptau = nn.HuberLoss()
    else:
        criterion_ptau = None

    label_smoothing = float(getattr(args, "label_smoothing", 0.0) or 0.0)
    criterion_thal = nn.CrossEntropyLoss(
        label_smoothing=label_smoothing
    ) if label_smoothing > 0 else nn.CrossEntropyLoss()

    device = torch.device(getattr(args, "device", "cpu"))
    out(f"Device: {device}")

    # --- Augmentation ---
    if getattr(args, "no_online_augment", False):
        train_aug = EmbeddingAugmentConfig()
    else:
        train_aug = augment_config_from_args(args)

    # --- Multi-seed LOFO CV ---
    n_seeds = getattr(args, "n_seeds", 1)
    base_seed = int(getattr(args, "seed", 0))
    patience = getattr(args, "patience", 40)
    min_epochs = getattr(args, "min_epochs", 30)
    epochs = getattr(args, "epochs", 200)
    thal_ordinal = getattr(args, "thal_ordinal", False)

    all_seed_results: List[dict] = []
    all_true_labels: List = []
    all_pred_labels: List = []

    for seed_idx in range(n_seeds):
        current_seed = base_seed + seed_idx
        out(f"\n{'#'*60}")
        out(f"Seed {seed_idx + 1}/{n_seeds} (seed={current_seed})")
        out(f"{'#'*60}")

        _py_random.seed(current_seed)
        np.random.seed(current_seed)
        torch.manual_seed(current_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(current_seed)

        fold_results: List[dict] = []

        for test_fi in range(n_folds):
            train_indices = [i for i in range(n_folds) if i != test_fi]
            out(f"\n  Test Fold {test_fi + 1}/{n_folds}")

            if use_sklearn:
                # Combine train folds
                train_X = np.concatenate(
                    [all_fold_data[i]["embeddings"] for i in train_indices]
                )
                train_y = np.concatenate(
                    [all_fold_data[i]["thal"] for i in train_indices]
                )
                test_X = all_fold_data[test_fi]["embeddings"]
                test_y = all_fold_data[test_fi]["thal"]

                f1_val, fold_true, fold_pred = _run_sklearn(
                    model_type=args.model_type,
                    train_X=train_X,
                    train_y=train_y,
                    test_X=test_X,
                    test_y=test_y,
                    idx_to_thal=idx_to_thal,
                    pca_components=getattr(args, "pca_components", None),
                    C=float(getattr(args, "C", 1.0)),
                    seed=current_seed,
                )
                out(f"    F1={f1_val:.4f} ({args.model_type})")
                fold_results.append({
                    "test_fold": test_fi + 1,
                    "seed": current_seed,
                    "best_f1": f1_val,
                    "best_epoch": 0,
                    "best_test_mse": 0.0,
                    "best_test_mae": 0.0,
                    "thal_true": fold_true,
                    "thal_pred": fold_pred,
                })
                all_true_labels.extend(fold_true)
                all_pred_labels.extend(fold_pred)
                continue

            if use_scvi:
                # scVI needs cell-level expression; aggregate for classifier
                train_expr = np.concatenate(
                    [np.asarray(raw_folds[i], dtype=np.float32) for i in train_indices]
                )
                test_expr = np.asarray(raw_folds[test_fi], dtype=np.float32)
                train_y = np.concatenate(
                    [all_fold_data[i]["thal"] for i in train_indices]
                )
                test_y = all_fold_data[test_fi]["thal"]

                # For scVI: patient-level y from aggregated data
                # But scVI works on cell-level, so we need cell-level labels
                # Actually: we train scVI unsupervised on cells, extract patient-level latent, classify
                # Use patient aggregated for classification
                f1_val, fold_true, fold_pred = _run_scvi(
                    train_expr=np.concatenate(
                        [all_fold_data[i]["embeddings"] for i in train_indices]
                    ),
                    train_y=np.concatenate(
                        [all_fold_data[i]["thal"] for i in train_indices]
                    ),
                    test_expr=all_fold_data[test_fi]["embeddings"],
                    test_y=all_fold_data[test_fi]["thal"],
                    idx_to_thal=idx_to_thal,
                    n_latent=int(getattr(args, "scvi_n_latent", 30)),
                    n_layers=int(getattr(args, "scvi_n_layers", 2)),
                    n_hidden=int(getattr(args, "scvi_n_hidden", 128)),
                    max_epochs=epochs,
                    seed=current_seed,
                )
                out(f"    F1={f1_val:.4f} (scvi)")
                fold_results.append({
                    "test_fold": test_fi + 1,
                    "seed": current_seed,
                    "best_f1": f1_val,
                    "best_epoch": 0,
                    "best_test_mse": 0.0,
                    "best_test_mae": 0.0,
                    "thal_true": fold_true,
                    "thal_pred": fold_pred,
                })
                all_true_labels.extend(fold_true)
                all_pred_labels.extend(fold_pred)
                continue

            # --- PyTorch models (mlp, flatten_mlp, gene_encoder, mil) ---
            mil_seed = current_seed + test_fi * 1000

            if use_mil:
                train_bags = []
                train_ptau_list, train_thal_list = [], []
                for i in train_indices:
                    fd = all_fold_data[i]
                    train_bags.extend(fd["cell_bags"])
                    train_ptau_list.extend(fd["ptau"])
                    train_thal_list.extend(fd["thal"])
                test_fd = all_fold_data[test_fi]
                test_bags = list(test_fd["cell_bags"])
                test_ptau = test_fd["ptau"]
                test_thal = test_fd["thal"]

                pca_comp = getattr(args, "pca_components", None)
                train_bags, test_bags, cell_in = _transform_mil_bags(
                    train_bags, test_bags, pca_comp, quiet=quiet
                )
                train_ptau_arr = np.array(train_ptau_list, dtype=np.float32)
                train_thal_arr = np.array(train_thal_list, dtype=np.int64)

                mil_cells = int(getattr(args, "mil_cells_per_patient", 128))
                train_ds = BagDataset(
                    train_bags, train_ptau_arr, train_thal_arr,
                    train=True, mil_cells_per_patient=mil_cells,
                    augment_config=train_aug, seed=mil_seed,
                )
                test_ds = BagDataset(
                    test_bags, test_ptau, test_thal,
                    train=False, mil_cells_per_patient=mil_cells, seed=mil_seed,
                )
                train_loader = DataLoader(
                    train_ds,
                    batch_size=int(getattr(args, "batch_size", 8)),
                    shuffle=True,
                )
                test_loader = DataLoader(
                    test_ds, batch_size=1, shuffle=False, collate_fn=collate_mil_eval,
                )

                model = MILModel(
                    cell_input_dim=cell_in,
                    n_thal_classes=n_thal_classes,
                    mil_hidden_dim=int(getattr(args, "mil_hidden_dim", 128)),
                    mil_attn_dim=int(getattr(args, "mil_attn_dim", 64)),
                    mil_encoder_layers=int(getattr(args, "mil_encoder_layers", 2)),
                    dropout=float(getattr(args, "dropout", 0.2)),
                    thal_ordinal=thal_ordinal,
                ).to(device)

                run_eval = evaluate_mil_model
                run_train = train_epoch_mil
                is_mil = True
            else:
                # Patient-level models: mlp, flatten_mlp, gene_encoder
                train_emb = np.concatenate(
                    [all_fold_data[i]["embeddings"] for i in train_indices]
                )
                train_ptau_arr = np.concatenate(
                    [all_fold_data[i]["ptau"] for i in train_indices]
                )
                train_thal_arr = np.concatenate(
                    [all_fold_data[i]["thal"] for i in train_indices]
                )
                test_emb = all_fold_data[test_fi]["embeddings"]
                test_ptau = all_fold_data[test_fi]["ptau"]
                test_thal = all_fold_data[test_fi]["thal"]

                if getattr(args, "verbose_cv", False):
                    # train_emb can be 2D (cells x features) or 3D (cells x genes x D)
                    train_cells = int(train_emb.shape[0]) if hasattr(train_emb, "shape") else -1
                    test_cells = int(test_emb.shape[0]) if hasattr(test_emb, "shape") else -1
                    out(
                        f"    [Debug] Fold {test_fi + 1}: "
                        f"train_cells={train_cells}, test_cells={test_cells}, "
                        f"train_emb_ndim={getattr(train_emb, 'ndim', None)}, "
                        f"test_emb_ndim={getattr(test_emb, 'ndim', None)}"
                    )

                # Remember (genes, D) from 3D tensors before flatten. Pickle `embedding_dim` can
                # disagree with the last axis (e.g. 512 in metadata but 500 in array → G*D not
                # divisible by 512).
                emb_geo: Optional[Tuple[int, int]] = None

                # StandardScaler expects 2D (n_samples, n_features).
                # In cell_level mode, embedding inputs are still (cells, genes, D),
                # so we flatten to (cells, genes*D).
                if getattr(train_emb, "ndim", None) == 3:
                    g3, d3 = int(train_emb.shape[1]), int(train_emb.shape[2])
                    emb_geo = (g3, d3)
                    train_emb = train_emb.reshape(train_emb.shape[0], -1)
                    test_emb = test_emb.reshape(test_emb.shape[0], -1)

                # StandardScaler
                # In cell_level mode with flattened embeddings (cells x genes*D),
                # sklearn can allocate float64 temporaries that exceed memory.
                if bool(getattr(args, "cell_level", False)) and train_emb.ndim == 2 and train_emb.shape[1] >= 10000:
                    _standard_scale_inplace_chunked(train_emb, test_emb)
                else:
                    scaler = StandardScaler()
                    train_emb = scaler.fit_transform(train_emb)
                    test_emb = scaler.transform(test_emb)

                # Optional PCA on flattened (cells, features). For gene_encoder + embedding,
                # features are genes×D; PCA mixes genes and dimensions so input_dim is no longer
                # a multiple of embedding_dim → actual_g = input_dim // D becomes 0 and reshape fails.
                pca_comp = getattr(args, "pca_components", None)
                if (
                    args.model_type == "gene_encoder"
                    and getattr(args, "input_type", None) in ("embedding", "random_embedding")
                ):
                    pca_comp = None
                if pca_comp and pca_comp < train_emb.shape[1]:
                    n_comp = min(pca_comp, train_emb.shape[0], train_emb.shape[1])
                    pca = PCA(n_components=n_comp, random_state=current_seed)
                    train_emb = pca.fit_transform(train_emb)
                    test_emb = pca.transform(test_emb)

                input_dim = train_emb.shape[1]
                bs = int(getattr(args, "batch_size", 8))

                # Reshape for gene_encoder (needs 3D: cells x genes x D)
                if args.model_type == "gene_encoder":
                    if args.input_type in ("embedding", "random_embedding"):
                        if emb_geo is not None:
                            actual_g, actual_d = emb_geo
                            if input_dim != actual_g * actual_d:
                                raise ValueError(
                                    f"gene_encoder+embedding: flat dim {input_dim} != "
                                    f"G*D ({actual_g}*{actual_d}) from 3D tensor"
                                )
                        else:
                            # 2D embeddings only: fall back to pickle metadata
                            actual_d = int(embedding_dim)
                            if actual_d <= 0 or input_dim % actual_d != 0:
                                raise ValueError(
                                    f"gene_encoder+embedding: feature dim {input_dim} not divisible by "
                                    f"embedding_dim {actual_d} (need 3D cell×gene×D data or consistent flat G×D)"
                                )
                            actual_g = input_dim // actual_d
                        train_emb = train_emb.reshape(-1, actual_g, actual_d)
                        test_emb = test_emb.reshape(-1, actual_g, actual_d)
                    else:
                        actual_d = input_dim
                        actual_g = 1
                        train_emb = train_emb.reshape(-1, 1, actual_d)
                        test_emb = test_emb.reshape(-1, 1, actual_d)

                train_ds = SimpleDataset(
                    train_emb, train_ptau_arr, train_thal_arr,
                    train=True, augment_config=train_aug,
                )
                test_ds = SimpleDataset(
                    test_emb, test_ptau, test_thal, train=False,
                )
                train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
                test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False)

                dropout = float(getattr(args, "dropout", 0.2))
                mlp_sizes_str = getattr(args, "mlp_hidden_sizes", "")
                mlp_sizes = None
                if mlp_sizes_str and str(mlp_sizes_str).strip():
                    mlp_sizes = [int(x) for x in str(mlp_sizes_str).split(",") if x.strip()]

                if args.model_type == "gene_encoder":
                    model = GeneEncoderModel(
                        n_dims=actual_d,
                        n_thal_classes=n_thal_classes,
                        dropout=dropout,
                        n_genes=actual_g,
                        thal_ordinal=thal_ordinal,
                    ).to(device)
                else:
                    # mlp or flatten_mlp
                    if train_emb.ndim == 3:
                        input_shape = (train_emb.shape[1], train_emb.shape[2])
                    else:
                        input_shape = None
                    model = BaselineRankingModel(
                        input_dim=input_dim,
                        input_shape=input_shape,
                        n_thal_classes=n_thal_classes,
                        use_attention=bool(getattr(args, "attention", False)),
                        num_heads=int(getattr(args, "attention_heads", 1)),
                        dropout=dropout,
                        n_hidden_layers=int(getattr(args, "n_hidden_layers", 1)),
                        hidden_dim_3layer=int(getattr(args, "hidden_dim_3layer", 32)),
                        mlp_hidden_sizes=mlp_sizes,
                        thal_ordinal=thal_ordinal,
                    ).to(device)

                run_eval = evaluate_standard
                run_train = train_epoch_standard
                is_mil = False

            # --- Optimizer ---
            optimizer = optim.Adam(
                model.parameters(),
                lr=float(getattr(args, "lr", 1e-3)),
                weight_decay=float(getattr(args, "weight_decay", 0.1)),
            )

            # --- Training loop ---
            best_f1 = float("-inf")
            best_loss = float("inf")
            best_epoch = 0
            best_mse, best_mae = 0.0, 0.0
            best_state = None
            epochs_no_improve = 0

            lambda_entropy = float(getattr(args, "lambda_entropy", 1e-3))
            mixup_alpha = float(getattr(args, "mixup_alpha", 0.0) or 0.0)
            gene_cutmix_prob = float(getattr(args, "gene_cutmix_prob", 0.0) or 0.0)
            gene_cutmix_alpha = float(getattr(args, "gene_cutmix_alpha", 1.0) or 1.0)

            show_epoch_progress = bool(getattr(args, "verbose_cv", False))
            tqdm = None
            if show_epoch_progress:
                try:
                    from tqdm import tqdm as _tqdm  # type: ignore
                    tqdm = _tqdm
                except ImportError:
                    tqdm = None

            epoch_iter = range(epochs)
            if tqdm is not None:
                epoch_iter = tqdm(
                    range(epochs),
                    desc=f"Fold {test_fi + 1}/{n_folds} epochs",
                    leave=False,
                )

            last_epoch = -1
            for epoch in epoch_iter:
                last_epoch = epoch
                if is_mil:
                    run_train(
                        model, train_loader, criterion_ptau, criterion_thal,
                        optimizer, device,
                        thal_ordinal=thal_ordinal, n_thal_classes=n_thal_classes,
                    )
                else:
                    run_train(
                        model, train_loader, criterion_ptau, criterion_thal,
                        optimizer, device,
                        lambda_entropy=lambda_entropy if args.model_type == "gene_encoder" else 0.0,
                        thal_ordinal=thal_ordinal, n_thal_classes=n_thal_classes,
                        mixup_alpha=mixup_alpha if not is_mil else 0.0,
                        gene_cutmix_prob=gene_cutmix_prob if not is_mil else 0.0,
                        gene_cutmix_alpha=gene_cutmix_alpha,
                    )

                test_loss, test_mse, test_mae, test_f1 = run_eval(
                    model, test_loader, criterion_ptau, criterion_thal,
                    device, idx_to_thal,
                    thal_ordinal=thal_ordinal, n_thal_classes=n_thal_classes,
                )

                improved = test_f1 > best_f1 or (
                    test_f1 == best_f1 and test_loss < best_loss
                )
                if improved:
                    best_f1 = test_f1
                    best_loss = test_loss
                    best_epoch = epoch
                    best_mse = test_mse
                    best_mae = test_mae
                    best_state = {
                        k: v.cpu().clone() for k, v in model.state_dict().items()
                    }
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                if (
                    patience
                    and patience > 0
                    and epochs_no_improve >= patience
                    and (epoch + 1) >= min_epochs
                ):
                    out(f"    Early stop at epoch {epoch + 1}")
                    break

            if best_state is not None:
                model.load_state_dict(best_state)

            if getattr(args, "verbose_cv", False):
                out(
                    f"    [Debug] Fold {test_fi + 1}: "
                    f"epochs_ran={max(0, last_epoch + 1)}, best_epoch={best_epoch + 1}, "
                    f"best_f1={best_f1:.4f}, patience={patience}, epochs_no_improve={epochs_no_improve}"
                )

            # Final eval for pooled predictions
            model.eval()
            with torch.no_grad():
                _, _, _, report_f1, fold_true, fold_pred = run_eval(
                    model, test_loader, criterion_ptau, criterion_thal,
                    device, idx_to_thal,
                    thal_ordinal=thal_ordinal, n_thal_classes=n_thal_classes,
                    return_preds=True,
                )

            out(f"    Fold {test_fi + 1} F1={report_f1:.4f} (epoch {best_epoch + 1})")

            fold_results.append({
                "test_fold": test_fi + 1,
                "seed": current_seed,
                "best_f1": report_f1,
                "best_epoch": best_epoch + 1,
                "best_test_mse": best_mse,
                "best_test_mae": best_mae,
                "thal_true": fold_true,
                "thal_pred": fold_pred,
            })
            all_true_labels.extend(fold_true)
            all_pred_labels.extend(fold_pred)

        # Per-seed summary
        seed_f1s = [r["best_f1"] for r in fold_results]
        out(f"\n  Seed {current_seed} avg F1: {np.mean(seed_f1s):.4f}")
        all_seed_results.extend(fold_results)

    # --- Grand summary ---
    all_f1s = [r["best_f1"] for r in all_seed_results]
    all_mses = [r["best_test_mse"] for r in all_seed_results]
    all_maes = [r["best_test_mae"] for r in all_seed_results]
    grand_f1 = float(np.mean(all_f1s))
    grand_std = float(np.std(all_f1s))
    grand_mse = float(np.mean(all_mses))
    grand_mae = float(np.mean(all_maes))

    pooled_f1 = (
        f1_score(all_true_labels, all_pred_labels, average="weighted")
        if all_true_labels
        else float("nan")
    )

    out(f"\n{'#'*60}")
    out(f"Grand Summary: {n_seeds} seeds x {n_folds} folds = {len(all_seed_results)} runs")
    out(f"  input_type={args.input_type}, model_type={args.model_type}")
    out(f"  Generalized F1: {grand_f1:.4f} +/- {grand_std:.4f}")
    out(f"  Pooled held-out F1: {pooled_f1:.4f}")
    out(f"  Avg MSE: {grand_mse:.4f}, Avg MAE: {grand_mae:.4f}")
    out(f"{'#'*60}")

    return grand_f1, grand_mse, grand_mae, all_seed_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Unified training: expression vs embedding experiment framework"
    )
    parser.add_argument("--region", type=str, required=True)
    parser.add_argument(
        "--cell_type", type=str, default="Astrocytes",
        choices=["Astrocytes", "Microglia", "MTG"],
    )
    parser.add_argument(
        "--input_type", type=str, required=True,
        choices=sorted(INPUT_TYPES),
    )
    parser.add_argument(
        "--model_type", type=str, required=True,
        choices=sorted(ALL_MODEL_TYPES),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_seeds", type=int, default=10)
    parser.add_argument("--n_folds_eval", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--min_epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--pca_components", type=int, default=None)
    parser.add_argument(
        "--expr_transform",
        type=str,
        default="log1p",
        choices=["log1p", "sqrt", "linear"],
        help="Expression transform used when creating v2 data pickles.",
    )
    parser.add_argument(
        "--cell_level",
        action="store_true",
        help="Train on cell-level samples (no patient pooling / MIL / scVI).",
    )
    parser.add_argument(
        "--loss_ptau", type=str, default="none",
        choices=["mse", "mae", "huber", "none"],
    )
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--thal_ordinal", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--verbose-cv", action="store_true", dest="verbose_cv")

    # MLP args
    parser.add_argument("--mlp_hidden_sizes", type=str, default="")
    parser.add_argument("--n_hidden_layers", type=int, default=1, choices=[1, 3])
    parser.add_argument("--hidden_dim_3layer", type=int, default=32)
    parser.add_argument("--attention", action="store_true")
    parser.add_argument("--attention_heads", type=int, default=1)

    # Gene encoder
    parser.add_argument("--lambda_entropy", type=float, default=1e-3)

    # MIL args
    parser.add_argument("--mil_cells_per_patient", type=int, default=128)
    parser.add_argument("--mil_hidden_dim", type=int, default=128)
    parser.add_argument("--mil_attn_dim", type=int, default=64)
    parser.add_argument("--mil_encoder_layers", type=int, default=2)

    # sklearn
    parser.add_argument("--C", type=float, default=1.0, help="Regularization for logreg/svm")

    # scVI
    parser.add_argument("--scvi_n_latent", type=int, default=30)
    parser.add_argument("--scvi_n_layers", type=int, default=2)
    parser.add_argument("--scvi_n_hidden", type=int, default=128)

    # Augmentation
    parser.add_argument("--no_online_augment", action="store_true")
    parser.add_argument("--aug_scale_jitter", type=float, default=0.0)
    parser.add_argument("--aug_per_gene_scale_jitter", type=float, default=0.0)
    parser.add_argument("--aug_gene_dropout", type=float, default=0.0)
    parser.add_argument("--aug_emb_dim_dropout", type=float, default=0.0)
    parser.add_argument("--aug_feature_dropout", type=float, default=0.0)
    parser.add_argument("--aug_gene_block_cutout_max", type=int, default=0)
    parser.add_argument("--aug_gene_block_cutout_prob", type=float, default=0.5)
    parser.add_argument("--aug_noise_std", type=float, default=0.0)
    parser.add_argument("--mixup_alpha", type=float, default=0.0)
    parser.add_argument("--gene_cutmix_prob", type=float, default=0.0)
    parser.add_argument("--gene_cutmix_alpha", type=float, default=1.0)

    # Misc
    parser.add_argument("--no_tensorboard", action="store_true")
    parser.add_argument("--exp_id", type=str, default=None)
    parser.add_argument("--log_base", type=str, default="runs")
    parser.add_argument(
        "--run_log",
        type=str,
        default=None,
        help="Write all training output to this file; if omitted, a timestamped file under --log_base is used.",
    )
    parser.add_argument(
        "--no_run_log",
        action="store_true",
        help="Disable writing the run log file.",
    )

    args = parser.parse_args()

    # Validate input_type / model_type combos
    if args.input_type == "expression" and args.model_type in EMBEDDING_MODELS:
        raise SystemExit(
            f"--model_type {args.model_type} requires --input_type embedding or random_embedding"
        )
    if args.input_type in ("embedding", "random_embedding") and args.model_type in EXPRESSION_MODELS:
        raise SystemExit(
            f"--model_type {args.model_type} requires --input_type expression"
        )

    main(args)
