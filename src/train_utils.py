"""Shared training utilities for train_model.py and train_model_v2.py.

Re-exports classes/functions from train_model.py and adds new helpers for
the v2 experiment framework (expression-only models, random embeddings, etc.).
"""

from __future__ import annotations

import math
import os
import pickle
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from collections import Counter


def label_dist(labels):
    """Compact class-count dict {class: n}, sorted by class."""
    c = Counter(int(x) for x in labels)
    return {k: c[k] for k in sorted(c)}


def full_classification_metrics(true_labels, pred_labels, probs=None, positive_index=-1):
    """f1 (weighted + macro), accuracy, and binary AUROC from a set of predictions.

    probs: per-sample full softmax vectors (list of length-K). AUROC is only defined for a
    binary target (uses probs[:, positive_index]); otherwise returned as NaN.
    """
    yt = [int(x) for x in true_labels]
    yp = [int(x) for x in pred_labels]
    m = {
        "f1_weighted": float(f1_score(yt, yp, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(yt, yp, average="macro", zero_division=0)),
        "acc": float(accuracy_score(yt, yp)),
        "auroc": float("nan"),
        "n": len(yt),
    }
    if probs is not None and len(set(yt)) == 2:
        try:
            P = np.asarray(probs, dtype=float)
            m["auroc"] = float(roc_auc_score(yt, P[:, positive_index]))
        except Exception:
            pass
    return m
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from baseline_model import BaselineRankingModel
from gene_encoder_model import (
    GeneEncoderModel,
    gene_entropy_loss,
    make_ordinal_targets,
    ordinal_logits_to_class,
)
from mil_model import MILModel

# ---------------------------------------------------------------------------
# Re-export from train_model (avoid circular import by lazy import)
# ---------------------------------------------------------------------------

def _lazy_train_model():
    import train_model as _tm
    return _tm


def get_embedding_augment_config_from_args(args):
    _tm = _lazy_train_model()
    return _tm._embedding_augment_config_from_args(args)


def get_embedding_dataset(*a, **kw):
    _tm = _lazy_train_model()
    return _tm.EmbeddingDataset(*a, **kw)


def get_mil_dataset(*a, **kw):
    _tm = _lazy_train_model()
    return _tm.MILDataset(*a, **kw)


def get_collate_mil_eval_batch():
    _tm = _lazy_train_model()
    return _tm.collate_mil_eval_batch


# ---------------------------------------------------------------------------
# Augmentation config (standalone, no dependency on train_model)
# ---------------------------------------------------------------------------

class EmbeddingAugmentConfig:
    """Train-time embedding augmentation config."""

    def __init__(
        self,
        scale_jitter_half_width: float = 0.0,
        per_gene_scale_jitter_half_width: float = 0.0,
        gene_dropout_p: float = 0.0,
        emb_dim_dropout_p: float = 0.0,
        feature_dropout_p: float = 0.0,
        gene_block_cutout_max: int = 0,
        gene_block_cutout_prob: float = 0.5,
        noise_std: float = 0.0,
    ):
        self.scale_jitter_half_width = scale_jitter_half_width
        self.per_gene_scale_jitter_half_width = per_gene_scale_jitter_half_width
        self.gene_dropout_p = gene_dropout_p
        self.emb_dim_dropout_p = emb_dim_dropout_p
        self.feature_dropout_p = feature_dropout_p
        self.gene_block_cutout_max = gene_block_cutout_max
        self.gene_block_cutout_prob = gene_block_cutout_prob
        self.noise_std = noise_std

    def any_enabled(self) -> bool:
        return (
            self.scale_jitter_half_width > 0
            or self.per_gene_scale_jitter_half_width > 0
            or self.gene_dropout_p > 0
            or self.emb_dim_dropout_p > 0
            or self.feature_dropout_p > 0
            or self.gene_block_cutout_max > 0
            or self.noise_std > 0
        )


def augment_config_from_args(args) -> EmbeddingAugmentConfig:
    return EmbeddingAugmentConfig(
        scale_jitter_half_width=float(getattr(args, "aug_scale_jitter", 0.0) or 0.0),
        per_gene_scale_jitter_half_width=float(
            getattr(args, "aug_per_gene_scale_jitter", 0.0) or 0.0
        ),
        gene_dropout_p=float(getattr(args, "aug_gene_dropout", 0.0) or 0.0),
        emb_dim_dropout_p=float(getattr(args, "aug_emb_dim_dropout", 0.0) or 0.0),
        feature_dropout_p=float(getattr(args, "aug_feature_dropout", 0.0) or 0.0),
        gene_block_cutout_max=int(getattr(args, "aug_gene_block_cutout_max", 0) or 0),
        gene_block_cutout_prob=float(
            getattr(args, "aug_gene_block_cutout_prob", 0.5) or 0.5
        ),
        noise_std=float(getattr(args, "aug_noise_std", 0.0) or 0.0),
    )


# ---------------------------------------------------------------------------
# Datasets (standalone versions)
# ---------------------------------------------------------------------------

def _augment_tensor(x: torch.Tensor, cfg: EmbeddingAugmentConfig) -> torch.Tensor:
    """Apply augmentation to a single sample tensor."""
    if not cfg.any_enabled():
        return x
    x = x.clone()
    dev, dt = x.device, x.dtype

    if cfg.scale_jitter_half_width > 0:
        a = cfg.scale_jitter_half_width
        u = 1.0 + (torch.rand(1, device=dev, dtype=dt) * (2 * a) - a)
        x.mul_(u)

    if x.dim() == 2:
        g, d = x.shape
        if cfg.per_gene_scale_jitter_half_width > 0:
            b = cfg.per_gene_scale_jitter_half_width
            u = 1.0 + (torch.rand(g, 1, device=dev, dtype=dt) * (2 * b) - b)
            x.mul_(u)
        if cfg.gene_dropout_p > 0:
            mask = (torch.rand(g, 1, device=dev, dtype=dt) > cfg.gene_dropout_p).to(dt)
            x.mul_(mask)
        if cfg.emb_dim_dropout_p > 0:
            mask = (torch.rand(1, d, device=dev, dtype=dt) > cfg.emb_dim_dropout_p).to(
                dt
            )
            x.mul_(mask)
        if (
            cfg.gene_block_cutout_max > 0
            and g > 0
            and torch.rand(1).item() < cfg.gene_block_cutout_prob
        ):
            cut_max = min(cfg.gene_block_cutout_max, g)
            if cut_max >= 1:
                cut_len = int(torch.randint(1, cut_max + 1, (1,)).item())
                start = int(torch.randint(0, max(1, g - cut_len + 1), (1,)).item())
                x[start : start + cut_len, :].zero_()
    elif x.dim() == 1:
        f = x.shape[0]
        if cfg.emb_dim_dropout_p > 0:
            mask = (torch.rand(f, device=dev, dtype=dt) > cfg.emb_dim_dropout_p).to(dt)
            x.mul_(mask)
        if cfg.feature_dropout_p > 0:
            mask = (torch.rand(f, device=dev, dtype=dt) > cfg.feature_dropout_p).to(dt)
            x.mul_(mask)

    if cfg.noise_std > 0:
        x.add_(torch.randn_like(x) * cfg.noise_std)
    return x


class SimpleDataset(Dataset):
    """Generic dataset for patient-level (or cell-level) samples."""

    def __init__(
        self,
        embeddings: np.ndarray,
        ptau: np.ndarray,
        thal: np.ndarray,
        train: bool = False,
        augment_config: Optional[EmbeddingAugmentConfig] = None,
        braak: Optional[np.ndarray] = None,
    ):
        self.embeddings = torch.FloatTensor(embeddings)
        self.ptau = torch.FloatTensor(ptau)
        self.thal = torch.LongTensor(thal)
        # -1 sentinel when braak is unused (single-task); ignored in loss/eval.
        self.braak = (torch.LongTensor(braak) if braak is not None
                      else torch.full((len(self.thal),), -1, dtype=torch.long))
        self.train = train
        self.augment_config = augment_config or EmbeddingAugmentConfig()

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        x = self.embeddings[idx]
        if self.train and self.augment_config.any_enabled():
            x = _augment_tensor(x, self.augment_config)
        return x, self.ptau[idx], self.thal[idx], self.braak[idx]


class BagDataset(Dataset):
    """MIL bag dataset: one sample = one patient's cell bag."""

    def __init__(
        self,
        cell_bags: Sequence[np.ndarray],
        ptau: np.ndarray,
        thal: np.ndarray,
        train: bool,
        mil_cells_per_patient: int,
        augment_config: Optional[EmbeddingAugmentConfig] = None,
        seed: int = 0,
        braak: Optional[np.ndarray] = None,
    ):
        self.cell_bags = [np.asarray(b, dtype=np.float32) for b in cell_bags]
        self.ptau = np.asarray(ptau, dtype=np.float32)
        self.thal = np.asarray(thal, dtype=np.int64)
        # -1 sentinel when braak is not used (single-task); ignored in loss/eval.
        self.braak = (
            np.asarray(braak, dtype=np.int64)
            if braak is not None
            else np.full(len(self.thal), -1, dtype=np.int64)
        )
        self.train = train
        self.K = int(mil_cells_per_patient)
        self.augment_config = augment_config or EmbeddingAugmentConfig()
        self._rng = np.random.RandomState(int(seed))

    def __len__(self):
        return len(self.cell_bags)

    def __getitem__(self, idx):
        bag = self.cell_bags[idx]
        n = bag.shape[0]
        if self.train:
            if n >= self.K:
                inds = self._rng.choice(n, size=self.K, replace=False)
            else:
                inds = self._rng.choice(n, size=self.K, replace=True)
            sampled = bag[inds]
            cells = [
                _augment_tensor(
                    torch.from_numpy(sampled[i]).float(), self.augment_config
                )
                for i in range(self.K)
            ]
            x = torch.stack(cells)
        else:
            x = torch.from_numpy(bag)
        return x, self.ptau[idx], self.thal[idx], self.braak[idx]


def collate_mil_eval(
    batch: List[Tuple[torch.Tensor, float, int, int]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.LongTensor, torch.LongTensor]:
    """Pad variable-length bags for eval."""
    bags, ptaus, thals, braaks = zip(*batch)
    max_l = max(b.shape[0] for b in bags)
    bsz = len(bags)
    feat_shape = bags[0].shape[1:]
    out = torch.zeros(bsz, max_l, *feat_shape, dtype=torch.float32)
    mask = torch.zeros(bsz, max_l, dtype=torch.bool)
    for i, bag in enumerate(bags):
        L = bag.shape[0]
        out[i, :L] = bag
        mask[i, :L] = True
    return (out, mask, torch.tensor(ptaus, dtype=torch.float32),
            torch.tensor(thals, dtype=torch.long), torch.tensor(braaks, dtype=torch.long))


# ---------------------------------------------------------------------------
# Null TensorBoard writer
# ---------------------------------------------------------------------------

class NullWriter:
    def add_scalar(self, *a, **kw):
        pass

    def flush(self):
        pass

    def close(self):
        pass


# ---------------------------------------------------------------------------
# Scaler / PCA helpers (memory-efficient, from train_model.py patterns)
# ---------------------------------------------------------------------------

def effective_row_chunk(
    row_chunk: int, n_features: int, max_bytes: int = 512 * 1024 * 1024
) -> int:
    bytes_per_row = n_features * 4
    safe = max(1, max_bytes // (bytes_per_row * 2))
    return min(row_chunk, safe)


def scaler_partial_fit_bag(
    scaler: StandardScaler, bag: np.ndarray, row_chunk: int = 8192
) -> None:
    n = bag.shape[0]
    if bag.ndim == 3:
        nf = bag.shape[1] * bag.shape[2]
        rc = effective_row_chunk(row_chunk, nf)
        for i in range(0, n, rc):
            slab = bag[i : i + rc]
            scaler.partial_fit(
                np.asarray(slab, dtype=np.float32).reshape(slab.shape[0], -1)
            )
    elif bag.ndim == 2:
        rc = effective_row_chunk(row_chunk, bag.shape[1])
        for i in range(0, n, rc):
            scaler.partial_fit(np.asarray(bag[i : i + rc], dtype=np.float32))
    else:
        raise ValueError(f"Expected 2D/3D bag, got ndim={bag.ndim}")


def scaler_transform_bag_inplace(
    scaler: StandardScaler,
    bag: np.ndarray,
    row_chunk: int = 8192,
    n_genes: Optional[int] = None,
    n_dims: Optional[int] = None,
) -> np.ndarray:
    """Scale bag in-place (avoids doubling memory)."""
    if bag.dtype != np.float32:
        bag = bag.astype(np.float32)
    n = bag.shape[0]
    if bag.ndim == 3:
        if n_genes is None or n_dims is None:
            n_genes, n_dims = bag.shape[1], bag.shape[2]
        mean = scaler.mean_.astype(np.float32).reshape(1, n_genes, n_dims)
        scale = scaler.scale_.astype(np.float32).reshape(1, n_genes, n_dims)
        scale = np.where(scale == 0, 1.0, scale)
        rc = effective_row_chunk(row_chunk, n_genes * n_dims)
        for i in range(0, n, rc):
            bag[i : i + rc] -= mean
            bag[i : i + rc] /= scale
    elif bag.ndim == 2:
        feat = bag.shape[1]
        mean = scaler.mean_.astype(np.float32).reshape(1, feat)
        scale = scaler.scale_.astype(np.float32).reshape(1, feat)
        scale = np.where(scale == 0, 1.0, scale)
        rc = effective_row_chunk(row_chunk, feat)
        for i in range(0, n, rc):
            bag[i : i + rc] -= mean
            bag[i : i + rc] /= scale
    return bag


# ---------------------------------------------------------------------------
# Random embedding generation
# ---------------------------------------------------------------------------

def generate_random_embeddings(
    expression_matrix: np.ndarray,
    n_genes: int,
    embedding_dim: int,
    seed: int = 12345,
) -> np.ndarray:
    """Create (n_cells, n_genes, D) from expression * random gene vectors.

    Uses a fixed seed so the same random embeddings are used across seeds/folds.
    """
    rng = np.random.RandomState(seed)
    random_emb = rng.randn(n_genes, embedding_dim).astype(np.float32)
    # expr[:, :, None] * emb[None, :, :]
    result = expression_matrix[:, :, None] * random_emb[None, :, :]
    return result.astype(np.float16)


# ---------------------------------------------------------------------------
# Training loop helpers
# ---------------------------------------------------------------------------

def train_epoch_standard(
    model: nn.Module,
    dataloader: DataLoader,
    criterion_ptau,
    criterion_thal: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    lambda_entropy: float = 0.0,
    thal_ordinal: bool = False,
    n_thal_classes: Optional[int] = None,
    mixup_alpha: float = 0.0,
    gene_cutmix_prob: float = 0.0,
    gene_cutmix_alpha: float = 1.0,
    criterion_braak: Optional[nn.Module] = None,
    braak_weight: float = 1.0,
) -> Tuple[float, float, float]:
    """Standard (non-MIL) training epoch. Returns (total_loss, ptau_loss, thal_loss).

    If criterion_braak is given, adds braak_weight * CE(braak) for multi-task; the
    model's 3rd output is then Braak logits (not gene-encoder attention weights).
    """
    model.train()
    total_loss = 0.0
    ptau_losses, thal_losses = [], []
    criterion_ordinal = nn.BCEWithLogitsLoss() if thal_ordinal else None

    for embeddings, ptau, thal, braak in dataloader:
        embeddings = embeddings.to(device)
        ptau = ptau.to(device)
        thal = thal.to(device)
        braak = braak.to(device)
        B = embeddings.size(0)

        batch_mix = None
        if (not thal_ordinal) and B > 1:
            if (
                gene_cutmix_prob > 0
                and embeddings.dim() == 3
                and float(torch.rand(1).item()) < gene_cutmix_prob
            ):
                G = embeddings.size(1)
                partner = torch.randperm(B, device=device)
                dist = torch.distributions.Beta(
                    torch.tensor(gene_cutmix_alpha, device=device),
                    torch.tensor(gene_cutmix_alpha, device=device),
                )
                r = float(dist.sample().item())
                cut_len = max(1, min(G, int(round(r * G))))
                start = int(
                    torch.randint(0, max(1, G - cut_len + 1), (1,), device=device).item()
                )
                lam = cut_len / float(G)
                x_new = embeddings.clone()
                x_new[:, start : start + cut_len, :] = embeddings[
                    partner, start : start + cut_len, :
                ]
                embeddings = x_new
                ptau = (1.0 - lam) * ptau + lam * ptau[partner]
                batch_mix = {"type": "cutmix", "lam": lam, "partner": partner}
            elif mixup_alpha > 0:
                dist = torch.distributions.Beta(
                    torch.tensor(mixup_alpha, device=device),
                    torch.tensor(mixup_alpha, device=device),
                )
                lam = float(dist.sample().item())
                idx = torch.randperm(B, device=device)
                embeddings = lam * embeddings + (1.0 - lam) * embeddings[idx]
                ptau = lam * ptau + (1.0 - lam) * ptau[idx]
                batch_mix = {"type": "mixup", "lam": lam, "partner": idx}

        optimizer.zero_grad()
        out = model(embeddings)
        ptau_pred, thal_pred = out[0], out[1]
        extra = out[2] if len(out) == 3 else None
        # 3rd output is Braak logits under multi-task, else gene-encoder attn weights.
        braak_pred = extra if criterion_braak is not None else None
        w = None if criterion_braak is not None else extra

        if thal_ordinal and n_thal_classes is not None:
            loss_thal = criterion_ordinal(
                thal_pred, make_ordinal_targets(thal, n_thal_classes)
            )
        elif batch_mix is not None:
            partner = batch_mix["partner"]
            lam = batch_mix["lam"]
            if batch_mix["type"] == "mixup":
                loss_thal = lam * criterion_thal(thal_pred, thal) + (
                    1.0 - lam
                ) * criterion_thal(thal_pred, thal[partner])
            else:
                loss_thal = (1.0 - lam) * criterion_thal(thal_pred, thal) + lam * criterion_thal(
                    thal_pred, thal[partner]
                )
        else:
            loss_thal = criterion_thal(thal_pred, thal)

        loss = loss_thal
        if criterion_braak is not None and braak_pred is not None:
            loss = loss + braak_weight * criterion_braak(braak_pred, braak)
        if criterion_ptau is not None:
            loss_ptau = criterion_ptau(ptau_pred, ptau)
            loss = loss + loss_ptau
            ptau_losses.append(loss_ptau.item())
        else:
            ptau_losses.append(0.0)
        if lambda_entropy > 0 and w is not None:
            loss = loss + lambda_entropy * gene_entropy_loss(w)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        thal_losses.append(loss_thal.item())

    return (
        total_loss / max(len(dataloader), 1),
        float(np.mean(ptau_losses)) if ptau_losses else 0.0,
        float(np.mean(thal_losses)) if thal_losses else 0.0,
    )


def evaluate_standard(
    model: nn.Module,
    dataloader: DataLoader,
    criterion_ptau,
    criterion_thal: nn.Module,
    device: torch.device,
    idx_to_thal: dict,
    *,
    thal_ordinal: bool = False,
    n_thal_classes: Optional[int] = None,
    return_preds: bool = False,
    criterion_braak: Optional[nn.Module] = None,
    idx_to_braak: Optional[dict] = None,
    braak_weight: float = 1.0,
    f1_average: str = "weighted",
):
    """Standard (non-MIL) evaluation. When idx_to_braak is given (multi-task), also
    returns the weighted Braak F1 (and, with return_preds, its true/pred labels)."""
    model.eval()
    total_loss = 0.0
    ptau_preds, ptau_targets = [], []
    thal_preds, thal_targets = [], []
    braak_preds, braak_targets = [], []
    thal_probs, braak_probs = [], []   # full per-sample softmax (for AUROC + confidence)
    criterion_ordinal = nn.BCEWithLogitsLoss() if thal_ordinal else None
    multitask = idx_to_braak is not None

    with torch.no_grad():
        for embeddings, ptau, thal, braak in dataloader:
            embeddings = embeddings.to(device)
            ptau = ptau.to(device)
            thal = thal.to(device)
            braak = braak.to(device)
            out = model(embeddings)
            ptau_pred, thal_pred = out[0], out[1]
            braak_pred = out[2] if (multitask and len(out) == 3) else None

            if thal_ordinal and n_thal_classes:
                thal_cls = ordinal_logits_to_class(thal_pred)
            else:
                thal_cls = torch.argmax(thal_pred, dim=1)

            if thal_ordinal and n_thal_classes:
                loss_thal = criterion_ordinal(
                    thal_pred, make_ordinal_targets(thal, n_thal_classes)
                )
            else:
                loss_thal = criterion_thal(thal_pred, thal)
            loss = loss_thal
            if criterion_ptau is not None:
                loss = loss + criterion_ptau(ptau_pred, ptau)
            total_loss += loss.item()

            # Full per-sample softmax (for confidence/entropy + AUROC). Ordinal -> a
            # pseudo two-class [1-p, p] from the single cumulative logit.
            if thal_ordinal:
                _p1 = torch.sigmoid(thal_pred[:, -1])
                thal_prob = torch.stack([1 - _p1, _p1], dim=1)
            else:
                thal_prob = torch.softmax(thal_pred, dim=1)

            ptau_preds.extend(ptau_pred.cpu().numpy())
            ptau_targets.extend(ptau.cpu().numpy())
            thal_preds.extend(thal_cls.cpu().numpy())
            thal_targets.extend(thal.cpu().numpy())
            thal_probs.extend(thal_prob.cpu().numpy())
            if braak_pred is not None:
                braak_preds.extend(torch.argmax(braak_pred, dim=1).cpu().numpy())
                braak_targets.extend(braak.cpu().numpy())
                braak_probs.extend(torch.softmax(braak_pred, dim=1).cpu().numpy())

    thal_preds_labels = [idx_to_thal[int(i)] for i in thal_preds]
    thal_targets_labels = [idx_to_thal[int(i)] for i in thal_targets]
    y_t = np.asarray(ptau_targets, dtype=np.float64)
    y_p = np.asarray(ptau_preds, dtype=np.float64)
    if not np.isfinite(y_t).all() or not np.isfinite(y_p).all():
        y_t = np.nan_to_num(y_t, nan=0.0, posinf=0.0, neginf=0.0)
        y_p = np.nan_to_num(y_p, nan=0.0, posinf=0.0, neginf=0.0)
    mse = mean_squared_error(y_t, y_p)
    mae = mean_absolute_error(y_t, y_p)
    f1 = f1_score(thal_targets_labels, thal_preds_labels, average=f1_average)

    braak_f1 = braak_true_labels = braak_pred_labels = None
    if multitask and braak_preds:
        braak_true_labels = [idx_to_braak[int(i)] for i in braak_targets]
        braak_pred_labels = [idx_to_braak[int(i)] for i in braak_preds]
        braak_f1 = f1_score(braak_true_labels, braak_pred_labels, average=f1_average)

    if return_preds:
        return (
            total_loss / max(len(dataloader), 1),
            mse,
            mae,
            f1,
            thal_targets_labels,
            thal_preds_labels,
            braak_f1,
            braak_true_labels,
            braak_pred_labels,
            [list(map(float, p)) for p in thal_probs],
            ([list(map(float, p)) for p in braak_probs] if multitask else None),
        )
    return total_loss / max(len(dataloader), 1), mse, mae, f1, braak_f1


def evaluate_mil_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion_ptau,
    criterion_thal: nn.Module,
    device: torch.device,
    idx_to_thal: dict,
    *,
    thal_ordinal: bool = False,
    n_thal_classes: Optional[int] = None,
    return_preds: bool = False,
    criterion_braak: Optional[nn.Module] = None,
    idx_to_braak: Optional[dict] = None,
    braak_weight: float = 1.0,
    f1_average: str = "weighted",
):
    """MIL evaluation (padded bags with mask).

    When idx_to_braak is provided (multi-task), also returns the weighted Braak F1
    as an extra trailing value.
    """
    model.eval()
    total_loss = 0.0
    ptau_preds, ptau_targets = [], []
    thal_preds_list, thal_targets_list = [], []
    braak_preds_list, braak_targets_list = [], []
    thal_probs, braak_probs = [], []   # full per-sample softmax (for AUROC + confidence)
    criterion_ordinal = nn.BCEWithLogitsLoss() if thal_ordinal else None
    multitask = idx_to_braak is not None

    with torch.no_grad():
        for batch in dataloader:
            bags, mask, ptau, thal, braak = batch
            bags = bags.to(device)
            mask = mask.to(device)
            ptau = ptau.to(device)
            thal = thal.to(device)
            braak = braak.to(device)

            ptau_pred, thal_pred, braak_pred = model(bags, mask=mask)

            if thal_ordinal and n_thal_classes:
                thal_cls = ordinal_logits_to_class(thal_pred)
            else:
                thal_cls = torch.argmax(thal_pred, dim=1)

            if thal_ordinal and n_thal_classes:
                loss_thal = criterion_ordinal(
                    thal_pred, make_ordinal_targets(thal, n_thal_classes)
                )
            else:
                loss_thal = criterion_thal(thal_pred, thal)
            loss = loss_thal
            if multitask and criterion_braak is not None and braak_pred is not None:
                loss = loss + braak_weight * criterion_braak(braak_pred, braak)
            if criterion_ptau is not None:
                loss = loss + criterion_ptau(ptau_pred, ptau)
            total_loss += loss.item()

            if thal_ordinal:
                _p1 = torch.sigmoid(thal_pred[:, -1])
                thal_prob = torch.stack([1 - _p1, _p1], dim=1)
            else:
                thal_prob = torch.softmax(thal_pred, dim=1)

            ptau_preds.extend(ptau_pred.cpu().numpy())
            ptau_targets.extend(ptau.cpu().numpy())
            thal_preds_list.extend(thal_cls.cpu().numpy())
            thal_targets_list.extend(thal.cpu().numpy())
            thal_probs.extend(thal_prob.cpu().numpy())
            if multitask and braak_pred is not None:
                braak_preds_list.extend(torch.argmax(braak_pred, dim=1).cpu().numpy())
                braak_targets_list.extend(braak.cpu().numpy())
                braak_probs.extend(torch.softmax(braak_pred, dim=1).cpu().numpy())

    thal_preds_labels = [idx_to_thal[int(i)] for i in thal_preds_list]
    thal_targets_labels = [idx_to_thal[int(i)] for i in thal_targets_list]
    y_t = np.asarray(ptau_targets, dtype=np.float64)
    y_p = np.asarray(ptau_preds, dtype=np.float64)
    if not np.isfinite(y_t).all() or not np.isfinite(y_p).all():
        y_t = np.nan_to_num(y_t, nan=0.0, posinf=0.0, neginf=0.0)
        y_p = np.nan_to_num(y_p, nan=0.0, posinf=0.0, neginf=0.0)
    mse = mean_squared_error(y_t, y_p)
    mae = mean_absolute_error(y_t, y_p)
    f1 = f1_score(thal_targets_labels, thal_preds_labels, average=f1_average)

    braak_f1 = None
    braak_true_labels = braak_pred_labels = None
    if multitask and braak_preds_list:
        braak_true_labels = [idx_to_braak[int(i)] for i in braak_targets_list]
        braak_pred_labels = [idx_to_braak[int(i)] for i in braak_preds_list]
        braak_f1 = f1_score(braak_true_labels, braak_pred_labels, average=f1_average)

    if return_preds:
        return (
            total_loss / max(len(dataloader), 1),
            mse,
            mae,
            f1,
            thal_targets_labels,
            thal_preds_labels,
            braak_f1,
            braak_true_labels,
            braak_pred_labels,
            [list(map(float, p)) for p in thal_probs],
            ([list(map(float, p)) for p in braak_probs] if multitask else None),
        )
    return total_loss / max(len(dataloader), 1), mse, mae, f1, braak_f1


def train_epoch_mil(
    model: nn.Module,
    dataloader: DataLoader,
    criterion_ptau,
    criterion_thal: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    thal_ordinal: bool = False,
    n_thal_classes: Optional[int] = None,
    criterion_braak: Optional[nn.Module] = None,
    braak_weight: float = 1.0,
) -> Tuple[float, float, float]:
    """MIL training epoch (fixed-size bags, no mask needed).

    If criterion_braak is given, adds braak_weight * CE(braak) for multi-task
    (Thal + Braak) training.
    """
    model.train()
    total_loss = 0.0
    ptau_losses, thal_losses = [], []
    criterion_ordinal = nn.BCEWithLogitsLoss() if thal_ordinal else None

    for bags, ptau, thal, braak in dataloader:
        bags = bags.to(device)
        ptau = ptau.to(device)
        thal = thal.to(device)
        braak = braak.to(device)

        optimizer.zero_grad()
        ptau_pred, thal_pred, braak_pred = model(bags)

        if thal_ordinal and n_thal_classes:
            loss_thal = criterion_ordinal(
                thal_pred, make_ordinal_targets(thal, n_thal_classes)
            )
        else:
            loss_thal = criterion_thal(thal_pred, thal)

        loss = loss_thal
        if criterion_braak is not None and braak_pred is not None:
            loss = loss + braak_weight * criterion_braak(braak_pred, braak)
        if criterion_ptau is not None:
            loss_ptau = criterion_ptau(ptau_pred, ptau)
            loss = loss + loss_ptau
            ptau_losses.append(loss_ptau.item())
        else:
            ptau_losses.append(0.0)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        thal_losses.append(loss_thal.item())

    return (
        total_loss / max(len(dataloader), 1),
        float(np.mean(ptau_losses)) if ptau_losses else 0.0,
        float(np.mean(thal_losses)) if thal_losses else 0.0,
    )
