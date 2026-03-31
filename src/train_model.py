"""Train model with k-fold cross-validation.
Predicts p-tau (regression) and Thal category (classification).
Uses BaselineRankingModel, GeneEncoderModel, or cell-level MILModel (--model mil).
"""

import argparse
import pickle
import warnings
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os
from tqdm import tqdm

from baseline_model import BaselineRankingModel
from gene_encoder_model import GeneEncoderModel, gene_entropy_loss, make_ordinal_targets, ordinal_logits_to_class
from mil_model import MILModel


class _NullSummaryWriter:
    """No-op stand-in for SummaryWriter when TensorBoard is disabled (e.g. autosearch)."""

    def add_scalar(self, *args, **kwargs):
        pass

    def close(self):
        pass


def _parse_mlp_hidden_sizes(s):
    """Parse comma-separated positive ints, e.g. '64,32,16,8,4'. Empty/whitespace -> None."""
    if s is None or not str(s).strip():
        return None
    parts = [p.strip() for p in str(s).split(",") if p.strip()]
    if not parts:
        return None
    out = []
    for p in parts:
        try:
            v = int(p)
        except ValueError as e:
            raise ValueError(f"Invalid mlp_hidden_sizes token {p!r} (expected int)") from e
        if v < 1:
            raise ValueError(f"mlp_hidden_sizes must be positive ints, got {v}")
        out.append(v)
    return out


def _get_gene_order_for_region(region, cell_type="Astrocytes"):
    """Return list of gene names in the same order as the pickle embedding dimensions (axis=1).
    Uses create_data's AnnData + embedding load so no need to re-run create_data."""
    from create_data import create_anndata, load_gene_embeddings
    embeddings_dir = os.path.join("GREmLN", "embeddings")
    if cell_type == "Astrocytes":
        emb_path = os.path.join(embeddings_dir, f"{region}_gene_embeddings.tsv")
    elif cell_type == "Microglia":
        emb_path = os.path.join(embeddings_dir, f"{region}_Microglia_gene_embeddings.tsv")
    else:
        emb_path = os.path.join(embeddings_dir, "MTG_gene_embeddings.tsv")
    if not os.path.isfile(emb_path):
        emb_path = os.path.join(embeddings_dir, "EC_gene_embeddings.tsv")
    adata = create_anndata(region, cell_type)
    embeddings_dict = load_gene_embeddings(emb_path, gene_list=adata.var_names.tolist())
    adata = adata[:, adata.var_names.isin(embeddings_dict.keys())].copy()
    return adata.var_names.tolist()


def _load_ranked_top_genes(ranked_genes_path, top_n):
    """Load top N gene names from ranked_genes.csv (column 'gene', rows in rank order)."""
    df = pd.read_csv(ranked_genes_path)
    if 'gene' not in df.columns:
        raise ValueError(f"ranked_genes CSV must have a 'gene' column. Found: {list(df.columns)}")
    return df['gene'].astype(str).head(top_n).tolist()


@dataclass
class EmbeddingAugmentConfig:
    """Train-time embedding augmentation (applied in __getitem__ when train=True)."""
    scale_jitter_half_width: float = 0.0  # multiply by U(1-a, 1+a); 0 = off
    per_gene_scale_jitter_half_width: float = 0.0  # 3D only: per-gene U(1-b, 1+b)
    gene_dropout_p: float = 0.0  # 3D: drop whole gene rows
    emb_dim_dropout_p: float = 0.0  # 3D: drop embedding dims (all genes); 2D: same on feature axis
    feature_dropout_p: float = 0.0  # 2D only: Bernoulli on flat features
    gene_block_cutout_max: int = 0  # 3D: max genes in contiguous zero block; 0 = off
    gene_block_cutout_prob: float = 0.5  # prob to apply block cutout when max > 0
    noise_std: float = 0.0  # additive N(0, noise_std^2)

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


def _embedding_augment_config_from_args(args) -> EmbeddingAugmentConfig:
    """Build config from argparse / SimpleNamespace."""
    return EmbeddingAugmentConfig(
        scale_jitter_half_width=float(getattr(args, "aug_scale_jitter", 0.0) or 0.0),
        per_gene_scale_jitter_half_width=float(getattr(args, "aug_per_gene_scale_jitter", 0.0) or 0.0),
        gene_dropout_p=float(getattr(args, "aug_gene_dropout", 0.0) or 0.0),
        emb_dim_dropout_p=float(getattr(args, "aug_emb_dim_dropout", 0.0) or 0.0),
        feature_dropout_p=float(getattr(args, "aug_feature_dropout", 0.0) or 0.0),
        gene_block_cutout_max=int(getattr(args, "aug_gene_block_cutout_max", 0) or 0),
        gene_block_cutout_prob=float(getattr(args, "aug_gene_block_cutout_prob", 0.5) or 0.5),
        noise_std=float(getattr(args, "aug_noise_std", 0.0) or 0.0),
    )


def augment_embedding_tensor(x: torch.Tensor, cfg: EmbeddingAugmentConfig) -> torch.Tensor:
    """Apply augmentation in order: multiplicative scales → masks → additive noise.
    x is a single sample: (n_genes, n_dims) or (n_features,).
    """
    if not cfg.any_enabled():
        return x
    x = x.clone()
    dev, dt = x.device, x.dtype

    # Global scale jitter: U(1-a, 1+a)
    if cfg.scale_jitter_half_width > 0:
        a = cfg.scale_jitter_half_width
        u = 1.0 + (torch.rand(1, device=dev, dtype=dt) * (2 * a) - a)
        x.mul_(u)

    if x.dim() == 3:
        raise ValueError("augment_embedding_tensor expects 2D (G,D) or 1D (F,) per sample, got 3D")

    if x.dim() == 2:
        g, d = x.shape[0], x.shape[1]
        if cfg.per_gene_scale_jitter_half_width > 0:
            b = cfg.per_gene_scale_jitter_half_width
            u = 1.0 + (torch.rand(g, 1, device=dev, dtype=dt) * (2 * b) - b)
            x.mul_(u)
        if cfg.gene_dropout_p > 0:
            mask = (torch.rand(g, 1, device=dev, dtype=dt) > cfg.gene_dropout_p).to(dt)
            x.mul_(mask)
        if cfg.emb_dim_dropout_p > 0:
            mask = (torch.rand(1, d, device=dev, dtype=dt) > cfg.emb_dim_dropout_p).to(dt)
            x.mul_(mask)
        if (
            cfg.gene_block_cutout_max > 0
            and g > 0
            and torch.rand(1, device=dev).item() < cfg.gene_block_cutout_prob
        ):
            cut_max = min(cfg.gene_block_cutout_max, g)
            if cut_max >= 1:
                cut_len = int(torch.randint(1, cut_max + 1, (1,), device=dev).item())
                start_max = max(0, g - cut_len)
                start = int(torch.randint(0, start_max + 1, (1,), device=dev).item())
                x[start : start + cut_len, :].zero_()
    elif x.dim() == 1:
        f = x.shape[0]
        if cfg.emb_dim_dropout_p > 0:
            mask = (torch.rand(f, device=dev, dtype=dt) > cfg.emb_dim_dropout_p).to(dt)
            x.mul_(mask)
        if cfg.feature_dropout_p > 0:
            mask = (torch.rand(f, device=dev, dtype=dt) > cfg.feature_dropout_p).to(dt)
            x.mul_(mask)
    else:
        raise ValueError(f"Unexpected embedding rank {x.dim()} for augmentation")

    if cfg.noise_std > 0:
        x.add_(torch.randn_like(x) * cfg.noise_std)
    return x


class EmbeddingDataset(Dataset):
    """Dataset for embeddings with ptau and thal targets; optional train-time augmentation."""

    def __init__(
        self,
        embeddings,
        ptau,
        thal,
        train: bool = False,
        augment_config: Optional[EmbeddingAugmentConfig] = None,
    ):
        self.embeddings = torch.FloatTensor(embeddings)
        self.ptau = torch.FloatTensor(ptau)
        self.thal = torch.LongTensor(thal)
        self.train = bool(train)
        self.augment_config = augment_config or EmbeddingAugmentConfig()

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        x = self.embeddings[idx]
        if self.train and self.augment_config.any_enabled():
            x = augment_embedding_tensor(x, self.augment_config)
        return x, self.ptau[idx], self.thal[idx]


class MILDataset(Dataset):
    """One sample = one patient bag of cells. Train: sample K cells; eval: full bag (variable length, use collate_mil_eval)."""

    def __init__(
        self,
        cell_bags: Sequence[np.ndarray],
        ptau: np.ndarray,
        thal: np.ndarray,
        train: bool,
        mil_cells_per_patient: int,
        augment_config: Optional[EmbeddingAugmentConfig] = None,
        seed: int = 0,
    ):
        self.cell_bags = [np.asarray(b, dtype=np.float32) for b in cell_bags]
        self.ptau = np.asarray(ptau, dtype=np.float32)
        self.thal = np.asarray(thal, dtype=np.int64)
        self.train = bool(train)
        self.K = int(mil_cells_per_patient)
        self.augment_config = augment_config or EmbeddingAugmentConfig()
        self._rng = np.random.RandomState(int(seed))

    def __len__(self) -> int:
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
                augment_embedding_tensor(torch.from_numpy(sampled[i]).float(), self.augment_config)
                for i in range(self.K)
            ]
            x = torch.stack(cells)
        else:
            x = torch.from_numpy(bag)
        return x, self.ptau[idx], self.thal[idx]


def collate_mil_eval_batch(
    batch: List[Tuple[torch.Tensor, float, int]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.LongTensor]:
    bags, ptaus, thals = zip(*batch)
    max_l = max(b.shape[0] for b in bags)
    bsz = len(bags)
    feat_shape = bags[0].shape[1:]
    out = torch.zeros(bsz, max_l, *feat_shape, dtype=torch.float32)
    mask = torch.zeros(bsz, max_l, dtype=torch.bool)
    for i, bag in enumerate(bags):
        L = bag.shape[0]
        out[i, :L] = bag
        mask[i, :L] = True
    ptau_t = torch.tensor(ptaus, dtype=torch.float32)
    thal_t = torch.tensor(thals, dtype=torch.long)
    return out, mask, ptau_t, thal_t


def _fit_pca_mil_subsample(
    train_bags_scaled: List[np.ndarray],
    n_dims: int,
    n_components: int,
    max_fit_rows: int,
    rng: np.random.Generator,
) -> PCA:
    """Fit PCA on up to max_fit_rows training rows (each row length n_dims) without full concat."""
    n_comp = int(min(n_components, n_dims))
    pool: list[np.ndarray] = []
    n_pool = 0
    for bag in train_bags_scaled:
        flat = bag.reshape(-1, n_dims).astype(np.float32, copy=False)
        pool.append(flat)
        n_pool += flat.shape[0]
        if n_pool >= max_fit_rows * 2:
            break
    if not pool:
        raise ValueError("MIL PCA: no training rows")
    big = np.concatenate(pool, axis=0)
    if big.shape[0] > max_fit_rows:
        idx = rng.choice(big.shape[0], size=max_fit_rows, replace=False)
        big = big[idx]
    pca = PCA(n_components=n_comp, svd_solver="randomized", random_state=int(rng.integers(0, 2**31 - 1)))
    pca.fit(big)
    return pca


def _pca_transform_bag_3d(bag: np.ndarray, pca: PCA, pca_components: int, row_chunk: int) -> np.ndarray:
    """Apply PCA along last dim; bag (n, G, D) -> (n, G, pca_components)."""
    n, g, d = bag.shape
    flat = bag.reshape(-1, d).astype(np.float32, copy=False)
    outs: list[np.ndarray] = []
    for i in range(0, flat.shape[0], row_chunk):
        outs.append(pca.transform(flat[i : i + row_chunk]))
    pc = np.concatenate(outs, axis=0).astype(np.float32)
    return pc.reshape(n, g, pca_components)


def _effective_row_chunk(row_chunk: int, n_features: int, max_chunk_bytes: int = 512 * 1024 * 1024) -> int:
    """Cap row_chunk so that one chunk (with sklearn's internal copy) stays within memory budget."""
    bytes_per_row = n_features * 4  # float32
    safe = max(1, max_chunk_bytes // (bytes_per_row * 2))  # ×2 for sklearn internal copy
    return min(row_chunk, safe)


def _mil_scaler_partial_fit_bag(
    scaler: StandardScaler,
    bag: np.ndarray,
    row_chunk: int,
) -> None:
    """partial_fit StandardScaler on one bag without allocating full (n_cells, n_feats) at once."""
    n = bag.shape[0]
    if bag.ndim == 3:
        n_feat = bag.shape[1] * bag.shape[2]
        rc = _effective_row_chunk(row_chunk, n_feat)
        for i in range(0, n, rc):
            slab = bag[i : i + rc]
            X = np.asarray(slab, dtype=np.float32).reshape(slab.shape[0], -1)
            scaler.partial_fit(X)
    elif bag.ndim == 2:
        rc = _effective_row_chunk(row_chunk, bag.shape[1])
        for i in range(0, n, rc):
            X = np.asarray(bag[i : i + rc], dtype=np.float32)
            scaler.partial_fit(X)
    else:
        raise ValueError(f"MIL scaler partial_fit: expected 2D/3D bag, got ndim={bag.ndim}")


def _mil_scaler_transform_bag(
    scaler: StandardScaler,
    bag: np.ndarray,
    row_chunk: int,
    *,
    n_genes: Optional[int] = None,
    n_dims: Optional[int] = None,
) -> np.ndarray:
    """Scale one bag **in-place** using scaler's mean_ / scale_, row-chunk at a time.

    Works directly on the 3D (n,G,D) or 2D (n,F) array — never allocates a
    second full-sized copy.  The bag is converted to float32 and returned.
    """
    if bag.dtype != np.float32:
        bag = bag.astype(np.float32)

    n = bag.shape[0]
    if bag.ndim == 3:
        if n_genes is None or n_dims is None:
            raise ValueError("MIL scaler transform 3D bag requires n_genes and n_dims")
        mean = scaler.mean_.astype(np.float32).reshape(1, n_genes, n_dims)
        scale = scaler.scale_.astype(np.float32).reshape(1, n_genes, n_dims)
        scale = np.where(scale == 0, 1.0, scale)
        rc = _effective_row_chunk(row_chunk, n_genes * n_dims)
        for i in range(0, n, rc):
            end = min(i + rc, n)
            bag[i:end] -= mean
            bag[i:end] /= scale
        return bag
    if bag.ndim == 2:
        feat = bag.shape[1]
        mean = scaler.mean_.astype(np.float32).reshape(1, feat)
        scale = scaler.scale_.astype(np.float32).reshape(1, feat)
        scale = np.where(scale == 0, 1.0, scale)
        rc = _effective_row_chunk(row_chunk, feat)
        for i in range(0, n, rc):
            end = min(i + rc, n)
            bag[i:end] -= mean
            bag[i:end] /= scale
        return bag
    raise ValueError(f"MIL scaler transform: expected 2D/3D bag, got ndim={bag.ndim}")


def _transform_mil_cell_bags(
    train_bags: List[np.ndarray],
    test_bags: List[np.ndarray],
    pca_components: Optional[int],
    out,
    mil_pca_max_fit_rows: int = 100_000,
    transform_row_chunk: int = 8192,
) -> Tuple[List[np.ndarray], List[np.ndarray], int]:
    """Fit StandardScaler (+ optional PCA) on training cells without allocating full (N, F) float64 matrix."""
    if not train_bags:
        raise ValueError("MIL: empty train_bags")
    ref = np.asarray(train_bags[0])
    rng = np.random.default_rng(42)

    if ref.ndim == 3:
        n_genes, n_dims = ref.shape[1], ref.shape[2]
        scaler = StandardScaler()
        for b in train_bags:
            _mil_scaler_partial_fit_bag(scaler, b, transform_row_chunk)
        train_after_scale = [
            _mil_scaler_transform_bag(
                scaler, b, transform_row_chunk, n_genes=n_genes, n_dims=n_dims
            )
            for b in train_bags
        ]
        test_after_scale = [
            _mil_scaler_transform_bag(
                scaler, b, transform_row_chunk, n_genes=n_genes, n_dims=n_dims
            )
            for b in test_bags
        ]
        if pca_components is not None:
            pca = _fit_pca_mil_subsample(
                train_after_scale, n_dims, pca_components, mil_pca_max_fit_rows, rng
            )
            n_out = pca.n_components_
            train_after_scale = [
                _pca_transform_bag_3d(b, pca, n_out, transform_row_chunk) for b in train_after_scale
            ]
            test_after_scale = [
                _pca_transform_bag_3d(b, pca, n_out, transform_row_chunk) for b in test_after_scale
            ]
            cell_in = n_genes * n_out
            out(
                f"Applied MIL: StandardScaler (streamed partial_fit) + PCA -> {n_out} dims/gene "
                f"(PCA fit on <= {mil_pca_max_fit_rows} sampled train rows)"
            )
        else:
            cell_in = n_genes * n_dims
            out("Applied MIL: StandardScaler on flattened cells (streamed partial_fit, no PCA)")
    elif ref.ndim == 2:
        feat_dim = ref.shape[1]
        scaler = StandardScaler()
        for b in train_bags:
            _mil_scaler_partial_fit_bag(scaler, b, transform_row_chunk)
        train_after_scale = [_mil_scaler_transform_bag(scaler, b, transform_row_chunk) for b in train_bags]
        test_after_scale = [_mil_scaler_transform_bag(scaler, b, transform_row_chunk) for b in test_bags]
        if pca_components is not None:
            pca_simple = _fit_pca_mil_subsample(
                train_after_scale, feat_dim, pca_components, mil_pca_max_fit_rows, rng
            )
            n_out = pca_simple.n_components_

            def _tf2(b: np.ndarray) -> np.ndarray:
                b32 = np.asarray(b, dtype=np.float32)
                o: list[np.ndarray] = []
                for i in range(0, b32.shape[0], transform_row_chunk):
                    o.append(pca_simple.transform(b32[i : i + transform_row_chunk]))
                return np.concatenate(o, axis=0).astype(np.float32)

            train_after_scale = [_tf2(b) for b in train_after_scale]
            test_after_scale = [_tf2(b) for b in test_after_scale]
            cell_in = n_out
            out(
                f"Applied MIL: StandardScaler (streamed) + PCA dim {cell_in} "
                f"(PCA fit on <= {mil_pca_max_fit_rows} sampled train rows)"
            )
        else:
            cell_in = feat_dim
            train_after_scale = [b.astype(np.float32, copy=False) for b in train_after_scale]
            test_after_scale = [b.astype(np.float32, copy=False) for b in test_after_scale]
            out(f"Applied MIL: StandardScaler on pooled cell vectors (no PCA), dim {cell_in}")
    else:
        raise ValueError(f"Unexpected MIL bag ndim {ref.ndim}")
    return train_after_scale, test_after_scale, cell_in


def evaluate_mil(
    model: nn.Module,
    dataloader: DataLoader,
    criterion_ptau,
    criterion_thal,
    device,
    idx_to_thal,
    thal_ordinal: bool = False,
    n_thal_classes=None,
    return_preds: bool = False,
):
    """Evaluate MIL model with padded cell bags and attention mask."""
    model.eval()
    total_loss = 0.0
    ptau_preds: list[float] = []
    ptau_targets: list[float] = []
    thal_preds: list[int] = []
    thal_targets: list[int] = []
    criterion_ordinal = nn.BCEWithLogitsLoss() if thal_ordinal else None

    with torch.no_grad():
        for embeddings, mask, ptau, thal in dataloader:
            embeddings = embeddings.to(device)
            mask = mask.to(device)
            ptau = ptau.to(device)
            thal = thal.to(device)

            ptau_pred, thal_pred = model(embeddings, mask=mask)
            if thal_ordinal and n_thal_classes is not None:
                thal_class_preds = ordinal_logits_to_class(thal_pred)
            else:
                thal_class_preds = torch.argmax(thal_pred, dim=1)

            if thal_ordinal and n_thal_classes is not None:
                ordinal_targets = make_ordinal_targets(thal, n_thal_classes)
                loss_thal = criterion_ordinal(thal_pred, ordinal_targets)
            else:
                loss_thal = criterion_thal(thal_pred, thal)
            loss = loss_thal
            if criterion_ptau is not None:
                loss_ptau = criterion_ptau(ptau_pred, ptau)
                loss = loss + loss_ptau
            total_loss += loss.item()

            ptau_preds.extend(ptau_pred.cpu().numpy().tolist())
            ptau_targets.extend(ptau.cpu().numpy().tolist())
            thal_preds.extend(thal_class_preds.cpu().numpy().tolist())
            thal_targets.extend(thal.cpu().numpy().tolist())

    thal_preds_labels = [idx_to_thal[i] for i in thal_preds]
    thal_targets_labels = [idx_to_thal[i] for i in thal_targets]
    mse = mean_squared_error(ptau_targets, ptau_preds)
    mae = mean_absolute_error(ptau_targets, ptau_preds)
    f1 = f1_score(thal_targets_labels, thal_preds_labels, average="weighted")
    if return_preds:
        return total_loss / max(len(dataloader), 1), mse, mae, f1, thal_targets_labels, thal_preds_labels
    return total_loss / max(len(dataloader), 1), mse, mae, f1


def apply_pooling(embeddings, method='mean'):
    """Apply pooling across axis=1 (n_genes) to get (n_samples, n_dims) shape.

    Args:
        embeddings: numpy array of shape (n_samples, n_genes, n_dims)
        method: 'mean' or 'max'

    Returns:
        Pooled embeddings of shape (n_samples, n_dims)
    """
    if method == 'mean':
        return np.mean(embeddings, axis=1)
    elif method == 'max':
        return np.max(embeddings, axis=1)
    else:
        raise ValueError(f"Unknown method: {method}")



def train_epoch(
    model,
    dataloader,
    criterion_ptau,
    criterion_thal,
    optimizer,
    device,
    lambda_entropy=0.0,
    thal_ordinal=False,
    n_thal_classes=None,
    mixup_alpha=0.0,
    gene_cutmix_prob=0.0,
    gene_cutmix_alpha=1.0,
):
    """Train for one epoch. Optional batch-level mixup or gene-axis CutMix (3D only); disabled if thal_ordinal."""
    model.train()
    total_loss = 0
    ptau_losses = []
    thal_losses = []
    criterion_ordinal = nn.BCEWithLogitsLoss() if thal_ordinal else None

    for embeddings, ptau, thal in dataloader:
        embeddings = embeddings.to(device)
        ptau = ptau.to(device)
        thal = thal.to(device)
        B = embeddings.size(0)

        batch_mix = None  # None | {"type": "mixup"|"cutmix", "lam": float, "partner": LongTensor}
        if (not thal_ordinal) and B > 1:
            if (
                gene_cutmix_prob > 0
                and embeddings.dim() == 3
                and float(torch.rand(1, device=device).item()) < gene_cutmix_prob
            ):
                G = embeddings.size(1)
                partner = torch.randperm(B, device=device)
                dist = torch.distributions.Beta(
                    torch.tensor(gene_cutmix_alpha, device=device),
                    torch.tensor(gene_cutmix_alpha, device=device),
                )
                r = float(dist.sample().item())
                cut_len = max(1, min(G, int(round(r * G))))
                start_max = G - cut_len
                start = int(torch.randint(0, start_max + 1, (1,), device=device).item())
                lam = cut_len / float(G)
                x_new = embeddings.clone()
                x_new[:, start : start + cut_len, :] = embeddings[partner, start : start + cut_len, :]
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
        w = out[2] if len(out) == 3 else None

        if thal_ordinal and n_thal_classes is not None:
            ordinal_targets = make_ordinal_targets(thal, n_thal_classes)
            loss_thal = criterion_ordinal(thal_pred, ordinal_targets)
        elif batch_mix is not None:
            partner = batch_mix["partner"]
            lam = batch_mix["lam"]
            if batch_mix["type"] == "mixup":
                loss_thal = lam * criterion_thal(thal_pred, thal) + (1.0 - lam) * criterion_thal(
                    thal_pred, thal[partner]
                )
            else:
                loss_thal = (1.0 - lam) * criterion_thal(thal_pred, thal) + lam * criterion_thal(
                    thal_pred, thal[partner]
                )
        else:
            loss_thal = criterion_thal(thal_pred, thal)

        loss = loss_thal
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
    return total_loss / len(dataloader), np.mean(ptau_losses), np.mean(thal_losses)


def evaluate(
    model,
    dataloader,
    criterion_ptau,
    criterion_thal,
    device,
    idx_to_thal,
    thal_ordinal: bool = False,
    n_thal_classes=None,
    return_preds: bool = False,
):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0
    ptau_preds = []
    ptau_targets = []
    thal_preds = []
    thal_targets = []
    criterion_ordinal = nn.BCEWithLogitsLoss() if thal_ordinal else None

    with torch.no_grad():
        for embeddings, ptau, thal in dataloader:
            embeddings = embeddings.to(device)
            ptau = ptau.to(device)
            thal = thal.to(device)

            out = model(embeddings)
            ptau_pred, thal_pred = out[0], out[1]
            if thal_ordinal and n_thal_classes is not None:
                thal_class_preds = ordinal_logits_to_class(thal_pred)
            else:
                thal_class_preds = torch.argmax(thal_pred, dim=1)

            if thal_ordinal and n_thal_classes is not None:
                ordinal_targets = make_ordinal_targets(thal, n_thal_classes)
                loss_thal = criterion_ordinal(thal_pred, ordinal_targets)
            else:
                loss_thal = criterion_thal(thal_pred, thal)
            loss = loss_thal
            if criterion_ptau is not None:
                loss_ptau = criterion_ptau(ptau_pred, ptau)
                loss = loss + loss_ptau
            total_loss += loss.item()

            ptau_preds.extend(ptau_pred.cpu().numpy())
            ptau_targets.extend(ptau.cpu().numpy())
            thal_preds.extend(thal_class_preds.cpu().numpy())
            thal_targets.extend(thal.cpu().numpy())

    # Convert back to original labels for F1 score using manual mapping
    thal_preds_labels = [idx_to_thal[i] for i in thal_preds]
    thal_targets_labels = [idx_to_thal[i] for i in thal_targets]

    # Metrics
    mse = mean_squared_error(ptau_targets, ptau_preds)
    mae = mean_absolute_error(ptau_targets, ptau_preds)
    f1 = f1_score(thal_targets_labels, thal_preds_labels, average='weighted')

    if return_preds:
        return total_loss / len(dataloader), mse, mae, f1, thal_targets_labels, thal_preds_labels
    return total_loss / len(dataloader), mse, mae, f1


def main(args):
    # Autosearch passes q=True (quiet) or q=False (verbose); CLI uses --verbose-cv for LOFO logs.
    if hasattr(args, "q"):
        quiet = bool(args.q)
    else:
        quiet = not getattr(args, "verbose_cv", False)

    def out(*a, **kw):
        if not quiet:
            print(*a, **kw)

    mlp_hidden_list = _parse_mlp_hidden_sizes(getattr(args, "mlp_hidden_sizes", ""))

    data_suffix = getattr(args, "data_suffix", "")
    data_path = os.path.join('data', 'model_data', f'{args.region}_data_for_model_training{data_suffix}.pkl')
    out(f"Loading data from {data_path}")
    if data_suffix and "augscaling" in str(data_suffix).lower():
        warnings.warn(
            f"Loading augmented pickle ({data_suffix!r}). Offline _augScaling is deprecated; "
            "use base .pkl and train_model --aug_* online augmentation. "
            "Delete stale *_patient_level_pooling_*_augScaling.pkl or use --recompute.",
            UserWarning,
            stacklevel=2,
        )

    with open(data_path, 'rb') as f:
        loaded = pickle.load(f)
    if len(loaded) == 5:
        embeddings_folds, patient_info_folds, ptau_folds, thal_folds, gene_names = loaded
    else:
        embeddings_folds, patient_info_folds, ptau_folds, thal_folds = loaded
        gene_names = None

    out(f"Loaded {len(embeddings_folds)} folds")

    # Optionally merge original folds into fewer evaluation folds (e.g., 4 instead of 8)
    n_folds_eval = getattr(args, "n_folds_eval", len(embeddings_folds))
    orig_n_folds = len(embeddings_folds)
    if n_folds_eval <= 0:
        raise ValueError("--n_folds_eval must be positive")
    if orig_n_folds % n_folds_eval != 0:
        raise ValueError(f"--n_folds_eval={n_folds_eval} must divide number of folds in data ({orig_n_folds})")
    group_size = orig_n_folds // n_folds_eval
    if n_folds_eval != orig_n_folds:
        def _merge_array_folds(folds_seq: List[np.ndarray]) -> List[np.ndarray]:
            merged: List[np.ndarray] = []
            for i in range(0, len(folds_seq), group_size):
                merged.append(np.concatenate(folds_seq[i : i + group_size], axis=0))
            return merged

        def _merge_patient_info_folds(folds_seq):
            merged_pi = []
            for i in range(0, len(folds_seq), group_size):
                combined = {}
                for sub in folds_seq[i : i + group_size]:
                    if hasattr(sub, "items"):
                        combined.update(dict(sub.items()))
                    else:
                        combined.update(dict(sub))
                merged_pi.append(combined)
            return merged_pi

        embeddings_folds = _merge_array_folds(list(embeddings_folds))
        ptau_folds = _merge_array_folds(list(ptau_folds))
        thal_folds = _merge_array_folds(list(thal_folds))
        patient_info_folds = _merge_patient_info_folds(list(patient_info_folds))
        out(f"Using n_folds_eval={n_folds_eval} (merged original {orig_n_folds} folds into groups of {group_size})")

    top_genes = getattr(args, 'top_genes', None)
    ranked_genes_path = getattr(args, 'ranked_genes_path', None)
    if top_genes is not None:
        if gene_names is None:
            cell_type = getattr(args, 'cell_type', 'Astrocytes')
            out(f"Resolving gene order for region={args.region}, cell_type={cell_type} (for top_genes filter)...")
            gene_names = _get_gene_order_for_region(args.region, cell_type=cell_type)
        if not ranked_genes_path:
            ranked_genes_path = os.path.join(
                "ranked_lists", "EC_Astrocytes_embedding_highest_lambda_entropy", "ranked_genes.csv"
            )
        if not os.path.isfile(ranked_genes_path):
            raise FileNotFoundError(f"ranked_genes_path not found: {ranked_genes_path}")
        top_gene_names = _load_ranked_top_genes(ranked_genes_path, top_genes)
        gene_to_idx = {str(g): i for i, g in enumerate(gene_names)}
        indices = [gene_to_idx[g] for g in top_gene_names if g in gene_to_idx]
        missing = [g for g in top_gene_names if g not in gene_to_idx]
        if missing:
            out(f"  Warning: {len(missing)} of top {top_genes} ranked genes not in data; {len(indices)} kept.")
        if not indices:
            raise ValueError("No ranked genes found in data gene list. Check ranked_genes_path and data region/cell_type.")
        for i in range(len(embeddings_folds)):
            embeddings_folds[i] = embeddings_folds[i][:, indices, :]
        out(f"  Filtered to top {len(indices)} genes from {ranked_genes_path}")

    model_kind = getattr(args, "model", "baseline")
    if getattr(args, "gene_encoder", False):
        model_kind = "gene_encoder"
    use_mil = model_kind == "mil"
    use_gene_encoder = model_kind == "gene_encoder"

    if use_gene_encoder and args.pooling is not None:
        raise ValueError("--model gene_encoder (or --gene_encoder) requires pooling=None; omit --pooling")

    device = torch.device(args.device)
    out(f"Using device: {device}")

    if args.loss_ptau is None or (isinstance(args.loss_ptau, str) and args.loss_ptau.lower() == 'none'):
        criterion_ptau = None
    elif use_gene_encoder:
        criterion_ptau = nn.HuberLoss()
    elif args.loss_ptau == 'mse':
        criterion_ptau = nn.MSELoss()
    elif args.loss_ptau == 'mae':
        criterion_ptau = nn.L1Loss()
    elif args.loss_ptau == 'huber':
        criterion_ptau = nn.HuberLoss()

    pool_str = args.pooling if args.pooling is not None else "None"
    top_suffix = f"_top{top_genes}" if top_genes is not None else ""
    data_cache_suffix = data_suffix if data_suffix else ""
    fold_suffix = f"_F{n_folds_eval}"
    if use_mil:
        cache_path = os.path.join(
            "data", "model_data", f"{args.region}_mil_cell_bags_{pool_str}{top_suffix}{fold_suffix}{data_cache_suffix}.pkl"
        )
    else:
        cache_path = os.path.join(
            "data", "model_data", f"{args.region}_patient_level_pooling_{pool_str}{top_suffix}{fold_suffix}{data_cache_suffix}.pkl"
        )
    use_cache = os.path.exists(cache_path) and (not getattr(args, "recompute", False))
    if use_cache and os.path.getmtime(cache_path) >= os.path.getmtime(data_path):
        out(f"Loading precomputed data from cache: {cache_path}")
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
        thal_to_idx = cache["thal_to_idx"]
        idx_to_thal = cache["idx_to_thal"]
        n_thal_classes = cache["n_thal_classes"]
        if use_mil:
            all_fold_mil_data = cache["all_fold_mil_data"]
            n_folds = len(all_fold_mil_data)
            all_fold_patient_data = None
        else:
            all_fold_patient_data = cache["all_fold_patient_data"]
            n_folds = len(all_fold_patient_data)
            all_fold_mil_data = None
        out(f"  {n_folds} folds loaded. Skip preparation.")
    else:
        all_thal_labels = []
        for thal_fold in thal_folds:
            if isinstance(thal_fold, (list, np.ndarray)):
                all_thal_labels.extend(thal_fold)
            else:
                all_thal_labels.extend(thal_fold.values if hasattr(thal_fold, "values") else thal_fold)
        all_thal_labels_int = [int(x) for x in all_thal_labels]
        thal_classes = sorted(set(all_thal_labels_int))
        thal_to_idx = {c: i for i, c in enumerate(thal_classes)}
        idx_to_thal = {i: c for c, i in thal_to_idx.items()}
        n_thal_classes = len(thal_classes)
        n_folds = len(embeddings_folds)
        all_fold_patient_data = []
        all_fold_mil_data: List[dict] = []

        out(f"\n{'='*60}")
        if use_mil:
            out("Pipeline (MIL): cells -> per-patient cell bags (no mean pool)")
        else:
            out("Pipeline: cells -> genes -> patient embedding (mean pool; StandardScaler+PCA in CV)")
        out(f"{'='*60}")

        for fold_idx in range(n_folds):
            out(f"\nPreparing fold {fold_idx + 1}/{n_folds}...")

            fold_embeddings = embeddings_folds[fold_idx]
            fold_ptau = ptau_folds[fold_idx]
            fold_thal = thal_folds[fold_idx]
            fold_patient_info = patient_info_folds[fold_idx]

            if args.pooling is None:
                pooled_embeddings = fold_embeddings
            else:
                pooled_embeddings = apply_pooling(fold_embeddings, method=args.pooling)
            out(f"  Pooled embeddings shape: {pooled_embeddings.shape}")

            patient_ptau = []
            patient_thal = []
            cell_idx = 0
            patient_iter = fold_patient_info.items() if hasattr(fold_patient_info, "items") else fold_patient_info
            if use_mil:
                patient_cell_bags: List[np.ndarray] = []
                for patient_idx, (_, cell_names) in enumerate(patient_iter):
                    n_cells_patient = len(cell_names)
                    patient_cell_embeddings = pooled_embeddings[cell_idx : cell_idx + n_cells_patient]
                    patient_cell_bags.append(np.array(patient_cell_embeddings, dtype=np.float32, copy=True))
                    patient_ptau.append(fold_ptau[patient_idx])
                    patient_thal.append(fold_thal[patient_idx])
                    cell_idx += n_cells_patient
                patient_ptau = np.array(patient_ptau)
                patient_thal = np.array(patient_thal)
                patient_ptau_log = np.log1p(patient_ptau)
                patient_thal_int = [int(t) for t in patient_thal]
                patient_thal_encoded = np.array([thal_to_idx[t] for t in patient_thal_int])
                all_fold_mil_data.append({
                    "cell_bags": patient_cell_bags,
                    "ptau": patient_ptau_log,
                    "thal": patient_thal_encoded,
                })
                out(f"  MIL: {len(patient_cell_bags)} patients, cell counts {[b.shape[0] for b in patient_cell_bags][:5]}...")
            else:
                patient_embeddings = []
                for patient_idx, (_, cell_names) in enumerate(patient_iter):
                    n_cells_patient = len(cell_names)
                    patient_cell_embeddings = pooled_embeddings[cell_idx : cell_idx + n_cells_patient]
                    patient_emb = np.mean(patient_cell_embeddings, axis=0)
                    patient_embeddings.append(patient_emb)
                    patient_ptau.append(fold_ptau[patient_idx])
                    patient_thal.append(fold_thal[patient_idx])
                    cell_idx += n_cells_patient

                patient_embeddings = np.array(patient_embeddings)
                patient_ptau = np.array(patient_ptau)
                patient_thal = np.array(patient_thal)
                out(f"  Patient embeddings shape: {patient_embeddings.shape}")

                patient_ptau_log = np.log1p(patient_ptau)
                patient_thal_int = [int(t) for t in patient_thal]
                patient_thal_encoded = np.array([thal_to_idx[t] for t in patient_thal_int])
                all_fold_patient_data.append({
                    "embeddings": patient_embeddings,
                    "ptau": patient_ptau_log,
                    "thal": patient_thal_encoded,
                })
                out(f"  Number of patients: {len(patient_ptau)}")

        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "wb") as f:
            if use_mil:
                pickle.dump({
                    "all_fold_mil_data": all_fold_mil_data,
                    "thal_to_idx": thal_to_idx,
                    "idx_to_thal": idx_to_thal,
                    "n_thal_classes": n_thal_classes,
                }, f, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                pickle.dump({
                    "all_fold_patient_data": all_fold_patient_data,
                    "thal_to_idx": thal_to_idx,
                    "idx_to_thal": idx_to_thal,
                    "n_thal_classes": n_thal_classes,
                }, f, protocol=pickle.HIGHEST_PROTOCOL)
        out(f"\nSaved cache to {cache_path} (reuse in next runs).")

    label_smoothing = float(getattr(args, "label_smoothing", 0.0) or 0.0)
    if label_smoothing > 0:
        criterion_thal = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    else:
        criterion_thal = nn.CrossEntropyLoss()

    if getattr(args, "no_online_augment", False):
        train_augment_cfg = EmbeddingAugmentConfig()
    else:
        train_augment_cfg = _embedding_augment_config_from_args(args)
    if train_augment_cfg.any_enabled():
        out(
            "Online embedding augmentation (train only): "
            f"scale_jitter={train_augment_cfg.scale_jitter_half_width}, "
            f"per_gene_scale={train_augment_cfg.per_gene_scale_jitter_half_width}, "
            f"gene_dropout={train_augment_cfg.gene_dropout_p}, "
            f"emb_dim_dropout={train_augment_cfg.emb_dim_dropout_p}, "
            f"feature_dropout={train_augment_cfg.feature_dropout_p}, "
            f"gene_block_cutout_max={train_augment_cfg.gene_block_cutout_max}, "
            f"noise_std={train_augment_cfg.noise_std}"
        )
    mixup_alpha = float(getattr(args, "mixup_alpha", 0.0) or 0.0)
    gene_cutmix_prob = float(getattr(args, "gene_cutmix_prob", 0.0) or 0.0)
    gene_cutmix_alpha = float(getattr(args, "gene_cutmix_alpha", 1.0) or 1.0)
    if use_mil and (mixup_alpha > 0 or gene_cutmix_prob > 0):
        out("MIL: batch mixup / gene CutMix disabled (use cell subsampling + augment instead).")
    elif mixup_alpha > 0 or gene_cutmix_prob > 0:
        out(
            f"Batch aug: mixup_alpha={mixup_alpha}, gene_cutmix_prob={gene_cutmix_prob}, "
            f"gene_cutmix_alpha={gene_cutmix_alpha} (off when --thal_ordinal)"
        )

    # Multi-seed + Leave-one-fold-out cross-validation
    n_seeds = getattr(args, 'n_seeds', 1)
    base_seed = int(getattr(args, 'seed', 0))
    all_seed_results: List[dict] = []
    all_true_labels: List[int] = []
    all_pred_labels: List[int] = []

    for seed_idx in range(n_seeds):
        current_seed = base_seed + seed_idx
        if n_seeds > 1:
            out(f"\n{'#'*60}")
            out(f"Seed {seed_idx + 1}/{n_seeds} (seed={current_seed})")
            out(f"{'#'*60}")

        import random as _seed_random
        _seed_random.seed(current_seed)
        np.random.seed(current_seed)
        torch.manual_seed(current_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(current_seed)

        fold_results: List[dict] = []

        for test_fold_idx in range(n_folds):
            out(f"\n{'='*60}")
            out(f"Test Fold {test_fold_idx + 1}/{n_folds}")
            out(f"Training on folds: {[i+1 for i in range(n_folds) if i != test_fold_idx]}")
            out(f"Testing on fold: {test_fold_idx + 1}")
            out(f"{'='*60}")

            train_fold_indices = [i for i in range(n_folds) if i != test_fold_idx]
            mil_seed = current_seed + test_fold_idx * 1000
            if use_mil:
                train_bags_flat: List[np.ndarray] = []
                train_ptau_list: List[float] = []
                train_thal_list: List[int] = []
                for i in train_fold_indices:
                    fd = all_fold_mil_data[i]
                    for j, bag in enumerate(fd["cell_bags"]):
                        train_bags_flat.append(bag)
                        train_ptau_list.append(float(fd["ptau"][j]))
                        train_thal_list.append(int(fd["thal"][j]))
                test_fd = all_fold_mil_data[test_fold_idx]
                test_bags = [np.asarray(b, dtype=np.float32) for b in test_fd["cell_bags"]]
                test_ptau = test_fd["ptau"]
                test_thal = test_fd["thal"]
                mil_pca_components = (
                    args.pca_components
                    if getattr(args, "mil_use_pca", False)
                    else None
                )
                if not getattr(args, "mil_use_pca", False) and args.pca_components is not None:
                    out(
                        "MIL: skipping --pca_components (full scaled embedding dims per gene; "
                        "pass --mil_use_pca to apply PCA to cell bags)"
                    )
                train_bags_t, test_bags_t, cell_input_dim = _transform_mil_cell_bags(
                    train_bags_flat, test_bags, mil_pca_components, out
                )
                train_ptau_arr = np.array(train_ptau_list, dtype=np.float32)
                train_thal_arr = np.array(train_thal_list, dtype=np.int64)
                out(
                    f"Train: {len(train_bags_t)} patients (~{sum(b.shape[0] for b in train_bags_t)} cells), "
                    f"Test: {len(test_bags_t)} patients"
                )
                train_dataset = MILDataset(
                    train_bags_t,
                    train_ptau_arr,
                    train_thal_arr,
                    train=True,
                    mil_cells_per_patient=args.mil_cells_per_patient,
                    augment_config=train_augment_cfg,
                    seed=mil_seed,
                )
                test_dataset = MILDataset(
                    test_bags_t,
                    test_ptau,
                    test_thal,
                    train=False,
                    mil_cells_per_patient=args.mil_cells_per_patient,
                    augment_config=train_augment_cfg,
                    seed=mil_seed,
                )
                train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
                test_loader = DataLoader(
                    test_dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    collate_fn=collate_mil_eval_batch,
                )
                model = MILModel(
                    cell_input_dim=cell_input_dim,
                    n_thal_classes=n_thal_classes,
                    mil_hidden_dim=args.mil_hidden_dim,
                    mil_attn_dim=args.mil_attn_dim,
                    mil_encoder_layers=args.mil_encoder_layers,
                    dropout=args.dropout,
                    thal_ordinal=getattr(args, "thal_ordinal", False),
                ).to(device)
                mixup_train = 0.0
                gene_cutmix_train = 0.0
                run_eval = evaluate_mil
            else:
                train_embeddings = np.concatenate([all_fold_patient_data[i]['embeddings'] for i in train_fold_indices], axis=0)
                train_ptau = np.concatenate([all_fold_patient_data[i]['ptau'] for i in train_fold_indices], axis=0)
                train_thal = np.concatenate([all_fold_patient_data[i]['thal'] for i in train_fold_indices], axis=0)
                test_embeddings = all_fold_patient_data[test_fold_idx]['embeddings']
                test_ptau = all_fold_patient_data[test_fold_idx]['ptau']
                test_thal = all_fold_patient_data[test_fold_idx]['thal']

                n_train, n_test = train_embeddings.shape[0], test_embeddings.shape[0]

                if len(train_embeddings.shape) == 3:
                    n_genes, n_dims = train_embeddings.shape[1], train_embeddings.shape[2]

                    train_2d = train_embeddings.reshape(n_train, -1)
                    test_2d = test_embeddings.reshape(n_test, -1)

                    scaler = StandardScaler()
                    train_scaled = scaler.fit_transform(train_2d)
                    test_scaled = scaler.transform(test_2d)

                    train_scaled = train_scaled.reshape(n_train, n_genes, n_dims)
                    test_scaled = test_scaled.reshape(n_test, n_genes, n_dims)

                    if args.pca_components is not None:
                        train_for_pca = train_scaled.reshape(-1, n_dims)
                        pca = PCA(n_components=args.pca_components)
                        train_pca = pca.fit_transform(train_for_pca)
                        test_pca = pca.transform(test_scaled.reshape(-1, n_dims))
                        train_embeddings = train_pca.reshape(n_train, n_genes, args.pca_components)
                        test_embeddings = test_pca.reshape(n_test, n_genes, args.pca_components)
                        out(
                            f"Applied StandardScaler (fit on train) + PCA 16->{args.pca_components} "
                            "(fit on scaled train), transform train+test"
                        )
                    else:
                        train_embeddings = train_scaled
                        test_embeddings = test_scaled
                        out(f"Applied StandardScaler (fit on train), transform train+test (no PCA, keeping {n_dims} dims)")
                else:
                    scaler = StandardScaler()
                    train_scaled = scaler.fit_transform(train_embeddings)
                    test_scaled = scaler.transform(test_embeddings)

                    if args.pca_components is not None:
                        pca = PCA(n_components=args.pca_components)
                        train_embeddings = pca.fit_transform(train_scaled)
                        test_embeddings = pca.transform(test_scaled)
                        out(
                            f"Applied StandardScaler (fit on train) + PCA 16->{args.pca_components} "
                            "(fit on scaled train), transform train+test"
                        )
                    else:
                        train_embeddings = train_scaled
                        test_embeddings = test_scaled
                        out(f"Applied StandardScaler (fit on train), transform train+test (no PCA, keeping 16 dims)")

                out(f"Train: {len(train_embeddings)} patients, Test: {len(test_embeddings)} patients")

                train_dataset = EmbeddingDataset(
                    train_embeddings, train_ptau, train_thal, train=True, augment_config=train_augment_cfg
                )
                test_dataset = EmbeddingDataset(test_embeddings, test_ptau, test_thal, train=False)

                train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
                test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

                if use_gene_encoder:
                    if len(train_embeddings.shape) != 3:
                        raise ValueError(
                            "--model gene_encoder requires pooling=None (3D input: n_patients, n_genes, n_dims)"
                        )
                    n_genes, n_dims = train_embeddings.shape[1], train_embeddings.shape[2]
                    model = GeneEncoderModel(
                        n_dims=n_dims,
                        n_thal_classes=n_thal_classes,
                        dropout=args.dropout,
                        n_genes=n_genes,
                        thal_ordinal=getattr(args, 'thal_ordinal', False),
                    ).to(device)
                else:
                    if len(train_embeddings.shape) == 3:
                        input_shape = (train_embeddings.shape[1], train_embeddings.shape[2])
                        input_dim = None
                    else:
                        input_shape = None
                        input_dim = train_embeddings.shape[1]

                    model = BaselineRankingModel(
                        input_dim=input_dim,
                        input_shape=input_shape,
                        n_thal_classes=n_thal_classes,
                        use_attention=args.attention,
                        num_heads=args.attention_heads if args.attention else 1,
                        dropout=args.dropout,
                        n_hidden_layers=args.n_hidden_layers,
                        hidden_dim_3layer=args.hidden_dim_3layer,
                        mlp_hidden_sizes=mlp_hidden_list,
                        thal_ordinal=getattr(args, 'thal_ordinal', False),
                    ).to(device)
                mixup_train = mixup_alpha
                gene_cutmix_train = gene_cutmix_prob
                run_eval = evaluate

            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

            if use_mil:
                model_suffix = '_mil'
            elif use_gene_encoder:
                model_suffix = '_geneenc'
            else:
                model_suffix = ''
            attention_suffix = '_attn' if args.attention else ''
            dropout_str = f'_drop{args.dropout}'
            wd_str = f'_wd{args.weight_decay}'
            if mlp_hidden_list:
                layers_str = '_MLP' + '_'.join(str(h) for h in mlp_hidden_list)
            elif args.n_hidden_layers == 3:
                layers_str = f'_L{args.n_hidden_layers}'
            else:
                layers_str = ''
            exp_id_str = f"_{args.exp_id}" if getattr(args, 'exp_id', None) else ''
            log_base = getattr(args, 'log_base', 'runs')
            no_tensorboard = getattr(args, "no_tensorboard", False)
            if no_tensorboard:
                writer = _NullSummaryWriter()
            else:
                log_dir = os.path.join(
                    log_base,
                    f"{args.region}_{args.pooling}_{args.loss_ptau}{model_suffix}{attention_suffix}"
                    f"{dropout_str}{wd_str}{layers_str}{exp_id_str}_seed{current_seed}_testfold{test_fold_idx + 1}",
                )
                if os.path.exists(log_dir):
                    try:
                        import shutil
                        shutil.rmtree(log_dir)
                    except (PermissionError, OSError) as e:
                        print(f"  Warning: Could not delete {log_dir}: {e}. TensorBoard may have it open. Continuing anyway.")
                writer = SummaryWriter(log_dir)

            best_test_loss = float('inf')
            best_test_f1 = float('-inf')
            best_epoch = 0
            best_test_mse = None
            best_test_mae = None
            best_state = None
            patience = getattr(args, 'patience', 20)
            min_epochs = getattr(args, 'min_epochs', 30)
            if patience is not None and patience <= 0:
                patience = None
            epochs_without_f1_improvement = 0

            thal_ordinal = getattr(args, 'thal_ordinal', False)
            for epoch in range(args.epochs):
                train_loss, train_ptau_loss, train_thal_loss = train_epoch(
                    model, train_loader, criterion_ptau, criterion_thal, optimizer, device,
                    lambda_entropy=args.lambda_entropy if use_gene_encoder else 0.0,
                    thal_ordinal=thal_ordinal, n_thal_classes=n_thal_classes,
                    mixup_alpha=mixup_train,
                    gene_cutmix_prob=gene_cutmix_train,
                    gene_cutmix_alpha=gene_cutmix_alpha,
                )
                test_loss, test_mse, test_mae, test_f1 = run_eval(
                    model, test_loader, criterion_ptau, criterion_thal, device, idx_to_thal,
                    thal_ordinal=thal_ordinal, n_thal_classes=n_thal_classes,
                )

                writer.add_scalar('Loss/Train', train_loss, epoch)
                writer.add_scalar('Loss/Test', test_loss, epoch)
                writer.add_scalar('Loss/Train_Ptau', train_ptau_loss, epoch)
                writer.add_scalar('Loss/Train_Thal', train_thal_loss, epoch)
                writer.add_scalar('Metrics/Test_MSE', test_mse, epoch)
                writer.add_scalar('Metrics/Test_MAE', test_mae, epoch)
                writer.add_scalar('Metrics/Test_F1', test_f1, epoch)

                improved = False
                if test_f1 > best_test_f1 or (test_f1 == best_test_f1 and test_loss < best_test_loss):
                    improved = True
                if improved:
                    best_test_f1 = test_f1
                    best_test_loss = test_loss
                    best_epoch = epoch
                    best_test_mse = test_mse
                    best_test_mae = test_mae
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    epochs_without_f1_improvement = 0
                else:
                    epochs_without_f1_improvement += 1

                if (epoch + 1) % 10 == 0 and getattr(args, "verbose", True) and not quiet:
                    out(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss:.4f}, "
                        f"Test Loss: {test_loss:.4f}, Test F1: {test_f1:.4f}, Test MAE: {test_mae:.4f}")

                if (
                    patience is not None
                    and epochs_without_f1_improvement >= patience
                    and (epoch + 1) >= min_epochs
                ):
                    if getattr(args, "verbose", True) and not quiet:
                        out(f"  Early stopping at epoch {epoch+1} (no F1 improvement for {patience} epochs, min_epochs={min_epochs}).")
                    break

            if best_state is not None:
                model.load_state_dict(best_state)

            # Final evaluation with best checkpoint to collect pooled predictions
            model.eval()
            with torch.no_grad():
                _, report_mse, report_mae, report_f1, fold_true, fold_pred = run_eval(
                    model,
                    test_loader,
                    criterion_ptau,
                    criterion_thal,
                    device,
                    idx_to_thal,
                    thal_ordinal=thal_ordinal,
                    n_thal_classes=n_thal_classes,
                    return_preds=True,
                )
            writer.close()

            if best_state is not None and getattr(args, 'save_best_model', False):
                save_dir = getattr(args, 'save_dir', 'models')
                os.makedirs(save_dir, exist_ok=True)
                save_name = f'{args.region}_testfold{test_fold_idx+1}_seed{current_seed}{model_suffix}{attention_suffix}{exp_id_str}.pt'
                save_path = os.path.join(save_dir, save_name)
                torch.save(best_state, save_path)
                out(f"  Saved best model to {save_path}")

            report_epoch = best_epoch + 1
            out(f"\nTest Fold {test_fold_idx + 1} Best F1: {report_f1:.4f} (epoch {report_epoch})")
            out(f"  MSE: {report_mse:.4f}, MAE: {report_mae:.4f}")

            fold_results.append({
                'test_fold': test_fold_idx + 1,
                'seed': current_seed,
                'best_f1': report_f1,
                'best_epoch': report_epoch,
                'best_test_mse': report_mse,
                'best_test_mae': report_mae,
                'thal_true': fold_true,
                'thal_pred': fold_pred,
            })

            all_true_labels.extend(fold_true)
            all_pred_labels.extend(fold_pred)

        # --- Per-seed summary ---
        seed_f1s = [r['best_f1'] for r in fold_results]
        seed_avg_f1 = np.mean(seed_f1s)
        out(f"\n{'='*60}")
        out(f"Seed {current_seed} LOFO Summary: per-fold best F1 = {[f'{f:.4f}' for f in seed_f1s]}")
        out(f"Seed {current_seed} Average F1: {seed_avg_f1:.4f}")
        out(f"{'='*60}")
        all_seed_results.extend(fold_results)

    # --- Grand summary across all seeds x folds ---
    out(f"\n{'#'*60}")
    out(f"Grand Summary ({n_seeds} seed(s) x {n_folds} folds = {len(all_seed_results)} runs)")
    out(f"{'#'*60}")

    all_f1s = [r['best_f1'] for r in all_seed_results]
    all_mses = [r['best_test_mse'] for r in all_seed_results]
    all_maes = [r['best_test_mae'] for r in all_seed_results]

    grand_avg_f1 = np.mean(all_f1s)
    grand_std_f1 = np.std(all_f1s)
    grand_avg_mse = np.mean(all_mses)
    grand_avg_mae = np.mean(all_maes)

    per_seed_avg_f1s = []
    for si in range(n_seeds):
        s = base_seed + si
        sf1 = [r['best_f1'] for r in all_seed_results if r['seed'] == s]
        per_seed_avg_f1s.append(np.mean(sf1))
    seed_of_f1_std = np.std(per_seed_avg_f1s) if len(per_seed_avg_f1s) > 1 else 0.0

    if n_seeds > 1:
        out(f"Per-seed avg F1: {[f'{f:.4f}' for f in per_seed_avg_f1s]}")
        out(f"Std of per-seed avg F1: {seed_of_f1_std:.4f}")

    for result in all_seed_results:
        out(f"\033[92mSeed {result['seed']} Fold {result['test_fold']}: F1={result['best_f1']:.4f}, "
            f"MSE={result['best_test_mse']:.4f}, MAE={result['best_test_mae']:.4f}\033[0m")

    from sklearn.metrics import f1_score as _f1_score_sklearn
    pooled_f1 = _f1_score_sklearn(all_true_labels, all_pred_labels, average="weighted") if all_true_labels else float("nan")

    out(f"\nGeneralized F1 (mean over {len(all_f1s)} seed-fold runs): {grand_avg_f1:.4f} +/- {grand_std_f1:.4f}")
    out(f"Pooled held-out F1 across all seeds x folds: {pooled_f1:.4f}")
    out(f"Average MSE: {grand_avg_mse:.4f}")
    out(f"Average MAE: {grand_avg_mae:.4f}")

    return grand_avg_f1, grand_avg_mse, grand_avg_mae, all_seed_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train model (baseline, gene encoder, or cell-level MIL)')
    parser.add_argument('--region', type=str, required=True, help='Region name (e.g., EC, ITG, PFC)')
    parser.add_argument(
        '--model',
        type=str,
        choices=['baseline', 'gene_encoder', 'mil'],
        default='baseline',
        help='baseline: patient-level MLP; gene_encoder: learnable gene pooling; mil: cell-level MIL bags',
    )
    parser.add_argument('--seed', type=int, default=0, help='RNG seed (MIL subsampling + reproducibility aids)')
    parser.add_argument('--pooling', type=str, choices=['mean', 'max'], default=None,
                       help='Pooling method: mean or max (default: None)')
    parser.add_argument('--loss_ptau', type=str, default='mse',
                       choices=['mse', 'mae', 'huber', 'none'],
                       help='Loss function for ptau prediction (none to omit ptau loss)')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--attention', action='store_true',
                       help='Use self-attention layer in the model')
    parser.add_argument('--attention_heads', type=int, default=1,
                       help='Number of attention heads (default: 1)')
    parser.add_argument('--dropout', type=float, default=0.2,
                       help='Dropout rate (default: 0.2)')
    parser.add_argument('--weight_decay', type=float, default=1e-1,
                       help='Weight decay for Adam optimizer (default: 1e-1)')
    parser.add_argument('--pca_components', type=int, default=None,
                       help='Number of PCA components; if not set, keep 16 dims (no PCA)')
    parser.add_argument(
        '--gene_encoder',
        action='store_true',
        help='Deprecated: use --model gene_encoder. GeneEncoderModel; requires pooling=None',
    )
    parser.add_argument('--mil_cells_per_patient', type=int, default=128,
                       help='MIL: number of cells sampled per patient per step (train only)')
    parser.add_argument('--mil_hidden_dim', type=int, default=128, help='MIL: cell encoder output width')
    parser.add_argument('--mil_attn_dim', type=int, default=64, help='MIL: attention bottleneck dim')
    parser.add_argument('--mil_encoder_layers', type=int, default=2,
                       help='MIL: number of linear layers in cell encoder (>=1)')
    parser.add_argument(
        '--mil_use_pca',
        action='store_true',
        help='MIL: after scaling, apply --pca_components per gene row (default: off; train on full D e.g. 16)',
    )
    parser.add_argument('--lambda_entropy', type=float, default=1e-3,
                       help='Gene entropy regularization weight (default: 1e-3, gene_encoder only)')
    parser.add_argument('--thal_ordinal', action='store_true',
                       help='Treat Thal as ordinal: use K-1 cumulative logits and BCE loss (default: nominal K-way CrossEntropy)')
    parser.add_argument('--recompute', action='store_true',
                       help='Force recompute patient-level data and overwrite cache (default: use cache if present and newer than data)')
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='Print per-epoch progress (default: True). Set to False when running many experiments.')
    parser.add_argument('--no_verbose', action='store_false', dest='verbose',
                       help='Disable per-epoch printing (quiet mode for batch runs)')
    parser.add_argument(
        '--verbose-cv',
        action='store_true',
        dest='verbose_cv',
        help='Print LOFO fold / data-load / summary stdout (default: quiet for less noise)',
    )
    parser.add_argument('--patience', type=int, default=40,
                       help='Early stopping: stop if held-out F1 does not improve for this many epochs (default: 40). Set to 0 to disable.')
    parser.add_argument('--min_epochs', type=int, default=30,
                       help='Minimum epochs before early stopping can trigger (default: 30).')
    parser.add_argument('--n_seeds', type=int, default=10,
                       help='Number of random seeds to run (default: 10). Each seed repeats full LOFO CV.')
    parser.add_argument('--n_folds_eval', type=int, default=8,
                       help='Number of CV folds to use by merging original folds (e.g., 4 or 8). Default: use all folds from data.')
    parser.add_argument('--n_hidden_layers', type=int, default=1, choices=[1, 3],
                       help='Baseline MLP: 1 (minimal) or 3 (original) when --mlp_hidden_sizes not set (default: 1).')
    parser.add_argument('--hidden_dim_3layer', type=int, default=32,
                       help='Hidden size when n_hidden_layers=3 (default: 32).')
    parser.add_argument(
        '--mlp_hidden_sizes',
        type=str,
        default='',
        help='Comma-separated baseline MLP widths, e.g. 64,32,16,8,4. When set, overrides --n_hidden_layers layout.',
    )
    parser.add_argument('--best_f1', action='store_true',
                       help='(Deprecated, now always on) Checkpoint selection uses best held-out F1.')
    parser.add_argument('--top_genes', type=int, default=None,
                       help='Use only top N genes from ranked_genes.csv; filters at train time (default: all genes).')
    parser.add_argument('--ranked_genes_path', type=str, default=None,
                       help='Path to ranked_genes.csv. Default when --top_genes set: ranked_lists/EC_Astrocytes_embedding_highest_lambda_entropy/ranked_genes.csv')
    parser.add_argument('--cell_type', type=str, default='Astrocytes',
                       help='Cell type for resolving gene order when --top_genes is used (default: Astrocytes).')
    parser.add_argument('--save_best_model', action='store_true',
                       help='Save best model state_dict to --save_dir (default: models/) per test fold.')
    parser.add_argument('--save_dir', type=str, default='models',
                       help='Directory to save best model when --save_best_model (default: models).')
    parser.add_argument('--data_suffix', type=str, default='',
                       help='Suffix for model-data .pkl (avoid _augScaling; use online --aug_* instead).')
    parser.add_argument('--no_online_augment', action='store_true',
                       help='Disable all EmbeddingDataset online augmentations (--aug_* ignored).')
    parser.add_argument('--aug_scale_jitter', type=float, default=0.0,
                       help='Global multiplicative jitter U(1-a,1+a) on embeddings (train only). 0=off.')
    parser.add_argument('--aug_per_gene_scale_jitter', type=float, default=0.0,
                       help='Per-gene U(1-b,1+b) scale on 3D (genes,dims) input. 0=off.')
    parser.add_argument('--aug_gene_dropout', type=float, default=0.0,
                       help='Probability to zero each gene row (3D input). 0=off.')
    parser.add_argument('--aug_emb_dim_dropout', type=float, default=0.0,
                       help='Probability to zero each embedding dim (all genes). 0=off.')
    parser.add_argument('--aug_feature_dropout', type=float, default=0.0,
                       help='Probability to zero each flat feature (2D pooled input). 0=off.')
    parser.add_argument('--aug_gene_block_cutout_max', type=int, default=0,
                       help='Max length of contiguous gene zero block (3D). 0=off.')
    parser.add_argument('--aug_gene_block_cutout_prob', type=float, default=0.5,
                       help='Probability to apply gene block cutout when max>0.')
    parser.add_argument('--aug_noise_std', type=float, default=0.0,
                       help='Additive Gaussian noise std (train only). 0=off.')
    parser.add_argument('--mixup_alpha', type=float, default=0.0,
                       help='Beta(alpha,alpha) for batch mixup on embeddings; 0=off. Off if --thal_ordinal.')
    parser.add_argument('--gene_cutmix_prob', type=float, default=0.0,
                       help='Per-batch prob of gene-axis CutMix (3D only). 0=off. Off if --thal_ordinal.')
    parser.add_argument('--gene_cutmix_alpha', type=float, default=1.0,
                       help='Beta(alpha,alpha) for random cut length fraction (gene CutMix).')
    parser.add_argument('--label_smoothing', type=float, default=0.0,
                       help='CrossEntropy label smoothing for Thal (0=off).')
    parser.add_argument('--no_tensorboard', action='store_true',
                       help='Disable TensorBoard logging (no SummaryWriter / event files).')
    args = parser.parse_args()
    if args.gene_encoder:
        args.model = 'gene_encoder'
    if args.model == 'mil' and args.mil_encoder_layers < 1:
        raise SystemExit('--mil_encoder_layers must be >= 1')

    main(args)
