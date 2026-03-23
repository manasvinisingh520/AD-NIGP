"""Train model with k-fold cross-validation.
Predicts p-tau (regression) and Thal category (classification).
Uses BaselineRankingModel or GeneEncoderModel.
"""

import argparse
import pickle
import warnings
from dataclasses import dataclass
from typing import Optional

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


def evaluate(model, dataloader, criterion_ptau, criterion_thal, device, idx_to_thal, thal_ordinal=False, n_thal_classes=None):
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

    return total_loss / len(dataloader), mse, mae, f1


def main(args):
    mlp_hidden_list = _parse_mlp_hidden_sizes(getattr(args, "mlp_hidden_sizes", ""))

    data_suffix = getattr(args, "data_suffix", "")
    data_path = os.path.join('data', 'model_data', f'{args.region}_data_for_model_training{data_suffix}.pkl')
    print(f"Loading data from {data_path}")
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

    print(f"Loaded {len(embeddings_folds)} folds")

    top_genes = getattr(args, 'top_genes', None)
    ranked_genes_path = getattr(args, 'ranked_genes_path', None)
    if top_genes is not None:
        if gene_names is None:
            cell_type = getattr(args, 'cell_type', 'Astrocytes')
            print(f"Resolving gene order for region={args.region}, cell_type={cell_type} (for top_genes filter)...")
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
            print(f"  Warning: {len(missing)} of top {top_genes} ranked genes not in data; {len(indices)} kept.")
        if not indices:
            raise ValueError("No ranked genes found in data gene list. Check ranked_genes_path and data region/cell_type.")
        for i in range(len(embeddings_folds)):
            embeddings_folds[i] = embeddings_folds[i][:, indices, :]
        print(f"  Filtered to top {len(indices)} genes from {ranked_genes_path}")

    if args.gene_encoder and args.pooling is not None:
        raise ValueError("--gene_encoder requires pooling=None (no gene pooling); omit --pooling")

    device = torch.device(args.device)
    print(f"Using device: {device}")

    if args.loss_ptau is None or (isinstance(args.loss_ptau, str) and args.loss_ptau.lower() == 'none'):
        criterion_ptau = None
    elif args.gene_encoder:
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
    cache_path = os.path.join("data", "model_data", f"{args.region}_patient_level_pooling_{pool_str}{top_suffix}{data_cache_suffix}.pkl")
    use_cache = os.path.exists(cache_path) and (not getattr(args, "recompute", False))
    if use_cache and os.path.getmtime(cache_path) >= os.path.getmtime(data_path):
        print(f"Loading precomputed patient-level data from cache: {cache_path}")
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
        all_fold_patient_data = cache["all_fold_patient_data"]
        thal_to_idx = cache["thal_to_idx"]
        idx_to_thal = cache["idx_to_thal"]
        n_thal_classes = cache["n_thal_classes"]
        n_folds = len(all_fold_patient_data)
        print(f"  {n_folds} folds loaded. Skip preparation.")
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

        print(f"\n{'='*60}")
        print("Pipeline: cells -> genes -> patient embedding (StandardScaler+PCA in CV)")
        print(f"{'='*60}")

        for fold_idx in range(n_folds):
            print(f"\nPreparing fold {fold_idx + 1}/{n_folds}...")

            fold_embeddings = embeddings_folds[fold_idx]
            fold_ptau = ptau_folds[fold_idx]
            fold_thal = thal_folds[fold_idx]
            fold_patient_info = patient_info_folds[fold_idx]

            if args.pooling is None:
                pooled_embeddings = fold_embeddings
            else:
                pooled_embeddings = apply_pooling(fold_embeddings, method=args.pooling)
            print(f"  Pooled embeddings shape: {pooled_embeddings.shape}")

            patient_embeddings = []
            patient_ptau = []
            patient_thal = []
            cell_idx = 0
            # `patient_info` is either a dict (patient_id -> cell_names) or an
            # ordered list of (patient_id, cell_names) tuples (allows duplicate IDs
            # when augmentations create multiple samples per patient).
            patient_iter = fold_patient_info.items() if hasattr(fold_patient_info, "items") else fold_patient_info
            for patient_idx, (patient_name, cell_names) in enumerate(patient_iter):
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
            print(f"  Patient embeddings shape: {patient_embeddings.shape}")

            patient_ptau_log = np.log1p(patient_ptau)
            patient_thal_int = [int(t) for t in patient_thal]
            patient_thal_encoded = np.array([thal_to_idx[t] for t in patient_thal_int])
            all_fold_patient_data.append({
                "embeddings": patient_embeddings,
                "ptau": patient_ptau_log,
                "thal": patient_thal_encoded,
            })
            print(f"  Number of patients: {len(patient_ptau)}")

        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump({
                "all_fold_patient_data": all_fold_patient_data,
                "thal_to_idx": thal_to_idx,
                "idx_to_thal": idx_to_thal,
                "n_thal_classes": n_thal_classes,
            }, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"\nSaved patient-level cache to {cache_path} (reuse in next runs).")

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
        print(
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
    if mixup_alpha > 0 or gene_cutmix_prob > 0:
        print(
            f"Batch aug: mixup_alpha={mixup_alpha}, gene_cutmix_prob={gene_cutmix_prob}, "
            f"gene_cutmix_alpha={gene_cutmix_alpha} (off when --thal_ordinal)"
        )

    # Leave-one-fold-out cross-validation
    fold_results = []

    for test_fold_idx in range(n_folds):
        print(f"\n{'='*60}")
        print(f"Test Fold {test_fold_idx + 1}/{n_folds}")
        print(f"Training on folds: {[i+1 for i in range(n_folds) if i != test_fold_idx]}")
        print(f"Testing on fold: {test_fold_idx + 1}")
        print(f"{'='*60}")

        train_fold_indices = [i for i in range(n_folds) if i != test_fold_idx]
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
                test_for_pca = test_scaled.reshape(-1, n_dims)
                pca = PCA(n_components=args.pca_components)
                train_pca = pca.fit_transform(train_for_pca)
                test_pca = pca.transform(test_for_pca)
                train_embeddings = train_pca.reshape(n_train, n_genes, args.pca_components)
                test_embeddings = test_pca.reshape(n_test, n_genes, args.pca_components)
                print(f"Applied StandardScaler (fit on train) + PCA 16->{args.pca_components} (fit on scaled train), transform train+test")
            else:
                train_embeddings = train_scaled
                test_embeddings = test_scaled
                print(f"Applied StandardScaler (fit on train), transform train+test (no PCA, keeping {n_dims} dims)")
        else:
            scaler = StandardScaler()
            train_scaled = scaler.fit_transform(train_embeddings)
            test_scaled = scaler.transform(test_embeddings)

            if args.pca_components is not None:
                pca = PCA(n_components=args.pca_components)
                train_embeddings = pca.fit_transform(train_scaled)
                test_embeddings = pca.transform(test_scaled)
                print(f"Applied StandardScaler (fit on train) + PCA 16->{args.pca_components} (fit on scaled train), transform train+test")
            else:
                train_embeddings = train_scaled
                test_embeddings = test_scaled
                print(f"Applied StandardScaler (fit on train), transform train+test (no PCA, keeping 16 dims)")

        print(f"Train: {len(train_embeddings)} patients, Test: {len(test_embeddings)} patients")

        train_dataset = EmbeddingDataset(
            train_embeddings, train_ptau, train_thal, train=True, augment_config=train_augment_cfg
        )
        test_dataset = EmbeddingDataset(test_embeddings, test_ptau, test_thal, train=False)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        if args.gene_encoder:
            if len(train_embeddings.shape) != 3:
                raise ValueError("--gene_encoder requires pooling=None (3D input: n_patients, n_genes, n_dims)")
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

        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        model_suffix = '_geneenc' if args.gene_encoder else ''
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
        log_dir = os.path.join(log_base, f'{args.region}_{args.pooling}_{args.loss_ptau}{model_suffix}{attention_suffix}{dropout_str}{wd_str}{layers_str}{exp_id_str}_testfold{test_fold_idx+1}')
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
        use_best_f1 = getattr(args, 'best_f1', False)
        if patience is not None and patience <= 0:
            patience = None
        epochs_without_improvement = 0

        thal_ordinal = getattr(args, 'thal_ordinal', False)
        for epoch in range(args.epochs):
            train_loss, train_ptau_loss, train_thal_loss = train_epoch(
                model, train_loader, criterion_ptau, criterion_thal, optimizer, device,
                lambda_entropy=args.lambda_entropy if args.gene_encoder else 0.0,
                thal_ordinal=thal_ordinal, n_thal_classes=n_thal_classes,
                mixup_alpha=mixup_alpha,
                gene_cutmix_prob=gene_cutmix_prob,
                gene_cutmix_alpha=gene_cutmix_alpha,
            )
            test_loss, test_mse, test_mae, test_f1 = evaluate(
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

            if use_best_f1:
                if test_loss < best_test_loss:
                    best_test_loss = test_loss
                    best_test_f1 = test_f1
                    best_epoch = epoch
                    best_test_mse = test_mse
                    best_test_mae = test_mae
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
            else:
                if test_loss < best_test_loss:
                    best_test_loss = test_loss
                    best_test_f1 = test_f1  # for diagnostic only
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

            """
            # (legacy: best_f1 controlled by --best_f1 above)
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                best_test_f1 = test_f1
                best_epoch = epoch
                best_test_mse = test_mse
                best_test_mae = test_mae
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            """
            if (epoch + 1) % 10 == 0 and getattr(args, "verbose", True):
                print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss:.4f}, "
                      f"Test Loss: {test_loss:.4f}, Test F1: {test_f1:.4f}, Test MAE: {test_mae:.4f}")

            if patience is not None and epochs_without_improvement >= patience:
                if getattr(args, "verbose", True):
                    print(f"  Early stopping at epoch {epoch+1} (no test loss improvement for {patience} epochs).")
                break

        """
        if best_state is not None:
            model.load_state_dict(best_state)
        """
        if use_best_f1 and best_state is not None:
            model.load_state_dict(best_state)
        writer.close()

        if best_state is not None and getattr(args, 'save_best_model', False):
            save_dir = getattr(args, 'save_dir', 'models')
            os.makedirs(save_dir, exist_ok=True)
            save_name = f'{args.region}_testfold{test_fold_idx+1}{model_suffix}{attention_suffix}{exp_id_str}.pt'
            save_path = os.path.join(save_dir, save_name)
            torch.save(best_state, save_path)
            print(f"  Saved best model to {save_path}")

        if use_best_f1 and best_state is not None:
            report_f1 = best_test_f1
            report_mse = best_test_mse
            report_mae = best_test_mae
            report_epoch = best_epoch + 1
            print(f"\nTest Fold {test_fold_idx + 1} Best F1: {report_f1:.4f} (epoch {report_epoch})")
            print(f"  MSE: {report_mse:.4f}, MAE: {report_mae:.4f}")
        else:
            # Final-epoch metrics (or fallback when --best_f1 but no improvement)
            model.eval()
            with torch.no_grad():
                test_loss, test_mse, test_mae, test_f1 = evaluate(
                    model, test_loader, criterion_ptau, criterion_thal, device, idx_to_thal,
                    thal_ordinal=thal_ordinal, n_thal_classes=n_thal_classes,
                )
            report_f1 = test_f1
            report_mse = test_mse
            report_mae = test_mae
            report_epoch = epoch + 1
            print(f"\nTest Fold {test_fold_idx + 1} Final F1: {report_f1:.4f} (last epoch)")
            if use_best_f1:
                print(f"  (no improvement in any epoch; reporting last epoch)")
            elif best_test_f1 != report_f1:
                print(f"  (best F1 during training was {best_test_f1:.4f})")
            print(f"  MSE: {report_mse:.4f}, MAE: {report_mae:.4f}")

        fold_results.append({
            'test_fold': test_fold_idx + 1,
            'best_f1': report_f1,
            'final_f1': report_f1,
            'best_epoch': report_epoch,
            'best_test_mse': report_mse,
            'best_test_mae': report_mae
        })

    print(f"\n{'='*60}")
    print("Leave-One-Fold-Out Cross-Validation Summary")
    print(f"{'='*60}")
    fold_f1s = [r['final_f1'] for r in fold_results]
    label = "best" if getattr(args, 'best_f1', False) else "final (last-epoch)"
    print(f"Per-fold {label} F1: {[f'{f:.4f}' for f in fold_f1s]}")
    for result in fold_results:
        print(f"\033[92mTest Fold {result['test_fold']}: F1={result['best_f1']:.4f}, "
              f"MSE={result['best_test_mse']:.4f}, MAE={result['best_test_mae']:.4f}\033[0m")

    avg_f1 = np.mean(fold_f1s)
    avg_mse = np.mean([r['best_test_mse'] for r in fold_results])
    avg_mae = np.mean([r['best_test_mae'] for r in fold_results])

    print(f"\nAverage F1: {avg_f1:.4f}")
    print(f"Average MSE: {avg_mse:.4f}")
    print(f"Average MAE: {avg_mae:.4f}")

    return avg_f1, avg_mse, avg_mae, fold_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train model (baseline or gene encoder)')
    parser.add_argument('--region', type=str, required=True, help='Region name (e.g., EC, ITG, PFC)')
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
    parser.add_argument('--gene_encoder', action='store_true',
                       help='Use GeneEncoderModel (gene encoder + attention pooling); requires pooling=None')
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
    parser.add_argument('--patience', type=int, default=20,
                       help='Early stopping: stop if test loss does not improve for this many epochs (default: 20). Set to 0 to disable.')
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
                       help='Use best F1 across epochs and restore best model; default is last-epoch F1.')
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
    args = parser.parse_args()

    main(args)
