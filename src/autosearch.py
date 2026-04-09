"""Generalized hyperparameter search across all regions, cell types, input types, and model families.

Calls train_model_v2.main(); supports Optuna, random, and LLM samplers.

Usage:
  python src/autosearch.py --region EC --cell_type Astrocytes --sampler optuna --n_trials 50
  python src/autosearch.py --region EC --cell_type Astrocytes --sampler llm --n_trials 20
  python src/autosearch.py --region EC --cell_type Astrocytes --input_type expression --sampler optuna --n_trials 30
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_model_v2 import (
    ALL_MODEL_TYPES,
    EMBEDDING_MODELS,
    EXPRESSION_MODELS,
    INPUT_TYPES,
    main as train_v2_main,
)

# ---------------------------------------------------------------------------
# Search-space constants
# ---------------------------------------------------------------------------

REGIONS_BY_CELL_TYPE = {
    "Astrocytes": ["EC", "ITG", "PFC", "V2", "V1"],
    "Microglia": ["CrossRegion", "ITG", "PFC"],
    "MTG": ["MTG"],
}

INPUT_TYPE_CHOICES = ["expression", "embedding", "random_embedding"]
EXPR_MODEL_CHOICES = ["logreg", "svm", "mlp"]
EMB_MODEL_CHOICES = ["flatten_mlp", "gene_encoder", "mil"]
ALL_SEARCHABLE_MODELS = EXPR_MODEL_CHOICES + EMB_MODEL_CHOICES

PCA_CHOICES = [None, 16, 50, 100]
MLP_MENU = ["", "64,32,16", "128,64,32", "256,128,64,32"]
C_CHOICES = [0.01, 0.1, 1.0, 10.0]
POOLING_CHOICES = ["mean", "max", "none"]

MIL_CELLS_CHOICES = [64, 128, 256, 512]
MIL_HIDDEN_DIM_CHOICES = [64, 128, 256]
MIL_ATTN_DIM_CHOICES = [32, 64, 128]
MIL_ENCODER_LAYERS_CHOICES = [1, 2, 3]

SCVI_N_LATENT_CHOICES = [10, 20, 30, 50]
SCVI_N_LAYERS_CHOICES = [1, 2, 3]
SCVI_N_HIDDEN_CHOICES = [64, 128, 256]

MLP_LLM_MAX_LAYERS = 24
MLP_LLM_MAX_WIDTH = 8192


# ---------------------------------------------------------------------------
# TrialParams
# ---------------------------------------------------------------------------


@dataclass
class TrialParams:
    input_type: str  # expression | embedding | random_embedding
    model_type: str  # logreg | svm | mlp | scvi | flatten_mlp | gene_encoder | mil

    # Common
    pca_components: Optional[int] = None
    dropout: float = 0.3
    weight_decay: float = 0.1
    lr: float = 1e-3
    batch_size: int = 8
    label_smoothing: float = 0.0
    thal_ordinal: bool = False

    # sklearn
    C: float = 1.0

    # MLP / flatten_mlp
    mlp_hidden_sizes: str = ""
    n_hidden_layers: int = 1
    hidden_dim_3layer: int = 32
    attention: bool = False

    # Gene encoder
    lambda_entropy: float = 1e-3

    # MIL
    mil_cells_per_patient: int = 128
    mil_hidden_dim: int = 128
    mil_attn_dim: int = 64
    mil_encoder_layers: int = 2

    # scVI
    scvi_n_latent: int = 30
    scvi_n_layers: int = 2
    scvi_n_hidden: int = 128

    # Augmentation
    aug_scale_jitter: float = 0.0
    aug_per_gene_scale_jitter: float = 0.0
    aug_gene_dropout: float = 0.0
    aug_emb_dim_dropout: float = 0.0
    aug_feature_dropout: float = 0.0
    aug_gene_block_cutout_max: int = 0
    aug_noise_std: float = 0.0
    mixup_alpha: float = 0.0
    gene_cutmix_prob: float = 0.0
    gene_cutmix_alpha: float = 1.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _nearest(x, choices):
    return min(choices, key=lambda c: abs((c or 0) - (x or 0)))


def _nearest_int(x, choices):
    return min(choices, key=lambda c: abs(c - x))


def _normalize_mlp_string(s: str) -> str:
    if not s or not str(s).strip():
        return ""
    parts = [p.strip() for p in str(s).split(",") if p.strip()]
    try:
        clamped = [max(1, min(MLP_LLM_MAX_WIDTH, int(p))) for p in parts[:MLP_LLM_MAX_LAYERS]]
    except ValueError:
        return ""
    return ",".join(str(x) for x in clamped)


# ---------------------------------------------------------------------------
# Suggest functions
# ---------------------------------------------------------------------------


def suggest_trial_optuna(trial, *, fixed_input_type: Optional[str] = None) -> TrialParams:
    if fixed_input_type:
        input_type = fixed_input_type
    else:
        input_type = trial.suggest_categorical("input_type", INPUT_TYPE_CHOICES)

    if input_type == "expression":
        # Use a distinct Optuna parameter name for expression-only models
        model_type = trial.suggest_categorical("model_type_expr", EXPR_MODEL_CHOICES)
    else:
        # And a separate name for embedding-based models to avoid
        # dynamic value spaces on a single parameter.
        model_type = trial.suggest_categorical("model_type_emb", EMB_MODEL_CHOICES)

    is_sklearn = model_type in ("logreg", "svm")
    is_scvi = model_type == "scvi"
    is_mil = model_type == "mil"
    is_gene_enc = model_type == "gene_encoder"
    is_mlp = model_type in ("mlp", "flatten_mlp")

    pca_components = trial.suggest_categorical("pca_components", PCA_CHOICES)
    dropout = trial.suggest_float("dropout", 0.1, 0.55)
    weight_decay = trial.suggest_categorical("weight_decay", [0.01, 0.1, 0.5, 1.0])
    lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [4, 8, 16])
    label_smoothing = trial.suggest_categorical("label_smoothing", [0.0, 0.05, 0.1])
    thal_ordinal = trial.suggest_categorical("thal_ordinal", [False, True])

    C = trial.suggest_categorical("C", C_CHOICES) if is_sklearn else 1.0
    mlp_hidden_sizes = trial.suggest_categorical("mlp_hidden_sizes", MLP_MENU) if is_mlp else ""
    n_hidden_layers = trial.suggest_categorical("n_hidden_layers", [1, 3]) if is_mlp else 1
    hidden_dim_3layer = trial.suggest_categorical("hidden_dim_3layer", [32, 64, 128]) if is_mlp else 32
    attention = trial.suggest_categorical("attention", [False, True]) if is_mlp else False

    lambda_entropy = trial.suggest_float("lambda_entropy", 1e-4, 0.2, log=True) if is_gene_enc else 1e-3

    mil_cells = trial.suggest_categorical("mil_cells_per_patient", MIL_CELLS_CHOICES) if is_mil else 128
    mil_hdim = trial.suggest_categorical("mil_hidden_dim", MIL_HIDDEN_DIM_CHOICES) if is_mil else 128
    mil_adim = trial.suggest_categorical("mil_attn_dim", MIL_ATTN_DIM_CHOICES) if is_mil else 64
    mil_layers = trial.suggest_categorical("mil_encoder_layers", MIL_ENCODER_LAYERS_CHOICES) if is_mil else 2

    scvi_lat = trial.suggest_categorical("scvi_n_latent", SCVI_N_LATENT_CHOICES) if is_scvi else 30
    scvi_lay = trial.suggest_categorical("scvi_n_layers", SCVI_N_LAYERS_CHOICES) if is_scvi else 2
    scvi_hid = trial.suggest_categorical("scvi_n_hidden", SCVI_N_HIDDEN_CHOICES) if is_scvi else 128

    # Augmentation (only for pytorch models on embeddings)
    is_torch_emb = input_type != "expression" and not is_sklearn and not is_scvi
    aug_scale = trial.suggest_categorical("aug_scale_jitter", [0.0, 0.05, 0.1, 0.15]) if is_torch_emb else 0.0
    aug_pg = trial.suggest_categorical("aug_per_gene_scale_jitter", [0.0, 0.05, 0.1]) if is_torch_emb else 0.0
    aug_ed = trial.suggest_categorical("aug_emb_dim_dropout", [0.0, 0.05, 0.1]) if is_torch_emb else 0.0
    aug_noise = trial.suggest_categorical("aug_noise_std", [0.0, 0.01, 0.02, 0.03]) if is_torch_emb else 0.0
    aug_gd = trial.suggest_categorical("aug_gene_dropout", [0.0, 0.05, 0.1]) if is_torch_emb else 0.0
    aug_block = trial.suggest_categorical("aug_gene_block_cutout_max", [0, 32, 128, 256]) if is_torch_emb else 0

    mixup_alpha = trial.suggest_categorical("mixup_alpha", [0.0, 0.2, 0.4]) if (is_torch_emb and not is_mil) else 0.0
    gene_cutmix_prob = trial.suggest_categorical("gene_cutmix_prob", [0.0, 0.25, 0.5]) if (is_torch_emb and not is_mil) else 0.0
    gene_cutmix_alpha = trial.suggest_float("gene_cutmix_alpha", 0.5, 2.0) if (is_torch_emb and not is_mil) else 1.0

    if thal_ordinal or is_mil:
        mixup_alpha = 0.0
        gene_cutmix_prob = 0.0

    return TrialParams(
        input_type=input_type,
        model_type=model_type,
        pca_components=pca_components,
        dropout=dropout,
        weight_decay=weight_decay,
        lr=lr,
        batch_size=batch_size,
        label_smoothing=label_smoothing,
        thal_ordinal=thal_ordinal,
        C=C,
        mlp_hidden_sizes=mlp_hidden_sizes,
        n_hidden_layers=n_hidden_layers,
        hidden_dim_3layer=hidden_dim_3layer,
        attention=attention,
        lambda_entropy=lambda_entropy,
        mil_cells_per_patient=mil_cells,
        mil_hidden_dim=mil_hdim,
        mil_attn_dim=mil_adim,
        mil_encoder_layers=mil_layers,
        scvi_n_latent=scvi_lat,
        scvi_n_layers=scvi_lay,
        scvi_n_hidden=scvi_hid,
        aug_scale_jitter=aug_scale,
        aug_per_gene_scale_jitter=aug_pg,
        aug_gene_dropout=aug_gd,
        aug_emb_dim_dropout=aug_ed,
        aug_feature_dropout=0.0,
        aug_gene_block_cutout_max=aug_block,
        aug_noise_std=aug_noise,
        mixup_alpha=mixup_alpha,
        gene_cutmix_prob=gene_cutmix_prob,
        gene_cutmix_alpha=gene_cutmix_alpha,
    )


def suggest_trial_random(rng, *, fixed_input_type: Optional[str] = None) -> TrialParams:
    if fixed_input_type:
        input_type = fixed_input_type
    else:
        input_type = rng.choice(INPUT_TYPE_CHOICES)

    if input_type == "expression":
        model_type = rng.choice(EXPR_MODEL_CHOICES)
    else:
        model_type = rng.choice(EMB_MODEL_CHOICES)

    is_sklearn = model_type in ("logreg", "svm")
    is_scvi = model_type == "scvi"
    is_mil = model_type == "mil"
    is_gene_enc = model_type == "gene_encoder"
    is_mlp = model_type in ("mlp", "flatten_mlp")

    pca_components = rng.choice(PCA_CHOICES)
    dropout = rng.uniform(0.1, 0.55)
    weight_decay = rng.choice([0.01, 0.1, 0.5, 1.0])
    lr = 10 ** rng.uniform(-4, math.log10(3e-3))
    batch_size = rng.choice([4, 8, 16])
    label_smoothing = rng.choice([0.0, 0.05, 0.1])
    thal_ordinal = rng.random() < 0.25

    C = rng.choice(C_CHOICES) if is_sklearn else 1.0
    mlp_hidden_sizes = rng.choice(MLP_MENU) if is_mlp else ""
    n_hidden_layers = rng.choice([1, 3]) if is_mlp else 1
    hidden_dim_3layer = rng.choice([32, 64, 128]) if is_mlp else 32
    attention = rng.random() < 0.5 if is_mlp else False

    lambda_entropy = 10 ** rng.uniform(-4, math.log10(0.2)) if is_gene_enc else 1e-3

    mil_cells = rng.choice(MIL_CELLS_CHOICES) if is_mil else 128
    mil_hdim = rng.choice(MIL_HIDDEN_DIM_CHOICES) if is_mil else 128
    mil_adim = rng.choice(MIL_ATTN_DIM_CHOICES) if is_mil else 64
    mil_layers = rng.choice(MIL_ENCODER_LAYERS_CHOICES) if is_mil else 2

    scvi_lat = rng.choice(SCVI_N_LATENT_CHOICES) if is_scvi else 30
    scvi_lay = rng.choice(SCVI_N_LAYERS_CHOICES) if is_scvi else 2
    scvi_hid = rng.choice(SCVI_N_HIDDEN_CHOICES) if is_scvi else 128

    is_torch_emb = input_type != "expression" and not is_sklearn and not is_scvi
    aug_scale = rng.choice([0.0, 0.05, 0.1, 0.15]) if is_torch_emb else 0.0
    aug_pg = rng.choice([0.0, 0.05, 0.1]) if is_torch_emb else 0.0
    aug_ed = rng.choice([0.0, 0.05, 0.1]) if is_torch_emb else 0.0
    aug_noise = rng.choice([0.0, 0.01, 0.02, 0.03]) if is_torch_emb else 0.0
    aug_gd = rng.choice([0.0, 0.05, 0.1]) if is_torch_emb else 0.0
    aug_block = rng.choice([0, 32, 128, 256]) if is_torch_emb else 0
    mixup_alpha = rng.choice([0.0, 0.2, 0.4]) if (is_torch_emb and not is_mil) else 0.0
    gene_cutmix_prob = rng.choice([0.0, 0.25, 0.5]) if (is_torch_emb and not is_mil) else 0.0
    gene_cutmix_alpha = rng.uniform(0.5, 2.0) if (is_torch_emb and not is_mil) else 1.0

    if thal_ordinal or is_mil:
        mixup_alpha = 0.0
        gene_cutmix_prob = 0.0

    return TrialParams(
        input_type=input_type,
        model_type=model_type,
        pca_components=pca_components,
        dropout=dropout,
        weight_decay=weight_decay,
        lr=lr,
        batch_size=batch_size,
        label_smoothing=label_smoothing,
        thal_ordinal=thal_ordinal,
        C=C,
        mlp_hidden_sizes=mlp_hidden_sizes,
        n_hidden_layers=n_hidden_layers,
        hidden_dim_3layer=hidden_dim_3layer,
        attention=attention,
        lambda_entropy=lambda_entropy,
        mil_cells_per_patient=mil_cells,
        mil_hidden_dim=mil_hdim,
        mil_attn_dim=mil_adim,
        mil_encoder_layers=mil_layers,
        scvi_n_latent=scvi_lat,
        scvi_n_layers=scvi_lay,
        scvi_n_hidden=scvi_hid,
        aug_scale_jitter=aug_scale,
        aug_per_gene_scale_jitter=aug_pg,
        aug_gene_dropout=aug_gd,
        aug_emb_dim_dropout=aug_ed,
        aug_feature_dropout=0.0,
        aug_gene_block_cutout_max=aug_block,
        aug_noise_std=aug_noise,
        mixup_alpha=mixup_alpha,
        gene_cutmix_prob=gene_cutmix_prob,
        gene_cutmix_alpha=gene_cutmix_alpha,
    )


def trial_params_from_llm_dict(raw: Dict[str, Any], *, fixed_input_type: Optional[str] = None) -> TrialParams:
    """Build TrialParams from an LLM-produced JSON object with clamping."""
    g = {k.replace("-", "_"): v for k, v in raw.items()}

    def b(key, default=False):
        v = g.get(key, default)
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.strip().lower() in ("1", "true", "yes")
        return bool(v)

    def f(key, default):
        try:
            return float(g.get(key, default))
        except (TypeError, ValueError):
            return default

    def i(key, default):
        try:
            return int(g.get(key, default))
        except (TypeError, ValueError):
            return default

    def s(key, default):
        v = g.get(key, default)
        return str(v).strip().lower() if v is not None else default

    input_type = fixed_input_type or s("input_type", "expression")
    if input_type not in INPUT_TYPE_CHOICES:
        input_type = "expression"

    model_type = s("model_type", "logreg")
    if input_type == "expression":
        if model_type not in EXPR_MODEL_CHOICES:
            model_type = "logreg"
    else:
        if model_type not in EMB_MODEL_CHOICES:
            model_type = "flatten_mlp"

    is_sklearn = model_type in ("logreg", "svm")
    is_mil = model_type == "mil"
    is_gene_enc = model_type == "gene_encoder"
    is_mlp = model_type in ("mlp", "flatten_mlp")
    is_scvi = model_type == "scvi"
    is_torch_emb = input_type != "expression" and not is_sklearn and not is_scvi

    pca_val = g.get("pca_components", None)
    pca_components = _nearest(pca_val, PCA_CHOICES) if pca_val is not None else None

    dropout = max(0.1, min(0.55, f("dropout", 0.3)))
    weight_decay = _nearest(f("weight_decay", 0.1), [0.01, 0.1, 0.5, 1.0])
    lr = max(1e-4, min(3e-3, f("lr", 1e-3)))
    batch_size = _nearest_int(i("batch_size", 8), [4, 8, 16])
    label_smoothing = _nearest(f("label_smoothing", 0.0), [0.0, 0.05, 0.1])
    thal_ordinal = b("thal_ordinal", False)

    C = _nearest(f("C", 1.0), C_CHOICES) if is_sklearn else 1.0

    if is_mlp:
        mlp_val = g.get("mlp_hidden_sizes", "")
        if isinstance(mlp_val, list):
            try:
                mlp_raw = ",".join(str(int(x)) for x in mlp_val)
            except (TypeError, ValueError):
                mlp_raw = ""
        else:
            mlp_raw = str(mlp_val or "").strip()
        mlp_hidden_sizes = _normalize_mlp_string(mlp_raw)
    else:
        mlp_hidden_sizes = ""

    n_hidden_layers = i("n_hidden_layers", 1) if is_mlp else 1
    if n_hidden_layers not in (1, 3):
        n_hidden_layers = 1
    hidden_dim_3layer = _nearest_int(i("hidden_dim_3layer", 32), [32, 64, 128]) if is_mlp else 32
    attention = b("attention", False) if is_mlp else False
    lambda_entropy = max(1e-4, min(0.2, f("lambda_entropy", 1e-3))) if is_gene_enc else 1e-3

    mil_cells = _nearest_int(i("mil_cells_per_patient", 128), MIL_CELLS_CHOICES) if is_mil else 128
    mil_hdim = _nearest_int(i("mil_hidden_dim", 128), MIL_HIDDEN_DIM_CHOICES) if is_mil else 128
    mil_adim = _nearest_int(i("mil_attn_dim", 64), MIL_ATTN_DIM_CHOICES) if is_mil else 64
    mil_layers = _nearest_int(i("mil_encoder_layers", 2), MIL_ENCODER_LAYERS_CHOICES) if is_mil else 2

    scvi_lat = _nearest_int(i("scvi_n_latent", 30), SCVI_N_LATENT_CHOICES) if is_scvi else 30
    scvi_lay = _nearest_int(i("scvi_n_layers", 2), SCVI_N_LAYERS_CHOICES) if is_scvi else 2
    scvi_hid = _nearest_int(i("scvi_n_hidden", 128), SCVI_N_HIDDEN_CHOICES) if is_scvi else 128

    aug_scale = _nearest(f("aug_scale_jitter", 0.0), [0.0, 0.05, 0.1, 0.15]) if is_torch_emb else 0.0
    aug_pg = _nearest(f("aug_per_gene_scale_jitter", 0.0), [0.0, 0.05, 0.1]) if is_torch_emb else 0.0
    aug_ed = _nearest(f("aug_emb_dim_dropout", 0.0), [0.0, 0.05, 0.1]) if is_torch_emb else 0.0
    aug_noise = _nearest(f("aug_noise_std", 0.0), [0.0, 0.01, 0.02, 0.03]) if is_torch_emb else 0.0
    aug_gd = _nearest(f("aug_gene_dropout", 0.0), [0.0, 0.05, 0.1]) if is_torch_emb else 0.0
    aug_block = _nearest_int(i("aug_gene_block_cutout_max", 0), [0, 32, 128, 256]) if is_torch_emb else 0
    mixup_alpha = _nearest(f("mixup_alpha", 0.0), [0.0, 0.2, 0.4]) if (is_torch_emb and not is_mil) else 0.0
    gene_cutmix_prob = _nearest(f("gene_cutmix_prob", 0.0), [0.0, 0.25, 0.5]) if (is_torch_emb and not is_mil) else 0.0
    gene_cutmix_alpha = max(0.5, min(2.0, f("gene_cutmix_alpha", 1.0)))

    if thal_ordinal or is_mil:
        mixup_alpha = 0.0
        gene_cutmix_prob = 0.0

    return TrialParams(
        input_type=input_type,
        model_type=model_type,
        pca_components=pca_components,
        dropout=dropout,
        weight_decay=weight_decay,
        lr=lr,
        batch_size=batch_size,
        label_smoothing=label_smoothing,
        thal_ordinal=thal_ordinal,
        C=C,
        mlp_hidden_sizes=mlp_hidden_sizes,
        n_hidden_layers=n_hidden_layers,
        hidden_dim_3layer=hidden_dim_3layer,
        attention=attention,
        lambda_entropy=lambda_entropy,
        mil_cells_per_patient=mil_cells,
        mil_hidden_dim=mil_hdim,
        mil_attn_dim=mil_adim,
        mil_encoder_layers=mil_layers,
        scvi_n_latent=scvi_lat,
        scvi_n_layers=scvi_lay,
        scvi_n_hidden=scvi_hid,
        aug_scale_jitter=aug_scale,
        aug_per_gene_scale_jitter=aug_pg,
        aug_gene_dropout=aug_gd,
        aug_emb_dim_dropout=aug_ed,
        aug_feature_dropout=0.0,
        aug_gene_block_cutout_max=aug_block,
        aug_noise_std=aug_noise,
        mixup_alpha=mixup_alpha,
        gene_cutmix_prob=gene_cutmix_prob,
        gene_cutmix_alpha=gene_cutmix_alpha,
    )


# ---------------------------------------------------------------------------
# Namespace builder
# ---------------------------------------------------------------------------


def params_to_namespace(
    p: TrialParams,
    *,
    region: str,
    cell_type: str,
    exp_id: str,
    log_base: str,
    epochs: int,
    patience: int,
    device: str,
    seed: int = 0,
    min_epochs: int = 30,
    quiet: bool = True,
    no_online_augment: bool = False,
    attention_heads: int = 1,
) -> SimpleNamespace:
    return SimpleNamespace(
        region=region,
        cell_type=cell_type,
        input_type=p.input_type,
        model_type=p.model_type,
        seed=seed,
        n_seeds=1,
        n_folds_eval=8,
        epochs=epochs,
        patience=patience,
        min_epochs=min_epochs,
        batch_size=p.batch_size,
        lr=p.lr,
        dropout=p.dropout,
        weight_decay=p.weight_decay,
        pca_components=p.pca_components,
        loss_ptau="none",
        label_smoothing=p.label_smoothing,
        thal_ordinal=p.thal_ordinal,
        device=device,
        verbose_cv=False,
        q=quiet,
        C=p.C,
        mlp_hidden_sizes=p.mlp_hidden_sizes,
        n_hidden_layers=p.n_hidden_layers,
        hidden_dim_3layer=p.hidden_dim_3layer,
        attention=p.attention,
        attention_heads=attention_heads,
        lambda_entropy=p.lambda_entropy,
        mil_cells_per_patient=p.mil_cells_per_patient,
        mil_hidden_dim=p.mil_hidden_dim,
        mil_attn_dim=p.mil_attn_dim,
        mil_encoder_layers=p.mil_encoder_layers,
        scvi_n_latent=p.scvi_n_latent,
        scvi_n_layers=p.scvi_n_layers,
        scvi_n_hidden=p.scvi_n_hidden,
        no_online_augment=no_online_augment,
        aug_scale_jitter=0.0 if no_online_augment else p.aug_scale_jitter,
        aug_per_gene_scale_jitter=0.0 if no_online_augment else p.aug_per_gene_scale_jitter,
        aug_gene_dropout=0.0 if no_online_augment else p.aug_gene_dropout,
        aug_emb_dim_dropout=0.0 if no_online_augment else p.aug_emb_dim_dropout,
        aug_feature_dropout=0.0 if no_online_augment else p.aug_feature_dropout,
        aug_gene_block_cutout_max=0 if no_online_augment else p.aug_gene_block_cutout_max,
        aug_gene_block_cutout_prob=0.5,
        aug_noise_std=0.0 if no_online_augment else p.aug_noise_std,
        mixup_alpha=0.0 if no_online_augment else p.mixup_alpha,
        gene_cutmix_prob=0.0 if no_online_augment else p.gene_cutmix_prob,
        gene_cutmix_alpha=p.gene_cutmix_alpha,
        no_tensorboard=True,
        exp_id=exp_id,
        log_base=log_base,
    )


# ---------------------------------------------------------------------------
# Run helper
# ---------------------------------------------------------------------------


def _run_one_trial(ns: SimpleNamespace) -> Tuple[float, Optional[str]]:
    try:
        avg_f1, _, _, _ = train_v2_main(ns)
        if avg_f1 is None or (isinstance(avg_f1, float) and math.isnan(avg_f1)):
            return float("-inf"), "invalid_f1"
        return float(avg_f1), None
    except Exception as e:
        return float("-inf"), f"{type(e).__name__}: {e}\n{traceback.format_exc()}"


# ---------------------------------------------------------------------------
# JSONL / CSV helpers
# ---------------------------------------------------------------------------


def _jsonl_append(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, default=str) + "\n")


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not os.path.isfile(path):
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _next_trial_id(rows: List[Dict[str, Any]]) -> int:
    best = -1
    for r in rows:
        try:
            best = max(best, int(r.get("trial", -1)))
        except (TypeError, ValueError):
            pass
    return best + 1


def _best_f1_footer(rows: List[Dict[str, Any]]) -> None:
    best_v, best_t, best_e = None, None, None
    for r in rows:
        if r.get("error"):
            continue
        try:
            v = float(r["value"])
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isfinite(v):
            continue
        if best_v is None or v > best_v:
            best_v, best_t, best_e = v, r.get("trial"), r.get("exp_id")
    parts = []
    if best_t is not None:
        parts.append(f"trial {best_t}")
    if best_e:
        parts.append(f"exp_id={best_e}")
    suffix = f" ({', '.join(parts)})" if parts else ""
    if best_v is None:
        print(f"\nBest F1 (this run): none{suffix}")
    else:
        print(f"\nBest F1 (this run): {best_v:.4f}{suffix}")


def _export_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    keys = sorted(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# LLM system prompt (dynamic)
# ---------------------------------------------------------------------------


def build_llm_system_prompt(region: str, cell_type: str, fixed_input_type: Optional[str] = None) -> str:
    input_desc = f'"{fixed_input_type}"' if fixed_input_type else 'one of: "expression", "embedding", "random_embedding"'
    if fixed_input_type == "expression":
        model_desc = ', '.join(f'"{m}"' for m in EXPR_MODEL_CHOICES)
    elif fixed_input_type in ("embedding", "random_embedding"):
        model_desc = ', '.join(f'"{m}"' for m in EMB_MODEL_CHOICES)
    else:
        model_desc = ', '.join(f'"{m}"' for m in ALL_SEARCHABLE_MODELS)

    return f"""You optimize hyperparameters for models predicting Thal stage from single-cell data.
Region: {region}, cell_type: {cell_type}. Objective: maximize mean LOFO F1 (loss_ptau=none, Thal only).

Output rules:
- Reply with ONE JSON array. Each element is one trial config (flat object).
- "input_type" must be {input_desc}.
- "model_type" must be one of: {model_desc}.

Model rules:
- Expression-only models (input_type="expression"):
  - "logreg": LogisticRegression. Search C (0.01/0.1/1.0/10.0), pca_components (null/16/50/100).
  - "svm": LinearSVC. Search C, pca_components.
  - "mlp": Patient-level MLP. Search mlp_hidden_sizes, n_hidden_layers (1/3), hidden_dim_3layer, dropout, lr, attention.
  - "scvi": scVI latent + LogReg. Search scvi_n_latent (10/20/30/50), scvi_n_layers (1/2/3), scvi_n_hidden (64/128/256).

- Embedding-based models (input_type="embedding" or "random_embedding"):
  - "flatten_mlp": Flatten (n_genes*D) -> MLP. Search mlp_hidden_sizes, dropout, lr.
  - "gene_encoder": Per-gene encoder + attention pooling. Search lambda_entropy (1e-4 to 0.2).
  - "mil": Cell-level MIL bags. Search mil_cells_per_patient (64/128/256/512), mil_hidden_dim (64/128/256), mil_attn_dim (32/64/128), mil_encoder_layers (1/2/3). Set mixup_alpha=0, gene_cutmix_prob=0.

- Embedding augmentation (only for embedding-based PyTorch models):
  - aug_scale_jitter: 0.0/0.05/0.1/0.15
  - aug_per_gene_scale_jitter: 0.0/0.05/0.1
  - aug_gene_dropout: 0.0/0.05/0.1
  - aug_noise_std: 0.0/0.01/0.02/0.03

- If thal_ordinal is true: mixup_alpha and gene_cutmix_prob must be 0.
- batch_size: 4/8/16. weight_decay: 0.01/0.1/0.5/1.0.
- pca_components: null/16/50/100.

Be creative: explore near the best trials, try boundary settings, and diversify."""


# ---------------------------------------------------------------------------
# LLM chat
# ---------------------------------------------------------------------------


def _call_llm_chat(
    *, system: str, user: str, model: str, api_key: str,
    base_url: Optional[str], timeout: float,
) -> str:
    from openai import OpenAI
    kw: Dict[str, Any] = {"api_key": api_key}
    if base_url:
        kw["base_url"] = base_url
    client = OpenAI(**kw)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.65,
        timeout=timeout,
    )
    return (resp.choices[0].message.content or "").strip()


def _extract_json_array(text: str) -> List[Dict[str, Any]]:
    text = text.strip()
    if "```" in text:
        blocks = re.findall(r"```(?:json)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
        for b in blocks:
            b = b.strip()
            if b.startswith("["):
                data = json.loads(b)
                if isinstance(data, list):
                    return [x for x in data if isinstance(x, dict)]
    m = re.search(r"\[[\s\S]*\]", text)
    if m:
        data = json.loads(m.group(0))
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]
    data = json.loads(text)
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    raise ValueError("Could not parse JSON array from LLM output")


def _format_history(rows: List[Dict[str, Any]], max_rows: int) -> str:
    tail = rows[-max_rows:] if len(rows) > max_rows else rows
    lines = []
    for r in tail:
        brief = {
            "trial": r.get("trial"),
            "value": r.get("value"),
            "error": (str(r.get("error", ""))[:120] + "...") if len(str(r.get("error", ""))) > 120 else r.get("error", ""),
            "input_type": r.get("input_type"),
            "model_type": r.get("model_type"),
            "pca_components": r.get("pca_components"),
            "lr": r.get("lr"),
            "dropout": r.get("dropout"),
            "batch_size": r.get("batch_size"),
        }
        lines.append(json.dumps(brief, default=str))
    return "\n".join(lines) if lines else "(no prior trials)"


# ---------------------------------------------------------------------------
# Optuna dashboard support
# ---------------------------------------------------------------------------


def _optuna_distributions(p: TrialParams, fixed_input_type: Optional[str]):
    from optuna.distributions import CategoricalDistribution, FloatDistribution
    d: Dict[str, Any] = {}
    params: Dict[str, Any] = {}

    if not fixed_input_type:
        d["input_type"] = CategoricalDistribution(tuple(INPUT_TYPE_CHOICES))
        params["input_type"] = p.input_type

    if p.input_type == "expression":
        d["model_type"] = CategoricalDistribution(tuple(EXPR_MODEL_CHOICES))
    else:
        d["model_type"] = CategoricalDistribution(tuple(EMB_MODEL_CHOICES))
    params["model_type"] = p.model_type

    d["pca_components"] = CategoricalDistribution(tuple(PCA_CHOICES))
    params["pca_components"] = p.pca_components
    d["dropout"] = FloatDistribution(0.1, 0.55)
    params["dropout"] = p.dropout
    d["weight_decay"] = CategoricalDistribution((0.01, 0.1, 0.5, 1.0))
    params["weight_decay"] = p.weight_decay
    d["lr"] = FloatDistribution(1e-4, 3e-3, log=True)
    params["lr"] = p.lr
    d["batch_size"] = CategoricalDistribution((4, 8, 16))
    params["batch_size"] = p.batch_size
    d["label_smoothing"] = CategoricalDistribution((0.0, 0.05, 0.1))
    params["label_smoothing"] = p.label_smoothing
    d["thal_ordinal"] = CategoricalDistribution((False, True))
    params["thal_ordinal"] = p.thal_ordinal

    return params, d


def _append_optuna_trial(study, p, *, f1, err, trial_id, exp_id, source, fixed_input_type):
    from optuna.trial import TrialState, create_trial
    params, distributions = _optuna_distributions(p, fixed_input_type)
    user_attrs = {"source": source, "jsonl_trial": trial_id, "exp_id": exp_id, "trial_params": asdict(p)}
    if err:
        user_attrs["error"] = err[:4000]
    v = float(f1)
    ok = not err and math.isfinite(v) and v > float("-inf")
    if ok:
        trial = create_trial(state=TrialState.COMPLETE, value=v, params=params, distributions=distributions, user_attrs=user_attrs)
    else:
        trial = create_trial(state=TrialState.FAIL, params=params, distributions=distributions, user_attrs=user_attrs)
    study.add_trial(trial)


# ---------------------------------------------------------------------------
# Sampler runners
# ---------------------------------------------------------------------------


def _default_storage_url(log_base: str, region: str, cell_type: str) -> str:
    db_path = os.path.join(os.path.abspath(log_base), f"{region}_{cell_type}_optuna.db")
    return "sqlite:///" + db_path.replace("\\", "/")


def _create_optuna_study(args: argparse.Namespace, storage: str):
    """Create or resume study; optionally delete same-named study first (--new_study)."""
    import optuna

    if getattr(args, "new_study", False):
        try:
            optuna.delete_study(study_name=args.study_name, storage=storage)
        except Exception:
            pass  # study may not exist (KeyError / storage-specific errors)
        print(
            f"  [optuna] Fresh study: removed '{args.study_name}' on this storage if it existed."
        )
    return optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction="maximize",
        load_if_exists=True,
    )


def run_optuna_sampler(args: argparse.Namespace) -> None:
    import optuna
    import torch
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.log_base, exist_ok=True)
    storage = args.storage or _default_storage_url(args.log_base, args.region, args.cell_type)
    study = _create_optuna_study(args, storage)

    jsonl_path = os.path.join(args.log_base, f"{args.study_name}_trials.jsonl")
    open(jsonl_path, "w", encoding="utf-8").close()
    csv_path = os.path.join(args.log_base, f"{args.study_name}_trials.csv")
    best_json_path = os.path.join(args.log_base, f"{args.study_name}_best_params.json")

    fixed_it = getattr(args, "input_type", None)
    trial_rows: List[Dict[str, Any]] = []

    def objective(trial):
        p = suggest_trial_optuna(trial, fixed_input_type=fixed_it)
        trial.set_user_attr("trial_params", asdict(p))
        trial.set_user_attr("source", "optuna")
        exp_id = f"autosearch_{args.study_name}_t{trial.number}"
        ns = params_to_namespace(
            p, region=args.region, cell_type=args.cell_type,
            exp_id=exp_id, log_base=args.log_base, epochs=args.epochs,
            patience=args.patience, device=device, seed=args.seed,
            quiet=not args.verbose, no_online_augment=args.no_online_augment,
            attention_heads=args.attention_heads,
        )
        f1, err = _run_one_trial(ns)
        row = {"trial": trial.number, "value": f1, "error": err or "", "exp_id": exp_id, **asdict(p)}
        trial_rows.append(row)
        _jsonl_append(jsonl_path, row)
        if err:
            print(f"  Trial {trial.number} FAILED: {err[:200]}")
        else:
            print(f"  Trial {trial.number} F1={f1:.4f} [{p.input_type}/{p.model_type}]")
        return f1

    study.optimize(objective, n_trials=args.n_trials)
    _export_csv(csv_path, trial_rows)
    try:
        best = study.best_trial
        payload = {
            "study_name": args.study_name, "storage": storage,
            "best_value": best.value, "best_trial": best.number,
            "trial_params": best.user_attrs.get("trial_params", {}),
        }
        with open(best_json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
    except ValueError:
        with open(best_json_path, "w", encoding="utf-8") as f:
            json.dump({"error": "no_completed_trials"}, f, indent=2)
    print(f"Wrote {best_json_path}, {csv_path}, {jsonl_path}")
    _best_f1_footer(trial_rows)


def run_random_sampler(args: argparse.Namespace) -> None:
    import random
    import torch
    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.log_base, exist_ok=True)
    jsonl_path = os.path.join(args.log_base, f"{args.study_name}_trials.jsonl")
    open(jsonl_path, "w", encoding="utf-8").close()
    csv_path = os.path.join(args.log_base, f"{args.study_name}_trials.csv")
    best_json_path = os.path.join(args.log_base, f"{args.study_name}_best_params.json")

    fixed_it = getattr(args, "input_type", None)
    optuna_study = None
    if not args.no_optuna:
        try:
            import optuna
            storage = args.storage or _default_storage_url(args.log_base, args.region, args.cell_type)
            optuna_study = _create_optuna_study(args, storage)
        except ImportError:
            pass

    trial_rows: List[Dict[str, Any]] = []
    best_f1 = float("-inf")
    best_payload = None

    for i in range(args.n_trials):
        p = suggest_trial_random(rng, fixed_input_type=fixed_it)
        exp_id = f"autosearch_{args.study_name}_r{i}"
        ns = params_to_namespace(
            p, region=args.region, cell_type=args.cell_type,
            exp_id=exp_id, log_base=args.log_base, epochs=args.epochs,
            patience=args.patience, device=device, seed=args.seed,
            quiet=not args.verbose, no_online_augment=args.no_online_augment,
            attention_heads=args.attention_heads,
        )
        f1, err = _run_one_trial(ns)
        row = {"trial": i, "value": f1, "error": err or "", "exp_id": exp_id, **asdict(p)}
        trial_rows.append(row)
        _jsonl_append(jsonl_path, row)
        if f1 > best_f1 and not err:
            best_f1 = f1
            best_payload = {"best_value": f1, "trial": i, "params": asdict(p), "exp_id": exp_id}
        print(f"  Random trial {i} F1={f1:.4f} [{p.input_type}/{p.model_type}]" + (f" err={err[:80]}" if err else ""))

        if optuna_study is not None:
            try:
                _append_optuna_trial(
                    optuna_study, p, f1=f1, err=err, trial_id=i, exp_id=exp_id,
                    source="random", fixed_input_type=fixed_it,
                )
            except Exception as e:
                print(f"  Optuna append failed: {e}")

    _export_csv(csv_path, trial_rows)
    if best_payload:
        with open(best_json_path, "w", encoding="utf-8") as f:
            json.dump({"study_name": args.study_name, "sampler": "random", **best_payload}, f, indent=2, default=str)
    print(f"\nWrote {best_json_path}, {csv_path}, {jsonl_path}")
    _best_f1_footer(trial_rows)


def run_llm_sampler(args: argparse.Namespace) -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        load_dotenv = None
    if load_dotenv:
        load_dotenv(Path(__file__).resolve().parent.parent / ".env")
        load_dotenv()

    import torch
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.log_base, exist_ok=True)
    jsonl_path = args.resume_jsonl or os.path.join(args.log_base, f"{args.study_name}_trials.jsonl")
    llm_log_path = os.path.join(args.log_base, f"{args.study_name}_llm_rounds.jsonl")

    fixed_it = getattr(args, "input_type", None)
    system_prompt = (
        Path(args.system_prompt_file).read_text(encoding="utf-8")
        if args.system_prompt_file
        else build_llm_system_prompt(args.region, args.cell_type, fixed_it)
    )

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key and args.base_url:
        api_key = "ollama"
    elif not api_key:
        raise SystemExit("Set OPENAI_API_KEY or pass --api_key")

    optuna_study = None
    if not args.no_optuna:
        try:
            import optuna
            storage = args.storage or _default_storage_url(args.log_base, args.region, args.cell_type)
            optuna_study = _create_optuna_study(args, storage)
        except ImportError:
            pass

    existing = _load_jsonl(jsonl_path)
    trial_id = _next_trial_id(existing)
    done = 0
    n_total = args.n_trials
    trial_rows: List[Dict[str, Any]] = []

    while done < n_total:
        n_prop = min(args.proposals_per_round, n_total - done)
        history = _load_jsonl(jsonl_path)
        hist_txt = _format_history(history, args.history_max_trials)
        user_msg = (
            f"Region={args.region}, cell_type={args.cell_type}.\n"
            f"Prior trials:\n{hist_txt}\n\n"
            f"Propose exactly {n_prop} new configurations as a JSON array."
        )

        proposals = None
        last_raw = ""
        print(f"\n--- LLM round: need {n_prop} config(s); {done}/{n_total} done ---")
        for attempt in range(args.llm_retries):
            try:
                if attempt > 0:
                    print(f"  LLM retry {attempt + 1}/{args.llm_retries}...")
                last_raw = _call_llm_chat(
                    system=system_prompt, user=user_msg,
                    model=args.llm_model, api_key=api_key,
                    base_url=args.base_url, timeout=args.llm_timeout,
                )
                proposals = _extract_json_array(last_raw)
                if proposals:
                    break
            except Exception as e:
                last_raw = f"<error {attempt}: {e}>"
                time.sleep(2 ** attempt)

        _jsonl_append(llm_log_path, {
            "utc": datetime.now(timezone.utc).isoformat(),
            "trial_start": trial_id, "n_requested": n_prop,
            "raw_preview": (last_raw[:2000] + "...") if len(last_raw) > 2000 else last_raw,
            "parsed_count": len(proposals) if proposals else 0,
        })

        if not proposals:
            raise SystemExit("LLM returned no valid JSON after retries.")

        for raw_cfg in proposals[:n_prop]:
            if done >= n_total:
                break
            p = trial_params_from_llm_dict(raw_cfg, fixed_input_type=fixed_it)
            exp_id = f"autosearch_llm_{args.study_name}_t{trial_id}"
            ns = params_to_namespace(
                p, region=args.region, cell_type=args.cell_type,
                exp_id=exp_id, log_base=args.log_base, epochs=args.epochs,
                patience=args.patience, device=device, seed=args.seed,
                quiet=not args.verbose, no_online_augment=args.no_online_augment,
                attention_heads=args.attention_heads,
            )
            f1, err = _run_one_trial(ns)
            row = {"trial": trial_id, "value": f1, "error": err or "", "exp_id": exp_id, **asdict(p)}
            _jsonl_append(jsonl_path, row)
            trial_rows.append(row)

            if optuna_study is not None:
                try:
                    _append_optuna_trial(
                        optuna_study, p, f1=f1, err=err, trial_id=trial_id,
                        exp_id=exp_id, source="llm", fixed_input_type=fixed_it,
                    )
                except Exception as e:
                    print(f"  Optuna append failed: {e}")

            if err:
                print(f"  LLM trial {trial_id} FAILED: {err[:200]}")
            else:
                print(f"  LLM trial {trial_id} F1={f1:.4f} [{p.input_type}/{p.model_type}]")
            trial_id += 1
            done += 1

    _best_f1_footer(trial_rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generalized autosearch: expression vs embedding, all regions"
    )
    all_known_regions = sorted({r for rs in REGIONS_BY_CELL_TYPE.values() for r in rs})
    parser.add_argument("--region", type=str, nargs="+", required=True,
                        help='One or more regions, or "all" to auto-select per cell_type.')
    parser.add_argument("--cell_type", type=str, nargs="+", default=["Astrocytes"],
                        choices=["Astrocytes", "Microglia", "MTG"],
                        help="One or more cell types (runs sequentially per type).")
    parser.add_argument("--input_type", type=str, nargs="+", default=None,
                        choices=sorted(INPUT_TYPES),
                        help="One or more input types to fix; omit to search all three.")
    parser.add_argument("--sampler", type=str, choices=["optuna", "random", "llm"], default="optuna")
    parser.add_argument(
        "--n_trials",
        type=int,
        default=20,
        help=(
            "Number of NEW trials Optuna will run. Trial numbers are global per study name in "
            "the DB (e.g. after 32 older trials, the next run starts at trial 32). "
            "Use --new_study to wipe this study name and start from 0."
        ),
    )
    parser.add_argument("--total_trials", type=int, default=None, help="Alias for --n_trials")
    parser.add_argument(
        "--new_study",
        action="store_true",
        help=(
            "Delete existing Optuna study with --study_name from this run's storage, then create "
            "a fresh study. Use when you want exactly --n_trials trials starting at number 0, "
            "without deleting the whole SQLite file (other studies in the same DB are kept)."
        ),
    )
    parser.add_argument("--study_name", type=str, default="autosearch_v2")
    parser.add_argument("--storage", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--log_base", type=str, default=os.path.join("ranking_model_runs", "autosearch_v2"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no_online_augment", action="store_true")
    parser.add_argument("--attention_heads", type=int, default=1)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--no_optuna", action="store_true")

    # LLM options
    llm = parser.add_argument_group("LLM (--sampler llm)")
    llm.add_argument("--proposals_per_round", type=int, default=3)
    llm.add_argument("--resume_jsonl", type=str, default=None)
    llm.add_argument("--history_max_trials", type=int, default=40)
    llm.add_argument("--model", type=str, default="gpt-4o-mini", dest="llm_model", metavar="MODEL")
    llm.add_argument("--api_key", type=str, default=None)
    llm.add_argument("--base_url", type=str, default=None)
    llm.add_argument("--system_prompt_file", type=str, default=None)
    llm.add_argument("--llm_retries", type=int, default=3)
    llm.add_argument("--llm_timeout", type=float, default=120.0)

    args = parser.parse_args()
    if args.total_trials is not None:
        args.n_trials = args.total_trials

    cell_types = args.cell_type          # list (nargs="+")
    input_types = args.input_type        # list or None
    raw_regions = args.region             # list (nargs="+")
    use_all_regions = len(raw_regions) == 1 and raw_regions[0].lower() == "all"

    combos: List[Tuple[str, str, Optional[str]]] = []
    for ct in cell_types:
        regions = REGIONS_BY_CELL_TYPE[ct] if use_all_regions else raw_regions
        if input_types is None:
            combos.extend((rg, ct, None) for rg in regions)
        else:
            combos.extend((rg, ct, it) for rg in regions for it in input_types)

    base_study = args.study_name
    base_storage = args.storage
    multi = len(combos) > 1

    print(f"Planned runs ({len(combos)}):")
    for i, (rg, ct, it) in enumerate(combos, 1):
        print(f"  {i}. region={rg}  cell_type={ct}  input_type={it or 'all'}")

    for rg, ct, it in combos:
        tag = f"{rg}_{ct}" + (f"_{it}" if it else "")
        args.region = rg
        args.cell_type = ct
        args.input_type = it
        args.study_name = f"{base_study}_{tag}" if multi else base_study
        args.storage = base_storage  # reset so _default_storage_url recalculates

        print(f"\n{'='*60}")
        print(f"  autosearch: region={rg}  cell_type={ct}"
              f"  input_type={it or 'all'}")
        print(f"  study={args.study_name}")
        print(f"{'='*60}\n")

        if args.sampler == "llm":
            run_llm_sampler(args)
        elif args.sampler == "optuna":
            run_optuna_sampler(args)
        else:
            run_random_sampler(args)


if __name__ == "__main__":
    main()
