"""
Autoresearch-style hyperparameter search for EC, loss_ptau=none, objective mean LOFO F1.

Calls train_model.main (fixed entrypoint); does not edit model code.
- Default: Optuna suggests trials (or --sampler random).
- --llm: OpenAI-compatible API proposes trial batches; same JSONL / optional Optuna append as before.

Usage:
  pip install -r requirements-autosearch.txt
  python src/autosearch_ec.py --n_trials 20 --log_base ranking_model_runs/autosearch_ec --epochs 150 --patience 25

  pip install -r requirements-autosearch-llm.txt
  python src/autosearch_ec.py --llm --n_trials 12 --proposals_per_round 3 --model gpt-4o-mini
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
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_model import _parse_mlp_hidden_sizes, main as train_main

REGION = "EC"
CELL_TYPE = "Astrocytes"
LOSS_PTAU = "none"

MLP_MENU = ["", "64,32,16", "128,64,32", "256,128,64,32"]
PCA_CHOICES = [2, 4, 8, 16]
POOLING_BASELINE = ["mean", "max", "none"]

# LLM-proposed MLP: same syntax as train_model --mlp_hidden_sizes; clamped for safety.
MLP_LLM_MAX_LAYERS = 24
MLP_LLM_MAX_WIDTH = 8192


@dataclass
class TrialParams:
    """Serializable trial configuration (for JSON/CSV)."""

    gene_encoder: bool
    attention: bool
    pooling: Optional[str]  # None for 3D / gene_encoder; "mean" or "max" for pooled baseline
    pca_components: int
    mlp_hidden_sizes: str
    n_hidden_layers: int
    hidden_dim_3layer: int
    dropout: float
    weight_decay: float
    lr: float
    lambda_entropy: float
    batch_size: int
    aug_scale_jitter: float
    aug_per_gene_scale_jitter: float
    aug_gene_dropout: float
    aug_emb_dim_dropout: float
    aug_feature_dropout: float
    aug_gene_block_cutout_max: int
    aug_noise_std: float
    mixup_alpha: float
    gene_cutmix_prob: float
    gene_cutmix_alpha: float
    label_smoothing: float
    thal_ordinal: bool


def _pooling_for_namespace(gene_encoder: bool, pooling_choice: str) -> Any:
    if gene_encoder:
        return None
    if pooling_choice == "none":
        return None
    return pooling_choice


def _nearest_in_sorted(x: float, choices: List[float]) -> float:
    return min(choices, key=lambda c: abs(c - x))


def _nearest_int(x: int, choices: List[int]) -> int:
    return min(choices, key=lambda c: abs(c - x))


def _normalize_llm_mlp_string(s: str) -> str:
    """Parse comma-separated layer widths; clamp depth and width; empty if invalid."""
    parsed = _parse_mlp_hidden_sizes(s)
    if not parsed:
        return ""
    clamped = [max(1, min(MLP_LLM_MAX_WIDTH, h)) for h in parsed[:MLP_LLM_MAX_LAYERS]]
    return ",".join(str(x) for x in clamped)


def trial_params_from_llm_dict(raw: Dict[str, Any]) -> TrialParams:
    """
    Build TrialParams from an LLM-produced JSON object.
    Clamps values to the same allowed sets as Optuna search; enforces architecture rules.
    """
    g = {k.replace("-", "_"): v for k, v in raw.items()}

    def b(key: str, default: bool = False) -> bool:
        v = g.get(key, default)
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.strip().lower() in ("1", "true", "yes")
        return bool(v)

    def f(key: str, default: float) -> float:
        try:
            return float(g.get(key, default))
        except (TypeError, ValueError):
            return default

    def i(key: str, default: int) -> int:
        try:
            return int(g.get(key, default))
        except (TypeError, ValueError):
            return default

    def s(key: str, default: str) -> str:
        v = g.get(key, default)
        if v is None:
            return default
        return str(v).strip().lower()

    def raw_str(key: str, default: str) -> str:
        v = g.get(key, default)
        if v is None:
            return default
        return str(v).strip()

    gene_encoder = b("gene_encoder", False)
    attention = b("attention", False)
    pooling_choice = s("pooling", "mean").lower()
    if pooling_choice not in POOLING_BASELINE:
        pooling_choice = "mean"

    lambda_entropy = f("lambda_entropy", 1e-3)
    lambda_entropy = max(1e-4, min(0.2, lambda_entropy))

    if gene_encoder:
        attention = False
        pooling_choice = "none"
    else:
        lambda_entropy = 1e-3

    pca_components = i("pca_components", 8)
    pca_components = _nearest_int(pca_components, PCA_CHOICES)

    mlp_val = g.get("mlp_hidden_sizes", "")
    if isinstance(mlp_val, list):
        try:
            mlp_raw = ",".join(str(int(x)) for x in mlp_val)
        except (TypeError, ValueError):
            mlp_raw = ""
    else:
        mlp_raw = raw_str("mlp_hidden_sizes", "") if mlp_val is not None else ""
    if gene_encoder:
        mlp_hidden_sizes = ""
    else:
        mlp_hidden_sizes = _normalize_llm_mlp_string(mlp_raw)

    n_hidden_layers = i("n_hidden_layers", 1)
    n_hidden_layers = 1 if gene_encoder else (n_hidden_layers if n_hidden_layers in (1, 3) else 1)

    hidden_dim_3layer = i("hidden_dim_3layer", 32)
    if gene_encoder:
        hidden_dim_3layer = 32
    else:
        hidden_dim_3layer = _nearest_int(hidden_dim_3layer, [32, 64, 128])

    dropout = max(0.2, min(0.55, f("dropout", 0.3)))
    wd = f("weight_decay", 0.1)
    weight_decay = _nearest_in_sorted(wd, [0.1, 0.5, 1.0])
    lr = max(1e-4, min(3e-3, f("lr", 1e-3)))
    batch_size = i("batch_size", 8)
    batch_size = _nearest_int(batch_size, [4, 8, 16])

    aug_scale = _nearest_in_sorted(f("aug_scale_jitter", 0.0), [0.0, 0.05, 0.1, 0.15])
    aug_pg = _nearest_in_sorted(f("aug_per_gene_scale_jitter", 0.0), [0.0, 0.05, 0.1])
    aug_ed = _nearest_in_sorted(f("aug_emb_dim_dropout", 0.0), [0.0, 0.05, 0.1])
    aug_noise = _nearest_in_sorted(f("aug_noise_std", 0.0), [0.0, 0.01, 0.02, 0.03])
    aug_block = i("aug_gene_block_cutout_max", 0)
    aug_block = _nearest_int(aug_block, [0, 32, 128, 256])

    is_3d = gene_encoder or (pooling_choice == "none")
    if is_3d:
        aug_gene_dropout = _nearest_in_sorted(f("aug_gene_dropout", 0.0), [0.0, 0.05, 0.1, 0.15])
        aug_feature_dropout = 0.0
        gene_cutmix_prob = _nearest_in_sorted(f("gene_cutmix_prob", 0.0), [0.0, 0.25, 0.5])
    else:
        aug_gene_dropout = 0.0
        aug_feature_dropout = _nearest_in_sorted(f("aug_feature_dropout", 0.0), [0.0, 0.05, 0.1, 0.15])
        gene_cutmix_prob = 0.0

    mixup_alpha = _nearest_in_sorted(f("mixup_alpha", 0.0), [0.0, 0.2, 0.4])
    gene_cutmix_alpha = max(0.5, min(2.0, f("gene_cutmix_alpha", 1.0)))
    label_smoothing = _nearest_in_sorted(f("label_smoothing", 0.0), [0.0, 0.05, 0.1])
    thal_ordinal = b("thal_ordinal", False)
    if thal_ordinal:
        mixup_alpha = 0.0
        gene_cutmix_prob = 0.0

    return TrialParams(
        gene_encoder=gene_encoder,
        attention=attention,
        pooling=_pooling_for_namespace(gene_encoder, pooling_choice),
        pca_components=pca_components,
        mlp_hidden_sizes=mlp_hidden_sizes,
        n_hidden_layers=n_hidden_layers,
        hidden_dim_3layer=hidden_dim_3layer,
        dropout=dropout,
        weight_decay=weight_decay,
        lr=lr,
        lambda_entropy=lambda_entropy if gene_encoder else 1e-3,
        batch_size=batch_size,
        aug_scale_jitter=aug_scale,
        aug_per_gene_scale_jitter=aug_pg,
        aug_gene_dropout=aug_gene_dropout,
        aug_emb_dim_dropout=aug_ed,
        aug_feature_dropout=aug_feature_dropout,
        aug_gene_block_cutout_max=aug_block,
        aug_noise_std=aug_noise,
        mixup_alpha=mixup_alpha,
        gene_cutmix_prob=gene_cutmix_prob,
        gene_cutmix_alpha=gene_cutmix_alpha,
        label_smoothing=label_smoothing,
        thal_ordinal=thal_ordinal,
    )


def suggest_trial_optuna(trial) -> TrialParams:
    """Sample one configuration (Optuna trial). Always suggest same param names for storage compatibility."""
    gene_encoder = trial.suggest_categorical("gene_encoder", [False, True])
    attention_s = trial.suggest_categorical("attention", [False, True])
    pooling_choice_s = trial.suggest_categorical("pooling", POOLING_BASELINE)
    lambda_entropy_s = trial.suggest_float("lambda_entropy", 1e-4, 0.2, log=True)

    if gene_encoder:
        attention = False
        pooling_choice = "none"
        lambda_entropy = float(lambda_entropy_s)
    else:
        attention = bool(attention_s)
        pooling_choice = str(pooling_choice_s)
        lambda_entropy = 1e-3

    pca_components = trial.suggest_categorical("pca_components", PCA_CHOICES)
    mlp_hidden_sizes = trial.suggest_categorical("mlp_hidden_sizes", MLP_MENU)
    n_hidden_layers = trial.suggest_categorical("n_hidden_layers", [1, 3])
    hidden_dim_3layer = trial.suggest_categorical("hidden_dim_3layer", [32, 64, 128])

    dropout = trial.suggest_float("dropout", 0.2, 0.55)
    weight_decay = trial.suggest_categorical("weight_decay", [0.1, 0.5, 1.0])
    lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [4, 8, 16])

    aug_scale_jitter = trial.suggest_categorical("aug_scale_jitter", [0.0, 0.05, 0.1, 0.15])
    aug_per_gene_scale_jitter = trial.suggest_categorical("aug_per_gene_scale_jitter", [0.0, 0.05, 0.1])
    aug_emb_dim_dropout = trial.suggest_categorical("aug_emb_dim_dropout", [0.0, 0.05, 0.1])
    aug_noise_std = trial.suggest_categorical("aug_noise_std", [0.0, 0.01, 0.02, 0.03])
    aug_gene_block_cutout_max = trial.suggest_categorical("aug_gene_block_cutout_max", [0, 32, 128, 256])

    is_3d = gene_encoder or (pooling_choice == "none")
    if is_3d:
        aug_gene_dropout = trial.suggest_categorical("aug_gene_dropout", [0.0, 0.05, 0.1, 0.15])
        aug_feature_dropout = 0.0
        gene_cutmix_prob = trial.suggest_categorical("gene_cutmix_prob", [0.0, 0.25, 0.5])
    else:
        aug_gene_dropout = 0.0
        aug_feature_dropout = trial.suggest_categorical("aug_feature_dropout", [0.0, 0.05, 0.1, 0.15])
        gene_cutmix_prob = 0.0

    mixup_alpha = trial.suggest_categorical("mixup_alpha", [0.0, 0.2, 0.4])
    gene_cutmix_alpha = trial.suggest_float("gene_cutmix_alpha", 0.5, 2.0)
    label_smoothing = trial.suggest_categorical("label_smoothing", [0.0, 0.05, 0.1])
    thal_ordinal = trial.suggest_categorical("thal_ordinal", [False, True])

    if thal_ordinal:
        mixup_alpha = 0.0
        gene_cutmix_prob = 0.0

    return TrialParams(
        gene_encoder=gene_encoder,
        attention=attention,
        pooling=_pooling_for_namespace(gene_encoder, pooling_choice),
        pca_components=pca_components,
        mlp_hidden_sizes=mlp_hidden_sizes if not gene_encoder else "",
        n_hidden_layers=n_hidden_layers if not gene_encoder else 1,
        hidden_dim_3layer=hidden_dim_3layer if not gene_encoder else 32,
        dropout=dropout,
        weight_decay=weight_decay,
        lr=lr,
        lambda_entropy=lambda_entropy,
        batch_size=batch_size,
        aug_scale_jitter=aug_scale_jitter,
        aug_per_gene_scale_jitter=aug_per_gene_scale_jitter,
        aug_gene_dropout=aug_gene_dropout,
        aug_emb_dim_dropout=aug_emb_dim_dropout,
        aug_feature_dropout=aug_feature_dropout,
        aug_gene_block_cutout_max=aug_gene_block_cutout_max,
        aug_noise_std=aug_noise_std,
        mixup_alpha=mixup_alpha,
        gene_cutmix_prob=gene_cutmix_prob,
        gene_cutmix_alpha=gene_cutmix_alpha,
        label_smoothing=label_smoothing,
        thal_ordinal=thal_ordinal,
    )


def suggest_trial_random(rng) -> TrialParams:
    """Match Optuna search space approximately with stdlib random."""
    gene_encoder = rng.random() < 0.5
    if gene_encoder:
        attention = False
        pooling_choice = "none"
        lambda_entropy = 10 ** rng.uniform(-4, math.log10(0.2))
    else:
        attention = rng.random() < 0.5
        pooling_choice = rng.choice(POOLING_BASELINE)
        lambda_entropy = 1e-3

    pca_components = rng.choice(PCA_CHOICES)
    mlp_hidden_sizes = "" if gene_encoder else rng.choice(MLP_MENU)
    n_hidden_layers = 1 if gene_encoder else rng.choice([1, 3])
    hidden_dim_3layer = 32 if gene_encoder else rng.choice([32, 64, 128])

    dropout = rng.uniform(0.2, 0.55)
    weight_decay = rng.choice([0.1, 0.5, 1.0])
    lr = 10 ** rng.uniform(-4, math.log10(3e-3))
    batch_size = rng.choice([4, 8, 16])

    aug_scale_jitter = rng.choice([0.0, 0.05, 0.1, 0.15])
    aug_per_gene_scale_jitter = rng.choice([0.0, 0.05, 0.1])
    aug_emb_dim_dropout = rng.choice([0.0, 0.05, 0.1])
    aug_noise_std = rng.choice([0.0, 0.01, 0.02, 0.03])
    aug_gene_block_cutout_max = rng.choice([0, 32, 128, 256])

    is_3d = gene_encoder or (pooling_choice == "none")
    if is_3d:
        aug_gene_dropout = rng.choice([0.0, 0.05, 0.1, 0.15])
        aug_feature_dropout = 0.0
        gene_cutmix_prob = rng.choice([0.0, 0.25, 0.5])
    else:
        aug_gene_dropout = 0.0
        aug_feature_dropout = rng.choice([0.0, 0.05, 0.1, 0.15])
        gene_cutmix_prob = 0.0

    mixup_alpha = rng.choice([0.0, 0.2, 0.4])
    gene_cutmix_alpha = rng.uniform(0.5, 2.0)
    label_smoothing = rng.choice([0.0, 0.05, 0.1])
    thal_ordinal = rng.random() < 0.25
    if thal_ordinal:
        mixup_alpha = 0.0
        gene_cutmix_prob = 0.0

    return TrialParams(
        gene_encoder=gene_encoder,
        attention=attention,
        pooling=_pooling_for_namespace(gene_encoder, pooling_choice),
        pca_components=pca_components,
        mlp_hidden_sizes=mlp_hidden_sizes,
        n_hidden_layers=n_hidden_layers,
        hidden_dim_3layer=hidden_dim_3layer,
        dropout=dropout,
        weight_decay=weight_decay,
        lr=lr,
        lambda_entropy=lambda_entropy,
        batch_size=batch_size,
        aug_scale_jitter=aug_scale_jitter,
        aug_per_gene_scale_jitter=aug_per_gene_scale_jitter,
        aug_gene_dropout=aug_gene_dropout,
        aug_emb_dim_dropout=aug_emb_dim_dropout,
        aug_feature_dropout=aug_feature_dropout,
        aug_gene_block_cutout_max=aug_gene_block_cutout_max,
        aug_noise_std=aug_noise_std,
        mixup_alpha=mixup_alpha,
        gene_cutmix_prob=gene_cutmix_prob,
        gene_cutmix_alpha=gene_cutmix_alpha,
        label_smoothing=label_smoothing,
        thal_ordinal=thal_ordinal,
    )


def params_to_namespace(
    p: TrialParams,
    *,
    exp_id: str,
    log_base: str,
    epochs: int,
    patience: int,
    device: str,
    data_suffix: str,
    no_online_augment: bool,
    attention_heads: int,
    quiet: bool = False,
) -> SimpleNamespace:
    """Build train_model args. Always best_f1=True; no_tensorboard=True (no event files under log_base)."""
    return SimpleNamespace(
        region=REGION,
        cell_type=CELL_TYPE,
        pooling=p.pooling,
        loss_ptau=LOSS_PTAU,
        batch_size=p.batch_size,
        epochs=epochs,
        lr=p.lr,
        device=device,
        dropout=p.dropout,
        weight_decay=p.weight_decay,
        attention=p.attention,
        attention_heads=attention_heads,
        pca_components=p.pca_components,
        gene_encoder=p.gene_encoder,
        lambda_entropy=p.lambda_entropy,
        log_base=log_base,
        exp_id=exp_id,
        verbose=False,
        n_hidden_layers=p.n_hidden_layers,
        hidden_dim_3layer=p.hidden_dim_3layer,
        mlp_hidden_sizes=p.mlp_hidden_sizes,
        patience=patience,
        best_f1=True,
        thal_ordinal=p.thal_ordinal,
        top_genes=None,
        ranked_genes_path=None,
        save_best_model=False,
        save_dir="models",
        data_suffix=data_suffix,
        recompute=False,
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
        label_smoothing=p.label_smoothing,
        no_tensorboard=True,
        q=quiet,
    )


def _run_one_trial(ns: SimpleNamespace) -> Tuple[float, Optional[str]]:
    try:
        avg_f1, _, _, _ = train_main(ns)
        if avg_f1 is None or (isinstance(avg_f1, float) and math.isnan(avg_f1)):
            return float("-inf"), "invalid_f1"
        return float(avg_f1), None
    except Exception as e:
        return float("-inf"), f"{type(e).__name__}: {e}\n{traceback.format_exc()}"


def _best_f1_from_trial_rows(
    rows: List[Dict[str, Any]],
) -> Tuple[Optional[float], Optional[Any], Optional[str]]:
    """Best mean F1 among rows with no error and finite value (this process only)."""
    best_v: Optional[float] = None
    best_trial: Optional[Any] = None
    best_exp: Optional[str] = None
    for r in rows:
        if r.get("error"):
            continue
        try:
            v = float(r["value"])
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isfinite(v) or v <= float("-inf"):
            continue
        if best_v is None or v > best_v:
            best_v = v
            best_trial = r.get("trial")
            best_exp = r.get("exp_id")
    return best_v, best_trial, best_exp


def _print_best_f1_run_footer(rows: List[Dict[str, Any]]) -> None:
    v, tid, eid = _best_f1_from_trial_rows(rows)
    parts: List[str] = []
    if tid is not None:
        parts.append(f"trial {tid}")
    if eid:
        parts.append(f"exp_id={eid}")
    suffix = f" ({', '.join(parts)})" if parts else ""
    if v is None:
        print("\nBest F1 (this run): none" + suffix)
    else:
        print(f"\nBest F1 (this run): {v:.4f}{suffix}")


def export_trials_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    keys = sorted(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


DEFAULT_LLM_SYSTEM_PROMPT = """You optimize hyperparameters for a PyTorch embedding classifier on brain region EC (Astrocytes).
Objective: maximize mean leave-one-fold-out F1 on Thal stage classification. Training uses loss_ptau=none (only Thal loss).

Output rules:
- Reply with ONE JSON array only (no markdown fences required but allowed). Each element is one trial config (flat object).
- Use only these pooling values for baseline (non gene_encoder): "mean", "max", "none".
- If gene_encoder is true: set attention false and pooling "none"; lambda_entropy is used (1e-4 to 0.2).
- If gene_encoder is false: lambda_entropy in JSON is ignored by the trainer; still you may omit it.
- If thal_ordinal is true: mixup_alpha and gene_cutmix_prob must be 0.
- For 3D input (gene_encoder true OR pooling "none"): use aug_gene_dropout / gene_cutmix_prob; set aug_feature_dropout 0.
- For 2D pooled input (pooling mean or max): use aug_feature_dropout; set aug_gene_dropout and gene_cutmix_prob 0.
- pca_components must be one of: 2, 4, 8, 16.
- mlp_hidden_sizes (baseline only; omit or "" if gene_encoder): comma-separated positive integers, layer widths from input toward the heads, e.g. "512,256,128,64" or "8192,4096,2048,1024,512". You choose depth and widths; server clamps each width to [1, MAX_WIDTH] and at most MAX_LAYERS layers (see user message for numbers). Prefer tapering (wide to narrow) for large flattened inputs.
- batch_size one of 4, 8, 16. weight_decay one of 0.1, 0.5, 1.0.

Be creative: explore near the best trials, try boundary settings, and diversify. Propose exactly N configs where N is given in the user message."""


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


def _next_jsonl_trial_id(rows: List[Dict[str, Any]]) -> int:
    best = -1
    for r in rows:
        try:
            best = max(best, int(r.get("trial", -1)))
        except (TypeError, ValueError):
            pass
    return best + 1


def _format_llm_history(rows: List[Dict[str, Any]], max_rows: int) -> str:
    tail = rows[-max_rows:] if len(rows) > max_rows else rows
    lines = []
    for r in tail:
        err = r.get("error") or ""
        v = r.get("value")
        brief = {
            "trial": r.get("trial"),
            "value": v,
            "error": (err[:120] + "…") if len(str(err)) > 120 else err,
            "gene_encoder": r.get("gene_encoder"),
            "attention": r.get("attention"),
            "pooling": r.get("pooling"),
            "pca_components": r.get("pca_components"),
            "mlp_hidden_sizes": r.get("mlp_hidden_sizes"),
            "lr": r.get("lr"),
            "dropout": r.get("dropout"),
            "weight_decay": r.get("weight_decay"),
            "batch_size": r.get("batch_size"),
            "thal_ordinal": r.get("thal_ordinal"),
        }
        lines.append(json.dumps(brief, default=str))
    return "\n".join(lines) if lines else "(no prior trials)"


def _extract_llm_json_array(text: str) -> List[Dict[str, Any]]:
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


def _call_llm_chat(
    *,
    system: str,
    user: str,
    model: str,
    api_key: str,
    base_url: Optional[str],
    timeout: float,
) -> str:
    from openai import OpenAI

    resolved_base = base_url or "https://api.openai.com/v1"
    print(f"Calling LLM (chat.completions): model={model!r}, base_url={resolved_base!r}, timeout={timeout}s")

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
    msg = resp.choices[0].message
    text = (msg.content or "").strip()
    print(f"LLM response received ({len(text)} chars).")
    return text


def _optuna_params_and_distributions_from_trial_params(
    p: TrialParams,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Match suggest_trial_optuna search space for Optuna dashboard compatibility."""
    from optuna.distributions import CategoricalDistribution, FloatDistribution

    pooling_choice = "none" if p.pooling is None else str(p.pooling)
    if pooling_choice not in POOLING_BASELINE:
        pooling_choice = "none"

    mlp = "" if p.gene_encoder else (p.mlp_hidden_sizes if p.mlp_hidden_sizes in MLP_MENU else "")

    params: Dict[str, Any] = {
        "gene_encoder": bool(p.gene_encoder),
        "attention": bool(p.attention),
        "pooling": pooling_choice,
        "lambda_entropy": float(p.lambda_entropy),
        "pca_components": int(p.pca_components),
        "mlp_hidden_sizes": mlp,
        "n_hidden_layers": int(p.n_hidden_layers),
        "hidden_dim_3layer": int(p.hidden_dim_3layer),
        "dropout": float(p.dropout),
        "weight_decay": float(p.weight_decay),
        "lr": float(p.lr),
        "batch_size": int(p.batch_size),
        "aug_scale_jitter": float(p.aug_scale_jitter),
        "aug_per_gene_scale_jitter": float(p.aug_per_gene_scale_jitter),
        "aug_emb_dim_dropout": float(p.aug_emb_dim_dropout),
        "aug_noise_std": float(p.aug_noise_std),
        "aug_gene_block_cutout_max": int(p.aug_gene_block_cutout_max),
        "mixup_alpha": float(p.mixup_alpha),
        "gene_cutmix_alpha": float(p.gene_cutmix_alpha),
        "label_smoothing": float(p.label_smoothing),
        "thal_ordinal": bool(p.thal_ordinal),
    }
    distributions: Dict[str, Any] = {
        "gene_encoder": CategoricalDistribution((False, True)),
        "attention": CategoricalDistribution((False, True)),
        "pooling": CategoricalDistribution(tuple(POOLING_BASELINE)),
        "lambda_entropy": FloatDistribution(1e-4, 0.2, log=True),
        "pca_components": CategoricalDistribution(tuple(PCA_CHOICES)),
        "mlp_hidden_sizes": CategoricalDistribution(tuple(MLP_MENU)),
        "n_hidden_layers": CategoricalDistribution((1, 3)),
        "hidden_dim_3layer": CategoricalDistribution((32, 64, 128)),
        "dropout": FloatDistribution(0.2, 0.55),
        "weight_decay": CategoricalDistribution((0.1, 0.5, 1.0)),
        "lr": FloatDistribution(1e-4, 3e-3, log=True),
        "batch_size": CategoricalDistribution((4, 8, 16)),
        "aug_scale_jitter": CategoricalDistribution((0.0, 0.05, 0.1, 0.15)),
        "aug_per_gene_scale_jitter": CategoricalDistribution((0.0, 0.05, 0.1)),
        "aug_emb_dim_dropout": CategoricalDistribution((0.0, 0.05, 0.1)),
        "aug_noise_std": CategoricalDistribution((0.0, 0.01, 0.02, 0.03)),
        "aug_gene_block_cutout_max": CategoricalDistribution((0, 32, 128, 256)),
        "mixup_alpha": CategoricalDistribution((0.0, 0.2, 0.4)),
        "gene_cutmix_alpha": FloatDistribution(0.5, 2.0),
        "label_smoothing": CategoricalDistribution((0.0, 0.05, 0.1)),
        "thal_ordinal": CategoricalDistribution((False, True)),
    }

    is_3d = bool(p.gene_encoder) or pooling_choice == "none"
    if is_3d:
        params["aug_gene_dropout"] = float(p.aug_gene_dropout)
        params["gene_cutmix_prob"] = float(p.gene_cutmix_prob)
        distributions["aug_gene_dropout"] = CategoricalDistribution((0.0, 0.05, 0.1, 0.15))
        distributions["gene_cutmix_prob"] = CategoricalDistribution((0.0, 0.25, 0.5))
    else:
        params["aug_feature_dropout"] = float(p.aug_feature_dropout)
        distributions["aug_feature_dropout"] = CategoricalDistribution((0.0, 0.05, 0.1, 0.15))

    return params, distributions


def _append_llm_trial_to_optuna(
    study: Any,
    p: TrialParams,
    *,
    f1: float,
    err: Optional[str],
    trial_id: int,
    exp_id: str,
) -> None:
    from optuna.trial import TrialState, create_trial

    params, distributions = _optuna_params_and_distributions_from_trial_params(p)
    user_attrs: Dict[str, Any] = {
        "source": "llm",
        "jsonl_trial": trial_id,
        "exp_id": exp_id,
        "trial_params": asdict(p),
    }
    if not p.gene_encoder and p.mlp_hidden_sizes:
        dash_mlp = str(params.get("mlp_hidden_sizes", ""))
        if p.mlp_hidden_sizes != dash_mlp:
            user_attrs["mlp_hidden_sizes_llm"] = p.mlp_hidden_sizes

    if err:
        user_attrs["error"] = err[:4000]

    v = float(f1)
    ok = not err and math.isfinite(v) and v > float("-inf")

    if ok:
        trial = create_trial(
            state=TrialState.COMPLETE,
            value=v,
            params=params,
            distributions=distributions,
            user_attrs=user_attrs,
        )
    else:
        trial = create_trial(
            state=TrialState.FAIL,
            params=params,
            distributions=distributions,
            user_attrs=user_attrs,
        )
    study.add_trial(trial)


def run_llm_autosearch(args: argparse.Namespace) -> None:
    """LLM proposes batches of configs; append JSONL; optional Optuna SQLite for dashboard."""
    try:
        from dotenv import load_dotenv as _load_dotenv
    except ImportError:
        _load_dotenv = None
    if _load_dotenv is not None:
        _repo_root = Path(__file__).resolve().parent.parent
        _load_dotenv(_repo_root / ".env")
        _load_dotenv()

    import numpy as np
    import torch

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.log_base, exist_ok=True)
    jsonl_path = args.resume_jsonl or os.path.join(args.log_base, f"{args.study_name}_trials.jsonl")
    llm_log_path = os.path.join(args.log_base, f"{args.study_name}_llm_rounds.jsonl")

    optuna_study = None
    optuna_storage: Optional[str] = None
    if not args.no_optuna:
        try:
            import optuna
        except ImportError as e:
            raise SystemExit(
                "Optuna is required for LLM runs unless you pass --no_optuna. "
                "Install: pip install -r requirements-autosearch.txt"
            ) from e
        db_path = os.path.join(os.path.abspath(args.log_base), f"{args.study_name}_llm_optuna.db")
        optuna_storage = args.optuna_storage or ("sqlite:///" + db_path.replace("\\", "/"))
        oname = args.optuna_study_name if args.optuna_study_name is not None else args.study_name
        optuna_study = optuna.create_study(
            study_name=oname,
            storage=optuna_storage,
            direction="maximize",
            load_if_exists=True,
        )
        print(f"LLM Optuna logging: storage={optuna_storage!r}, study_name={oname!r} (e.g. optuna-dashboard)")

    if args.system_prompt_file:
        system_prompt = Path(args.system_prompt_file).read_text(encoding="utf-8")
    else:
        system_prompt = DEFAULT_LLM_SYSTEM_PROMPT

    api_key = args.api_key if args.api_key is not None else os.environ.get("OPENAI_API_KEY")
    if not api_key and args.base_url:
        api_key = "ollama"
    elif not api_key:
        raise SystemExit("Set OPENAI_API_KEY or pass --api_key (use any string with local --base_url).")

    existing = _load_jsonl(jsonl_path)
    trial_id = _next_jsonl_trial_id(existing)
    done = 0
    n_total = args.n_trials
    trial_rows: List[Dict[str, Any]] = []

    while done < n_total:
        n_prop = min(args.proposals_per_round, n_total - done)
        history = _load_jsonl(jsonl_path)
        hist_txt = _format_llm_history(history, args.history_max_trials)
        user_msg = (
            f"Region={REGION}, cell_type={CELL_TYPE}, loss_ptau={LOSS_PTAU}.\n"
            f"Prior trials (compact, newest last):\n{hist_txt}\n\n"
            f"Propose exactly {n_prop} new configurations as a JSON array of {n_prop} objects.\n"
            f"MLP safety caps: at most {MLP_LLM_MAX_LAYERS} layers, each width at most {MLP_LLM_MAX_WIDTH}.\n"
            f"Allowed reference: pca {PCA_CHOICES}, pooling {POOLING_BASELINE}."
        )

        proposals: Optional[List[Dict[str, Any]]] = None
        last_raw = ""
        print(f"\n--- LLM round: need {n_prop} new config(s); {done}/{n_total} trials done ---")
        for attempt in range(args.llm_retries):
            try:
                if attempt > 0:
                    print(f"  LLM retry {attempt + 1}/{args.llm_retries}...")
                last_raw = _call_llm_chat(
                    system=system_prompt,
                    user=user_msg,
                    model=args.llm_model,
                    api_key=api_key,
                    base_url=args.base_url,
                    timeout=args.llm_timeout,
                )
                proposals = _extract_llm_json_array(last_raw)
                if len(proposals) >= 1:
                    break
            except Exception as e:
                last_raw = f"<error {attempt}: {e}>"
                time.sleep(2**attempt)

        _jsonl_append(
            llm_log_path,
            {
                "utc": datetime.now(timezone.utc).isoformat(),
                "trial_start": trial_id,
                "n_requested": n_prop,
                "raw_preview": (last_raw[:2000] + "…") if len(last_raw) > 2000 else last_raw,
                "parsed_count": len(proposals) if proposals else 0,
            },
        )

        if not proposals:
            raise SystemExit("LLM returned no valid JSON array after retries.")

        proposals = proposals[:n_prop]

        for raw_cfg in proposals:
            if done >= n_total:
                break
            p = trial_params_from_llm_dict(raw_cfg)
            exp_id = f"autosearch_llm_{args.study_name}_t{trial_id}"
            ns = params_to_namespace(
                p,
                exp_id=exp_id,
                log_base=args.log_base,
                epochs=args.epochs,
                patience=args.patience,
                device=device,
                data_suffix=args.data_suffix,
                no_online_augment=args.no_online_augment,
                attention_heads=args.attention_heads,
                quiet=args.q,
            )
            f1, err = _run_one_trial(ns)
            row = {
                "trial": trial_id,
                "value": f1,
                "error": err or "",
                "exp_id": exp_id,
                "llm_round": True,
                **asdict(p),
            }
            _jsonl_append(jsonl_path, row)
            trial_rows.append(row)
            print(f"  Trial {trial_id} F1={f1:.4f}" + (f" ERR={str(err)[:80]}" if err else ""))
            if optuna_study is not None:
                try:
                    _append_llm_trial_to_optuna(
                        optuna_study,
                        p,
                        f1=f1,
                        err=err,
                        trial_id=trial_id,
                        exp_id=exp_id,
                    )
                except Exception as e:
                    print(f"  Optuna append failed (trial {trial_id}): {type(e).__name__}: {e}")
            trial_id += 1
            done += 1

    tail = f"\nFinished {done} LLM trials. Log: {jsonl_path}"
    if optuna_storage is not None:
        tail += f"\nLLM Optuna DB: {optuna_storage}"
    print(tail)
    _print_best_f1_run_footer(trial_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="EC autosearch: maximize mean F1 (loss_ptau=none)")
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument(
        "--total_trials",
        type=int,
        default=None,
        help="Alias for --n_trials (e.g. LLM runs)",
    )
    parser.add_argument("--study_name", type=str, default="ec_f1_none")
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Optuna storage URL (sampler=optuna); default sqlite under --log_base",
    )
    parser.add_argument("--sampler", type=str, choices=["optuna", "random"], default="optuna")
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Propose trials via OpenAI-compatible API instead of Optuna/random sampler",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--log_base", type=str, default=os.path.join("ranking_model_runs", "autosearch_ec"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data_suffix", type=str, default="")
    parser.add_argument("--no_online_augment", action="store_true")
    parser.add_argument("--attention_heads", type=int, default=1)
    parser.add_argument(
        "--q",
        action="store_true",
        dest="q",
        help="Suppress train_model LOFO/data-load stdout (trial / LLM lines from this script still print)",
    )
    llm = parser.add_argument_group("LLM (--llm only)")
    llm.add_argument("--proposals_per_round", type=int, default=3)
    llm.add_argument(
        "--resume_jsonl",
        type=str,
        default=None,
        help="Append to this jsonl (default: log_base/<study_name>_trials.jsonl)",
    )
    llm.add_argument("--history_max_trials", type=int, default=40)
    llm.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        dest="llm_model",
        metavar="MODEL",
        help="LLM model id (OpenAI-compatible)",
    )
    llm.add_argument("--api_key", type=str, default=None, help="Default: env OPENAI_API_KEY")
    llm.add_argument("--base_url", type=str, default=None, help="OpenAI-compatible base URL (e.g. Ollama)")
    llm.add_argument("--system_prompt_file", type=str, default=None)
    llm.add_argument("--llm_retries", type=int, default=3)
    llm.add_argument("--llm_timeout", type=float, default=120.0)
    llm.add_argument(
        "--no_optuna",
        action="store_true",
        help="With --llm: do not append trials to SQLite *_llm_optuna.db",
    )
    llm.add_argument(
        "--optuna_storage",
        type=str,
        default=None,
        help="With --llm: Optuna storage URL; default sqlite .../<study_name>_llm_optuna.db",
    )
    llm.add_argument(
        "--optuna_study_name",
        type=str,
        default=None,
        help="With --llm: study name in RDB (default: same as --study_name)",
    )
    args = parser.parse_args()
    if args.total_trials is not None:
        args.n_trials = args.total_trials

    if args.llm and args.sampler == "random":
        raise SystemExit("--llm cannot be used with --sampler random.")

    if args.llm:
        run_llm_autosearch(args)
        return

    import random

    import numpy as np
    import torch

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.log_base, exist_ok=True)
    # Fresh JSONL per invocation (Optuna DB still holds history if load_if_exists).
    jsonl_path = os.path.join(args.log_base, f"{args.study_name}_trials.jsonl")
    open(jsonl_path, "w", encoding="utf-8").close()
    csv_path = os.path.join(args.log_base, f"{args.study_name}_trials.csv")
    best_json_path = os.path.join(args.log_base, f"{args.study_name}_best_params.json")

    if args.sampler == "optuna":
        try:
            import optuna
        except ImportError as e:
            raise SystemExit(
                "Optuna is required for --sampler optuna. Install: pip install -r requirements-autosearch.txt"
            ) from e

        storage = args.storage or f"sqlite:///{os.path.join(args.log_base, args.study_name + '_optuna.db')}"
        study = optuna.create_study(
            study_name=args.study_name,
            storage=storage,
            direction="maximize",
            load_if_exists=True,
        )

        trial_rows: List[Dict[str, Any]] = []

        def objective(trial: optuna.Trial) -> float:
            p = suggest_trial_optuna(trial)
            trial.set_user_attr("trial_params", asdict(p))
            exp_id = f"autosearch_{args.study_name}_t{trial.number}"
            ns = params_to_namespace(
                p,
                exp_id=exp_id,
                log_base=args.log_base,
                epochs=args.epochs,
                patience=args.patience,
                device=device,
                data_suffix=args.data_suffix,
                no_online_augment=args.no_online_augment,
                attention_heads=args.attention_heads,
                quiet=args.q,
            )
            f1, err = _run_one_trial(ns)
            row = {
                "trial": trial.number,
                "value": f1,
                "error": err or "",
                "exp_id": exp_id,
                **asdict(p),
            }
            trial_rows.append(row)
            with open(jsonl_path, "a", encoding="utf-8") as jf:
                jf.write(json.dumps(row, default=str) + "\n")
            if err:
                print(f"  Trial {trial.number} FAILED: {err[:200]}...")
            else:
                print(f"  Trial {trial.number} F1={f1:.4f}")
            return f1

        study.optimize(objective, n_trials=args.n_trials)

        export_trials_csv(csv_path, trial_rows)
        try:
            best = study.best_trial
            best_payload = {
                "study_name": args.study_name,
                "storage": storage,
                "best_value": best.value,
                "best_trial": best.number,
                "optuna_params": study.best_params,
                "trial_params": best.user_attrs.get("trial_params", {}),
                "exported_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            with open(best_json_path, "w", encoding="utf-8") as f:
                json.dump(best_payload, f, indent=2, default=str)
        except ValueError:
            best_payload = {
                "study_name": args.study_name,
                "error": "no_completed_trials",
                "exported_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            with open(best_json_path, "w", encoding="utf-8") as f:
                json.dump(best_payload, f, indent=2, default=str)
            print("\nNo successful Optuna trials (all failed or -inf).")
        print(f"Wrote {best_json_path}, {csv_path}, {jsonl_path}")
        _print_best_f1_run_footer(trial_rows)

    else:
        trial_rows = []
        best_f1 = float("-inf")
        best_payload = None
        for i in range(args.n_trials):
            p = suggest_trial_random(rng)
            exp_id = f"autosearch_{args.study_name}_r{i}"
            ns = params_to_namespace(
                p,
                exp_id=exp_id,
                log_base=args.log_base,
                epochs=args.epochs,
                patience=args.patience,
                device=device,
                data_suffix=args.data_suffix,
                no_online_augment=args.no_online_augment,
                attention_heads=args.attention_heads,
                quiet=args.q,
            )
            f1, err = _run_one_trial(ns)
            row = {
                "trial": i,
                "value": f1,
                "error": err or "",
                "exp_id": exp_id,
                **asdict(p),
            }
            trial_rows.append(row)
            with open(jsonl_path, "a", encoding="utf-8") as jf:
                jf.write(json.dumps(row, default=str) + "\n")
            if f1 > best_f1 and err is None:
                best_f1 = f1
                best_payload = {"best_value": f1, "trial": i, "params": asdict(p), "exp_id": exp_id}
            print(f"  Random trial {i} F1={f1:.4f}" + (f" err={err[:80]}" if err else ""))

        export_trials_csv(csv_path, trial_rows)
        if best_payload:
            with open(best_json_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "study_name": args.study_name,
                        "sampler": "random",
                        "seed": args.seed,
                        **best_payload,
                    },
                    f,
                    indent=2,
                    default=str,
                )
        print(f"\nWrote {best_json_path}, {csv_path}, {jsonl_path}")
        _print_best_f1_run_footer(trial_rows)


if __name__ == "__main__":
    main()
