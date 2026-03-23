# EC autonomous search (autoresearch-style)

Inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch): a **fixed training entrypoint** (`train_model.main`), a **single objective** (mean leave-one-fold-out **F1** on Thal), and comparable **epoch/patience** budgets per trial. This repo does **not** auto-edit Python model code; the driver only **samples hyperparameters** and logs results.

## Objective and constraints

- **Region:** `EC`
- **Cell type:** `Astrocytes`
- **`loss_ptau`:** `none` (Thal classification drives training loss for this study)
- **Metric:** `train_model.main` → **average F1** across test folds (same as your experiment CSVs). Training always runs with `best_f1=True` inside the driver so the saved checkpoint aligns with the F1 objective.

## Setup

```bash
pip install -r requirements-autosearch.txt
```

(`optuna` is only needed for `--sampler optuna`. For zero extra deps use `--sampler random`.)

## Run (Optuna, recommended)

```bash
python src/autosearch_ec.py --n_trials 30 --log_base ranking_model_runs/autosearch_ec --epochs 200 --patience 30
```

Artifacts under `--log_base`:

- `{study_name}_optuna.db` — SQLite study (resume with same `--study_name` + path)
- `{study_name}_trials.jsonl` — one JSON object per trial (this run only)
- `{study_name}_trials.csv` — flat table for spreadsheets
- `{study_name}_best_params.json` — best value + `trial_params` dict

## Run (random search, no Optuna)

```bash
python src/autosearch_ec.py --sampler random --n_trials 50 --seed 1 --log_base ranking_model_runs/autosearch_ec_rand
```

## Useful flags

| Flag | Effect |
|------|--------|
| `--no_online_augment` | Forces all `--aug_*` / mixup / cutmix strengths to 0 in the trial (still uses base pickle) |
| `--data_suffix` | Same as `train_model` (avoid `_augScaling`; prefer online aug) |
| `--device cuda` | Override auto device |
| `--study_name` | Optuna study name + file prefix |

## Search space (edit in code)

All sampling lives in `src/autosearch_ec.py`: `suggest_trial_optuna` / `suggest_trial_random`.

Rules enforced there:

- **`gene_encoder=True`:** `pooling=None`, `attention=False`, `lambda_entropy` sampled (used by gene encoder).
- **Baseline:** `pooling` in `mean` / `max` / `none` (3D when `none`), `attention` sampled, `mlp_hidden_sizes` from a small menu.
- **3D inputs** (gene encoder or `pooling=none`): gene dropout, gene CutMix prob; **2D pooled:** feature dropout instead.
- **`thal_ordinal=True`:** mixup and gene CutMix forced off (matches `train_model.train_epoch`).

## Extending the search

1. Add categories or ranges in `suggest_trial_optuna` (and mirror in `suggest_trial_random`).
2. Add CLI passthroughs only if you need them from shell; otherwise keep the space in one file for agents/humans to diff.
3. For much faster **screening**, reduce `--epochs` / `--patience` (same for all trials so rankings stay comparable).

## Data notes

- Use **base** `data/model_data/EC_data_for_model_training.pkl` with online `--aug_*` rather than offline `_augScaling` pickles (see `train_model` warnings).
- After changing pickles, use `train_model --recompute` once if patient-level cache is stale.
