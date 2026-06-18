# Autosearch v2 run directory

This folder holds **Phase 2** Optuna hyperparameter searches from `src/autosearch.py` (default `--log_base ranking_model_runs/autosearch_v2`). Each study writes artifacts here; per-trial `train_model_v2` logs live in subfolders named `autosearch_<study_name>_t<trial>`.

The table below is a **snapshot** of completed studies found via `*_best_params.json` in this directory. Trial counts come from the matching `*_trials.csv` (data rows only).

## Metrics (two F1s)

`train_model_v2` prints both in the **Grand Summary** of each per-trial log:

| Metric | Meaning |
|--------|---------|
| **Generalized F1** | Mean of per-fold test F1s (same quantity returned as `avg_f1` / what Optuna maximizes for `n_seeds=1`). |
| **Pooled held-out F1** | Single F1 computed on pooled predictions across held-out folds (see log line `Pooled held-out F1:`). |

**Optuna `best_value`** in `*_best_params.json` matches **Generalized F1** (rounded) for the best trial. Pooled F1 is **not** stored in JSON/CSV; the table below takes it from `autosearch_<study>_t<best_trial>/train_model_v2_*.log`.

## Layout (per study)

| Artifact | Pattern |
|----------|---------|
| Best trial summary | `{study_name}_best_params.json` |
| All trials (CSV) | `{study_name}_trials.csv` |
| All trials (JSONL) | `{study_name}_trials.jsonl` |
| Optuna SQLite (default) | `{region}_{cell_type}_optuna.db` (under this same directory) |
| Per-trial training log | `autosearch_{study_name}_t{N}/train_model_v2_*.log` |

See `doc/EXPERIMENT_README.md` and `.cursor/skills/v2-experiment-protocol/SKILL.md` for the full pipeline.

## Studies present in this folder (snapshot)

| Study name | Generalized F1 | Pooled held-out F1 | Best trial | Input / model | Trials (CSV) | Optuna DB |
|------------|------------------|--------------------|------------|---------------|--------------|-----------|
| `v2_EC_Astrocytes_expr` | 0.6014 | 0.5650 | 26 | expression / mlp | 20 | `EC_Astrocytes_optuna.db` |
| `v2_EC_Astrocytes_emb` | 0.5875 | 0.5644 | 22 | embedding / flatten_mlp | 47 | `EC_Astrocytes_optuna.db` |
| `v2_ITG_Astrocytes_expr` | 0.5701 | 0.5312 | 23 | expression / mlp | 20 | `ITG_Astrocytes_optuna.db` |
| `v2_ITG_Astrocytes_emb` | 0.5868 | 0.5499 | 42 | embedding / flatten_mlp | 65 | `ITG_Astrocytes_optuna.db` |
| `v2_PFC_Astrocytes_expr` | 0.5205 | 0.5273 | 87 | expression / mlp | 50 | `PFC_Astrocytes_optuna.db` |
| `v2_PFC_Astrocytes_emb` | 0.5889 | 0.5821 | 39 | embedding / flatten_mlp | 65 | `PFC_Astrocytes_optuna.db` |
| `v2_V2_Astrocytes_expr` | 0.5486 | 0.5528 | 43 | expression / mlp | 20 | `V2_Astrocytes_optuna.db` |
| `v2_V2_Astrocytes_emb` | 0.6111 | 0.5380 | 36 | embedding / flatten_mlp | 101 | `V2_Astrocytes_optuna.db` |
| `v2_V1_Astrocytes_expr` | 0.5326 | 0.4869 | 31 | expression / mlp | 20 | `V1_Astrocytes_optuna.db` |
| `v2_V1_Astrocytes_emb` | 0.5396 | 0.5082 | 37 | embedding / mil | 20 | `V1_Astrocytes_optuna.db` |
| `v2_CrossRegion_Microglia_expr` | 0.3562 | 0.3702 | 0 | expression / svm | 225 | `CrossRegion_Microglia_optuna.db` |
| `v2_ITG_Microglia_expr` | 0.5236 | 0.5441 | 0 | expression / svm | 250 | `ITG_Microglia_optuna.db` |
| `v2_PFC_Microglia_expr` | 0.5507 | 0.5524 | 13 | expression / svm | 225 | `PFC_Microglia_optuna.db` |
| `autosearch_v2` (Microglia · ITG · embedding, partial) | 0.7792 | 0.7710 | 6 | embedding / mil | 8 / 50 partial | `ITG_Microglia_optuna.db` |
| `autosearch_v2` (Microglia · PFC · embedding, partial) | 0.7208 | 0.7194 | 19 | embedding / mil | 21 / 50 partial | `PFC_Microglia_optuna.db` |
| `autosearch_v2` (Microglia · CrossRegion · embedding, partial) | 0.6521 | 0.6837 | 13 | embedding / mil | 16 / 50 partial | `CrossRegion_Microglia_optuna.db` |
| `autosearch_v2_EC_Astrocytes_expression` | 0.6056 | 0.5787 | 92 | expression / mlp | 100 | `EC_Astrocytes_optuna.db` |
| `v2b_EC_Astrocytes_expression` | 0.5014 | 0.5333 | 49 | expression / mlp | 50 | `EC_Astrocytes_optuna.db` |
| `autosearch_v2` | 0.6271 | 0.5998 | 47 | expression / svm | 1044 | `PFC_Microglia_optuna.db` |
| `v2b` | 0.5236 | 0.5441 | 5 | expression / svm | 1028 | `ITG_Microglia_optuna.db` |

**Notes**

- Generalized F1 in the table is taken from the log (4 decimals) so it aligns with the Grand Summary; it matches `best_value` in JSON up to float rounding.
- Fold-wise **std** for Generalized F1 is in the same log line (`Generalized F1: mean +/- std`).
- Several studies can share the same `{region}_{cell_type}_optuna.db`; Optuna distinguishes runs by **study name**. Use `optuna-dashboard` and pick the study from the dropdown.
- Rows named `autosearch_v2` and `v2b` use **generic** study names while pointing at **Microglia** DBs—prefer explicit names like `v2_<REGION>_<CELL>_<expr|emb>` for new work.
- The three **Microglia · … · embedding, partial** rows are **in-progress** regional sweeps (default study name `autosearch_v2`, 50 trials targeted). **Generalized** and **Pooled** F1 are from the **best completed trial so far** (Grand Summary in `autosearch_autosearch_v2_t<best>/train_model_v2_*.log` with `argv` showing `Microglia` and `input_type=embedding`). There is no separate `*_best_params.json` for these yet; trial counts are **completed embedding trials with `STATUS: completed`** in those folders, not a clean CSV slice.
- Full hyperparameters for Phase 3 are in each `*_best_params.json` under `trial_params`.

## Example commands (recreate or extend)

Single slice (expression only, 50 trials):

```bash
python src/autosearch.py --region EC --cell_type Astrocytes --input_type expression \
  --sampler optuna --n_trials 50 --study_name v2_EC_Astrocytes_expr
```

Many slices in one process (both cell types, all regions per type, expression + embedding, 100 trials each study):

```bash
python src/autosearch.py --region all --cell_type Astrocytes Microglia \
  --input_type expression embedding --sampler optuna --n_trials 100
```

Fresh Optuna study on existing storage (wipes same-named study if present):

```bash
python src/autosearch.py ... --new_study
```

## Refreshing the metrics table

After new runs, re-parse Grand Summary lines from the **best trial** log for each `*_best_params.json`. From repo root (adjust `base` if you moved the folder):

```python
import json, glob, os, re

base = "ranking_model_runs/autosearch_v2"
gen_re = re.compile(r"Generalized F1:\s*([\d.]+)\s*\+/-")
pool_re = re.compile(r"Pooled held-out F1:\s*([\d.]+)")

for jp in sorted(glob.glob(os.path.join(base, "*_best_params.json"))):
    d = json.load(open(jp))
    if d.get("error"):
        continue
    sn, bt = d["study_name"], d["best_trial"]
    sub = os.path.join(base, f"autosearch_{sn}_t{bt}")
    logs = sorted(glob.glob(os.path.join(sub, "train_model_v2_*.log")))
    if not logs:
        print(sn, "NO_LOG")
        continue
    txt = open(logs[-1], encoding="utf-8", errors="replace").read()
    g = gen_re.search(txt)
    p = pool_re.search(txt)
    print(sn, g.group(1) if g else "?", p.group(1) if p else "?")
```

Paste the printed pairs back into the markdown table.
