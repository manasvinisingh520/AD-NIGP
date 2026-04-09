---
name: v2-experiment-protocol
description: >-
  Runs the sf2026 v2 embedding-vs-expression pipeline (create_data_v2, autosearch,
  train_model_v2) across phases in doc/EXPERIMENT_README.md Step 4, per region and
  per cell type. Produces comparable summaries and a per-region results table.
  Use when the user asks to run the full experiment protocol, all regions for
  Astrocytes/Microglia/MTG, Phase 2-4 autosearch + seeded training, or to compare
  expression vs embedding vs random_embedding results.
---

# v2 experiment protocol (Step 4)

Authoritative reference: doc/EXPERIMENT_README.md Step 4.

## Critical setup

- **Python:** Always use `C:\Users\Manish\.conda\envs\sf2026\python.exe` (Python 3.10). System Python is 3.7 and cannot read the v2 pickles (protocol 5).
- **Working directory:** All commands from repo root `sf2026/`.
- **SQLite contention:** Expression and embedding autosearch for the **same region** share one Optuna DB. Never run them in parallel. Run expression first, then embedding per region. Different regions CAN run in parallel.

## Region sets by cell type

| cell_type  | regions                     |
|------------|-----------------------------|
| Astrocytes | EC, ITG, PFC, V2, V1       |
| Microglia  | CrossRegion, ITG, PFC       |
| MTG        | MTG                         |

## Phase 1 -- Data generation (skip if pickles exist)

Check `data/model_data/{region}[_{cell_type}]_v2_data_log1p.pkl` for each region.

```bash
python src/create_data_v2.py --cell_type Astrocytes
python src/create_data_v2.py --cell_type Microglia
```

## Phase 2 -- Autosearch (per region, expression then embedding)

For each (region, cell_type), run expression THEN embedding (sequential to avoid DB lock):

```bash
python src/autosearch.py --region REGION --cell_type CELL \
  --input_type expression --sampler optuna --n_trials N \
  --study_name v2_REGION_CELL_expr

python src/autosearch.py --region REGION --cell_type CELL \
  --input_type embedding --sampler optuna --n_trials N \
  --study_name v2_REGION_CELL_emb
```

**Parallelism:** Different regions can run simultaneously (separate DBs). Same region expr+emb must be sequential.

Best params written to: `ranking_model_runs/autosearch_v2/{study_name}_best_params.json`

## Phase 3 -- Seeded training

Use `scripts/run_phase3.py` which reads all best_params JSONs and runs train_model_v2 with n_seeds=10:

```bash
python scripts/run_phase3.py --cell_type Astrocytes --cell_type Microglia [--include_random] [--dry-run]
```

The script maps JSON params to CLI flags automatically. Logs go to `runs/`.

## Phase 4 -- Report

1. Grep `runs/train_model_v2_*.log` for "Grand Summary", "Generalized F1", "Pooled held-out F1"
2. Build per-region table: Region | Expression F1 | Embedding F1
3. Write report to `doc/reports/{cell_type}_v2_experiment_report.md`

## Agent execution pattern

When user says "run for CELL_TYPE" or "run for all":

```
[ ] Confirm cell types and regions from table above
[ ] Phase 1: verify pickles exist; run create_data_v2 if missing
[ ] Phase 2: for each region, run expr autosearch then emb autosearch (sequentially per region, parallel across regions)
[ ] Phase 3: python scripts/run_phase3.py --cell_type X
[ ] Phase 4: collect Grand Summary from logs, write comparison report
```

Key lessons from previous runs:
- MIL model on large regions (V1) can hang for 30+ min per trial; kill and use proxy params from another region if stuck
- Microglia "Thal" column contains CERAD strings (none/sparse/moderate/frequent), mapped to 0-3 in create_data_v2.py
- All 4 CERAD classes for Microglia vs 6 Thal classes for Astrocytes; train_model_v2 auto-detects class count
