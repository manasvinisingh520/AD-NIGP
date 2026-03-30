# EC autonomous search (autoresearch-style)

Inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch): a **fixed training entrypoint** (`train_model.main`), a **single objective** (mean leave-one-fold-out **F1** on Thal), and comparable **epoch/patience** budgets per trial. This repo does **not** auto-edit Python model code; the driver only **samples hyperparameters** and logs results.

## Objective and constraints

- **Region:** `EC`
- **Cell type:** `Astrocytes`
- **`loss_ptau`:** `none` (Thal classification drives training loss for this study)
- **Metric:** `train_model.main` → **average F1** across test folds (same as your experiment CSVs). Training always runs with `best_f1=True` inside the driver so the saved checkpoint aligns with the F1 objective.
- **TensorBoard:** Autosearch sets `no_tensorboard=True` on each trial so no `SummaryWriter` event files are created (faster, less disk I/O). Normal `python src/train_model.py` still logs to TensorBoard unless you pass `--no_tensorboard`.

## Setup

```bash
pip install -r requirements-autosearch.txt
```

- **`--sampler optuna`** (default), **`--sampler random`**, and **`--sampler llm`** all log trials to the same SQLite file by default: `{log_base}/ec_optuna.db` (unless you pass **`--no_optuna`**, which skips Optuna entirely for random and LLM—no `optuna` package needed). The **Optuna study name** inside that file is **`--study_name`** (use different study names to keep sampler runs separate in one DB).
- For **`--sampler random`** or **`--sampler llm`** with SQLite logging, `optuna` is required (same install as above).

## Run (Optuna, recommended)

```bash
python src/autosearch_ec.py --n_trials 30 --log_base ranking_model_runs/autosearch_ec --epochs 200 --patience 30
```

Artifacts under `--log_base`:

- `ec_optuna.db` — shared SQLite storage (default path under `--log_base`; override with `--storage` / `--optuna_storage`). Each **`--study_name`** is a separate Optuna study in that file when not `--no_optuna`.
- `{study_name}_trials.jsonl` — one JSON object per trial (this run only)
- `{study_name}_trials.csv` — flat table for spreadsheets
- `{study_name}_best_params.json` — best value + `trial_params` dict

## Optuna Dashboard

Install the web UI, then point it at **`{log_base}/ec_optuna.db`** by default (use a **forward-slash** URL; three slashes after `sqlite:` for an absolute path).

```bash
pip install optuna-dashboard
```

On **Windows**, if `gunicorn` is installed the dashboard may default to it and crash with `No module named 'fcntl'` (Unix-only). Pass **`--server wsgiref`** so it uses the stdlib server (fine for local use). The command below includes that flag.

**One database** holds sampler-driven trials, random-search appends, and LLM appends (when SQLite logging is enabled). In the dashboard, trial **`user_attrs`** include **`source`**: `optuna` (native sampler), `random`, or `llm`.

```bash
optuna-dashboard --server wsgiref sqlite:///I:/sf2026/ranking_model_runs/autosearch_ec/ec_optuna.db
```

Replace with your `--log_base` path and drive letter. Override the file with **`--storage`** or **`--optuna_storage`** (see Useful flags).

**Multiple studies in one file:** the default DB is always **`ec_optuna.db`**. If you use different `--study_name` values across runs (e.g. `rando`, `optuna`, `llm`), each name is a separate study **inside that same file**. In Optuna Dashboard, use the **study selector / dropdown** to switch—do not expect a single merged list unless you reused one study name. If you only see one study, you are likely opening an **old** database (for example a legacy `*_optuna.db` path) instead of **`{log_base}/ec_optuna.db`**.

Open the URL printed in the terminal (default **http://127.0.0.1:8080**). Disable SQLite for **`--sampler llm`** or random with **`--no_optuna`**.

### Troubleshooting (`no such table: version_info`)

That error means SQLite opened the file, but it is **not** a valid Optuna RDB (missing Optuna schema). Typical causes:

1. **Empty or corrupt DB** — delete the broken `.db` and run autosearch again so `optuna.create_study` recreates it (you will lose that study’s history in that file).
2. **Not an Optuna database** — another program created a `.db` with the same name; rename/delete and recreate via autosearch.

### Troubleshooting (`No module named 'fcntl'` on Windows)

`optuna-dashboard` may pick **gunicorn** when it is installed; gunicorn does not run on Windows. Use:

```bash
optuna-dashboard --server wsgiref sqlite:///I:/path/to/your_study.db
```

## Run (random search)

```bash
python src/autosearch_ec.py --sampler random --n_trials 50 --seed 1 --log_base ranking_model_runs/autosearch_ec_rand
```

By default this **also appends** each trial to `{log_base}/ec_optuna.db` (same as the Optuna sampler). Use **`--no_optuna`** to skip SQLite and avoid the `optuna` dependency.

## LLM-in-the-loop (`--sampler llm`)

An OpenAI-compatible API proposes **batches** of trials; each JSON object is validated by `trial_params_from_llm_dict` in `autosearch_ec.py`, then `train_model.main` runs like the Optuna path (EC, `loss_ptau=none`, mean LOFO F1, TensorBoard off).

1. **Install:** `pip install -r requirements-autosearch-llm.txt` (adds `openai`, `python-dotenv`, etc.).
2. **API key:** put `OPENAI_API_KEY=...` in project-root `.env`, or export it. For **Ollama:** `--base_url http://localhost:11434/v1` and `--api_key ollama` (or any non-empty string).
3. **Example:**

```bash
python src/autosearch_ec.py --sampler llm --n_trials 24 --proposals_per_round 3 --study_name ec_llm --log_base ranking_model_runs/autosearch_ec_llm --epochs 150 --patience 25 --model gpt-4o-mini
```

**LLM artifacts:** `{study_name}_trials.jsonl` (append; use `--resume_jsonl` to continue), `{study_name}_llm_rounds.jsonl` (raw replies), and the **same** `{log_base}/ec_optuna.db` as other modes (unless `--no_optuna`). Override storage with **`--storage`** / **`--optuna_storage`**.

**Prompts:** `--system_prompt_file` replaces the built-in system prompt. The model must return a **single JSON array** of flat objects (trial configs). Main fields: `gene_encoder`, `attention`, `pooling`, `pca_components`, `mlp_hidden_sizes` (comma-separated widths for baseline; clamped), `n_hidden_layers`, `hidden_dim_3layer`, `dropout`, `weight_decay`, `lr`, `batch_size`, `lambda_entropy` (gene encoder only), aug / mixup / cutmix / `label_smoothing` / `thal_ordinal` — same semantics as `suggest_trial_optuna` (see source for allowed values).

## Useful flags

| Flag | Effect |
|------|--------|
| `--no_online_augment` | Forces all `--aug_*` / mixup / cutmix strengths to 0 in the trial (still uses base pickle) |
| `--data_suffix` | Same as `train_model` (avoid `_augScaling`; prefer online aug) |
| `--device cuda` | Override auto device |
| `--study_name` | Optuna study name in the RDB + prefix for jsonl/csv/best_params artifacts |
| `-v` / `--verbose` | Print full `train_model` LOFO / data-load logs (default is quiet; trial lines from autosearch still print) |
| `--no_optuna` | Skip SQLite logging for `--sampler random` and `--sampler llm` |
| `--sampler` | `optuna` (default), `random`, or `llm` |
| `--storage` | Optuna storage URL (first choice; default `sqlite:///…/<log_base>/ec_optuna.db`) |
| `--optuna_storage` | Alternative storage URL if `--storage` unset |

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
