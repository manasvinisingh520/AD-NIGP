# Runs 20 trials each: random, optuna, and llm samplers into the same SQLite file
# (default: <LogBase>/ec_optuna.db) with separate Optuna study names: rando, optuna, llm.
#
# Prerequisites: pip install -r requirements-autosearch.txt and requirements-autosearch-llm.txt
# LLM: set OPENAI_API_KEY (or .env) or pass -BaseUrl / -ApiKey for a local OpenAI-compatible server.
#
# Usage (from repo root): .\run_autosearch_ec_three_samplers.ps1
# Optional: .\run_autosearch_ec_three_samplers.ps1 -LogBase "ranking_model_runs/my_ec_run" -Epochs 200

param(
    [string]$LogBase = "ranking_model_runs/autosearch_ec_1",
    [int]$NTrials = 100,
    [int]$Epochs = 200,
    [int]$Patience = 30,
    [int]$Seed = 0,
    [string]$LlmModel = "gpt-4o-mini",
    [int]$ProposalsPerRound = 3,
    [string]$BaseUrl = "",
    [string]$ApiKey = ""
)

$ErrorActionPreference = "Stop"
$RepoRoot = $PSScriptRoot
Set-Location $RepoRoot

$dbPathPlanned = [System.IO.Path]::GetFullPath((Join-Path (Join-Path $RepoRoot $LogBase) "ec_optuna.db"))
Write-Host @"
IMPORTANT — Optuna Dashboard:
  Open THIS file only: $dbPathPlanned
  (Older runs may have created *other* .db files; they are NOT updated by this script.)
  This DB holds three studies (rando, optuna, llm). Use the dashboard's study dropdown to switch.
"@ -ForegroundColor Yellow

$common = @(
    "src/autosearch_ec.py",
    "--log_base", $LogBase,
    "--n_trials", "$NTrials",
    "--epochs", "$Epochs",
    "--patience", "$Patience",
    "--seed", "$Seed"
)
if ($BaseUrl) { $common += @("--base_url", $BaseUrl) }
if ($ApiKey) { $common += @("--api_key", $ApiKey) }

Write-Host "=== random sampler, study_name=rando ===" -ForegroundColor Cyan
& python @common --sampler random --study_name rando
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "=== optuna sampler, study_name=optuna ===" -ForegroundColor Cyan
& python @common --sampler optuna --study_name optuna
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "=== llm sampler, study_name=llm (needs API) ===" -ForegroundColor Cyan
& python @common --sampler llm --study_name llm --model $LlmModel --proposals_per_round $ProposalsPerRound
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$dbPath = $dbPathPlanned
Write-Host "Done. Optuna DB: $dbPath" -ForegroundColor Green
Write-Host "Dashboard: optuna-dashboard --server wsgiref sqlite:///$($dbPath.Replace('\', '/'))" -ForegroundColor Green
if (Test-Path $dbPath) {
    $dbUrl = "sqlite:///" + ($dbPath -replace "\\", "/")
    & python -c "import optuna; s=optuna.study.get_all_study_summaries(storage=r'$dbUrl'); print('Studies in DB:', [x.study_name for x in s])"
}
