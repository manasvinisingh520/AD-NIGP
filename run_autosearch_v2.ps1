# Generalized autosearch runner: random, optuna, and llm samplers for any region/cell_type.
# All three studies share one SQLite DB per region+cell_type for the Optuna dashboard.
#
# Usage (from repo root):
#   .\run_autosearch_v2.ps1
#   .\run_autosearch_v2.ps1 -Region EC -CellType Astrocytes -InputType expression
#   .\run_autosearch_v2.ps1 -Region ITG -CellType Microglia -NTrials 50 -Epochs 300

param(
    [string]$Region = "EC",
    [string]$CellType = "Astrocytes",
    [string]$InputType = "",
    [string]$LogBase = "",
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

if (-not $LogBase) {
    $LogBase = "ranking_model_runs/autosearch_v2_${Region}_${CellType}"
}

$dbPathPlanned = [System.IO.Path]::GetFullPath(
    (Join-Path (Join-Path $RepoRoot $LogBase) "${Region}_${CellType}_optuna.db")
)
Write-Host @"
IMPORTANT - Optuna Dashboard:
  DB: $dbPathPlanned
  Studies: rando, optuna, llm  (use the dashboard dropdown to switch)
"@ -ForegroundColor Yellow

$common = @(
    "src/autosearch.py",
    "--region", $Region,
    "--cell_type", $CellType,
    "--log_base", $LogBase,
    "--n_trials", "$NTrials",
    "--epochs", "$Epochs",
    "--patience", "$Patience",
    "--seed", "$Seed"
)
if ($InputType) { $common += @("--input_type", $InputType) }
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

Write-Host "Done. Optuna DB: $dbPathPlanned" -ForegroundColor Green
Write-Host "Dashboard: optuna-dashboard --server wsgiref sqlite:///$($dbPathPlanned.Replace('\', '/'))" -ForegroundColor Green
if (Test-Path $dbPathPlanned) {
    $dbUrl = "sqlite:///" + ($dbPathPlanned -replace "\\", "/")
    & python -c "import optuna; s=optuna.study.get_all_study_summaries(storage=r'$dbUrl'); print('Studies in DB:', [x.study_name for x in s])"
}
