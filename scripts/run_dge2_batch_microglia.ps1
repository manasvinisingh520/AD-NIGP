# PowerShell script to run dge2.py for Microglia with multiple parameter combinations
# Runs for bins [100, 250, 500], regions [EC, ITG, PFC, V2, V1, CrossRegion], and 1 seed

# Verify we're in the correct directory
if (-not (Test-Path ".\src\dge2.py")) {
    Write-Host "Error: Cannot find .\src\dge2.py" -ForegroundColor Red
    Write-Host "Please run this script from the project root directory." -ForegroundColor Red
    exit 1
}

# Parameters
$cell_type = "Microglia"
$bins = @(100)
$regions = @("EC", "ITG", "PFC", "V2", "V1", "CrossRegion")
#$regions = @("V2")

# Use a single seed
$seeds = @(100)  # Single seed value

# Count total runs
$totalRuns = $bins.Count * $regions.Count * $seeds.Count
$currentRun = 0

$separator = "============================================================"
Write-Host $separator
Write-Host "DGE2 Batch Runner - Microglia"
Write-Host $separator
Write-Host "Cell Type: $cell_type"
Write-Host "Bins: $($bins -join ', ')"
Write-Host "Regions: $($regions -join ', ')"
Write-Host "Seeds: $($seeds -join ', ')"
Write-Host "Total runs: $totalRuns"
Write-Host $separator
Write-Host ""

# Create logs directory
$logDir = "logs\dge2_batch_microglia_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
Write-Host "Log directory: $logDir"
Write-Host ""

# Track start time
$startTime = Get-Date

# Loop through all combinations
foreach ($bin in $bins) {
    foreach ($region in $regions) {
        foreach ($seed in $seeds) {
            $currentRun++
            
            # Create log file for this run
            $logFileName = "dge2_microglia_{0}_bins{1}_seed{2}.log" -f $region, $bin, $seed
            $logFile = Join-Path $logDir $logFileName
            $errorLogFileName = "dge2_microglia_{0}_bins{1}_seed{2}_error.log" -f $region, $bin, $seed
            $errorLogFile = Join-Path $logDir $errorLogFileName
            
            Write-Host "[$currentRun/$totalRuns] Running: cell_type=$cell_type, region=$region, bins=$bin, seed=$seed"
            
            # Run the command and capture output
            try {
                # Run command and redirect both stdout and stderr to log file
                $process = Start-Process -FilePath "python" `
                    -ArgumentList ".\src\dge2.py", "-cell_type", $cell_type, "-region", $region, "-seed", $seed, "-bins", $bin `
                    -NoNewWindow `
                    -Wait `
                    -PassThru `
                    -RedirectStandardOutput $logFile `
                    -RedirectStandardError $errorLogFile
                
                if ($process.ExitCode -eq 0) {
                    Write-Host "  [OK] Success" -ForegroundColor Green
                } else {
                    Write-Host "  [FAIL] Failed with exit code $($process.ExitCode)" -ForegroundColor Red
                }
            }
            catch {
                Write-Host "  [ERROR] Error: $_" -ForegroundColor Red
            }
            
            Write-Host ""
        }
    }
}

# Calculate elapsed time
$endTime = Get-Date
$elapsed = $endTime - $startTime

Write-Host $separator
Write-Host "Batch run completed!"
Write-Host "Total runs: $totalRuns"
Write-Host "Elapsed time: $($elapsed.Hours)h $($elapsed.Minutes)m $($elapsed.Seconds)s"
Write-Host "Log directory: $logDir"
Write-Host $separator

