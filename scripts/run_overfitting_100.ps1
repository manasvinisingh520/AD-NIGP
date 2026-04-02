# Run run_overfitting_experiments.py 100 times and collect results
# Output format: geFalse_attnFalse_mae_pca16_highest_lambda_entropy  F1=0.3028, MSE=0.1621, MAE=0.2823

$numRuns = 100
$outputFile = "overfitting_100_runs.txt"

# Clear/create output file
"" | Out-File -FilePath $outputFile -Encoding utf8
Write-Host "Running $numRuns times. Results -> $outputFile"
Write-Host ""

for ($i = 1; $i -le $numRuns; $i++) {
    Write-Host "[$i/$numRuns] Running..." -NoNewline
    $logBase = "ec_junk_run$i"
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = "python"
    $psi.Arguments = "-u .\src\run_overfitting_experiments.py --region EC --log_base $logBase --epochs 200"
    $psi.UseShellExecute = $false
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.WorkingDirectory = $PSScriptRoot
    $p = [System.Diagnostics.Process]::Start($psi)
    $outputStr = $p.StandardOutput.ReadToEnd() + "`n" + $p.StandardError.ReadToEnd()
    $p.WaitForExit()
    # Strip ANSI escape codes (e.g. from train_model)
    $outputStr = $outputStr -replace '\x1b\[[0-9;]*m', ''

    $lines = @()
    # Pattern 1: Best model per combo (from run_overfitting_experiments)
    $matches = [regex]::Matches($outputStr, '(\S+)\s+F1=([\d.]+),\s*MSE=([\d.]+),\s*MAE=([\d.]+)')
    foreach ($m in $matches) {
        $runName = $m.Groups[1].Value.Trim()
        $f1 = $m.Groups[2].Value
        $mse = $m.Groups[3].Value
        $mae = $m.Groups[4].Value
        $lines += "$runName  F1=$f1, MSE=$mse, MAE=$mae"
    }
    # Pattern 2: Average F1/MSE/MAE from train_model (fallback)
    if ($lines.Count -eq 0) {
        $mF1 = [regex]::Match($outputStr, 'Average F1:\s*([\d.]+)')
        $mMse = [regex]::Match($outputStr, 'Average MSE:\s*([\d.]+)')
        $mMae = [regex]::Match($outputStr, 'Average MAE:\s*([\d.]+)')
        if ($mF1.Success -and $mMse.Success -and $mMae.Success) {
            $lines += "geFalse_attnFalse_mae_pca16_highest_lambda_entropy  F1=$($mF1.Groups[1].Value), MSE=$($mMse.Groups[1].Value), MAE=$($mMae.Groups[1].Value)"
        }
    }

    if ($lines.Count -gt 0) {
        $runResult = "Run $i : " + ($lines -join " | ")
        $runResult | Add-Content -Path $outputFile
        Write-Host " done -> " -NoNewline
        Write-Host $lines[0]
    } else {
        "Run $i : (no result parsed)" | Add-Content -Path $outputFile
        Write-Host " (no result parsed)"
        if ($i -eq 1) {
            $outputStr | Out-File -FilePath "debug_run1_output.txt" -Encoding utf8
            Write-Host "`n  (Run 1 raw output saved to debug_run1_output.txt for inspection)"
        }
    }
}

Write-Host ""
Write-Host "Done. Results saved to $outputFile"
