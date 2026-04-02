# DGE2 Batch Runner Script

PowerShell script to run `dge2.py` with multiple parameter combinations.

## Usage

**Run from PowerShell:**
```powershell
.\run_dge2_batch.ps1
```

Or right-click the file and select "Run with PowerShell"

## Parameters

The script runs the following combinations:

- **Bins**: 100, 250, 500 (cells per pseudobulk bin)
- **Regions**: EC, ITG, PFC, V2, V1
- **Seeds**: 10 random seeds (7-digit numbers)

**Total runs**: 3 bins × 5 regions × 10 seeds = **150 runs**

## Output

### Results
Results are saved to: `results/dge5/`
- Format: `dge_results_{region}_bins{bins}_seed{seed}_{group1}_vs_{group2}.csv`

### Logs
Logs are saved to: `logs/dge2_batch_{timestamp}/`
- Each run has its own log file: `dge2_{region}_bins{bins}_seed{seed}.log`
- Error logs: `dge2_{region}_bins{bins}_seed{seed}_error.log`

## Example Output

```
============================================================
DGE2 Batch Runner
============================================================
Bins: 100, 250, 500
Regions: EC, ITG, PFC, V2, V1
Seeds: 1234567, 2345678, 3456789, ...
Total runs: 150
Log directory: logs\dge2_batch_20250101_120000
============================================================

[1/150] Running: region=EC, bins=100, seed=1234567
  ✓ Success

[2/150] Running: region=EC, bins=100, seed=2345678
  ✓ Success
...
```

## Notes

- The script runs sequentially (one at a time)
- Each run may take several minutes depending on data size
- Total runtime: ~150 runs × average time per run
- You can stop the script at any time (Ctrl+C)
- Failed runs are logged with error codes

## Customization

To modify parameters, edit `run_dge2_batch.ps1`:

```powershell
$bins = @(100, 250, 500)  # Change bins here
$regions = @("EC", "ITG", "PFC", "V2", "V1")  # Change regions here
# Seeds are randomly generated - modify the random seed or range in the script
```

