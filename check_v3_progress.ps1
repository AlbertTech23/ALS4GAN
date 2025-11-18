# V3 Training Progress Monitor
# Run this script to check training progress

Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host "  V3 Training Progress Monitor" -ForegroundColor Cyan
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host ""

# Check if training is running
$process = Get-Process | Where-Object {$_.CommandLine -like "*train_s4gan_salak_v3.py*"}
if ($process) {
    Write-Host "✓ Training is RUNNING" -ForegroundColor Green
    Write-Host "  PID: $($process.Id)" -ForegroundColor Yellow
    Write-Host "  CPU: $([math]::Round($process.CPU, 2))s" -ForegroundColor Yellow
    Write-Host ""
} else {
    Write-Host "✗ Training is NOT running" -ForegroundColor Red
    Write-Host ""
}

# Check checkpoint directory
$checkpoint_dir = "C:\_albert\ALS4GAN\checkpoints_v3"
if (Test-Path $checkpoint_dir) {
    Write-Host "Checkpoints:" -ForegroundColor Cyan
    $checkpoints = Get-ChildItem $checkpoint_dir -Filter "*.pth" | Sort-Object LastWriteTime -Descending
    if ($checkpoints) {
        foreach ($ckpt in $checkpoints | Select-Object -First 5) {
            $age = (Get-Date) - $ckpt.LastWriteTime
            $age_str = if ($age.TotalHours -lt 1) {
                "$([math]::Round($age.TotalMinutes, 0))m ago"
            } else {
                "$([math]::Round($age.TotalHours, 1))h ago"
            }
            Write-Host "  $($ckpt.Name.PadRight(30)) - $age_str" -ForegroundColor Yellow
        }
    } else {
        Write-Host "  No checkpoints yet..." -ForegroundColor Gray
    }
    Write-Host ""
} else {
    Write-Host "Checkpoint directory not found" -ForegroundColor Red
    Write-Host ""
}

# Check wandb logs
$wandb_runs = Get-ChildItem "C:\_albert\ALS4GAN\wandb" -Directory | Where-Object {$_.Name -like "run-*"} | Sort-Object LastWriteTime -Descending
if ($wandb_runs) {
    Write-Host "Recent WandB Runs:" -ForegroundColor Cyan
    foreach ($run in $wandb_runs | Select-Object -First 3) {
        $age = (Get-Date) - $run.LastWriteTime
        $age_str = if ($age.TotalHours -lt 1) {
            "$([math]::Round($age.TotalMinutes, 0))m ago"
        } else {
            "$([math]::Round($age.TotalHours, 1))h ago"
        }
        Write-Host "  $($run.Name) - $age_str" -ForegroundColor Yellow
    }
    Write-Host ""
}

# Expected timeline
Write-Host "Expected Timeline:" -ForegroundColor Cyan
Write-Host "  Iteration  500: Val mIoU ~0.28-0.32, ST_px ~15k-25k" -ForegroundColor Gray
Write-Host "  Iteration 2000: Val mIoU ~0.48-0.55, ST_px ~25k-35k" -ForegroundColor Gray
Write-Host "  Iteration 5000: Val mIoU ~0.58-0.65, ST_px ~30k-45k" -ForegroundColor Gray
Write-Host "  Iteration 10000: Val mIoU ~0.65-0.72, ST_px ~35k-50k" -ForegroundColor Gray
Write-Host "  Iteration 20000: Val mIoU ~0.70-0.76, ST_px ~35k-50k (TARGET)" -ForegroundColor Green
Write-Host ""

Write-Host "Tips:" -ForegroundColor Cyan
Write-Host "  • Training takes ~22-24 hours for 50k iterations" -ForegroundColor Gray
Write-Host "  • Check GPU usage: nvidia-smi" -ForegroundColor Gray
Write-Host "  • View latest log: Get-Content checkpoints_v3\training.log -Tail 50" -ForegroundColor Gray
Write-Host "  • Kill training: Stop-Process -Name python" -ForegroundColor Gray
Write-Host ""
Write-Host "=" * 80 -ForegroundColor Cyan
