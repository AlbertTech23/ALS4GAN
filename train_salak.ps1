# Train S4GAN on Salak Dataset
# Quick launch script for training

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  S4GAN Training for Salak Dataset" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Configuration
$PYTHON_PATH = "C:\Users\IoT-C504-03\miniconda3\envs\als4gan_env\python.exe"
$DATA_ROOT = "C:/_albert/s4GAN/patchify/temp_patches"
$LABELED_LIST = "C:/_albert/ALS4GAN/labeled_files_50.txt"
$CLASS_MAPPING = "C:/_albert/ALS4GAN/class_mapping.csv"
$CHECKPOINT_DIR = "C:/_albert/ALS4GAN/checkpoints"

Write-Host "Configuration:" -ForegroundColor Yellow
Write-Host "  Python: $PYTHON_PATH"
Write-Host "  Data root: $DATA_ROOT"
Write-Host "  Labeled list: $LABELED_LIST"
Write-Host "  Checkpoints: $CHECKPOINT_DIR"
Write-Host ""

# Check if paths exist
$allPathsExist = $true

if (-not (Test-Path $PYTHON_PATH)) {
    Write-Host "  ERROR: Python executable not found: $PYTHON_PATH" -ForegroundColor Red
    $allPathsExist = $false
}

if (-not (Test-Path $DATA_ROOT)) {
    Write-Host "  ERROR: Data root not found: $DATA_ROOT" -ForegroundColor Red
    $allPathsExist = $false
}

if (-not (Test-Path $LABELED_LIST)) {
    Write-Host "  ERROR: Labeled list not found: $LABELED_LIST" -ForegroundColor Red
    $allPathsExist = $false
}

if (-not (Test-Path $CLASS_MAPPING)) {
    Write-Host "  ERROR: Class mapping not found: $CLASS_MAPPING" -ForegroundColor Red
    $allPathsExist = $false
}

if (-not $allPathsExist) {
    Write-Host ""
    Write-Host "Please fix the paths above and try again." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "  All paths verified!" -ForegroundColor Green
Write-Host ""

# Ask for batch size
Write-Host "Select batch size:" -ForegroundColor Yellow
Write-Host "  1. Batch size 4 (safe, slower)"
Write-Host "  2. Batch size 8 (recommended)"
Write-Host "  3. Batch size 16 (fast, may OOM)"
Write-Host "  4. Custom"
Write-Host ""
$batchChoice = Read-Host "Enter choice (1-4)"

switch ($batchChoice) {
    "1" { $BATCH_SIZE = 4 }
    "2" { $BATCH_SIZE = 8 }
    "3" { $BATCH_SIZE = 16 }
    "4" { 
        $BATCH_SIZE = Read-Host "Enter custom batch size"
        $BATCH_SIZE = [int]$BATCH_SIZE
    }
    default { 
        Write-Host "Invalid choice. Using default: 8" -ForegroundColor Yellow
        $BATCH_SIZE = 8
    }
}

Write-Host ""
Write-Host "Using batch size: $BATCH_SIZE" -ForegroundColor Green
Write-Host ""

# Ask for number of steps
Write-Host "Training duration:" -ForegroundColor Yellow
Write-Host "  1. Quick test (1000 steps, ~5 min)"
Write-Host "  2. Short training (10000 steps, ~2 hours)"
Write-Host "  3. Medium training (25000 steps, ~6 hours)"
Write-Host "  4. Full training (50000 steps, ~15 hours)"
Write-Host "  5. Custom"
Write-Host ""
$stepsChoice = Read-Host "Enter choice (1-5)"

switch ($stepsChoice) {
    "1" { $NUM_STEPS = 1000 }
    "2" { $NUM_STEPS = 10000 }
    "3" { $NUM_STEPS = 25000 }
    "4" { $NUM_STEPS = 50000 }
    "5" { 
        $NUM_STEPS = Read-Host "Enter custom number of steps"
        $NUM_STEPS = [int]$NUM_STEPS
    }
    default { 
        Write-Host "Invalid choice. Using default: 50000" -ForegroundColor Yellow
        $NUM_STEPS = 50000
    }
}

Write-Host ""
Write-Host "Training for: $NUM_STEPS steps" -ForegroundColor Green
Write-Host ""

# Confirm before starting
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Ready to start training with:" -ForegroundColor Cyan
Write-Host "  Batch size: $BATCH_SIZE"
Write-Host "  Training steps: $NUM_STEPS"
Write-Host "  Estimated time: ~$([math]::Round($NUM_STEPS * 1.0 / 3600, 1)) hours"
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
$confirm = Read-Host "Start training? (y/n)"

if ($confirm -ne "y" -and $confirm -ne "Y") {
    Write-Host "Training cancelled." -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 0
}

Write-Host ""
Write-Host "Starting training..." -ForegroundColor Green
Write-Host ""

# Change to ALS4GAN directory
cd C:\_albert\ALS4GAN

# Run training
& $PYTHON_PATH tools\train_s4gan_salak.py `
    --data-root $DATA_ROOT `
    --labeled-list $LABELED_LIST `
    --class-mapping $CLASS_MAPPING `
    --num-classes 7 `
    --batch-size $BATCH_SIZE `
    --num-steps $NUM_STEPS `
    --threshold-st 0.2 `
    --checkpoint-dir $CHECKPOINT_DIR `
    --eval-every 1000 `
    --save-pred-every 5000 `
    --wandb-project "als4gan-salak" `
    --random-mirror `
    --random-scale

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Training completed!" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Checkpoints saved to: $CHECKPOINT_DIR" -ForegroundColor Green
Write-Host "Best model: $CHECKPOINT_DIR\best_model.pth" -ForegroundColor Green
Write-Host ""
Read-Host "Press Enter to exit"
