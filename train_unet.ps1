# ============================================================
# UNet++ with MobileNetV2 Training - Advanced Lightweight Model
# ============================================================
#
# ARCHITECTURE:
# • UNet++ with MobileNetV2 encoder (BEST OF BOTH WORLDS)
# • Dense nested skip connections (superior to standard UNet)
# • Parameters: ~5-7M (7× lighter than DeepLabV3+)
# • Multi-Head ensemble (3 heads)
# • Expected: 0.65-0.70 mIoU (matching V3 with less compute)
#
# ADVANTAGES OVER STANDARD UNET:
# • Dense connections: Better feature fusion
# • More accurate: ~2-3% higher than standard UNet
# • Nested decoder: Gradual feature transformation
# • Still lightweight and fast
#
# IMPROVEMENTS FROM V3.0 (applied here):
# ✅ Lowered confidence thresholds: 0.45-0.60
# ✅ Patience: 40 evaluations (~20k iterations)
# ✅ ST weight: 1.5 (stronger self-training)
# ✅ Training: 75k iterations
# ✅ Multi-scale: 256-320 (narrower range)
# ✅ Learning rate: 0.0003
# ✅ Warmup: 1500 iterations
# ✅ Batch size: 12 (balanced for UNet++)
# ============================================================

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "UNet++ with MobileNetV2 Training - Salak Dataset" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "🚀 Advanced Lightweight Architecture" -ForegroundColor Green
Write-Host "  • UNet++ with dense nested skip connections" -ForegroundColor Yellow
Write-Host "  • Parameters: ~5-7M vs 40M (7× reduction)" -ForegroundColor Yellow
Write-Host "  • Better than standard UNet (~2-3% higher mIoU)" -ForegroundColor Yellow
Write-Host "  • Same V3.1 improvements applied" -ForegroundColor Yellow
Write-Host ""

# Configuration
$DATA_ROOT = "C:/_albert/s4GAN/patchify/temp_patches"
$CLASS_MAPPING = "C:/_albert/ALS4GAN/class_mapping.csv"
$CHECKPOINT_DIR = "C:/_albert/ALS4GAN/checkpoints_unetpp"
$WANDB_PROJECT = "als4gan-salak"
$WANDB_RUN_NAME = "unetpp_mobilenetv2_improved"

# V3.1 Improved Hyperparameters
$BATCH_SIZE = 12          # Balanced for UNet++ (slightly more params than UNet)
$NUM_STEPS = 75000        # Same as V3.1
$LEARNING_RATE = 0.0003   # Same as V3.1
$WARMUP_ITERS = 1500      # Same as V3.1
$ST_WEIGHT = 1.5          # Same as V3.1
$PATIENCE = 40            # Same as V3.1
$SCALE_MIN = 256          # Same as V3.1
$SCALE_MAX = 320          # Same as V3.1

Write-Host "📋 Configuration:" -ForegroundColor Cyan
Write-Host "  Data Root: $DATA_ROOT" -ForegroundColor White
Write-Host "  Checkpoint Dir: $CHECKPOINT_DIR" -ForegroundColor White
Write-Host "  Batch Size: $BATCH_SIZE (balanced for UNet++)" -ForegroundColor White
Write-Host "  Training Steps: $NUM_STEPS" -ForegroundColor White
Write-Host "  Learning Rate: $LEARNING_RATE" -ForegroundColor White
Write-Host "  Warmup Iterations: $WARMUP_ITERS" -ForegroundColor White
Write-Host "  ST Weight: $ST_WEIGHT" -ForegroundColor White
Write-Host "  Patience: $PATIENCE evaluations (~20k iterations)" -ForegroundColor White
Write-Host "  Multi-Scale: ${SCALE_MIN}-${SCALE_MAX}px" -ForegroundColor White
Write-Host ""

# Create checkpoint directory
if (-not (Test-Path $CHECKPOINT_DIR)) {
    Write-Host "📁 Creating checkpoint directory..." -ForegroundColor Yellow
    New-Item -ItemType Directory -Path $CHECKPOINT_DIR -Force | Out-Null
}

# Confirm before starting
Write-Host "⚠️  Ready to start UNet++ training..." -ForegroundColor Yellow
Write-Host "   This will run for ~22-28 hours" -ForegroundColor Yellow
Write-Host "   Checkpoints saved to: $CHECKPOINT_DIR" -ForegroundColor Yellow
Write-Host ""
$response = Read-Host "Continue? (y/n)"
if ($response -ne "y") {
    Write-Host "❌ Training cancelled." -ForegroundColor Red
    exit
}

Write-Host ""
Write-Host "🚀 Starting UNet++ with MobileNetV2 training..." -ForegroundColor Green
Write-Host ""

# Run training with V3.1 improvements
python tools/train_unet_salak.py `
  --data-root $DATA_ROOT `
  --class-mapping $CLASS_MAPPING `
  --num-classes 7 `
  --batch-size $BATCH_SIZE `
  --num-steps $NUM_STEPS `
  --learning-rate $LEARNING_RATE `
  --warmup-iters $WARMUP_ITERS `
  --st-loss-weight $ST_WEIGHT `
  --early-stop-patience $PATIENCE `
  --scale-min $SCALE_MIN `
  --scale-max $SCALE_MAX `
  --checkpoint-dir $CHECKPOINT_DIR `
  --wandb-project $WANDB_PROJECT `
  --wandb-run-name $WANDB_RUN_NAME `
  --use-multi-head `
  --use-combined-loss `
  --use-class-weights `
  --multi-scale `
  --use-ema `
  --random-mirror `
  --random-scale

$exitCode = $LASTEXITCODE
Write-Host ""
if ($exitCode -eq 0) {
    Write-Host "✅ Training completed successfully!" -ForegroundColor Green
    Write-Host "📊 Check results in: $CHECKPOINT_DIR" -ForegroundColor Cyan
} else {
    Write-Host "❌ Training failed with exit code: $exitCode" -ForegroundColor Red
}

exit $exitCode
