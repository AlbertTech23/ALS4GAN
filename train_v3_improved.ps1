# V3.1 Improved Training - Aiming for 70%+ mIoU
# This script runs the improved V3 with optimized hyperparameters

Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host "  V3.1 IMPROVED TRAINING - Target: 70%+ mIoU" -ForegroundColor Cyan
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host ""

Write-Host "📊 V3.0 Results:" -ForegroundColor Yellow
Write-Host "  Best mIoU: 0.6214 (62.14%) at iteration 4,000" -ForegroundColor Gray
Write-Host "  Stopped early at iteration 11,500" -ForegroundColor Gray
Write-Host "  Issue: Thresholds too strict, early stopping too aggressive" -ForegroundColor Gray
Write-Host ""

Write-Host "🚀 V3.1 Improvements:" -ForegroundColor Green
Write-Host "  ✅ Lowered confidence thresholds (0.60-0.70 → 0.45-0.60)" -ForegroundColor Gray
Write-Host "  ✅ Extended training (50k → 75k iterations)" -ForegroundColor Gray
Write-Host "  ✅ Increased patience (15 → 40 evaluations = 20k iters)" -ForegroundColor Gray
Write-Host "  ✅ Increased ST weight (1.0 → 1.5)" -ForegroundColor Gray
Write-Host "  ✅ Narrower multi-scale (256-384 → 256-320)" -ForegroundColor Gray
Write-Host "  ✅ Slightly higher LR (0.00025 → 0.0003)" -ForegroundColor Gray
Write-Host "  ✅ Longer warmup (1000 → 1500 iterations)" -ForegroundColor Gray
Write-Host ""

Write-Host "🎯 Expected Results:" -ForegroundColor Cyan
Write-Host "  Best mIoU: 0.70-0.74 (70-74%)" -ForegroundColor Green
Write-Host "  Training time: ~30-35 hours" -ForegroundColor Yellow
Write-Host "  Should NOT stop early this time" -ForegroundColor Green
Write-Host ""

Write-Host "Press any key to start training..." -ForegroundColor Yellow
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
Write-Host ""

Write-Host "🚀 Starting V3.1 training..." -ForegroundColor Green
Write-Host ""

# Run the improved training
C:\Users\IoT-C504-03\miniconda3\envs\als4gan_env\python.exe tools\train_s4gan_salak_v3.py `
  --data-root "C:/_albert/s4GAN/patchify/temp_patches" `
  --class-mapping "C:/_albert/ALS4GAN/class_mapping.csv" `
  --num-classes 7 `
  --batch-size 8 `
  --num-steps 75000 `
  --checkpoint-dir "C:/_albert/ALS4GAN/checkpoints_v3_improved" `
  --early-stop-patience 40 `
  --st-loss-weight 1.5 `
  --scale-min 256 `
  --scale-max 320 `
  --learning-rate 0.0003 `
  --warmup-iters 1500 `
  --eval-every 500 `
  --save-pred-every 5000 `
  --save-latest-every 100

Write-Host ""
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host "  Training Complete!" -ForegroundColor Cyan
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host ""
Write-Host "Check results in: C:\_albert\ALS4GAN\checkpoints_v3_improved" -ForegroundColor Yellow
Write-Host "Best model: best_model_ema.pth" -ForegroundColor Green
