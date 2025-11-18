# UNet-MobileNetV2 vs DeepLabV3+ Comparison
C:\Users\IoT-C504-03\miniconda3\envs\als4gan_env\python.exe tools/train_unet_salak.py --data-root "C:/_albert/s4GAN/patchify/temp_patches" --class-mapping "C:/_albert/ALS4GAN/class_mapping.csv" --num-classes 7 --batch-size 12 --num-steps 75000 --learning-rate 0.0003 --warmup-iters 1500 --st-loss-weight 1.5 --early-stop-patience 40 --scale-min 256 --scale-max 320 --checkpoint-dir "C:/_albert/ALS4GAN/checkpoints_unetpp" --wandb-project "als4gan-salak" --wandb-run-name "unetpp_mobilenetv2_improved" --use-multi-head --use-combined-loss --use-class-weights --multi-scale --use-ema --random-mirror --random-scale

## Quick Reference

| Feature | DeepLabV3+ (V3) | UNet-MobileNetV2 | Winner |
|---------|-----------------|------------------|--------|
| **Parameters** | 40.35M | ~3.5M | 🏆 UNet (11× lighter) |
| **Training Speed** | Baseline | ~2× faster | 🏆 UNet |
| **Memory Usage** | ~8GB VRAM | ~4GB VRAM | 🏆 UNet |
| **Expected mIoU** | 0.65-0.70 | 0.63-0.68 | 🏆 V3 (slightly better) |
| **Inference Speed** | Baseline | ~3× faster | 🏆 UNet |
| **Edge Deployment** | ❌ Too large | ✅ Mobile-ready | 🏆 UNet |
| **Training Time** | ~30-35 hours | ~20-25 hours | 🏆 UNet |

## Architecture Details

### DeepLabV3+ ResNet50 (V3)
```
Encoder: ResNet50 (pretrained ImageNet)
├─ Block 1-4: Standard ResNet layers
├─ ASPP: Atrous Spatial Pyramid Pooling
│   ├─ 1×1 conv
│   ├─ 3×3 atrous conv (rate=6)
│   ├─ 3×3 atrous conv (rate=12)
│   ├─ 3×3 atrous conv (rate=18)
│   └─ Global pooling
└─ Decoder: Upsampling + Low-level features

Multi-Head: 3 classification heads
Parameters: 40,354,247
```

**Strengths**:
- Proven architecture (SOTA on many datasets)
- Atrous convolutions capture multi-scale context
- Strong on complex scenes
- Higher accuracy potential

**Weaknesses**:
- Large model size (40M parameters)
- Slow training (~30-35 hours)
- High memory usage (~8GB VRAM)
- Not suitable for edge deployment

---

### UNet-MobileNetV2
```
Encoder: MobileNetV2 (pretrained ImageNet)
├─ Layer 0-1: 32 channels (128×128)
├─ Layer 2-3: 24 channels (64×64)
├─ Layer 4-6: 32 channels (32×32)
├─ Layer 7-13: 96 channels (16×16)
└─ Layer 14-17: 320 channels (8×8)

Bridge: ConvBlock 320→512

Decoder: UNet skip connections
├─ Up1 + Skip(96ch): 512→256 (16×16)
├─ Up2 + Skip(32ch): 256→128 (32×32)
├─ Up3 + Skip(24ch): 128→64 (64×64)
└─ Up4 + Skip(32ch): 64→32 (128×128)

Output: 32→7 classes (256×256)

Multi-Head: 3 classification heads
Parameters: 3,502,023
```

**Strengths**:
- Extremely lightweight (3.5M parameters)
- Fast training (~20-25 hours, 40% faster)
- Low memory usage (~4GB VRAM)
- Mobile/edge deployment ready
- Skip connections preserve details
- MobileNetV2 designed for efficiency

**Weaknesses**:
- Slightly lower accuracy (expected ~2-3% below V3)
- Less capacity for very complex patterns
- Fewer parameters may limit performance ceiling

---

## Training Configuration

Both models use the **same improved hyperparameters**:

| Hyperparameter | Value | Purpose |
|----------------|-------|---------|
| **Confidence Thresholds** | 0.45-0.60 | Lower = more pseudo-labels |
| **Patience** | 40 evaluations | ~20k iterations tolerance |
| **ST Weight** | 1.5 | Stronger self-training signal |
| **Total Steps** | 75,000 | Longer training for better convergence |
| **Multi-Scale** | 256-320px | Narrower range for stability |
| **Learning Rate** | 0.0003 | Slightly higher for faster convergence |
| **Warmup** | 1,500 iterations | Gentle start for stability |
| **Batch Size** | V3: 8, UNet: 16 | UNet can use larger batches |

**Shared Features**:
- ✅ Multi-Head ensemble (3 heads with dropout diversity)
- ✅ Combined Loss (CE + Dice + Focal)
- ✅ Class-wise confidence thresholds
- ✅ EMA (0.9995) for stability
- ✅ Gradient clipping (10.0)
- ✅ Cosine LR schedule
- ✅ Class weighting for imbalance
- ✅ NO discriminator (removed complexity)

---

## Expected Performance

### V3 (DeepLabV3+) - RUNNING
**Status**: Training in `checkpoints_v3_improved/`  
**Current**: ~15,000-20,000 iterations (estimated)  
**Expected**: 0.65-0.70 mIoU (without background)  
**Training Time**: ~30-35 hours total  

### UNet-MobileNetV2 - READY TO START
**Status**: Script ready, not yet started  
**Expected**: 0.63-0.68 mIoU (without background)  
**Training Time**: ~20-25 hours total  
**Checkpoint Dir**: `checkpoints_unet/`  

### Comparison Notes
- UNet may achieve ~2-3% lower mIoU than V3
- BUT: UNet trains 40% faster (saves ~10 hours)
- UNet uses half the memory (can increase batch size)
- UNet is deployment-ready (edge devices, mobile)

---

## When to Use Each Model

### Use DeepLabV3+ (V3) if:
- ✅ Maximum accuracy is critical
- ✅ You have sufficient compute resources
- ✅ Training time is not a constraint
- ✅ Deployment is server-side only
- ✅ You need SOTA performance

### Use UNet-MobileNetV2 if:
- ✅ Fast training is important
- ✅ Edge/mobile deployment needed
- ✅ Limited GPU memory available
- ✅ Inference speed matters
- ✅ Good accuracy is sufficient (not SOTA required)

---

## How to Start UNet Training

1. **Ensure V3 is running** (don't interrupt it!)
   ```powershell
   # Check V3 status
   Get-Process python
   ```

2. **Open NEW PowerShell terminal**
   ```powershell
   cd C:\_albert\ALS4GAN
   ```

3. **Run UNet training script**
   ```powershell
   .\train_unet.ps1
   ```

4. **Monitor both trainings**
   - V3: `checkpoints_v3_improved/`
   - UNet: `checkpoints_unet/`
   - W&B: `als4gan-salak` project

---

## Results Tracking

### V3 (DeepLabV3+)
- **Checkpoint Dir**: `C:/_albert/ALS4GAN/checkpoints_v3_improved/`
- **W&B Run**: `v3_improved_<timestamp>`
- **Expected Completion**: After ~30-35 hours from start
- **Best Model**: `best_model_ema.pth` (highest Val mIoU)

### UNet-MobileNetV2
- **Checkpoint Dir**: `C:/_albert/ALS4GAN/checkpoints_unet/`
- **W&B Run**: `unet_mobilenetv2_improved`
- **Expected Completion**: After ~20-25 hours from start
- **Best Model**: `best_model_ema.pth` (highest Val mIoU)

---

## Final Comparison (After Both Complete)

After both trainings finish, compare:

1. **Accuracy Metrics**:
   - mIoU (foreground only, excluding class 0)
   - Per-class IoU
   - Training stability (loss curves)

2. **Training Efficiency**:
   - Total training time
   - Iterations per second
   - Memory usage

3. **Model Size**:
   - Parameters: V3 (40M) vs UNet (3.5M)
   - Checkpoint file size

4. **Inference Speed** (test on same hardware):
   - Batch inference time
   - Single image latency

**Decision Criteria**:
- If mIoU difference < 2%: **Choose UNet** (much more efficient)
- If mIoU difference > 5%: **Choose V3** (significantly better)
- If deploying to edge: **Choose UNet** (only viable option)
- If server-only: **Choose V3** (maximize accuracy)

---

## Next Steps

1. ✅ **V3 Training**: Already running
2. 🔄 **Start UNet Training**: Run `.\train_unet.ps1`
3. ⏳ **Wait for Both**: Monitor via W&B
4. 📊 **Compare Results**: Use metrics from both runs
5. 🎯 **Choose Best Model**: Based on use case (accuracy vs efficiency)
6. 🚀 **Deploy**: Use chosen model for inference

---

## Questions?

- **Can I run both at once?** Yes! They use separate checkpoint directories.
- **Will UNet interfere with V3?** No, completely independent.
- **Which will finish first?** UNet (~20-25 hours) before V3 (~30-35 hours).
- **Can I stop UNet if V3 is better?** Yes, just close the terminal.
- **Should I use both?** Consider ensemble if both perform well!

---

**Created**: For comparing lightweight UNet with heavier DeepLabV3+  
**Purpose**: Help choose best model for Salak dataset segmentation  
**Recommendation**: Wait for both results, then decide based on your deployment needs.
