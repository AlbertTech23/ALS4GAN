# S4GAN Training Monitoring Guide

## 🎯 What to Watch During Training

### **1. Loss Values (Every 100 Iterations)**

```
Iter 500/50000 | Loss_CE: 2.470 | Loss_FM: 0.037 | Loss_ST: 0.000 | Loss_D: 0.184 | ST_Count: 2
```

#### **Loss_CE (Cross-Entropy Loss)**
- **What it means**: How well the model predicts labeled data
- **Good trend**: Should decrease over time
- **Target values**:
  - First 1000 iterations: 3.0-5.0 (high is normal)
  - After 5000 iterations: 1.0-2.0
  - After 20000 iterations: 0.5-1.0
  - End of training: < 0.5
- **⚠️ Warning signs**:
  - Stays > 4.0 after 5000 iterations → Model not learning
  - Sudden spike (e.g., 1.0 → 10.0) → Training instability
  - `NaN` or `inf` → Training crashed, reduce learning rate

#### **Loss_FM (Feature Matching Loss)**
- **What it means**: How realistic the predictions look (adversarial component)
- **Good trend**: Should decrease slowly
- **Target values**:
  - First 1000 iterations: 0.001-0.01
  - After 10000 iterations: 0.005-0.02
  - End of training: 0.01-0.05
- **Note**: Multiplied by `lambda_fm=0.1`, so actual contribution is small
- **⚠️ Warning signs**:
  - Stays at 0.000 → Discriminator not working
  - Increases continuously → GAN training unstable

#### **Loss_ST (Self-Training Loss)**
- **What it means**: Loss from high-confidence predictions on unlabeled data
- **Good trend**: Should activate after iteration 1000, then decrease
- **Target values**:
  - First 1000 iterations: 0.000 (disabled by design)
  - After 1000 iterations: 1.0-3.0 (if ST_Count > 0)
  - After 20000 iterations: 0.3-1.0
  - End of training: < 0.5
- **Note**: Only computed when ST_Count > 0
- **⚠️ Warning signs**:
  - Always 0.000 after iter 1000 → Confidence threshold too high
  - Very high (> 5.0) → Poor quality pseudo-labels

#### **Loss_D (Discriminator Loss)**
- **What it means**: How well discriminator distinguishes real vs fake
- **Good trend**: Should stabilize around 0.3-0.7
- **Target values**:
  - First 1000 iterations: 0.5-0.7 (random guessing is 0.693)
  - After 5000 iterations: 0.3-0.6
  - Steady state: 0.2-0.5
- **⚠️ Warning signs**:
  - Stays at 0.693 → Discriminator not training
  - Drops to 0.0 → Discriminator too strong (generator can't fool it)
  - Above 1.0 → Training unstable

#### **ST_Count (Self-Training Count)**
- **What it means**: Number of unlabeled samples with confidence > threshold (0.2)
- **Good trend**: Should increase over time
- **Target values**:
  - First 1000 iterations: 0-2 (low confidence is normal)
  - After 5000 iterations: 2-8 (with batch size 4)
  - After 20000 iterations: 5-16
  - End of training: 10-16
- **Note**: With batch size 4, max possible is 16 (4 images × 4 batches)
- **⚠️ Warning signs**:
  - Always 0 → Threshold too high (try 0.1 instead of 0.2)
  - Suddenly drops to 0 → Model degrading

---

### **2. Evaluation Metrics (Every 500 Iterations)**

```
============================================================
Evaluating at iteration 500...
============================================================
Training mIoU: 0.3523
Validation mIoU: 0.3201
✓ New best validation mIoU: 0.3201
============================================================
```

#### **Training mIoU (Mean Intersection over Union)**
- **What it means**: How accurately model segments training data
- **Range**: 0.0 (worst) to 1.0 (perfect)
- **Target values**:
  - Iteration 500: 0.25-0.40
  - Iteration 5000: 0.45-0.60
  - Iteration 10000: 0.55-0.70
  - Iteration 25000: 0.65-0.80
  - Iteration 50000: 0.70-0.85
- **Good sign**: Steadily increasing
- **⚠️ Warning signs**:
  - < 0.20 after 5000 iterations → Model not learning
  - Decreases over time → Model degrading

#### **Validation mIoU**
- **What it means**: How well model generalizes to unseen data
- **Range**: 0.0 (worst) to 1.0 (perfect)
- **Target values**:
  - Should be **slightly lower** than training mIoU (0.02-0.10 gap is normal)
  - Iteration 500: 0.20-0.35
  - Iteration 5000: 0.40-0.55
  - Iteration 10000: 0.50-0.65
  - Iteration 25000: 0.60-0.75
  - Iteration 50000: 0.65-0.80

#### **Overfitting Detection**
Compare Training vs Validation mIoU:

| Scenario | Train mIoU | Val mIoU | Gap | Status |
|----------|------------|----------|-----|--------|
| **Good fit** | 0.75 | 0.72 | 0.03 | ✅ Excellent |
| **Good fit** | 0.68 | 0.62 | 0.06 | ✅ Good |
| **Slight overfit** | 0.80 | 0.68 | 0.12 | ⚠️ Acceptable |
| **Overfitting** | 0.85 | 0.60 | 0.25 | ❌ Problem |
| **Underfitting** | 0.35 | 0.33 | 0.02 | ⚠️ Both low |

**What to do if overfitting:**
- Reduce learning rate
- Add more data augmentation
- Increase dropout
- Stop training earlier

---

### **3. Learning Rates**

Logged to Wandb every 100 iterations:
- **Generator LR**: Starts at 2.5e-4, decreases polynomially to ~0 by iteration 50000
- **Discriminator LR**: Starts at 1.0e-4, decreases polynomially

**Normal behavior**: Both should decrease smoothly over time

---

## 📊 Wandb Dashboard - What to Monitor

### **Key Charts:**

#### **1. Training Loss/**
- **Cross Entropy**: Should decrease steadily
- **Feature Matching**: Small fluctuations, overall stable
- **Self-Training**: Activates after iter 1000, then decreases
- **Discriminator**: Should stabilize around 0.3-0.6
- **Total Generator**: Sum of all losses, should decrease

#### **2. Metrics/**
- **Training mIoU**: Should increase over time
- **Validation mIoU**: Should track training mIoU (slightly lower)
- **Gap between Train/Val**: Should stay < 0.15

#### **3. Self-Training/**
- **Confidence Count**: Should increase over time
- Target: 5-16 samples per batch by end of training

#### **4. Learning Rate/**
- **Generator**: Smooth polynomial decay
- **Discriminator**: Smooth polynomial decay

---

## 🎯 Expected Progress Timeline

### **Iteration 0-1000: Warm-up Phase**
- ✅ Loss_CE: 3.0-5.0 (high)
- ✅ Loss_D: 0.5-0.7 (learning)
- ✅ ST_Count: 0-2 (low confidence)
- ✅ Val mIoU: 0.20-0.35
- **What's happening**: Model learning basic features

### **Iteration 1000-5000: Early Learning**
- ✅ Loss_CE: 1.5-2.5
- ✅ Loss_ST: Starts activating (if ST_Count > 0)
- ✅ Loss_D: 0.3-0.6
- ✅ ST_Count: 2-8
- ✅ Val mIoU: 0.35-0.55
- **What's happening**: Model improving, semi-supervised learning begins

### **Iteration 5000-20000: Main Training**
- ✅ Loss_CE: 0.8-1.5
- ✅ Loss_ST: 0.5-2.0
- ✅ Loss_D: 0.2-0.5
- ✅ ST_Count: 5-12
- ✅ Val mIoU: 0.55-0.70
- **What's happening**: Model refining predictions, leveraging unlabeled data

### **Iteration 20000-50000: Convergence**
- ✅ Loss_CE: 0.3-0.8
- ✅ Loss_ST: 0.2-1.0
- ✅ Loss_D: 0.2-0.4
- ✅ ST_Count: 8-16
- ✅ Val mIoU: 0.65-0.80
- **What's happening**: Model fine-tuning, approaching optimal performance

---

## 🚨 Common Problems & Solutions

### **Problem 1: Val mIoU Not Improving**
**Symptoms:**
- Val mIoU stuck at 0.20-0.30 after 10k iterations
- Training mIoU increasing but validation stagnant

**Solutions:**
1. Check if labels are correct (review visualizations)
2. Increase data augmentation (`--random-mirror` `--random-scale`)
3. Reduce learning rate by 50%
4. Check class imbalance (some classes might be ignored)

### **Problem 2: Training Unstable (Loss Spikes)**
**Symptoms:**
- Sudden loss spikes (e.g., 1.0 → 20.0)
- NaN or inf values
- Model performance degrading

**Solutions:**
1. Reduce learning rate: `--learning-rate 1.0e-4` (half of default)
2. Reduce discriminator LR: `--learning-rate-D 5.0e-5`
3. Check for corrupted data (empty masks)
4. Restart from last checkpoint

### **Problem 3: Self-Training Not Working (ST_Count Always 0)**
**Symptoms:**
- ST_Count stays 0 after iteration 1000
- Loss_ST always 0.000

**Solutions:**
1. Lower threshold: `--threshold-st 0.1` (instead of 0.2)
2. Wait longer (may activate after 5000 iterations)
3. Check if model is learning at all (check val mIoU)

### **Problem 4: Overfitting (Train >> Val)**
**Symptoms:**
- Train mIoU: 0.85, Val mIoU: 0.55 (gap > 0.20)
- Validation mIoU decreasing while training increasing

**Solutions:**
1. Stop training earlier (use best_model.pth, not final_model.pth)
2. Add regularization
3. Use more labeled data if available
4. Increase val_split: `--val-split 0.3`

### **Problem 5: Out of Memory (CUDA OOM)**
**Symptoms:**
- Training crashes with "CUDA out of memory"
- System freezes

**Solutions:**
1. ✅ **Already done**: Reduce batch size to 4
2. If still crashing: Reduce to batch size 2
3. Close other applications
4. Reduce image size (not recommended)

---

## 📈 What "Good" Looks Like

### **Console Output Example (Iteration 10000):**
```
Iter 10000/50000 | Loss_CE: 0.823 | Loss_FM: 0.015 | Loss_ST: 0.645 | Loss_D: 0.312 | ST_Count: 8

============================================================
Evaluating at iteration 10000...
============================================================
Training mIoU: 0.6234
Validation mIoU: 0.5876
✓ New best validation mIoU: 0.5876 (previous: 0.5654)
============================================================
```

**Analysis:**
- ✅ Loss_CE decreasing (was ~3.0 at start)
- ✅ Loss_ST active and reasonable (0.645)
- ✅ Loss_D stable (0.312)
- ✅ ST_Count increasing (was 2-4 earlier)
- ✅ Train mIoU: 0.62 (good progress)
- ✅ Val mIoU: 0.59 (close to train, no overfitting)
- ✅ Val mIoU improving (new best!)

This is **excellent progress!** Continue training.

---

## 🎯 When to Stop Training

### **Stop Early (Before 50k Iterations) If:**
1. ✅ Validation mIoU hasn't improved for 10,000 iterations
2. ✅ Validation mIoU starts decreasing (overfitting)
3. ✅ You're satisfied with current performance (e.g., val mIoU > 0.75)

**Use `best_model.pth` for inference** (not final_model.pth!)

### **Continue Training If:**
1. Validation mIoU still improving
2. Train/Val gap < 0.15
3. Haven't reached 50,000 iterations yet

---

## 💾 Files to Keep After Training

**Essential:**
- ✅ `best_model.pth` - Best model (use this for inference!)
- ✅ `best_model_D.pth` - Corresponding discriminator
- ✅ `checkpoint_50000.pth` - Last full checkpoint (for resume/extension)

**Optional (can delete to save space):**
- ❌ `latest_checkpoint.pth` - Only needed during training
- ❌ `checkpoint_5000.pth`, `checkpoint_10000.pth`, etc. - Intermediate backups
- ❌ `final_model.pth` - Usually worse than best_model.pth

---

## 📧 Summary Checklist

Before closing the training, verify:

- [ ] Best validation mIoU recorded (check Wandb or console)
- [ ] `best_model.pth` exists in checkpoint directory
- [ ] Training completed without errors
- [ ] Final train/val gap < 0.15 (no severe overfitting)
- [ ] Wandb run finished successfully

**Your model is ready to use!** 🎉
