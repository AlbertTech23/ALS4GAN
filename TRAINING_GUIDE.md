# Training S4GAN on Salak Dataset - Complete Guide

## 🎯 Overview
This guide covers training the S4GAN model on your Salak dataset with **370,647 patches** (12,856 labeled + 357,791 unlabeled) for semi-supervised semantic segmentation.

### 📋 Quick Facts
- **Total patches**: 370,647 (256×256 pixels each)
- **Labeled patches**: 12,856 (from 50 base images)
- **Unlabeled patches**: 357,791 (for semi-supervised learning)
- **Classes**: 7 (including background)
- **Training duration**: ~4-5 hours with batch size 16 on RTX 4060 Ti
- **Recommended iterations**: 50,000

### 🔑 Key Updates (Latest Version)
✅ **Automatic full dataset loading** - No need for `--labeled-list` argument  
✅ **370k+ patches loaded** - All labeled + unlabeled data  
✅ **Enhanced checkpointing** - Resume training from any interruption  
✅ **Better mIoU tracking** - Evaluation every 500 iterations  
✅ **Latest checkpoint** - Auto-save every 100 iterations for crash recovery  

---

## 🚀 Quick Start

### 1. Test the Dataset First
```powershell
cd C:\_albert\ALS4GAN
C:\Users\IoT-C504-03\miniconda3\envs\als4gan_env\python.exe data\test_dataloader.py
```

**Expected Output:**
```
📊 TOTAL PATCHES: 370647 images, 12856 masks
📊 Unlabeled patches: 357791

TEST 1: Loading patches from labeled list (50 base images)...
  Total samples: 50

TEST 2: Loading ALL patches (full dataset)...
  Total samples: 370647
  Images with masks: 12856
  Images without masks (unlabeled): 357791
```

### 2. Start Training (Recommended Settings)

**Basic Training (Batch Size 4):**
```powershell
cd C:\_albert\ALS4GAN

C:\Users\IoT-C504-03\miniconda3\envs\als4gan_env\python.exe tools\train_s4gan_salak.py `
  --data-root "C:/_albert/s4GAN/patchify/temp_patches" `
  --class-mapping "C:/_albert/ALS4GAN/class_mapping.csv" `
  --num-classes 7 `
  --batch-size 4 `
  --num-steps 50000 `
  --threshold-st 0.1 `
  --checkpoint-dir "C:/_albert/ALS4GAN/checkpoints" `
  --eval-every 500 `
  --save-pred-every 5000 `
  --save-latest-every 100 `
  --wandb-project "als4gan-salak"
```

**Faster Training (Batch Size 16 - if GPU memory allows):**
```powershell
C:\Users\IoT-C504-03\miniconda3\envs\als4gan_env\python.exe tools\train_s4gan_salak.py `
  --data-root "C:/_albert/s4GAN/patchify/temp_patches" `
  --class-mapping "C:/_albert/ALS4GAN/class_mapping.csv" `
  --num-classes 7 `
  --batch-size 16 `
  --num-steps 50000 `
  --threshold-st 0.1 `
  --checkpoint-dir "C:/_albert/ALS4GAN/checkpoints" `
  --eval-every 500 `
  --save-pred-every 5000 `
  --save-latest-every 100 `
  --wandb-project "als4gan-salak"
```

**Key Changes from Previous Version:**
- ✅ **NO `--labeled-list` argument** → Loads ALL 370k patches automatically
- ✅ **50,000 iterations** (increased from 40k)
- ✅ **Evaluation every 500 iterations** (for better mIoU tracking)
- ✅ **Latest checkpoint every 100 iterations** (for resume capability)
- ✅ **Periodic checkpoints every 5,000 iterations**

---

## 📊 What the Training Does

### Data Loading Strategy
The training script now loads **ALL patches** from your dataset:

**Dataset Statistics:**
- **Total patches**: 370,647
- **Labeled patches** (with masks): 12,856 from 50 base images
- **Unlabeled patches** (no masks): 357,791
- **Image size**: 256×256 pixels
- **Classes**: 7 (including background)

### Data Split (Automatic 80/20 on Labeled Data)
- **Training**: 80% of labeled patches (~10,285 patches)
- **Validation**: 20% of labeled patches (~2,571 patches)
- **Unlabeled**: All 357,791 unlabeled patches used for semi-supervised learning

### Losses Tracked
1. **Cross-Entropy Loss**: Supervised learning on labeled data
2. **Feature Matching Loss**: Ensures realistic predictions (adversarial)
3. **Self-Training Loss**: Learns from high-confidence predictions on unlabeled data
4. **Discriminator Loss**: Adversarial training to improve realism

### Metrics Tracked (Wandb)
- **Training mIoU**: Mean Intersection over Union on training set (every 500 iterations)
- **Validation mIoU**: Mean IoU on validation set (overfitting indicator)
- **Learning Rates**: For both Generator and Discriminator
- **Self-Training Count**: How many unlabeled samples pass confidence threshold (≥0.2)
- **Per-Class IoU**: IoU for each of the 7 classes

### Charts You'll See in Wandb
1. **"Training Loss vs Iteration"**: Shows all loss components
2. **"Training mIoU vs Iteration"**: Performance on training data
3. **"Validation mIoU vs Iteration"**: Performance on validation data
   - If training mIoU >> validation mIoU → **Overfitting**
   - If both low → **Underfitting**
   - If both high and close → **Good fit!**

---

## 🔧 Important Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--batch-size` | 4 | Increase if you have enough GPU memory (8-16 for RTX 4060 Ti) |
| `--threshold-st` | 0.2 | Self-training threshold (0.0-1.0). Higher = more conservative |
| `--num-steps` | 50000 | Total training iterations (~15 hours with batch size 4) |
| `--eval-every` | 500 | Evaluate validation mIoU every N iterations |
| `--save-pred-every` | 5000 | Save full checkpoint every N iterations (checkpoint_5000.pth, checkpoint_10000.pth, ...) |
| `--save-latest-every` | 100 | Save latest checkpoint every N iterations (for crash recovery) |
| `--val-split` | 0.2 | Validation split ratio (20% of labeled data) |

### Adjusting for Your GPU

**RTX 4060 Ti (16GB VRAM)**:
- Batch size 4: Safe, ~15 hours for 50k iterations
- Batch size 8: Faster, ~8 hours
- Batch size 16: **Optimal**, ~4-5 hours
- Batch size 32: May run out of memory

Try starting with batch size 16:
```powershell
--batch-size 16
```

---

## 💾 Model Checkpoints & Saving Strategy

### Files Saved During Training

The training script saves **3 types** of checkpoint files:

#### 1. **Latest Checkpoint** (Every 100 iterations)
- **File**: `latest_checkpoint.pth`
- **Purpose**: Resume training after crashes/interruptions
- **Contains**: Model weights, optimizer state, iteration counter, best mIoU
- **Use**: For continuing interrupted training

#### 2. **Best Model** (When validation mIoU improves)
- **Files**: `best_model.pth`, `best_model_D.pth`
- **Purpose**: Best performing model on validation set
- **Contains**: Model weights only
- **Use**: For inference/testing after training

#### 3. **Periodic Checkpoints** (Every 5,000 iterations)
- **Files**: `checkpoint_5000.pth`, `checkpoint_10000.pth`, ..., `checkpoint_50000.pth`
- **Purpose**: Backups at regular intervals
- **Contains**: Full checkpoint (model + optimizer + iteration)
- **Use**: Fallback if latest checkpoint corrupts, or to resume from specific iteration

#### 4. **Final Model** (At the end of training)
- **Files**: `final_model.pth`, `final_model_D.pth`
- **Purpose**: Model at the end of 50,000 iterations
- **Contains**: Model weights only

### Checkpoint Directory Structure
```
checkpoints/
├── latest_checkpoint.pth       # Updated every 100 iterations (RESUMABLE)
├── latest_model.pth            # Model weights only (every 100 iter)
├── latest_model_D.pth          # Discriminator weights (every 100 iter)
├── best_model.pth              # Best model based on validation mIoU
├── best_model_D.pth            # Best discriminator
├── checkpoint_5000.pth         # Full checkpoint at iteration 5000
├── checkpoint_10000.pth        # Full checkpoint at iteration 10000
├── checkpoint_15000.pth        # ...
├── ...
├── checkpoint_50000.pth        # Full checkpoint at iteration 50000
├── final_model.pth             # Final model at end of training
└── final_model_D.pth           # Final discriminator
```

---

## 🔄 Resuming Training After Interruption

### Scenario: Training Stopped at Iteration 12,345

#### **Option 1: Resume from Latest Checkpoint (Recommended)**
```powershell
C:\Users\IoT-C504-03\miniconda3\envs\als4gan_env\python.exe tools\train_s4gan_salak.py `
  --data-root "C:/_albert/s4GAN/patchify/temp_patches" `
  --class-mapping "C:/_albert/ALS4GAN/class_mapping.csv" `
  --num-classes 7 `
  --batch-size 16 `
  --num-steps 50000 `
  --checkpoint-dir "C:/_albert/ALS4GAN/checkpoints" `
  --restore-from "C:/_albert/ALS4GAN/checkpoints/latest_checkpoint.pth" `
  --wandb-project "als4gan-salak"
```

**What happens:**
1. Loads model weights from checkpoint
2. Loads optimizer state (learning rate, momentum)
3. Resumes from iteration 12,300 (rounded down to last save at iteration 12,300)
4. Continues training to iteration 50,000
5. **Total iterations still 50,000** (not 50,000 + 12,300)

#### **Option 2: Resume from Specific Periodic Checkpoint**
```powershell
--restore-from "C:/_albert/ALS4GAN/checkpoints/checkpoint_10000.pth"
```

**What happens:**
- Resumes from exactly iteration 10,000
- Continues to iteration 50,000

#### **Option 3: Continue Training Beyond 50,000 Iterations**
If you want to train LONGER (e.g., from 50k to 100k):

```powershell
C:\Users\IoT-C504-03\miniconda3\envs\als4gan_env\python.exe tools\train_s4gan_salak.py `
  --data-root "C:/_albert/s4GAN/patchify/temp_patches" `
  --class-mapping "C:/_albert/ALS4GAN/class_mapping.csv" `
  --num-classes 7 `
  --batch-size 16 `
  --num-steps 100000 `
  --checkpoint-dir "C:/_albert/ALS4GAN/checkpoints_extended" `
  --restore-from "C:/_albert/ALS4GAN/checkpoints/checkpoint_50000.pth" `
  --wandb-project "als4gan-salak"
```

**What happens:**
- Loads checkpoint from iteration 50,000
- Continues training from 50,000 → 100,000
- Saves to new checkpoint directory

---

## 📈 Monitoring Training

### Via Wandb Dashboard

1. **First time**: Script will prompt for API key
   - Go to https://wandb.ai/authorize
   - Copy your API key
   - Paste when prompted

2. **View live charts**:
   - Go to https://wandb.ai
   - Navigate to project: `als4gan-salak`
   - You'll see real-time updates every 100 iterations

### Via Console Output

```
Iter 1000/40000 | Loss_CE: 0.523 | Loss_FM: 0.012 | Loss_ST: 0.234 | Loss_D: 0.145 | ST_Count: 2
============================================================
Evaluating at iteration 1000...
============================================================
Training mIoU: 0.4523
Validation mIoU: 0.4201
✓ New best validation mIoU: 0.4201
============================================================
```

**What to watch**:
- ✅ **Loss_CE decreasing**: Model is learning
- ✅ **ST_Count > 0**: Self-training is working
- ✅ **Val mIoU increasing**: Model generalizes well
- ⚠️ **Train mIoU >> Val mIoU**: Overfitting (reduce learning rate or add regularization)

---

## 🎯 Expected Results

### Good Training Signs
✅ Validation mIoU > 0.6 (60%) after 25k iterations  
✅ Gap between train and val mIoU < 0.1  
✅ Losses steadily decreasing  
✅ Self-training count > 0 consistently  

### Performance Benchmarks (Estimated)
| Iteration | Expected Val mIoU | Status |
|-----------|-------------------|--------|
| 5,000 | 0.35-0.45 | Early learning |
| 10,000 | 0.45-0.55 | Improving |
| 25,000 | 0.60-0.70 | Good performance |
| 50,000 | 0.65-0.75 | Near convergence |

**Note**: Results may vary based on dataset quality and labeling accuracy.

---

## � Understanding the Checkpoint Mechanism (In-Depth)

### How Checkpointing Works

#### **What Gets Saved in a Full Checkpoint?**
```python
checkpoint = {
    'iteration': 12345,                    # Current iteration number
    'model_state': model.state_dict(),     # Generator weights
    'model_D_state': model_D.state_dict(), # Discriminator weights
    'optimizer_state': optimizer.state_dict(),     # Optimizer state (learning rate, momentum)
    'optimizer_D_state': optimizer_D.state_dict(), # Discriminator optimizer state
    'best_val_miou': 0.6523,               # Best validation mIoU so far
}
```

#### **Why Save Optimizer State?**
The optimizer stores important training dynamics:
- **Learning rate**: Polynomial decay means LR decreases over time
- **Momentum**: SGD with momentum accumulates gradients
- **Parameter history**: Helps maintain training stability

Without optimizer state, resuming training would:
- ❌ Reset learning rate to initial value (too high!)
- ❌ Lose momentum accumulation
- ❌ Cause training instability and worse convergence

#### **What Happens When Training Stops?**

**Scenario**: Training stops at iteration 12,345

1. **Last `latest_checkpoint.pth` saved**: Iteration 12,300 (rounded down to last save at 12,300)
2. **Periodic checkpoint exists**: `checkpoint_10000.pth` at iteration 10,000
3. **Best model saved**: `best_model.pth` with validation mIoU = 0.6523 (maybe from iteration 11,800)

**Your options:**
- Resume from 12,300 (latest checkpoint) - **Recommended**
- Resume from 10,000 (periodic checkpoint) - Loses 2,300 iterations of progress
- Start fresh with best model - Loses optimizer state, training may be unstable

#### **Resume Training Process (Step-by-Step)**

When you run with `--restore-from`:

```powershell
--restore-from "C:/_albert/ALS4GAN/checkpoints/latest_checkpoint.pth"
```

**What the code does:**

```python
# 1. Load checkpoint file
checkpoint = torch.load('latest_checkpoint.pth')

# 2. Restore model weights
model.load_state_dict(checkpoint['model_state'])
model_D.load_state_dict(checkpoint['model_D_state'])

# 3. Restore optimizer state (IMPORTANT!)
optimizer.load_state_dict(checkpoint['optimizer_state'])
optimizer_D.load_state_dict(checkpoint['optimizer_D_state'])

# 4. Get iteration counter
start_iter = checkpoint['iteration']  # 12,300

# 5. Resume training loop
for i_iter in range(start_iter, num_steps):  # 12,300 → 50,000
    # Training continues...
```

**Result**: Training continues seamlessly from iteration 12,300 → 50,000

#### **Edge Case: Corrupted Checkpoint**

If `latest_checkpoint.pth` is corrupted (e.g., power loss during save):

**Fallback strategy:**
1. Use most recent periodic checkpoint: `checkpoint_10000.pth`
2. Or use second-latest: `checkpoint_5000.pth`
3. Or start from `best_model.pth` (weights only, no optimizer state)

**Command to resume from fallback:**
```powershell
--restore-from "C:/_albert/ALS4GAN/checkpoints/checkpoint_10000.pth"
```

#### **Checkpoint File Sizes**

Typical file sizes:
- `latest_checkpoint.pth`: ~500 MB (full checkpoint)
- `checkpoint_10000.pth`: ~500 MB (full checkpoint)
- `best_model.pth`: ~250 MB (weights only)
- `latest_model.pth`: ~250 MB (weights only)

**Disk space needed**: ~5 GB for all checkpoints (10 periodic + latest + best + final)

#### **Checkpoint Cleanup**

After training completes, you can delete intermediate checkpoints to save space:

**Keep:**
- ✅ `best_model.pth` - Best performing model
- ✅ `best_model_D.pth` - Corresponding discriminator
- ✅ `final_model.pth` - Final model
- ✅ `checkpoint_50000.pth` - Last full checkpoint (for extending training)

**Can delete:**
- ❌ `latest_checkpoint.pth` - No longer needed
- ❌ `checkpoint_5000.pth`, `checkpoint_10000.pth`, ... - Intermediate backups
- ❌ `latest_model.pth`, `latest_model_D.pth` - Redundant with final model

---

## 🔧 Advanced: Modifying Training Mid-Stream

### Change Batch Size After Resume

You **can** change batch size when resuming:

```powershell
# Original training with batch size 4
--batch-size 4

# Resume with batch size 16 (faster)
--restore-from "checkpoints/latest_checkpoint.pth" --batch-size 16
```

**Effect**: Training speed changes, but model convergence continues normally.

### Change Learning Rate After Resume

You **cannot** easily change learning rate because:
- Optimizer state contains learning rate schedule
- Polynomial decay depends on iteration number

**Workaround**: If you need to adjust, modify the code or train longer with extended iterations.

---

## 💾 Checkpoints Saved (Summary)

### Automatic Saves During Training
| File | Frequency | Contains | Purpose |
|------|-----------|----------|---------|
| `latest_checkpoint.pth` | Every 100 iter | Full checkpoint | Resume after crashes |
| `latest_model.pth` | Every 100 iter | Weights only | Quick model access |
| `checkpoint_N.pth` | Every 5000 iter | Full checkpoint | Periodic backups |
| `best_model.pth` | When val mIoU improves | Weights only | **Best model for inference** |
| `final_model.pth` | End of training | Weights only | Final trained model |

**Use `best_model.pth` for testing/deployment** (not final_model.pth)!

---

### Warning Signs
⚠️ Validation mIoU plateaus while training mIoU increases → **Overfitting**  
⚠️ Both mIoUs stay low (< 0.3) → **Underfitting** (increase model capacity or train longer)  
⚠️ ST_Count always 0 → Threshold too high (reduce `--threshold-st`)  
⚠️ Losses oscillating wildly → Learning rate too high  

---

## 🔧 Troubleshooting

### "CUDA out of memory"
→ Reduce `--batch-size` to 2 or 1

### "Wandb login failed"
→ Get API key from https://wandb.ai/authorize
→ Or use `--no-wandb` to disable wandb

### "Dataset initialization failed"
→ Run test_dataloader.py first to verify dataset
→ Check that salak-1-* folders exist

### "Low validation mIoU"
→ Train longer (increase `--num-steps`)
→ Adjust `--threshold-st` (try 0.1 or 0.3)
→ Increase batch size if possible

---

## 📊 Understanding the Wandb Charts

### Chart 1: "Training Loss/Cross Entropy"
- **What**: Supervised learning loss on labeled data
- **Good**: Steadily decreasing
- **Target**: < 0.5 after 10k iterations

### Chart 2: "Metrics/Training mIoU"
- **What**: Performance on training set
- **Good**: Increasing over time
- **Target**: > 0.7 (70%)

### Chart 3: "Metrics/Validation mIoU" ⭐ MOST IMPORTANT
- **What**: Performance on held-out validation set
- **Good**: Increasing and close to training mIoU
- **Target**: > 0.6 (60%)
- **Use for**: Detecting overfitting

### Chart 4: "Self-Training/Confidence Count"
- **What**: Number of unlabeled samples used per iteration
- **Good**: > 0 and stable
- **Meaning**: Semi-supervised learning is active

---

## 🎓 Advanced Tips

### Faster Training
```powershell
--batch-size 16 `
--eval-every 2000  # Less frequent evaluation
```

### More Conservative (Less Overfitting)
```powershell
--threshold-st 0.3 `  # Higher threshold
--val-split 0.3  # More validation data
```

### More Aggressive (Faster Learning)
```powershell
--threshold-st 0.1 `  # Lower threshold
--lambda-st 2.0  # Higher self-training weight
```

---

## 📝 Next Steps After Training

1. **Check best_model.pth validation mIoU**
   - If > 0.6: Great! Ready for inference
   - If 0.4-0.6: Okay, might need more training
   - If < 0.4: Need to debug (check data, hyperparameters)

2. **Visualize predictions** (we'll create inference script)

3. **Fine-tune** if needed with different hyperparameters

4. **Deploy** the model for full dataset inference

---

## 🚀 Ready to Train?

1. ✅ Test dataset with `test_dataloader.py`
2. ✅ Verify visualizations look correct
3. ✅ Run training command above
4. ✅ Monitor Wandb dashboard
5. ✅ Wait ~12 hours for completion

**Good luck! 🎯**

---

*For questions or issues, check the console output and Wandb logs first.*
