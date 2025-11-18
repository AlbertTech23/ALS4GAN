# Training Analysis - First 2000 Iterations

**Date**: November 11, 2025  
**Iterations Analyzed**: 0 → 2000  
**Dataset**: 370,647 patches (12,856 labeled, 357,791 unlabeled)  
**Batch Size**: 4  
**Model**: S4GAN (DeepLabV2-ResNet101)

---

## 📊 Performance Summary

### **Validation mIoU Trend**
| Iteration | Train mIoU | Val mIoU | Status |
|-----------|-----------|----------|--------|
| 500  | 0.1438 | **0.1436** | ⬆️ Baseline |
| 1000 | 0.1826 | **0.1802** | ⬆️ **+25.5% improvement** (BEST) |
| 1500 | 0.1550 | **0.1536** | ⬇️ -14.8% (decline) |
| 2000 | 0.1189 | **0.1151** | ⬇️ -25.1% (decline) |

**🔴 CRITICAL FINDING**: Model peaked at iteration 1000 and has been declining for 1000 iterations.

---

## 🔍 Loss Analysis

### **Cross Entropy Loss (Loss_CE)**
```
Iter 0:    3.483  (baseline)
Iter 500:  1.412  ⬇️ -59.4%
Iter 1000: 2.572  ⬆️ +82.1% (spike)
Iter 1500: 1.475  ⬇️ -42.6%
Iter 2000: 1.225  ⬇️ -17.0%
```
**Analysis**: 
- Good initial reduction (0→500)
- **Concerning spike at 1000** despite best mIoU
- Currently stabilizing around 1.2-1.5 range
- Target: Should be < 0.5 by 10k iterations

### **Feature Matching Loss (Loss_FM)**
```
Iter 0:    0.001
Iter 500:  0.032  ⬆️ +3100%
Iter 1000: 0.029  
Iter 1500: 0.026  ⬇️ Stable
Iter 2000: 0.079  ⬆️ +172% (spike)
```
**Analysis**:
- Increasing from near-zero to 0.03-0.08 range
- Large spike at 2000 (0.079) indicates generator-discriminator mismatch
- This is actually expected in early training (discriminator learning features)

### **Self-Training Loss (Loss_ST)**
```
Iter 0-500:   0.000  (no confident predictions)
Iter 1100:    0.295  (first activation)
Iter 1200:    0.224
Iter 1500:    0.188
Iter 1600+:   0.000  (deactivated again)
```
**Analysis**:
- Brief activation around iterations 1100-1500
- **Currently inactive** (threshold=0.2 too strict?)
- ST_Count = 0 at iter 1600-2000 means no unlabeled data used
- **This is a problem** - not leveraging the 357k unlabeled patches

### **Discriminator Loss (Loss_D)**
```
Iter 0:    0.693  (random baseline ~ln(2))
Iter 500:  0.174  ⬇️ Strong learning
Iter 1000: 0.708  ⬆️ Reset (discriminator confused)
Iter 1500: 0.315  
Iter 2000: 0.079  ⬇️ Very low
```
**Analysis**:
- Loss_D=0.079 is **TOO LOW** = discriminator winning too hard
- Ideal range: 0.3-0.5 (balanced adversarial game)
- When D wins, generator can't learn from adversarial signal

### **Self-Training Count**
```
Iter 0-500:   3-4 pixels per batch
Iter 1000:    3 pixels
Iter 1100-1500: 2-4 pixels (ST active)
Iter 1600-2000: 0 pixels (ST dead)
```
**Analysis**:
- Extremely low confidence counts
- With threshold=0.2, model finds almost no confident predictions
- **Problem**: 357k unlabeled patches are not being used

---

## 🚨 Key Problems Identified

### **1. Self-Training Not Working** (CRITICAL)
**Symptoms:**
- Loss_ST = 0.000 for most iterations
- ST_Count = 0 (no confident predictions)
- Only using 12,856 labeled patches, ignoring 357,791 unlabeled

**Root Cause:**
- Threshold 0.2 is too conservative for early training
- Model confidence is low, so no pseudo-labels generated

**Impact:**
- Not benefiting from semi-supervised learning
- Essentially running as supervised-only training
- Missing the core advantage of S4GAN

### **2. Discriminator Imbalance**
**Symptoms:**
- Loss_D dropping to 0.079 (too low)
- Large FM loss spikes (0.079 at iter 2000)

**Root Cause:**
- Discriminator learning too fast relative to generator
- Generator can't fool discriminator → poor adversarial signal

**Impact:**
- Generator not improving from adversarial training
- Feature matching loss unstable

### **3. Validation Performance Collapse**
**Symptoms:**
- Best mIoU at iteration 1000 (0.1802)
- Declined 36% by iteration 2000 (0.1151)
- Training mIoU also declining

**Root Cause:**
- Without self-training, overfitting to 12,856 labeled samples
- Discriminator dominance preventing generator improvement
- Possible learning rate too high causing instability

**Impact:**
- Model getting worse over time
- Risk of complete training failure if continues

---

## ✅ Recommended Immediate Actions

### **Priority 1: Enable Self-Training** 🔥
**Current:** `--threshold-st 0.2`  
**Recommended:** `--threshold-st 0.1` or even `0.05`

**Rationale:**
- At 0.2, ST is dead (0 count)
- Lower threshold will activate pseudo-labeling
- Original S4GAN paper uses dynamic thresholds or lower values early

**Expected Impact:**
- ST_Count should increase to 1000-5000 per batch
- Loss_ST will activate and contribute to training
- Start leveraging unlabeled data

### **Priority 2: Rebalance Discriminator** 🔥
**Options:**
- Lower discriminator learning rate (currently tied to generator LR)
- Increase discriminator update interval (update D every N iterations instead of every iteration)
- Add label smoothing for discriminator (already in code?)

**Recommended:** Check discriminator LR schedule in code

### **Priority 3: Learning Rate Adjustment**
**Current behavior:** Both models using same schedule  
**Recommendation:**
- Check `lr_scheduler.py` - might be decaying too fast
- Consider warmup schedule for first 2k iterations
- Discriminator LR should be lower than generator

### **Priority 4: Monitor for Early Stopping**
**Action:** If mIoU continues declining past iteration 3000:
- Stop training
- Load best checkpoint (iteration 1000)
- Restart with adjusted hyperparameters

---

## 📈 Expected vs Actual Performance

### **Iteration 2000 - Expected:**
- Val mIoU: **0.25-0.35** (if ST working)
- Loss_CE: **1.5-2.0**
- Loss_ST: **0.3-0.6** (active)
- ST_Count: **2000-5000** pixels/batch

### **Iteration 2000 - Actual:**
- Val mIoU: **0.1151** ❌ (61% below expected)
- Loss_CE: **1.225** ✓ (acceptable)
- Loss_ST: **0.000** ❌ (completely inactive)
- ST_Count: **0** ❌ (not using unlabeled data)

---

## 🎯 Revised Training Strategy

### **Option A: Continue with Threshold Adjustment** (Recommended)
1. **Stop current training** (no point continuing if ST is dead)
2. **Restart from iteration 1000 checkpoint** (best model)
3. **Change threshold**: `--threshold-st 0.1`
4. **Monitor**: ST should activate within 100 iterations
5. **Target**: ST_Count > 1000 by iteration 1500

### **Option B: Debug Self-Training Code**
Check `train_s4gan_salak.py` around self-training logic:
- Is confidence calculation correct?
- Are predictions normalized (softmax)?
- Is threshold applied to probability or logits?

### **Option C: Switch to Time-Based Threshold**
Modify code to reduce threshold over time:
```python
threshold_st = max(0.05, 0.2 - (i_iter / 10000) * 0.15)  # 0.2 → 0.05 over 10k iters
```

---

## 📊 Success Criteria for Next 1000 Iterations

**By Iteration 3000, you should see:**
✓ ST_Count > 500 (if threshold=0.1)  
✓ Loss_ST = 0.2-0.5 (active contribution)  
✓ Val mIoU > 0.20 (improvement from current)  
✓ Train/Val gap < 0.05 (not overfitting)  
✓ Loss_D = 0.3-0.5 (balanced adversarial)  

**Red Flags to Stop Training:**
❌ Val mIoU < 0.10 (getting worse)  
❌ ST_Count = 0 for 500 consecutive iterations  
❌ Loss_D < 0.05 (discriminator too strong)  
❌ Train mIoU - Val mIoU > 0.15 (overfitting)  

---

## 🔬 Detailed Recommendations

### **1. Self-Training Threshold Analysis**

**Current threshold (0.2):**
- Requires 20% confidence to use pseudo-label
- At iteration 2000, model has ~14% average confidence → no ST
- Too conservative for early training

**Recommended threshold schedule:**
```
Iter 0-5000:    threshold = 0.05  (very permissive)
Iter 5000-15k:  threshold = 0.10  (moderate)
Iter 15k-30k:   threshold = 0.15  (strict)
Iter 30k-50k:   threshold = 0.20  (very strict)
```

**Why:** Model confidence increases over training. Start permissive, increase strictness.

### **2. Check Softmax Temperature**

Look for this in the code:
```python
# Pseudo-label generation
probs = F.softmax(outputs_remain / temperature, dim=1)
```

If temperature > 1.0, it reduces confidence → harder to pass threshold.  
**Recommendation:** Use temperature=1.0 for first 10k iterations.

### **3. Discriminator Learning Rate**

Check if discriminator uses same LR as generator:
```python
lr_D = args.learning_rate_D or args.learning_rate * 0.1
```

**Recommendation:** Discriminator LR should be 0.1x generator LR (one magnitude lower).

### **4. Batch Size vs Threshold Interaction**

With batch_size=4 and threshold=0.2:
- Each batch has 4 × 256 × 256 = 262,144 pixels
- ST_Count=0 means 0/262,144 pixels meet threshold
- Even 1% of pixels = 2,621 pixels (should see this number)

**Conclusion:** Model has <0.0001% pixels above 20% confidence = extremely low confidence.

---

## 💡 Next Steps

1. **Immediate:** Stop current training (it's declining)
2. **Restart:** Load checkpoint from iteration 1000 (best model so far)
3. **Adjust:** Change `--threshold-st 0.2` to `--threshold-st 0.05`
4. **Monitor:** Watch for ST_Count to become > 0 within 100 iterations
5. **Evaluate:** Run evaluation every 500 iterations to catch peak performance
6. **Decide:** By iteration 3000:
   - If improving → continue with current settings
   - If still declining → investigate discriminator LR or data issues

---

## 📝 Code Changes Needed

### **Quick Fix (Restart with Different Threshold):**
```powershell
# Just change one flag in your launch command:
--threshold-st 0.05  # instead of 0.2
```

### **Better Fix (Dynamic Threshold):**
Add to training loop in `train_s4gan_salak.py`:
```python
# Around line 300, before self-training block
threshold_st = max(0.05, args.threshold_st - (i_iter / 25000) * 0.15)
# This goes: 0.2 → 0.05 over 25k iterations
```

---

## 🎓 Learning Points

1. **Self-training needs tuning:** Threshold is the most critical hyperparameter
2. **Monitor ST_Count:** If it's 0, you're not doing semi-supervised learning
3. **Best model ≠ latest model:** Best was at 1000, not 2000
4. **Discriminator balance matters:** Loss_D should stay in 0.3-0.5 range
5. **Early stopping is valid:** Better to stop and adjust than continue declining

---

**Conclusion:** Current training is NOT using 96% of your data (unlabeled patches). Fix self-training threshold to unlock the full power of semi-supervised learning. Restart from iteration 1000 with threshold=0.05.
