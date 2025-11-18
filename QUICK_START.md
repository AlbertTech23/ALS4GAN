# Quick Start - Testing Your Dataset

## ⚡ Quick Commands

### 1. Navigate to project
```powershell
cd C:\_albert\ALS4GAN
```

### 2. Run the data loader test
```powershell
C:\Users\IoT-C504-03\miniconda3\envs\als4gan_env\python.exe data\test_dataloader.py
```

---

## 📝 Before You Run

### Update these paths in `data/test_dataloader.py` (line ~30):

```python
# Where your patch images and masks are located
DATA_ROOT = r"C:/_albert/s4GAN/patchify/temp_patches"  # ← CHANGE THIS!

# These should already be correct:
LABELED_LIST = r"C:/_albert/ALS4GAN/labeled_files_50.txt"
CLASS_MAPPING = r"C:/_albert/ALS4GAN/class_mapping.csv"
NUM_CLASSES = 7
```

---

## 📁 Your Dataset Structure Should Be:

```
{DATA_ROOT}/
├── images/
│   ├── DJI_101_0155.JPG
│   ├── DJI_101_0175.JPG
│   └── ... (all patch images)
└── masks/
    ├── DJI_101_0155_mask.png  (or DJI_101_0155.png)
    ├── DJI_101_0175_mask.png
    └── ... (corresponding masks)
```

---

## ✅ What the Test Does

1. Checks if all paths exist ✓
2. Loads your class mapping from CSV ✓
3. Initializes the custom dataset ✓
4. Creates a PyTorch DataLoader ✓
5. Loads 3 batches (12 images total) ✓
6. Validates shapes and values ✓
7. Saves visualizations to `data/test_output/` ✓

---

## 🎯 What Success Looks Like

```
============================================================
✓✓✓ ALL TESTS PASSED! ✓✓✓
============================================================

Your dataset loader is working correctly!
Visualization saved to: C:/_albert/ALS4GAN/data/test_output

Next steps:
  1. Review the visualizations
  2. Verify masks align with images
  3. Proceed with training using train_s4gan.py
```

---

## 🔍 Check Your Output

After running, check:
- `data/test_output/visualization_*.png` - Visual check of image + mask
- `data/test_output/distribution_*.txt` - Class distribution stats

Make sure:
- ✓ Images are not corrupted
- ✓ Masks align with images
- ✓ All 7 classes appear
- ✓ No unexpected class labels

---

## 🚨 Common Issues

### "Data root does not exist"
→ Update `DATA_ROOT` path in `test_dataloader.py`

### "Failed to load image"
→ Check if images are in `{DATA_ROOT}/images/` folder
→ Verify filename matches `labeled_files_50.txt`

### "Failed to load label"
→ Check if masks are in `{DATA_ROOT}/masks/` folder
→ Ensure mask names match: `{image_name}_mask.png` or `{image_name}.png`

### "Found labels outside valid range"
→ Mask colors don't match `class_mapping.csv`
→ Check RGB values are exact matches

---

## 📊 Your Classes (from class_mapping.csv)

| Index | Class Name      | RGB Color        |
|-------|-----------------|------------------|
| 0     | __background__  | (0, 0, 0)        |
| 1     | Badan Air       | (255, 50, 50)    |
| 2     | Bangunan        | (255, 225, 50)   |
| 3     | Jalan           | (109, 255, 50)   |
| 4     | Pohon Berinang  | (50, 255, 167)   |
| 5     | Snake Fruit     | (50, 167, 255)   |
| 6     | Tanah Terbuka   | (109, 50, 255)   |

---

## 📂 Files Created

1. **`data/custom_dataset.py`** - Your dataset loader class
2. **`data/test_dataloader.py`** - Test script (run this!)
3. **`DATASET_SETUP_GUIDE.md`** - Detailed guide
4. **`QUICK_START.md`** - This file!

---

## 🎓 Understanding Your Setup

**Your Dataset:**
- ~370k total patches
- 50 labeled patches (listed in `labeled_files_50.txt`)
- ~370k - 50 unlabeled patches
- 7 semantic classes

**S4GAN Training (next step):**
- Uses 50 labeled patches for supervised learning
- Uses unlabeled patches for semi-supervised learning
- Discriminator helps improve predictions
- No Active Learning needed!

---

## 🚀 Next Steps After Test Passes

Once the test is successful:

1. **Review visualizations** - Make sure everything looks good
2. **I'll help you** create a training script
3. **Modify train_s4gan.py** to use your custom dataset
4. **Start training** the S4GAN model

---

## 💬 Ready?

**Run the test now:**
```powershell
C:\Users\IoT-C504-03\miniconda3\envs\als4gan_env\python.exe data\test_dataloader.py
```

**Tell me:**
1. Does it pass? ✓ or ✗
2. What's the output?
3. Do visualizations look correct?

Then we'll proceed to training! 🎯

---

*For detailed explanations, see `DATASET_SETUP_GUIDE.md`*
