# Dataset Setup and Testing Guide for ALS4GAN

This guide will help you set up your custom remote sensing dataset and test the S4GAN data loader (skipping the Active Learning part).

## 📋 Table of Contents
- [Understanding the Code](#understanding-the-code)
- [Dataset Requirements](#dataset-requirements)
- [Folder Structure Setup](#folder-structure-setup)
- [Testing the Data Loader](#testing-the-data-loader)
- [Next Steps: Training](#next-steps-training)
- [Troubleshooting](#troubleshooting)

---

## 🔍 Understanding the Code

### How S4GAN Works (from `train_s4gan.py`)

The S4GAN (Semi-Supervised Semantic Segmentation with GANs) training process:

1. **Generator (G)**: DeepLabV2 with ResNet-101 backbone
   - Takes images as input
   - Outputs semantic segmentation predictions

2. **Discriminator (D)**: Distinguishes between real and fake segmentation maps
   - **Real**: Ground truth masks from labeled data
   - **Fake**: Predictions from the generator on unlabeled data

3. **Training Losses**:
   - **Cross-Entropy Loss (CE)**: Supervised loss on labeled data
   - **Feature Matching Loss (FM)**: Matches features between real and fake
   - **Self-Training Loss (ST)**: Uses high-confidence predictions on unlabeled data
   - **Adversarial Loss (D)**: Discriminator loss

4. **Semi-Supervised Strategy**:
   - Uses a small portion of labeled data (2-5% in your case)
   - Leverages unlabeled data through self-training with confidence thresholding
   - Discriminator guides the generator to produce realistic segmentation maps

### Key Files

- **`data/custom_dataset.py`**: Your custom dataset loader (newly created)
- **`tools/train_s4gan.py`**: Main training script
- **`model/deeplabv2.py`**: Generator architecture
- **`model/discriminator.py`**: Discriminator architecture
- **`labeled_files_50.txt`**: List of labeled image filenames
- **`class_mapping.csv`**: Your class definitions and RGB colors

---

## 📦 Dataset Requirements

### What You Have
- **~370k patches** from ~1000 base images
- **7 classes** (including background):
  ```
  0: __background__ (0, 0, 0)
  1: Badan Air (255, 50, 50)
  2: Bangunan (255, 225, 50)
  3: Jalan (109, 255, 50)
  4: Pohon Berinang (50, 255, 167)
  5: Snake Fruit (50, 167, 255)
  6: Tanah Terbuka (109, 50, 255)
  ```
- **2-5% labeled** patches listed in `labeled_files_50.txt`

### Expected Format
- **Images**: RGB patches (any size, will be resized to 320×320)
- **Masks**: RGB masks where each pixel color corresponds to a class
- **File extensions**: `.jpg`, `.JPG`, `.png`, `.PNG`, `.tif`, `.TIF`

---

## 📁 Folder Structure Setup

### Option 1: Recommended Structure

Organize your dataset in the ALS4GAN folder:

```
C:/_albert/ALS4GAN/
├── data/
│   ├── custom_dataset.py         (✓ Created)
│   ├── test_dataloader.py        (✓ Created)
│   └── remote_sensing_dataset/   (← CREATE THIS)
│       ├── images/               (← Put patch images here)
│       │   ├── DJI_101_0155.JPG
│       │   ├── DJI_101_0175.JPG
│       │   └── ...
│       └── masks/                (← Put mask images here)
│           ├── DJI_101_0155_mask.png
│           ├── DJI_101_0175_mask.png
│           └── ...
├── labeled_files_50.txt          (✓ Already exists)
├── class_mapping.csv             (✓ Already exists)
└── checkpoints/                  (For saving models)
```

### Option 2: Use Existing Location

If you want to keep data at `C:/_albert/s4GAN/patchify/temp_patches`:

```
C:/_albert/s4GAN/patchify/temp_patches/
├── images/
│   ├── DJI_101_0155.JPG
│   └── ...
└── masks/
    ├── DJI_101_0155_mask.png
    └── ...
```

Then update the path in `test_dataloader.py`.

### Important Notes on File Naming

The dataset loader expects matching filenames between images and masks:

**Example 1** (with `_mask` suffix):
- Image: `DJI_101_0155.JPG`
- Mask: `DJI_101_0155_mask.png`

**Example 2** (same name, different folder):
- Image: `DJI_101_0155.JPG`
- Mask: `DJI_101_0155.png`

**The loader will try both conventions automatically.**

---

## 🧪 Testing the Data Loader

### Step 1: Organize Your Dataset

1. **Copy your patch images** to the `images/` folder
2. **Copy your mask images** to the `masks/` folder
3. **Verify** that filenames in `labeled_files_50.txt` match your actual files

### Step 2: Update Test Configuration

Edit `data/test_dataloader.py` and update these lines (around line 30):

```python
# Path to your dataset root (contains images/ and masks/ folders)
DATA_ROOT = r"C:/_albert/ALS4GAN/data/remote_sensing_dataset"  # ← UPDATE THIS

# Path to labeled files list
LABELED_LIST = r"C:/_albert/ALS4GAN/labeled_files_50.txt"  # Already correct

# Path to class mapping CSV
CLASS_MAPPING = r"C:/_albert/ALS4GAN/class_mapping.csv"  # Already correct

# Number of classes (should match your class_mapping.csv)
NUM_CLASSES = 7  # Already correct
```

### Step 3: Run the Test

Open PowerShell and run:

```powershell
cd C:\_albert\ALS4GAN

C:\Users\IoT-C504-03\miniconda3\envs\als4gan_env\python.exe data\test_dataloader.py
```

### Step 4: Check the Output

The test will:

1. ✓ Check if all paths exist
2. ✓ Load class mapping from CSV
3. ✓ Initialize the dataset
4. ✓ Create a DataLoader
5. ✓ Load 3 batches of data
6. ✓ Validate data shapes and values
7. ✓ Generate visualizations

**Expected Output:**

```
============================================================
Testing Custom Remote Sensing Dataset Loader
============================================================

Configuration:
  Data root: C:/_albert/ALS4GAN/data/remote_sensing_dataset
  Labeled list: C:/_albert/ALS4GAN/labeled_files_50.txt
  Class mapping: C:/_albert/ALS4GAN/class_mapping.csv
  Number of classes: 7
  Batch size: 4

Checking paths...
  ✓ Data root exists
  ✓ Labeled list file exists
  ✓ Class mapping file exists

CUDA Information:
  CUDA available: True
  CUDA device: NVIDIA GeForce RTX 4060 Ti
  CUDA version: 12.9

============================================================
Initializing Dataset...
============================================================
Loaded 7 classes:
  __background__: [0, 0, 0]
  Badan Air: [255, 50, 50]
  ...

✓ Dataset initialized successfully!
  Total samples: 50

Creating DataLoader...
✓ DataLoader created successfully!

============================================================
Testing Data Loading (loading 3 batches)...
============================================================

Batch 1/3:
  Images shape: torch.Size([4, 3, 320, 320])
  Labels shape: torch.Size([4, 320, 320])
  ...
  ✓ Batch 1 loaded successfully!

...

============================================================
✓✓✓ ALL TESTS PASSED! ✓✓✓
============================================================

Your dataset loader is working correctly!
Visualization saved to: C:/_albert/ALS4GAN/data/test_output
```

### Step 5: Review Visualizations

Check the output folder: `C:/_albert/ALS4GAN/data/test_output/`

You should see:
- `visualization_DJI_101_0155.png` - Shows image, label indices, and colored mask
- `distribution_DJI_101_0155.txt` - Shows class distribution statistics

**Verify that**:
- Images look correct
- Masks are properly aligned with images
- Class colors match your expectations
- No unexpected classes appear

---

## 🚀 Next Steps: Training

Once the data loader test passes, you can modify `train_s4gan.py` to use your custom dataset.

### Create a Training Script

I'll help you create a modified training script that:
1. Uses your custom dataset
2. Skips Active Learning
3. Uses your configuration

Would you like me to:
1. Create a training wrapper script?
2. Show you how to modify `train_s4gan.py`?
3. Create a simple training config file?

---

## 🔧 Troubleshooting

### Problem: "Failed to load image: ..."

**Cause**: Image file not found or wrong path

**Solutions**:
1. Check if the file exists at the specified path
2. Verify filename in `labeled_files_50.txt` matches actual file
3. Check file extension (`.JPG` vs `.jpg`)
4. Ensure images are in the `images/` subfolder

### Problem: "Failed to load label: ..."

**Cause**: Mask file not found

**Solutions**:
1. Check mask filename convention:
   - Try `{name}_mask.png`
   - Try `{name}.png`
2. Verify masks are in the `masks/` subfolder
3. Check if mask files exist for all images in `labeled_files_50.txt`

### Problem: "Found labels outside valid range"

**Cause**: Mask contains colors not in `class_mapping.csv`

**Solutions**:
1. Check if mask RGB values exactly match those in `class_mapping.csv`
2. Verify no anti-aliasing or compression artifacts in masks
3. Ensure masks are saved as PNG (lossless)

### Problem: Import errors when running test

**Cause**: Missing dependencies in conda environment

**Solutions**:
```powershell
# Activate environment
conda activate als4gan_env

# Install missing packages
conda install pandas matplotlib pillow opencv
```

### Problem: CUDA out of memory

**Cause**: Batch size too large for your GPU

**Solutions**:
1. Reduce `BATCH_SIZE` in test script (try 2 or 1)
2. Reduce `crop_size` to (256, 256) or (224, 224)

---

## 📝 Summary Checklist

Before proceeding to training:

- [ ] Dataset organized in correct folder structure
- [ ] All files in `labeled_files_50.txt` exist in `images/` folder
- [ ] Corresponding masks exist in `masks/` folder
- [ ] File naming convention is consistent
- [ ] `class_mapping.csv` matches your mask colors
- [ ] Test script runs without errors
- [ ] Visualizations look correct
- [ ] CUDA is available and working
- [ ] All class labels are in valid range [0-6]

---

## 💡 Questions?

If you have issues:

1. **Check the error message** - often tells you exactly what's wrong
2. **Verify paths** - most issues are path-related
3. **Check file names** - case-sensitive on some systems
4. **Review visualizations** - ensure masks are correct
5. **Ask for help** - provide the full error message

**Ready to train?** Let me know if the test passes, and I'll help you set up training!

---

## 🎯 Key Takeaways

1. **S4GAN = Generator + Discriminator** for semi-supervised learning
2. **Small labeled set** (2-5%) + **Large unlabeled set** (95-98%)
3. **Self-training** uses high-confidence predictions as pseudo-labels
4. **Feature matching** ensures realistic segmentation maps
5. **No Active Learning needed** - we'll use the labeled set as-is

---

*Last updated: November 10, 2025*
