# ALS4GAN Source Code Analysis Summary

## 🎯 Project Overview

**ALS4GAN** = Active Learning for Semi-Supervised Semantic Segmentation with GANs

**Your Goal**: Use only the **S4GAN** (Semi-Supervised GAN) part, skipping Active Learning.

---

## 🏗️ Architecture

### 1. Generator (Segmentation Network)
**File**: `model/deeplabv2.py`
- **Architecture**: DeepLabV2 with ResNet-101 backbone
- **Input**: RGB images (320×320)
- **Output**: Semantic segmentation maps (7 classes)
- **Key Features**:
  - Atrous Spatial Pyramid Pooling (ASPP)
  - Multi-scale predictions
  - Pre-trained on ImageNet

### 2. Discriminator
**File**: `model/discriminator.py`
- **Purpose**: Distinguish real vs. fake segmentation maps
- **Input**: Concatenated [softmax predictions + RGB image]
- **Output**: 
  - Classification score (0-1): fake vs. real
  - Feature maps for Feature Matching loss

### 3. Training Strategy
**File**: `tools/train_s4gan.py`

**Three Data Streams**:
1. **Labeled data** (`trainloader`) - For supervised learning
2. **Unlabeled data** (`trainloader_remain`) - For self-training
3. **Ground truth** (`trainloader_gt`) - For discriminator real examples

**Four Losses**:
1. **Cross-Entropy (CE)**: Supervised loss on labeled data
   ```python
   loss_ce = CrossEntropy2d(pred_labeled, gt_labels)
   ```

2. **Feature Matching (FM)**: Match generator features to real data
   ```python
   loss_fm = L1(mean(D_features_real), mean(D_features_fake))
   ```

3. **Self-Training (ST)**: Use high-confidence predictions as pseudo-labels
   ```python
   if D_confidence > threshold_st:
       loss_st = CrossEntropy2d(pred_unlabeled, pseudo_labels)
   ```

4. **Adversarial (D)**: Train discriminator
   ```python
   loss_D = BCE(D(real), 1) + BCE(D(fake), 0)
   ```

**Total Generator Loss**:
```python
loss_G = loss_ce + λ_fm * loss_fm + λ_st * loss_st
```

---

## 📊 Data Flow

### Training Loop (Simplified)

```
For each iteration:
    
    1. TRAIN GENERATOR:
       ├─ Load labeled batch
       ├─ Forward pass → predictions
       ├─ Compute CE loss on labeled data
       │
       ├─ Load unlabeled batch  
       ├─ Forward pass → predictions
       ├─ Discriminator scores predictions
       ├─ Select high-confidence predictions (> threshold)
       ├─ Compute ST loss on selected predictions
       │
       ├─ Load GT batch for discriminator
       ├─ Compute FM loss (feature matching)
       │
       └─ Backprop: loss_ce + λ_fm*loss_fm + λ_st*loss_st
    
    2. TRAIN DISCRIMINATOR:
       ├─ Feed fake predictions → D → 0 (fake)
       ├─ Feed real GT maps → D → 1 (real)
       └─ Backprop: loss_D
```

### Data Loading

**Original Code** (supports UCM and DeepGlobe):
```python
if dataset_name == 'ucm':
    train_dataset = UCMDataSet(...)
elif dataset_name == 'deepglobe':
    train_dataset = DeepGlobeDataSet(...)
```

**Your Custom Dataset** (NEW):
```python
from data.custom_dataset import CustomDataSet

train_dataset = CustomDataSet(
    root=DATA_ROOT,
    list_path=LABELED_LIST,
    class_mapping_csv=CLASS_MAPPING,
    module='s4gan',
    crop_size=(320, 320),
    mean=(128, 128, 128)
)
```

---

## 🔑 Key Parameters

### From `train_s4gan.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--num-classes` | Required | Number of classes (7 for you) |
| `--threshold-st` | Required | Self-training threshold (0.0-1.0) |
| `--labeled-ratio` | None | Ratio of labeled data (use 50/370k) |
| `--lambda-fm` | 0.1 | Weight for feature matching loss |
| `--lambda-st` | 1.0 | Weight for self-training loss |
| `--learning-rate` | 2.5e-4 | Generator learning rate |
| `--learning-rate-D` | 1e-4 | Discriminator learning rate |
| `--batch-size` | 1 | Batch size |
| `--num-steps` | 40000 | Total training iterations |
| `--input-size` | 320,320 | Input image size |

---

## 📁 Dataset Classes

### DeepGlobeDataSet
```python
# For DeepGlobe Land Cover Classification dataset
- 6 classes: urban, agriculture, rangeland, forest, water, barren
- RGB masks → class indices via color mapping
- Expected structure:
  root/
    DeepGlobe_Images/{name}_sat.jpg
    DeepGlobe_Labels/{name}_mask.png
```

### UCMDataSet
```python
# For UC Merced Land Use dataset
- 21 classes: agricultural, airplane, beach, etc.
- Class from filename pattern (e.g., "agricultural05")
- Expected structure:
  root/
    UCMerced_Images/{name}.tif
    UCMerced_Labels/{name}.png
```

### CustomDataSet (YOUR NEW CLASS)
```python
# For your remote sensing dataset
- 7 classes: background, Badan Air, Bangunan, etc.
- RGB masks → class indices via class_mapping.csv
- Expected structure:
  root/
    images/{name}.JPG
    masks/{name}_mask.png
```

---

## 🧩 How Data is Processed

### Input Pipeline:

```
1. Read image (BGR)
   ├─ cv2.imread()
   └─ Resize to 320×320

2. Read mask (RGB)
   ├─ cv2.imread()
   ├─ Resize to 320×320
   └─ encode_segmap() → class indices

3. Preprocessing:
   ├─ image = image - mean  (mean subtraction)
   ├─ BGR → RGB conversion
   └─ HWC → CHW transpose

4. Data Augmentation (if enabled):
   ├─ Random scaling (0.5 - 1.5x)
   ├─ Random cropping
   └─ Random horizontal flip

5. Output:
   ├─ image: (C, H, W) float32 tensor
   ├─ label: (H, W) int tensor
   ├─ size: original size
   ├─ name: filename
   └─ index: sample index
```

### Class Encoding:

```python
def encode_segmap(mask):
    """Convert RGB mask to class indices"""
    # Input: (H, W, 3) RGB values
    # Output: (H, W) class indices [0, 1, 2, ..., 6]
    
    for class_idx, color in enumerate(class_colors):
        matches = np.all(mask == color, axis=-1)
        label_mask[matches] = class_idx
    
    return label_mask
```

---

## 🎛️ Training Configuration

### Hyperparameters (from paper):

- **Batch size**: 1 (due to memory constraints)
- **Learning rate (G)**: 2.5e-4 with polynomial decay
- **Learning rate (D)**: 1e-4
- **Optimizer (G)**: SGD with momentum 0.9
- **Optimizer (D)**: Adam with β=(0.9, 0.99)
- **Weight decay**: 5e-4
- **λ_fm**: 0.1 (feature matching)
- **λ_st**: 1.0 (self-training)
- **Threshold_st**: 0.2-0.6 (typical range)

### Label Ratio Handling:

**If `labeled_ratio` is provided**:
```python
partial_size = int(labeled_ratio * total_size)
train_ids = np.arange(total_size)
np.random.shuffle(train_ids)

labeled_ids = train_ids[:partial_size]
unlabeled_ids = train_ids[partial_size:]
```

**If `active_learning` is True** (SKIP THIS):
```python
# Uses active_learning_images_array to select labeled samples
# NOT NEEDED for your use case
```

**If neither** (your case with labeled_files_50.txt):
```python
# All samples in list are labeled
# No splitting needed
```

---

## 🔍 Understanding Self-Training

### Confidence-Based Selection:

```python
def find_good_maps(D_outs, pred_all, threshold_st):
    """
    Select high-confidence predictions for self-training
    
    Args:
        D_outs: Discriminator confidence scores [B]
        pred_all: Predictions on unlabeled data [B, C, H, W]
        threshold_st: Confidence threshold (e.g., 0.2)
    
    Returns:
        pred_sel: Selected predictions [N, C, H, W]
        label_sel: Pseudo labels [N, H, W]
        count: Number selected
    """
    
    indexes = [i for i, score in enumerate(D_outs) if score > threshold_st]
    
    if count > 0:
        # Convert predictions to pseudo labels
        label_sel = argmax(pred_sel, dim=1)
        return pred_sel, label_sel, count
    
    return None, None, 0
```

**Key Insight**: 
- Only predictions that "fool" the discriminator (D_out > threshold) are used
- These are considered high-quality pseudo labels
- Acts as automatic quality control for self-training

---

## 📦 Your Dataset Specifics

### Class Mapping:

| Index | Class Name | RGB | Notes |
|-------|------------|-----|-------|
| 0 | __background__ | (0,0,0) | Black |
| 1 | Badan Air | (255,50,50) | Reddish |
| 2 | Bangunan | (255,225,50) | Yellow |
| 3 | Jalan | (109,255,50) | Green |
| 4 | Pohon Berinang | (50,255,167) | Cyan |
| 5 | Snake Fruit | (50,167,255) | Blue |
| 6 | Tanah Terbuka | (109,50,255) | Purple |

### Data Split:

- **Total**: ~370k patches
- **Labeled**: 50 patches (~0.01%)
- **Unlabeled**: ~370k - 50 patches (~99.99%)

**This is PERFECT for semi-supervised learning!**

---

## 🚦 Current Status

### ✅ Completed:

1. ✓ Environment setup (als4gan_env)
2. ✓ CUDA verification (RTX 4060 Ti)
3. ✓ Dataset preparation (patches ready)
4. ✓ Class mapping defined
5. ✓ Labeled files list created

### 📝 Created Files:

1. ✓ **`data/custom_dataset.py`** - Your dataset loader
2. ✓ **`data/test_dataloader.py`** - Test script
3. ✓ **`DATASET_SETUP_GUIDE.md`** - Detailed guide
4. ✓ **`QUICK_START.md`** - Quick reference
5. ✓ **`SOURCE_CODE_ANALYSIS.md`** - This file

### ⏭️ Next Steps:

1. **Organize dataset** into proper folder structure
2. **Run test script** to verify data loading
3. **Review visualizations** to ensure correctness
4. **Modify train_s4gan.py** to use CustomDataSet
5. **Start training** with your configuration

---

## 🎯 Training Command (Preview)

Once ready, you'll run something like:

```powershell
C:\Users\IoT-C504-03\miniconda3\envs\als4gan_env\python.exe tools\train_s4gan.py `
  --dataset custom `
  --num-classes 7 `
  --data-dir "C:/_albert/ALS4GAN/data/remote_sensing_dataset" `
  --data-list "C:/_albert/ALS4GAN/labeled_files_50.txt" `
  --threshold-st 0.2 `
  --batch-size 4 `
  --num-steps 40000 `
  --checkpoint-dir "C:/_albert/ALS4GAN/checkpoints" `
  --restore-from "path/to/pretrained/resnet101.pth"
```

**Note**: You'll need a pretrained ResNet-101 model to start. We'll address this next.

---

## 🤔 Questions Answered

### Q: Can I skip Active Learning?
**A**: Yes! Just use `--labeled-ratio None` or don't provide it. The code will use all samples in `data-list` as labeled.

### Q: How do I use my custom dataset?
**A**: We created `CustomDataSet` class that loads from your folder structure.

### Q: What's the minimum GPU memory needed?
**A**: 
- Batch size 1: ~4GB
- Batch size 4: ~8GB
- Your RTX 4060 Ti (16GB) can handle batch size 8-16

### Q: How long will training take?
**A**: 
- 40k iterations × ~1-2 sec/iter = ~11-22 hours
- Checkpoints saved every 5k iterations

### Q: Do I need pretrained weights?
**A**: Yes, ResNet-101 pretrained on ImageNet. We'll download this.

---

## 📚 Key Insights

1. **Semi-supervised learning** works GREAT with small labeled sets
2. **Discriminator** acts as a quality filter for pseudo labels
3. **Feature matching** prevents mode collapse
4. **Self-training threshold** is critical (0.2 is a good start)
5. **Your dataset ratio** (0.01% labeled) is perfect for this approach

---

## 🎓 Understanding S4GAN vs. Supervised Learning

### Supervised Only:
```
50 labeled images → Train → Model
                    ↓
                  Limited generalization
```

### S4GAN (Semi-Supervised):
```
50 labeled images ────┐
                      ├─→ Train G → Model
370k unlabeled images ┘      ↕
                             D (discriminator)
                             
Better generalization!
```

**Why it works**:
- Labeled data provides ground truth
- Unlabeled data provides diversity
- Discriminator ensures quality
- Feature matching prevents overfitting

---

## 🛠️ Code Modifications Needed

To use your custom dataset, you'll need to modify `train_s4gan.py`:

### Change 1: Import custom dataset
```python
# Add at top
from data.custom_dataset import CustomDataSet
```

### Change 2: Add dataset option
```python
if dataset_name == 'ucm':
    train_dataset = UCMDataSet(...)
elif dataset_name == 'deepglobe':
    train_dataset = DeepGlobeDataSet(...)
elif dataset_name == 'custom':  # NEW
    train_dataset = CustomDataSet(
        root=args.data_dir,
        list_path=args.data_list,
        class_mapping_csv=args.class_mapping,  # NEW argument
        module='s4gan',
        crop_size=input_size,
        scale=args.random_scale,
        mirror=args.random_mirror,
        mean=IMG_MEAN
    )
else:
    raise NotImplementedError(...)
```

### Change 3: Add class mapping argument
```python
parser.add_argument("--class-mapping", type=str, required=True,
                    help="Path to class_mapping.csv")
```

**I can help you make these changes once the test passes!**

---

## 📖 References

- **Original Paper**: Semi-Supervised Semantic Segmentation with GANs
- **DeepLabV2**: "DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs"
- **ResNet**: "Deep Residual Learning for Image Recognition"

---

## ✨ Summary

You now have:
1. ✅ Understanding of how S4GAN works
2. ✅ Custom dataset loader for your data
3. ✅ Test script to verify everything
4. ✅ Clear path forward

**Next action**: Run the test script and let me know the results!

---

*Ready to test? See `QUICK_START.md` for commands!*
