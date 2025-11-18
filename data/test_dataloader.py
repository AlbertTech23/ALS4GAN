"""
Test script for Salak Dataset Loader
This script tests if the data loader can correctly read your salak remote sensing dataset

Usage:
    C:\\Users\\IoT-C504-03\\miniconda3\\envs\\als4gan_env\\python.exe test_dataloader.py
"""

import sys
import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from torch.utils import data

# Add parent directory to path to import salak_dataset
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.salak_dataset import SalakDataSet


def test_dataloader():
    """Test the custom dataset loader"""
    
    print("="*60)
    print("Testing Salak Remote Sensing Dataset Loader")
    print("="*60)
    
    # ========================================
    # CONFIGURATION - MODIFY THESE PATHS
    # ========================================
    
    # Path to your dataset root (contains salak-1-1, salak-1-2, ..., salak-1-6 folders)
    DATA_ROOT = r"C:/_albert/s4GAN/patchify/temp_patches"
    
    # Path to labeled files list
    LABELED_LIST = r"C:/_albert/ALS4GAN/labeled_files_50.txt"
    
    # Path to class mapping CSV
    CLASS_MAPPING = r"C:/_albert/ALS4GAN/class_mapping.csv"
    
    # Number of classes (should match your class_mapping.csv)
    NUM_CLASSES = 7  # Including __background__
    
    # Batch size for testing
    BATCH_SIZE = 4
    
    # Number of batches to test
    NUM_TEST_BATCHES = 3
    
    # Output directory for visualization
    OUTPUT_DIR = r"C:/_albert/ALS4GAN/data/test_output"
    
    # ========================================
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"\nConfiguration:")
    print(f"  Data root: {DATA_ROOT}")
    print(f"  Labeled list: {LABELED_LIST}")
    print(f"  Class mapping: {CLASS_MAPPING}")
    print(f"  Number of classes: {NUM_CLASSES}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Output directory: {OUTPUT_DIR}")
    
    # Check if paths exist
    print(f"\nChecking paths...")
    if not os.path.exists(DATA_ROOT):
        print(f"  ❌ ERROR: Data root does not exist: {DATA_ROOT}")
        print(f"     Please update DATA_ROOT in this script to point to your dataset folder")
        return False
    else:
        print(f"  ✓ Data root exists")
    
    if not os.path.exists(LABELED_LIST):
        print(f"  ❌ ERROR: Labeled list file does not exist: {LABELED_LIST}")
        return False
    else:
        print(f"  ✓ Labeled list file exists")
    
    if not os.path.exists(CLASS_MAPPING):
        print(f"  ❌ ERROR: Class mapping file does not exist: {CLASS_MAPPING}")
        return False
    else:
        print(f"  ✓ Class mapping file exists")
    
    # Check CUDA availability
    print(f"\nCUDA Information:")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")
    
    # Create dataset
    print(f"\n{'='*60}")
    print("Initializing Dataset...")
    print(f"{'='*60}")
    
    try:
        # Test 1: Load patches from labeled list only (50 base images)
        print("\n🔍 TEST 1: Loading patches from labeled list (50 base images)...")
        dataset_labeled = SalakDataSet(
            root=DATA_ROOT,
            list_path=LABELED_LIST,
            class_mapping_csv=CLASS_MAPPING,
            module='s4gan',
            crop_size=(256, 256),
            mean=(128, 128, 128),
            scale=False,
            mirror=False,
            ignore_label=255
        )
        print(f"\n✓ Labeled dataset initialized!")
        print(f"  Total samples: {len(dataset_labeled)}")
        
        # Test 2: Load ALL patches (no labeled list)
        print(f"\n{'='*60}")
        print(f"🔍 TEST 2: Loading ALL patches (full dataset)...")
        print(f"{'='*60}")
        dataset_all = SalakDataSet(
            root=DATA_ROOT,
            list_path=None,  # No labeled list = load everything
            class_mapping_csv=CLASS_MAPPING,
            module='s4gan',
            crop_size=(256, 256),
            mean=(128, 128, 128),
            scale=False,
            mirror=False,
            ignore_label=255
        )
        print(f"\n✓ Full dataset initialized!")
        print(f"  Total samples: {len(dataset_all)}")
        
        # Use the full dataset for detailed testing
        dataset = dataset_all
        
    except Exception as e:
        print(f"\n❌ ERROR: Failed to initialize dataset")
        print(f"   {type(e).__name__}: {e}")
        print(f"\nPossible issues:")
        print(f"  1. Check if your dataset folder structure matches:")
        print(f"     {DATA_ROOT}/")
        print(f"       salak-1-1/")
        print(f"         images/")
        print(f"           DJI_101_0155.JPG")
        print(f"           ...")
        print(f"         masks/")
        print(f"           DJI_101_0155_mask.png")
        print(f"           ...")
        print(f"       salak-1-2/")
        print(f"         images/")
        print(f"         masks/")
        print(f"       ...")
        print(f"  2. Verify image/mask file naming convention")
        print(f"  3. Check file extensions (.jpg, .JPG, .png, etc.)")
        import traceback
        traceback.print_exc()
        return False
    
    print(f"\n✓ Dataset initialized successfully!")
    print(f"  Total samples: {len(dataset)}")
    
    # Create data loader
    print(f"\nCreating DataLoader...")
    try:
        dataloader = data.DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )
        print(f"✓ DataLoader created successfully!")
    except Exception as e:
        print(f"❌ ERROR: Failed to create DataLoader: {e}")
        return False
    
    # Test loading batches
    print(f"\n{'='*60}")
    print(f"Testing Data Loading (loading {NUM_TEST_BATCHES} batches)...")
    print(f"{'='*60}")
    
    for batch_idx, batch_data in enumerate(dataloader):
        if batch_idx >= NUM_TEST_BATCHES:
            break
        
        try:
            images, labels, sizes, names, indices = batch_data
            
            print(f"\nBatch {batch_idx + 1}/{NUM_TEST_BATCHES}:")
            print(f"  Images shape: {images.shape}")
            print(f"  Labels shape: {labels.shape}")
            print(f"  Images dtype: {images.dtype}")
            print(f"  Labels dtype: {labels.dtype}")
            print(f"  Image value range: [{images.min():.2f}, {images.max():.2f}]")
            print(f"  Label value range: [{labels.min():.0f}, {labels.max():.0f}]")
            print(f"  Unique labels in batch: {torch.unique(labels).numpy()}")
            print(f"  Sample names: {names[:min(3, len(names))]}")
            
            # Validate data
            assert images.shape[0] <= BATCH_SIZE, "Batch size mismatch"
            assert images.shape[1] == 3, "Expected 3 channels (RGB)"
            assert images.shape[2] == 256, "Expected height 256"
            assert images.shape[3] == 256, "Expected width 256"
            assert labels.shape[1] == 256, "Expected label height 256"
            assert labels.shape[2] == 256, "Expected label width 256"
            
            # Check if labels are within valid range
            unique_labels = torch.unique(labels)
            valid_labels = all(l < NUM_CLASSES or l == 255 for l in unique_labels)
            if not valid_labels:
                print(f"  ⚠ WARNING: Found labels outside valid range [0, {NUM_CLASSES-1}]: {unique_labels.numpy()}")
            
            # Visualize first sample in batch
            if batch_idx == 0:
                visualize_sample(images[0], labels[0], names[0], OUTPUT_DIR, dataset.class_names)
            
            print(f"  ✓ Batch {batch_idx + 1} loaded successfully!")
            
        except Exception as e:
            print(f"  ❌ ERROR loading batch {batch_idx + 1}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    print(f"\n{'='*60}")
    print(f"✓✓✓ ALL TESTS PASSED! ✓✓✓")
    print(f"{'='*60}")
    print(f"\nYour dataset loader is working correctly!")
    print(f"Visualization saved to: {OUTPUT_DIR}")
    print(f"\nNext steps:")
    print(f"  1. Review the visualizations in {OUTPUT_DIR}")
    print(f"  2. Verify that the masks are correctly aligned with images")
    print(f"  3. Proceed with training using train_s4gan.py")
    
    return True


def visualize_sample(image, label, name, output_dir, class_names):
    """Visualize a single sample (image + mask)"""
    
    print(f"\n  Generating visualization for: {name}")
    
    # Denormalize image
    # Image is in CHW format and has been mean-subtracted
    image_np = image.numpy().transpose(1, 2, 0)  # CHW -> HWC
    image_np = image_np[:, :, ::-1]  # BGR -> RGB
    image_np = image_np + 128  # Add back mean
    image_np = np.clip(image_np, 0, 255).astype(np.uint8)
    
    # Label is already in HW format
    label_np = label.numpy().astype(np.int32)
    
    # Create colored mask for visualization
    num_classes = len(class_names)
    colors = plt.cm.get_cmap('tab10', num_classes)
    colored_mask = np.zeros((label_np.shape[0], label_np.shape[1], 3))
    
    for class_idx in range(num_classes):
        mask = (label_np == class_idx)
        colored_mask[mask] = colors(class_idx)[:3]
    
    # Handle ignore label (255) - show in black
    ignore_mask = (label_np == 255)
    colored_mask[ignore_mask] = [0, 0, 0]
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot image
    axes[0].imshow(image_np)
    axes[0].set_title(f'Image: {name}')
    axes[0].axis('off')
    
    # Plot label (raw)
    im = axes[1].imshow(label_np, cmap='tab10', vmin=0, vmax=num_classes-1)
    axes[1].set_title('Label (Class Indices)')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Plot colored mask
    axes[2].imshow(colored_mask)
    axes[2].set_title('Label (Colored)')
    axes[2].axis('off')
    
    # Add class legend
    legend_text = "Classes:\n"
    for idx, class_name in enumerate(class_names[:min(num_classes, 10)]):
        legend_text += f"{idx}: {class_name}\n"
    
    fig.text(0.98, 0.5, legend_text, va='center', ha='left', 
             fontsize=8, family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, f'visualization_{name.replace(".JPG", "").replace(".jpg", "")}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved visualization to: {output_path}")
    
    # Save class distribution
    unique, counts = np.unique(label_np, return_counts=True)
    dist_text = f"\nClass distribution in {name}:\n"
    dist_text += "-" * 40 + "\n"
    for u, c in zip(unique, counts):
        if u == 255:
            dist_text += f"Ignore (255): {c:6d} pixels ({c/label_np.size*100:5.2f}%)\n"
        elif u < len(class_names):
            dist_text += f"Class {u} ({class_names[u]:20s}): {c:6d} pixels ({c/label_np.size*100:5.2f}%)\n"
    print(dist_text)
    
    # Save distribution to text file
    dist_path = os.path.join(output_dir, f'distribution_{name.replace(".JPG", "").replace(".jpg", "")}.txt')
    with open(dist_path, 'w') as f:
        f.write(dist_text)


if __name__ == '__main__':
    print("\n" + "="*60)
    print("Salak Dataset Loader Test")
    print("="*60)
    
    success = test_dataloader()
    
    if success:
        print("\n✓ Test completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Test failed. Please check the errors above.")
        print("\nCommon issues:")
        print("  1. Dataset folder structure mismatch")
        print("  2. Image/mask file naming mismatch")
        print("  3. Wrong file paths in configuration")
        print("  4. Missing files listed in labeled_files_50.txt")
        sys.exit(1)
