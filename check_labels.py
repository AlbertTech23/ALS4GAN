"""
Quick diagnostic to check for invalid labels in the dataset
"""
import sys
import numpy as np
from tqdm import tqdm

sys.path.insert(0, 'c:\_albert\ALS4GAN')

from data.salak_dataset import SalakDataSet

# Initialize dataset
dataset = SalakDataSet(
    root="C:/_albert/s4GAN/patchify/temp_patches",
    list_path=None,
    class_mapping_csv="C:/_albert/ALS4GAN/class_mapping.csv",
    crop_size=(256, 256),
    mean=(0, 0, 0),
    scale=False,
    mirror=False,
    ignore_label=255
)

print(f"Total patches: {len(dataset)}")
print(f"Number of classes expected: 7 (indices 0-6)")
print(f"Ignore label: 255")
print()

# Check a sample of masks for invalid labels
invalid_found = []
unique_labels = set()

sample_size = min(1000, len(dataset))
print(f"Checking {sample_size} random samples...")

indices = np.random.choice(len(dataset), sample_size, replace=False)

for idx in tqdm(indices):
    _, label, _, _, _ = dataset[idx]
    
    # Get unique values
    unique = np.unique(label)
    unique_labels.update(unique.tolist())
    
    # Check for invalid labels (should be 0-6 or 255)
    invalid = unique[(unique > 6) & (unique != 255)]
    if len(invalid) > 0:
        invalid_found.append({
            'idx': idx,
            'invalid_labels': invalid.tolist(),
            'all_unique': unique.tolist()
        })

print()
print("=" * 80)
print("RESULTS:")
print("=" * 80)
print(f"All unique labels found: {sorted(unique_labels)}")
print()

if invalid_found:
    print(f"⚠️  FOUND {len(invalid_found)} samples with INVALID labels!")
    print()
    print("Examples:")
    for i, item in enumerate(invalid_found[:5]):
        print(f"  Sample {item['idx']}: Invalid={item['invalid_labels']}, All unique={item['all_unique']}")
    print()
    print("RECOMMENDATION:")
    print("  The dataset contains labels outside [0-6, 255] range.")
    print("  This is causing the CUDA error. Options:")
    print("  1. Fix the dataset labels to only contain valid values")
    print("  2. The loss functions now clamp invalid values, so training should work")
else:
    print("✓ No invalid labels found in sampled patches!")
    print("  The issue might be:")
    print("  1. Rare invalid labels not in this sample")
    print("  2. Data augmentation creating invalid values")
    print("  3. The loss functions now have safety checks, so should work now")

print("=" * 80)
