"""
Salak Dataset Loader for Remote Sensing Dataset
Adapted for S4GAN semi-supervised semantic segmentation
Handles multi-folder structure: salak-1-1, salak-1-2, ..., salak-1-6
"""

import cv2
import numpy as np
import pandas as pd
import random
import os
import os.path as osp
from torch.utils import data
from PIL import Image
import glob


class SalakDataSet(data.Dataset):
    """
    Salak Dataset for Remote Sensing patches
    
    Folder structure:
        root/
            salak-1-1/
                images/
                masks/
            salak-1-2/
                images/
                masks/
            ...
            salak-1-6/
                images/
                masks/
    
    Args:
        root: Root directory containing salak-1-* folders
        list_path: Path to text file containing list of labeled images
        class_mapping_csv: Path to CSV file with class mapping
        module: 's4gan' for semantic segmentation
        crop_size: Size to crop/resize images (default: 320x320)
        mean: Mean values for normalization (default: 128, 128, 128)
        scale: Whether to apply random scaling
        mirror: Whether to apply random mirroring
        ignore_label: Label value to ignore (default: 255)
    """
    
    def __init__(self, root, list_path, class_mapping_csv, module='s4gan', 
                 crop_size=(320, 320), mean=(128, 128, 128), 
                 scale=False, mirror=False, ignore_label=255):
        
        self.module = module
        self.root = root
        self.list_path = list_path
        self.crop_h, self.crop_w = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        
        # Read the list of labeled image filenames (if provided)
        if list_path is not None:
            with open(list_path, 'r') as f:
                self.img_ids = [i_id.strip() for i_id in f.readlines() if i_id.strip()]
        else:
            # No labeled list - will load all patches
            self.img_ids = []
        
        # Load class mapping from CSV
        self.load_class_mapping(class_mapping_csv)
        
        # Build file list
        self.files = []
        self._build_file_list()
        
        print(f"SalakDataSet initialized:")
        print(f"  Root: {self.root}")
        print(f"  Total images: {len(self.files)}")
        print(f"  Number of classes: {len(self.class_colors)}")
        print(f"  Crop size: {crop_size}")
        
        # Count how many have masks
        with_masks = sum(1 for f in self.files if f['label'] is not None)
        print(f"  Images with masks: {with_masks}")
        print(f"  Images without masks (unlabeled): {len(self.files) - with_masks}")
        
    def load_class_mapping(self, csv_path):
        """Load class names and colors from CSV"""
        df = pd.read_csv(csv_path)
        self.class_names = df['class_name'].tolist()
        
        # Parse RGB colors from the CSV
        self.class_colors = []
        for idx, row in df.iterrows():
            color_str = row['class_color']
            # Parse "255 50 50" -> [255, 50, 50]
            rgb = [int(c) for c in color_str.strip().split()]
            self.class_colors.append(rgb)
        
        self.num_classes = len(self.class_colors)
        print(f"Loaded {self.num_classes} classes:")
        for name, color in zip(self.class_names, self.class_colors):
            print(f"  {name}: {color}")
    
    def _build_file_list(self):
        """
        Build list of image and label file paths
        Searches across salak-1-1, salak-1-2, ..., salak-1-6 folders
        Each salak folder has images/ and masks/ subfolders
        
        For S4GAN training:
        - If labeled_list is provided: Load only patches from those base images
        - Otherwise: Load ALL patches from all folders
        """
        # Get all salak-1-* folders
        salak_folders = sorted(glob.glob(osp.join(self.root, "salak-1-*")))
        
        if not salak_folders:
            print(f"WARNING: No salak-1-* folders found in {self.root}")
            return
        
        print(f"Found {len(salak_folders)} salak folders: {[osp.basename(f) for f in salak_folders]}")
        
        # Count total patches
        total_image_patches = 0
        total_mask_patches = 0
        print("\nCounting total patches:")
        for salak_folder in salak_folders:
            images_dir = osp.join(salak_folder, "images")
            masks_dir = osp.join(salak_folder, "masks")
            if osp.exists(images_dir):
                img_count = len([f for f in os.listdir(images_dir) if f.lower().endswith('.png')])
                total_image_patches += img_count
            else:
                img_count = 0
            if osp.exists(masks_dir):
                mask_count = len([f for f in os.listdir(masks_dir) if f.lower().endswith('.png')])
                total_mask_patches += mask_count
            else:
                mask_count = 0
            print(f"  {osp.basename(salak_folder)}: {img_count} images, {mask_count} masks")
        
        print(f"\n📊 TOTAL PATCHES: {total_image_patches} images, {total_mask_patches} masks")
        print(f"📊 Unlabeled patches: {total_image_patches - total_mask_patches}")
        
        # Decide loading strategy
        if self.img_ids:
            # Load only patches from labeled base images
            print(f"\n🔍 Loading patches from {len(self.img_ids)} labeled base images...\n")
            self._load_from_labeled_list(salak_folders)
        else:
            # Load ALL patches for training
            print(f"\n🔍 Loading ALL patches for training...\n")
            self._load_all_patches(salak_folders)
    
    def _load_from_labeled_list(self, salak_folders):
        """Load only patches from the labeled base image list"""
        for name in self.img_ids:
            img_file = None
            label_file = None
            found_folder = None
            
            # Try to find exact match first
            base_name = osp.splitext(name)[0]
            
            # Search for the image file across all salak folders
            for salak_folder in salak_folders:
                images_dir = osp.join(salak_folder, "images")
                
                if not osp.exists(images_dir):
                    continue
                
                # Try different extensions and naming patterns
                potential_names = [
                    name,  # Exact match
                    base_name + '.jpg',
                    base_name + '.JPG',
                    base_name + '.png',
                    base_name + '.PNG',
                    base_name + '.tif',
                    base_name + '.TIF',
                    # Try with common patch suffixes (if patchified)
                    base_name + '_0.jpg',
                    base_name + '_0.JPG',
                ]
                
                for potential_name in potential_names:
                    potential_img = osp.join(images_dir, potential_name)
                    if osp.exists(potential_img):
                        img_file = potential_img
                        found_folder = salak_folder
                        break
                
                if img_file:
                    break
            
            # If still not found, try recursive search (in case of nested structure)
            if not img_file:
                for salak_folder in salak_folders:
                    images_dir = osp.join(salak_folder, "images")
                    if osp.exists(images_dir):
                        # Search recursively for any file containing the base name
                        for root, dirs, files in os.walk(images_dir):
                            for file in files:
                                if base_name in file and file.lower().endswith(('.jpg', '.jpeg', '.png', '.tif')):
                                    img_file = osp.join(root, file)
                                    found_folder = salak_folder
                                    break
                            if img_file:
                                break
                    if img_file:
                        break
            
            if not img_file:
                print(f"WARNING: Image not found: {name}")
                continue
            
            # Look for corresponding mask in the same folder
            if self.module == 's4gan':
                masks_dir = osp.join(found_folder, "masks")
                
                # Check if masks directory exists and is not empty
                if osp.exists(masks_dir) and os.listdir(masks_dir):
                    # Get all available mask files for this base name
                    available_masks = []
                    for root, dirs, files in os.walk(masks_dir):
                        for file in files:
                            if base_name in file and file.lower().endswith(('.png', '.PNG')):
                                available_masks.append(osp.join(root, file))
                    
                    if available_masks:
                        # Pick a random mask from available ones (for augmentation diversity)
                        label_file = np.random.choice(available_masks)
                        
                        # Update the image file to match the mask patch
                        # Extract the mask filename to find corresponding image
                        mask_filename = osp.basename(label_file)
                        mask_base = osp.splitext(mask_filename)[0]
                        
                        # Look for image with same patch name
                        images_dir = osp.join(found_folder, "images")
                        for potential_ext in ['.png', '.PNG', '.jpg', '.JPG', '.jpeg', '.JPEG']:
                            potential_img = osp.join(images_dir, mask_base + potential_ext)
                            if osp.exists(potential_img):
                                img_file = potential_img
                                break
                        
                        # Verify the image exists
                        if not osp.exists(img_file):
                            print(f"WARNING: Image not found for mask {mask_filename} in {found_folder}")
                            label_file = None
                    else:
                        # No masks found for this base name
                        print(f"WARNING: Mask not found for {name} in {found_folder}")
                        label_file = None
                else:
                    # Empty masks folder - treat as unlabeled
                    label_file = None
            else:
                label_file = None
            
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name,
                "folder": osp.basename(found_folder) if found_folder else "unknown"
            })
    
    def _load_all_patches(self, salak_folders):
        """Load ALL patches from all salak folders for training"""
        for salak_folder in salak_folders:
            images_dir = osp.join(salak_folder, "images")
            masks_dir = osp.join(salak_folder, "masks")
            
            if not osp.exists(images_dir):
                continue
            
            # Get all image files in this folder
            image_files = [f for f in os.listdir(images_dir) if f.lower().endswith('.png')]
            
            # Build a set of available mask basenames for quick lookup
            available_masks = set()
            if osp.exists(masks_dir):
                mask_files = [f for f in os.listdir(masks_dir) if f.lower().endswith('.png')]
                available_masks = {osp.splitext(f)[0] for f in mask_files}
            
            # Add each image patch
            for img_filename in image_files:
                img_path = osp.join(images_dir, img_filename)
                img_base = osp.splitext(img_filename)[0]
                
                # Check if corresponding mask exists
                if img_base in available_masks:
                    mask_path = osp.join(masks_dir, img_base + '.png')
                    if osp.exists(mask_path):
                        label_file = mask_path
                    else:
                        label_file = None
                else:
                    label_file = None
                
                self.files.append({
                    "img": img_path,
                    "label": label_file,
                    "name": img_filename,
                    "folder": osp.basename(salak_folder)
                })
    
    def __len__(self):
        return len(self.files)
    
    def generate_scale_label(self, image, label):
        """Apply random scaling to image and label"""
        f_scale = 0.5 + random.randint(0, 11) / 10.0
        image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_LINEAR)
        if self.module == 's4gan':
            label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_NEAREST)
        return image, label
    
    def encode_segmap(self, mask):
        """
        Encode RGB segmentation mask to class indices
        
        Args:
            mask: RGB mask (H, W, 3) with colors corresponding to classes
                  OR grayscale mask (H, W, 3) with class indices replicated across channels
        
        Returns:
            label_mask: Class indices (H, W) with values 0 to num_classes-1
        """
        # Check if mask is already grayscale (all channels identical = class indices)
        if len(mask.shape) == 3:
            # Check if all channels are identical (grayscale stored as BGR)
            if np.allclose(mask[:,:,0], mask[:,:,1]) and np.allclose(mask[:,:,1], mask[:,:,2]):
                # Already grayscale with class indices - just extract one channel
                label_mask = mask[:,:,0].astype(np.int32)
                return label_mask
        
        # Otherwise, decode RGB colors to class indices
        mask = mask.astype(int)
        label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
        
        for class_idx, color in enumerate(self.class_colors):
            # Find pixels matching this class color
            matches = np.all(mask == color, axis=-1)
            label_mask[matches] = class_idx
        
        label_mask = label_mask.astype(int)
        return label_mask
    
    def __getitem__(self, index):
        """
        Get a single data sample
        
        Returns:
            image: Preprocessed image tensor (C, H, W)
            label: Encoded label mask (H, W)
            size: Original image size
            name: Image filename
            index: Sample index
        """
        datafiles = self.files[index]
        
        # Load image
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to load image: {datafiles['img']}")
        
        # Resize to standard size (256x256)
        image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_CUBIC)
        
        if self.module == 's4gan':
            # Load label/mask if available
            if datafiles["label"] is not None:
                label = cv2.imread(datafiles["label"], cv2.IMREAD_COLOR)
                if label is None:
                    # Failed to load mask - treat as unlabeled
                    label = np.full((256, 256), self.ignore_label, dtype=np.int32)
                else:
                    # Resize mask
                    label = cv2.resize(label, (256, 256), interpolation=cv2.INTER_NEAREST)
                    # Convert RGB mask to class indices
                    label = self.encode_segmap(label)
            else:
                # No mask available - unlabeled data
                label = np.full((256, 256), self.ignore_label, dtype=np.int32)
        else:
            label = np.zeros((256, 256), dtype=np.int32)
        
        size = image.shape
        name = datafiles["name"]
        
        # Apply random scaling if enabled
        if self.scale:
            image, label = self.generate_scale_label(image, label)
        
        # Convert to float and subtract mean
        image = np.asarray(image, np.float32)
        image -= self.mean
        
        if self.module == 's4gan':
            img_h, img_w = label.shape
        else:
            img_h, img_w = image.shape[0], image.shape[1]
        
        # Pad if necessary
        pad_h = max(self.crop_h - img_h, 0)
        pad_w = max(self.crop_w - img_w, 0)
        
        if pad_h > 0 or pad_w > 0:
            img_pad = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w,
                                        cv2.BORDER_CONSTANT,
                                        value=(0.0, 0.0, 0.0))
            if self.module == 's4gan':
                label_pad = cv2.copyMakeBorder(label, 0, pad_h, 0, pad_w,
                                              cv2.BORDER_CONSTANT,
                                              value=(self.ignore_label,))
            else:
                label_pad = label
        else:
            img_pad, label_pad = image, label
        
        # Random crop
        img_h, img_w = img_pad.shape[0], img_pad.shape[1]
        h_off = random.randint(0, img_h - self.crop_h)
        w_off = random.randint(0, img_w - self.crop_w)
        
        image = np.asarray(img_pad[h_off:h_off + self.crop_h, w_off:w_off + self.crop_w],
                          np.float32)
        
        if self.module == 's4gan':
            label = np.asarray(label_pad[h_off:h_off + self.crop_h, w_off:w_off + self.crop_w],
                              np.float32)
        
        # Convert BGR to RGB
        image = image[:, :, ::-1]
        # Transpose to CHW format for PyTorch
        image = image.transpose((2, 0, 1))
        
        # Random mirroring
        if self.is_mirror:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            if self.module == 's4gan':
                label = label[:, ::flip]
        
        return image.copy(), label.copy(), np.array(size), name, index
