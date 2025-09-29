import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset, DataLoader
import os
from pathlib import Path
import sys

from .augmentation_function import *
import warnings
warnings.filterwarnings('ignore')


# ==================== Improved NII.GZ File Reading Dataset ====================
class MedicalImageDataset(Dataset):
    """
    Medical Image NII.GZ File Reading Dataset (Integrated data augmentation, supports 3D and 4D images)
    Supports standard directory structure:
    data_root/
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── val/
    │   ├── images/
    │   └── labels/
    └── test/
        ├── images/
        └── labels/
    """
    def __init__(self, data_root, split='train', image_suffix=['.nii.gz', '.nii'], 
             mask_suffix=['.nii.gz', '.nii'], transform=None, target_size=None, 
             num_classes=None, debug_labels=True, use_augmentation=True,
             augmentation_type='medical'):
        """
        Args:
            data_root: Data root directory
            split: Data split ('train', 'val', 'test')
            image_suffix: Image file suffix, can be string or list (default: ['.nii.gz', '.nii'])
            mask_suffix: Label file suffix, can be string or list (default: ['.nii.gz', '.nii'])
            transform: Additional data augmentation transforms (applied after built-in augmentation)
            target_size: Target size (D, H, W), no resizing if None
            num_classes: Number of classes for label range checking
            debug_labels: Whether to enable label debugging mode
            use_augmentation: Whether to use data augmentation (only effective for training set)
            augmentation_type: Data augmentation type ('light', 'medium', 'heavy', 'medical', 'medical_heavy')
        """
        self.data_root = Path(data_root)
        self.split = split
        self.external_transform = transform
        self.target_size = target_size
        self.num_classes = num_classes
        self.debug_labels = debug_labels
        
        # Data augmentation settings: only enabled for training set
        self.use_augmentation = use_augmentation and (split == 'train')
        if self.use_augmentation:
            if augmentation_type == 'light':
                self.augmentation_transform = get_light_augmentation()
            elif augmentation_type == 'medium':
                self.augmentation_transform = get_medium_augmentation()
            elif augmentation_type == 'heavy':
                self.augmentation_transform = get_heavy_augmentation()
            elif augmentation_type == 'medical':
                self.augmentation_transform = get_medical_augmentation()
            elif augmentation_type == 'medical_heavy':
                self.augmentation_transform = get_medical_artifact_heavy()
            elif augmentation_type == 'custom':
                self.augmentation_transform = get_custom_augmentation()
            else:
                print(f"⚠️ Unknown augmentation type: {augmentation_type}, using default medical")
                self.augmentation_transform = get_medical_augmentation()
            
            print(f"Training set enabled {augmentation_type} data augmentation")
        else:
            self.augmentation_transform = None
            if split == 'train':
                print("Training set data augmentation not enabled")
            else:
                print(f"{split.upper()} set data augmentation not enabled (correct)")
        
        # Set image and label directories
        self.image_dir = self.data_root / split / 'images'
        self.mask_dir = self.data_root / split / 'labels'
        
        # Check if directories exist
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory does not exist: {self.image_dir}")
        if not self.mask_dir.exists():
            raise FileNotFoundError(f"Label directory does not exist: {self.mask_dir}")
        
        # Ensure suffix is in list format
        if isinstance(image_suffix, str):
            image_suffix = [image_suffix]
        if isinstance(mask_suffix, str):
            mask_suffix = [mask_suffix]
        
        # Find paired files (supports multiple suffixes)
        self.image_files = []
        for suffix in image_suffix:
            self.image_files.extend(list(self.image_dir.glob(f'*{suffix}')))
        
        # Remove duplicate files
        self.image_files = list(set(self.image_files))
        
        self.pairs = []
        
        for img_file in self.image_files:
            # Get filename (without extension)
            img_stem = img_file.stem
            if img_stem.endswith('.nii'):  # Handle .nii.gz case
                img_stem = img_stem[:-4]
            
            # Try to find corresponding label file (try all possible suffixes)
            mask_found = False
            for suffix in mask_suffix:
                mask_file = self.mask_dir / f"{img_stem}{suffix}"
                if mask_file.exists():
                    self.pairs.append((img_file, mask_file))
                    mask_found = True
                    break
            
            if not mask_found:
                print(f"Warning: Cannot find corresponding label file {img_stem}")
        
        print(f"[{split.upper()}] Found {len(self.pairs)} data pairs (supports .nii and .nii.gz)")
        
        # Analyze label distribution of entire dataset (only execute on first load)
        if self.debug_labels and len(self.pairs) > 0:
            self.analyze_label_distribution()
    
    def analyze_label_distribution(self):
        """Analyze label distribution of entire dataset"""
        #print(f"\n=== [{self.split.upper()}] Label Distribution Analysis ===")
        all_unique_values = set()
        problematic_files = []
        
        # Analyze label distribution of first 5 files
        sample_size = min(5, len(self.pairs))
        for i in range(sample_size):
            _, mask_path = self.pairs[i]
            try:
                mask = self.load_nii_image(mask_path)
                unique_vals = np.unique(mask)
                all_unique_values.update(unique_vals)
                
                # Check for abnormal values
                max_val = np.max(unique_vals)
                min_val = np.min(unique_vals)
                
                if min_val < 0 or (self.num_classes and max_val >= self.num_classes):
                    problematic_files.append((mask_path.name, unique_vals))
                
                #print(f"File {mask_path.name}: Label value range [{min_val:.1f}, {max_val:.1f}], unique values: {len(unique_vals)}")
                
            except Exception as e:
                print(f"Error occurred while analyzing file {mask_path}: {e}")
        
        # print(f"All unique label values: {sorted(list(all_unique_values))}")
        
        # if self.num_classes:
        #     print(f"Expected number of classes: {self.num_classes} (range: 0 to {self.num_classes-1})")
        #     invalid_values = [v for v in all_unique_values if v < 0 or v >= self.num_classes]
        #     if invalid_values:
        #         print(f"⚠️  Found invalid label values: {invalid_values}")
                
        # if problematic_files:
        #     #print(f"⚠️  Found {len(problematic_files)} problematic files:")
        #     for filename, vals in problematic_files:
        #         print(f"   - {filename}: {vals}")
        
        print("=== Analysis completed ===\n")
    
    def __len__(self):
        return len(self.pairs)
    
    def load_nii_image(self, file_path):
        """Load NII.GZ file, supports 3D and 4D images"""
        try:
            nii_img = nib.load(str(file_path))
            img_data = nii_img.get_fdata()
            
            # Convert to numpy array and ensure float32 type
            img_data = np.array(img_data, dtype=np.float32)
            
            # Check image dimensions and handle 4D case
            if img_data.ndim == 4:
                # For 4D images, take first time point or first slice of last dimension
                #print(f"Detected 4D image {file_path.name}, shape: {img_data.shape}")
                
                # Common 4D medical image formats: (x, y, z, time) or (x, y, z, channel)
                # Usually we take first time point or channel
                img_data = img_data[:, :, :, 0]
                #print(f"Converted to 3D image, new shape: {img_data.shape}")
                
            elif img_data.ndim < 3:
                raise ValueError(f"Image dimension too low: {img_data.ndim}D, needs at least 3D")
            elif img_data.ndim > 4:
                raise ValueError(f"Unsupported image dimension: {img_data.ndim}D, supports 3D or 4D")
            
            return img_data
        except Exception as e:
            print(f"Error occurred while reading file {file_path}: {e}")
            raise
    
    def normalize_image(self, image):
        """Image normalization"""
        # Remove outliers (optional)
        p1, p99 = np.percentile(image, (1, 99))
        image = np.clip(image, p1, p99)
        
        # Z-score normalization
        mean = np.mean(image)
        std = np.std(image)
        if std > 0:
            image = (image - mean) / std
        
        return image
    
    def resize_volume(self, volume, target_size):
        """Resize 3D volume, supports different input dimensions"""
        if target_size is None:
            return volume
            
        from scipy.ndimage import zoom
        
        # Ensure input is 3D
        if volume.ndim != 3:
            raise ValueError(f"resize_volume only supports 3D volumes, received {volume.ndim}D")
        
        current_size = volume.shape
        
        # Ensure target_size is also 3D
        if len(target_size) != 3:
            raise ValueError(f"target_size must be 3D (D, H, W), received {len(target_size)}D")
        
        zoom_factors = [t/c for t, c in zip(target_size, current_size)]
        
        # Use nearest neighbor interpolation for labels, linear interpolation for images
        if len(np.unique(volume)) < 10:  # Assume it's a label
            resized = zoom(volume, zoom_factors, order=0)
        else:  # Assume it's an image
            resized = zoom(volume, zoom_factors, order=1)
            
        return resized
    
    def clean_labels(self, mask, file_path=None):
        """Clean label data, only keep 0 and 1 - simplified version"""
        # Ensure mask is 3D
        if mask.ndim != 3:
            print(f"Warning: Label dimension abnormal ({mask.ndim}D), expected 3D")
            if mask.ndim == 4:
                # If it's a 4D label, take first channel/time point
                mask = mask[:, :, :, 0]
                print(f"4D label converted to 3D")
        
        # Remove NaN values and infinities
        mask = np.nan_to_num(mask, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Simple but effective binarization: all non-zero values become 1
        mask[mask > 0] = 1
        
        # Ensure integer type
        mask = mask.astype(np.int64)
        
        # Simple validation
        unique_vals = np.unique(mask)
        #if self.debug_labels and file_path:
            #print(f"📋 Label processing {Path(file_path).name}: {unique_vals}")
        
        # Final check
        if not set(unique_vals).issubset({0, 1}):
            print(f"⚠️ Found abnormal label values: {unique_vals}")
            # Force binarization again
            mask = (mask > 0).astype(np.int64)
        
        return mask
    
    def handle_dimension_mismatch(self, image, mask, file_path=None):
        """Handle dimension mismatch between image and label"""
        if image.shape != mask.shape:
            print(f"Dimension mismatch - Image: {image.shape}, Label: {mask.shape}")
            
            # If only one is 4D, reduce dimension first
            if image.ndim == 4 and mask.ndim == 3:
                print("Image is 4D, label is 3D - converting image to 3D")
                image = image[:, :, :, 0]
            elif image.ndim == 3 and mask.ndim == 4:
                print("Image is 3D, label is 4D - converting label to 3D") 
                mask = mask[:, :, :, 0]
            
            # Check dimensions again
            if image.shape != mask.shape:
                print(f"Dimensions still mismatched after adjustment - Image: {image.shape}, Label: {mask.shape}")
                # Can choose further processing strategies, such as cropping or padding
                min_shape = [min(i, m) for i, m in zip(image.shape, mask.shape)]
                print(f"Cropping both to common minimum size: {min_shape}")
                
                image = image[:min_shape[0], :min_shape[1], :min_shape[2]]
                mask = mask[:min_shape[0], :min_shape[1], :min_shape[2]]
        
        return image, mask
    
    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        
        try:
            # Load image and label
            image = self.load_nii_image(img_path)
            mask = self.load_nii_image(mask_path)
            
            # Handle dimension mismatch
            image, mask = self.handle_dimension_mismatch(image, mask, img_path.name)
            
            # Resize (only resize after ensuring both are 3D)
            if self.target_size is not None:
                image = self.resize_volume(image, self.target_size)
                mask = self.resize_volume(mask, self.target_size)
            
            # Normalize image
            image = self.normalize_image(image)
            
            # Clean and validate labels
            mask = self.clean_labels(mask, mask_path.name)
            
            # Add channel dimension: [D, H, W] -> [1, D, H, W]
            image = image[np.newaxis, ...]
            
            # Convert to tensor
            image = torch.from_numpy(image).float()
            mask = torch.from_numpy(mask).long()
            
            # Final tensor validation: ensure only 0 and 1
            mask_unique = torch.unique(mask)
            if not all(val in [0, 1] for val in mask_unique.tolist()):
                raise ValueError(f"Tensor contains non-binary label values: {mask_unique.tolist()}, expected only [0, 1]")
            
            # Apply built-in data augmentation (only for training set)
            if self.augmentation_transform is not None:
                image, mask = self.augmentation_transform(image, mask)
            
            # Apply additional external transforms
            if self.external_transform is not None:
                image, mask = self.external_transform(image, mask)
            
            return {
                'image': image,
                'mask': mask,
                'image_path': str(img_path),
                'mask_path': str(mask_path)
            }
            
        except Exception as e:
            print(f"❌ Error occurred while processing file pair {img_path.name} / {mask_path.name}: {e}")
            print(f"Detailed error information: {type(e).__name__}: {str(e)}")
            raise


# ==================== Data Loading Utility Functions ====================
def create_data_loaders(data_root, batch_size=2, target_size=(64, 64, 64), 
                       num_workers=2, use_augmentation=True, augmentation_type='medical'):
    """
    Create training, validation and test data loaders
    Note: This version enforces binary classification, only keeps labels 0 and 1, and only applies data augmentation to training set
    Supports 3D and 4D medical image formats
    
    Args:
        data_root: Data root directory
        batch_size: Batch size
        target_size: Target size (D, H, W)
        num_workers: Number of data loading threads
        use_augmentation: Whether to use data augmentation for training set
        augmentation_type: Data augmentation type ('light', 'medium', 'heavy', 'medical', 'medical_heavy')
    """
    data_loaders = {}
    
    for split in ['train', 'val', 'test']:
        split_path = Path(data_root) / split
        if split_path.exists():
            # Only enable detailed debugging for training set
            debug_mode = (split == 'train')
            
            dataset = MedicalImageDataset(
                data_root=data_root,
                split=split,
                target_size=target_size,
                num_classes=2,  # Fixed to binary classification
                debug_labels=debug_mode,
                use_augmentation=use_augmentation,
                augmentation_type=augmentation_type
            )
            
            shuffle = True if split == 'train' else False
            
            data_loaders[split] = DataLoader(
                dataset, 
                batch_size=batch_size, 
                shuffle=shuffle, 
                num_workers=num_workers,
                pin_memory=True
            )
        else:
            print(f"Warning: {split} directory does not exist, skipping")
    
    return data_loaders


# ==================== Debugging Utility Functions ====================
def debug_data_loader(data_loader, num_samples=3):
    """
    Debug data loader, check several batches of data
    This version is designed for binary classification, includes data augmentation information, supports 3D/4D images
    """
    print("\n=== DataLoader Binary Classification Debug Information (supports 3D/4D) ===")
    print(f"Batch size: {data_loader.batch_size}")
    print(f"Dataset size: {len(data_loader.dataset)}")
    print(f"Number of batches: {len(data_loader)}")
    print("Label description: 0=background, 1=foreground")
    print("Dimension support: 3D and 4D medical images (4D will be automatically converted to 3D)")
    
    # Check if data augmentation is enabled
    if hasattr(data_loader.dataset, 'augmentation_transform') and data_loader.dataset.augmentation_transform:
        print("🔄 Data augmentation: Enabled")
        print(f"Augmentation type: {len(data_loader.dataset.augmentation_transform.transforms)} transforms")
    else:
        print("❌ Data augmentation: Disabled")
    
    for i, batch in enumerate(data_loader):
        if i >= num_samples:
            break
            
        images = batch['image']
        masks = batch['mask']
        
        print(f"\n--- Batch {i+1} ---")
        print(f"Image shape: {images.shape}, dtype: {images.dtype}")
        print(f"Label shape: {masks.shape}, dtype: {masks.dtype}")
        print(f"Image value range: [{images.min():.3f}, {images.max():.3f}]")
        print(f"Label value range: [{masks.min()}, {masks.max()}]")
        print(f"Label unique values: {torch.unique(masks).tolist()}")
        
        # Check binary labels
        unique_labels = torch.unique(masks).tolist()
        if set(unique_labels).issubset({0, 1}):
            print("✅ Labels correct: only contains 0 and 1")
            
            # Count pixel distribution
            total_pixels = masks.numel()
            background_pixels = (masks == 0).sum().item()
            foreground_pixels = (masks == 1).sum().item()
            
            bg_percentage = (background_pixels / total_pixels) * 100
            fg_percentage = (foreground_pixels / total_pixels) * 100
            
            print(f"   Background(0): {background_pixels} pixels ({bg_percentage:.2f}%)")
            print(f"   Foreground(1): {foreground_pixels} pixels ({fg_percentage:.2f}%)")
        else:
            print(f"❌ Labels incorrect: contains non-binary values {unique_labels}")
    
    print("=== Debugging completed ===\n")


# ==================== Usage Examples and Test Functions ====================
def test_augmented_dataset():
    """Test dataset with integrated data augmentation (supports 3D/4D images)"""
    # Assume your data path
    data_root = r"D:\unet3d\dataset"
    
    print("🧪 Starting test of different levels of data augmentation (supports 3D/4D images)...")
    
    augmentation_types = ['light', 'medium', 'heavy', 'medical', 'medical_heavy']
    
    for aug_type in augmentation_types:
        print(f"\n{'='*50}")
        print(f"🔄 Testing {aug_type.upper()} data augmentation")
        print(f"{'='*50}")
        
        try:
            # Create data loaders
            data_loaders = create_data_loaders(
                data_root=data_root,
                batch_size=1,  # Use small batch for testing
                target_size=(32, 32, 32),  # Use small size for quick testing
                use_augmentation=True,
                augmentation_type=aug_type
            )
            
            # Test training set data loader
            if 'train' in data_loaders:
                print(f"\n📊 {aug_type} augmentation effect test:")
                debug_data_loader(data_loaders['train'], num_samples=1)
            else:
                print("⚠️ Training set data not found")
                
        except FileNotFoundError as e:
            print(f"❌ Data path error: {e}")
            print("Please confirm your data directory structure as follows:")
            print("data_root/")
            print("├── train/")
            print("│   ├── images/")
            print("│   └── labels/")
            print("├── val/")
            print("│   ├── images/")
            print("│   └── labels/")
            print("└── test/")
            print("    ├── images/")
            print("    └── labels/")
            break
        except Exception as e:
            print(f"❌ Error occurred while testing {aug_type}: {e}")