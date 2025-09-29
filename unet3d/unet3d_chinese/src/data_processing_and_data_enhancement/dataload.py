import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset, DataLoader
import os
from pathlib import Path
import sys
# 導入你的模組（根據實際路徑調整）
sys.path.append(r"D:\unet3d")
from .augmentation_function import *
import warnings
warnings.filterwarnings('ignore')

#i love you~~~~~~~~~~~~~~
# ==================== 改良版 NII.GZ 檔案讀取 Dataset ====================
class MedicalImageDataset(Dataset):
    """
    醫學影像 NII.GZ 檔案讀取 Dataset (整合數據增強功能，支援3D和4D圖像)
    支援標準目錄結構：
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
            data_root: 資料根目錄
            split: 資料分割 ('train', 'val', 'test')
            image_suffix: 影像檔案後綴，可以是字串或列表 (預設: ['.nii.gz', '.nii'])
            mask_suffix: 標籤檔案後綴，可以是字串或列表 (預設: ['.nii.gz', '.nii'])
            transform: 額外的資料增強變換（會在內建增強之後執行）
            target_size: 目標尺寸 (D, H, W)，若為 None 則不調整大小
            num_classes: 類別數量，用於檢查標籤範圍
            debug_labels: 是否啟用標籤除錯模式
            use_augmentation: 是否使用數據增強（只對訓練集有效）
            augmentation_type: 數據增強類型 ('light', 'medium', 'heavy', 'medical', 'medical_heavy')
        """
        self.data_root = Path(data_root)
        self.split = split
        self.external_transform = transform
        self.target_size = target_size
        self.num_classes = num_classes
        self.debug_labels = debug_labels
        
        # 數據增強設定：只對訓練集啟用
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
                print(f"⚠️ 未知的增強類型: {augmentation_type}, 使用預設 medical")
                self.augmentation_transform = get_medical_augmentation()
            
            print(f"訓練集啟用 {augmentation_type} 數據增強")
        else:
            self.augmentation_transform = None
            if split == 'train':
                print("訓練集未啟用數據增強")
            else:
                print(f"{split.upper()} 集未啟用數據增強（正確）")
        
        # 設定影像和標籤目錄
        self.image_dir = self.data_root / split / 'images'
        self.mask_dir = self.data_root / split / 'labels'
        
        # 檢查目錄是否存在
        if not self.image_dir.exists():
            raise FileNotFoundError(f"影像目錄不存在: {self.image_dir}")
        if not self.mask_dir.exists():
            raise FileNotFoundError(f"標籤目錄不存在: {self.mask_dir}")
        
        # 確保 suffix 是列表格式
        if isinstance(image_suffix, str):
            image_suffix = [image_suffix]
        if isinstance(mask_suffix, str):
            mask_suffix = [mask_suffix]
        
        # 尋找配對的檔案（支援多種後綴）
        self.image_files = []
        for suffix in image_suffix:
            self.image_files.extend(list(self.image_dir.glob(f'*{suffix}')))
        
        # 去除重複檔案
        self.image_files = list(set(self.image_files))
        
        self.pairs = []
        
        for img_file in self.image_files:
            # 取得檔名（不含副檔名）
            img_stem = img_file.stem
            if img_stem.endswith('.nii'):  # 處理 .nii.gz 的情況
                img_stem = img_stem[:-4]
            
            # 嘗試尋找對應的標籤檔案（嘗試所有可能的後綴）
            mask_found = False
            for suffix in mask_suffix:
                mask_file = self.mask_dir / f"{img_stem}{suffix}"
                if mask_file.exists():
                    self.pairs.append((img_file, mask_file))
                    mask_found = True
                    break
            
            if not mask_found:
                print(f"警告: 找不到對應的標籤檔案 {img_stem}")
        
        print(f"[{split.upper()}] 找到 {len(self.pairs)} 對資料 (支援 .nii 和 .nii.gz)")
        
        # 分析整個資料集的標籤分佈（只在第一次載入時執行）
        if self.debug_labels and len(self.pairs) > 0:
            self.analyze_label_distribution()
    
    def analyze_label_distribution(self):
        """分析整個資料集的標籤分佈"""
        #print(f"\n=== [{self.split.upper()}] 標籤分佈分析 ===")
        all_unique_values = set()
        problematic_files = []
        
        # 分析前5個檔案的標籤分佈
        sample_size = min(5, len(self.pairs))
        for i in range(sample_size):
            _, mask_path = self.pairs[i]
            try:
                mask = self.load_nii_image(mask_path)
                unique_vals = np.unique(mask)
                all_unique_values.update(unique_vals)
                
                # 檢查是否有異常值
                max_val = np.max(unique_vals)
                min_val = np.min(unique_vals)
                
                if min_val < 0 or (self.num_classes and max_val >= self.num_classes):
                    problematic_files.append((mask_path.name, unique_vals))
                
                #print(f"檔案 {mask_path.name}: 標籤值範圍 [{min_val:.1f}, {max_val:.1f}], 唯一值: {len(unique_vals)}")
                
            except Exception as e:
                print(f"分析檔案 {mask_path} 時發生錯誤: {e}")
        
        # print(f"所有唯一標籤值: {sorted(list(all_unique_values))}")
        
        # if self.num_classes:
        #     print(f"預期類別數: {self.num_classes} (範圍: 0 到 {self.num_classes-1})")
        #     invalid_values = [v for v in all_unique_values if v < 0 or v >= self.num_classes]
        #     if invalid_values:
        #         print(f"⚠️  發現無效標籤值: {invalid_values}")
                
        # if problematic_files:
        #     #print(f"⚠️  發現 {len(problematic_files)} 個問題檔案:")
        #     for filename, vals in problematic_files:
        #         print(f"   - {filename}: {vals}")
        
        print("=== 分析完成 ===\n")
    
    def __len__(self):
        return len(self.pairs)
    
    def load_nii_image(self, file_path):
        """載入 NII.GZ 檔案，支援3D和4D圖像"""
        try:
            nii_img = nib.load(str(file_path))
            img_data = nii_img.get_fdata()
            
            # 轉換為 numpy array 並確保是 float32 類型
            img_data = np.array(img_data, dtype=np.float32)
            
            # 檢查圖像維度並處理4D情況
            if img_data.ndim == 4:
                # 對於4D圖像，取第一個時間點或最後一個維度的第一個切片
                #print(f"檢測到4D圖像 {file_path.name}，形狀: {img_data.shape}")
                
                # 常見的4D醫學影像格式：(x, y, z, time) 或 (x, y, z, channel)
                # 通常我們取第一個時間點或通道
                img_data = img_data[:, :, :, 0]
                #print(f"轉換為3D圖像，新形狀: {img_data.shape}")
                
            elif img_data.ndim < 3:
                raise ValueError(f"圖像維度過低: {img_data.ndim}D，需要至少3D")
            elif img_data.ndim > 4:
                raise ValueError(f"不支援的圖像維度: {img_data.ndim}D，支援3D或4D")
            
            return img_data
        except Exception as e:
            print(f"讀取檔案 {file_path} 時發生錯誤: {e}")
            raise
    
    def normalize_image(self, image):
        """影像標準化"""
        # 移除異常值（可選）
        p1, p99 = np.percentile(image, (1, 99))
        image = np.clip(image, p1, p99)
        
        # Z-score 標準化
        mean = np.mean(image)
        std = np.std(image)
        if std > 0:
            image = (image - mean) / std
        
        return image
    
    def resize_volume(self, volume, target_size):
        """調整 3D 體積大小，支援不同輸入維度"""
        if target_size is None:
            return volume
            
        from scipy.ndimage import zoom
        
        # 確保輸入是3D的
        if volume.ndim != 3:
            raise ValueError(f"resize_volume 只支援3D體積，收到 {volume.ndim}D")
        
        current_size = volume.shape
        
        # 確保 target_size 也是3D的
        if len(target_size) != 3:
            raise ValueError(f"target_size 必須是3D (D, H, W)，收到 {len(target_size)}D")
        
        zoom_factors = [t/c for t, c in zip(target_size, current_size)]
        
        # 對於標籤使用最近鄰插值，對於影像使用線性插值
        if len(np.unique(volume)) < 10:  # 假設是標籤
            resized = zoom(volume, zoom_factors, order=0)
        else:  # 假設是影像
            resized = zoom(volume, zoom_factors, order=1)
            
        return resized
    
    def clean_labels(self, mask, file_path=None):
        """清理標籤資料，只保留0和1 - 簡化版本"""
        # 確保mask是3D的
        if mask.ndim != 3:
            print(f"警告: 標籤維度異常 ({mask.ndim}D)，預期3D")
            if mask.ndim == 4:
                # 如果是4D標籤，取第一個通道/時間點
                mask = mask[:, :, :, 0]
                print(f"4D標籤已轉換為3D")
        
        # 移除 NaN 值和無窮大值
        mask = np.nan_to_num(mask, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 簡單但有效的二元化：所有非零值都變成1
        mask[mask > 0] = 1
        
        # 確保是整數類型
        mask = mask.astype(np.int64)
        
        # 簡單驗證
        unique_vals = np.unique(mask)
        #if self.debug_labels and file_path:
            #print(f"📋 標籤處理 {Path(file_path).name}: {unique_vals}")
        
        # 最終檢查
        if not set(unique_vals).issubset({0, 1}):
            print(f"⚠️ 發現異常標籤值: {unique_vals}")
            # 再次強制二元化
            mask = (mask > 0).astype(np.int64)
        
        return mask
    
    def handle_dimension_mismatch(self, image, mask, file_path=None):
        """處理影像和標籤維度不匹配的問題"""
        if image.shape != mask.shape:
            print(f"維度不匹配 - 影像: {image.shape}, 標籤: {mask.shape}")
            
            # 如果只是其中一個是4D，先降維
            if image.ndim == 4 and mask.ndim == 3:
                print("影像是4D，標籤是3D - 將影像轉為3D")
                image = image[:, :, :, 0]
            elif image.ndim == 3 and mask.ndim == 4:
                print("影像是3D，標籤是4D - 將標籤轉為3D") 
                mask = mask[:, :, :, 0]
            
            # 再次檢查維度
            if image.shape != mask.shape:
                print(f"調整後維度仍不匹配 - 影像: {image.shape}, 標籤: {mask.shape}")
                # 可以選擇進一步的處理策略，例如裁剪或填充
                min_shape = [min(i, m) for i, m in zip(image.shape, mask.shape)]
                print(f"將兩者裁剪到共同最小尺寸: {min_shape}")
                
                image = image[:min_shape[0], :min_shape[1], :min_shape[2]]
                mask = mask[:min_shape[0], :min_shape[1], :min_shape[2]]
        
        return image, mask
    
    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        
        try:
            # 載入影像和標籤
            image = self.load_nii_image(img_path)
            mask = self.load_nii_image(mask_path)
            
            # 處理維度不匹配問題
            image, mask = self.handle_dimension_mismatch(image, mask, img_path.name)
            
            # 調整大小（確保都是3D後才調整）
            if self.target_size is not None:
                image = self.resize_volume(image, self.target_size)
                mask = self.resize_volume(mask, self.target_size)
            
            # 標準化影像
            image = self.normalize_image(image)
            
            # 清理和驗證標籤
            mask = self.clean_labels(mask, mask_path.name)
            
            # 添加通道維度：[D, H, W] -> [1, D, H, W]
            image = image[np.newaxis, ...]
            
            # 轉換為 tensor
            image = torch.from_numpy(image).float()
            mask = torch.from_numpy(mask).long()
            
            # 最終驗證 tensor：確保只有0和1
            mask_unique = torch.unique(mask)
            if not all(val in [0, 1] for val in mask_unique.tolist()):
                raise ValueError(f"Tensor 包含非二元標籤值: {mask_unique.tolist()}, 預期只有 [0, 1]")
            
            # 應用內建數據增強（只對訓練集）
            if self.augmentation_transform is not None:
                image, mask = self.augmentation_transform(image, mask)
            
            # 應用額外的外部變換
            if self.external_transform is not None:
                image, mask = self.external_transform(image, mask)
            
            return {
                'image': image,
                'mask': mask,
                'image_path': str(img_path),
                'mask_path': str(mask_path)
            }
            
        except Exception as e:
            print(f"❌ 處理檔案對 {img_path.name} / {mask_path.name} 時發生錯誤: {e}")
            print(f"詳細錯誤資訊: {type(e).__name__}: {str(e)}")
            raise


# ==================== 資料載入工具函數 ====================
def create_data_loaders(data_root, batch_size=2, target_size=(64, 64, 64), 
                       num_workers=2, use_augmentation=True, augmentation_type='medical'):
    """
    創建訓練、驗證和測試資料載入器
    注意：此版本強制執行二元分類，只保留標籤0和1，並且只對訓練集進行數據增強
    支援3D和4D醫學影像格式
    
    Args:
        data_root: 資料根目錄
        batch_size: 批次大小
        target_size: 目標尺寸 (D, H, W)
        num_workers: 資料載入執行緒數
        use_augmentation: 是否對訓練集使用數據增強
        augmentation_type: 數據增強類型 ('light', 'medium', 'heavy', 'medical', 'medical_heavy')
    """
    data_loaders = {}
    
    for split in ['train', 'val', 'test']:
        split_path = Path(data_root) / split
        if split_path.exists():
            # 只在訓練集啟用詳細除錯
            debug_mode = (split == 'train')
            
            dataset = MedicalImageDataset(
                data_root=data_root,
                split=split,
                target_size=target_size,
                num_classes=2,  # 固定為二元分類
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
            print(f"警告: {split} 目錄不存在，跳過")
    
    return data_loaders


# ==================== 除錯工具函數 ====================
def debug_data_loader(data_loader, num_samples=3):
    """
    除錯資料載入器，檢查幾個批次的資料
    此版本專為二元分類設計，包含數據增強資訊，支援3D/4D圖像
    """
    print("\n=== DataLoader 二元分類除錯資訊 (支援3D/4D) ===")
    print(f"批次大小: {data_loader.batch_size}")
    print(f"資料集大小: {len(data_loader.dataset)}")
    print(f"批次數量: {len(data_loader)}")
    print("標籤說明: 0=背景, 1=前景")
    print("維度支援: 3D和4D醫學影像（4D會自動轉換為3D）")
    
    # 檢查是否有數據增強
    if hasattr(data_loader.dataset, 'augmentation_transform') and data_loader.dataset.augmentation_transform:
        print("🔄 數據增強: 已啟用")
        print(f"增強類型: {len(data_loader.dataset.augmentation_transform.transforms)} 種變換")
    else:
        print("❌ 數據增強: 未啟用")
    
    for i, batch in enumerate(data_loader):
        if i >= num_samples:
            break
            
        images = batch['image']
        masks = batch['mask']
        
        print(f"\n--- 批次 {i+1} ---")
        print(f"影像 shape: {images.shape}, dtype: {images.dtype}")
        print(f"標籤 shape: {masks.shape}, dtype: {masks.dtype}")
        print(f"影像值範圍: [{images.min():.3f}, {images.max():.3f}]")
        print(f"標籤值範圍: [{masks.min()}, {masks.max()}]")
        print(f"標籤唯一值: {torch.unique(masks).tolist()}")
        
        # 檢查二元標籤
        unique_labels = torch.unique(masks).tolist()
        if set(unique_labels).issubset({0, 1}):
            print("✅ 標籤正確：只包含0和1")
            
            # 統計像素分佈
            total_pixels = masks.numel()
            background_pixels = (masks == 0).sum().item()
            foreground_pixels = (masks == 1).sum().item()
            
            bg_percentage = (background_pixels / total_pixels) * 100
            fg_percentage = (foreground_pixels / total_pixels) * 100
            
            print(f"   背景(0): {background_pixels} 像素 ({bg_percentage:.2f}%)")
            print(f"   前景(1): {foreground_pixels} 像素 ({fg_percentage:.2f}%)")
        else:
            print(f"❌ 標籤錯誤：包含非二元值 {unique_labels}")
    
    print("=== 除錯完成 ===\n")


# ==================== 使用範例和測試函數 ====================
def test_augmented_dataset():
    """測試整合數據增強的資料集（支援3D/4D圖像）"""
    # 假設您的資料路徑
    data_root = r"D:\unet3d\dataset"
    
    print("🧪 開始測試不同等級的數據增強 (支援3D/4D圖像)...")
    
    augmentation_types = ['light', 'medium', 'heavy', 'medical', 'medical_heavy']
    
    for aug_type in augmentation_types:
        print(f"\n{'='*50}")
        print(f"🔄 測試 {aug_type.upper()} 數據增強")
        print(f"{'='*50}")
        
        try:
            # 創建資料載入器
            data_loaders = create_data_loaders(
                data_root=data_root,
                batch_size=1,  # 使用小批次進行測試
                target_size=(32, 32, 32),  # 使用小尺寸進行快速測試
                use_augmentation=True,
                augmentation_type=aug_type
            )
            
            # 測試訓練集資料載入器
            if 'train' in data_loaders:
                print(f"\n📊 {aug_type} 增強效果測試:")
                debug_data_loader(data_loaders['train'], num_samples=1)
            else:
                print("⚠️ 未找到訓練集資料")
                
        except FileNotFoundError as e:
            print(f"❌ 資料路徑錯誤: {e}")
            print("請確認您的資料目錄結構如下:")
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
            print(f"❌ 測試 {aug_type} 時發生錯誤: {e}")


