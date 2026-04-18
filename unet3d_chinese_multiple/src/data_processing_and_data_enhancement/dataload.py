import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset, DataLoader
import os
from pathlib import Path
import sys
from collections import OrderedDict
from copy import deepcopy
from typing import Union, Tuple, List
from scipy.ndimage import map_coordinates
from skimage.transform import resize
import pandas as pd

# 導入你的模組（根據實際路徑調整）
sys.path.append(r"D:\unet3d")
from .augmentation_function import *
import warnings
warnings.filterwarnings('ignore')

# ==================== nnUNet 風格的 Resampling 函數 ====================
ANISO_THRESHOLD = 3  # 各向異性閾值

def get_do_separate_z(spacing: Union[Tuple[float, ...], List[float], np.ndarray], 
                      anisotropy_threshold=ANISO_THRESHOLD):
    """判斷是否需要分離處理 z 軸"""
    do_separate_z = (np.max(spacing) / np.min(spacing)) > anisotropy_threshold
    return do_separate_z


def get_lowres_axis(spacing: Union[Tuple[float, ...], List[float], np.ndarray]):
    """找出各向異性的軸"""
    axis = np.where(max(spacing) / np.array(spacing) == 1)[0]
    return axis


def compute_new_shape(old_shape: Union[Tuple[int, ...], List[int], np.ndarray],
                      old_spacing: Union[Tuple[float, ...], List[float], np.ndarray],
                      new_spacing: Union[Tuple[float, ...], List[float], np.ndarray]) -> np.ndarray:
    """根據 spacing 計算新的形狀"""
    assert len(old_spacing) == len(old_shape)
    assert len(old_shape) == len(new_spacing)
    new_shape = np.array([int(round(i / j * k)) for i, j, k in zip(old_spacing, new_spacing, old_shape)])
    return new_shape


def determine_do_sep_z_and_axis(
        force_separate_z: bool,
        current_spacing,
        new_spacing,
        separate_z_anisotropy_threshold: float = ANISO_THRESHOLD) -> Tuple[bool, Union[int, None]]:
    """決定是否需要分離 z 軸處理以及哪個軸"""
    if force_separate_z is not None:
        do_separate_z = force_separate_z
        if force_separate_z:
            axis = get_lowres_axis(current_spacing)
        else:
            axis = None
    else:
        if get_do_separate_z(current_spacing, separate_z_anisotropy_threshold):
            do_separate_z = True
            axis = get_lowres_axis(current_spacing)
        elif get_do_separate_z(new_spacing, separate_z_anisotropy_threshold):
            do_separate_z = True
            axis = get_lowres_axis(new_spacing)
        else:
            do_separate_z = False
            axis = None

    if axis is not None:
        if len(axis) == 3:
            do_separate_z = False
            axis = None
        elif len(axis) == 2:
            do_separate_z = False
            axis = None
        else:
            axis = axis[0]
    return do_separate_z, axis


def resample_data_or_seg(data: np.ndarray, 
                         new_shape: Union[Tuple[float, ...], List[float], np.ndarray],
                         is_seg: bool = False, 
                         axis: Union[None, int] = None, 
                         order: int = 3,
                         do_separate_z: bool = False, 
                         order_z: int = 0, 
                         dtype_out=None):
    """
    nnUNet 風格的重採樣函數
    
    Args:
        data: 輸入資料 (c, x, y, z) 或 (x, y, z)
        new_shape: 目標形狀
        is_seg: 是否為分割標籤
        axis: 各向異性軸
        order: 插值階數（影像）
        do_separate_z: 是否分離處理 z 軸
        order_z: z 軸插值階數
        dtype_out: 輸出資料型別
    """
    # 確保資料是 4D (c, x, y, z)
    if data.ndim == 3:
        data = data[np.newaxis, ...]
        squeeze_output = True
    else:
        squeeze_output = False
    
    assert data.ndim == 4, "data must be (c, x, y, z)"
    assert len(new_shape) == data.ndim - 1

    if is_seg:
        # 對於分割標籤，使用最近鄰插值
        resize_fn = lambda img, shape, order, **kwargs: resize(img, shape, order=0, preserve_range=True, anti_aliasing=False)
        kwargs = {}
    else:
        resize_fn = resize
        kwargs = {'mode': 'edge', 'anti_aliasing': False}
    
    shape = np.array(data[0].shape)
    new_shape = np.array(new_shape)
    
    if dtype_out is None:
        dtype_out = data.dtype
    
    reshaped_final = np.zeros((data.shape[0], *new_shape), dtype=dtype_out)
    
    if np.any(shape != new_shape):
        data = data.astype(float, copy=False)
        
        if do_separate_z:
            assert axis is not None, 'If do_separate_z, we need to know what axis is anisotropic'
            
            # 決定 2D 平面的形狀
            if axis == 0:
                new_shape_2d = new_shape[1:]
            elif axis == 1:
                new_shape_2d = new_shape[[0, 2]]
            else:
                new_shape_2d = new_shape[:-1]

            for c in range(data.shape[0]):
                tmp = deepcopy(new_shape)
                tmp[axis] = shape[axis]
                reshaped_here = np.zeros(tmp)
                
                # 先在 2D 平面上 resize
                for slice_id in range(shape[axis]):
                    if axis == 0:
                        reshaped_here[slice_id] = resize_fn(data[c, slice_id], new_shape_2d, order, **kwargs)
                    elif axis == 1:
                        reshaped_here[:, slice_id] = resize_fn(data[c, :, slice_id], new_shape_2d, order, **kwargs)
                    else:
                        reshaped_here[:, :, slice_id] = resize_fn(data[c, :, :, slice_id], new_shape_2d, order, **kwargs)
                
                # 然後單獨處理 z 軸
                if shape[axis] != new_shape[axis]:
                    rows, cols, dim = new_shape[0], new_shape[1], new_shape[2]
                    orig_rows, orig_cols, orig_dim = reshaped_here.shape

                    # align_corners=False
                    row_scale = float(orig_rows) / rows
                    col_scale = float(orig_cols) / cols
                    dim_scale = float(orig_dim) / dim

                    map_rows, map_cols, map_dims = np.mgrid[:rows, :cols, :dim]
                    map_rows = row_scale * (map_rows + 0.5) - 0.5
                    map_cols = col_scale * (map_cols + 0.5) - 0.5
                    map_dims = dim_scale * (map_dims + 0.5) - 0.5

                    coord_map = np.array([map_rows, map_cols, map_dims])
                    
                    if not is_seg or order_z == 0:
                        reshaped_final[c] = map_coordinates(reshaped_here, coord_map, order=order_z, mode='nearest')
                    else:
                        # 對於分割標籤，使用多數投票
                        unique_labels = np.sort(pd.unique(reshaped_here.ravel()))
                        for i, cl in enumerate(unique_labels):
                            reshaped_final[c][np.round(
                                map_coordinates((reshaped_here == cl).astype(float), coord_map, order=order_z,
                                                mode='nearest')) > 0.5] = cl
                else:
                    reshaped_final[c] = reshaped_here
        else:
            # 不分離 z 軸，直接 3D resize
            for c in range(data.shape[0]):
                reshaped_final[c] = resize_fn(data[c], new_shape, order, **kwargs)
        
        if squeeze_output:
            reshaped_final = reshaped_final[0]
        
        return reshaped_final
    else:
        if squeeze_output:
            return data[0]
        return data


# ==================== 改良版 NII.GZ 檔案讀取 Dataset ====================
class MedicalImageDataset(Dataset):
    """
    醫學影像 NII.GZ 檔案讀取 Dataset (使用 nnUNet 風格的 resampling)
    """
    def __init__(self, data_root, split='train', image_suffix=['.nii.gz', '.nii'], 
                 mask_suffix=['.nii.gz', '.nii'], transform=None, target_size=None, 
                 num_classes=None, debug_labels=True, use_augmentation=True,
                 augmentation_type='medical', spacing=None, force_separate_z=None,
                ):
        """
        Args:
            data_root: 資料根目錄
            split: 資料分割 ('train', 'val', 'test')
            image_suffix: 影像檔案後綴
            mask_suffix: 標籤檔案後綴
            transform: 額外的資料增強變換
            target_size: 目標尺寸 (D, H, W)
            num_classes: 類別數量
            debug_labels: 是否啟用標籤除錯
            use_augmentation: 是否使用數據增強
            augmentation_type: 數據增強類型
            spacing: 原始資料的 spacing (z, y, x)，用於判斷各向異性
            force_separate_z: 強制是否分離 z 軸處理
        """
        self.data_root = Path(data_root)
        self.split = split
        self.external_transform = transform
        self.target_size = target_size
        self.num_classes = num_classes
        self.debug_labels = debug_labels
        self.spacing = spacing if spacing is not None else [1.0, 1.0, 1.0]  # 預設等向性
        self.force_separate_z = force_separate_z

        
        
        # 數據增強設定
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
            
            print(f"✅ 訓練集啟用 {augmentation_type} 數據增強 (nnUNet 風格 resampling)")
        else:
            self.augmentation_transform = None
            if split == 'train':
                print("❌ 訓練集未啟用數據增強")
            else:
                print(f"✅ {split.upper()} 集未啟用數據增強（正確）")
        
        # 設定影像和標籤目錄
        self.image_dir = self.data_root / split / 'images'
        self.mask_dir = self.data_root / split / 'labels'
        
        # 檢查目錄
        if not self.image_dir.exists():
            raise FileNotFoundError(f"影像目錄不存在: {self.image_dir}")
        if not self.mask_dir.exists():
            raise FileNotFoundError(f"標籤目錄不存在: {self.mask_dir}")
        
        # 確保 suffix 是列表
        if isinstance(image_suffix, str):
            image_suffix = [image_suffix]
        if isinstance(mask_suffix, str):
            mask_suffix = [mask_suffix]
        
        # 尋找配對的檔案
        self.image_files = []
        for suffix in image_suffix:
            self.image_files.extend(list(self.image_dir.glob(f'*{suffix}')))
        
        self.image_files = list(set(self.image_files))
        self.pairs = []
        
        for img_file in self.image_files:
            img_stem = img_file.stem
            if img_stem.endswith('.nii'):
                img_stem = img_stem[:-4]
            
            mask_found = False
            for suffix in mask_suffix:
                mask_file = self.mask_dir / f"{img_stem}{suffix}"
                if mask_file.exists():
                    self.pairs.append((img_file, mask_file))
                    mask_found = True
                    break
            
            if not mask_found:
                print(f"⚠️ 找不到對應的標籤檔案 {img_stem}")
        
        print(f"📊 [{split.upper()}] 找到 {len(self.pairs)} 對資料")
        
        if self.debug_labels and len(self.pairs) > 0:
            self.analyze_label_distribution()
    
    def analyze_label_distribution(self):
        """分析標籤分佈"""
        all_unique_values = set()
        sample_size = min(5, len(self.pairs))
        
        for i in range(sample_size):
            _, mask_path = self.pairs[i]
            try:
                mask = self.load_nii_image(mask_path)
                unique_vals = np.unique(mask)
                all_unique_values.update(unique_vals)
            except Exception as e:
                print(f"❌ 分析檔案 {mask_path} 時發生錯誤: {e}")
        
        print(f"🔍 標籤值範圍: {sorted(list(all_unique_values))}")
    
    def __len__(self):
        return len(self.pairs)
    
    def load_nii_image(self, file_path):
        """載入 NII.GZ 檔案並提取 spacing"""
        try:
            nii_img = nib.load(str(file_path))
            img_data = nii_img.get_fdata()
            img_data = np.array(img_data, dtype=np.float32)
            
            # 提取 spacing 資訊（如果還沒設定）
            if self.spacing is None or np.allclose(self.spacing, [1.0, 1.0, 1.0]):
                file_spacing = nii_img.header.get_zooms()[:3]  # (x, y, z) 或 (z, y, x)
                # 轉換為 (z, y, x) 順序
                self.spacing = [float(file_spacing[2]), float(file_spacing[1]), float(file_spacing[0])]
                # if self.debug_labels:
                #     print(f"📏 從檔案提取 spacing: {self.spacing} (z, y, x)")
            
            if img_data.ndim == 4:
                img_data = img_data[:, :, :, 0]
            elif img_data.ndim < 3:
                raise ValueError(f"圖像維度過低: {img_data.ndim}D")
            elif img_data.ndim > 4:
                raise ValueError(f"不支援的圖像維度: {img_data.ndim}D")
              
            # ✅ 將 nibabel 預設的 (X, Y, Z) 轉置為 (Z, Y, X) 以匹配 spacing 定義
            img_data = np.transpose(img_data, (2, 1, 0))
                               
            return img_data
        except Exception as e:
            print(f"❌ 讀取檔案 {file_path} 時發生錯誤: {e}")
            raise
    
    def normalize_image(self, image):
        """影像標準化"""
        p1, p99 = np.percentile(image, (1, 99))
        image = np.clip(image, p1, p99)
        
        mean = np.mean(image)
        std = np.std(image)
        if std > 0:
            image = (image - mean) / std
        
        return image
    
    def resize_volume(self, volume, target_size, is_seg=False):
        """
        使用 nnUNet 風格的 resampling
        支援各向異性處理
        """
        if target_size is None:
            return volume
        
        # 確保輸入是 3D
        if volume.ndim != 3:
            raise ValueError(f"resize_volume 只支援3D體積，收到 {volume.ndim}D")
        
        current_size = volume.shape
        
        if len(target_size) != 3:
            raise ValueError(f"target_size 必須是3D (D, H, W)")
        
        # 計算新的 spacing
        current_spacing = np.array(self.spacing)
        new_spacing = current_spacing * (np.array(current_size) / np.array(target_size))
        
        # 決定是否需要分離 z 軸
        do_separate_z, axis = determine_do_sep_z_and_axis(
            self.force_separate_z, 
            current_spacing, 
            new_spacing
        )
        
        # 使用 nnUNet 風格的 resampling
        order = 0 if is_seg else 3  # 標籤用最近鄰，影像用三次插值
        order_z = 0  # z 軸通常使用最近鄰或線性插值
        
        resized = resample_data_or_seg(
            volume,
            target_size,
            is_seg=is_seg,
            axis=axis,
            order=order,
            do_separate_z=do_separate_z,
            order_z=order_z
        )
        
        return resized
    
    def clean_labels(self, mask, file_path=None):
        """清理標籤（支援多標籤）"""
        if mask.ndim == 4:
            mask = mask[:, :, :, 0]

        mask = np.nan_to_num(mask, nan=0.0, posinf=0.0, neginf=0.0)
        mask = mask.astype(np.int64)

        # 檢查是否有超出 num_classes 範圍的標籤
        unique_vals = np.unique(mask)
        max_expected = self.num_classes - 1 if self.num_classes else 255
        if unique_vals.max() > max_expected:
            print(f"⚠️  發現超出範圍的標籤值: {unique_vals} (最大允許: {max_expected})")

        return mask
    
    def handle_dimension_mismatch(self, image, mask, file_path=None):
        """處理維度不匹配"""
        if image.shape != mask.shape:
            if image.ndim == 4 and mask.ndim == 3:
                image = image[:, :, :, 0]
            elif image.ndim == 3 and mask.ndim == 4:
                mask = mask[:, :, :, 0]
            
            if image.shape != mask.shape:
                min_shape = [min(i, m) for i, m in zip(image.shape, mask.shape)]
                image = image[:min_shape[0], :min_shape[1], :min_shape[2]]
                mask = mask[:min_shape[0], :min_shape[1], :min_shape[2]]
        
        return image, mask
    
    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        
        try:
            # 載入影像和標籤
            image = self.load_nii_image(img_path)
            mask = self.load_nii_image(mask_path)
            
            # 處理維度不匹配
            image, mask = self.handle_dimension_mismatch(image, mask, img_path.name)
            

            image = self.resize_volume(image, self.target_size, is_seg=False)
            mask = self.resize_volume(mask, self.target_size, is_seg=True)
            
            # 標準化影像
            image = self.normalize_image(image)
            
            # 清理標籤
            mask = self.clean_labels(mask, mask_path.name)
            
            # 添加通道維度
            image = image[np.newaxis, ...]
            
            # 轉換為 tensor
            image = torch.from_numpy(image).float()
            mask = torch.from_numpy(mask).long()
            
            # 驗證
            # 驗證標籤範圍
            mask_unique = torch.unique(mask)
            max_label = mask_unique.max().item()
            if self.num_classes and max_label >= self.num_classes:
                raise ValueError(
                    f"標籤最大值 {max_label} >= num_classes {self.num_classes}，"
                    f"請確認資料或調整 n_classes 設定"
                )
            
            # 應用數據增強
            if self.augmentation_transform is not None:
                image, mask = self.augmentation_transform(image, mask)
            
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
            raise


# ==================== 資料載入工具函數 ====================
def create_data_loaders(data_root, batch_size=2, target_size=(64, 64, 64), 
                       num_workers=2, use_augmentation=True, augmentation_type='medical',
                       spacing=None, force_separate_z=None, num_classes=4,
                       ):
    """
    創建資料載入器（使用 nnUNet 風格的 resampling）
    
    Args:
        spacing: 原始資料的 spacing (z, y, x)，例如 [3.0, 1.0, 1.0] 表示 z 軸解析度較低
        force_separate_z: 強制是否分離 z 軸處理（None 則自動判斷）
    """
    data_loaders = {}
    
    for split in ['train', 'val', 'test']:
        split_path = Path(data_root) / split
        if split_path.exists():
            debug_mode = (split == 'train')
            
            dataset = MedicalImageDataset(
                data_root=data_root,
                split=split,
                target_size=target_size,
                num_classes=num_classes,
                debug_labels=debug_mode,
                use_augmentation=use_augmentation,
                augmentation_type=augmentation_type,
                spacing=spacing,
                force_separate_z=force_separate_z,
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
            print(f"⚠️ {split} 目錄不存在，跳過")
    
    return data_loaders


# ==================== 除錯工具函數 ====================
def debug_data_loader(data_loader, num_samples=3):
    """除錯資料載入器"""
    print("\n=== DataLoader 除錯資訊 (nnUNet 風格 resampling) ===")
    print(f"批次大小: {data_loader.batch_size}")
    print(f"資料集大小: {len(data_loader.dataset)}")
    print(f"批次數量: {len(data_loader)}")
    
    if hasattr(data_loader.dataset, 'augmentation_transform') and data_loader.dataset.augmentation_transform:
        print("🔄 數據增強: 已啟用")
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
        print(f"標籤唯一值: {torch.unique(masks).tolist()}")
        
        unique_labels = torch.unique(masks).tolist()
        if set(unique_labels).issubset({0, 1}):
            print("✅ 標籤正確：只包含0和1")
            
            total_pixels = masks.numel()
            bg_pixels = (masks == 0).sum().item()
            fg_pixels = (masks == 1).sum().item()
            
            print(f"   背景(0): {bg_pixels} ({bg_pixels/total_pixels*100:.2f}%)")
            print(f"   前景(1): {fg_pixels} ({fg_pixels/total_pixels*100:.2f}%)")
        else:
            print(f"❌ 標籤錯誤：包含非二元值 {unique_labels}")
    
    print("=== 除錯完成 ===\n")


# ==================== 測試函數 ====================
def test_nnunet_style_dataset():
    """測試 nnUNet 風格的資料集"""
    data_root = r"D:\unet3d\dataset"
    
    print("🧪 測試 nnUNet 風格 resampling...")
    
    # 範例：假設你的資料 z 軸解析度較低（各向異性）
    spacing = [3.0, 1.0, 1.0]  # (z, y, x)
    
    try:
        data_loaders = create_data_loaders(
            data_root=data_root,
            batch_size=1,
            target_size=(32, 64, 64),
            use_augmentation=True,
            augmentation_type='medical',
            spacing=spacing,  # 提供 spacing 資訊
            force_separate_z=None  # 自動判斷
        )
        
        if 'train' in data_loaders:
            print("\n📊 測試結果:")
            debug_data_loader(data_loaders['train'], num_samples=2)
        else:
            print("⚠️ 未找到訓練集資料")
            
    except Exception as e:
        print(f"❌ 測試時發生錯誤: {e}")


if __name__ == '__main__':
    test_nnunet_style_dataset()
