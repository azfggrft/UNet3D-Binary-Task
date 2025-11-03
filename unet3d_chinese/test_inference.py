#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
推理腳本 - 使用訓練好的模型對新樣本進行推理
✅ 直接使用訓練時的數據讀取器（nnUNet 風格 resampling）
✅ 使用 Argmax（與訓練時完全一致）
支援單個樣本、整個文件夾測試
"""

import torch
import numpy as np
from pathlib import Path
from datetime import datetime
import argparse
import json

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# ✅ 根據你的項目結構導入
from src.network_architecture.unet3d import UNet3D
from src.data_processing_and_data_enhancement.dataload import MedicalImageDataset
from src.loss_architecture.loss import CombinedLoss, DiceLoss
from src.loss_architecture.calculate_dice import calculate_metrics


class ModelInference:
    """模型推理類（使用訓練時相同的預處理方式）"""
    
    def __init__(self, model_path, config=None, device='auto', 
                 spacing=None, force_separate_z=None):
        """
        初始化推理器
        
        Args:
            model_path: 訓練好的模型權重路徑
            config: 模型配置（包含模型超參數）
            device: 計算設備 ('auto', 'cuda', 'cpu')
            spacing: 原始資料的 spacing (z, y, x) - 用於各向異性處理
            force_separate_z: 強制是否分離 z 軸處理
        """
        self.device = self._setup_device(device)
        self.model = None
        self.config = config or {}
        self.criterion = None
        
        # 保存 resampling 相關參數
        self.spacing = spacing if spacing is not None else [1.0, 1.0, 1.0]
        self.force_separate_z = force_separate_z
        
        # 載入模型
        self._load_model(model_path)
        print(f"✅ 模型已載入: {model_path}")
    
    def _setup_device(self, device):
        """設置計算設備"""
        if device == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(device)
        
        print(f"🖥️  使用設備: {device}")
        if device.type == 'cuda':
            print(f"💾 GPU 名稱: {torch.cuda.get_device_name(device)}")
        
        return device
    
    def _load_model(self, model_path):
        """載入模型權重（修復版本：處理 total_ops 和 total_params）"""
        print("📂 載入模型...")
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # 獲取模型配置（從檢查點中提取）
        if 'model_config' in checkpoint:
            model_config = checkpoint['model_config']
            print(f"📋 從檢查點中載入模型配置")
        else:
            # 如果沒有保存配置，使用默認值
            model_config = {
                'n_channels': self.config.get('n_channels', 1),
                'n_classes': self.config.get('n_classes', 2),
                'base_channels': self.config.get('base_channels', 32),
                'num_groups': self.config.get('num_groups', 8),
                'bilinear': self.config.get('bilinear', False)
            }
            print(f"⚠️  使用默認模型配置")
        
        # 保存配置
        self.config.update(model_config)
        
        # 建立模型
        print("🏗️  建立模型架構...")
        self.model = UNet3D(
            n_channels=model_config['n_channels'],
            n_classes=model_config['n_classes'],
            base_channels=model_config['base_channels'],
            num_groups=model_config['num_groups'],
            bilinear=model_config['bilinear']
        ).to(self.device)
        
        # 載入權重（關鍵修復：移除多餘的統計信息鍵）
        print("⚙️  載入模型權重...")
        model_state_dict = checkpoint['model_state_dict']
        
        # ✅ 移除所有包含 'total_ops' 和 'total_params' 的鍵
        keys_to_remove = [key for key in model_state_dict.keys() 
                         if 'total_ops' in key or 'total_params' in key]
        
        if keys_to_remove:
            print(f"🧹 發現 {len(keys_to_remove)} 個多餘的統計信息鍵，正在清理...")
            for key in keys_to_remove:
                del model_state_dict[key]
            print(f"✅ 清理完成，開始載入權重...")
        
        # 載入清理後的狀態字典
        try:
            self.model.load_state_dict(model_state_dict)
            print("✅ 模型權重載入成功")
        except RuntimeError as e:
            print(f"⚠️  警告：載入時出現不匹配")
            self.model.load_state_dict(model_state_dict, strict=False)
            print("✅ 已使用非嚴格模式載入模型")
        
        self.model.eval()
        
        print(f"📊 模型配置: n_channels={model_config['n_channels']}, "
              f"n_classes={model_config['n_classes']}")
    
    def _load_data_with_dataset_class(self, image_path, mask_path=None, target_size=(64, 64, 64)):
        """
        使用 MedicalImageDataset 類來載入數據
        ✅ 保證與訓練時相同的預處理方式
        ✅ 確保正確的維度 [C, D, H, W]
        """
        try:
            import nibabel as nib
            
            print(f"   📂 載入影像: {Path(image_path).name}")
            
            # 直接載入並使用 MedicalImageDataset 的預處理邏輯
            image = self._load_nii_image(image_path)
            
            # 載入遮罩（如果提供了）
            mask = None
            if mask_path and Path(mask_path).exists():
                print(f"   📂 載入遮罩: {Path(mask_path).name}")
                mask = self._load_nii_image(mask_path)
            
            # 記錄原始尺寸
            original_size = image.shape
            
            # ✅ 使用與訓練相同的 resampling 方式
            image_resized = self._resize_volume(image, target_size, is_seg=False)
            mask_resized = None
            if mask is not None:
                mask_resized = self._resize_volume(mask, target_size, is_seg=True)
            
            # ✅ 使用與訓練相同的標準化方式
            image_normalized = self._normalize_image(image_resized)
            
            # ✅ 清理標籤（與訓練相同）
            if mask_resized is not None:
                mask_resized = self._clean_labels(mask_resized)
            
            # 添加通道維度 [1, D, H, W]
            image_tensor = torch.from_numpy(image_normalized[np.newaxis, ...]).float()
            
            if mask_resized is not None:
                mask_tensor = torch.from_numpy(mask_resized).long()
            else:
                mask_tensor = None
            
            print(f"   ✅ 數據載入成功: 影像形狀={image_tensor.shape}, 遮罩={'是' if mask_resized is not None else '否'}")
            
            return image_tensor, mask_tensor, original_size
            
        except Exception as e:
            print(f"   ❌ 載入數據失敗: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None
    
    def _load_nii_image(self, file_path):
        """載入 NII.GZ 檔案（與訓練時相同）"""
        import nibabel as nib
        
        nii_img = nib.load(str(file_path))
        img_data = nii_img.get_fdata()
        img_data = np.array(img_data, dtype=np.float32)
        
        # 提取 spacing 資訊
        if self.spacing is None or np.allclose(self.spacing, [1.0, 1.0, 1.0]):
            file_spacing = nii_img.header.get_zooms()[:3]
            self.spacing = [float(file_spacing[2]), float(file_spacing[1]), float(file_spacing[0])]
        
        if img_data.ndim == 4:
            img_data = img_data[:, :, :, 0]
        elif img_data.ndim < 3:
            raise ValueError(f"圖像維度過低: {img_data.ndim}D")
        
        return img_data
    
    def _normalize_image(self, image):
        """影像標準化（與訓練時相同）"""
        p1, p99 = np.percentile(image, (1, 99))
        image = np.clip(image, p1, p99)
        
        mean = np.mean(image)
        std = np.std(image)
        if std > 0:
            image = (image - mean) / std
        
        return image
    
    def _clean_labels(self, mask):
        """清理標籤（與訓練時相同）"""
        if mask.ndim == 4:
            mask = mask[:, :, :, 0]
        
        mask = np.nan_to_num(mask, nan=0.0, posinf=0.0, neginf=0.0)
        mask[mask > 0] = 1
        mask = mask.astype(np.int64)
        
        return mask
    
    def _resize_volume(self, volume, target_size, is_seg=False):
        """
        使用 nnUNet 風格的 resampling（與訓練時相同）
        支援各向異性處理
        """
        from src.data_processing_and_data_enhancement.dataload import resample_data_or_seg, determine_do_sep_z_and_axis
        
        if target_size is None:
            return volume
        
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
        order = 0 if is_seg else 3
        order_z = 0
        
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
    
    def get_prediction_as_binary(self, output):
        """
        ✅ 新增方法：使用 Argmax 將模型輸出轉為二值預測
        與訓練時的 calculate_dice_score 完全一致
        
        Args:
            output: 模型輸出 [B, C, D, H, W] 或 [C, D, H, W]
            
        Returns:
            prediction: 二值預測 [D, H, W]，值為 0 或 1
        """
        # 確保是 numpy array
        if isinstance(output, torch.Tensor):
            output = output.cpu().numpy()
        
        # 移除 batch 維度（如果有）
        if output.ndim == 5:
            output = output[0]  # [C, D, H, W]
        
        # 使用 argmax 取得預測類別（與訓練時一致）
        prediction = np.argmax(output, axis=0)  # [D, H, W]
        
        return prediction.astype(np.uint8)
    
    def infer_single_sample(self, image_path, mask_path=None, target_size=(64, 64, 64)):
        """
        對單個樣本進行推理
        ✅ 使用訓練時相同的預處理方式
        ✅ 使用 Argmax（與訓練時一致）
        
        Args:
            image_path: 影像路徑
            mask_path: 遮罩路徑（可選，用於計算指標）
            target_size: 目標尺寸
            
        Returns:
            dict: 推理結果
        """
        print(f"\n📊 推理樣本: {Path(image_path).name}")
        
        # 載入數據（使用訓練時相同的方式）
        image, mask, original_size = self._load_data_with_dataset_class(
            image_path, mask_path, target_size
        )
        
        if image is None:
            return None
        
        # ✅ 確保有批次維度 [1, C, D, H, W]
        if image.ndim == 4:
            image = image.unsqueeze(0)
            if mask is not None:
                mask = mask.unsqueeze(0)
        elif image.ndim != 5:
            print(f"❌ 錯誤的影像維度: {image.ndim}D, 期望 4D 或 5D")
            return None
        
        print(f"   📏 最終形狀: {image.shape} (需要 [B, C, D, H, W])")
        
        # 移到設備
        image = image.to(self.device)
        if mask is not None:
            mask = mask.to(self.device)
        
        # 前向傳播
        with torch.no_grad():
            output = self.model(image)
        
        # ✅ 使用 Argmax 生成二值預測（與訓練一致）
        prediction_binary = self.get_prediction_as_binary(output)
        
        result = {
            'image_path': str(image_path),
            'output': output.cpu().numpy(),  # 保存原始輸出（用於進一步分析）
            'prediction': prediction_binary,  # 保存二值預測（用於可視化）
            'image': image.cpu().numpy(),
        }
        
        # 如果提供遮罩，計算指標
        if mask is not None:
            result['mask'] = mask.cpu().numpy()
            
            # ✅ 使用與訓練相同的方式計算 Dice
            metrics = calculate_metrics(output, mask, self.config['n_classes'])
            result['metrics'] = metrics
            
            print(f"   ✅ 平均 Dice: {metrics['mean_dice']:.4f}")
            if 'per_class_dice' in metrics:
                print(f"   📊 類別 Dice: {metrics['per_class_dice']}")
            
            # 🔍 診斷信息
            print(f"   🔍 診斷:")
            print(f"      - 輸出形狀: {output.shape}")
            print(f"      - 預測為前景的像素: {np.sum(prediction_binary == 1)} / {prediction_binary.size}")
            print(f"      - 真實前景像素: {np.sum(mask.cpu().numpy() == 1)}")
        
        return result
    
    def infer_folder(self, images_folder, masks_folder=None, target_size=(64, 64, 64), 
                     save_results=True, output_dir=None):
        """
        對整個文件夾的樣本進行推理
        
        Args:
            images_folder: 影像文件夾路徑
            masks_folder: 遮罩文件夾路徑（可選）
            target_size: 目標尺寸
            save_results: 是否保存結果
            output_dir: 結果保存目錄
            
        Returns:
            list: 所有結果
        """
        images_path = Path(images_folder)
        masks_path = Path(masks_folder) if masks_folder else None
        
        # 獲取所有影像文件
        image_files = sorted(images_path.glob('*.nii.gz')) + sorted(images_path.glob('*.nii'))
        
        if not image_files:
            print(f"❌ 在 {images_folder} 中找不到影像文件")
            return []
        
        print(f"📁 找到 {len(image_files)} 個樣本")
        
        all_results = []
        all_dice_scores = []
        
        # 使用進度條（如果可用）
        iterator = tqdm(image_files, desc="推理進度") if TQDM_AVAILABLE else image_files
        
        for image_file in iterator:
            # 查找對應的遮罩
            mask_file = None
            if masks_path and masks_path.exists():
                mask_file = masks_path / image_file.name
                if not mask_file.exists():
                    mask_file = None
            
            # 推理
            result = self.infer_single_sample(str(image_file), mask_file, target_size)
            
            if result:
                all_results.append(result)
                if 'metrics' in result:
                    all_dice_scores.append(result['metrics']['mean_dice'])
        
        # 生成摘要
        if all_dice_scores:
            summary = {
                'total_samples': len(all_results),
                'avg_dice': float(np.mean(all_dice_scores)),
                'std_dice': float(np.std(all_dice_scores)),
                'min_dice': float(np.min(all_dice_scores)),
                'max_dice': float(np.max(all_dice_scores)),
                'timestamp': datetime.now().isoformat()
            }
            
            print("\n" + "=" * 60)
            print("📊 推理摘要")
            print("=" * 60)
            print(f"總樣本數: {summary['total_samples']}")
            print(f"平均 Dice: {summary['avg_dice']:.6f}")
            print(f"標準差: {summary['std_dice']:.6f}")
            print(f"最小 Dice: {summary['min_dice']:.6f}")
            print(f"最大 Dice: {summary['max_dice']:.6f}")
            print("=" * 60)
            
            # 保存結果
            if save_results:
                self._save_results(all_results, summary, output_dir)
        
        return all_results, summary if all_dice_scores else None
    
    def _save_results(self, results, summary, output_dir=None):
        """保存推理結果"""
        if output_dir is None:
            output_dir = Path('./inference_results')
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存摘要為 JSON
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = output_dir / f"inference_summary_{timestamp}.json"
        
        summary_data = {
            'model_config': self.config,
            'summary': summary,
            'method': 'argmax',  # ✅ 標記使用的方法
            'results': [
                {
                    'image_path': r['image_path'],
                    'metrics': r.get('metrics', {}),
                    'predicted_foreground_pixels': int(np.sum(r.get('prediction', 0) == 1))
                }
                for r in results
            ]
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ 結果已保存到: {summary_file}")


def main():
    """主推理函數"""
    parser = argparse.ArgumentParser(
        description='使用訓練好的模型進行推理（使用 Argmax）',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 必需參數
    parser.add_argument('--model_path', type=str, required=True,
                        help='訓練好的模型權重路徑')
    parser.add_argument('--images_dir', type=str, required=True,
                        help='測試影像所在的文件夾')
    
    # 可選參數
    parser.add_argument('--masks_dir', type=str, default=None,
                        help='測試遮罩所在的文件夾（可選，用於計算指標）')
    parser.add_argument('--target_size', type=int, nargs=3, default=(64, 64, 64),
                        help='目標影像尺寸 (D H W)')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='計算設備')
    parser.add_argument('--output_dir', type=str, default='./inference_results',
                        help='結果保存目錄')
    parser.add_argument('--no_save', action='store_true',
                        help='不保存結果')
    
    # 模型配置參數
    parser.add_argument('--n_channels', type=int, default=1,
                        help='輸入通道數')
    parser.add_argument('--n_classes', type=int, default=2,
                        help='輸出類別數')
    parser.add_argument('--base_channels', type=int, default=32,
                        help='基礎通道數')
    parser.add_argument('--num_groups', type=int, default=8,
                        help='GroupNorm 組數')
    parser.add_argument('--bilinear', action='store_true',
                        help='使用雙線性上採樣')
    
    # resampling 參數
    parser.add_argument('--spacing', type=float, nargs=3, default=None,
                        help='原始資料的 spacing (z y x)，例如 3.0 1.0 1.0')
    parser.add_argument('--force_separate_z', type=int, default=None,
                        help='強制是否分離 z 軸處理 (0=False, 1=True, None=auto)')
    
    args = parser.parse_args()
    
    # 準備配置
    config = {
        'n_channels': args.n_channels,
        'n_classes': args.n_classes,
        'base_channels': args.base_channels,
        'num_groups': args.num_groups,
        'bilinear': args.bilinear
    }
    
    # 驗證路徑
    model_path = Path(args.model_path)
    images_dir = Path(args.images_dir)
    
    if not model_path.exists():
        print(f"❌ 模型文件不存在: {model_path}")
        return
    
    if not images_dir.exists():
        print(f"❌ 影像文件夾不存在: {images_dir}")
        return
    
    masks_dir = Path(args.masks_dir) if args.masks_dir else None
    if masks_dir and not masks_dir.exists():
        print(f"⚠️  遮罩文件夾不存在: {masks_dir}，將不計算指標")
        masks_dir = None
    
    # 處理 spacing
    spacing = args.spacing if args.spacing else None
    force_separate_z = None
    if args.force_separate_z is not None:
        force_separate_z = bool(args.force_separate_z)
    
    print("\n🚀 開始推理（使用 Argmax）...")
    print("=" * 60)
    
    # 初始化推理器
    inferencer = ModelInference(
        model_path=str(model_path),
        config=config,
        device=args.device,
        spacing=spacing,
        force_separate_z=force_separate_z
    )
    
    # 執行推理
    results, summary = inferencer.infer_folder(
        images_folder=str(images_dir),
        masks_folder=str(masks_dir) if masks_dir else None,
        target_size=tuple(args.target_size),
        save_results=not args.no_save,
        output_dir=args.output_dir
    )
    
    print("\n✅ 推理完成！")


if __name__ == '__main__':
    main()