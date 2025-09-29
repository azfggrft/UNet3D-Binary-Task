#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3D UNet 預測腳本
載入訓練好的模型權重，對指定資料夾中的 nii.gz or nii 檔案進行預測
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import nibabel as nib
from pathlib import Path
import sys
import argparse
import warnings
import json
from typing import List, Tuple, Optional, Dict
import time
from tqdm import tqdm
import os

# 忽略警告
warnings.filterwarnings('ignore')

from src.network_architecture.net_module import *
from src.network_architecture.unet3d import UNet3D
from trainer import EnhancedUNet3DTrainer

class UNet3DPredictor:
    """3D UNet 預測器類別"""
    
    def __init__(self, model_path: str, device: str = 'auto'):
        """
        初始化預測器
        
        Args:
            model_path: 模型權重檔案路徑 (.pth)
            device: 計算設備 ('auto', 'cpu', 'cuda')
        """
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"模型檔案不存在: {model_path}")
        
        # 設置設備
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"使用設備: {self.device}")
        if self.device.type == 'cuda':
            print(f"GPU 名稱: {torch.cuda.get_device_name(self.device)}")
            print(f"GPU 記憶體: {torch.cuda.get_device_properties(self.device).total_memory / 1024**3:.1f} GB")
        
        # 載入模型
        self.model = None
        self.model_config = None
        self._load_model()
        
        print("模型載入完成，準備進行預測")
    
    def safe_torch_load(self, path):
        """安全的 torch.load 函數，兼容 PyTorch 2.6+"""
        try:
            return torch.load(path, map_location=self.device, weights_only=False)
        except Exception as e1:
            try:
                print(f"使用 weights_only=False 載入失敗，嘗試 weights_only=True")
                return torch.load(path, map_location=self.device, weights_only=True)
            except Exception as e2:
                print(f"正在設置安全全域變數並重新載入...")
                torch.serialization.add_safe_globals([
                    'numpy._core.multiarray.scalar',
                    'numpy.core.multiarray.scalar',
                    'numpy.dtype',
                    'numpy.ndarray',
                    'collections.OrderedDict'
                ])
                return torch.load(path, map_location=self.device, weights_only=True)
    
    def _clean_state_dict(self, state_dict):
        """清理狀態字典，移除 thop 添加的額外鍵值"""
        # 需要移除的鍵值模式
        keys_to_remove = []
        for key in state_dict.keys():
            if 'total_ops' in key or 'total_params' in key:
                keys_to_remove.append(key)
        
        # 移除這些鍵值
        for key in keys_to_remove:
            del state_dict[key]
            
        return state_dict
    
    def _load_model(self):
        """載入模型權重"""
        print(f"載入模型權重: {self.model_path}")
        
        try:
            checkpoint = self.safe_torch_load(self.model_path)
            
            # 提取模型配置
            if 'model_config' in checkpoint:
                self.model_config = checkpoint['model_config']
                print(f"載入模型配置: {self.model_config}")
            else:
                # 使用預設配置
                self.model_config = {
                    'n_channels': 1,
                    'n_classes': 2,
                    'base_channels': 64,
                    'num_groups': 8,
                    'bilinear': False
                }
                print("使用預設模型配置")
            
            # 建立模型
            self.model = UNet3D(
                n_channels=self.model_config.get('n_channels', 1),
                n_classes=self.model_config.get('n_classes', 2),
                base_channels=self.model_config.get('base_channels', 64),
                num_groups=self.model_config.get('num_groups', 8),
                bilinear=self.model_config.get('bilinear', False)
            ).to(self.device)
            
            # 清理狀態字典並載入權重
            model_state_dict = checkpoint['model_state_dict']
            cleaned_state_dict = self._clean_state_dict(model_state_dict)
            self.model.load_state_dict(cleaned_state_dict)
            self.model.eval()
            
            # 顯示模型資訊
            total_params, trainable_params = self.model.get_model_size()
            print(f"模型參數: {total_params:,} ({trainable_params:,} 可訓練)")
            
            # 顯示訓練資訊
            if 'epoch' in checkpoint:
                print(f"模型訓練到第 {checkpoint['epoch']} epoch")
            if 'best_val_dice' in checkpoint:
                print(f"最佳驗證 Dice: {checkpoint['best_val_dice']:.4f}")
                
        except Exception as e:
            print(f"載入模型時發生錯誤: {e}")
            raise
    
    def load_nii_image(self, file_path: Path) -> np.ndarray:
        """載入 NII.GZ 檔案"""
        try:
            nii_img = nib.load(str(file_path))
            img_data = nii_img.get_fdata()
            img_data = np.array(img_data, dtype=np.float32)
            
            # 處理不同維度的影像
            if img_data.ndim == 4:
                # 🔧 若是4D影像 (X, Y, Z, T)，只取第0個時間點
                print(f"⚠️ 偵測到4D影像 {img_data.shape}，自動取最後一維的 index=0")
                img_data = img_data[..., 0]
            elif img_data.ndim == 3:
                # ✅ 3D影像，維持原樣
                print(f"✅ 偵測到3D影像 {img_data.shape}")
            elif img_data.ndim == 2:
                # 🔧 若是2D影像，添加深度維度 (H, W) -> (1, H, W)
                print(f"⚠️ 偵測到2D影像 {img_data.shape}，添加深度維度")
                img_data = img_data[np.newaxis, ...]
            elif img_data.ndim > 4:
                # 🚨 超過4D的影像，取前3個維度
                print(f"⚠️ 偵測到{img_data.ndim}D影像 {img_data.shape}，只取前3個維度")
                img_data = img_data[..., 0, 0] if img_data.ndim == 5 else img_data
                # 如果還是超過3D，繼續降維直到3D
                while img_data.ndim > 3:
                    img_data = img_data[..., 0]
            else:
                raise ValueError(f"不支援的影像維度: {img_data.ndim}D")
            
            # 確保最終結果是3D
            if img_data.ndim != 3:
                raise ValueError(f"處理後的影像維度不正確: {img_data.ndim}D，期望3D")
                
            print(f"📊 最終影像形狀: {img_data.shape}")
            return img_data
            
        except Exception as e:
            print(f"讀取檔案 {file_path} 時發生錯誤: {e}")
            raise
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """影像標準化"""
        # 移除異常值
        p1, p99 = np.percentile(image, (1, 99))
        image = np.clip(image, p1, p99)
        
        # Z-score 標準化
        mean = np.mean(image)
        std = np.std(image)
        if std > 0:
            image = (image - mean) / std
        
        return image
    
    def resize_volume(self, volume: np.ndarray, target_size: Tuple[int, int, int]) -> np.ndarray:
        """調整 3D 體積大小"""
        if target_size is None:
            return volume
            
        from scipy.ndimage import zoom
        
        current_size = volume.shape
        zoom_factors = [t/c for t, c in zip(target_size, current_size)]
        resized = zoom(volume, zoom_factors, order=1)  # 線性插值
        
        return resized
    
    def preprocess_image(self, image: np.ndarray, target_size: Optional[Tuple[int, int, int]] = None) -> torch.Tensor:
        """預處理單張影像"""
        # 調整大小
        if target_size is not None:
            image = self.resize_volume(image, target_size)
        
        # 標準化
        image = self.normalize_image(image)
        
        # 添加通道維度和批次維度：[D, H, W] -> [1, 1, D, H, W]
        image = image[np.newaxis, np.newaxis, ...]
        
        # 轉換為 tensor
        image = torch.from_numpy(image).float().to(self.device)
        
        return image
    
    def postprocess_prediction(self, prediction: torch.Tensor, original_size: Tuple[int, int, int]) -> np.ndarray:
        """後處理預測結果"""
        # 移除批次維度和通道維度
        if len(prediction.shape) == 5:  # [1, C, D, H, W]
            prediction = prediction.squeeze(0)
        
        # 如果是多類別，取最大概率的類別
        if prediction.shape[0] > 1:
            prediction = torch.argmax(prediction, dim=0)
        else:
            # 二分類情況
            prediction = torch.sigmoid(prediction.squeeze(0))
            prediction = (prediction > 0.5).float()
        
        # 轉為 numpy
        prediction = prediction.cpu().numpy().astype(np.uint8)
        
        # 調整回原始大小
        if prediction.shape != original_size:
            from scipy.ndimage import zoom
            zoom_factors = [o/p for o, p in zip(original_size, prediction.shape)]
            prediction = zoom(prediction, zoom_factors, order=0)  # 最近鄰插值
            prediction = prediction.astype(np.uint8)
        
        return prediction
    
    def predict_single_image(self, image_path: Path, target_size: Optional[Tuple[int, int, int]] = None) -> Tuple[np.ndarray, Dict]:
        """預測單張影像"""
        # 載入原始影像
        original_image = self.load_nii_image(image_path)
        original_size = original_image.shape
        
        # 預處理
        input_tensor = self.preprocess_image(original_image, target_size)
        
        # 預測
        with torch.no_grad():
            start_time = time.time()
            prediction = self.model(input_tensor)
            inference_time = time.time() - start_time
        
        # 後處理
        prediction_mask = self.postprocess_prediction(prediction, original_size)
        
        # 計算統計資訊
        stats = {
            'original_size': original_size,
            'input_size': input_tensor.shape[2:],  # 去掉批次和通道維度
            'inference_time': inference_time,
            'foreground_pixels': int(np.sum(prediction_mask > 0)),
            'total_pixels': int(prediction_mask.size),
            'foreground_ratio': float(np.sum(prediction_mask > 0) / prediction_mask.size)
        }
        
        return prediction_mask, stats
    
    def save_prediction(self, prediction: np.ndarray, original_nii_path: Path, output_path: Path):
        """保存預測結果為 NII.GZ 格式"""
        try:
            # 載入原始檔案以保持相同的仿射變換和頭部資訊
            original_nii = nib.load(str(original_nii_path))
            
            # 創建新的 NII 影像
            prediction_nii = nib.Nifti1Image(
                prediction, 
                original_nii.affine, 
                original_nii.header
            )
            
            # 更新資料類型
            prediction_nii.set_data_dtype(np.uint8)
            
            # 保存
            nib.save(prediction_nii, str(output_path))
            print(f"預測結果已保存: {output_path}")
            
        except Exception as e:
            print(f"保存預測結果時發生錯誤: {e}")
            raise
    
    def predict_folder(self, 
                      input_folder: str, 
                      output_folder: str, 
                      target_size: Optional[Tuple[int, int, int]] = None,
                      file_pattern: str = "*.nii.gz",
                      save_stats: bool = True) -> Dict:
        """
        預測整個資料夾中的所有 NII.GZ 檔案
        
        Args:
            input_folder: 輸入資料夾路徑
            output_folder: 輸出資料夾路徑
            target_size: 目標尺寸 (D, H, W)，None 表示保持原始大小
            file_pattern: 檔案模式
            save_stats: 是否保存統計資訊
        
        Returns:
            dict: 預測統計資訊
        """
        input_path = Path(input_folder)
        output_path = Path(output_folder)
        
        # 檢查輸入資料夾
        if not input_path.exists():
            raise FileNotFoundError(f"輸入資料夾不存在: {input_path}")
        
        # 創建輸出資料夾
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 尋找所有 NII.GZ 檔案
        nii_files = list(input_path.glob(file_pattern))
        if not nii_files:
            raise ValueError(f"在 {input_path} 中找不到符合 {file_pattern} 的檔案")
        
        print(f"找到 {len(nii_files)} 個檔案需要預測")
        print(f"輸出資料夾: {output_path}")
        if target_size:
            print(f"目標尺寸: {target_size}")
        
        # 預測統計
        all_stats = {}
        total_time = 0
        successful_predictions = 0
        
        # 使用進度條
        for nii_file in tqdm(nii_files, desc="預測進度"):
            try:
                # 預測
                prediction, stats = self.predict_single_image(nii_file, target_size)
                
                # 生成輸出檔案名稱
                output_file = output_path / f"pred_{nii_file.name}"
                
                # 保存預測結果
                self.save_prediction(prediction, nii_file, output_file)
                
                # 記錄統計
                stats['input_file'] = str(nii_file)
                stats['output_file'] = str(output_file)
                all_stats[nii_file.name] = stats
                
                total_time += stats['inference_time']
                successful_predictions += 1
                
                # 顯示簡要資訊
                tqdm.write(f"✅ {nii_file.name}: {stats['foreground_ratio']:.1%} 前景像素, "
                          f"{stats['inference_time']:.2f}s")
                
            except Exception as e:
                tqdm.write(f"❌ 預測 {nii_file.name} 失敗: {e}")
                continue
        
        # 計算總體統計
        summary_stats = {
            'total_files': len(nii_files),
            'successful_predictions': successful_predictions,
            'failed_predictions': len(nii_files) - successful_predictions,
            'total_inference_time': total_time,
            'average_inference_time': total_time / successful_predictions if successful_predictions > 0 else 0,
            'model_config': self.model_config,
            'target_size': target_size
        }
        
        print(f"\n預測完成!")
        print(f"成功: {successful_predictions}/{len(nii_files)}")
        print(f"總推理時間: {total_time:.2f} 秒")
        print(f"平均推理時間: {summary_stats['average_inference_time']:.2f} 秒/檔案")
        
        # 保存統計資訊
        if save_stats:
            stats_file = output_path / 'prediction_stats.json'
            full_stats = {
                'summary': summary_stats,
                'individual_files': all_stats
            }
            
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(full_stats, f, ensure_ascii=False, indent=2)
            
            print(f"統計資訊已保存: {stats_file}")
        
        return summary_stats

def main():
    """主函數"""
    parser = argparse.ArgumentParser(
        description='3D UNet 預測腳本',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('model_path', type=str, help='模型權重檔案路徑 (.pth)')
    parser.add_argument('input_folder', type=str, help='輸入資料夾路徑')
    parser.add_argument('output_folder', type=str, help='輸出資料夾路徑')
    
    parser.add_argument('--target_size', type=int, nargs=3, metavar=('D', 'H', 'W'),
                       help='目標尺寸 (深度 高度 寬度)，例如: --target_size 64 64 64')
    parser.add_argument('--device', type=str, default='auto', 
                       choices=['auto', 'cpu', 'cuda'], help='計算設備')
    parser.add_argument('--pattern', type=str, default='*.nii.gz',
                       help='檔案模式')
    parser.add_argument('--no_stats', action='store_true',
                       help='不保存統計資訊')
    
    args = parser.parse_args()
    
    print("🧠 3D UNet 預測系統")
    print("=" * 50)
    
    try:
        # 創建預測器
        predictor = UNet3DPredictor(args.model_path, args.device)
        
        # 轉換目標尺寸
        target_size = tuple(args.target_size) if args.target_size else None
        
        # 執行預測
        stats = predictor.predict_folder(
            input_folder=args.input_folder,
            output_folder=args.output_folder,
            target_size=target_size,
            file_pattern=args.pattern,
            save_stats=not args.no_stats
        )
        
        print("\n🎉 所有預測完成!")
        
    except KeyboardInterrupt:
        print("\n⏹️ 預測被用戶中斷")
    except Exception as e:
        print(f"\n❌ 預測過程中發生錯誤: {e}")
        import traceback
        traceback.print_exc()

def predict_single_example():
    """單檔案預測範例"""
    print("單檔案預測範例:")
    
    # 參數設定
    model_path = r"model.pth"
    image_path = r"image.nii.gz"
    output_path = r"predictions\pred_image.nii.gz"
    target_size = (64, 64, 64)  # 或 None 保持原始大小
    
    try:
        # 創建預測器
        predictor = UNet3DPredictor(model_path, device='auto')
        
        # 預測單張影像
        prediction, stats = predictor.predict_single_image(
            Path(image_path), 
            target_size
        )
        
        # 保存結果
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        predictor.save_prediction(prediction, Path(image_path), Path(output_path))
        
        print(f"預測統計: {stats}")
        print("預測完成!")
        
    except Exception as e:
        print(f"預測失敗: {e}")

if __name__ == "__main__":
    main()