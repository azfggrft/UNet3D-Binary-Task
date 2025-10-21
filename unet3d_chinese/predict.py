#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
簡化版 3D UNet 預測腳本（使用 nnUNet 風格 resampling）
直接執行即可，所有配置在腳本開頭
"""

import torch
import numpy as np
import nibabel as nib
from pathlib import Path
import sys
import warnings
import time
from tqdm import tqdm

warnings.filterwarnings('ignore')

# 🔥 導入改良版預測器（使用 nnUNet 風格資料處理）
from predictor import UNet3DPredictor

"""
使用說明:

1. 批量預測（預設）:
   python simple_predict.py

2. 修改模型配置:
   直接編輯 CONFIG['model_config'] 中的參數:
   - n_channels: 輸入通道數
   - n_classes: 輸出類別數  
   - base_channels: 基礎通道數（影響模型大小）
   - num_groups: GroupNorm群組數
   - bilinear: 上採樣方式

3. 修改 spacing 配置（重要！）:
   直接編輯 CONFIG['spacing']:
   - [1.0, 1.0, 1.0]: 等向性資料
   - [3.0, 1.0, 1.0]: z軸解析度較低（常見於醫學影像）
   - None: 自動從 NII 檔案讀取

4. 常見配置範例:
   # 小型模型（節省記憶體）:
   'base_channels': 32, 'num_groups': 4
   
   # 大型模型（更好效果）:
   'base_channels': 128, 'num_groups': 16
   
   # 多分類任務:
   'n_classes': 5  # 例如5個類別
   
   # 各向異性資料:
   'spacing': [3.0, 1.0, 1.0]  # z軸解析度較低

5. 在其他腳本中匯入使用:
   from simple_predict import predict_single_file, predict_images
   predict_single_file("image.nii.gz", "output.nii.gz")

輸出檔案:
- pred_*.nii.gz: 預測結果
- prediction_stats.json: 統計資訊（如果啟用）

注意事項:
- 使用 nnUNet 風格 resampling，自動處理各向異性資料
- 修改模型配置後，確保與訓練時的配置相符
- 如果權重載入失敗，會自動嘗試寬鬆模式
- base_channels 必須能被 num_groups 整除
- spacing 會影響 resampling 行為（各向異性偵測）
"""

# ==================== 配置參數 ====================
CONFIG = {
    # 模型和路徑
    'model_path': r"D:\UNet\unet3d_test\train_end\best_val_loss_model.pth",  # 模型權重路徑
    'input_folder': r"D:\UNet\dataset\test\images",     # 輸入資料夾
    'output_folder': r"D:\UNet\unet3d_test\predict",    # 輸出資料夾
    
    # 影像處理
    'target_size': (64, 64, 64),    # 目標尺寸，設為 None 保持原始大小
    'file_pattern': "*.nii",         # 檔案模式
    
    # ========== nnUNet 風格參數 ==========
    'spacing': None,                 # spacing (z, y, x)，例如 [3.0, 1.0, 1.0]
                                     # None = 自動從檔案讀取
                                     # [1.0, 1.0, 1.0] = 等向性
                                     # [3.0, 1.0, 1.0] = z軸解析度低（常見）
    
    # 系統設定
    'device': 'auto',               # 'auto', 'cpu', 'cuda'
    'save_stats': True,             # 是否保存統計資訊
    
    # ========== 訓練時的模型配置參數（可在此修改） ==========
    'model_config': {
        'n_channels': 1,        # 輸入通道數（1=灰階影像, 3=RGB影像）
        'n_classes': 2,         # 輸出類別數（2=二分類背景+前景, 3=三分類等）
        'base_channels': 32,    # 基礎通道數（影響模型容量：32/64/128）
        'num_groups': 8,        # GroupNorm的群組數量（通常為8或16）
        'bilinear': False       # 上採樣方式（False=轉置卷積, True=雙線性插值）
    }
}


class CustomUNet3DPredictor(UNet3DPredictor):
    """自定義 3D UNet 預測器，支援外部模型配置（使用 nnUNet 風格 resampling）"""
    
    def __init__(self, model_path: str, device: str = 'auto', 
                 spacing: list = None, custom_model_config: dict = None):
        """
        初始化預測器
        
        Args:
            model_path: 模型權重檔案路徑
            device: 計算設備
            spacing: spacing (z, y, x)，用於 nnUNet 風格 resampling
            custom_model_config: 自定義模型配置參數
        """
        self.custom_model_config = custom_model_config
        # 先設置 spacing，再調用父類初始化
        self.spacing = spacing if spacing is not None else [1.0, 1.0, 1.0]
        super().__init__(model_path, device, spacing=self.spacing)
    
    def _load_model(self):
        """載入模型權重（支援自定義配置）"""
        print(f"載入模型權重: {self.model_path}")
        
        try:
            checkpoint = self.safe_torch_load(self.model_path)
            
            # 提取檢查點中的模型配置
            if 'model_config' in checkpoint:
                saved_model_config = checkpoint['model_config']
                print(f"檢查點中的模型配置: {saved_model_config}")
            else:
                saved_model_config = {}
                print("檢查點中沒有找到模型配置")
            
            # 如果提供了自定義配置，則使用自定義配置
            if self.custom_model_config:
                print("🔧 使用自定義模型配置:")
                self.model_config = self.custom_model_config.copy()
                for key, value in self.model_config.items():
                    print(f"  {key}: {value}")
            else:
                # 否則使用預設配置邏輯
                self.model_config = {
                    'n_channels': saved_model_config.get('n_channels', 1),
                    'n_classes': saved_model_config.get('n_classes', 2),
                    'base_channels': saved_model_config.get('base_channels', 64),
                    'num_groups': saved_model_config.get('num_groups', 8),
                    'bilinear': saved_model_config.get('bilinear', False)
                }
                print("使用檢查點或預設模型配置")
            
            print(f"最終模型配置: {self.model_config}")
            
            # 建立模型
            from src.network_architecture.unet3d import UNet3D
            self.model = UNet3D(
                n_channels=self.model_config['n_channels'],
                n_classes=self.model_config['n_classes'],
                base_channels=self.model_config['base_channels'],
                num_groups=self.model_config['num_groups'],
                bilinear=self.model_config['bilinear']
            ).to(self.device)
            
            # 清理狀態字典並載入權重
            model_state_dict = checkpoint['model_state_dict']
            cleaned_state_dict = self._clean_state_dict(model_state_dict)
            
            # 嘗試載入權重
            try:
                self.model.load_state_dict(cleaned_state_dict, strict=True)
                print("✅ 模型權重載入成功（嚴格模式）")
            except RuntimeError as e:
                print(f"⚠️ 嚴格模式載入失敗，嘗試寬鬆模式: {str(e)[:100]}...")
                missing_keys, unexpected_keys = self.model.load_state_dict(cleaned_state_dict, strict=False)
                if missing_keys:
                    print(f"缺失的權重鍵: {len(missing_keys)} 個")
                if unexpected_keys:
                    print(f"多餘的權重鍵: {len(unexpected_keys)} 個")
                print("⚠️ 模型權重載入完成（寬鬆模式）")
            
            self.model.eval()
            
            # 顯示模型資訊
            total_params, trainable_params = self.model.get_model_size()
            print(f"模型參數: {total_params:,} ({trainable_params:,} 可訓練)")
            
            # 顯示訓練資訊
            if 'epoch' in checkpoint:
                print(f"模型訓練到第 {checkpoint['epoch']} epoch")
            if 'best_val_dice' in checkpoint:
                print(f"最佳驗證 Dice: {checkpoint['best_val_dice']:.4f}")
            if 'train_loss' in checkpoint:
                print(f"訓練損失: {checkpoint['train_loss']:.4f}")
            if 'val_loss' in checkpoint:
                print(f"驗證損失: {checkpoint['val_loss']:.4f}")
                
        except Exception as e:
            print(f"載入模型時發生錯誤: {e}")
            print("請檢查以下項目：")
            print("1. 模型檔案是否存在且完整")
            print("2. 模型配置參數是否正確")
            print("3. PyTorch 版本是否相容")
            raise


def predict_images():
    """執行影像預測（使用 nnUNet 風格 resampling）"""
    print("3D UNet 預測系統啟動（nnUNet 風格 resampling）")
    print("=" * 50)
    
    # 顯示配置
    print("配置參數:")
    for key, value in CONFIG.items():
        if key == 'model_config':
            print(f"  {key}:")
            for sub_key, sub_value in value.items():
                print(f"    {sub_key}: {sub_value}")
        elif key == 'spacing':
            if value is None:
                print(f"  {key}: None (自動從檔案讀取)")
            else:
                print(f"  {key}: {value} (z, y, x)")
        else:
            print(f"  {key}: {value}")
    print("-" * 50)
    
    try:
        # 建立自定義預測器
        print("載入模型...")
        predictor = CustomUNet3DPredictor(
            model_path=CONFIG['model_path'],
            device=CONFIG['device'],
            spacing=CONFIG['spacing'],
            custom_model_config=CONFIG['model_config']
        )
        
        # 執行批量預測
        print("開始預測...")
        stats = predictor.predict_folder(
            input_folder=CONFIG['input_folder'],
            output_folder=CONFIG['output_folder'],
            target_size=CONFIG['target_size'],
            file_pattern=CONFIG['file_pattern'],
            save_stats=CONFIG['save_stats']
        )
        
        print("\n📊 預測統計:")
        print(f"  成功預測: {stats['successful_predictions']} 個檔案")
        print(f"  失敗預測: {stats['failed_predictions']} 個檔案")
        print(f"  總耗時: {stats['total_inference_time']:.2f} 秒")
        print(f"  平均耗時: {stats['average_inference_time']:.2f} 秒/檔案")
        print(f"  Resampling 方法: {stats.get('resampling_method', 'nnUNet_style')}")
        
        print(f"\n✅ 結果已保存到: {CONFIG['output_folder']}")
        if CONFIG['save_stats']:
            print(f"✅ 統計資訊已保存: {Path(CONFIG['output_folder']) / 'prediction_stats.json'}")
        print("🎉 預測完成!")
        
    except FileNotFoundError as e:
        print(f"❌ 檔案或資料夾不存在: {e}")
        print("請檢查以下路徑:")
        print(f"  模型檔案: {CONFIG['model_path']}")
        print(f"  輸入資料夾: {CONFIG['input_folder']}")
    
    except Exception as e:
        print(f"❌ 預測過程中發生錯誤: {e}")
        import traceback
        traceback.print_exc()


def predict_single_file(image_path, output_path=None):
    """
    預測單個檔案的便利函數（使用 nnUNet 風格 resampling）
    
    Args:
        image_path: 輸入影像路徑
        output_path: 輸出路徑（可選）
    """
    if output_path is None:
        input_path = Path(image_path)
        output_path = input_path.parent / f"pred_{input_path.name}"
    
    try:
        print(f"📂 輸入檔案: {image_path}")
        print(f"📂 輸出檔案: {output_path}")
        
        # 建立自定義預測器
        predictor = CustomUNet3DPredictor(
            model_path=CONFIG['model_path'], 
            device=CONFIG['device'],
            spacing=CONFIG['spacing'],
            custom_model_config=CONFIG['model_config']
        )
        
        # 預測
        print("🔄 執行預測...")
        prediction, stats = predictor.predict_single_image(
            Path(image_path), 
            CONFIG['target_size']
        )
        
        # 保存
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        predictor.save_prediction(prediction, Path(image_path), Path(output_path))
        
        print(f"✅ 預測完成: {output_path}")
        print(f"📊 統計資訊:")
        print(f"  原始尺寸: {stats['original_size']}")
        print(f"  輸入尺寸: {stats['input_size']}")
        print(f"  Spacing: {stats.get('spacing', 'N/A')}")
        print(f"  前景比例: {stats['foreground_ratio']:.1%}")
        print(f"  推理時間: {stats['inference_time']:.2f} 秒")
        
        return prediction, stats
        
    except Exception as e:
        print(f"❌ 單檔案預測失敗: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def validate_config():
    """驗證配置參數的合理性"""
    config = CONFIG['model_config']
    
    print("🔍 驗證模型配置...")
    
    warnings_found = False
    
    # 檢查參數範圍
    if config['n_channels'] < 1:
        print("⚠️ 警告: n_channels 應該 >= 1")
        warnings_found = True
    
    if config['n_classes'] < 2:
        print("⚠️ 警告: n_classes 應該 >= 2")
        warnings_found = True
    
    if config['base_channels'] not in [16, 32, 64, 128, 256]:
        print(f"⚠️ 警告: base_channels={config['base_channels']} 不是常見值 (16,32,64,128,256)")
        warnings_found = True
    
    if config['base_channels'] % config['num_groups'] != 0:
        print(f"⚠️ 警告: base_channels({config['base_channels']}) 應該能被 num_groups({config['num_groups']}) 整除")
        warnings_found = True
    
    # 檢查 spacing
    if CONFIG['spacing'] is not None:
        spacing = CONFIG['spacing']
        if len(spacing) != 3:
            print(f"⚠️ 警告: spacing 應該是 3 個值 (z, y, x)，當前: {spacing}")
            warnings_found = True
        elif any(s <= 0 for s in spacing):
            print(f"⚠️ 警告: spacing 值應該都 > 0，當前: {spacing}")
            warnings_found = True
        else:
            # 檢查是否為各向異性
            max_s = max(spacing)
            min_s = min(spacing)
            if max_s / min_s > 3.0:
                print(f"ℹ️ 資訊: 偵測到各向異性資料 (ratio={max_s/min_s:.1f})，將使用分離 z 軸處理")
    
    if not warnings_found:
        print("✅ 配置驗證完成，無警告")
    else:
        print("⚠️ 配置驗證完成，發現警告（請檢查）")
    
    return not warnings_found


def show_help():
    """顯示使用說明"""
    help_text = """
    ╔═══════════════════════════════════════════════════════════════╗
    ║         3D UNet 簡易預測工具（nnUNet 風格 resampling）        ║
    ╚═══════════════════════════════════════════════════════════════╝
    
    📋 使用方式:
    
    1. 批量預測（預設）:
       python simple_predict.py
    
    2. 在 Python 中使用:
       from simple_predict import predict_single_file, predict_images
       
       # 預測單個檔案
       predict_single_file("image.nii.gz", "output.nii.gz")
       
       # 批量預測
       predict_images()
    
    ⚙️ 重要配置參數（在腳本開頭的 CONFIG 中修改）:
    
    model_config:
      - n_channels: 輸入通道數（1=灰階, 3=RGB）
      - n_classes: 輸出類別數（2=二分類, 3+=多分類）
      - base_channels: 模型容量（32=小, 64=中, 128=大）
      - num_groups: GroupNorm 群組數（通常 8 或 16）
      - bilinear: 上採樣方式（False=轉置卷積, True=雙線性）
    
    spacing:
      - None: 自動從 NII 檔案讀取（推薦）
      - [1.0, 1.0, 1.0]: 等向性資料
      - [3.0, 1.0, 1.0]: 各向異性（z軸解析度較低，常見）
    
    🎯 常見配置範例:
    
    # 小型模型（節省記憶體）:
    'base_channels': 32, 'num_groups': 4
    
    # 大型模型（更好效果）:
    'base_channels': 128, 'num_groups': 16
    
    # 各向異性資料（CT 掃描）:
    'spacing': [3.0, 1.0, 1.0]
    
    # 多分類任務（5個類別）:
    'n_classes': 5
    
    ⚠️ 注意事項:
    - 模型配置必須與訓練時相同
    - base_channels 必須能被 num_groups 整除
    - spacing 會影響 resampling 行為（各向異性偵測）
    - 使用 nnUNet 風格 resampling 可自動處理各向異性資料
    
    📁 輸出檔案:
    - pred_*.nii.gz: 預測結果（與輸入同名，加上 pred_ 前綴）
    - prediction_stats.json: 詳細統計資訊
    
    """
    print(help_text)


def main():
    """主函數"""
    import sys
    
    # 檢查是否需要顯示幫助
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help', 'help']:
        show_help()
        return
    
    print("\n" + "="*60)
    print("  簡化版 3D UNet 預測工具（nnUNet 風格 resampling）")
    print("="*60 + "\n")
    
    # 驗證配置
    config_valid = validate_config()
    print()
    
    if not config_valid:
        response = input("⚠️ 發現配置警告，是否繼續？ (y/n): ").strip().lower()
        if response != 'y':
            print("❌ 預測已取消")
            return
        print()
    
    # 執行批量預測
    predict_images()


if __name__ == '__main__':
    main()