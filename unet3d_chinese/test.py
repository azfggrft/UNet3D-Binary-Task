#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
推理測試腳本 - 使用優化版本（修復 Windows 路徑問題）
✅ 直接使用訓練時的數據讀取和預處理方式（nnUNet 風格）
✅ 新增功能：保存 Dice 分數到 txt、生成可視化結果
只需要填寫配置參數，其他都是自動化的
"""

import torch
from pathlib import Path
import numpy as np
from datetime import datetime
import json
try:
    import nibabel as nib
    from scipy import ndimage
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("⚠️  警告：缺少可視化依賴 (nibabel, scipy, matplotlib)，將無法生成圖片")

from test_inference import ModelInference  # 優先使用優化版本


# ========================================
# ⬇️  填寫你的配置（必需修改）
# ========================================

# 1️⃣ 模型路徑（必需）
# ✅ 使用正斜杠 / 或雙反斜杠 \\ 來避免轉義問題
MODEL_PATH = r"best_val_dice_model.pth"        #讀取訓練好的權重

# 2️⃣ 測試數據路徑（必需）
IMAGES_DIR = r"D:/UNet/dataset/test/images"              # 測試影像文件夾
MASKS_DIR = r"D:/UNet/dataset/test/labels"               # 測試遮罩文件夾（可選）

# 3️⃣ 模型配置（必需 - 根據你的訓練配置填寫）
MODEL_CONFIG = {
    'n_channels': 1,           # 輸入通道數（灰階=1, RGB=3）
    'n_classes': 2,            # 輸出類別數（二分類=2）
    'base_channels': 32,       # 基礎通道數
    'num_groups': 8,           # GroupNorm 組數
    'bilinear': False          # 是否使用雙線性上採樣
}

# 4️⃣ 其他參數（可選）
TARGET_SIZE = (64, 64, 64)    # 影像尺寸 (深度, 高度, 寬度) - 必須與訓練時相同
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'  # 自動選擇設備
OUTPUT_DIR = r"./result"  # 結果保存目錄
SAVE_RESULTS = True           # 是否保存結果

# 5️⃣ 可視化參數（新增）
GENERATE_VISUALIZATIONS = True   # 是否生成可視化圖片
VIZ_SLICE_MODE = 'max'        # 'middle'=中間切片, 'max'=最大激活切片, 'all'=所有切片
VIZ_DPI = 300                    # 圖片解析度 (DPI)
VIZ_CMAP = 'viridis'                 # 預測遮罩色圖：'hot', 'jet', 'viridis', 'gray', 'bone' 等

# 6️⃣ nnUNet 風格的 resampling 參數（可選）
SPACING = None                # 原始資料的 spacing (z, y, x)
FORCE_SEPARATE_Z = None       # 強制是否分離 z 軸處理

# ========================================


class VisualizationHelper:
    """可視化輔助類"""
    
    @staticmethod
    def load_nii_image(file_path):
        """載入 NII 影像"""
        if not VISUALIZATION_AVAILABLE:
            return None
        try:
            nii_img = nib.load(str(file_path))
            return nii_img.get_fdata()
        except Exception as e:
            print(f"   ⚠️  無法載入原始 NII 影像: {e}")
            return None
    
    @staticmethod
    def find_best_slice(mask, direction='z'):
        """找到最佳的展示切片（含有最多預測的切片）"""
        if mask.ndim != 3:
            return mask.shape[0] // 2
        
        if direction == 'z':
            slice_sums = np.sum(mask, axis=(1, 2))
        elif direction == 'y':
            slice_sums = np.sum(mask, axis=(0, 2))
        else:  # x
            slice_sums = np.sum(mask, axis=(0, 1))
        
        return np.argmax(slice_sums) if np.max(slice_sums) > 0 else mask.shape[0] // 2
    
    @staticmethod
    def normalize_image(img):
        """標準化圖像到 0-1"""
        img_min = np.min(img)
        img_max = np.max(img)
        if img_max - img_min > 0:
            return (img - img_min) / (img_max - img_min)
        return img
    
    @staticmethod
    def generate_comparison_image(image, ground_truth, prediction, output_path, dice_score, viz_cmap='viridis'):
        """Generate comparison image with 4 subplots: original image, ground truth, prediction, overlap with label counts"""
        if not VISUALIZATION_AVAILABLE:
            print(f"   ⚠️  Skipping visualization (missing dependencies)")
            return
        
        try:
            # Ensure 3D data
            if image.ndim != 3 or ground_truth.ndim != 3 or prediction.ndim != 3:
                print(f"   ⚠️  Incorrect data dimensions, skipping visualization")
                return
            
            # Select best slice
            if VIZ_SLICE_MODE == 'middle':
                slice_idx = image.shape[0] // 2
            elif VIZ_SLICE_MODE == 'max':
                slice_idx = VisualizationHelper.find_best_slice(ground_truth + prediction)
            else:
                slice_idx = image.shape[0] // 2
            
            # Extract slices
            img_slice = image[slice_idx, :, :]
            gt_slice = ground_truth[slice_idx, :, :]
            pred_slice = prediction[slice_idx, :, :]
            
            # Normalize image
            img_slice = VisualizationHelper.normalize_image(img_slice)
            
            # Create figure with larger size
            fig, axes = plt.subplots(2, 2, figsize=(14, 14), dpi=VIZ_DPI)
            fig.suptitle(f'{Path(output_path).stem} (Dice: {dice_score:.4f})', fontsize=16, fontweight='bold')
            
            # Subplot 1: Original Image
            axes[0, 0].imshow(img_slice, cmap='gray')
            axes[0, 0].set_title('Original Image', fontweight='bold', fontsize=12)
            axes[0, 0].axis('off')
            
            # Subplot 2: Ground Truth
            axes[0, 1].imshow(img_slice, cmap='gray')
            axes[0, 1].imshow(gt_slice, cmap='Reds', alpha=0.6)
            axes[0, 1].set_title('Ground Truth', fontweight='bold', fontsize=12)
            axes[0, 1].axis('off')
            
            # Add label count for ground truth
            gt_count = np.sum(gt_slice > 0)
            axes[0, 1].text(0.02, 0.98, f'Labels: {gt_count}', 
                           transform=axes[0, 1].transAxes,
                           fontsize=11, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
            
            # Subplot 3: Prediction
            axes[1, 0].imshow(img_slice, cmap='gray')
            axes[1, 0].imshow(pred_slice, cmap=viz_cmap, alpha=0.6)
            axes[1, 0].set_title('Prediction', fontweight='bold', fontsize=12)
            axes[1, 0].axis('off')
            
            # Add label count for prediction
            pred_count = np.sum(pred_slice > 0)
            axes[1, 0].text(0.02, 0.98, f'Labels: {pred_count}', 
                           transform=axes[1, 0].transAxes,
                           fontsize=11, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
            
            # Subplot 4: Overlap (Ground Truth=Red, Prediction=Blue)
            axes[1, 1].imshow(img_slice, cmap='gray')
            
            # Ground Truth - Red
            gt_overlay = np.zeros((*gt_slice.shape, 3), dtype=np.float32)
            gt_overlay[gt_slice > 0, 0] = 1.0  # Red
            axes[1, 1].imshow(gt_overlay, alpha=0.5)
            
            # Prediction - Blue
            pred_overlay = np.zeros((*pred_slice.shape, 3), dtype=np.float32)
            pred_overlay[pred_slice > 0, 2] = 1.0  # Blue
            axes[1, 1].imshow(pred_overlay, alpha=0.5)
            
            axes[1, 1].set_title('Overlap (Red=Ground Truth, Blue=Prediction)', fontweight='bold', fontsize=12)
            axes[1, 1].axis('off')
            
            # Add label information for overlap
            intersection = np.sum((gt_slice > 0) & (pred_slice > 0))
            axes[1, 1].text(0.02, 0.98, f'GT Labels: {gt_count}\nPred Labels: {pred_count}\nIntersection: {intersection}', 
                           transform=axes[1, 1].transAxes,
                           fontsize=10, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
            
            # Add legend
            red_patch = mpatches.Patch(color='red', label='Ground Truth', alpha=0.6)
            blue_patch = mpatches.Patch(color='blue', label='Prediction', alpha=0.6)
            fig.legend(handles=[red_patch, blue_patch], loc='lower center', ncol=2, fontsize=11)
            
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.08)
            
            # Save figure
            plt.savefig(output_path, dpi=VIZ_DPI, bbox_inches='tight')
            plt.close()
            
            print(f"   ✅ Saved visualization: {Path(output_path).name}")
            
        except Exception as e:
            print(f"   ⚠️  Error generating visualization: {e}")
            import traceback
            traceback.print_exc()


def run_inference_with_visualization():
    """執行推理的主函數（添加可視化）"""

    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print("\n" + "="*70)
    print("🚀 開始推理測試（使用訓練時相同的預處理方式）")
    print("="*70)
    
    # 驗證配置
    print("\n📋 驗證配置...")
    if not Path(MODEL_PATH).exists():
        print(f"❌ 模型文件不存在: {MODEL_PATH}")
        return None
    
    if not Path(IMAGES_DIR).exists():
        print(f"❌ 影像文件夾不存在: {IMAGES_DIR}")
        return None
    
    masks_dir = MASKS_DIR if Path(MASKS_DIR).exists() else None
    if MASKS_DIR and not masks_dir:
        print(f"⚠️  遮罩文件夾不存在: {MASKS_DIR}，將不計算指標")
    
    print(f"✅ 配置驗證通過")
    print(f"   📁 模型路徑: {MODEL_PATH}")
    print(f"   📁 影像文件夾: {IMAGES_DIR}")
    if masks_dir:
        print(f"   📁 遮罩文件夾: {MASKS_DIR}")
    print(f"   🖥️  計算設備: {DEVICE}")
    print(f"   📊 模型配置: {MODEL_CONFIG}")
    print(f"   📏 目標尺寸: {TARGET_SIZE}")
    if SPACING:
        print(f"   🔄 Spacing (z,y,x): {SPACING}")
    print(f"   🎨 生成可視化: {GENERATE_VISUALIZATIONS}")
    
    try:
        # 初始化推理器
        print("\n🔧 初始化推理器...")
        inferencer = ModelInference(
            model_path=MODEL_PATH,
            config=MODEL_CONFIG,
            device=DEVICE,
            spacing=SPACING,
            force_separate_z=FORCE_SEPARATE_Z
        )
        
        # 執行推理
        print("\n▶️  執行推理...")
        results, summary = inferencer.infer_folder(
            images_folder=IMAGES_DIR,
            masks_folder=masks_dir,
            target_size=TARGET_SIZE,
            save_results=SAVE_RESULTS,
            output_dir=OUTPUT_DIR
        )
        
        # 生成可視化和報告
        if summary:
            print("\n📊 後處理結果...")
            _save_detailed_reports(results, summary, OUTPUT_DIR, masks_dir)
            
            if GENERATE_VISUALIZATIONS and VISUALIZATION_AVAILABLE:
                print("\n🎨 生成可視化圖片...")
                _generate_all_visualizations(results, IMAGES_DIR, masks_dir, OUTPUT_DIR)
            
            # 顯示結果
            print("\n" + "="*70)
            print("✅ 推理完成！")
            print("="*70)
            print(f"總樣本數: {summary['total_samples']}")
            print(f"平均 Dice: {summary['avg_dice']:.6f} ({summary['avg_dice']*100:.2f}%)")
            print(f"標準差: {summary['std_dice']:.6f}")
            print(f"最佳 Dice: {summary['max_dice']:.6f}")
            print(f"最差 Dice: {summary['min_dice']:.6f}")
            print("="*70)
            
            # 顯示個別結果
            print("\n📊 個別結果:")
            print(f"{'樣本名稱':<40} {'Dice分數':<15}")
            print("-" * 55)
            for result in results:
                image_name = Path(result['image_path']).name
                if 'metrics' in result:
                    dice = result['metrics']['mean_dice']
                    print(f"{image_name:<40} {dice:<15.6f}")
                else:
                    print(f"{image_name:<40} {'N/A':<15}")
            
            # 顯示保存位置
            if SAVE_RESULTS:
                output_path = Path(OUTPUT_DIR)
                print(f"\n💾 結果已保存到: {output_path}")
                print(f"   - JSON 報告: {list(output_path.glob('inference_summary_*.json'))}")
                print(f"   - TXT 報告: {list(output_path.glob('dice_scores_*.txt'))}")
                if GENERATE_VISUALIZATIONS:
                    print(f"   - 可視化圖片: {list(output_path.glob('viz_*.png'))}")
            
            return results, summary
        else:
            print("❌ 推理失敗或沒有計算指標")
            return results, None
    
    except Exception as e:
        print(f"\n❌ 推理過程中出錯: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def _save_detailed_reports(results, summary, output_dir, masks_dir):
    """保存詳細的 txt 報告和 JSON 報告"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # ========== 保存 TXT 報告 ==========
    txt_file = output_path / f"dice_scores_{timestamp}.txt"
    
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("🔬 推理結果報告\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"模型配置:\n")
        f.write(f"  - 輸入通道數: {MODEL_CONFIG['n_channels']}\n")
        f.write(f"  - 輸出類別數: {MODEL_CONFIG['n_classes']}\n")
        f.write(f"  - 基礎通道數: {MODEL_CONFIG['base_channels']}\n")
        f.write(f"  - GroupNorm 組數: {MODEL_CONFIG['num_groups']}\n")
        f.write(f"  - 雙線性上採樣: {MODEL_CONFIG['bilinear']}\n")
        f.write(f"\n")
        
        # 摘要統計
        f.write("="*70 + "\n")
        f.write("📊 摘要統計\n")
        f.write("="*70 + "\n")
        f.write(f"總樣本數: {summary['total_samples']}\n")
        f.write(f"平均 Dice 分數: {summary['avg_dice']:.6f}\n")
        f.write(f"標準差: {summary['std_dice']:.6f}\n")
        f.write(f"最高 Dice 分數: {summary['max_dice']:.6f}\n")
        f.write(f"最低 Dice 分數: {summary['min_dice']:.6f}\n")
        f.write(f"\n")
        
        # 個別樣本詳細結果
        f.write("="*70 + "\n")
        f.write("📋 個別樣本詳細結果\n")
        f.write("="*70 + "\n")
        f.write(f"{'序號':<5} {'樣本名稱':<35} {'Dice分數':<15}\n")
        f.write("-"*55 + "\n")
        
        for idx, result in enumerate(results, 1):
            image_name = Path(result['image_path']).name
            if 'metrics' in result:
                dice = result['metrics']['mean_dice']
                f.write(f"{idx:<5} {image_name:<35} {dice:<15.6f}\n")
            else:
                f.write(f"{idx:<5} {image_name:<35} {'N/A':<15}\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("📝 注釋\n")
        f.write("="*70 + "\n")
        f.write(f"- Dice 分數範圍: 0-1 (越接近 1 越好)\n")
        f.write(f"- 可視化圖片保存位置: {output_path / 'visualizations'}\n")
        f.write(f"- JSON 詳細報告: inference_summary_{timestamp}.json\n")
    
    print(f"   ✅ 已保存 TXT 報告: {txt_file}")
    
    return txt_file


def _generate_all_visualizations(results, images_dir, masks_dir, output_dir):
    """為所有結果生成可視化圖片"""
    viz_dir = Path(output_dir) / 'visualizations'
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    images_path = Path(images_dir)
    masks_path = Path(masks_dir) if masks_dir else None
    
    for idx, result in enumerate(results, 1):
        try:
            if 'output' in result:
                output_debug = result['output']
                if output_debug.ndim == 5:
                    output_debug = output_debug[0, 0]
                elif output_debug.ndim == 4:
                    output_debug = output_debug[0]
                
                output_sigmoid_debug = 1 / (1 + np.exp(-output_debug))
                
                # if idx == 1:  # 只打印第一個樣本
                #     print(f"\n🔍 診斷信息 (樣本 {idx}):")
                #     print(f"   輸出值範圍: {output_debug.min():.4f} ~ {output_debug.max():.4f}")
                #     print(f"   Sigmoid 範圍: {output_sigmoid_debug.min():.4f} ~ {output_sigmoid_debug.max():.4f}")
                #     for threshold in [0.1, 0.2, 0.3, 0.4, 0.5]:
                #         count = np.sum(output_sigmoid_debug > threshold)
                #         print(f"   閾值 {threshold}: {count} 個像素")

            image_path = Path(result['image_path'])
            image_name = image_path.stem
            
            # 載入原始影像
            original_image = VisualizationHelper.load_nii_image(image_path)
            if original_image is None:
                continue
            
            # 載入真實標籤
            if masks_path and masks_path.exists():
                mask_path = masks_path / image_path.name
                if mask_path.exists():
                    ground_truth = VisualizationHelper.load_nii_image(mask_path)
                else:
                    ground_truth = None
            else:
                ground_truth = None
            
            # 從結果中提取預測
            if 'output' in result:
                # output 形狀為 [1, 1, D, H, W]（batch, channel, depth, height, width）
                output = result['output']
                if output.ndim == 5:
                    output = output[0, 0]  # 移除 batch 和 channel 維度
                elif output.ndim == 4:
                    output = output[0]  # 只移除 batch 維度
                
                # 應用 softmax 並取最大值
                output_softmax = 1 / (1 + np.exp(-output))  # 簡單的 sigmoid
                prediction = (output_softmax > 0.5).astype(np.uint8)
            else:
                prediction = None
            
            if original_image is None or prediction is None:
                continue
            
            # 確保尺寸一致（如果原始影像尺寸不同，需要調整）
            if original_image.shape != prediction.shape:
                # 使用最近鄰插值調整預測尺寸到原始影像尺寸
                from scipy.ndimage import zoom
                scale_factors = np.array(original_image.shape) / np.array(prediction.shape)
                prediction = zoom(prediction.astype(float), scale_factors, order=0, mode='constant', cval=0.0).astype(np.uint8)
            
            if ground_truth is not None and ground_truth.shape != original_image.shape:
                from scipy.ndimage import zoom
                scale_factors = np.array(original_image.shape) / np.array(ground_truth.shape)
                ground_truth = zoom(ground_truth.astype(float), scale_factors, order=0).astype(np.uint8)
            
            # 生成可視化
            dice_score = result['metrics']['mean_dice'] if 'metrics' in result else 0.0
            output_path = viz_dir / f"viz_{idx:03d}_{image_name}.png"
            
            if ground_truth is not None:
                VisualizationHelper.generate_comparison_image(
                    original_image, ground_truth, prediction, output_path, dice_score, viz_cmap=VIZ_CMAP
                )
            else:
                print(f"   ⚠️  跳過 {image_name}（沒有真實標籤）")
        
        except Exception as e:
            print(f"   ⚠️  生成 {Path(result['image_path']).name} 的可視化時出錯: {e}")


def run_single_sample_inference(image_path, mask_path=None):
    """
    推理單個樣本的函數
    
    Args:
        image_path: 單個影像的路徑
        mask_path: 對應的遮罩路徑（可選）
        
    Returns:
        dict: 推理結果
    """
    
    print("\n" + "="*70)
    print("🚀 推理單個樣本（使用訓練時相同的預處理方式）")
    print("="*70)
    
    try:
        # 初始化推理器
        print("🔧 初始化推理器...")
        inferencer = ModelInference(
            model_path=MODEL_PATH,
            config=MODEL_CONFIG,
            device=DEVICE,
            spacing=SPACING,
            force_separate_z=FORCE_SEPARATE_Z
        )
        
        # 推理單個樣本
        print(f"\n▶️  推理樣本: {Path(image_path).name}")
        result = inferencer.infer_single_sample(
            image_path=image_path,
            mask_path=mask_path,
            target_size=TARGET_SIZE
        )
        
        if result:
            print("\n" + "="*70)
            print("✅ 推理完成！")
            print("="*70)
            
            if 'metrics' in result:
                metrics = result['metrics']
                print(f"平均 Dice: {metrics['mean_dice']:.6f}")
                if 'per_class_dice' in metrics:
                    print(f"類別 Dice: {metrics['per_class_dice']}")
            
            print(f"輸出形狀: {result['output'].shape}")
            print("="*70)
            
            return result
        else:
            print("❌ 推理失敗")
            return None
    
    except Exception as e:
        print(f"❌ 推理過程中出錯: {e}")
        import traceback
        traceback.print_exc()
        return None


# ========================================
# 主程式
# ========================================

if __name__ == '__main__':
    import sys
    
    # 檢查命令行參數
    if len(sys.argv) > 1:
        if sys.argv[1] == 'single' and len(sys.argv) > 2:
            # 推理單個樣本
            image_path = sys.argv[2]
            mask_path = sys.argv[3] if len(sys.argv) > 3 else None
            result = run_single_sample_inference(image_path, mask_path)
        else:
            print("❌ 不支援的命令")
            print("\n使用方式:")
            print("  python test_inference_enhanced.py           # 推理整個文件夾")
            print("  python test_inference_enhanced.py single <image_path> [mask_path]  # 推理單個樣本")
    else:
        # 推理整個文件夾（含可視化）
        results, summary = run_inference_with_visualization()
    
    """
    ✨ 新增功能：
    
    1️⃣ 自動生成 TXT 報告：
       - 保存所有樣本的 Dice 分數
       - 計算平均值、標準差、最大值、最小值
       - 包含模型配置和時間戳
       - 檔案名稱：dice_scores_YYYYMMDD_HHMMSS.txt
    
    2️⃣ 自動生成可視化圖片：
       - 四個子圖：原始影像、真實標籤、預測標籤、重疊
       - 支援多種切片選擇模式
       - 自動調整預測結果尺寸到原始影像尺寸
       - 檔案名稱：viz_XXX_sample_name.png
       - 保存位置：{OUTPUT_DIR}/visualizations/
    
    3️⃣ 原函數名稱保持不變：
       - run_inference_with_visualization() (新增，原本 run_inference())
       - run_single_sample_inference() (保留)
    
    📊 TXT 報告內容包括：
       - 模型配置（通道數、類別數、基礎通道等）
       - 摘要統計（平均、標準差、最大、最小 Dice）
       - 個別樣本結果（序號、名稱、Dice 分數）
       - 時間戳和注釋
    
    🎨 可視化特性：
       - 智能選擇最佳切片顯示
       - 支援彩色疊加（紅色=真實，藍色=預測）
       - 高分辨率輸出（可配置 DPI）
       - 自動處理尺寸不匹配
    
    ⚠️  依賴要求：
       - nibabel (載入 NII 文件)
       - scipy (圖像縮放)
       - matplotlib (繪圖)
       
       安裝：pip install nibabel scipy matplotlib
    """