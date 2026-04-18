
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import argparse
from pathlib import Path
import random
import numpy as np
import time
import json
import warnings
warnings.filterwarnings('ignore')

# 進度條庫
try:
    from tqdm import tqdm, trange
    TQDM_AVAILABLE = True
except ImportError:
    print("建議安裝 tqdm 以獲得更好的進度條體驗: pip install tqdm")
    TQDM_AVAILABLE = False

from src.network_architecture.net_module import *
from src.network_architecture.unet3d import UNet3D
from src.loss_architecture.loss import DiceLoss, CombinedLoss
from src.loss_architecture.calculate_dice import calculate_dice_score, calculate_metrics
from src.data_processing_and_data_enhancement.visualizer import UNet3DVisualizer

class EnhancedUNet3DTrainer:
    """修復 PyTorch 2.6 載入問題的增強版 3D UNet 訓練管理器"""
    
    def __init__(self, model, data_loaders, optimizer, criterion, device, 
                 save_dir='./checkpoints', log_interval=10, save_interval=10,
                 scheduler=None, visualize=True, plot_interval=10, 
                 use_progress_bar=True, training_config=None):  # 新增參數
        
        self.model = model.to(device) if model is not None else None
        self.data_loaders = data_loaders
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device if device is not None else torch.device('cpu')
        self.scheduler = scheduler
        self.save_dir = Path(save_dir) if save_dir is not None else Path('./checkpoints')
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.visualize = visualize
        self.plot_interval = plot_interval
        self.use_progress_bar = use_progress_bar and TQDM_AVAILABLE
        self.training_start_time = None
        self.training_end_time = None
        self.total_training_time = 0
        
        # 保存訓練配置（新增）
        self.training_config = training_config if training_config is not None else {}
        
        # 計算模型複雜度資訊
        self.model_info = self._calculate_model_complexity()
        
        # 初始化視覺化器
        if self.visualize:
            viz_dir = self.save_dir / 'visualizations'
            self.visualizer = UNet3DVisualizer(save_dir=viz_dir)
            print(f"視覺化功能已啟用，圖片保存至: {viz_dir}")
        
        # 訓練歷史記錄
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_dice': [],
            'val_dice': [],
            'learning_rate': [],
            'epoch_time': [],
            'gpu_memory': []
        }
        
        # 最佳模型追蹤
        self.best_val_dice = float('-inf')
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        
        self.start_time = None
        self.epoch_start_time = None
        
        # 進度條相關
        self.epoch_pbar = None
        self.train_pbar = None
        self.val_pbar = None

    def _calculate_model_complexity(self):
        """計算模型複雜度資訊"""
        try:
            # 獲取模型參數數量
            if hasattr(self.model, 'get_model_size'):
                total_params, trainable_params = self.model.get_model_size()
            else:
                total_params = sum(p.numel() for p in self.model.parameters())
                trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            # 嘗試計算FLOPs（可選）
            try:
                from thop import profile, clever_format
                
                # 創建一個示例輸入
                if hasattr(self.model, 'n_channels'):
                    n_channels = self.model.n_channels
                else:
                    n_channels = 1
                
                # 使用實際的訓練尺寸來計算FLOPs
                target_size = self.training_config.get('target_size', (64, 64, 64))
                sample_input = torch.randn(1, n_channels, *target_size).to(self.device)
                
                # 計算FLOPs
                flops, params = profile(self.model, inputs=(sample_input,), verbose=False)
                flops = flops / 2 / 1e9
                flops_str = f"{flops:.3f} GFLOPs"
                
                # 測試推理時間
                self.model.eval()
                inference_times = []
                
                with torch.no_grad():
                    # 暖身運行
                    for _ in range(3):
                        _ = self.model(sample_input)
                    
                    # 實際測量
                    for _ in range(10):
                        start_time = time.time()
                        _ = self.model(sample_input)
                        torch.cuda.synchronize() if self.device.type == 'cuda' else None
                        end_time = time.time()
                        inference_times.append(end_time - start_time)
                
                avg_inference_time = np.mean(inference_times)
                
            except Exception as e:
                print(f"計算FLOPs時發生錯誤: {e}")
                flops_str = "無法計算"
                avg_inference_time = 0
            
            return {
                'total_params': total_params,
                'trainable_params': trainable_params,
                'model_size_mb': total_params * 4 / 1024 / 1024,  # FP32
                'flops': flops_str,
                'avg_inference_time': avg_inference_time
            }
            
        except Exception as e:
            print(f"計算模型複雜度時發生錯誤: {e}")
            return {
                'total_params': 'Unknown',
                'trainable_params': 'Unknown', 
                'model_size_mb': 0,
                'flops': 'Unknown',
                'avg_inference_time': 0
            }
    
    def safe_torch_load(self, path):
        """安全的 torch.load 函數，兼容 PyTorch 2.6+"""
        try:
            # 首先嘗試使用 weights_only=False（適用於信任的檢查點）
            return torch.load(path, map_location=self.device, weights_only=False)
        except Exception as e1:
            try:
                # 如果失敗，嘗試使用 weights_only=True
                print(f"使用 weights_only=False 載入失敗，嘗試 weights_only=True")
                return torch.load(path, map_location=self.device, weights_only=True)
            except Exception as e2:
                # 最後嘗試使用安全全域變數
                print(f"正在設置安全全域變數並重新載入...")
                torch.serialization.add_safe_globals([
                    'numpy._core.multiarray.scalar',
                    'numpy.core.multiarray.scalar',
                    'numpy.dtype',
                    'numpy.ndarray',
                    'collections.OrderedDict'
                ])
                return torch.load(path, map_location=self.device, weights_only=True)
    
    def log_gpu_memory(self):
        """記錄GPU記憶體使用情況"""
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated(self.device) / 1024**3
            memory_cached = torch.cuda.memory_reserved(self.device) / 1024**3
            self.history['gpu_memory'].append({
                'allocated': memory_allocated,
                'cached': memory_cached
            })
        else:
            self.history['gpu_memory'].append({'allocated': 0, 'cached': 0})
    
    def _format_dice_score(self, dice_score):
        """
        安全地格式化 Dice 分數，處理列表和單一數值情況
        
        Args:
            dice_score: 可能是單一數值或列表
            
        Returns:
            str: 格式化後的字符串
        """
        if isinstance(dice_score, (list, tuple)):
            if len(dice_score) == 1:
                # 只有一個前景類別的情況
                return f'{dice_score[0]:.4f}'
            else:
                # 多類別的情況，顯示平均值
                mean_dice = np.mean(dice_score)
                return f'{mean_dice:.4f}'
        else:
            # 單一數值的情況
            return f'{dice_score:.4f}'
    
    def train_one_epoch(self, epoch):
        """訓練一個 epoch - 修改為樣本級別平均"""
        self.epoch_start_time = time.time()
        self.model.train()
        train_loss = 0.0
        train_dice_scores = []  # ✅ 改為存儲每個樣本的分數
        
        num_batches = len(self.data_loaders['train'])
        
        # 檢查是否在 warmup 階段
        has_warmup = hasattr(self, 'warmup_scheduler') and self.warmup_scheduler is not None
        warmup_epochs = getattr(self, 'warmup_epochs', 0) if has_warmup else 0
        is_warmup_stage = has_warmup and epoch < warmup_epochs
        
        # 創建訓練進度條
        if self.use_progress_bar:
            desc_prefix = "🔥 Warmup" if is_warmup_stage else "🚀 Train"
            self.train_pbar = tqdm(
                enumerate(self.data_loaders['train']), 
                total=num_batches,
                desc=f"{desc_prefix} E{epoch+1:3d}",
                leave=False,
                ncols=100,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}'
            )
        else:
            stage_info = "Warmup 訓練" if is_warmup_stage else "正常訓練"
            print(f"\n開始{stage_info} Epoch {epoch+1}")
        
        data_iter = self.train_pbar if self.use_progress_bar else enumerate(self.data_loaders['train'])
        
        for batch_idx, batch in data_iter:
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            # 確保標籤為二元分類
            # masks[masks > 0] = 1
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            
            loss = self.criterion(outputs, masks)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            train_loss += loss.item()
            
            # ✅ 修改：計算每個樣本的 Dice 分數（與驗證和測試一致）
            with torch.no_grad():
                batch_size = images.size(0)
                batch_dice_scores = []  # 當前批次的 Dice 分數
                
                for sample_idx in range(batch_size):
                    single_output = outputs[sample_idx:sample_idx+1]
                    single_mask = masks[sample_idx:sample_idx+1]
                    
                    # 使用 calculate_metrics 保持一致性
                    single_metrics = calculate_metrics(
                        single_output, single_mask, num_classes=self.model.n_classes
                    )
                    single_dice = single_metrics['mean_dice']
                    train_dice_scores.append(single_dice)  # 添加到總列表
                    batch_dice_scores.append(single_dice)   # 添加到批次列表
                
                # 計算當前批次的平均 Dice（用於進度條顯示）
                current_batch_dice = np.mean(batch_dice_scores)
            
            # 更新進度條 - 顯示當前批次的平均 Dice
            if self.use_progress_bar:
                current_lr = self.optimizer.param_groups[0]['lr']
                
                postfix_data = {
                    'Loss': f'{loss.item():.4f}',
                    'Dice': f'{current_batch_dice:.4f}',  # ✅ 使用當前批次平均
                    'LR': f'{current_lr:.2e}'
                }
                
                if is_warmup_stage:
                    postfix_data['Stage'] = f'W{epoch+1}/{warmup_epochs}'
                    
                self.train_pbar.set_postfix(postfix_data)
                
            elif batch_idx % 5 == 0:
                progress = 100.0 * batch_idx / num_batches
                current_lr = self.optimizer.param_groups[0]['lr']
                
                stage_prefix = "[WARMUP]" if is_warmup_stage else "[TRAIN]"
                print(f'\r  {stage_prefix} 批次 [{batch_idx:3d}/{num_batches}] {progress:5.1f}% | '
                    f'損失: {loss.item():.4f} | Dice: {current_batch_dice:.4f} | LR: {current_lr:.6f}', end='')
        
        # 關閉訓練進度條
        if self.use_progress_bar and self.train_pbar:
            self.train_pbar.close()
        
        avg_train_loss = train_loss / num_batches
        
        # ✅ 計算樣本級別平均 Dice（所有樣本的平均）
        avg_train_dice = np.mean(train_dice_scores) if train_dice_scores else 0.0
        
        epoch_time = time.time() - self.epoch_start_time
        self.history['epoch_time'].append(epoch_time)
        self.log_gpu_memory()
        
        if not self.use_progress_bar:
            stage_info = "Warmup 訓練" if is_warmup_stage else "訓練"
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f'\n  {stage_info}完成 | 平均損失: {avg_train_loss:.4f} | 平均Dice: {avg_train_dice:.4f} | '
                f'當前LR: {current_lr:.6f} | 耗時: {epoch_time:.1f}s')
        
        return avg_train_loss, avg_train_dice
    
    def validate_one_epoch(self, epoch):
        """驗證一個 epoch - 修改為樣本級別平均"""
        if 'val' not in self.data_loaders:
            return 0.0, 0.0
            
        self.model.eval()
        val_loss = 0.0
        val_dice_scores = []  # ✅ 改為存儲每個樣本的分數
        
        num_batches = len(self.data_loaders['val'])
        
        # 創建驗證進度條
        if self.use_progress_bar:
            self.val_pbar = tqdm(
                enumerate(self.data_loaders['val']), 
                total=num_batches,
                desc=f"✨ Valid E{epoch+1:3d}",
                leave=False,
                ncols=100,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}'
            )
        else:
            print(f"開始驗證...")
        
        data_iter = self.val_pbar if self.use_progress_bar else enumerate(self.data_loaders['val'])
        
        with torch.no_grad():
            for batch_idx, batch in data_iter:
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                # masks[masks > 0] = 1
                
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                val_loss += loss.item()
                
                # ✅ 計算每個樣本的 Dice 分數（與訓練和測試一致）
                batch_size = images.size(0)
                batch_dice_scores = []  # 當前批次的 Dice 分數
                
                for sample_idx in range(batch_size):
                    single_output = outputs[sample_idx:sample_idx+1]
                    single_mask = masks[sample_idx:sample_idx+1]
                    
                    # 使用 calculate_metrics 保持一致性
                    single_metrics = calculate_metrics(
                        single_output, single_mask, num_classes=self.model.n_classes
                    )
                    single_dice = single_metrics['mean_dice']
                    val_dice_scores.append(single_dice)      # 添加到總列表
                    batch_dice_scores.append(single_dice)    # 添加到批次列表
                
                # 計算當前批次的平均 Dice（用於進度條顯示）
                current_batch_dice = np.mean(batch_dice_scores)
                
                # 更新進度條
                if self.use_progress_bar:
                    self.val_pbar.set_postfix({
                        'Loss': f'{loss.item():.4f}',
                        'Dice': f'{current_batch_dice:.4f}'
                    })
                elif batch_idx % 3 == 0:
                    progress = 100.0 * batch_idx / num_batches
                    print(f'\r  驗證進度 {progress:5.1f}% | 當前Dice: {current_batch_dice:.4f}', end='')
        
        # 關閉驗證進度條
        if self.use_progress_bar and self.val_pbar:
            self.val_pbar.close()
        
        avg_val_loss = val_loss / num_batches
        
        # ✅ 計算樣本級別平均 Dice（所有樣本的平均）
        avg_val_dice = np.mean(val_dice_scores) if val_dice_scores else 0.0
        
        if not self.use_progress_bar:
            print(f'\n  驗證完成 | 平均損失: {avg_val_loss:.4f} | 平均Dice: {avg_val_dice:.4f}')
        
        return avg_val_loss, avg_val_dice
    
    def visualize_epoch_results(self, epoch, sample_predictions=None):
        """視覺化當前epoch的結果"""
        if not self.visualize:
            return
        
        try:
            if epoch % self.plot_interval == 0:
                if self.use_progress_bar:
                    tqdm.write("更新訓練曲線...")
                else:
                    print("更新訓練曲線...")
                
                # 檢查visualizer的可用方法
                if hasattr(self.visualizer, 'plot_training_curves'):
                    self.visualizer.plot_training_curves(
                        self.history,
                        title=f"Training curve (up to Epoch {epoch+1})",
                        save_name=f"training_curves_epoch_{epoch+1}.png"
                    )
                else:
                    print("視覺化器沒有 plot_training_curves 方法")
            
            if sample_predictions is not None:
                images, masks, predictions = sample_predictions
                if hasattr(self.visualizer, 'plot_3d_predictions'):
                    self.visualizer.plot_3d_predictions(
                        images, masks, predictions,
                        save_name=f"predictions_epoch_{epoch+1}.png",
                        max_samples=4
                    )
                else:
                    print("視覺化器沒有 plot_3d_predictions 方法")
                
        except Exception as e:
            if self.use_progress_bar:
                tqdm.write(f"視覺化過程出現錯誤: {e}")
            else:
                print(f"視覺化過程出現錯誤: {e}")
            # 不中斷訓練，繼續執行
    
    def save_checkpoint(self, epoch, is_dice_best=False, is_loss_best=False):
        """保存檢查點 - 使用實際的訓練配置而非模型預設值"""
        # 從訓練配置中提取模型配置（優先使用實際配置）
        if self.training_config:
            model_config = {
                'n_channels': self.training_config.get('n_channels', getattr(self.model, 'n_channels', 1)),
                'n_classes': self.training_config.get('n_classes', getattr(self.model, 'n_classes', 2)),
                'base_channels': self.training_config.get('base_channels', getattr(self.model, 'base_channels', 64)),
                'num_groups': self.training_config.get('num_groups', getattr(self.model, 'num_groups', 8)),
                'bilinear': self.training_config.get('bilinear', getattr(self.model, 'bilinear', False))
            }
        else:
            # 降級方案：從模型動態獲取
            model_config = {
                'n_channels': getattr(self.model, 'n_channels', 1),
                'n_classes': getattr(self.model, 'n_classes', 2),
                'base_channels': getattr(self.model, 'base_channels', 64),
                'num_groups': getattr(self.model, 'num_groups', 8),
                'bilinear': getattr(self.model, 'bilinear', False)
            }
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.history['train_loss'],
            'val_losses': self.history['val_loss'],
            'train_dice': self.history['train_dice'],
            'val_dice': self.history['val_dice'],
            'learning_rate': self.history['learning_rate'],
            'epoch_time': self.history['epoch_time'],  # 新增
            'gpu_memory': self.history['gpu_memory'],  # 新增：GPU記憶體歷史
            'best_val_dice': self.best_val_dice,
            'best_val_loss': self.best_val_loss,
            'total_training_time': self.total_training_time,
            
            # 使用實際的訓練配置
            'model_config': model_config,
            
            # 額外保存完整的訓練配置（可選）
            'training_config': self.training_config,
            
            # 模型複雜度資訊
            'model_info': self.model_info
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # 每個 epoch 都存 latest
        latest_path = self.save_dir / 'latest_checkpoint.pth'
        torch.save(checkpoint, latest_path)

        # 每隔 save_interval 存一份帶編號的檢查點
        if (epoch + 1) % self.save_interval == 0:
            epoch_path = self.save_dir / f'epoch_{epoch+1}_checkpoint.pth'
            torch.save(checkpoint, epoch_path)
            print(f"定期檢查點已保存: {epoch_path}")
        
        
        if is_dice_best:
            best_path = self.save_dir / 'best_val_dice_model.pth'
            torch.save(checkpoint, best_path)
            print(f"最佳模型已保存(DICE): {best_path}")
        
        if is_loss_best:
            best_loss_path = self.save_dir / 'best_val_loss_model.pth'
            torch.save(checkpoint, best_loss_path)
            print(f"最佳模型已保存(LOSS): {best_loss_path}")
            
        # 定期顯示保存資訊
        if (epoch + 1) % (self.save_interval * 2) == 0:
            print(f"檢查點已保存至: {latest_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """載入模型檢查點 - 修復 PyTorch 2.6 兼容性問題"""
        try:
            checkpoint = self.safe_torch_load(checkpoint_path)
        except Exception as e:
            print(f"載入檢查點失敗: {e}")
            raise
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_dice = checkpoint.get('best_val_dice', 0.0)

         # 恢復完整歷史記錄
        self.history['train_loss'] = checkpoint.get('train_losses', [])
        self.history['val_loss'] = checkpoint.get('val_losses', [])
        self.history['train_dice'] = checkpoint.get('train_dice', [])
        self.history['val_dice'] = checkpoint.get('val_dice', [])
        self.history['learning_rate'] = checkpoint.get('learning_rate', [])
        self.history['epoch_time'] = checkpoint.get('epoch_time', [])  # 新增
        self.history['gpu_memory'] = checkpoint.get('gpu_memory', [])  # 新增：恢復GPU記憶體歷史
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        start_epoch = checkpoint['epoch'] + 1
        print(f"載入檢查點，從第 {start_epoch} epoch 開始")
        print(f"歷史最佳Dice: {self.best_val_dice:.4f}")
        # 顯示 GPU 記憶體資訊
        if self.history['gpu_memory']:
            latest_gpu = self.history['gpu_memory'][-1]
            print(f"上次訓練 GPU 記憶體: 已分配 {latest_gpu['allocated']:.2f} GB, 已緩存 {latest_gpu['cached']:.2f} GB")
        return start_epoch
    
    def get_sample_predictions(self):
        """獲取樣本預測結果用於視覺化"""
        if 'val' not in self.data_loaders:
            return None
        
        self.model.eval()
        with torch.no_grad():
            for batch in self.data_loaders['val']:
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                # masks[masks > 0] = 1
                
                outputs = self.model(images)
                
                return (images[:4], masks[:4], outputs[:4])
        return None
    
    def train(self, num_epochs, resume_from=None, early_stopping_patience=None):
        """訓練模型 - 修復 warmup 學習率顯示問題"""
        print(f"開始訓練 {num_epochs} epochs")
        
        # 記錄訓練開始時間
        self.training_start_time = time.time()
        
        start_epoch = 0
        
        # 檢查是否有 warmup scheduler
        has_warmup = hasattr(self, 'warmup_scheduler') and self.warmup_scheduler is not None
        warmup_epochs = getattr(self, 'warmup_epochs', 0) if has_warmup else 0
        
        if has_warmup:
            print(f"🔥 Warmup 已啟用: 前 {warmup_epochs} epochs 將使用 warmup 調度")
        
        # 從檢查點恢復
        if resume_from and Path(resume_from).exists():
            checkpoint = self.safe_torch_load(resume_from)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            
            # 恢復歷史記錄
            self.history['train_loss'] = checkpoint.get('train_losses', [])
            self.history['val_loss'] = checkpoint.get('val_losses', [])
            self.history['train_dice'] = checkpoint.get('train_dice', [])
            self.history['val_dice'] = checkpoint.get('val_dice', [])
            
            # 恢復學習率歷史（如果存在）
            if 'learning_rate' in checkpoint:
                self.history['learning_rate'] = checkpoint['learning_rate']
            else:
                # 如果沒有學習率歷史，用當前學習率填充
                current_lr = self.optimizer.param_groups[0]['lr']
                self.history['learning_rate'] = [current_lr] * len(self.history['train_loss'])
            
            self.best_val_dice = checkpoint.get('best_val_dice', 0)
            
            # 恢復已訓練時間
            self.total_training_time = checkpoint.get('total_training_time', 0)
            
            print(f"從 epoch {start_epoch} 恢復訓練")
            print(f"已訓練時間: {self.total_training_time/3600:.2f} 小時")
            print(f"當前學習率: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # 如果有 warmup，需要調整 warmup scheduler 的狀態
            if has_warmup and start_epoch < warmup_epochs:
                self.warmup_scheduler.current_epoch = start_epoch
                print(f"🔥 恢復 warmup 狀態: epoch {start_epoch}/{warmup_epochs}")
        
        # 早停機制
        early_stopping_counter = 0
        
        try:
            for epoch in range(start_epoch, num_epochs):
                epoch_start_time = time.time()
                
                # ==================== 關鍵修改：在每個 epoch 開始時調用 warmup scheduler ====================
                if has_warmup:
                    if epoch < warmup_epochs:
                        # Warmup 階段：調用 warmup scheduler
                        self.warmup_scheduler.step(epoch)
                        warmup_stage = True
                        
                        # 獲取當前的學習率和動量用於顯示
                        current_lrs = self.warmup_scheduler.get_lr()
                        current_momenta = self.warmup_scheduler.get_momentum()
                        
                        if self.use_progress_bar:
                            tqdm.write(f"🔥 Warmup 階段 [{epoch+1}/{warmup_epochs}] - LR: {current_lrs[0]:.6f}, Momentum: {current_momenta[0]:.3f}")
                        else:
                            print(f"🔥 Warmup 階段 [{epoch+1}/{warmup_epochs}] - LR: {current_lrs[0]:.6f}, Momentum: {current_momenta[0]:.3f}")
                    else:
                        warmup_stage = False
                else:
                    warmup_stage = False
                
                # 訓練一個epoch
                train_loss, train_dice = self.train_one_epoch(epoch)
                val_loss, val_dice = self.validate_one_epoch(epoch)
                
                # 記錄歷史
                self.history['train_loss'].append(train_loss)
                self.history['val_loss'].append(val_loss)
                self.history['train_dice'].append(train_dice)
                self.history['val_dice'].append(val_dice)
                
                # 記錄學習率（在 warmup 或主調度器之後）
                current_lr = self.optimizer.param_groups[0]['lr']
                self.history['learning_rate'].append(current_lr)
                
                epoch_time = time.time() - epoch_start_time
                self.total_training_time += epoch_time
                
                # ==================== 學習率調度 ====================
                if not warmup_stage and self.scheduler:
                    # 只有在非 warmup 階段才調用主調度器
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        #self.scheduler.step(val_dice)
                        self.scheduler.step(val_loss) #改為以val_loss當標準(早停)，配合ReduceLROnPlateau的mode='min'(train.py)
                    else:
                        self.scheduler.step()
                
                # 檢查是否是最佳模型（修復：分別獨立判斷）
                is_dice_best = val_dice > self.best_val_dice
                is_loss_best = val_loss < self.best_val_loss
                
                # 更新最佳 Dice
                if is_dice_best:
                    self.best_val_dice = val_dice
                    early_stopping_counter = 0
                
                # 更新最佳 Loss（獨立判斷，不使用 elif）
                if is_loss_best:
                    self.best_val_loss = val_loss
                    early_stopping_counter = 0
                
                # 如果兩者都沒改進，才增加計數器
                if not is_dice_best and not is_loss_best:
                    early_stopping_counter += 1
                
                # 日誌輸出 - 增強版：顯示當前階段和動量資訊
                if (epoch + 1) % self.log_interval == 0 or is_dice_best or is_loss_best:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    
                    # 獲取動量資訊
                    momentum_info = ""
                    if 'momentum' in self.optimizer.param_groups[0]:
                        momentum = self.optimizer.param_groups[0]['momentum']
                        momentum_info = f", Momentum: {momentum:.3f}"
                    elif 'betas' in self.optimizer.param_groups[0]:
                        beta1 = self.optimizer.param_groups[0]['betas'][0]
                        momentum_info = f", Beta1: {beta1:.3f}"
                    
                    # 階段標記
                    stage_info = "🔥 [WARMUP]" if warmup_stage else "🚀 [NORMAL]"
                    
                    log_msg = (f"{stage_info} Epoch [{epoch+1}/{num_epochs}] "
                            f"Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f} | "
                            f"Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f} | "
                            f"LR: {current_lr:.6f}{momentum_info} | Time: {epoch_time:.1f}s")
                    
                    if is_dice_best:
                        log_msg += " ⭐ [NEW BEST DICE]"
                        
                    if is_loss_best:
                        log_msg += " 🌸 [NEW BEST LOSS]"

                    if self.use_progress_bar:
                        tqdm.write(log_msg)
                    else:
                        print(log_msg)

                # 保存檢查點
                self.save_checkpoint(epoch, is_dice_best=is_dice_best, is_loss_best=is_loss_best)
                
                # 視覺化 - 包含預測結果
                if self.visualize and (epoch + 1) % self.plot_interval == 0:
                    try:
                        # 繪製訓練曲線
                        self.visualizer.plot_training_curves(self.history, title=f"Training curve (up to Epoch {epoch+1})" , save_name=f"training_curves_epoch_{epoch+1:03d}.png")
                        # 獲取並視覺化預測結果
                        sample_predictions = self.get_sample_predictions()
                        if sample_predictions is not None:
                            images, masks, predictions = sample_predictions
                            if hasattr(self.visualizer, 'plot_3d_predictions'):
                                self.visualizer.plot_3d_predictions(
                                    images, masks, predictions,
                                    save_name=f"predictions_epoch_{epoch+1:03d}.png",
                                    max_samples=3
                                )
                                if self.use_progress_bar:
                                    tqdm.write(f"已生成第 {epoch+1} epoch 的預測結果圖")
                                else:
                                    print(f"已生成第 {epoch+1} epoch 的預測結果圖")
                    except Exception as e:
                        error_msg = f"視覺化過程出現錯誤: {e}"
                        if self.use_progress_bar:
                            tqdm.write(error_msg)
                        else:
                            print(error_msg)
                
                # 早停檢查
                if early_stopping_patience and early_stopping_counter >= early_stopping_patience:
                    print(f"早停觸發！已連續 {early_stopping_patience} epochs 無改善")
                    break
            
            # 記錄訓練結束時間
            self.training_end_time = time.time()
            
            print(f"訓練完成！")
            print(f"總訓練時間: {self.total_training_time/3600:.2f} 小時")
            print(f"最佳驗證 Dice: {self.best_val_dice:.4f}")
            
            # 最終視覺化
            if self.visualize:
                try:
                    # 根據visualizer的實際方法來調用
                    if hasattr(self.visualizer, 'create_final_dashboard'):
                        self.visualizer.create_final_dashboard(self.history)
                    elif hasattr(self.visualizer, 'create_training_dashboard'):
                        self.visualizer.create_training_dashboard(self.history, "final_training_dashboard.png")
                    else:
                        # 退回到基本的訓練曲線繪製
                        self.visualizer.plot_training_curves(self.history, len(self.history['train_loss']))
                except Exception as e:
                    print(f"最終視覺化生成失敗: {e}")
                    print("跳過視覺化步驟")
        
        except KeyboardInterrupt:
            print("訓練被中斷")
            self.training_end_time = time.time()
        
        return self.history

    
    def test(self, checkpoint_path=None, save_results=True):
        """測試模型並生成視覺化結果，同時保存測試報告"""
        if 'test' not in self.data_loaders:
            print("沒有測試資料集")
            return None
        
        if checkpoint_path:
            try:
                checkpoint = self.safe_torch_load(checkpoint_path)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print(f"載入測試模型: {checkpoint_path}")
            except Exception as e:
                print(f"載入測試模型失敗: {e}")
                return None
        
        self.model.eval()
        test_dice_scores = []
        test_losses = []
        all_predictions = []
        test_details = []  # 記錄詳細資訊
        
        num_batches = len(self.data_loaders['test'])
        
        # 創建測試進度條
        if self.use_progress_bar:
            test_pbar = tqdm(
                enumerate(self.data_loaders['test']),
                total=num_batches,
                desc="測試模型",
                ncols=100,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}'
            )
        else:
            print("開始測試...")
        
        data_iter = test_pbar if self.use_progress_bar else enumerate(self.data_loaders['test'])
        
        with torch.no_grad():
            for batch_idx, batch in data_iter:
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                # masks[masks > 0] = 1
                
                outputs = self.model(images)

                # 計算批次損失（用於進度顯示）
                batch_loss = self.criterion(outputs, masks)
                test_losses.append(batch_loss.item())

                # 計算批次Dice分數（用於進度顯示）
                batch_metrics = calculate_metrics(outputs, masks, self.model.n_classes)
                batch_dice_score = batch_metrics['mean_dice']
                test_dice_scores.append(batch_dice_score)

                # 為每個樣本計算個別的損失和Dice分數
                batch_size = images.size(0)
                individual_losses = []
                individual_dice_scores = []

                for sample_idx in range(batch_size):
                    # 提取單個樣本
                    single_image = images[sample_idx:sample_idx+1]
                    single_mask = masks[sample_idx:sample_idx+1]
                    single_output = outputs[sample_idx:sample_idx+1]
                    
                    # 計算單個樣本的損失
                    single_loss = self.criterion(single_output, single_mask)
                    individual_losses.append(single_loss.item())
                    
                    # 計算單個樣本的Dice分數
                    single_metrics = calculate_metrics(single_output, single_mask, self.model.n_classes)
                    single_dice = single_metrics['mean_dice']
                    individual_dice_scores.append(single_dice)

                # 記錄詳細資訊（每個檔案）
                if 'image_path' in batch:
                    for i, img_path in enumerate(batch['image_path']):
                        if i < len(individual_losses):
                            test_details.append({
                                'file': Path(img_path).name,
                                'loss': individual_losses[i],
                                'dice': individual_dice_scores[i],
                                'dice_per_class': single_metrics['dice_per_class'],
                                'batch_idx': batch_idx,
                                'sample_idx': i
                            })
                else:
                    # 如果沒有檔案路徑，記錄每個樣本的資訊
                    for i in range(len(individual_losses)):
                        test_details.append({
                            'file': f'batch_{batch_idx:03d}_sample_{i:02d}',
                            'loss': individual_losses[i],
                            'dice': individual_dice_scores[i],
                            'dice_per_class': single_metrics['dice_per_class'],
                            'batch_idx': batch_idx,
                            'sample_idx': i
                        })
                
                # 收集預測結果用於視覺化
                if batch_idx < 3 and self.visualize:
                    all_predictions.append((images.cpu(), masks.cpu(), outputs.cpu()))
                
                # 更新進度條
                if self.use_progress_bar:
                    test_pbar.set_postfix({
                        'Loss': f'{batch_loss.item():.4f}',
                        'Dice': f'{batch_dice_score:.4f}'
                    })
                elif batch_idx % 5 == 0:
                    print(f"測試進度: {batch_idx+1}/{num_batches}")
        
        # 關閉測試進度條
        if self.use_progress_bar and 'test_pbar' in locals():
            test_pbar.close()
        
        # ============== 重點修復：基於個別檔案計算真實統計 ==============
        if test_details:
            # 使用個別檔案的真實數據
            individual_losses = [detail['loss'] for detail in test_details]
            individual_dice_scores = [detail['dice'] for detail in test_details]
            
            avg_test_loss = np.mean(individual_losses)
            avg_test_dice = np.mean(individual_dice_scores)
            
            # 計算統計指標
            loss_std = np.std(individual_losses)
            dice_std = np.std(individual_dice_scores)
            min_loss = np.min(individual_losses)
            max_loss = np.max(individual_losses)
            min_dice = np.min(individual_dice_scores)
            max_dice = np.max(individual_dice_scores)
            
            # 用於返回和保存的數據
            final_losses = individual_losses
            final_dice_scores = individual_dice_scores
            sample_count = len(test_details)
            unit_description = "個檔案"
        else:
            # 降級到批次數據
            avg_test_loss = np.mean(test_losses)
            avg_test_dice = np.mean(test_dice_scores)
            
            loss_std = np.std(test_losses)
            dice_std = np.std(test_dice_scores)
            min_loss = np.min(test_losses)
            max_loss = np.max(test_losses)
            min_dice = np.min(test_dice_scores)
            max_dice = np.max(test_dice_scores)
            
            final_losses = test_losses
            final_dice_scores = test_dice_scores
            sample_count = len(test_losses)
            unit_description = "個批次"
        
        result_msg = f"""
        測試結果:
        測試樣本數: {sample_count} {unit_description}
        平均損失: {avg_test_loss:.4f}
        平均 Dice 分數: {avg_test_dice:.4f} ({avg_test_dice * 100:.2f}%)
        損失範圍: {min_loss:.4f} ~ {max_loss:.4f} (標準差: {loss_std:.4f})
        Dice範圍: {min_dice:.4f} ~ {max_dice:.4f} (標準差: {dice_std:.4f})"""
        
        if self.use_progress_bar:
            tqdm.write(result_msg)
        else:
            print(result_msg)
        
        # 保存測試結果到文件
        if save_results:
            self._save_test_results(
                avg_test_loss, avg_test_dice,
                final_losses, final_dice_scores,
                test_details, checkpoint_path,
                # 傳遞統計指標
                loss_std, dice_std, min_loss, max_loss, min_dice, max_dice,
                sample_count, unit_description
            )
        
        # 生成測試視覺化
        if self.visualize and all_predictions:
            viz_msg = "生成測試結果視覺化..."
            if self.use_progress_bar:
                tqdm.write(viz_msg)
            else:
                print(viz_msg)
            
            try:
                for i, (images, masks, predictions) in enumerate(all_predictions):
                    if hasattr(self.visualizer, 'plot_3d_predictions'):
                        self.visualizer.plot_3d_predictions(
                            images, masks, predictions,
                            save_name=f"test_results_batch_{i+1}.png"
                        )
                    else:
                        print("視覺化器沒有 plot_3d_predictions 方法，跳過測試結果視覺化")
                        break
            except Exception as e:
                error_msg = f"測試結果視覺化失敗: {e}"
                if self.use_progress_bar:
                    tqdm.write(error_msg)
                else:
                    print(error_msg)
        
        return {
            'avg_loss': avg_test_loss,
            'avg_dice': avg_test_dice,
            'all_losses': final_losses,
            'all_dice_scores': final_dice_scores,
            'details': test_details,
            'stats': {
                'loss_std': loss_std,
                'dice_std': dice_std,
                'min_loss': min_loss,
                'max_loss': max_loss,
                'min_dice': min_dice,
                'max_dice': max_dice,
                'sample_count': sample_count,
                'unit': unit_description
            }
        }

    def _save_test_results(self, avg_loss, avg_dice, all_losses, all_dice, details, model_path,
                        loss_std, dice_std, min_loss, max_loss, min_dice, max_dice,
                        sample_count, unit_description):
        """增強版測試結果保存 - 使用實際的訓練配置"""
        import datetime
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 從檢查點載入額外資訊
        training_info = {}
        model_config = {}
        model_complexity = {}
        full_training_config = {}  # 新增：完整訓練配置
        
        if model_path and Path(model_path).exists():
            try:
                checkpoint = self.safe_torch_load(model_path)
                training_info = {
                    'total_epochs': checkpoint.get('epoch', 'Unknown') + 1,
                    'best_val_dice': checkpoint.get('best_val_dice', 'Unknown'),
                    'total_training_time': checkpoint.get('total_training_time', 0),
                    'final_train_loss': checkpoint.get('train_losses', [])[-1] if checkpoint.get('train_losses') else 'Unknown',
                    'final_val_loss': checkpoint.get('val_losses', [])[-1] if checkpoint.get('val_losses') else 'Unknown'
                }
                # 優先使用檢查點中的實際配置
                model_config = checkpoint.get('model_config', {})
                full_training_config = checkpoint.get('training_config', {})
                model_complexity = checkpoint.get('model_info', self.model_info)
            except:
                pass
        
        # 如果檢查點中沒有配置，使用當前 trainer 的配置
        if not model_config and self.training_config:
            model_config = {
                'n_channels': self.training_config.get('n_channels', 1),
                'n_classes': self.training_config.get('n_classes', 2),
                'base_channels': self.training_config.get('base_channels', 32),
                'num_groups': self.training_config.get('num_groups', 8),
                'bilinear': self.training_config.get('bilinear', False)
            }
        # 新增：從檢查點或當前歷史獲取 GPU 記憶體資訊
        gpu_memory_stats = {}
        if model_path and Path(model_path).exists():
            try:
                checkpoint = self.safe_torch_load(model_path)
                gpu_memory_history = checkpoint.get('gpu_memory', self.history.get('gpu_memory', []))
            except:
                gpu_memory_history = self.history.get('gpu_memory', [])
        else:
            gpu_memory_history = self.history.get('gpu_memory', [])
        
        # 計算 GPU 記憶體統計
        if gpu_memory_history and len(gpu_memory_history) > 0:
            allocated_list = [m['allocated'] for m in gpu_memory_history if 'allocated' in m]
            cached_list = [m['cached'] for m in gpu_memory_history if 'cached' in m]
            
            if allocated_list:
                gpu_memory_stats = {
                    'avg_allocated': np.mean(allocated_list),
                    'max_allocated': np.max(allocated_list),
                    'min_allocated': np.min(allocated_list),
                    'avg_cached': np.mean(cached_list),
                    'max_cached': np.max(cached_list),
                    'min_cached': np.min(cached_list),
                    'final_allocated': allocated_list[-1],
                    'final_cached': cached_list[-1]
                }
        
        # 保存到txt文件
        txt_file = self.save_dir / f"test_results_{timestamp}.txt"
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("3D UNet 模型完整測試報告\n")
            f.write("=" * 70 + "\n")
            f.write(f"測試時間: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"模型路徑: {model_path or '當前訓練模型'}\n")
            f.write("\n")
            
            # 模型基本資訊（使用實際配置）
            f.write("=" * 25 + " 模型基本資訊 " + "=" * 25 + "\n")
            f.write(f"輸入通道數 (n_channels): {model_config.get('n_channels', 'Unknown')}\n")
            f.write(f"輸出類別數 (n_classes): {model_config.get('n_classes', 'Unknown')}\n")
            f.write(f"基礎通道數 (base_channels): {model_config.get('base_channels', 'Unknown')}\n")
            f.write(f"GroupNorm 組數 (num_groups): {model_config.get('num_groups', 'Unknown')}\n")
            f.write(f"雙線性上採樣 (bilinear): {model_config.get('bilinear', 'Unknown')}\n")
            f.write("\n")
               
            # 如果有完整的訓練配置，額外顯示關鍵訓練參數
            if full_training_config:
                f.write("=" * 24 + " 訓練配置參數 " + "=" * 24 + "\n")
                f.write(f"批次大小 (batch_size): {full_training_config.get('batch_size', 'Unknown')}\n")
                f.write(f"學習率 (learning_rate): {full_training_config.get('learning_rate', 'Unknown')}\n")
                f.write(f"優化器 (optimizer): {full_training_config.get('optimizer', 'Unknown')}\n")
                f.write(f"損失函數 (loss_type): {full_training_config.get('loss_type', 'Unknown')}\n")
                f.write(f"數據增強 (use_augmentation): {full_training_config.get('use_augmentation', 'Unknown')}\n")
                if full_training_config.get('use_augmentation'):
                    f.write(f"增強類型 (augmentation_type): {full_training_config.get('augmentation_type', 'Unknown')}\n")
                f.write(f"影像尺寸 (target_size): {full_training_config.get('target_size', 'Unknown')}\n")
                f.write("\n")
            
            # 模型複雜度資訊
            f.write("=" * 25 + " 模型複雜度資訊 " + "=" * 24 + "\n")
            f.write(f"總參數量: {model_complexity.get('total_params', 'Unknown'):,}\n")
            f.write(f"可訓練參數: {model_complexity.get('trainable_params', 'Unknown'):,}\n")
            f.write(f"模型大小: {model_complexity.get('model_size_mb', 0):.2f} MB (FP32)\n")
            f.write(f"GLOPs: {model_complexity.get('flops', 'Unknown')}\n")
            f.write(f"平均推理時間: {model_complexity.get('avg_inference_time', 0)*1000:.2f} ms\n")
            f.write("\n")
            
            # 新增：GPU 記憶體使用統計區塊（在模型複雜度資訊之後）
            if gpu_memory_stats:
                f.write("=" * 24 + " GPU 記憶體使用統計 " + "=" * 23 + "\n")
                f.write(f"平均已分配記憶體: {gpu_memory_stats['avg_allocated']:.2f} GB\n")
                f.write(f"峰值已分配記憶體: {gpu_memory_stats['max_allocated']:.2f} GB\n")
                f.write(f"最小已分配記憶體: {gpu_memory_stats['min_allocated']:.2f} GB\n")
                f.write(f"平均已緩存記憶體: {gpu_memory_stats['avg_cached']:.2f} GB\n")
                f.write(f"峰值已緩存記憶體: {gpu_memory_stats['max_cached']:.2f} GB\n")
                f.write(f"最小已緩存記憶體: {gpu_memory_stats['min_cached']:.2f} GB\n")
                f.write(f"最終已分配記憶體: {gpu_memory_stats['final_allocated']:.2f} GB\n")
                f.write(f"最終已緩存記憶體: {gpu_memory_stats['final_cached']:.2f} GB\n")
                f.write("\n")
            
            # 訓練資訊
            f.write("=" * 26 + " 訓練資訊 " + "=" * 26 + "\n")
            if training_info:
                f.write(f"訓練輪數: {training_info.get('total_epochs', 'Unknown')} epochs\n")
                total_time = training_info.get('total_training_time', 0)
                if total_time > 0:
                    hours = int(total_time // 3600)
                    minutes = int((total_time % 3600) // 60)
                    seconds = int(total_time % 60)
                    f.write(f"總訓練時間: {hours:02d}:{minutes:02d}:{seconds:02d} ({total_time/3600:.2f} 小時)\n")
                    f.write(f"平均每epoch時間: {total_time/training_info.get('total_epochs', 1):.1f} 秒\n")
                f.write(f"訓練集最終損失: {training_info.get('final_train_loss', 'Unknown')}\n")
                f.write(f"驗證集最終損失: {training_info.get('final_val_loss', 'Unknown')}\n")
                f.write(f"最佳驗證Dice: {training_info.get('best_val_dice', 'Unknown')}\n")
            else:
                f.write("無法載入訓練資訊\n")
            f.write("\n")
            
            # 測試結果
            f.write("=" * 27 + " 測試結果 " + "=" * 27 + "\n")
            f.write(f"測試樣本數: {sample_count} {unit_description}\n")
            f.write(f"平均測試損失: {avg_loss:.6f}\n")
            f.write(f"平均測試Dice (所有前景): {avg_dice:.6f} ({avg_dice * 100:.2f}%)\n")
            f.write("\n")

            # 新增：每個標籤的個別 Dice
            if details:
                num_fg = len(details[0].get('dice_per_class', [avg_dice]))
                if num_fg > 1:
                    f.write("=" * 24 + " 各標籤 Dice 分數 " + "=" * 24 + "\n")
                    for cls_idx in range(num_fg):
                        cls_dices = [
                            d['dice_per_class'][cls_idx]
                            for d in details
                            if 'dice_per_class' in d and cls_idx < len(d['dice_per_class'])
                        ]
                        if cls_dices:
                            f.write(
                                f"  標籤 {cls_idx + 1}: "
                                f"平均 {np.mean(cls_dices):.6f}  "
                                f"最小 {np.min(cls_dices):.6f}  "
                                f"最大 {np.max(cls_dices):.6f}  "
                                f"標準差 {np.std(cls_dices):.6f}\n"
                            )
                    f.write("\n")
            
            # 統計分析
            f.write("=" * 27 + " 統計分析 " + "=" * 27 + "\n")
            f.write(f"損失 - 標準差: {loss_std:.6f}\n")
            f.write(f"損失 - 最小值: {min_loss:.6f}\n")
            f.write(f"損失 - 最大值: {max_loss:.6f}\n")
            f.write(f"Dice - 標準差: {dice_std:.6f}\n")
            f.write(f"Dice - 最小值: {min_dice:.6f}\n")
            f.write(f"Dice - 最大值: {max_dice:.6f}\n")
            f.write("\n")
            
            # 性能評估
            f.write("=" * 27 + " 性能評估 " + "=" * 27 + "\n")
            if avg_dice >= 0.90:
                performance = "優秀"
                evaluation = "模型表現出色，可以用於實際應用"
            elif avg_dice >= 0.80:
                performance = "良好" 
                evaluation = "模型表現良好，可考慮進一步優化"
            elif avg_dice >= 0.75:
                performance = "中等"
                evaluation = "模型表現中等，建議調整超參數或增加數據"
            else:
                performance = "需要改進"
                evaluation = "模型表現不佳，建議檢查數據質量和模型架構"
                
            f.write(f"整體評級: {performance} (Dice: {avg_dice:.4f})\n")
            f.write(f"評估建議: {evaluation}\n")
            f.write("\n")
            
            # 詳細結果（如果不太多）
            if details and len(details) <= 30:
                f.write("=" * 25 + " 詳細測試結果 " + "=" * 25 + "\n")
                f.write(f"{'檔案/批次':<35} {'損失':<12} {'Dice分數':<12}\n")
                f.write("-" * 65 + "\n")
                for detail in details:
                    f.write(f"{detail['file']:<35} {detail['loss']:<12.6f} {detail['dice']:<12.6f}\n")
            
            f.write("\n" + "=" * 70 + "\n")
            f.write("報告結束\n")
        
        print(f"完整測試報告已保存到: {txt_file}")
        
        # JSON格式保存（包含所有數據）
        json_file = self.save_dir / f"test_results_{timestamp}.json"
        results_data = {
            'timestamp': datetime.datetime.now().isoformat(),
            'model_path': str(model_path) if model_path else 'current_model',
            'model_config': model_config,
            'training_config': full_training_config,  # 新增：完整訓練配置
            'model_complexity': model_complexity,
            'training_info': training_info,
            'gpu_memory_stats': gpu_memory_stats,  # 新增：GPU記憶體統計
            'gpu_memory_history': gpu_memory_history,  # 新增：完整GPU記憶體歷史
            'test_summary': {
                'avg_loss': float(avg_loss),
                'avg_dice': float(avg_dice),
                'sample_count': sample_count,
                'unit': unit_description,
                'loss_std': float(loss_std),
                'dice_std': float(dice_std),
                'best_dice': float(max_dice),
                'worst_dice': float(min_dice),
                'min_loss': float(min_loss),
                'max_loss': float(max_loss)
            },
            'all_losses': [float(x) for x in all_losses],
            'all_dice_scores': [float(x) for x in all_dice],
            'details': details
        }
        
        import json
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, ensure_ascii=False, indent=2)
        
        print(f"完整JSON數據已保存到: {json_file}")
