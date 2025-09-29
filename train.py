#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3D UNet 簡化訓練腳本 - 含 Warmup 和進階優化器設定
只需要調整參數，所有訓練函數都在 trainer 中
"""

import torch
import sys
import argparse
from pathlib import Path
import random
import numpy as np
import math

from src.network_architecture.net_module import *  # UNet3D 模型定義
from trainer import *  # 訓練工具和 trainer
from src.network_architecture.unet3d import UNet3D
from src.data_processing_and_data_enhancement.dataload import MedicalImageDataset, create_data_loaders

"""
🚀 Enhanced 3D UNet 訓練腳本 - 整合視覺化功能與 Warmup

新增進階功能：
✅ Warmup 學習率調度 - 漸進式學習率提升
✅ Momentum Warmup - 動量參數漸進調整
✅ Bias 參數特殊處理 - 不同的學習率設定
✅ 參數分組優化 - 權重和偏置分別處理
✅ 多種學習率調度器 (Step, ReduceLROnPlateau, Cosine)
✅ 進階優化器支援 (Adam, AdamW, SGD with momentum)

Warmup 機制說明：
🔥 前 N 個 epoch 使用較低的學習率和動量，逐漸提升到目標值
🌡️  有助於穩定訓練初期，避免梯度爆炸
⚖️  Bias 參數使用不同的 warmup 策略

參數建議範圍：
📈 momentum: 0.3-0.98 (SGD), 0.6-0.98 (Adam beta1)
🔥 warmup_epochs: 1-5 epochs (可以是小數，如 2.5)
🌡️  warmup_momentum: 0.0-0.95 (初始動量)
⚖️  warmup_bias_lr: 0.0-0.2 (bias 學習率乘數)

使用說明:
1. 基本使用（含 warmup）：
   python train.py

2. 自訂 warmup 參數：
   python train.py --warmup_epochs 3.0 --warmup_momentum 0.1 --momentum 0.9

3. 停用 warmup：
   python train.py --warmup_epochs 0

4. 使用不同調度器：
   python train.py --scheduler cosine

5. 完整進階設定：
   python train.py --epochs 100 --lr 1e-3 --momentum 0.95 \
                   --warmup_epochs 2.5 --warmup_momentum 0.1 \
                   --warmup_bias_lr 0.15 --scheduler cosine

新增輸出內容：
📈 Warmup 學習率和動量變化曲線
📊 不同參數組的學習率追蹤
🔥 Warmup 階段的詳細日誌

訓練階段說明：
1️⃣ Warmup 階段 (0 - warmup_epochs)：
   - 學習率從 0 線性增加到目標值
   - Momentum 從 warmup_momentum 增加到目標值
   - Bias 參數使用特殊的學習率設定

2️⃣ 正常訓練階段 (warmup_epochs - 總epochs)：
   - 使用目標學習率和動量
   - 根據設定的調度器調整學習率

效果：
✅ 更穩定的訓練開始
✅ 更好的收斂性能
✅ 減少訓練初期的不穩定性
✅ 支援更大的學習率設定

注意事項：
⚠️  Warmup 會稍微延長訓練時間
⚠️  需要根據資料集大小調整 warmup_epochs
⚠️  某些小資料集可能不需要 warmup
⚠️  SGD 和 Adam 的最佳 momentum 設定不同
"""


# ==================== Warmup 學習率調度器 ====================
class WarmupLRScheduler:
    """
    Warmup 學習率調度器
    支援 momentum 和 bias lr 的 warmup
    """
    def __init__(self, optimizer, warmup_epochs, warmup_momentum, warmup_bias_lr, base_lr, base_momentum):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.warmup_momentum = warmup_momentum
        self.warmup_bias_lr = warmup_bias_lr
        self.base_lr = base_lr
        self.base_momentum = base_momentum
        self.current_epoch = 0
        
        # 保存原始參數組設定
        self.param_groups_info = []
        for i, param_group in enumerate(optimizer.param_groups):
            info = {
                'base_lr': param_group.get('lr', base_lr),
                'base_momentum': param_group.get('momentum', base_momentum),
                'is_bias': 'bias' in str(param_group.get('name', '')).lower()
            }
            self.param_groups_info.append(info)
    
    def step(self, epoch=None):
        """更新學習率和 momentum"""
        if epoch is not None:
            self.current_epoch = epoch
        else:
            self.current_epoch += 1
        
        if self.current_epoch < self.warmup_epochs:
            # Warmup 階段 - 修復：從第1步開始，而不是從0開始
            warmup_ratio = (self.current_epoch + 1) / self.warmup_epochs
            
            for i, param_group in enumerate(self.optimizer.param_groups):
                info = self.param_groups_info[i]
                
                # 學習率 warmup（bias 和一般參數不同處理）
                if info['is_bias']:
                    # bias 參數使用特殊的 warmup
                    target_lr = self.warmup_bias_lr * info['base_lr']
                    param_group['lr'] = target_lr * warmup_ratio
                else:
                    # 一般參數的線性 warmup
                    param_group['lr'] = info['base_lr'] * warmup_ratio
                
                # Momentum warmup（如果優化器支援）
                if 'momentum' in param_group:
                    param_group['momentum'] = self.warmup_momentum + (info['base_momentum'] - self.warmup_momentum) * warmup_ratio
                elif 'betas' in param_group:  # Adam 系列
                    # 對於 Adam，調整 beta1（相當於 momentum）
                    original_betas = param_group['betas']
                    new_beta1 = self.warmup_momentum + (info['base_momentum'] - self.warmup_momentum) * warmup_ratio
                    param_group['betas'] = (new_beta1, original_betas[1])
        
        else:
            # Warmup 完成，恢復正常設定
            for i, param_group in enumerate(self.optimizer.param_groups):
                info = self.param_groups_info[i]
                param_group['lr'] = info['base_lr']
                
                if 'momentum' in param_group:
                    param_group['momentum'] = info['base_momentum']
                elif 'betas' in param_group:
                    param_group['betas'] = (info['base_momentum'], param_group['betas'][1])
    
    def get_lr(self):
        """取得當前學習率"""
        return [param_group['lr'] for param_group in self.optimizer.param_groups]
    
    def get_momentum(self):
        """取得當前 momentum"""
        momenta = []
        for param_group in self.optimizer.param_groups:
            if 'momentum' in param_group:
                momenta.append(param_group['momentum'])
            elif 'betas' in param_group:
                momenta.append(param_group['betas'][0])
            else:
                momenta.append(0.0)
        return momenta
    
    def get_lr(self):
        """取得當前學習率"""
        return [param_group['lr'] for param_group in self.optimizer.param_groups]
    
    def get_momentum(self):
        """取得當前 momentum"""
        momenta = []
        for param_group in self.optimizer.param_groups:
            if 'momentum' in param_group:
                momenta.append(param_group['momentum'])
            elif 'betas' in param_group:
                momenta.append(param_group['betas'][0])
            else:
                momenta.append(0.0)
        return momenta


# ==================== 訓練參數配置 ====================
def get_config():
    """
    🔧 在這裡調整所有訓練參數
    """
    config = {
        # === 資料相關參數 ===
        'data_root': r"D:\unet3d_chinese\dataset",  # 🔧 修改為你的資料根目錄
        'target_size': (64, 64, 64),     # 🔧 目標影像尺寸 (D, H, W)
        'batch_size': 8,                 # 🔧 批次大小（根據顯卡記憶體調整）
        'num_workers': 4,                # 🔧 資料載入執行緒數
        'use_augmentation': False,        # 🔧 是否啟用數據增強（只對訓練集）
        'augmentation_type': 'custom',    # 🔧 數據增強類型 ('light', 'medium', 'heavy', 'medical', 'medical_heavy', 'custom')
        
        # === 模型相關參數 ===
        'n_channels': 1,                 # 🔧 輸入通道數（灰階影像為1，RGB為3）
        'n_classes': 2,                  # 🔧 輸出類別數（二分類為2，多分類根據需求）
        'base_channels': 64,             # 🔧 基礎通道數（可以是16, 32, 64）
        'num_groups': 8,                 # 🔧 GroupNorm 組數
        'bilinear': False,               # 🔧 是否使用雙線性上採樣
        
        # === 訓練相關參數 ===
        'num_epochs': 300,               # 🔧 訓練 epoch 數
        'learning_rate': 1e-3,           # 🔧 學習率 (SGD=1E-2, Adam=1E-3)
        'weight_decay': 5e-4,            # 🔧 權重衰減
        'optimizer': 'adam',             # 🔧 優化器 ('adam', 'adamw', 'sgd')
        
        # === 進階優化器參數 ===
        'momentum': 0.937,               # 🔧 SGD momentum 或 Adam beta1 (0.3, 0.6, 0.98)
        'warmup_epochs': 3.0,            # 🔧 warmup epochs（支援小數）
        'warmup_momentum': 0.8,          # 🔧 warmup 初始 momentum (0.0, 0.95)
        'warmup_bias_lr': 0.1,           # 🔧 warmup 初始 bias lr 乘數 (0.0, 0.2)
        
        # === 視覺化相關參數 ===
        'enable_visualization': True,    # 🔧 是否啟用視覺化
        'plot_interval': 20,             # 🔧 視覺化更新間隔（每幾個epoch）
        'early_stopping_patience': 31,   # 🔧 早停耐心值
        
        # === 損失函數參數 ===
        'loss_type': 'combined',         # 🔧 損失函數類型 ('dice', 'ce', 'combined')
        'ce_weight': 0.4,                # 🔧 Cross Entropy 權重
        'dice_weight': 0.6,              # 🔧 Dice Loss 權重
        
        # === 學習率調度器 ===
        'scheduler': 'reduce_on_plateau', # 🔧 調度器類型 ('step', 'reduce_on_plateau', 'cosine', None)
        'scheduler_patience': 10,         # 🔧 ReduceLROnPlateau 耐心值
        'scheduler_factor': 0.5,          # 🔧 學習率衰減因子
        'step_size': 10,                  # 🔧 StepLR 步長
        'cosine_t_max': None,             # 🔧 Cosine 調度器最大週期（None則使用總epoch數）
        
        # === 保存和日誌 ===
        'save_dir': r"D:\unet3d_chinese\train_end",     # 🔧 模型保存目錄
        'log_interval': 1,               # 🔧 日誌輸出間隔
        'save_interval': 200,            # 🔧 模型保存間隔
        'resume_from': None,             # 🔧 從檢查點恢復訓練（路徑或None）
        
        # === 其他設定 ===
        'seed': 42,                      # 🔧 隨機種子
        'device': 'auto',                # 🔧 設備選擇 ('auto', 'cpu', 'cuda:0')
        'run_test': True,                # 🔧 訓練完成後是否執行測試
    }
    return config

def set_seed(seed):
    """設置隨機種子以確保可重現性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def validate_config(config):
    """驗證配置參數"""
    print("🔍 驗證配置參數...")
    
    data_root = Path(config['data_root'])
    if not data_root.exists():
        raise FileNotFoundError(f"❌ 資料目錄不存在: {data_root}")
    
    train_images = data_root / 'train' / 'images'
    train_labels = data_root / 'train' / 'labels'
    
    if not train_images.exists():
        raise FileNotFoundError(f"❌ 訓練影像目錄不存在: {train_images}")
    if not train_labels.exists():
        raise FileNotFoundError(f"❌ 訓練標籤目錄不存在: {train_labels}")
    
    print("✅ 基本配置驗證通過")
    
    for split in ['val', 'test']:
        images_dir = data_root / split / 'images'
        labels_dir = data_root / split / 'labels'
        if images_dir.exists() and labels_dir.exists():
            print(f"✅ 找到 {split} 資料集")
        else:
            print(f"⚠️  {split} 資料集不完整或不存在")
    
    # 驗證新增的參數
    if not (0.0 <= config['momentum'] <= 1.0):
        print(f"⚠️  警告: momentum 值 {config['momentum']} 超出建議範圍 [0.0, 1.0]")
    
    if config['warmup_epochs'] < 0:
        print(f"❌ 錯誤: warmup_epochs 不能為負數: {config['warmup_epochs']}")
        raise ValueError("warmup_epochs 必須 >= 0")
    
    if not (0.0 <= config['warmup_momentum'] <= 1.0):
        print(f"⚠️  警告: warmup_momentum 值 {config['warmup_momentum']} 超出建議範圍 [0.0, 1.0]")
    
    if not (0.0 <= config['warmup_bias_lr'] <= 1.0):
        print(f"⚠️  警告: warmup_bias_lr 值 {config['warmup_bias_lr']} 超出建議範圍 [0.0, 1.0]")
    
    print("🔍 執行資料完整性檢查...")
    check_data_integrity(config)

def check_data_integrity(config):
    """檢查資料完整性和標籤值範圍"""
    try:
        test_dataset = MedicalImageDataset(
            data_root=config['data_root'],
            split='train',
            target_size=config['target_size'],
            use_augmentation=config['use_augmentation'],
            augmentation_type=config['augmentation_type']
        )
        
        if len(test_dataset) > 0:
            sample = test_dataset[0]
            image = sample['image']
            mask = sample['mask']
            
            print(f"📊 影像形狀: {image.shape}")
            print(f"📊 標籤形狀: {mask.shape}")
            print(f"📊 影像值範圍: [{image.min().item():.3f}, {image.max().item():.3f}]")
            print(f"📊 標籤值範圍: [{mask.min().item()}, {mask.max().item()}]")
            print(f"📊 唯一標籤值: {torch.unique(mask).tolist()}")
            
            max_label = mask.max().item()
            if max_label >= config['n_classes']:
                print(f"⚠️  警告: 最大標籤值 {max_label} 超出類別數 {config['n_classes']}")
                print(f"💡 建議: 將 n_classes 設為 {max_label + 1} 或檢查標籤資料")
            
            min_label = mask.min().item()
            if min_label < 0:
                print(f"❌ 錯誤: 發現負數標籤值 {min_label}")
                raise ValueError("標籤值不能為負數")
            
            print("✅ 資料完整性檢查通過")
        else:
            print("❌ 找不到任何訓練資料")
            
    except Exception as e:
        print(f"⚠️  資料完整性檢查失敗: {e}")
        print("💡 建議繼續執行，但請注意可能出現的問題")

def print_config_summary(config):
    """顯示配置摘要"""
    print("\n" + "=" * 60)
    print("📋 訓練配置摘要")
    print("=" * 60)
    print(f"📁 資料根目錄: {config['data_root']}")
    print(f"🎯 影像尺寸: {config['target_size']}")
    print(f"📦 批次大小: {config['batch_size']}")
    print(f"🏗️  模型通道: {config['n_channels']} -> {config['n_classes']}")
    print(f"🎮 訓練輪數: {config['num_epochs']}")
    print(f"⚡ 學習率: {config['learning_rate']}")
    print(f"🚀 優化器: {config['optimizer'].upper()}")
    print(f"📈 Momentum: {config['momentum']}")
    print(f"🔥 Warmup epochs: {config['warmup_epochs']}")
    print(f"🌡️  Warmup momentum: {config['warmup_momentum']}")
    print(f"⚖️  Warmup bias lr: {config['warmup_bias_lr']}")
    print(f"🎯 損失函數: {config['loss_type']}")
    print(f"📅 調度器: {config['scheduler']}")
    print(f"💾 保存目錄: {config['save_dir']}")
    print(f"🎨 視覺化: {'啟用' if config['enable_visualization'] else '停用'}")
    if config['enable_visualization']:
        print(f"📊 視覺化間隔: 每 {config['plot_interval']} epochs")
    print(f"🔄 數據增強: {'啟用' if config['use_augmentation'] else '停用'}")
    if config['use_augmentation']:
        print(f"🎭 增強類型: {config['augmentation_type']}")
    print("=" * 60)

def setup_optimizer_with_param_groups(model, config):
    """
    設置優化器，支援不同參數組的不同設定
    為 bias 參數和其他參數設定不同的學習率
    """
    # 分離 bias 參數和其他參數
    bias_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'bias' in name.lower():
                bias_params.append(param)
            else:
                other_params.append(param)
    
    # 建立參數組
    param_groups = [
        {
            'params': other_params,
            'lr': config['learning_rate'],
            'weight_decay': config['weight_decay'],
            'name': 'weights'
        },
        {
            'params': bias_params,
            'lr': config['learning_rate'],
            'weight_decay': 0.0,  # bias 通常不使用 weight decay
            'name': 'bias'
        }
    ]
    
    print(f"📊 參數分組: 權重參數 {len(other_params)} 個, bias 參數 {len(bias_params)} 個")
    
    # 根據優化器類型設定
    if config['optimizer'].lower() == 'adam':
        optimizer = torch.optim.Adam(
            param_groups,
            betas=(config['momentum'], 0.999)  # 使用 momentum 作為 beta1
        )
    elif config['optimizer'].lower() == 'adamw':
        optimizer = torch.optim.AdamW(
            param_groups,
            betas=(config['momentum'], 0.999)  # 使用 momentum 作為 beta1
        )
    elif config['optimizer'].lower() == 'sgd':
        # 為 SGD 添加 momentum 參數
        for group in param_groups:
            group['momentum'] = config['momentum']
        
        optimizer = torch.optim.SGD(param_groups)
    else:
        raise ValueError(f"不支援的優化器: {config['optimizer']}")
    
    return optimizer

def create_lr_scheduler(optimizer, config, warmup_scheduler=None):
    """創建學習率調度器（在 warmup 之後使用）"""
    if not config['scheduler']:
        return None
    
    if config['scheduler'] == 'step':
        from torch.optim.lr_scheduler import StepLR
        scheduler = StepLR(
            optimizer,
            step_size=config['step_size'],
            gamma=config['scheduler_factor']
        )
        print(f"📅 主調度器: StepLR (每 {config['step_size']} epoch 衰減 {config['scheduler_factor']})")
        
    elif config['scheduler'] == 'reduce_on_plateau':
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='max',
            patience=config['scheduler_patience'],
            factor=config['scheduler_factor'],
        )
        print(f"📉 主調度器: ReduceLROnPlateau (耐心值: {config['scheduler_patience']})")
        
    elif config['scheduler'] == 'cosine':
        from torch.optim.lr_scheduler import CosineAnnealingLR
        t_max = config.get('cosine_t_max', config['num_epochs'])
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=t_max,
            eta_min=config['learning_rate'] * 0.01  # 最小學習率為初始值的1%
        )
        print(f"🌊 主調度器: CosineAnnealingLR (T_max: {t_max})")
        
    else:
        raise ValueError(f"不支援的調度器: {config['scheduler']}")
    
    return scheduler

def create_trainer_from_config(config):
    """根據配置創建訓練器"""
    print("🏗️  根據配置創建增強版訓練器...")
    
    # 設置設備
    if config['device'] == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(config['device'])
    
    print(f"🖥️  使用設備: {device}")
    if device.type == 'cuda':
        print(f"🔢 GPU 名稱: {torch.cuda.get_device_name(device)}")
        print(f"💾 GPU 記憶體: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f} GB")
    
    # 創建模型
    print("🏗️  建立模型...")
    model = UNet3D(
        n_channels=config['n_channels'],
        n_classes=config['n_classes'],
        base_channels=config['base_channels'],
        num_groups=config['num_groups'],
        bilinear=config['bilinear']
    ).to(device)
    
    total_params, trainable_params = model.get_model_size()
    total_params, trainable_params = model.get_model_size()
    print(f"📊 模型參數: {total_params:,} ({trainable_params:,} 可訓練)")
    print(f"💾 估計大小: {total_params * 4 / 1024 / 1024:.1f} MB (FP32)")

    # 計算模型計算量
    try:
        from thop import profile
        sample_input = torch.randn(1, config['n_channels'], *config['target_size']).to(device)
        flops, _ = profile(model, inputs=(sample_input,), verbose=False)
        # 除以2來修正重複計算（MAC vs FLOPs）
        flops = flops / 2 / 1e9  # 轉換為 GFLOPs
        print(f"🔢 模型計算量: {flops:.3f}G")
        del sample_input
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    except:
        print("⚠️ 無法計算 FLOPs (需安裝 thop: pip install thop)")
     
    # 創建資料載入器（使用配置中的數據增強參數）
    print("📁 建立資料載入器...")
    data_loaders = create_data_loaders(
        data_root=config['data_root'],
        batch_size=config['batch_size'],
        target_size=config['target_size'],
        num_workers=config['num_workers'],
        use_augmentation=config['use_augmentation'],
        augmentation_type=config['augmentation_type']
    )
    
    # 設置優化器（支援參數分組）
    print("⚡ 設置進階優化器...")
    optimizer = setup_optimizer_with_param_groups(model, config)
    print(f"⚡ 優化器: {config['optimizer'].upper()}")
    print(f"📈 基礎學習率: {config['learning_rate']}")
    print(f"📈 Momentum/Beta1: {config['momentum']}")
    
    # 設置 Warmup 調度器
    warmup_scheduler = None
    if config['warmup_epochs'] > 0:
        warmup_scheduler = WarmupLRScheduler(
            optimizer=optimizer,
            warmup_epochs=config['warmup_epochs'],
            warmup_momentum=config['warmup_momentum'],
            warmup_bias_lr=config['warmup_bias_lr'],
            base_lr=config['learning_rate'],
            base_momentum=config['momentum']
        )
        print(f"🔥 Warmup 調度器: {config['warmup_epochs']} epochs")
        print(f"🌡️  Warmup momentum: {config['warmup_momentum']} -> {config['momentum']}")
        print(f"⚖️  Warmup bias lr 乘數: {config['warmup_bias_lr']}")
    
    # 設置主學習率調度器
    main_scheduler = create_lr_scheduler(optimizer, config, warmup_scheduler)
    
    # 設置損失函數
    if config['loss_type'] == 'dice':
        criterion = DiceLoss()
        print("🎯 損失函數: Dice Loss")
        
    elif config['loss_type'] == 'ce':
        criterion = torch.nn.CrossEntropyLoss()
        print("🎯 損失函數: Cross Entropy Loss")
        
    elif config['loss_type'] == 'combined':
        criterion = CombinedLoss(
            ce_weight=config['ce_weight'],
            dice_weight=config['dice_weight']
        )
        print(f"🎯 損失函數: Combined Loss (CE: {config['ce_weight']}, Dice: {config['dice_weight']})")
        
    else:
        raise ValueError(f"不支援的損失函數: {config['loss_type']}")
    
    # 創建增強版訓練器並傳遞完整配置
    trainer = EnhancedUNet3DTrainer(
        model=model,
        data_loaders=data_loaders,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        save_dir=config['save_dir'],
        log_interval=config['log_interval'],
        save_interval=config['save_interval'],
        scheduler=main_scheduler,
        visualize=config['enable_visualization'],
        plot_interval=config['plot_interval'],
        training_config=config  # 🔑 關鍵：傳遞完整配置
    )
    
    # 將 warmup scheduler 添加到 trainer 中
    if warmup_scheduler:
        trainer.warmup_scheduler = warmup_scheduler
        trainer.warmup_epochs = config['warmup_epochs']
        print("✅ Warmup 調度器已整合到訓練器中")
    
    return trainer

def main():
    """主訓練函數"""
    print("🚀 Enhanced 3D UNet 訓練系統啟動 (含 Warmup)")
    print("=" * 60)
    
    # 設置 CUDA 調試模式
    import os
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    print("🔧 啟用 CUDA 調試模式")
    
    config = get_config()
    
    set_seed(config['seed'])
    print(f"🎲 設置隨機種子: {config['seed']}")
    
    validate_config(config)
    print_config_summary(config)
    
    trainer = create_trainer_from_config(config)
    
    print("\n🚀 開始增強版訓練 (含 Warmup)...")
    print("=" * 70)
    
    try:
        # 執行訓練（包含視覺化和早停）
        trainer.train(
            num_epochs=config['num_epochs'],
            resume_from=config['resume_from'],
            early_stopping_patience=config['early_stopping_patience']
        )
        
        # 訓練完成後進行測試
        if config['run_test'] and 'test' in trainer.data_loaders:
            print("\n" + "=" * 70)
            print("開始測試最佳模型...")
            best_model_path = Path(config['save_dir']) / 'best_model.pth'
            if best_model_path.exists():
                # 執行測試並自動保存結果
                test_results = trainer.test(str(best_model_path), save_results=True)
                
                if test_results:
                    # 在控制台顯示結果
                    print(f"\n最終測試結果:")
                    print(f"平均損失: {test_results['avg_loss']:.4f}")
                    print(f"平均 Dice 分數: {test_results['avg_dice']:.4f}")
                    print(f"Dice 分數百分比: {test_results['avg_dice'] * 100:.2f}%")
                
            else:
                print("找不到最佳模型檔案")
        
        print("\n" + "🎉" * 20)
        print("✅ 所有任務完成！")
        print("📊 訓練曲線和視覺化結果已保存至: " + str(Path(config['save_dir']) / 'visualizations'))
        print("🎉" * 20)
        
    except KeyboardInterrupt:
        print("\n⏹️  訓練被用戶中斷")
        print("💾 保存中斷狀態...")
        try:
            trainer.save_checkpoint(len(trainer.history['train_loss']), is_best=False)
            print("✅ 狀態已保存，可使用 resume_from 參數恢復訓練")
        except:
            print("❌ 保存狀態失敗")
        
    except Exception as e:
        print(f"\n❌ 訓練過程中發生錯誤: {e}")
        import traceback
        traceback.print_exc()
        print("\n💡 建議檢查：")
        print("   - 標籤值是否在正確範圍內 [0, n_classes-1]")
        print("   - 資料格式是否正確")
        print("   - 顯卡記憶體是否足夠（可降低 batch_size）")
        print("   - 檔案路徑是否正確")
        print("   - 是否有 NaN 或異常值在資料中")

def parse_args():
    """解析命令行參數"""
    parser = argparse.ArgumentParser(
        description='Enhanced 3D UNet 訓練腳本 - 整合視覺化功能與 Warmup',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--data_root', type=str, help='資料根目錄')
    parser.add_argument('--epochs', type=int, help='訓練 epoch 數')
    parser.add_argument('--batch_size', type=int, help='批次大小')
    parser.add_argument('--lr', type=float, help='學習率')
    parser.add_argument('--resume', type=str, help='從檢查點恢復訓練')
    parser.add_argument('--save_dir', type=str, help='模型保存目錄')
    parser.add_argument('--device', type=str, help='計算設備')
    parser.add_argument('--n_classes', type=int, help='輸出類別數')
    parser.add_argument('--loss_type', type=str, choices=['dice', 'ce', 'combined'], help='損失函數類型')
    parser.add_argument('--no_viz', action='store_true', help='停用視覺化功能')
    parser.add_argument('--plot_interval', type=int, help='視覺化更新間隔')
    parser.add_argument('--no_augmentation', action='store_true', help='停用數據增強')
    parser.add_argument('--augmentation_type', type=str, 
                        choices=['light', 'medium', 'heavy', 'medical', 'medical_heavy'],
                        help='數據增強類型')
    
    # 新增 Warmup 相關參數
    parser.add_argument('--momentum', type=float, help='SGD momentum 或 Adam beta1')
    parser.add_argument('--warmup_epochs', type=float, help='Warmup epochs 數（支援小數）')
    parser.add_argument('--warmup_momentum', type=float, help='Warmup 初始 momentum')
    parser.add_argument('--warmup_bias_lr', type=float, help='Warmup bias 學習率乘數')
    parser.add_argument('--scheduler', type=str, 
                        choices=['step', 'reduce_on_plateau', 'cosine', None],
                        help='學習率調度器類型')
    
    return parser.parse_args()

def update_config_from_args(config, args):
    """使用命令行參數更新配置"""
    updated = False
    
    if args.data_root:
        config['data_root'] = args.data_root
        updated = True
    if args.epochs:
        config['num_epochs'] = args.epochs
        updated = True
    if args.batch_size:
        config['batch_size'] = args.batch_size
        updated = True
    if args.lr:
        config['learning_rate'] = args.lr
        updated = True
    if args.resume:
        config['resume_from'] = args.resume
        updated = True
    if args.save_dir:
        config['save_dir'] = args.save_dir
        updated = True
    if args.device:
        config['device'] = args.device
        updated = True
    if args.n_classes:
        config['n_classes'] = args.n_classes
        updated = True
    if args.loss_type:
        config['loss_type'] = args.loss_type
        updated = True
    if args.no_viz:
        config['enable_visualization'] = False
        updated = True
    if args.plot_interval:
        config['plot_interval'] = args.plot_interval
        updated = True
    if args.no_augmentation:
        config['use_augmentation'] = False
        updated = True
    if args.augmentation_type:
        config['augmentation_type'] = args.augmentation_type
        updated = True
    
    # 新增 Warmup 參數更新
    if args.momentum is not None:
        config['momentum'] = args.momentum
        updated = True
    if args.warmup_epochs is not None:
        config['warmup_epochs'] = args.warmup_epochs
        updated = True
    if args.warmup_momentum is not None:
        config['warmup_momentum'] = args.warmup_momentum
        updated = True
    if args.warmup_bias_lr is not None:
        config['warmup_bias_lr'] = args.warmup_bias_lr
        updated = True
    if args.scheduler is not None:
        config['scheduler'] = args.scheduler
        updated = True
    
    if updated:
        print("📝 使用命令行參數覆蓋配置")
    
    return config

if __name__ == '__main__':
    args = parse_args()
    config = get_config()
    config = update_config_from_args(config, args)
    
    globals()['get_config'] = lambda: config
    main()

