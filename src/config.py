"""
NFL 球员轨迹预测项目配置文件
"""

import os
from pathlib import Path
import torch


class Config:
    """项目配置类，管理所有超参数和路径"""
    
    # 路径配置
    DATA_DIR = Path("data/")
    OUTPUT_DIR = Path("./outputs")
    MODEL_DIR = OUTPUT_DIR / "models"
    FIGURE_DIR = Path("./figures")
    
    # 创建必要的目录
    OUTPUT_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)
    FIGURE_DIR.mkdir(exist_ok=True)
    
    # 训练配置
    SEED = 42
    N_FOLDS = 10
    BATCH_SIZE = 256
    EPOCHS = 200
    PATIENCE = 30
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-5
    
    # 模型架构配置
    WINDOW_SIZE = 10
    HIDDEN_DIM = 128
    MAX_FUTURE_HORIZON = 94
    
    # Transformer 超参数
    N_HEADS = 4
    N_LAYERS = 2
    
    # ResidualMLP Head 超参数
    MLP_HIDDEN_DIM = 256
    N_RES_BLOCKS = 2
    
    # 场地配置
    FIELD_X_MIN, FIELD_X_MAX = 0.0, 120.0
    FIELD_Y_MIN, FIELD_Y_MAX = 0.0, 53.3
    
    # 特征工程配置
    K_NEIGH = 6
    RADIUS = 30.0
    TAU = 8.0
    N_ROUTE_CLUSTERS = 7
    
    # 设备配置
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 调试模式
    DEBUG = False
    if DEBUG:
        N_FOLDS = 2
        EPOCHS = 10
    
    # 模型持久化配置
    SAVE_ARTIFACTS = True
    LOAD_ARTIFACTS = True
    LOAD_DIR = None
    
    @classmethod
    def display(cls):
        """打印所有配置"""
        print("=" * 80)
        print("配置信息")
        print("=" * 80)
        
        print("\n[路径配置]")
        print(f"  DATA_DIR:      {cls.DATA_DIR}")
        print(f"  OUTPUT_DIR:    {cls.OUTPUT_DIR}")
        print(f"  MODEL_DIR:     {cls.MODEL_DIR}")
        
        print("\n[训练配置]")
        print(f"  SEED:          {cls.SEED}")
        print(f"  N_FOLDS:       {cls.N_FOLDS}")
        print(f"  BATCH_SIZE:    {cls.BATCH_SIZE}")
        print(f"  EPOCHS:        {cls.EPOCHS}")
        print(f"  PATIENCE:      {cls.PATIENCE}")
        print(f"  LEARNING_RATE: {cls.LEARNING_RATE}")
        
        print("\n[模型配置]")
        print(f"  WINDOW_SIZE:   {cls.WINDOW_SIZE}")
        print(f"  HIDDEN_DIM:    {cls.HIDDEN_DIM}")
        print(f"  N_HEADS:       {cls.N_HEADS}")
        print(f"  N_LAYERS:      {cls.N_LAYERS}")
        print(f"  MLP_HIDDEN_DIM: {cls.MLP_HIDDEN_DIM}")
        
        print("\n[设备配置]")
        print(f"  DEVICE:        {cls.DEVICE}")
        print(f"  CUDA Available: {torch.cuda.is_available()}")
        
        print("=" * 80)


def set_seed(seed=42):
    """设置所有随机种子以保证可复现性"""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"随机种子已设置为 {seed}")
