# 使用指南

本项目提供了模块化的代码实现，可以直接导入使用。

## 快速开始

### 1. 环境配置

```bash
pip install -r requirements.txt
```

### 2. 数据准备

```bash
# 下载竞赛数据
kaggle competitions download -c nfl-big-data-bowl-2026-prediction

# 解压到 data 目录
mkdir -p data
unzip nfl-big-data-bowl-2026-prediction.zip -d data/
```

### 3. 训练模型

```python
from src.config import Config, set_seed
from src.models import STTransformer, TemporalHuber

# 设置随机种子
set_seed(Config.SEED)

# 查看配置
Config.display()

# 训练流程见完整实现
```

## 核心模块说明

### config.py
配置管理模块，包含所有超参数：
- 数据路径配置
- 训练超参数（batch size, learning rate等）
- 模型架构参数（hidden_dim, n_heads等）
- 特征工程参数（K_NEIGH, RADIUS等）

### models.py
模型定义模块：
- `STTransformer`: 时空 Transformer 主模型
- `ResidualMLPHead`: 残差 MLP 输出头
- `TemporalHuber`: 时间加权 Huber 损失函数

### utils.py
工具函数模块：
- 数据处理函数
- 可视化函数
- 评估指标计算

## 模型架构

完整的模型实现在 `src/models.py` 中：

```python
model = STTransformer(
    input_dim=167,      # 输入特征维度
    hidden_dim=128,     # Transformer 隐藏维度
    horizon=94,         # 预测时间范围
    window_size=10,     # 输入窗口大小
    n_heads=4,          # 注意力头数
    n_layers=2          # Transformer 层数
)
```

## 训练配置

所有训练配置在 `src/config.py` 中定义：

```python
SEED = 42
N_FOLDS = 10
BATCH_SIZE = 256
EPOCHS = 200
PATIENCE = 30
LEARNING_RATE = 1e-3
```

## 完整实现

完整的训练和推理实现包含以下步骤：

1. 数据加载和预处理
2. 特征工程（167 维特征）
3. 几何基线计算
4. 序列化和标准化
5. 模型训练（10-Fold 交叉验证）
6. 模型集成和推理

详细实现逻辑请参考 README.md 中的 Action 部分。

## 关键特征

### 几何特征（13 维）
这是本项目的核心创新，通过物理规则计算几何基线：
- geo_endpoint_x/y: 几何目标位置
- geo_vector_x/y: 到目标的向量
- geo_distance: 到目标的距离
- geo_required_vx/vy: 所需速度
- geo_velocity_error_x/y: 速度误差
- geo_required_ax/ay: 所需加速度
- geo_alignment: 速度对齐度

### 镜像 WR 特征
防守球员镜像接球手特征：
- mirror_wr_vx/vy: 接球手速度
- mirror_offset_x/y: 相对位置偏移
- mirror_wr_dist: 到接球手的距离

## 性能

- 竞赛排名: 142 / 2980 (Top 4.8%)
- 奖牌: Silver Medal
- 模型参数: 约 210 万
- 训练时间: 约 2 小时/Fold
- 推理速度: <50ms/play

## 许可证

MIT License

