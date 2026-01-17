# NFL Player Trajectory Prediction - 项目设置指南

## 🚀 快速开始

### 1. 克隆或下载项目

```bash
# 如果使用 Git
git clone <your-repo-url>
cd nfl-player-trajectory-prediction

# 或者直接解压下载的文件夹
cd nfl-player-trajectory-prediction
```

### 2. 创建虚拟环境

```bash
# 使用 venv
python -m venv venv

# 激活虚拟环境
# macOS/Linux:
source venv/bin/activate

# Windows:
venv\Scripts\activate
```

### 3. 安装依赖

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. 下载数据

```bash
# 确保已安装 Kaggle API
pip install kaggle

# 配置 Kaggle 凭证（如果还没配置）
# 1. 从 https://www.kaggle.com/account 下载 kaggle.json
# 2. 放置到 ~/.kaggle/kaggle.json (macOS/Linux) 或 C:\Users\<YourUsername>\.kaggle\kaggle.json (Windows)
# 3. chmod 600 ~/.kaggle/kaggle.json (macOS/Linux)

# 下载竞赛数据
kaggle competitions download -c nfl-big-data-bowl-2026-prediction

# 解压到 data 目录
unzip nfl-big-data-bowl-2026-prediction.zip -d data/
```

### 5. 运行 Notebooks

```bash
# 启动 Jupyter
jupyter notebook

# 或使用 JupyterLab
jupyter lab
```

按顺序运行：
1. `notebooks/01_data_exploration.ipynb` - 数据探索
2. `notebooks/02_feature_engineering.ipynb` - 特征工程
3. `notebooks/03_model_training.ipynb` - 模型训练
4. `notebooks/04_results_analysis.ipynb` - 结果分析

---

## 📁 项目结构说明

```
nfl-player-trajectory-prediction/
│
├── README.md                          # 项目主页（面试官首先看的）
├── SETUP.md                           # 本文件：环境配置指南
├── requirements.txt                   # Python 依赖包
├── .gitignore                        # Git 忽略文件
│
├── notebooks/                         # Jupyter Notebooks（讲故事）
│   ├── 01_data_exploration.ipynb     # 数据探索与可视化
│   ├── 02_feature_engineering.ipynb  # 特征工程详解
│   ├── 03_model_training.ipynb       # 模型训练流程
│   └── 04_results_analysis.ipynb     # 结果分析与消融实验
│
├── src/                              # 源代码（工程化）
│   ├── __init__.py                   # 包初始化
│   ├── config.py                     # 配置管理
│   ├── features.py                   # 特征工程函数
│   ├── models.py                     # 模型架构定义
│   ├── training.py                   # 训练与验证逻辑
│   └── utils.py                      # 工具函数
│
├── docs/                             # 文档
│   ├── methodology.md                # 方法论详解（核心创新）
│   ├── architecture.png              # 架构图
│   └── results.md                    # 完整结果报告
│
├── figures/                          # 可视化图表
│   ├── eda/                         # EDA 图表
│   ├── model/                       # 模型相关
│   └── results/                     # 结果可视化
│
├── data/                             # 数据目录（不提交到 Git）
│   ├── train/                       # 训练数据
│   └── test/                        # 测试数据
│
└── outputs/                          # 输出目录（不提交到 Git）
    └── models/                      # 训练好的模型
```

---

## 🔧 常见问题

### Q1: ImportError: No module named 'src'

**解决方案**：
```bash
# 在项目根目录运行
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 或在 Notebook 中添加
import sys
sys.path.append('..')
```

### Q2: CUDA out of memory

**解决方案**：
- 减小 `BATCH_SIZE`（在 `src/config.py` 中）
- 或使用 CPU（自动检测）

### Q3: 数据文件找不到

**解决方案**：
- 确保数据在 `data/` 目录
- 检查 `src/config.py` 中的 `DATA_DIR` 路径

---

## 📝 使用说明

### 方式 1：使用 Notebooks（推荐用于展示）

按顺序运行 `notebooks/` 中的文件，每个 Notebook 都有详细的说明和可视化。

### 方式 2：使用 Python 脚本（推荐用于训练）

```python
# train.py
from src.config import Config, set_seed
from src.training import train_full_pipeline

# 设置随机种子
set_seed(Config.SEED)

# 训练模型
train_full_pipeline(Config)
```

### 方式 3：导入为库

```python
from src import STTransformer, Config
from src.utils import visualize_play

# 使用模型
model = STTransformer(
    input_dim=167,
    hidden_dim=128,
    horizon=94,
    window_size=10,
    n_heads=4,
    n_layers=2
)
```

---

## 🎯 面试展示建议

### 如果面试官想快速了解（5-10 分钟）：
1. 打开 **README.md**，展示：
   - 核心创新（几何神经突破）
   - 性能指标
   - 架构图

2. 打开 **docs/methodology.md**，讲解：
   - 物理先验的设计
   - 为什么比纯数据驱动好

### 如果面试官想看代码（15-30 分钟）：
1. **src/models.py**：展示模型架构
2. **src/features.py**：展示特征工程（如果你补充了）
3. **notebooks/03_model_training.ipynb**：展示训练流程

### 如果面试官想看分析（30-45 分钟）：
1. **notebooks/01_data_exploration.ipynb**：数据理解
2. **notebooks/04_results_analysis.ipynb**：结果和消融实验

---

## 📊 预期输出

训练完成后，你会得到：

### 1. 模型文件
```
outputs/models/
├── model_fold1.pt
├── model_fold2.pt
├── ...
├── model_fold10.pt
├── scaler_fold1.pkl
├── ...
└── route_kmeans.pkl
```

### 2. 可视化图表
```
figures/
├── eda/
│   ├── player_position_distribution.png
│   └── distance_to_ball_distribution.png
├── model/
│   └── architecture.png
└── results/
    ├── training_curves.png
    └── ablation_study.png
```

### 3. 性能指标
```
Cross-Validation Results:
  Fold 1: RMSE = 0.547
  Fold 2: RMSE = 0.543
  ...
  Average: 0.545 ± 0.008
```

---

## 🤝 贡献

如果你发现 bug 或有改进建议：
1. Fork 本仓库
2. 创建 feature 分支
3. 提交 Pull Request

---

## 📧 联系方式

- **GitHub**: [@yourusername](https://github.com/yourusername)
- **Email**: your.email@example.com
- **LinkedIn**: [Your Name](https://linkedin.com/in/yourprofile)

---

**祝你面试顺利！🎉**

