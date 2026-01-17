# 方法论详解：几何神经突破（Geometric Neural Breakthrough）

## 目录
1. [核心思想](#核心思想)
2. [几何规则设计](#几何规则设计)
3. [深度学习修正](#深度学习修正)
4. [特征工程策略](#特征工程策略)
5. [模型架构](#模型架构)
6. [训练策略](#训练策略)

---

## 核心思想

### 问题：为什么传统方法不够好？

在橄榄球比赛中，球员的移动遵循一定的物理和战术规则：
- **接球手**会向球的落点移动
- **防守球员**会跟随接球手移动
- **其他球员**通常保持原有的运动轨迹

传统深度学习方法完全从数据中学习这些模式，但存在以下问题：
1. **数据效率低**：需要大量数据才能学习基本的物理规律
2. **泛化能力差**：在训练集中没见过的战术配置上表现不佳
3. **训练不稳定**：从零开始学习复杂的时空依赖关系

### 解决方案：物理先验 + 学习修正

我们的方法将问题分解为两部分：

```
最终预测 = 几何基线（物理规则） + 学习修正（深度学习）
```

**优势**：
- ✅ 几何基线提供了强先验，降低学习难度
- ✅ 模型只需学习对基线的修正，问题空间更小
- ✅ 即使修正失败，几何基线也能保证合理的预测

---

## 几何规则设计

### 1. 接球手规则（Receiver Rule）

**假设**：接球手会尽力向球的落点移动

```python
geo_endpoint_x = ball_land_x
geo_endpoint_y = ball_land_y
```

**物理意义**：
- 接球手的目标是接到球
- 最优策略是直接向球落点移动
- 实际轨迹可能因防守、碰撞等偏离，但趋势不变

### 2. 防守球员规则（Defender Rule）

**假设**：防守球员保持与接球手的相对位置

```python
# 计算防守球员与接球手的偏移量
mirror_offset_x = defender_x - receiver_x
mirror_offset_y = defender_y - receiver_y

# 防守球员的几何终点 = 球落点 + 偏移量
geo_endpoint_x = ball_land_x + mirror_offset_x
geo_endpoint_y = ball_land_y + mirror_offset_y
```

**物理意义**：
- 防守球员需要盯防接球手
- 保持相对位置可以最大化干扰接球
- 这是最常见的防守策略（人盯人）

### 3. 其他球员规则（Momentum Rule）

**假设**：其他球员保持当前的速度和方向（惯性）

```python
# 计算剩余时间
time_remaining = num_frames_output / 10.0

# 根据当前速度外推
geo_endpoint_x = current_x + velocity_x * time_remaining
geo_endpoint_y = current_y + velocity_y * time_remaining
```

**物理意义**：
- 不直接参与传球的球员通常继续原有动作
- 例如：阻挡球员继续阻挡，跑位球员继续跑位

### 4. 场地边界约束

所有几何终点都裁剪到场地范围内：

```python
geo_endpoint_x = clip(geo_endpoint_x, 0, 120)
geo_endpoint_y = clip(geo_endpoint_y, 0, 53.3)
```

---

## 深度学习修正

### 需要修正的场景

几何基线假设理想情况，但实际中存在多种因素需要修正：

#### 1. 防守压力（Defensive Pressure）
- 接球手被贴身防守时，无法直线向球移动
- 需要绕路或减速

#### 2. 球员碰撞（Collision Avoidance）
- 多个球员在相近位置时，需要避开彼此
- 不能穿过其他球员

#### 3. 战术变化（Tactical Variations）
- 假动作：接球手可能先向错误方向移动
- 区域防守：防守球员不完全盯人

#### 4. 体能差异（Physical Constraints）
- 不同球员的速度、加速度上限不同
- 需要考虑球员的身体指标

### 修正特征设计

为了让模型学习修正，我们设计了 13 个几何特征：

| 特征 | 含义 | 用途 |
|-----|-----|-----|
| `geo_endpoint_x/y` | 几何终点坐标 | 告诉模型"理想目标"在哪 |
| `geo_vector_x/y` | 到几何终点的向量 | 目标方向 |
| `geo_distance` | 到几何终点的距离 | 还有多远 |
| `geo_required_vx/vy` | 达到终点所需速度 | 速度是否足够 |
| `geo_velocity_error_x/y` | 速度误差 | 当前速度偏离多少 |
| `geo_required_ax/ay` | 达到终点所需加速度 | 需要多大加速度 |
| `geo_alignment` | 速度与几何路径对齐度 | 是否朝正确方向 |

模型看到这些特征后，可以判断：
- "接球手离球很近，但速度不足" → 预测加速
- "防守球员在接球手和球之间" → 预测绕路
- "几何路径被其他球员阻挡" → 预测改变方向

---

## 特征工程策略

### 特征分类体系

我们将 167 个特征分为 8 大类：

#### 1. 基础物理特征 (20)
- **目的**：提供球员的基本状态
- **示例**：位置、速度、加速度、方向
- **工程技巧**：
  - 计算动量 = 速度 × 体重
  - 计算动能 = 0.5 × 体重 × 速度²

#### 2. 球相关特征 (12)
- **目的**：描述球员与球的关系
- **示例**：到球的距离、角度、接近速度
- **工程技巧**：
  - 计算速度对齐度（速度向量与球方向的点积）
  - 预测按当前速度会到达的位置

#### 3. 对手交互特征 (15)
- **目的**：建模球员间的交互
- **示例**：最近对手距离、防守压力、镜像特征
- **工程技巧**：
  - 镜像 WR 特征：防守球员"镜像"最近的接球手
  - 防守压力 = 1 / (最近对手距离 + 0.5)

#### 4. 路线模式特征 (8)
- **目的**：识别战术路线类型
- **方法**：K-Means 聚类
- **特征**：
  - 轨迹直线度
  - 最大转向角度
  - 路线深度和宽度

#### 5. GNN 邻居嵌入 (18)
- **目的**：捕捉局部空间关系
- **方法**：基于距离的加权聚合
- **特征**：
  - 队友/对手的相对位置
  - 队友/对手的相对速度
  - 最近 K 个邻居的距离

#### 6. 时序特征 (70)
- **目的**：建模时间依赖
- **技巧**：
  - **Lag 特征**：过去 1-5 帧的状态
  - **滚动统计**：3/5 帧窗口的均值、标准差
  - **EMA 平滑**：指数移动平均
  - **差分特征**：相邻帧的变化量

#### 7. 几何特征 (13) ⭐
- **目的**：提供物理先验
- **详见**：[深度学习修正](#深度学习修正)

#### 8. 角色编码 (11)
- **目的**：区分不同角色的球员
- **编码**：One-hot 编码 + 角色交互特征

---

## 模型架构

### ST-Transformer 架构

我们的模型是一个 **时空 Transformer（Spatio-Temporal Transformer）**：

```
输入序列 (B, 10, 167)
    ↓
[1] 特征投影
    ↓ (B, 10, 128)
[2] + 位置编码
    ↓
[3] Transformer Encoder
    ↓ (B, 10, 128)
[4] 注意力池化
    ↓ (B, 128)
[5] ResidualMLP Head
    ↓ (B, 188)
[6] Reshape + Cumsum
    ↓
输出轨迹 (B, 94, 2)
```

### 模块详解

#### 1. 特征投影（Feature Projection）
```python
self.input_projection = nn.Linear(167, 128)
```
- 将高维特征压缩到隐藏空间
- 提取特征间的线性组合

#### 2. 位置编码（Positional Encoding）
```python
self.pos_embed = nn.Parameter(torch.randn(1, 10, 128))
```
- **可学习**的位置编码（不是固定的 sin/cos）
- 让模型知道每帧在时间序列中的位置

#### 3. Transformer Encoder
```python
TransformerEncoderLayer(
    d_model=128,
    nhead=4,
    dim_feedforward=512,
    activation='gelu'
)
```
- **多头注意力**：捕捉不同时间帧之间的依赖关系
- **前馈网络**：非线性变换
- **残差连接 + LayerNorm**：稳定训练

#### 4. 注意力池化（Attention Pooling）
```python
self.pool_attn = nn.MultiheadAttention(128, num_heads=4)
self.pool_query = nn.Parameter(torch.randn(1, 1, 128))
```
- 将序列 (10, 128) 压缩为单个向量 (128)
- 使用**可学习的 query**，自动学习关注哪些帧

#### 5. ResidualMLP Head（创新点！）
```python
ResidualMLPHead(
    input_dim=128,
    hidden_dim=256,
    output_dim=188,  # 94 frames × 2 coords
    n_res_blocks=2
)
```
- **残差块**：避免梯度消失
- **深层网络**：128 → 256 → 256 → 188
- **优于简单线性层**：能学习更复杂的映射

#### 6. 累积和（Cumulative Sum）
```python
out = torch.cumsum(out, dim=1)
```
- 模型预测**相对位移** (Δx, Δy)
- 通过累积和转换为**绝对坐标** (x, y)
- **优势**：更容易学习（位移的变化比绝对坐标更规律）

### 为什么用 Transformer 而不是 RNN？

| 方面 | Transformer | RNN/LSTM |
|-----|------------|----------|
| **并行计算** | ✅ 可并行 | ❌ 串行 |
| **长期依赖** | ✅ 直接连接 | ⚠️ 梯度消失 |
| **灵活注意力** | ✅ 自动学习关注点 | ❌ 固定权重 |
| **训练速度** | ✅ 快 | ❌ 慢 |
| **实验结果** | **0.545 RMSE** | 0.571 RMSE |

---

## 训练策略

### 1. 交叉验证

**Group K-Fold (K=10)**
```python
GroupKFold(n_splits=10)
```
- **分组依据**：`game_id`
- **原因**：同一场比赛的不同 play 高度相关，必须分组
- **好处**：避免数据泄漏，更准确评估泛化能力

### 2. 损失函数

**Temporal Huber Loss**
```python
TemporalHuber(delta=0.5, time_decay=0.03)
```

#### Huber Loss
- 对小误差使用 L2（平方误差）
- 对大误差使用 L1（绝对误差）
- **好处**：对异常值更鲁棒

#### 时间衰减权重
```python
weight(t) = exp(-0.03 × t)
```
- 近期帧权重高（t=0 时 weight=1.0）
- 远期帧权重低（t=90 时 weight≈0.07）
- **原因**：近期预测更重要且更准确

### 3. 优化器

**AdamW**
```python
AdamW(lr=1e-3, weight_decay=1e-5)
```
- **AdamW**：带权重衰减的 Adam
- **学习率**：1e-3（经过调优）
- **权重衰减**：1e-5（轻微正则化）

### 4. 学习率调度

**ReduceLROnPlateau**
```python
ReduceLROnPlateau(patience=5, factor=0.5)
```
- 验证损失不下降时，学习率 × 0.5
- 帮助模型在后期精细调整

### 5. 早停（Early Stopping）

**Patience = 30**
```python
if val_loss not improve for 30 epochs:
    stop training
```
- 防止过拟合
- 节省时间

### 6. 数据归一化

**StandardScaler（每个 Fold 独立）**
```python
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_val = scaler.transform(X_val)
```
- **重要**：验证集使用训练集的 scaler
- **防止泄漏**：每个 Fold 独立缩放

---

## 消融实验设计

### 实验设置

| 实验 | 修改 | 目的 |
|-----|-----|-----|
| **完整模型** | 无 | Baseline |
| **去除几何特征** | 移除 13 个 `geo_*` 特征 | 验证几何先验的价值 |
| **去除 GNN 嵌入** | 移除 18 个 `gnn_*` 特征 | 验证邻居交互的价值 |
| **替换为 GRU** | 用 GRU 替换 Transformer | 验证 Transformer 的优势 |
| **去除 ResidualMLP** | 用单层线性层替换 | 验证残差结构的价值 |
| **简单 Baseline** | 只用动量规则外推 | 对比深度学习的提升 |

### 结果分析

```
完整模型:        0.545 RMSE  (基准)
去除几何特征:     0.578 RMSE  (+0.033) ← 几何特征最重要！
去除 GNN:       0.562 RMSE  (+0.017)
替换为 GRU:     0.571 RMSE  (+0.026)
去除 ResidualMLP: 0.553 RMSE (+0.008)
简单 Baseline:   0.823 RMSE  (+0.278) ← 深度学习提升巨大
```

**结论**：
1. **几何特征贡献最大**（-3.3% RMSE）
2. **Transformer 优于 RNN**（-2.6% RMSE）
3. **GNN 邻居信息很重要**（-1.7% RMSE）
4. **ResidualMLP 带来稳定提升**（-0.8% RMSE）

---

## 总结

### 核心创新

1. **物理先验融合**：将领域知识编码为几何规则
2. **学习修正策略**：模型只需学习对基线的修正
3. **时空 Transformer**：高效建模序列依赖
4. **ResidualMLP Head**：深层网络提升表达能力
5. **精心设计的特征**：167 维特征覆盖所有重要信息

### 适用性

这个方法可以推广到其他领域：
- **篮球**：球员向球移动，防守盯人
- **足球**：进攻向球门，防守回防
- **无人机**：目标点导航 + 障碍避让
- **机器人**：路径规划 + 动态障碍

**核心思想**：
> 当问题有明确的物理规则时，不要让模型从零学习，而是让模型学习对规则的修正。

---

*本文档详细介绍了几何神经突破的方法论。如有疑问，欢迎提 Issue！*

