"""
NFL 球员轨迹预测神经网络模型
"""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """残差块：前馈网络 + 快捷连接"""
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x):
        return x + self.ffn(self.norm(x))


class ResidualMLPHead(nn.Module):
    """
    残差多层感知机输出头
    
    参数:
        input_dim: 输入维度
        hidden_dim: MLP 内部隐藏维度
        output_dim: 输出维度
        n_res_blocks: 残差块数量
        dropout: Dropout 概率
    """
    def __init__(self, input_dim, hidden_dim, output_dim, n_res_blocks=2, dropout=0.2):
        super().__init__()
        
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU()
        )
        
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, hidden_dim * 2, dropout) for _ in range(n_res_blocks)]
        )
        
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.input_layer(x)
        x = self.residual_blocks(x)
        x = self.output_norm(x)
        x = self.output_layer(x)
        return x


class STTransformer(nn.Module):
    """
    时空 Transformer 模型
    
    用于学习球员轨迹的时空依赖关系
    
    架构:
        输入 → 特征投影 → 位置编码 → Transformer Encoder
        → 注意力池化 → ResidualMLP → 累积和 → 输出
    
    参数:
        input_dim: 输入特征维度
        hidden_dim: Transformer 隐藏维度
        horizon: 预测时间范围
        window_size: 输入窗口大小
        n_heads: 多头注意力头数
        n_layers: Transformer 层数
        dropout: Dropout 概率
    """
    def __init__(self, input_dim, hidden_dim, horizon, window_size, n_heads, n_layers, dropout=0.1):
        super().__init__()
        self.horizon = horizon
        self.hidden_dim = hidden_dim
        
        # 特征投影
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # 可学习的位置编码
        self.pos_embed = nn.Parameter(torch.randn(1, window_size, hidden_dim))
        self.embed_dropout = nn.Dropout(dropout)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )
        
        # 注意力池化
        self.pool_ln = nn.LayerNorm(hidden_dim)
        self.pool_attn = nn.MultiheadAttention(hidden_dim, num_heads=n_heads, batch_first=True)
        self.pool_query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        # ResidualMLP 输出头
        from .config import Config
        config = Config()
        self.head = ResidualMLPHead(
            input_dim=hidden_dim,
            hidden_dim=config.MLP_HIDDEN_DIM,
            output_dim=horizon * 2,
            n_res_blocks=config.N_RES_BLOCKS,
            dropout=0.2
        )
    
    def forward(self, x):
        """
        参数:
            x: (Batch, Window, Features)
        
        返回:
            out: (Batch, Horizon, 2) - 预测的轨迹坐标
        """
        B, S, _ = x.shape
        
        # 特征投影和位置编码
        x_embed = self.input_projection(x)
        x = x_embed + self.pos_embed[:, :S, :]
        x = self.embed_dropout(x)
        
        # Transformer 编码
        h = self.transformer_encoder(x)
        
        # 注意力池化
        q = self.pool_query.expand(B, -1, -1)
        ctx, _ = self.pool_attn(q, self.pool_ln(h), self.pool_ln(h))
        ctx = ctx.squeeze(1)
        
        # 输出预测
        out = self.head(ctx)
        out = out.view(B, self.horizon, 2)
        
        # 累积和：相对位移转换为绝对位置
        out = torch.cumsum(out, dim=1)
        
        return out


class TemporalHuber(nn.Module):
    """
    时间加权 Huber 损失
    
    对近期帧的预测误差给予更高权重，远期帧权重指数衰减
    
    参数:
        delta: Huber 损失的阈值
        time_decay: 时间衰减系数
    """
    def __init__(self, delta=0.5, time_decay=0.03):
        super().__init__()
        self.delta = delta
        self.time_decay = time_decay
    
    def forward(self, pred, target, mask):
        """
        参数:
            pred: (B, T, 2) - 预测值
            target: (B, T, 2) - 真实值
            mask: (B, T) - 有效帧掩码
        
        返回:
            loss: 标量损失值
        """
        err = pred - target
        abs_err = torch.abs(err)
        huber = torch.where(
            abs_err <= self.delta,
            0.5 * err * err,
            self.delta * (abs_err - 0.5 * self.delta)
        )
        
        # 时间加权
        if self.time_decay > 0:
            L = pred.size(1)
            t = torch.arange(L, device=pred.device).float()
            weight = torch.exp(-self.time_decay * t).view(1, L, 1)
            huber = huber * weight
            mask = mask.unsqueeze(-1) * weight
        else:
            mask = mask.unsqueeze(-1)
        
        return (huber * mask).sum() / (mask.sum() + 1e-8)


def count_parameters(model):
    """统计模型参数数量"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"总参数量: {total:,}")
    print(f"可训练参数量: {trainable:,}")
    
    return total, trainable
