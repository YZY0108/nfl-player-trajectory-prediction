"""
Utility Functions for NFL Player Trajectory Prediction
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def get_velocity(speed, direction_deg):
    """
    将速度标量和方向角转换为速度向量
    
    Args:
        speed: 速度标量（码/秒）
        direction_deg: 方向角（度，0度=北，顺时针）
    
    Returns:
        (vx, vy): 速度分量
    """
    theta = np.deg2rad(direction_deg)
    vx = speed * np.sin(theta)
    vy = speed * np.cos(theta)
    return vx, vy


def height_to_feet(height_str):
    """
    将身高字符串转换为英尺
    
    Args:
        height_str: 身高字符串，格式为 "6-2" (6英尺2英寸)
    
    Returns:
        height_feet: 身高（英尺）
    """
    try:
        ft, inches = map(int, str(height_str).split('-'))
        return ft + inches / 12
    except:
        return 6.0  # 默认值


def clip_to_field(x, y, x_min=0.0, x_max=120.0, y_min=0.0, y_max=53.3):
    """
    将坐标裁剪到场地范围内
    
    Args:
        x, y: 坐标
        x_min, x_max: x 轴范围
        y_min, y_max: y 轴范围
    
    Returns:
        (x_clipped, y_clipped): 裁剪后的坐标
    """
    x_clipped = np.clip(x, x_min, x_max)
    y_clipped = np.clip(y, y_min, y_max)
    return x_clipped, y_clipped


def compute_rmse(pred, target):
    """
    计算 RMSE（Root Mean Squared Error）
    
    Args:
        pred: 预测值，shape (N, 2) 或 (N,)
        target: 真实值，shape (N, 2) 或 (N,)
    
    Returns:
        rmse: RMSE 值
    """
    if pred.ndim == 2:
        # 2D 情况：计算欧式距离
        sq_errors = (pred[:, 0] - target[:, 0]) ** 2 + (pred[:, 1] - target[:, 1]) ** 2
    else:
        # 1D 情况
        sq_errors = (pred - target) ** 2
    
    return np.sqrt(np.mean(sq_errors))


def plot_trajectory(ax, trajectory, label='', color='blue', marker='o', alpha=0.7):
    """
    在给定的坐标轴上绘制轨迹
    
    Args:
        ax: Matplotlib 坐标轴
        trajectory: 轨迹点，shape (T, 2)
        label: 图例标签
        color: 颜色
        marker: 标记样式
        alpha: 透明度
    """
    ax.plot(trajectory[:, 0], trajectory[:, 1], 
            color=color, marker=marker, label=label, 
            alpha=alpha, linewidth=2, markersize=4)
    
    # 标记起点和终点
    ax.scatter(trajectory[0, 0], trajectory[0, 1], 
              color=color, s=100, marker='o', edgecolor='black', zorder=10)
    ax.scatter(trajectory[-1, 0], trajectory[-1, 1], 
              color=color, s=100, marker='*', edgecolor='black', zorder=10)


def plot_field(ax, x_range=(0, 120), y_range=(0, 53.3)):
    """
    绘制橄榄球场地
    
    Args:
        ax: Matplotlib 坐标轴
        x_range: x 轴范围（码）
        y_range: y 轴范围（码）
    """
    # 场地边界
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    
    # 场地线
    ax.axvline(x=0, color='white', linewidth=2)
    ax.axvline(x=120, color='white', linewidth=2)
    ax.axhline(y=0, color='white', linewidth=2)
    ax.axhline(y=53.3, color='white', linewidth=2)
    
    # 每10码的线
    for x in range(10, 120, 10):
        ax.axvline(x=x, color='white', linewidth=1, alpha=0.3)
    
    # 设置背景色
    ax.set_facecolor('#2E8B57')  # 草地绿
    
    # 标签
    ax.set_xlabel('X (yards)', fontsize=12)
    ax.set_ylabel('Y (yards)', fontsize=12)
    ax.grid(False)


def visualize_play(input_df, output_df, game_id, play_id, save_path=None):
    """
    可视化一个 play 的球员轨迹
    
    Args:
        input_df: 输入数据（投球前）
        output_df: 输出数据（投球后）
        game_id: 比赛 ID
        play_id: Play ID
        save_path: 保存路径（可选）
    """
    fig, ax = plt.subplots(figsize=(15, 8))
    plot_field(ax)
    
    # 筛选数据
    input_play = input_df[(input_df['game_id'] == game_id) & (input_df['play_id'] == play_id)]
    output_play = output_df[(output_df['game_id'] == game_id) & (output_df['play_id'] == play_id)]
    
    # 绘制每个球员的轨迹
    for nfl_id in input_play['nfl_id'].unique():
        player_input = input_play[input_play['nfl_id'] == nfl_id].sort_values('frame_id')
        player_output = output_play[output_play['nfl_id'] == nfl_id].sort_values('frame_id')
        
        if len(player_output) == 0:
            continue
        
        # 确定颜色
        role = player_input['player_role'].iloc[-1]
        if role == 'Targeted Receiver':
            color = 'red'
        elif role == 'Defensive Coverage':
            color = 'blue'
        else:
            color = 'gray'
        
        # 绘制轨迹
        trajectory = np.vstack([
            player_input[['x', 'y']].values,
            player_output[['x', 'y']].values
        ])
        
        plot_trajectory(ax, trajectory, label=role if nfl_id == input_play['nfl_id'].iloc[0] else '', 
                       color=color, alpha=0.6)
    
    # 绘制球落点
    if 'ball_land_x' in input_play.columns:
        ball_x = input_play['ball_land_x'].iloc[0]
        ball_y = input_play['ball_land_y'].iloc[0]
        ax.scatter(ball_x, ball_y, color='orange', s=300, marker='*', 
                  edgecolor='black', linewidth=2, label='Ball Landing', zorder=20)
    
    ax.legend(loc='upper right', fontsize=10)
    ax.set_title(f'Play Visualization - Game {game_id}, Play {play_id}', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved to {save_path}")
    
    plt.show()


def plot_training_curves(history, save_path=None):
    """
    绘制训练曲线
    
    Args:
        history: 训练历史，字典格式 {'train_loss': [...], 'val_loss': [...]}
        save_path: 保存路径（可选）
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = range(1, len(history['train_loss']) + 1)
    ax.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training Curves', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved to {save_path}")
    
    plt.show()


def plot_feature_importance(feature_names, importances, top_n=20, save_path=None):
    """
    绘制特征重要性
    
    Args:
        feature_names: 特征名称列表
        importances: 特征重要性值
        top_n: 显示前 N 个特征
        save_path: 保存路径（可选）
    """
    # 排序
    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]
    
    # 绘图
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(top_n), top_importances, color='steelblue', alpha=0.8)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_features)
    ax.invert_yaxis()
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title(f'Top {top_n} Feature Importance', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved to {save_path}")
    
    plt.show()


def create_submission(predictions, test_template, save_path='submission.csv'):
    """
    创建提交文件
    
    Args:
        predictions: 预测结果，DataFrame 或 array
        test_template: 测试模板
        save_path: 保存路径
    """
    if isinstance(predictions, np.ndarray):
        submission = test_template.copy()
        submission['x'] = predictions[:, 0]
        submission['y'] = predictions[:, 1]
    else:
        submission = predictions
    
    submission.to_csv(save_path, index=False)
    print(f"✓ Submission saved to {save_path}")
    print(f"  Shape: {submission.shape}")
    print(f"  Columns: {list(submission.columns)}")
    
    return submission


class EarlyStopping:
    """早停机制"""
    def __init__(self, patience=10, min_delta=0.0, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model = None
    
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'  EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_model = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0
        
        return self.early_stop

