"""
NFL 球员轨迹预测项目工具包
"""

__version__ = '1.0.0'
__author__ = 'Yuan Zhiyi'

from .config import Config, set_seed
from .models import STTransformer, TemporalHuber, ResidualMLPHead, count_parameters
from .utils import (
    get_velocity,
    height_to_feet,
    clip_to_field,
    compute_rmse,
    plot_trajectory,
    plot_field,
    visualize_play,
    plot_training_curves,
    plot_feature_importance,
    create_submission,
    EarlyStopping
)

__all__ = [
    'Config',
    'set_seed',
    'STTransformer',
    'TemporalHuber',
    'ResidualMLPHead',
    'count_parameters',
    'get_velocity',
    'height_to_feet',
    'clip_to_field',
    'compute_rmse',
    'plot_trajectory',
    'plot_field',
    'visualize_play',
    'plot_training_curves',
    'plot_feature_importance',
    'create_submission',
    'EarlyStopping'
]
