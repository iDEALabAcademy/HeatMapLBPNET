"""
Model Builder Module
Factory functions for creating LBPNet models
"""

from typing import Dict, Any
from .lbpnet_rp import LBPNetRP
from .lbpnet_conv1x1 import LBPNetConv1x1


def build_model(config: Dict[str, Any]):
    """
    Build LBPNet model based on configuration
    
    Args:
        config: Model configuration dictionary
    
    Returns:
        LBPNet model instance
    """
    model_type = config.get('model', 'lbpnet_rp')
    if model_type == 'lbpnet_rp':
        return LBPNetRP(config)
    elif model_type == 'lbpnet_conv1x1':
        return LBPNetConv1x1(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def get_default_config() -> Dict[str, Any]:
    """Default config (legacy, no heatmap or adaptive-P)."""
    return {
        'model': 'lbpnet_rp',
        'data': {'train_ratio': 0.85, 'val_ratio': 0.15, 'seed': 42},
        'lbp_layer': {
            'num_patterns': 1,
            'num_points': 8,
            'window': 5,
            'share_across_channels': True,
            'mode': 'bits',
            'alpha_init': 0.2,
            'learn_alpha': True,
            'offset_init_std': 0.8,
        },
        'blocks': {
            'stages': 8,
            'channels_per_stage': [37, 40, 80, 80, 160, 160, 320, 320],
            'downsample_at': [1, 3, 5, 7],
            'rp_config': {'n_bits_per_out': 4, 'seed': 42},
        },
        'head': {'hidden': 512, 'dropout_rate': 0.2, 'num_classes': 10},
    }


# Previous heatmap-biased adaptive-P preset removed. Use get_default_config() and
# override adaptive-P related keys in 'lbp_layer' directly, e.g.:
# cfg = get_default_config(); cfg['lbp_layer'].update({
#     'adaptive_p_enable': True,
#     'adaptive_p_thresholds': [0.2, 0.5, 0.7],  # maps internally to adaptive_bins
#     'adaptive_p_values': [2, 4, 6, 8],         # ignored (fixed set) retained for backward compat
#     'apply_only_stage0': True,
# })
