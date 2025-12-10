"""
Data module
Includes dataset creation and dataloader helpers.

This wrapper supports AMNIST cropped dataset (19x18).
Uses exact bbox cropping from avg_top40_AMNIST/bbox.npy.
"""

import os
from typing import Optional, Tuple

# ----------------------------------------------------------
# Dataset imports
# ----------------------------------------------------------
from .amnist_dataset_cropped import (
	get_amnist_datasets,
	get_amnist_dataloaders,
)

from .ckplus_dataset import (
	get_ckplus_datasets as _get_ckplus_datasets_full,
	get_ckplus_dataloaders as _get_ckplus_dataloaders_full,
)

from .cropped_ckplus_dataset import (
	get_ckplus_datasets as _get_ckplus_datasets_cropped,
	get_ckplus_dataloaders as _get_ckplus_dataloaders_cropped,
)

# Variant dispatcher for CK+
def _pick_variant(config, full_fn, cropped_fn):
	"""Choose full or cropped based on config['data'].get('variant')"""
	variant = config.get('data', {}).get('variant', 'full')
	if variant == 'cropped':
		return cropped_fn(config)
	else:
		return full_fn(config)

def get_ckplus_datasets(config):
	return _pick_variant(config, _get_ckplus_datasets_full, _get_ckplus_datasets_cropped)

def get_ckplus_dataloaders(config):
	return _pick_variant(config, _get_ckplus_dataloaders_full, _get_ckplus_dataloaders_cropped)

# ----------------------------------------------------------
# Heatmap utilities
# ----------------------------------------------------------
from .heatmap import (
	get_heatmap_numpy,
	get_heatmap_torch,
	HeatmapProvider,
)

__all__ = [
	'get_amnist_datasets',
	'get_amnist_dataloaders',
	'get_ckplus_datasets',
	'get_ckplus_dataloaders',
	'get_heatmap_numpy',
	'get_heatmap_torch',
	'HeatmapProvider',
]
