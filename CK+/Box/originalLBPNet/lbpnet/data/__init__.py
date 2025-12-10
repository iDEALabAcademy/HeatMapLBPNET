"""
Data module
Includes dataset creation and dataloader helpers.

This wrapper lets you switch between:
- Cropped AMNIST (bbox, grayscale, normalized) and Full AMNIST (full image, grayscale, normalized)
- Cropped CK+ (bbox, grayscale, normalized) and Full CK+ (48x48 grayscale, normalized)
- SVHN (10 digit classes, 32x32 RGB/grayscale, normalized)

Selection is controlled by either:
- config['data']['variant'] in {'cropped', 'full'}
- or environment variable AMNIST_VARIANT/CKPLUS_VARIANT in {'cropped', 'full'}

Default: 'full' to preserve current behavior.
"""

import os
from typing import Optional, Tuple

# Import AMNIST implementations and dispatch at runtime
from .amnist_dataset import (
	get_amnist_datasets as _get_full_amnist_datasets,
	get_amnist_dataloaders as _get_full_amnist_dataloaders,
)
from .cropped_amnist_dataset import (
	get_amnist_datasets as _get_cropped_amnist_datasets,
	get_amnist_dataloaders as _get_cropped_amnist_dataloaders,
)

# Import CK+ implementations
from .ckplus_dataset import (
	get_ckplus_datasets as _get_full_ckplus_datasets,
	get_ckplus_dataloaders as _get_full_ckplus_dataloaders,
)
from .cropped_ckplus_dataset import (
	get_ckplus_datasets as _get_cropped_ckplus_datasets,
	get_ckplus_dataloaders as _get_cropped_ckplus_dataloaders,
)

# Import SVHN dataset
from .svhn_dataset import (
	get_svhn_datasets,
	get_svhn_dataloaders,
)

# Import cropped SVHN dataset
from .cropped_svhn_dataset import (
	get_svhn_datasets as get_cropped_svhn_datasets,
	get_svhn_dataloaders as get_cropped_svhn_dataloaders,
)

def _pick_variant(config, env_var='AMNIST_VARIANT') -> str:
	data_cfg = config.get('data', {}) if isinstance(config, dict) else {}
	variant = str(data_cfg.get('variant', os.environ.get(env_var, 'full'))).lower()
	if variant in ('cropped', 'crop', 'bbox'):
		return 'cropped'
	return 'full'

def get_amnist_datasets(config: dict, data_dir: str = '/home/sgram/Heatmap/AMNIST/New folder', download: bool = False):
	variant = _pick_variant(config, 'AMNIST_VARIANT')
	if variant == 'cropped':
		return _get_cropped_amnist_datasets(config, data_dir=data_dir, download=download)
	return _get_full_amnist_datasets(config, data_dir=data_dir)

def get_amnist_dataloaders(config: dict, data_dir: str = '/home/sgram/Heatmap/AMNIST/New folder', download: bool = False, **kwargs):
	variant = _pick_variant(config, 'AMNIST_VARIANT')
	if variant == 'cropped':
		return _get_cropped_amnist_dataloaders(config, data_dir=data_dir, download=download, **kwargs)
	return _get_full_amnist_dataloaders(config, data_dir=data_dir, **kwargs)

def get_ckplus_datasets(config: dict, data_dir: str = '/home/sgram/Heatmap/CK+/ck_dataset', download: bool = False):
	variant = _pick_variant(config, 'CKPLUS_VARIANT')
	if variant == 'cropped':
		return _get_cropped_ckplus_datasets(config, data_dir=data_dir, download=download)
	return _get_full_ckplus_datasets(config, data_dir=data_dir)

def get_ckplus_dataloaders(config: dict, data_dir: str = '/home/sgram/Heatmap/CK+/ck_dataset', download: bool = False, **kwargs):
	variant = _pick_variant(config, 'CKPLUS_VARIANT')
	if variant == 'cropped':
		return _get_cropped_ckplus_dataloaders(config, data_dir=data_dir, download=download, **kwargs)
	return _get_full_ckplus_dataloaders(config, data_dir=data_dir, **kwargs)

__all__ = [
	'get_amnist_datasets', 
	'get_amnist_dataloaders',
	'get_ckplus_datasets',
	'get_ckplus_dataloaders',
	'get_svhn_datasets',
	'get_svhn_dataloaders',
	'get_cropped_svhn_datasets',
	'get_cropped_svhn_dataloaders',
]
