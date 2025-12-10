"""
CK+ Dataset Module (Cropped version)
Handles CK+ dataset loading using the exact global bbox crop (grayscale, [-1,1])
while preserving the original API and splitting/loader logic.
"""

# ---- begin: make external crop_CK+ visible ----
import sys, os
import importlib.util

_CROP_DIR = "/home/sgram/Heatmap/CK+/Box"
_CROP_FILE = os.path.join(_CROP_DIR, "crop_CK+.py")

try:
    spec = importlib.util.spec_from_file_location("crop_ckplus_module", _CROP_FILE)
    crop_ckplus_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(crop_ckplus_module)
    CKPlusCropExact = crop_ckplus_module.CKPlusCropExact
except Exception as e:
    raise ImportError(
        f"Failed to import CKPlusCropExact from {_CROP_FILE}. "
        f"Check the path and file name. Original error: {e}"
    )
# ---- end: make external crop_CK+ visible ----


import os
from typing import Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets

# ---------------------------
# Utilities
# ---------------------------

def _get_targets_tensor(ds) -> torch.Tensor:
    """Extract targets from ImageFolder dataset"""
    if hasattr(ds, 'targets'):
        t = ds.targets
    else:
        # Fallback: build targets list from dataset
        t = [ds[i][1] for i in range(len(ds))]
    return t if isinstance(t, torch.Tensor) else torch.as_tensor(t)


def stratified_split_indices(targets: torch.Tensor, n_val: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    assert 0 < n_val < len(targets)
    g = torch.Generator().manual_seed(int(seed))
    targets_np = targets.detach().cpu().numpy()
    idx_all = np.arange(len(targets_np))
    val_idx_list = []
    num_classes = int(targets_np.max()) + 1

    for c in range(num_classes):
        cls_idx = idx_all[targets_np == c]
        if len(cls_idx) == 0:
            continue
        n_take = int(round(len(cls_idx) * n_val / len(targets_np)))
        n_take = min(n_take, len(cls_idx))
        perm = torch.randperm(len(cls_idx), generator=g).numpy()
        val_idx_list.append(cls_idx[perm[:n_take]])

    val_idx = np.concatenate(val_idx_list) if len(val_idx_list) > 0 else np.empty((0,), dtype=np.int64)

    # fix rounding errors to hit exactly n_val
    if len(val_idx) != n_val:
        rest = np.setdiff1d(idx_all, val_idx, assume_unique=False)
        g2 = np.random.default_rng(int(seed))
        if len(val_idx) > n_val:
            drop = g2.choice(val_idx, size=len(val_idx) - n_val, replace=False)
            val_idx = np.setdiff1d(val_idx, drop, assume_unique=False)
        else:
            add = g2.choice(rest, size=n_val - len(val_idx), replace=False)
            val_idx = np.concatenate([val_idx, add])

    train_idx = np.setdiff1d(idx_all, val_idx, assume_unique=False)
    return train_idx, val_idx


def _build_hist(indices: Optional[np.ndarray], targets: torch.Tensor, num_classes: int = 7):
    used = targets.detach().cpu() if indices is None else targets.detach().cpu()[torch.as_tensor(indices, dtype=torch.long)]
    counts = torch.bincount(used, minlength=num_classes).to(torch.int64)
    total = int(counts.sum().item())
    perc = (counts.float() / max(1, total) * 100.0).tolist()
    return counts.tolist(), perc, total


# ---------------------------
# Public API
# ---------------------------

def get_ckplus_datasets(
    config: dict,
    data_dir: str = '/home/sgram/Heatmap/CK+/ck_dataset',
    download: bool = True  # kept for signature compatibility; ignored in cropped pipeline
) -> Tuple[Dataset, Optional[Dataset], Dataset]:
    """
    Get CK+ train, validation, and test datasets (CROPPED, grayscale, normalized).
    - Uses exact global bbox cropping (no resize/square) via CKPlusCropExact.
    - Preserves original splitting behavior.
    """

    # ---- configuration (preserve original keys; add cropped-specific with sensible defaults)
    data_cfg = config.get('data', {}) if isinstance(config, dict) else {}

    # Original split config
    train_ratio = data_cfg.get('train_ratio', config.get('train_ratio', 0.8))
    val_ratio = data_cfg.get('val_ratio', config.get('val_ratio', 0.1))
    seed = int(data_cfg.get('seed', config.get('seed', 42)))
    val_size = int(data_cfg.get('val_size', -1))       # -1 => ratio
    split_seed = int(data_cfg.get('split_seed', seed))
    stratified = bool(data_cfg.get('stratified', True))

    # Cropped pipeline config
    local_root = data_cfg.get('local_root', data_dir)
    bbox_path  = data_cfg.get('bbox_path', '/home/sgram/Heatmap/CK+/Box/avg_top40_CK+/bbox.npy')
    normalize  = bool(data_cfg.get('normalize', True))

    # reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ---- load bbox and sanity-check size
    bbox = np.load(os.path.abspath(bbox_path)).astype(int).tolist()  # [t,l,b,r], inclusive
    t, l, b, r = bbox
    h_box = b - t + 1
    w_box = r - l + 1

    # ---- base datasets (no transforms; wrapper handles grayscale+normalize)
    base_full = datasets.ImageFolder(root=local_root, transform=None)
    targets = _get_targets_tensor(base_full)
    N = len(base_full)

    # ---- determine split
    if val_size == -1:
        # ratio mode
        gen = torch.Generator().manual_seed(split_seed)
        train_sz = int(float(train_ratio) * N)
        val_sz = int(float(val_ratio) * N)
        test_sz = N - train_sz - val_sz
        train_subset, val_subset, test_subset = torch.utils.data.random_split(
            range(N), [train_sz, val_sz, test_sz], generator=gen
        )
        train_indices = list(train_subset.indices)
        val_indices = list(val_subset.indices)
        test_indices = list(test_subset.indices)
    elif val_size == 0:
        train_indices = list(range(N))
        val_indices = None
        test_indices = []
    else:
        if stratified:
            tr_idx, va_idx = stratified_split_indices(targets, n_val=val_size, seed=split_seed)
            train_indices = tr_idx.tolist()
            val_indices = va_idx.tolist()
            # Remaining for test
            all_idx = set(range(N))
            train_val_idx = set(train_indices + val_indices)
            test_indices = list(all_idx - train_val_idx)
        else:
            gen = torch.Generator().manual_seed(split_seed)
            perm = torch.randperm(N, generator=gen).tolist()
            val_indices = perm[:val_size]
            # Split remaining into train/test
            remaining = perm[val_size:]
            train_sz = int(0.8 * len(remaining))
            train_indices = remaining[:train_sz]
            test_indices = remaining[train_sz:]

    # ---- wrap with cropped grayscale dataset (exact bbox, normalize [-1,1])
    base_train = datasets.ImageFolder(root=local_root, transform=None)
    base_val = datasets.ImageFolder(root=local_root, transform=None)
    base_test = datasets.ImageFolder(root=local_root, transform=None)
    
    train_full_wrapped = CKPlusCropExact(base_train, tuple(bbox), normalize=normalize)
    eval_full_wrapped  = CKPlusCropExact(base_val, tuple(bbox), normalize=normalize)
    test_wrapped       = CKPlusCropExact(base_test, tuple(bbox), normalize=normalize)

    # ---- build subsets for train/val/test
    train_dataset = Subset(train_full_wrapped, train_indices)
    val_dataset = Subset(eval_full_wrapped, val_indices) if val_indices is not None else None
    test_dataset = Subset(test_wrapped, test_indices)

    # ---- logging sizes and class histograms
    print(f"📊 CK+ Dataset sizes (cropped {h_box}x{w_box}):")
    print(f"   Train set: {len(train_dataset)} samples")
    print(f"   Validation set: {0 if val_dataset is None else len(val_dataset)} samples")
    print(f"   Test set: {len(test_dataset)} samples")

    num_classes = len(base_full.classes)
    print(f"   Classes ({num_classes}): {base_full.classes}")
    
    mother_counts, mother_perc, _ = _build_hist(None, targets, num_classes)
    tr_counts, tr_perc, _ = _build_hist(np.array(train_indices, dtype=np.int64), targets, num_classes)
    print(f"   [train] cls_hist={tr_counts}, perc={[round(p,2) for p in tr_perc]}")

    if val_dataset is not None and val_indices is not None:
        va_counts, va_perc, _ = _build_hist(np.array(val_indices, dtype=np.int64), targets, num_classes)
        print(f"   [val]   cls_hist={va_counts}, perc={[round(p,2) for p in va_perc]}")
        diffs = [abs(va_perc[i] - mother_perc[i]) for i in range(num_classes)]
        max_diff = max(diffs) if diffs else 0.0
        # For small datasets (< 1000 samples), allow 10pp deviation; otherwise 2pp
        tolerance = 10.0 if N < 1000 else 2.0
        assert max_diff <= tolerance + 1e-6, f"Val class distribution deviates by >{tolerance}pp (max={max_diff:.2f}pp)"

    te_counts, te_perc, _ = _build_hist(np.array(test_indices, dtype=np.int64), targets, num_classes)
    print(f"   [test]  cls_hist={te_counts}, perc={[round(p,2) for p in te_perc]}")

    return train_dataset, val_dataset, test_dataset


def get_ckplus_dataloaders(
    config: dict,
    data_dir: str = '/home/sgram/Heatmap/CK+/ck_dataset',
    download: bool = True,  # kept for signature compatibility; ignored in cropped pipeline
    **kwargs
) -> Tuple[DataLoader, Optional[DataLoader], DataLoader]:
    """
    Get CK+ dataloaders (cropped), preserving original DataLoader behavior.
    """

    # datasets
    train_dataset, val_dataset, test_dataset = get_ckplus_datasets(
        config, data_dir, download
    )

    # training config passthrough
    training_config = config.get('training', {})
    data_cfg = config.get('data', {}) if isinstance(config, dict) else {}
    batch_size = training_config.get('batch_size', 32)
    num_workers = int(training_config.get('num_workers', data_cfg.get('num_workers', 4)))
    pin_memory = bool(training_config.get('pin_memory', data_cfg.get('pin_memory', True)))

    # deterministic DataLoader behavior
    base_seed = int(config.get('seed', 42))
    g = torch.Generator()
    g.manual_seed(base_seed)

    def _worker_init_fn(worker_id: int):
        import numpy as _np
        s = base_seed + worker_id
        _np.random.seed(s)
        torch.manual_seed(s)

    # loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=g,
        worker_init_fn=_worker_init_fn,
        **kwargs
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            generator=g,
            worker_init_fn=_worker_init_fn,
            **kwargs
        )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=g,
        worker_init_fn=_worker_init_fn,
        **kwargs
    )

    return train_loader, val_loader, test_loader
