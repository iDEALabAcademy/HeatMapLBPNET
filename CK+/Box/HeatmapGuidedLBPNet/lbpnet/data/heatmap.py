# heatmap.py
# Lightweight utilities for loading and preparing global (or per-image) heatmaps.

import os
from typing import Optional, Tuple, Literal, Union, Dict

import numpy as np
import torch
import torch.nn.functional as F

# ---- Default path (changeable) ----
DEFAULT_HEATMAP_PATH = "/home/sgram/Heatmap/SVHN/Box/avg_top40_SVHN/cropped_heatmaps/heatmap_crop.npy"

NormalizeMode = Optional[Literal["auto", "yes", "no"]]


def _is_normalized(arr: np.ndarray, eps: float = 1e-6) -> bool:
    """Heuristic: consider normalized if values lie in [0,1] within tolerance."""
    if arr.size == 0:
        return True
    vmin, vmax = float(np.min(arr)), float(np.max(arr))
    return (vmin >= -eps) and (vmax <= 1.0 + eps)


def _minmax_normalize(arr: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Min–max normalize to [0,1]; safe for constant arrays."""
    vmin, vmax = float(np.min(arr)), float(np.max(arr))
    denom = max(vmax - vmin, eps)
    return (arr - vmin) / denom


class HeatmapProvider:
    """
    File-backed heatmap loader with small conveniences:
      - optional (auto) normalization to [0,1] (skipped if already normalized)
      - optional resizing to a target HxW
      - returns numpy or torch tensors; optional batching/channel dims for torch
      - simple in-memory cache to avoid reloading each call
    """

    def __init__(self, path: str = DEFAULT_HEATMAP_PATH):
        self.path = path
        self._cache: Dict[Tuple[str, str], np.ndarray] = {}

    def load_numpy(
        self,
        path: Optional[str] = None,
        normalize: NormalizeMode = "auto",
        expected_shape: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """
        Load heatmap as float32 numpy array of shape (H, W).
        normalize:
          - "auto": normalize only if not already in [0,1]
          - "yes": force min–max normalization
          - "no" or None: never normalize
        """
        hp = path or self.path
        key = (hp, str(normalize))
        if key in self._cache:
            arr = self._cache[key]
        else:
            if not os.path.isfile(hp):
                raise FileNotFoundError(f"Heatmap file not found: {hp}")
            arr = np.load(hp)
            if arr.ndim != 2:
                raise ValueError(f"Expected heatmap with shape (H, W), got {arr.shape}")
            arr = arr.astype(np.float32, copy=False)

            if normalize == "yes":
                arr = _minmax_normalize(arr)
            elif normalize == "auto":
                if not _is_normalized(arr):
                    arr = _minmax_normalize(arr)
            # else: "no" or None → leave as-is

            self._cache[key] = arr

        if expected_shape is not None and tuple(arr.shape) != tuple(expected_shape):
            raise ValueError(f"Heatmap shape {arr.shape} does not match expected {expected_shape}")

        return arr

    def load_torch(
        self,
        path: Optional[str] = None,
        normalize: NormalizeMode = "auto",
        device: Optional[torch.device] = None,
        add_bc: bool = True,
        target_hw: Optional[Tuple[int, int]] = None,
        mode: Literal["bilinear", "nearest"] = "bilinear",
    ) -> torch.Tensor:
        """
        Load heatmap as a torch tensor (float32).
        - add_bc=True → returns shape [1,1,H,W] (broadcast-friendly). If False → [H,W].
        - target_hw: if provided, resizes to (H, W) using F.interpolate (expects [1,1,H,W]).
        """
        arr = self.load_numpy(path=path, normalize=normalize)
        th = torch.from_numpy(arr)  # [H, W], float32

        if add_bc:
            th = th.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]

        if target_hw is not None:
            if th.ndim != 4:
                th = th.unsqueeze(0).unsqueeze(0)  # ensure [1,1,H,W]
            th = F.interpolate(th, size=target_hw, mode=mode, align_corners=False if mode == "bilinear" else None)

        if device is not None:
            th = th.to(device)

        return th


# ---- Convenience top-level functions ----

_global_provider = HeatmapProvider(DEFAULT_HEATMAP_PATH)

def get_heatmap_numpy(
    path: Optional[str] = None,
    normalize: NormalizeMode = "auto",
    expected_shape: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """One-shot numpy loader."""
    return _global_provider.load_numpy(path=path, normalize=normalize, expected_shape=expected_shape)

def get_heatmap_torch(
    path: Optional[str] = None,
    normalize: NormalizeMode = "auto",
    device: Optional[Union[str, torch.device]] = None,
    add_bc: bool = True,
    target_hw: Optional[Tuple[int, int]] = None,
    mode: Literal["bilinear", "nearest"] = "bilinear",
) -> torch.Tensor:
    """One-shot torch loader; returns [1,1,H,W] by default (good for broadcasting)."""
    dev = torch.device(device) if isinstance(device, str) else device
    return _global_provider.load_torch(
        path=path, normalize=normalize, device=dev, add_bc=add_bc, target_hw=target_hw, mode=mode
    )

# ---- Local (pivot-aware) window mean utilities ----

def _ensure_bc1x(t: torch.Tensor) -> torch.Tensor:
    """
    Ensure shape [1,1,H,W]. Accepts [H,W], [1,H,W], or [1,1,H,W].
    """
    if t.ndim == 2:
        return t.unsqueeze(0).unsqueeze(0)
    if t.ndim == 3:
        return t.unsqueeze(0)
    if t.ndim == 4:
        return t
    raise ValueError(f"expected 2D/3D/4D tensor, got {t.shape}")

def window_mean_from_heatmap(
    heatmap: Union[torch.Tensor, None] = None,
    *,
    path: Optional[str] = None,
    normalize: NormalizeMode = "auto",
    device: Optional[Union[str, torch.device]] = None,
    target_hw: Optional[Tuple[int, int]] = None,
    kernel_size: Union[int, Tuple[int, int]] = 5,
    exclude_center: bool = False,
    pad_mode: str = "reflect",
) -> torch.Tensor:
    """
    Compute per-pixel local mean of the heatmap aligned to the pivot (current pixel).

    Returns a tensor of shape [1,1,H,W] where each location is the mean of its KxK
    neighborhood (optionally excluding the pivot). This is 'same' padding behavior.

    Args:
        heatmap: torch tensor of shape [H,W], [1,H,W], or [1,1,H,W]. If None, loads via `path`.
        path: optional .npy path to load when heatmap is None (falls back to DEFAULT_HEATMAP_PATH).
        normalize: passed to loader when using `path` (see load_torch).
        device: move output to this device.
        target_hw: (H,W) to resize heatmap before pooling.
        kernel_size: int or (kh, kw). Odd sizes recommended for true center alignment.
        exclude_center: if True, exclude the pivot from the mean (divide by K*K-1).
        pad_mode: padding for borders; 'reflect' or 'replicate' are good choices.
    """
    # 1) Prepare heatmap tensor [1,1,H,W]
    if heatmap is None:
        th = get_heatmap_torch(path=path, normalize=normalize, device=device, add_bc=True, target_hw=target_hw)
    else:
        th = _ensure_bc1x(heatmap)
        if target_hw is not None:
            th = F.interpolate(th, size=target_hw, mode="bilinear", align_corners=False)
        if device is not None:
            th = th.to(device)

    _, _, H, W = th.shape

    # 2) Kernel + padding
    if isinstance(kernel_size, int):
        kh = kw = int(kernel_size)
    else:
        kh, kw = int(kernel_size[0]), int(kernel_size[1])

    k = torch.ones((1, 1, kh, kw), dtype=th.dtype, device=th.device)
    denom = float(kh * kw)
    if exclude_center:
        cy, cx = kh // 2, kw // 2
        k[0, 0, cy, cx] = 0.0
        denom = max(denom - 1.0, 1.0)  # avoid /0 for 1x1

    pad_h = kh // 2
    pad_w = kw // 2
    th_p = F.pad(th, (pad_w, pad_w, pad_h, pad_h), mode=pad_mode)

    # 3) Convolution → local mean
    local_sum = F.conv2d(th_p, k, stride=1, padding=0)  # [1,1,H,W]
    local_mean = local_sum / denom
    return local_mean

@torch.no_grad()
def adaptive_p_mask_from_mean(
    win_mean: torch.Tensor,
    bins=(0.2, 0.5, 0.7),
    max_points: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Turn a per-pixel window mean (shape [1,1,H,W]) into:
      - P_map: int map in {2,4,6,8} per pixel, shape [1,1,H,W]
      - mask:  binary neighbor mask shape [1,1,8,H,W] where first P bits are 1

    Binning rule:
      mean < bins[0]     -> P=2
      bins[0]<= mean < b1-> P=4
      b1      <= mean < b2-> P=6
      mean >= bins[2]     -> P=8
    """
    if win_mean.ndim != 4 or win_mean.shape[0:2] != (1, 1):
        raise ValueError(f"expected win_mean shape [1,1,H,W], got {tuple(win_mean.shape)}")

    b0, b1, b2 = float(bins[0]), float(bins[1]), float(bins[2])
    H, W = win_mean.shape[-2], win_mean.shape[-1]

    # Compute P_map in {2,4,6,8}
    P_map = torch.empty_like(win_mean, dtype=torch.int64)
    P_map[win_mean <  b0] = 2
    P_map[(win_mean >= b0) & (win_mean <  b1)] = 4
    P_map[(win_mean >= b1) & (win_mean <  b2)] = 6
    P_map[win_mean >= b2] = 8
    P_map.clamp_(2, max_points)

    # Build mask [1,1,8,H,W]: enable first P neighbors in canonical order
    mask = torch.zeros((1, 1, max_points, H, W), dtype=torch.bool, device=win_mean.device)
    # Vectorized fill: for k=0..7, mask[k] = (P_map > k)
    ks = torch.arange(max_points, device=win_mean.device).view(1, 1, max_points, 1, 1)
    mask = (P_map.unsqueeze(2) > ks)

    return mask, P_map
