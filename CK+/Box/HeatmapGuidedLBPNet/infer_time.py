#!/usr/bin/env python3
"""
Inference script for LBPNet on SVHN with toggleable variants.

Toggles:
  - USE_CROP:           use bbox→29x18 cropped model vs 32x32 full model
  - USE_ADAPTIVE_P:     use adaptive-P (needs heatmap) vs fixed-P only
  - USE_P_DECAY_CHANS:  use the "P-decay per stage" architecture vs baseline

The actual config is always taken from train_original_model.get_config()
by selecting a MODEL_PRESET that matches the toggle combination.
"""

# ============================================================
# ================== USER CONFIGURATION ======================
# ============================================================

IMAGE_PATH = "/home/sgram/Heatmap/SVHN/Box/binary_Ding_P_new/inference_input_images/183.png"
DEBUG_DIR  = "/home/sgram/Heatmap/SVHN/Box/binary_Ding_P_new/inference_debug"
DEVICE_INDEX = 1          # preferred GPU index
BASE_SIZE    = 32         # 32x32 SVHN grid before bbox

# ---- Architecture / training variant toggles ----
USE_CROP            = True   # True  -> cropped 29x18 model; False -> 32x32 full model
USE_ADAPTIVE_P      = True   # True  -> uses adaptive-P + heatmap; False -> fixed P only
USE_P_DECAY_CHANNELS = True  # True  -> preset with P-decay per stage; False -> baseline P schedule

# ---- Mapping from (toggles) -> preset name + paths ----
# Fill in the entries you actually trained.
PRESET_MAP = {
    # Example: cropped + adaptive-P + P-decay (your current main model)
    (True, True, True): {
        "MODEL_PRESET": "paper_svhn_rp_cropped_P",
        "CKPT_PATH": "/home/sgram/Heatmap/SVHN/Box/binary_Ding_P_new/outputs_svhn_cropped_hm_P_decay_less_channel/best_model.pth",
        "BBOX_PATH": "/home/sgram/Heatmap/SVHN/Box/avg_top40_SVHN/bbox.npy",
        "HEATMAP_PATH": "/home/sgram/Heatmap/SVHN/Box/avg_top40_SVHN/cropped_heatmaps/heatmap_crop.npy",
    },

    # TODO: add other combos when you have them trained, e.g.:
    # (True, False, True): {...}   # cropped, no adaptive-P, P-decay
    # (True, True, False): {...}   # cropped, adaptive-P, no P-decay
    # (False, False, False): {...} # 32x32 baseline, etc.
    (False, False, False): {
        "MODEL_PRESET": "paper_svhn_rp",
        "CKPT_PATH": "/home/sgram/Heatmap/SVHN/Box/binary_Ding/outputs_svhn_original/best_model.pth",
        "BBOX_PATH": "/home/sgram/Heatmap/SVHN/Box/avg_top40_SVHN/bbox.npy",
        "HEATMAP_PATH": "/home/sgram/Heatmap/SVHN/Box/avg_top40_SVHN/cropped_heatmaps/heatmap_crop.npy",
    },
    
}

# ============================================================

import os
import time
import math
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast

import sys
import importlib.util

def dynamic_get_config():
    if USE_CROP:
        model_path = "/home/sgram/Heatmap/SVHN/Box/binary_Ding_P_new/train_original_model_cropped.py"
        module_name = "train_original_model_cropped"
    else:
        model_path = "/home/sgram/Heatmap/SVHN/Box/binary_Ding/train_original_model.py"
        module_name = "train_original_model"
    spec = importlib.util.spec_from_file_location(module_name, model_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.get_config

get_config = dynamic_get_config()
from lbpnet.models import build_model


# ============================================================
# ====================== BBOX HANDLING =======================
# ============================================================

def load_bbox_svhn(bbox_path: str, img_w: int, img_h: int):
    """
    SVHN bbox format: [top, left, height, width]
    Returns (x0, y0, x1, y1) clamped to image bounds.
    """
    bbox = np.load(bbox_path, allow_pickle=True)

    if isinstance(bbox, np.ndarray) and bbox.dtype == object and bbox.shape == ():
        item = bbox.item()
        if isinstance(item, dict) and "bbox" in item:
            bbox = np.array(item["bbox"])
        else:
            raise ValueError(f"bbox.npy dict has no 'bbox' key: {item}")

    bbox = np.array(bbox, dtype=float).reshape(-1)
    if bbox.size != 4:
        raise ValueError(f"Expected 4 entries in bbox.npy, got {bbox}")

    top, left, height, width = bbox
    top = int(round(top))
    left = int(round(left))
    height = int(round(height))
    width = int(round(width))

    x0 = left
    y0 = top
    x1 = left + width
    y1 = top + height

    x0 = max(0, min(x0, img_w - 1))
    y0 = max(0, min(y0, img_h - 1))
    x1 = max(x0 + 1, min(x1, img_w))
    y1 = max(y0 + 1, min(y1, img_h))

    return x0, y0, x1, y1


# ============================================================
# =================== IMAGE PREPROCESSING ====================
# ============================================================

def preprocess_image(
    image_path: str,
    bbox_path: str,
    crop_h: int,
    crop_w: int,
    save_debug_path: str,
    base_size: int = 32,
    use_crop: bool = True,
):
    """
    Pipeline:
      - Load original PNG (any size / aspect)
      - Convert to grayscale
      - If already 32x32: use directly
        Else:
          * aspect-ratio preserving resize so min(H,W)=base_size
          * center-crop 32x32 patch
      - If use_crop=True:
          * apply SVHN bbox on this 32x32 patch
          * resize bbox crop to (crop_h, crop_w) if needed
        Else:
          * use 32x32 patch directly (and make sure crop_h=crop_w=32 in config)
      - Normalize to [-1, 1]
      - Return [1,1,H,W] tensor and preprocessing time

    Also prints:
      - original size
      - 32x32 patch size
      - final model-input size
    """
    t_pre_start = time.time()

    # 1) Load original image
    img_original = Image.open(image_path)
    orig_w, orig_h = img_original.size
    print(f"[DEBUG] Original image size: {orig_h}x{orig_w}")

    # Convert to grayscale
    img_gray = img_original.convert("L")

    # 2) Get a 32x32 patch on the SVHN grid
    if orig_w == base_size and orig_h == base_size:
        img_32 = img_gray
        print("[DEBUG] Image is already 32x32; no resize/crop before bbox.")
    else:
        min_side = min(orig_w, orig_h)
        scale = base_size / float(min_side)
        new_w = int(math.ceil(orig_w * scale))
        new_h = int(math.ceil(orig_h * scale))
        img_resized = img_gray.resize((new_w, new_h), resample=Image.BILINEAR)
        print(f"[DEBUG] After aspect-preserving resize: {new_h}x{new_w}")

        cx = new_w // 2
        cy = new_h // 2
        x0 = cx - base_size // 2
        y0 = cy - base_size // 2
        x0 = max(0, min(x0, new_w - base_size))
        y0 = max(0, min(y0, new_h - base_size))
        x1 = x0 + base_size
        y1 = y0 + base_size
        print(f"[DEBUG] Center-crop box on resized: x0={x0}, y0={y0}, x1={x1}, y1={y1}")
        img_32 = img_resized.crop((x0, y0, x1, y1))

    w32, h32 = img_32.size
    print(f"[DEBUG] Size after 32x32 step (image used for bbox): {h32}x{w32}")

    # Optionally save the 32x32 patch for debugging
    os.makedirs(os.path.dirname(save_debug_path), exist_ok=True)
    img_32.save(os.path.join(os.path.dirname(save_debug_path), "cropped_32x32.png"))


    # 3) Apply bbox-based crop (if using cropped model)
    if use_crop:
        x0, y0, x1, y1 = load_bbox_svhn(bbox_path, w32, h32)
        crop = img_32.crop((x0, y0, x1, y1))
        if crop.size != (crop_w, crop_h):
            crop = crop.resize((crop_w, crop_h), resample=Image.BILINEAR)
        # Save the cropped 29x18 image
        cropped_29x18_path = os.path.join(os.path.dirname(save_debug_path), "cropped_29x18.png")
        crop.save(cropped_29x18_path)
        print(f"[DEBUG] Saved cropped 29x18 image to: {cropped_29x18_path}")
        print(f"[DEBUG] Final cropped size (model input): {crop.size[1]}x{crop.size[0]}")
    else:
        crop = img_32
        if crop.size != (crop_w, crop_h):
            crop = crop.resize((crop_w, crop_h), resample=Image.BILINEAR)
        print(f"[DEBUG] Using full 32x32 (or resized) as model input: {crop.size[1]}x{crop.size[0]}")

    # 4) Save final input image
    crop.save(save_debug_path)
    print(f"[DEBUG] Saved model-input image to: {save_debug_path}")

    # 5) Normalize to [-1,1]
    arr = np.array(crop, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)

    t_pre_end = time.time()
    preprocess_time = (t_pre_end - t_pre_start) * 1000.0
    return tensor, preprocess_time


# ============================================================
# ==================== MODEL LOADING =========================
# ============================================================

def load_model(config, ckpt_path, device, heatmap_path: str, use_adaptive_p: bool):
    """
    Build model from config, attach/remove heatmap_path depending on
    use_adaptive_p, and load weights with BN-safe dummy forward.
    """
    lbp_cfg = config.get("lbp_layer", {})
    if use_adaptive_p:
        if heatmap_path is None or heatmap_path == "":
            raise ValueError("USE_ADAPTIVE_P=True but no HEATMAP_PATH provided")
        lbp_cfg["heatmap_path"] = heatmap_path
        os.environ["GLOBAL_HEATMAP_PATH"] = heatmap_path
    else:
        lbp_cfg.pop("heatmap_path", None)
        os.environ.pop("GLOBAL_HEATMAP_PATH", None)
    config["lbp_layer"] = lbp_cfg

    model = build_model(config).to(device)

    img_size = config.get("image_size", [29, 18])
    if isinstance(img_size, (list, tuple)) and len(img_size) == 2:
        H, W = int(img_size[0]), int(img_size[1])
    else:
        H = W = int(img_size)

    model.eval()
    with torch.no_grad():
        dummy = torch.zeros(1, 1, H, W, device=device)
        _ = model(dummy)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    raw_state = ckpt.get("model_state_dict", ckpt)
    incompat = model.load_state_dict(raw_state, strict=False)
    print(f"Loaded checkpoint '{ckpt_path}'")
    print("  missing:", len(incompat.missing_keys))
    print("  unexpected:", len(incompat.unexpected_keys))

    if hasattr(model, "set_ste"):
        model.set_ste(True, True)

    return model


# ============================================================
# ========================== MAIN ============================
# ============================================================

def main():
    # ---- Choose preset based on toggles ----
    key = (USE_CROP, USE_ADAPTIVE_P, USE_P_DECAY_CHANNELS)
    if key not in PRESET_MAP:
        raise ValueError(f"No PRESET_MAP entry for toggles {key}. "
                         f"Add one in PRESET_MAP with MODEL_PRESET + paths.")

    preset_cfg = PRESET_MAP[key]
    model_preset = preset_cfg["MODEL_PRESET"]
    ckpt_path    = preset_cfg["CKPT_PATH"]
    bbox_path    = preset_cfg.get("BBOX_PATH", "")
    heatmap_path = preset_cfg.get("HEATMAP_PATH", "")

    # Share preset selection with train_config logic
    os.environ["MODEL_PRESET"] = model_preset

    # ---- Device ----
    device = torch.device("cuda:1")
    print(f"Using device: {device}")

    # ---- Config & model ----
    config = get_config()
    image_size = config["image_size"]
    if isinstance(image_size, (list, tuple)) and len(image_size) == 2:
        H, W = int(image_size[0]), int(image_size[1])
    else:
        H = W = int(image_size)

    print(f"Model preset: {config.get('_source', model_preset)}")
    print(f"Model expects input size: {H}x{W}")
    print(f"Setup: Cropped={'Yes' if USE_CROP else 'No'}, Adaptive_P={'Yes' if USE_ADAPTIVE_P else 'No'}, P_decay={'Yes' if USE_P_DECAY_CHANNELS else 'No'}")
    print(f"USE_CROP={USE_CROP}, USE_ADAPTIVE_P={USE_ADAPTIVE_P}, USE_P_DECAY_CHANNELS={USE_P_DECAY_CHANNELS}")

    model = load_model(config, ckpt_path, device, heatmap_path, use_adaptive_p=USE_ADAPTIVE_P)

    debug_path = os.path.join(DEBUG_DIR, "preprocessed_input.png")

    # ---- Preprocess ----
    img_tensor, preprocess_time = preprocess_image(
        IMAGE_PATH,
        bbox_path,
        crop_h=H,
        crop_w=W,
        save_debug_path=debug_path,
        base_size=BASE_SIZE,
        use_crop=USE_CROP,
    )
    print(f"Preprocess time   : {preprocess_time:.3f} ms")

    img_tensor = img_tensor.to(device)

    # ---- Inference timing (forward only) ----
    with torch.no_grad():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.time()

        with autocast(enabled=(config.get("amp", False) and device.type == "cuda")):
            logits = model(img_tensor)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t1 = time.time()

    inference_ms = (t1 - t0) * 1000.0
    probs = F.softmax(logits, dim=1)
    pred = int(probs.argmax(1).item())
    conf = float(probs[0, pred].item())

    print("\n=== INFERENCE RESULT ===")
    print(f"Predicted class : {pred}")
    print(f"Confidence      : {conf:.4f}")
    print(f"Preprocess time : {preprocess_time:.3f} ms")
    print(f"Inference time  : {inference_ms:.3f} ms")
    print(f"Probabilities   : {probs.cpu().numpy()}")


if __name__ == "__main__":
    main()
