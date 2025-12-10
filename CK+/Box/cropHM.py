# crop_ckplus_heatmap.py
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

# ========================== Configuration ==========================
TOP_K_PERCENT = 40  # for folder naming consistency, not used for CK+

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Folder containing bbox.npy (from your CK+ bbox script)
ROI_DIR = os.path.join(BASE_DIR, f"avg_top40_CK+")

# Path to the original continuous heatmap
HEATMAP_PATH = "/home/sgram/Heatmap/CK+/global_heatmap_avg/global_avg_heatmap.npy"

# Output folder inside avg_top40_CK+/cropped_heatmaps/
OUT_DIR = os.path.join(ROI_DIR, "cropped_heatmaps")
os.makedirs(OUT_DIR, exist_ok=True)
# ===================================================================

def load_bbox(bbox_path):
    if not os.path.exists(bbox_path):
        raise FileNotFoundError(f"bbox file not found: {bbox_path}")
    bbox = np.load(bbox_path).astype(np.int32)
    if bbox.shape != (4,):
        raise ValueError(f"bbox.npy must be a 1D array of 4 ints, got {bbox.shape}")
    if np.any(bbox < 0):
        return None
    return tuple(bbox.tolist())  # (t, l, b, r)

def crop_with_inclusive_bbox(arr2d, bbox):
    t, l, b, r = bbox
    return arr2d[t:b+1, l:r+1]

def resize_square_bilinear(arr2d, side):
    x = torch.from_numpy(arr2d).float().unsqueeze(0).unsqueeze(0)
    y = F.interpolate(x, size=(side, side), mode='bilinear', align_corners=False)
    return y.squeeze(0).squeeze(0).numpy()

def save_vis(img2d, path, cmap='hot', title=None, add_colorbar=True):
    plt.figure()
    im = plt.imshow(img2d, cmap=cmap, interpolation='nearest')
    if title:
        plt.title(title)
    plt.axis('off')
    if add_colorbar:
        plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight')
    plt.close()

def main():
    # 1) Load bbox from ROI_DIR
    bbox_path = os.path.join(ROI_DIR, "bbox.npy")
    bbox = load_bbox(bbox_path)

    if bbox is None:
        empty = np.zeros((0, 0), dtype=np.float32)
        np.save(os.path.join(OUT_DIR, "heatmap_crop.npy"), empty)
        np.save(os.path.join(OUT_DIR, "heatmap_crop_square.npy"), empty)
        print(f"No ROI (bbox was [-1,-1,-1,-1]). Saved empty crops in {OUT_DIR}")
        return

    # 2) Load continuous heatmap
    heatmap = np.load(HEATMAP_PATH).astype(np.float32)
    print(f"Loaded heatmap from: {HEATMAP_PATH}")

    if heatmap.ndim != 2:
        raise ValueError(f"Expected 2D heatmap, got shape {heatmap.shape}")

    # 3) Crop using same inclusive box
    heatmap_crop = crop_with_inclusive_bbox(heatmap, bbox)
    h, w = heatmap_crop.shape

    # 4) Also provide a square version (for visual comparability / optional use)
    side = max(h, w)
    heatmap_crop_square = resize_square_bilinear(heatmap_crop, side)

    # 5) Save arrays
    np.save(os.path.join(OUT_DIR, "heatmap_crop.npy"), heatmap_crop)
    np.save(os.path.join(OUT_DIR, "heatmap_crop_square.npy"), heatmap_crop_square)

    # 6) Visualizations
    t, l, b, r = bbox
    fig, ax = plt.subplots()
    ax.imshow(heatmap, cmap='hot', interpolation='nearest')
    ax.add_patch(plt.Rectangle((l - 0.5, t - 0.5),
                               (r - l + 1), (b - t + 1),
                               edgecolor='red', linewidth=2, fill=False))
    ax.set_title(f"Full Heatmap with ROI (CK+)")
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "full_heatmap_with_bbox.png"), bbox_inches='tight')
    plt.close(fig)

    save_vis(heatmap_crop,
             os.path.join(OUT_DIR, f"heatmap_crop_{h}x{w}.png"),
             title=f"Cropped Heatmap ({h}x{w})")

    save_vis(heatmap_crop_square,
             os.path.join(OUT_DIR, f"heatmap_crop_square_{side}x{side}.png"),
             title=f"Square Heatmap ({side}x{side})")

    print(f"\nSaved results in: {OUT_DIR}")
    print(f" - heatmap_crop.npy                shape={heatmap_crop.shape}")
    print(f" - heatmap_crop_square.npy         shape={heatmap_crop_square.shape}")
    print(f" - full_heatmap_with_bbox.png")
    print(f" - heatmap_crop_{h}x{w}.png")
    print(f" - heatmap_crop_square_{side}x{side}.png\n")

if __name__ == "__main__":
    main()
