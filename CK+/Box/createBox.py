"""
Purpose:
    Given a global CK+ heatmap mask (binary .npy),
    compute its bounding box and save:
        • bbox.npy  → [top, left, bottom, right]
        • crop.npy  → cropped region of the mask
        • bbox_visualization.png → visualization

Input:
    MASK_PATH: path to global binary mask (e.g. global_avg_binary_mask.npy)
Output:
    ./avg_top40_CK+/  (or any folder name you choose)

"""

import os
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------
# Configuration
# Path to your binary mask file (global CK+ mask)
MASK_PATH = r"/home/sgram/Heatmap/CK+/global_heatmap_avg/global_avg_binary_mask.npy"

# Folder name for saving results
OUTPUT_DIR_NAME = "avg_top40_CK+"

# Optional padding (in pixels) around bounding box
MARGIN = 0
# -------------------------------------------------------

def bbox_from_mask(mask: np.ndarray):
    """Return [top, left, bottom, right] of nonzero region."""
    ys, xs = np.where(mask != 0)
    if ys.size == 0:
        return None
    return int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())

def expand_bbox(bbox, margin, H, W):
    """Expand box by margin but keep within bounds."""
    if margin <= 0:
        return bbox
    t, l, b, r = bbox
    t = max(0, t - margin)
    l = max(0, l - margin)
    b = min(H - 1, b + margin)
    r = min(W - 1, r + margin)
    return t, l, b, r

def draw_bbox(mask, bbox, output_path):
    """Save mask with red rectangle drawn around bbox."""
    t, l, b, r = bbox
    fig, ax = plt.subplots()
    ax.imshow(mask, cmap="gray", interpolation="nearest", vmin=0, vmax=1)
    rect = plt.Rectangle((l - 0.5, t - 0.5),
                         (r - l + 1), (b - t + 1),
                         edgecolor="red", linewidth=2, fill=False)
    ax.add_patch(rect)
    ax.set_title("CK+ Global Mask Bounding Box")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

def main(mask_path, output_dir_name, margin=0):
    # Save outputs inside the folder where the script is running
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, output_dir_name)
    os.makedirs(output_dir, exist_ok=True)

    mask_path = os.path.abspath(mask_path)
    mask = np.load(mask_path)
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape {mask.shape}")

    bbox = bbox_from_mask(mask)
    if bbox is None:
        np.save(os.path.join(output_dir, "bbox.npy"), np.array([-1, -1, -1, -1], dtype=np.int32))
        np.save(os.path.join(output_dir, "crop.npy"), np.zeros((0, 0), dtype=mask.dtype))
        print("No nonzero pixels found — saved empty bbox and crop.")
        return

    H, W = mask.shape
    bbox = expand_bbox(bbox, margin, H, W)
    t, l, b, r = bbox
    crop = mask[t:b+1, l:r+1]

    bbox_out = os.path.join(output_dir, "bbox.npy")
    crop_out = os.path.join(output_dir, "crop.npy")
    image_out = os.path.join(output_dir, "bbox_visualization.png")

    np.save(bbox_out, np.array([t, l, b, r], dtype=np.int32))
    np.save(crop_out, crop)
    draw_bbox(mask, bbox, image_out)

    print(f"\nResults saved in: {output_dir}")
    print(f"   bbox.npy → [top, left, bottom, right] = {(t, l, b, r)}")
    print(f"   crop.npy → shape = {crop.shape}")
    print(f"   bbox_visualization.png → red box visualization\n")

if __name__ == "__main__":
    main(MASK_PATH, output_dir_name=OUTPUT_DIR_NAME, margin=MARGIN)
