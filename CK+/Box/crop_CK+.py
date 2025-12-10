import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms

# ========================== Configuration ==========================
DATA_ROOT = "/home/sgram/Heatmap/CK+/ck_dataset"
BBOX_PATH = "/home/sgram/Heatmap/CK+/Box/avg_top40_CK+/bbox.npy"
NORMALIZE = True
BATCH_SIZE = 128
NUM_WORKERS = 4
PIN_MEMORY = True
PERSISTENT_WORKERS = False
DEBUG_SAVE_SAMPLES = 5
DEBUG_DIR = "ck+_crop_debug"
SEED = 42
# ===================================================================

class CKPlusCropExact(Dataset):
    """
    CK+ dataset wrapper:
      - Convert to grayscale (1×H×W)
      - Exact bbox crop (inclusive)
      - Normalize to [-1,1] (optional)
    Output tensors have shape [1, h_box, w_box]
    """
    def __init__(self, base_ds, bbox, normalize=True):
        self.base = base_ds
        self.bbox = tuple(int(x) for x in bbox)
        self.normalize = normalize
        self.to_tensor = transforms.ToTensor()
        self.norm = transforms.Normalize(mean=[0.5], std=[0.5]) if normalize else transforms.Lambda(lambda x: x)
        t, l, b, r = self.bbox
        self.h_box = b - t + 1
        self.w_box = r - l + 1

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        pil_gray = img.convert("L")
        img_u8 = np.array(pil_gray, dtype=np.uint8)
        t, l, b, r = self.bbox
        cropped = img_u8[t:b+1, l:r+1]
        x = self.to_tensor(Image.fromarray(cropped, mode="L"))
        x = self.norm(x)
        return x, label

def save_debug_examples(base_ds, wrapper, bbox, out_dir, n=5):
    os.makedirs(out_dir, exist_ok=True)
    n = min(n, len(wrapper))
    idxs = np.linspace(0, len(wrapper) - 1, n, dtype=int)
    for i, idx in enumerate(idxs):
        img, lab = base_ds[idx]
        raw_gray = np.array(img.convert("L"))
        t, l, b, r = bbox
        # Save original with bbox
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.imshow(raw_gray, cmap="gray", interpolation="nearest")
        rect = plt.Rectangle((l - 0.5, t - 0.5), (r - l + 1), (b - t + 1),
                             edgecolor="red", linewidth=2, fill=False)
        ax.add_patch(rect)
        ax.set_title(f"Orig {idx} Label {lab}")
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{i:02d}_orig_label{lab}.png"), bbox_inches="tight")
        plt.close(fig)
        # Save cropped
        x, y = wrapper[idx]
        x01 = (x * 0.5 + 0.5) if wrapper.normalize else x
        arr = (x01.squeeze(0).numpy() * 255).astype(np.uint8)
        Image.fromarray(arr, mode="L").save(os.path.join(out_dir, f"{i:02d}_cropped_exact_label{y}.png"))

def build_ckplus_loaders(
    data_root=DATA_ROOT,
    bbox_path=BBOX_PATH,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    normalize=NORMALIZE,
    debug_save_samples=DEBUG_SAVE_SAMPLES,
    debug_dir=DEBUG_DIR,
    seed=SEED,
):
    bbox = np.load(os.path.abspath(bbox_path)).astype(int).tolist()  # [t,l,b,r]
    t, l, b, r = bbox
    # Use ImageFolder for CK+
    base_ds = datasets.ImageFolder(root=data_root, transform=transforms.Lambda(lambda x: x))
    wrapped = CKPlusCropExact(base_ds, tuple(bbox), normalize=normalize)
    # Split into train (70%), val (15%), test (15%)
    total_size = len(wrapped)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    train_ds, val_ds, test_ds = random_split(
        wrapped, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )
    if debug_save_samples > 0:
        save_debug_examples(base_ds, wrapped, tuple(bbox), debug_dir, n=debug_save_samples)
    common = dict(batch_size=batch_size, num_workers=num_workers,
                  pin_memory=PIN_MEMORY, persistent_workers=PERSISTENT_WORKERS)
    return {
        "train": DataLoader(train_ds, shuffle=True, **common),
        "val": DataLoader(val_ds, shuffle=False, **common),
        "test": DataLoader(test_ds, shuffle=False, **common),
    }

if __name__ == "__main__":
    loaders = build_ckplus_loaders()
    for name in ["train", "val", "test"]:
        xb, yb = next(iter(loaders[name]))
        print(f"{name}: images {tuple(xb.shape)}, labels {tuple(yb.shape)}")
    print(f"\nData root: {DATA_ROOT}")
    print(f"BBox path: {BBOX_PATH}")
    print(" Using CK+ ImageFolder. Exact-box cropping only (no squaring/resizing).")
