#!/usr/bin/env python3
"""
Per-class pixel importance scoring for CK+ dataset using 3x3 patch masking.
Generates per-pixel importance maps by measuring accuracy drop when patches are masked.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from train_full import CKEmotionCNN
from tqdm import tqdm

# ---------------------- Configuration ----------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIR = 'ck_dataset'
MODEL_PATH = 'CK_full_CNN_outputs/best_model.pth'
ACC_FILE = 'CK_full_CNN_outputs/test_results.txt'
OUTPUT_DIR = 'per_class_heatmaps'
PATCH_SIZE = 16
IMAGE_SIZE = 48
PATCH_MASK_VALUE = 0.0
PATCHED_SAMPLE_CLASS = 'happy'
PATCHED_SAMPLE_INDEX = 0
PATCHED_SAMPLE_DIR = 'patched_sample'

# Class names for CK+ dataset
CLASS_NAMES = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise']

# ---------------------- Model Setup ----------------------
print(f"🖥️  Using device: {DEVICE}")
print("🔧 Loading trained model...")
model = CKEmotionCNN(num_classes=len(CLASS_NAMES)).to(DEVICE)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print("✅ Model loaded successfully")

# ---------------------- Baseline Accuracies ----------------------
print("\n📊 Parsing baseline per-class accuracies...")
baseline_accuracies = {}
with open(ACC_FILE, 'r') as f:
    in_section = False
    for line in f:
        if 'Per-class Test Accuracy:' in line:
            in_section = True
            continue
        if in_section and ':' in line and '/' in line:
            # Parse line like: "anger          :  14/ 16 =  87.50%"
            parts = line.split(':')
            if len(parts) == 2:
                class_name = parts[0].strip()
                if class_name in CLASS_NAMES:
                    acc_str = parts[1].split('=')[1].strip().rstrip('%')
                    baseline_accuracies[class_name] = float(acc_str)

print("Baseline accuracies:")
for cls, acc in baseline_accuracies.items():
    print(f"  {cls:15s}: {acc:.2f}%")

# ---------------------- Load Test Images & Group by Class ----------------------
print("\n📂 Loading test images by class...")
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def group_by_class(data_dir, transform, class_names):
    """Load and group images by class"""
    class_to_images = {cls: [] for cls in class_names}
    
    for cls in class_names:
        class_dir = os.path.join(data_dir, cls)
        if not os.path.exists(class_dir):
            print(f"⚠️  Warning: {class_dir} not found")
            continue
            
        for fname in os.listdir(class_dir):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(class_dir, fname)
                img = plt.imread(img_path)
                
                # Normalize to [0, 1] if needed
                if img.max() > 1.0:
                    img = img / 255.0
                
                # Convert to PIL Image
                img = transforms.ToPILImage()(img) if isinstance(img, np.ndarray) else img
                img = transform(img)
                class_to_images[cls].append(img)
    
    return class_to_images

def denormalize_image(img_array):
    """Convert normalized [-1, 1] grayscale values back to [0, 1]."""
    return np.clip(img_array * 0.5 + 0.5, 0.0, 1.0)

def save_patched_sample_grid(class_to_images, class_name, patch_size, output_dir, sample_index=0):
    """Save a 3x3 grid of occluded images for the requested class."""
    images = class_to_images.get(class_name, [])
    if len(images) == 0:
        print(f"⚠️  Cannot create patched sample grid: no images found for '{class_name}'")
        return
    if sample_index >= len(images):
        print(f"⚠️  Sample index {sample_index} out of bounds for class '{class_name}' (only {len(images)} images)")
        return
    sample_tensor = images[sample_index].clone()
    sample_tensor = sample_tensor.squeeze(0)  # remove channel dimension
    image_np = sample_tensor.cpu().numpy()
    height, width = image_np.shape
    if patch_size > min(height, width):
        print(f"⚠️  Patch size {patch_size} too large for sample image of size {height}x{width}")
        return
    row_offsets = [0, max(0, (height - patch_size) // 2), max(0, height - patch_size)]
    col_offsets = [0, max(0, (width - patch_size) // 2), max(0, width - patch_size)]
    position_labels = [
        'upper left', 'upper middle', 'upper right',
        'middle left', 'middle middle', 'middle right',
        'lower left', 'lower middle', 'lower right'
    ]
    patched_images = []
    for r_idx, i in enumerate(row_offsets):
        for c_idx, j in enumerate(col_offsets):
            patched = image_np.copy()
            patched[i:i+patch_size, j:j+patch_size] = PATCH_MASK_VALUE
            label = position_labels[r_idx * 3 + c_idx]
            patched_images.append((patched, label))
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(3, 3, figsize=(6, 6))
    for ax, (img, label) in zip(axes.flatten(), patched_images):
        ax.imshow(denormalize_image(img), cmap='gray', vmin=0.0, vmax=1.0)
        ax.set_title(label, fontsize=8)
        ax.axis('off')
    fig.tight_layout()
    output_path = os.path.join(output_dir, f"{class_name}_patched_subgrid.png")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"🧩 Saved patched sample grid to {output_path}")

class_to_images = group_by_class(DATA_DIR, transform, CLASS_NAMES)

# Print counts
for cls, images in class_to_images.items():
    print(f"  {cls:15s}: {len(images):3d} images")

print(f"\n🧪 Creating patched sample grid for '{PATCHED_SAMPLE_CLASS}' class...")
save_patched_sample_grid(
    class_to_images=class_to_images,
    class_name=PATCHED_SAMPLE_CLASS,
    patch_size=PATCH_SIZE,
    output_dir=PATCHED_SAMPLE_DIR,
    sample_index=PATCHED_SAMPLE_INDEX
)

# ---------------------- Evaluate Patch Drop ----------------------
def evaluate_patch_drop(model, images, labels, top_left, patch_size):
    """
    Mask a patch and evaluate accuracy.
    
    Args:
        model: Trained model
        images: Batch of images [N, C, H, W]
        labels: True labels [N]
        top_left: (i, j) top-left corner of patch
        patch_size: Size of square patch to mask
    
    Returns:
        Accuracy percentage after masking
    """
    i, j = top_left
    model.eval()
    with torch.no_grad():
        masked_images = images.clone()
        # Mask patch to the configured fill value in normalized space
        masked_images[:, 0, i:i+patch_size, j:j+patch_size] = PATCH_MASK_VALUE
        outputs = model(masked_images.to(DEVICE))
        _, predicted = torch.max(outputs, 1)
        correct = (predicted.cpu() == labels).sum().item()
    return 100.0 * correct / len(images)

# ---------------------- Main Loop ----------------------
print(f"\n🔍 Computing per-pixel importance maps using {PATCH_SIZE}x{PATCH_SIZE} patches...")
importance_maps = {}

for cls in tqdm(CLASS_NAMES, desc='Processing classes'):
    if len(class_to_images[cls]) == 0:
        print(f"\n⚠️  Skipping {cls}: no images found")
        continue
    
    print(f"\n📊 Processing class: {cls}")
    images = torch.stack(class_to_images[cls])
    labels = torch.tensor([CLASS_NAMES.index(cls)] * len(images))
    
    # Initialize pixel score arrays
    pixel_scores = np.zeros((IMAGE_SIZE, IMAGE_SIZE))
    pixel_counts = np.zeros((IMAGE_SIZE, IMAGE_SIZE))
    
    # Scan all possible patch positions
    max_pos = IMAGE_SIZE - PATCH_SIZE + 1
    for i in tqdm(range(max_pos), desc=f'  {cls} rows', leave=False):
        for j in range(max_pos):
            # Evaluate accuracy drop when this patch is masked
            acc = evaluate_patch_drop(model, images, labels, top_left=(i, j), patch_size=PATCH_SIZE)
            drop = baseline_accuracies[cls] - acc
            
            # Distribute the drop score to all pixels in the patch
            for di in range(PATCH_SIZE):
                for dj in range(PATCH_SIZE):
                    pixel_scores[i+di, j+dj] += drop
                    pixel_counts[i+di, j+dj] += 1
    
    # Average scores per pixel (divide by number of patches covering each pixel)
    with np.errstate(divide='ignore', invalid='ignore'):
        final_scores = np.divide(pixel_scores, pixel_counts)
        final_scores[np.isnan(final_scores)] = 0.0
    
    # Normalize to [0, 1]
    final_scores -= final_scores.min()
    if final_scores.max() > 0:
        final_scores /= final_scores.max()
    
    importance_maps[cls] = final_scores
    print(f"  ✅ {cls}: min={final_scores.min():.4f}, max={final_scores.max():.4f}, mean={final_scores.mean():.4f}")

# ---------------------- Save Heatmaps ----------------------
print(f"\n💾 Saving heatmaps to {OUTPUT_DIR}/")
os.makedirs(OUTPUT_DIR, exist_ok=True)

for cls in CLASS_NAMES:
    if cls not in importance_maps:
        continue
    
    # Save as PNG
    plt.figure(figsize=(8, 8))
    plt.imshow(importance_maps[cls], cmap='hot', interpolation='nearest')
    plt.title(f"Class '{cls}' Per-Pixel Importance Map\n(from {PATCH_SIZE}x{PATCH_SIZE} patches)")
    plt.colorbar(label='Importance Score')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{cls}_heatmap.png"), dpi=150)
    plt.close()
    
    # Save as NPY
    np.save(os.path.join(OUTPUT_DIR, f"{cls}_heatmap.npy"), importance_maps[cls])
    
    print(f"  ✅ Saved {cls}_heatmap.png and {cls}_heatmap.npy")

print(f"\n✅ All class-wise per-pixel importance maps saved to {OUTPUT_DIR}/")
print(f"   - {len(importance_maps)} heatmaps generated")
print(f"   - Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
print(f"   - Patch size: {PATCH_SIZE}x{PATCH_SIZE}")
