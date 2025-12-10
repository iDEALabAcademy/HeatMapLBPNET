import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import random
from torchvision import transforms

from train_full import CKEmotionCNN

# ---------------------- Helpers ----------------------
def denormalize_tensor(tensor):
    """Convert normalized [-1, 1] tensor to numpy array in [0, 1]."""
    return tensor.mul(0.5).add(0.5).clamp(0.0, 1.0)

# ---------------------- Dataset / Model Metadata ----------------------
CLASS_NAMES = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise']
AVG_CLASSES = [cls for cls in CLASS_NAMES if cls != 'ignore']
IMAGE_SIZE = 48
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIR = os.path.join(os.path.dirname(__file__), "ck_dataset")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "CK_full_CNN_outputs", "best_model.pth")

# ---------------------- Configuration ----------------------
THRESHOLD_PERCENT = 40  # keep top-X% pixels
HEATMAP_DIR = os.path.join(os.path.dirname(__file__), "per_class_heatmaps")
GLOBAL_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "global_heatmap_avg")
SAMPLE_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "sample_masked_images_global_avg")

# ---------------------- Load Per-Class Heatmaps ----------------------
heatmaps = []
for cls in AVG_CLASSES:
    path = os.path.join(HEATMAP_DIR, f"{cls}_heatmap.npy")
    heatmap = np.load(path)  # shape: (48, 48)
    heatmaps.append(heatmap)
    print(f"  ✓ Loaded heatmap for '{cls}'")


# Compute global average heatmap
global_heatmap = np.mean(heatmaps, axis=0)

# Normalize to [0, 1] before thresholding
global_heatmap -= global_heatmap.min()
if global_heatmap.max() > 0:
    global_heatmap /= global_heatmap.max()

# ---------------------- Save Global Heatmap ----------------------
os.makedirs(GLOBAL_OUTPUT_DIR, exist_ok=True)

plt.imshow(global_heatmap, cmap='hot')
plt.title("Global Per-Pixel Importance Heatmap")
plt.colorbar()
plt.tight_layout()
plt.savefig(os.path.join(GLOBAL_OUTPUT_DIR, "global_avg_heatmap.png"))
plt.close()

np.save(os.path.join(GLOBAL_OUTPUT_DIR, "global_avg_heatmap.npy"), global_heatmap)

# ---------------------- Create Binary Mask ----------------------
flat_heatmap = global_heatmap.flatten()
threshold_value = np.percentile(flat_heatmap, 100 - THRESHOLD_PERCENT)
binary_mask = (global_heatmap >= threshold_value).astype(np.float32)
np.save(os.path.join(GLOBAL_OUTPUT_DIR, "global_avg_binary_mask.npy"), binary_mask)

plt.imshow(binary_mask, cmap='gray')
plt.title(f"Global Binary Mask (Top {THRESHOLD_PERCENT}% pixels)")
plt.tight_layout()
plt.savefig(os.path.join(GLOBAL_OUTPUT_DIR, "global_avg_binary_mask.png"))
plt.close()

# ---------------------- Load Model ----------------------
print("\n🧠 Loading CK+ classifier...")
model = CKEmotionCNN(num_classes=len(CLASS_NAMES)).to(DEVICE)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# ---------------------- Load All Test Images ----------------------
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Group test images by class
class_to_images = {cls: [] for cls in CLASS_NAMES}
for cls in CLASS_NAMES:
    class_dir = os.path.join(DATA_DIR, cls)
    if not os.path.isdir(class_dir):
        print(f"⚠️  Warning: missing class directory {class_dir}")
        continue
    for fname in os.listdir(class_dir):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(class_dir, fname)
            img = plt.imread(img_path)
            if img.max() > 1.0:
                img = img / 255.0
            img = transforms.ToPILImage()(img) if isinstance(img, np.ndarray) else img
            img = transform(img)
            class_to_images[cls].append(img)

# Flatten all test images and labels
all_images = []
all_labels = []
for cls in CLASS_NAMES:
    all_images.extend(class_to_images[cls])
    all_labels.extend([CLASS_NAMES.index(cls)] * len(class_to_images[cls]))

if len(all_images) == 0:
    raise RuntimeError("No images loaded from CK+ dataset. Check DATA_DIR.")

all_images = torch.stack(all_images)
all_labels = torch.tensor(all_labels)

# ---------------------- Evaluate on Masked Test Set (with Per-Class Accuracy) ----------------------
print("\nEvaluating on globally masked test set...")

binary_mask_tensor = torch.tensor(binary_mask).unsqueeze(0).to(DEVICE)  # shape: (1, 48, 48)
total_correct = 0
total_samples = 0

# Per-class tracking
class_correct = {cls: 0 for cls in CLASS_NAMES}
class_total = {cls: 0 for cls in CLASS_NAMES}

model.eval()
with torch.no_grad():
    batch_size = 128
    for i in range(0, len(all_images), batch_size):
        images = all_images[i:i+batch_size].to(DEVICE)
        labels = all_labels[i:i+batch_size].to(DEVICE)
        masked_images = images.clone() * binary_mask_tensor
        outputs = model(masked_images)
        _, predicted = torch.max(outputs, 1)
        total_correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)
        for k in range(labels.size(0)):
            label = labels[k].item()
            pred = predicted[k].item()
            class_total[CLASS_NAMES[label]] += 1
            if pred == label:
                class_correct[CLASS_NAMES[label]] += 1

# Print overall accuracy
acc = 100 * total_correct / total_samples
print(f"\nGlobal masked test accuracy (top {THRESHOLD_PERCENT}% pixels): {acc:.2f}%")

# Print per-class accuracy
print("\n--- Per-Class Accuracy ---")
for cls in CLASS_NAMES:
    if class_total[cls] > 0:
        acc_cls = 100 * class_correct[cls] / class_total[cls]
    else:
        acc_cls = 0.0
    print(f"Class {cls}: {acc_cls:.2f}%")

# ---------------------- Save One Masked Sample per Class ----------------------
print("\nSaving one masked sample per class...")
os.makedirs(SAMPLE_OUTPUT_DIR, exist_ok=True)
binary_mask_tensor_cpu = binary_mask_tensor.cpu()

for cls in CLASS_NAMES:
    if len(class_to_images[cls]) == 0:
        continue
    img = random.choice(class_to_images[cls])  # shape: (1, 48, 48)
    masked_img = img.clone() * binary_mask_tensor_cpu
    display_img = denormalize_tensor(masked_img.squeeze(0)).numpy()
    plt.imshow(display_img, cmap='gray', vmin=0.0, vmax=1.0)
    plt.title(f"Class {cls} - Global Masked")
    plt.axis('off')
    save_path = os.path.join(SAMPLE_OUTPUT_DIR, f"class_{cls}_masked.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

print("Saved random masked samples to 'sample_masked_images_global_avg/' directory.")
