import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Directory containing heatmaps
heatmap_dir = '/home/sgram/Heatmap/CK+/per_class_heatmaps'

# Class names in order: top row (4), bottom row (3) with happy in the middle of top row
top_row = ['anger', 'contempt', 'sadness', 'disgust']
bottom_row = ['fear', 'happy', 'surprise']
all_classes = top_row + bottom_row

# Load all heatmap numpy arrays
heatmaps = []
for cls in all_classes:
    npy_path = os.path.join(heatmap_dir, f'{cls}_heatmap.npy')
    heatmaps.append(np.load(npy_path))

# Create figure with subplots
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

# Plot top row (4 images)
for i, cls in enumerate(top_row):
    im = axes[0, i].imshow(heatmaps[i], cmap='hot')
    axes[0, i].set_title(cls.capitalize(), fontsize=24, fontweight='bold')
    axes[0, i].axis('off')
    cbar = plt.colorbar(im, ax=axes[0, i], fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=16)

# Plot bottom row (3 images)
for i, cls in enumerate(bottom_row):
    im = axes[1, i].imshow(heatmaps[4 + i], cmap='hot')
    axes[1, i].set_title(cls.capitalize(), fontsize=24, fontweight='bold')
    axes[1, i].axis('off')
    cbar = plt.colorbar(im, ax=axes[1, i], fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=16)

# Hide the last subplot in bottom row
axes[1, 3].axis('off')

plt.tight_layout()

# Save figure
output_path = os.path.join(heatmap_dir, 'all_classes_heatmap_grid.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"Saved combined heatmap visualization to: {output_path}")
