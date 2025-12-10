import kagglehub
import os
import shutil
from pathlib import Path
from PIL import Image

# Set download path to current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(current_dir, "ck_dataset")

# Download latest version
print(f"Downloading CK+ dataset to {dataset_dir}...")
path = kagglehub.dataset_download("shuvoalok/ck-dataset")

print(f"Dataset downloaded to cache: {path}")

# Move/copy to CK+ folder if not already there
if os.path.abspath(path) != os.path.abspath(dataset_dir):
    print(f"Moving dataset to {dataset_dir}...")
    if os.path.exists(dataset_dir):
        shutil.rmtree(dataset_dir)
    shutil.copytree(path, dataset_dir)
    print(f"✅ Dataset ready at: {dataset_dir}")
    
    # Delete from cache to save space
    print(f"Removing cache at {path}...")
    try:
        shutil.rmtree(path)
        print("✅ Cache cleaned")
    except Exception as e:
        print(f"⚠️  Could not remove cache: {e}")
else:
    print(f"✅ Dataset already at: {dataset_dir}")

# Save debug images (1 per class)
debug_dir = os.path.join(current_dir, "debug_images")
os.makedirs(debug_dir, exist_ok=True)
print(f"\n📸 Saving debug images to {debug_dir}...")

# Find all class folders
class_folders = [d for d in Path(dataset_dir).iterdir() if d.is_dir()]
class_folders.sort()

for class_folder in class_folders:
    class_name = class_folder.name
    # Find first image in this class
    image_files = list(class_folder.glob("*.png")) + list(class_folder.glob("*.jpg")) + list(class_folder.glob("*.jpeg"))
    
    if image_files:
        # Take the first image
        src_image = image_files[0]
        dst_image = os.path.join(debug_dir, f"{class_name}_sample.png")
        
        # Copy/convert to PNG
        img = Image.open(src_image)
        img.save(dst_image)
        print(f"  ✅ {class_name}: {src_image.name} -> {dst_image}")
    else:
        print(f"  ⚠️  {class_name}: No images found")

print(f"\n✅ Debug images saved in {debug_dir}")