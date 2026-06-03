import tarfile
import pickle
import os
import numpy as np
from PIL import Image
import random

# CIFAR-10 files are already extracted to cifar-10-batches-py/
# No need to extract again

# Find the data directory
print("Looking for CIFAR-10 batch files...")
data_dir = None

# Check common paths
possible_paths = [
    'cifar-10-batches-py',  # Extracted directly to root
    os.path.join('cifar-10-python', 'cifar-10-batches-py'),  # Inside subfolder
    'cifar-10-python'  # Or maybe just at root
]

for path in possible_paths:
    if os.path.exists(os.path.join(path, 'data_batch_1')):
        data_dir = path
        print(f"Found CIFAR-10 data in: {data_dir}")
        break

# If still not found, search recursively
if data_dir is None:
    for root, dirs, files in os.walk('.'):
        if 'data_batch_1' in files:
            data_dir = root
            print(f"Found CIFAR-10 data in: {data_dir}")
            break

if data_dir is None:
    print(f"ERROR: Could not find CIFAR-10 batch files!")
    print(f"Current directory contents:")
    for item in os.listdir('.'):
        if os.path.isdir(item):
            print(f"  [{item}]")
        else:
            print(f"  {item}")
    exit(1)

all_data = []
all_labels = []

# Load all training batches
for batch_num in range(1, 6):
    batch_file = os.path.join(data_dir, f'data_batch_{batch_num}')
    with open(batch_file, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
        all_data.append(batch[b'data'])
        all_labels.append(batch[b'labels'])

# Load test batch too (for more samples)
test_file = os.path.join(data_dir, 'test_batch')
with open(test_file, 'rb') as f:
    test_batch = pickle.load(f, encoding='bytes')
    all_data.append(test_batch[b'data'])
    all_labels.append(test_batch[b'labels'])

# Concatenate all data
all_data = np.concatenate(all_data, axis=0)
all_labels = np.concatenate(all_labels, axis=0)

print(f"Total CIFAR-10 images: {len(all_data)}")

# Class mappings (CIFAR-10 classes)
# 0: airplane, 1: automobile, 2: bird, 3: cat, 4: deer, 5: dog, 6: frog, 7: horse, 8: ship, 9: truck
class_map = {
    3: 'cat',
    5: 'dog',
    7: 'horse'
}

# Create images directory if it doesn't exist
os.makedirs('images', exist_ok=True)

# Extract and save 50 images from each class
for class_idx, class_name in class_map.items():
    print(f"\nProcessing class: {class_name} (index {class_idx})")
    
    # Get indices for this class
    class_indices = np.where(all_labels == class_idx)[0]
    print(f"Found {len(class_indices)} {class_name} images")
    
    # Sample 50 random images
    sample_indices = np.random.choice(class_indices, size=min(50, len(class_indices)), replace=False)
    
    # Save each image
    for i, idx in enumerate(sample_indices, 1):
        # CIFAR-10 data is 3072 = 32*32*3 (flattened RGB)
        img_data = all_data[idx].reshape(3, 32, 32)  # (C, H, W)
        img_array = np.transpose(img_data, (1, 2, 0))  # (H, W, C)
        
        # Convert to PIL Image and save
        img = Image.fromarray(img_array.astype('uint8'), 'RGB')
        img_path = os.path.join('images', f'{class_name}_{i}.jpg')
        img.save(img_path, 'JPEG')
    
    print(f"Saved {len(sample_indices)} {class_name} images to ./images/")

print("\n✓ Done! Images are ready in ./images/ folder")
print("You can now click 'Load CIFAR10 Sample' in the web app to load them for training.")
