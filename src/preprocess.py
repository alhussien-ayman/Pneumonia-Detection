import cv2
import numpy as np
import os
from pathlib import Path


def load_image(image_path: str) -> np.ndarray:
    """Load image from path as grayscale."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")
    return img


def resize_image(img: np.ndarray, size: tuple = (64, 64)) -> np.ndarray:
    """Resize image using bilinear interpolation."""
    return cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)


def normalize_image(img: np.ndarray) -> np.ndarray:
    """Min-max normalize pixel values to [0, 1]."""
    img = img.astype(np.float32)
    min_val, max_val = img.min(), img.max()
    if max_val - min_val == 0:
        return np.zeros_like(img, dtype=np.float32)
    return (img - min_val) / (max_val - min_val)


def preprocess(image_path: str, size: tuple = (64, 64)) -> np.ndarray:
    """Full preprocessing pipeline for a single image."""
    img = load_image(image_path)
    img = resize_image(img, size)
    img = normalize_image(img)
    return img


def load_dataset(data_dir: str, size: tuple = (64, 64)):
    """
    Load all images from train/val/test directory structure.
    Expected structure:
        data_dir/
            NORMAL/    -> label 0
            PNEUMONIA/ -> label 1
    Returns: X (numpy array of preprocessed images), y (labels), paths
    """
    images, labels, paths = [], [], []
    label_map = {"NORMAL": 0, "PNEUMONIA": 1}

    for class_name, label in label_map.items():
        class_dir = Path(data_dir) / class_name
        if not class_dir.exists():
            print(f"Warning: {class_dir} not found, skipping.")
            continue
        for img_file in class_dir.glob("*.jpeg"):
            try:
                img = preprocess(str(img_file), size)
                images.append(img)
                labels.append(label)
                paths.append(str(img_file))
            except Exception as e:
                print(f"Skipping {img_file}: {e}")

    X = np.array(images)
    y = np.array(labels)
    return X, y, paths


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Single image preprocessing
        img = preprocess(sys.argv[1])
        print(f"Preprocessed shape: {img.shape}, range: [{img.min():.3f}, {img.max():.3f}]")
    else:
        # Load and preprocess full dataset
        data_base = Path("data/chest_xray")
        
        for split in ["train", "val", "test"]:
            split_dir = data_base / split
            if split_dir.exists():
                print(f"\nProcessing {split} dataset...")
                X, y, paths = load_dataset(str(split_dir), size=(64, 64))
                print(f"  Loaded {len(X)} images")
                print(f"  Shape: {X.shape}")
                print(f"  Labels: {len([l for l in y if l == 0])} NORMAL, {len([l for l in y if l == 1])} PNEUMONIA")
                print(f"  Pixel range: [{X.min():.3f}, {X.max():.3f}]")
            else:
                print(f"Warning: {split_dir} not found")
