"""
preprocess.py
-------------
Image loading and preprocessing pipeline for chest X-ray pneumonia detection.
Handles CLAHE lung masking, image resizing, and dataset loading.
"""

import os
import glob as gb
import numpy as np
import cv2

# Label encoding
LABEL_CODE = {'NORMAL': 0, 'PNEUMONIA': 1}
LABEL_NAMES = {0: 'Normal', 1: 'Pneumonia'}
IMAGE_SIZE = (256, 256)


# ---------------------------------------------------------------------------
# Core image processing functions
# ---------------------------------------------------------------------------

def create_lung_mask(image_gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply CLAHE enhancement followed by Otsu thresholding to isolate lung fields.

    Args:
        image_gray: Grayscale image array (H x W).

    Returns:
        (masked_img, mask): The masked lung region and the binary mask.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(image_gray)

    blurred = cv2.GaussianBlur(equalized, (5, 5), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    masked_img = cv2.bitwise_and(image_gray, image_gray, mask=mask)
    return masked_img, mask


def load_image(img_path: str, size: tuple = IMAGE_SIZE) -> np.ndarray | None:
    """
    Load and resize a grayscale image from disk.

    Args:
        img_path: Path to the JPEG image.
        size: Target (width, height) for resizing.

    Returns:
        Resized grayscale image, or None if the file cannot be read.
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return cv2.resize(img, size)


def preprocess_image(img_path: str) -> np.ndarray | None:
    """
    Full preprocessing for a single image: load → resize → CLAHE mask.

    Args:
        img_path: Path to the JPEG image.

    Returns:
        Masked grayscale image array, or None on failure.
    """
    img = load_image(img_path)
    if img is None:
        return None
    masked_img, _ = create_lung_mask(img)
    return masked_img


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_split(data_path: str, verbose: bool = True) -> tuple[list, list, list]:
    """
    Load all images from a split directory (e.g., train/, test/, val/).

    Expected directory structure:
        data_path/
            NORMAL/
                *.jpeg
            PNEUMONIA/
                *.jpeg

    Args:
        data_path: Path to the split folder.
        verbose: Print progress per class folder.

    Returns:
        (images, paths, labels) — parallel lists of preprocessed images,
        file paths, and integer labels.
    """
    images, paths, labels = [], [], []

    for folder in sorted(os.listdir(data_path)):
        folder_path = os.path.join(data_path, folder)
        if not os.path.isdir(folder_path) or folder not in LABEL_CODE:
            continue

        files = gb.glob(os.path.join(folder_path, '*.jpeg'))
        if verbose:
            print(f'  [{folder}] found {len(files)} images …')

        for file in files:
            img = preprocess_image(file)
            if img is not None:
                images.append(img)
                paths.append(file)
                labels.append(LABEL_CODE[folder])

    return images, paths, labels


def load_dataset(base_dir: str, verbose: bool = True) -> dict:
    """
    Load train, test and val splits from the standard chest_xray directory layout.

    Args:
        base_dir: Root directory containing train/, test/, val/ subdirectories.
        verbose: Print progress messages.

    Returns:
        Dict with keys 'train', 'test', 'val'; each value is a dict with
        keys 'images', 'paths', 'labels'.
    """
    dataset = {}
    for split in ('train', 'test', 'val'):
        split_path = os.path.join(base_dir, split)
        if not os.path.isdir(split_path):
            print(f'  [WARNING] Split directory not found: {split_path}')
            continue
        if verbose:
            print(f'\nLoading {split.upper()} split …')
        imgs, paths, labels = load_split(split_path, verbose=verbose)
        dataset[split] = {
            'images': imgs,
            'paths': paths,
            'labels': np.array(labels),
        }
        if verbose:
            n = len(labels)
            n0 = labels.count(0)
            n1 = labels.count(1)
            print(f'  → {n} images  (Normal: {n0}, Pneumonia: {n1})')

    return dataset