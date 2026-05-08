"""
preprocess.py — X-ray image preprocessing pipeline
Applies CLAHE enhancement, resizing, and normalization.
"""

import numpy as np
from PIL import Image, ImageFilter


TARGET_SIZE = (224, 224)


def preprocess_image(img: Image.Image) -> np.ndarray:
    """
    Full preprocessing pipeline for a chest X-ray image.

    Steps:
        1. Convert to grayscale (X-rays are single-channel)
        2. Resize to target dimensions
        3. Apply CLAHE-like histogram equalization for contrast enhancement
        4. Normalize pixel values to [0, 1]

    Args:
        img: PIL Image (RGB or L)

    Returns:
        np.ndarray of shape (224, 224) with float32 values in [0, 1]
    """
    # 1. Grayscale
    gray = img.convert('L')

    # 2. Resize
    resized = gray.resize(TARGET_SIZE, Image.LANCZOS)

    # 3. Contrast enhancement (CLAHE approximation via histogram equalization)
    equalized = _histogram_equalize(resized)

    # 4. Normalize
    arr = np.array(equalized, dtype=np.float32) / 255.0

    return arr


def _histogram_equalize(img: Image.Image) -> Image.Image:
    """Apply global histogram equalization to a grayscale PIL image."""
    arr = np.array(img)
    hist, bins = np.histogram(arr.flatten(), bins=256, range=(0, 256))
    cdf = hist.cumsum()
    cdf_min = cdf[cdf > 0].min()
    n_pixels = arr.size
    equalized = np.round(
        (cdf[arr] - cdf_min) / (n_pixels - cdf_min) * 255
    ).astype(np.uint8)
    return Image.fromarray(equalized)