"""
features.py — Handcrafted feature extraction for pneumonia detection.

Extracts a rich set of texture, statistical, and gradient-based features
from a preprocessed grayscale chest X-ray image.
"""

import numpy as np
from scipy import ndimage


def extract_features(img_array: np.ndarray) -> np.ndarray:
    """
    Extract a 1-D feature vector from a preprocessed image.

    Args:
        img_array: np.ndarray of shape (H, W), float32 in [0, 1]

    Returns:
        np.ndarray: 1-D feature vector
    """
    features = []

    # --- 1. Global Statistics ---
    features += _global_stats(img_array)

    # --- 2. Region-of-Interest Statistics (center lung area) ---
    features += _roi_stats(img_array)

    # --- 3. Gradient / Edge Features ---
    features += _gradient_features(img_array)

    # --- 4. Texture Features (GLCM-inspired) ---
    features += _texture_features(img_array)

    # --- 5. Histogram Features ---
    features += _histogram_features(img_array)

    return np.array(features, dtype=np.float32)


# ─────────────────────────── helpers ────────────────────────────────────────

def _global_stats(arr: np.ndarray) -> list:
    flat = arr.flatten()
    return [
        float(np.mean(flat)),
        float(np.std(flat)),
        float(np.median(flat)),
        float(np.percentile(flat, 25)),
        float(np.percentile(flat, 75)),
        float(np.min(flat)),
        float(np.max(flat)),
        float(np.var(flat)),
    ]


def _roi_stats(arr: np.ndarray) -> list:
    h, w = arr.shape
    r1, r2 = h // 4, 3 * h // 4
    c1, c2 = w // 4, 3 * w // 4
    roi = arr[r1:r2, c1:c2].flatten()
    return [
        float(np.mean(roi)),
        float(np.std(roi)),
        float(np.median(roi)),
        float(np.var(roi)),
    ]


def _gradient_features(arr: np.ndarray) -> list:
    gx = ndimage.sobel(arr, axis=1)
    gy = ndimage.sobel(arr, axis=0)
    magnitude = np.hypot(gx, gy)
    return [
        float(np.mean(magnitude)),
        float(np.std(magnitude)),
        float(np.max(magnitude)),
        float(np.sum(magnitude > 0.1) / magnitude.size),  # edge density
    ]


def _texture_features(arr: np.ndarray) -> list:
    """Simplified GLCM-inspired contrast and homogeneity."""
    # Quantize to 8 levels
    q = (arr * 7).astype(np.uint8)
    h, w = q.shape
    # Horizontal co-occurrence
    pairs_h = list(zip(q[:, :-1].flatten(), q[:, 1:].flatten()))
    # Vertical co-occurrence
    pairs_v = list(zip(q[:-1, :].flatten(), q[1:, :].flatten()))

    def contrast(pairs):
        return float(np.mean([(int(a) - int(b)) ** 2 for a, b in pairs]))

    def homogeneity(pairs):
        return float(np.mean([1 / (1 + abs(int(a) - int(b))) for a, b in pairs]))

    return [
        contrast(pairs_h),
        contrast(pairs_v),
        homogeneity(pairs_h),
        homogeneity(pairs_v),
    ]


def _histogram_features(arr: np.ndarray, bins: int = 16) -> list:
    hist, _ = np.histogram(arr, bins=bins, range=(0, 1), density=True)
    return hist.tolist()