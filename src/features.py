"""
features.py

Optimized feature extraction pipeline for Chest X-ray classification.
Updated for higher resolution (128x128) and improved texture scaling.

USES:
✔ HOG  -> edge/shape structure (Optimized for larger resolution)
✔ LBP  -> multiscale texture patterns (Captures finer interstitial details)
✔ Statistical grayscale features
"""

import cv2
import numpy as np

from scipy.stats import skew, kurtosis
from skimage.feature import hog, local_binary_pattern


# ─────────────────────────────────────────────
# HOG FEATURES
# ─────────────────────────────────────────────
def hog_features(image_bgr: np.ndarray) -> np.ndarray:
    """
    Histogram of Oriented Gradients (HOG)
    
    Captures:
    - Edges and lung structure
    - Consolidation patterns
    """

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Increased pixels_per_cell to (16, 16) to maintain structural 
    # context at the higher 128x128 resolution.
    features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(16, 16), 
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        visualize=False,
        feature_vector=True
    )

    return features.astype(np.float32)


# ─────────────────────────────────────────────
# LBP FEATURES (MULTISCALE)
# ─────────────────────────────────────────────
def lbp_features(
    image_bgr: np.ndarray,
    n_bins: int = 64
) -> np.ndarray:
    """
    Local Binary Pattern (LBP)
    
    Updated to capture texture at two different scales (R=1 and R=3)
    to better identify subtle interstitial pneumonia patterns.
    """

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    
    all_hists = []
    
    # Extract textures at multiple scales: 
    # Small scale (P=8, R=1) for fine detail, Large scale (P=24, R=3) for broader patterns.
    for P, R in [(8, 1), (24, 3)]:
        lbp = local_binary_pattern(
            gray,
            P=P,
            R=R,
            method='uniform'
        )

        hist, _ = np.histogram(
            lbp.ravel(),
            bins=n_bins,
            range=(0, P + 2),
            density=True
        )
        all_hists.append(hist)

    return np.concatenate(all_hists).astype(np.float32)


# ─────────────────────────────────────────────
# STATISTICAL FEATURES
# ─────────────────────────────────────────────
def statistical_features(image_bgr: np.ndarray) -> np.ndarray:
    """
    Global grayscale statistics.
    """

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    pixels = gray.astype(np.float32).ravel()

    feats = [
        np.mean(pixels),
        np.std(pixels),
        skew(pixels),
        kurtosis(pixels),

        np.percentile(pixels, 10),
        np.percentile(pixels, 25),
        np.percentile(pixels, 50),
        np.percentile(pixels, 75),
        np.percentile(pixels, 90),
    ]

    return np.array(feats, dtype=np.float32)


# ─────────────────────────────────────────────
# MASTER FEATURE EXTRACTOR
# ─────────────────────────────────────────────
def extract_features(image_bgr: np.ndarray) -> np.ndarray:
    """
    Concatenate HOG, Multiscale LBP, and Statistics into one vector.
    """

    hog_feat = hog_features(image_bgr)
    lbp_feat = lbp_features(image_bgr)
    stat_feat = statistical_features(image_bgr)

    features = np.concatenate([
        hog_feat,
        lbp_feat,
        stat_feat
    ])

    return features.astype(np.float32)


# ─────────────────────────────────────────────
# DATASET FEATURE MATRIX
# ─────────────────────────────────────────────
def build_feature_matrix(images: np.ndarray) -> np.ndarray:
    """
    Convert image stack (N, H, W, 3) to feature matrix (N, n_features).
    """

    features = []
    total = len(images)

    for i, img in enumerate(images):
        if i % 500 == 0:
            print(f"[INFO] Extracting features: {i}/{total}")

        features.append(extract_features(img))

    return np.array(features, dtype=np.float32)


# ─────────────────────────────────────────────
# TEST RUN
# ─────────────────────────────────────────────
if __name__ == '__main__':
    # Test with the new 128x128 target resolution
    dummy = np.random.randint(
        0,
        256,
        (128, 128, 3),
        dtype=np.uint8
    )

    f = extract_features(dummy)
    print("Feature vector shape:", f.shape)