"""
features.py
-----------
Feature extraction pipeline for chest X-ray pneumonia detection.
Covers HOG (structural edges), GLCM (texture / cloudiness), PCA compression,
and feature concatenation / scaling.
"""

import numpy as np
import cv2
from skimage.feature import hog, graycomatrix, graycoprops
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Per-image feature extractors
# ---------------------------------------------------------------------------

def extract_hog_features(image: np.ndarray, img_size: tuple = (128, 128)) -> np.ndarray:
    """
    Compute HOG (Histogram of Oriented Gradients) features from a grayscale image.

    Args:
        image: Grayscale image array.
        img_size: Resize target before computing HOG.

    Returns:
        1-D HOG feature vector.
    """
    img_resized = cv2.resize(image, img_size)
    fd = hog(
        img_resized,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=False,
        block_norm='L2-Hys',
    )
    return fd


def extract_glcm_features(image: np.ndarray, img_size: tuple = (128, 128)) -> np.ndarray:
    """
    Compute GLCM (Gray-Level Co-occurrence Matrix) texture features.

    Distances [1, 3] × angles [0°, 45°, 90°, 135°] → 5 properties × 8 combos = 40 values.

    Args:
        image: Grayscale image array.
        img_size: Resize target before computing GLCM.

    Returns:
        1-D GLCM feature vector (contrast, dissimilarity, homogeneity, energy, correlation).
    """
    img_resized = cv2.resize(image, img_size)
    img_binned = (img_resized / 8).astype(np.uint8)
    glcm = graycomatrix(
        img_binned,
        distances=[1, 3],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=32,
        symmetric=True,
        normed=True,
    )

    contrast = graycoprops(glcm, 'contrast').flatten()
    dissimilarity = graycoprops(glcm, 'dissimilarity').flatten()
    homogeneity = graycoprops(glcm, 'homogeneity').flatten()
    energy = graycoprops(glcm, 'energy').flatten()
    correlation = graycoprops(glcm, 'correlation').flatten()

    return np.hstack([contrast, dissimilarity, homogeneity, energy, correlation])


def extract_features_single(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract both HOG and GLCM features for one preprocessed image.

    Args:
        image: Preprocessed (CLAHE-masked) grayscale image.

    Returns:
        (hog_features, glcm_features) as 1-D arrays.
    """
    hog_feats = extract_hog_features(image)
    glcm_feats = extract_glcm_features(image)
    return hog_feats, glcm_feats


# ---------------------------------------------------------------------------
# Batch feature extraction
# ---------------------------------------------------------------------------

def extract_features_batch(
    images: list,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract HOG and GLCM features for a list of preprocessed images.

    Args:
        images: List of preprocessed grayscale images.
        verbose: Print progress every 500 images.

    Returns:
        (X_hog, X_glcm) — 2-D arrays of shape (n_samples, n_features).
    """
    X_hog, X_glcm = [], []

    for i, img in enumerate(images):
        if verbose and i % 500 == 0:
            print(f'  Extracting features … {i}/{len(images)}')
        hog_f, glcm_f = extract_features_single(img)
        X_hog.append(hog_f)
        X_glcm.append(glcm_f)

    return np.array(X_hog), np.array(X_glcm)


# ---------------------------------------------------------------------------
# PCA + Scaling pipeline
# ---------------------------------------------------------------------------

def build_feature_pipeline(
    X_train_hog: np.ndarray,
    X_train_glcm: np.ndarray,
    X_test_hog: np.ndarray,
    X_test_glcm: np.ndarray,
    pca_components: int = 50,
    use_pca: bool = False,
) -> tuple[np.ndarray, np.ndarray, PCA | None, StandardScaler]:
    """
    Optionally apply PCA to HOG features, concatenate with GLCM, then scale.

    NOTE: Based on project findings, PCA on HOG features discards ~64 % of
    variance and hurts recall.  Set use_pca=False (default) to use raw HOG.

    Args:
        X_train_hog: HOG features for training set.
        X_train_glcm: GLCM features for training set.
        X_test_hog: HOG features for test set.
        X_test_glcm: GLCM features for test set.
        pca_components: Number of PCA components (only used when use_pca=True).
        use_pca: Whether to compress HOG via PCA before concatenation.

    Returns:
        (X_train_scaled, X_test_scaled, fitted_pca_or_None, fitted_scaler)
    """
    if use_pca:
        print(f'  Applying PCA (n_components={pca_components}) to HOG features …')
        pca = PCA(n_components=pca_components, random_state=42)
        X_train_hog_reduced = pca.fit_transform(X_train_hog)
        X_test_hog_reduced = pca.transform(X_test_hog)
        cum_var = np.cumsum(pca.explained_variance_ratio_)[-1]
        print(f'  PCA retained {cum_var * 100:.1f} % of variance.')
    else:
        print('  Using raw HOG features (PCA skipped).')
        pca = None
        X_train_hog_reduced = X_train_hog
        X_test_hog_reduced = X_test_hog

    X_train_combined = np.hstack([X_train_hog_reduced, X_train_glcm])
    X_test_combined = np.hstack([X_test_hog_reduced, X_test_glcm])
    print(f'  Combined feature vector size: {X_train_combined.shape[1]}')

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_combined)
    X_test_scaled = scaler.transform(X_test_combined)

    return X_train_scaled, X_test_scaled, pca, scaler


def get_feature_names(
    n_hog: int,
    n_glcm: int,
    pca: PCA | None = None,
) -> np.ndarray:
    """
    Build human-readable feature names for the combined feature vector.

    Args:
        n_hog: Number of HOG features (raw or PCA-reduced).
        n_glcm: Number of GLCM features.
        pca: Fitted PCA object (or None for raw HOG).

    Returns:
        Array of feature name strings.
    """
    if pca is not None:
        hog_names = [f'HOG_PCA_{i + 1}' for i in range(pca.n_components_)]
    else:
        hog_names = [f'HOG_{i + 1}' for i in range(n_hog)]

    glcm_names = [f'GLCM_{i + 1}' for i in range(n_glcm)]
    return np.array(hog_names + glcm_names)