import numpy as np
import cv2
from scipy.stats import skew, kurtosis
from skimage.feature import graycomatrix, graycoprops


# ── 1. Statistical Features (7) ────────────────────────────────
def statistical_features(img_norm: np.ndarray) -> np.ndarray:
    """Mean, variance, skewness, kurtosis, 25th/50th/75th percentiles."""
    flat = img_norm.ravel()
    return np.array([
        flat.mean(),
        flat.var(),
        skew(flat),
        kurtosis(flat),
        np.percentile(flat, 25),
        np.percentile(flat, 50),
        np.percentile(flat, 75),
    ])


# ── 2. GLCM Texture Features (~50) ─────────────────────────────
def glcm_features(img_norm: np.ndarray) -> np.ndarray:
    """
    Gray-Level Co-occurrence Matrix features.
    4 properties × 4 angles × multiple distances = ~50 features.
    """
    img8 = (img_norm * 255).astype(np.uint8)
    distances = [1, 2]
    angles    = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    properties = ['contrast', 'homogeneity', 'energy', 'correlation']

    glcm = graycomatrix(img8, distances=distances, angles=angles,
                        symmetric=True, normed=True)

    feats = []
    for prop in properties:
        feats.append(graycoprops(glcm, prop).ravel())

    return np.concatenate(feats)


# ── 3. Edge Features via Canny (2) ─────────────────────────────
def edge_features(img_norm: np.ndarray) -> np.ndarray:
    """Edge density and mean edge intensity using Canny detector."""
    img8  = (img_norm * 255).astype(np.uint8)
    edges = cv2.Canny(img8, threshold1=100, threshold2=200)

    edge_density    = edges.mean() / 255.0
    active_pixels   = edges[edges > 0]
    mean_edge_intensity = active_pixels.mean() / 255.0 if len(active_pixels) > 0 else 0.0

    return np.array([edge_density, mean_edge_intensity])


# ── 4. Histogram Features (32) ─────────────────────────────────
def histogram_features(img_norm: np.ndarray, bins: int = 32) -> np.ndarray:
    """32-bin intensity histogram of normalized pixel values."""
    hist, _ = np.histogram(img_norm.ravel(), bins=bins, range=(0.0, 1.0))
    return hist.astype(np.float32) / (hist.sum() + 1e-8)  # normalize


# ── Master Extractor (73 total) ─────────────────────────────────
def extract_features(img_norm: np.ndarray) -> np.ndarray:
    """
    Extract all 73 features from a single normalized 64x64 image.
    Breakdown:
      - Statistical : 7
      - GLCM        : 32  (4 props × 4 angles × 2 distances)
      - Edge (Canny): 2
      - Histogram   : 32
    """
    stat  = statistical_features(img_norm)   
    glcm  = glcm_features(img_norm)          
    edge  = edge_features(img_norm)          
    hist  = histogram_features(img_norm)     

    feature_vector = np.concatenate([stat, glcm, edge, hist])
    return feature_vector


def build_feature_matrix(images: np.ndarray) -> np.ndarray:
    """
    Build feature matrix from array of preprocessed images.
    Args:
        images: shape (N, 64, 64) float32 normalized images
    Returns:
        X: shape (N, n_features)
    """
    features = []
    for i, img in enumerate(images):
        if i % 500 == 0:
            print(f"  Extracting features: {i}/{len(images)}")
        features.append(extract_features(img))
    return np.array(features)


if __name__ == "__main__":
    # Quick test with a random image
    dummy_img = np.random.rand(64, 64).astype(np.float32)
    feats = extract_features(dummy_img)
    print(f"Feature vector shape: {feats.shape}")
    print(f"Expected 73 features, got: {len(feats)}")
