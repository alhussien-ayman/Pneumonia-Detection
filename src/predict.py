import cv2
import numpy as np
import joblib
import json
import csv
import os
import sys
from datetime import datetime

# Import updated extract_features from your modified features.py
from features import extract_features

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH   = os.path.join(ROOT_DIR, "models", "pneumonia_model.pkl")
RESULTS_JSON = os.path.join(ROOT_DIR, "results", "results.json")
RESULTS_CSV  = os.path.join(ROOT_DIR, "results", "results.csv")

# Clinical priority threshold for Pneumonia probability
THRESHOLD = 0.85 
# Target size must match IMG_SIZE used in train.py
IMG_SIZE = (128, 128) 

# ─────────────────────────────────────────────
def predict_image(image_path: str, model_path: str = MODEL_PATH) -> dict:
    """
    Full inference pipeline: Load -> Extract -> Scale -> PCA -> Predict.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}. Run train.py first.")

    # 1. Load Model Artefact (Now includes Scaler)
    artefact = joblib.load(model_path)
    scaler   = artefact["scaler"]
    pca      = artefact["pca"]
    model    = artefact["model"]

    # 2. Load and Basic Preprocess (Resize)
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    image_resized = cv2.resize(image, IMG_SIZE)

    # 3. Feature Extraction (HOG + LBP + Stats)
    # Ensure this matches the logic in your updated features.py
    feats = extract_features(image_resized).reshape(1, -1)

    # 4. Scaling (CRITICAL: Must use the scaler from training)
    feats_scaled = scaler.transform(feats)

    # 5. PCA Transformation
    feats_pca = pca.transform(feats_scaled)

    # 6. Predict
    label_int = model.predict(feats_pca)[0]
    # For KNN/SVM, we get probability for class 1 (Pneumonia)
    proba = model.predict_proba(feats_pca)[0, 1]

    # 7. Clinical Priority Logic
    if label_int == 1:
        if proba >= THRESHOLD:
            priority = "HIGH PRIORITY — Escalate for immediate review"
        else:
            priority = "UNCERTAIN — Refer to clinician"
    else:
        priority = "LOW RISK — Routine follow-up"

    return {
        "image":           os.path.basename(image_path),
        "label":           "Pneumonia" if label_int == 1 else "Normal",
        "raw_prediction":  int(label_int),
        "confidence":      round(float(proba), 4),
        "priority":        priority,
        "timestamp":       datetime.now().isoformat(),
    }


# ── STORAGE UTILITIES ───────────────────────────────────────────

def save_results_json(results: list, path: str = RESULTS_JSON):
    existing = []
    if os.path.exists(path):
        with open(path, "r") as f:
            try:
                existing = json.load(f)
            except json.JSONDecodeError:
                existing = []
    existing.extend(results)
    with open(path, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"Results appended to {path}")


def save_results_csv(results: list, path: str = RESULTS_CSV):
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        if write_header:
            writer.writeheader()
        writer.writerows(results)
    print(f"Results appended to {path}")


def print_result(result: dict):
    print("\n" + "═"*55)
    print(f"  IMAGE      : {result['image']}")
    print(f"  PREDICTION : {result['label']}")
    print(f"  CONFIDENCE : {result['confidence']*100:.1f}%")
    print(f"  STATUS     : {result['priority']}")
    print("═"*55)


# ── CLI EXECUTION ────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path> [image_path2 ...]")
        sys.exit(1)

    image_paths = sys.argv[1:]
    all_results = []

    for path in image_paths:
        if not os.path.exists(path):
            print(f"File not found: {path}, skipping.")
            continue
        try:
            result = predict_image(path)
            print_result(result)
            all_results.append(result)
        except Exception as e:
            print(f"Error processing {path}: {e}")

    if all_results:
        save_results_json(all_results)
        save_results_csv(all_results)