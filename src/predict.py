"""
predict.py
----------
Production inference script for chest X-ray pneumonia detection.

Loads a saved model + scaler (+ optional PCA), preprocesses an input image,
extracts features, and returns the prediction with confidence.

Usage (CLI):
    # Single image
    python src/predict.py --image path/to/xray.jpeg \
                          --models_dir models

    # Batch — all images in a directory
    python src/predict.py --image_dir path/to/images/ \
                          --models_dir models \
                          --output results/predictions.csv
"""

import os
import argparse
import pickle
import csv
import numpy as np

from preprocess import preprocess_image, LABEL_NAMES
from features import extract_hog_features, extract_glcm_features


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def _load(path: str):
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_pipeline(models_dir: str) -> dict:
    """
    Load the full inference pipeline (model, scaler, optional PCA).

    Returns a dict with keys: 'model', 'scaler', 'pca' (may be None).
    """
    rf_path  = os.path.join(models_dir, 'rf_model.pkl')
    svm_path = os.path.join(models_dir, 'svm_model.pkl')
    scaler_path = os.path.join(models_dir, 'scaler.pkl')
    pca_path    = os.path.join(models_dir, 'pca.pkl')

    # Prefer Balanced RF; fall back to SVM
    if os.path.exists(rf_path):
        model = _load(rf_path)
        model_name = 'Balanced Random Forest'
    elif os.path.exists(svm_path):
        model = _load(svm_path)
        model_name = 'SVM C=0.01'
    else:
        raise FileNotFoundError(
            f'No model file found in {models_dir}. '
            'Run train.py first.'
        )

    scaler = _load(scaler_path)
    pca    = _load(pca_path) if os.path.exists(pca_path) else None

    return {'model': model, 'scaler': scaler, 'pca': pca, 'model_name': model_name}


# ---------------------------------------------------------------------------
# Single-image prediction
# ---------------------------------------------------------------------------

def predict_image(img_path: str, pipeline: dict) -> dict:
    """
    Run the full pipeline on one image and return a prediction dict.

    Args:
        img_path: Path to a JPEG chest X-ray.
        pipeline: Dict returned by load_pipeline().

    Returns:
        {
            'path': str,
            'label': 'Normal' | 'Pneumonia',
            'label_int': 0 | 1,
            'confidence': float,   # probability of predicted class
            'normal_prob': float,
            'pneumonia_prob': float,
        }
    """
    model  = pipeline['model']
    scaler = pipeline['scaler']
    pca    = pipeline['pca']

    # Preprocess
    img = preprocess_image(img_path)
    if img is None:
        return {'path': img_path, 'error': 'Could not read image.'}

    # Feature extraction
    hog_feats  = extract_hog_features(img)
    glcm_feats = extract_glcm_features(img)

    # PCA (optional)
    if pca is not None:
        hog_feats = pca.transform(hog_feats.reshape(1, -1))[0]

    # Combine + scale
    combined = np.hstack([hog_feats, glcm_feats]).reshape(1, -1)
    scaled   = scaler.transform(combined)

    # Predict
    label_int = int(model.predict(scaled)[0])
    label     = LABEL_NAMES[label_int]

    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(scaled)[0]
    else:
        # SVC without probability: use decision_function sign only
        df   = model.decision_function(scaled)[0]
        prob = float(1 / (1 + np.exp(-df)))         # sigmoid approximation
        probs = np.array([1 - prob, prob])

    return {
        'path':          img_path,
        'label':         label,
        'label_int':     label_int,
        'confidence':    float(probs[label_int]),
        'normal_prob':   float(probs[0]),
        'pneumonia_prob': float(probs[1]),
    }


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='Run pneumonia prediction on chest X-rays.')
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument('--image',     help='Path to a single JPEG image.')
    grp.add_argument('--image_dir', help='Directory of JPEG images (batch mode).')
    p.add_argument('--models_dir', default='models',
                   help='Directory containing saved model and scaler.')
    p.add_argument('--output', default=None,
                   help='(Batch mode) Path to write CSV results.')
    return p.parse_args()


def main():
    args     = parse_args()
    pipeline = load_pipeline(args.models_dir)
    print(f'Loaded model: {pipeline["model_name"]}')

    if args.image:
        result = predict_image(args.image, pipeline)
        if 'error' in result:
            print(f'Error: {result["error"]}')
        else:
            print(f'\nImage : {result["path"]}')
            print(f'Result: {result["label"]} '
                  f'(confidence: {result["confidence"] * 100:.1f} %)')
            print(f'  Normal prob    : {result["normal_prob"] * 100:.1f} %')
            print(f'  Pneumonia prob : {result["pneumonia_prob"] * 100:.1f} %')

    else:
        import glob
        files = glob.glob(os.path.join(args.image_dir, '*.jpeg'))
        if not files:
            files = glob.glob(os.path.join(args.image_dir, '*.jpg'))
        print(f'Found {len(files)} images in {args.image_dir}')

        results = []
        for i, f in enumerate(files):
            res = predict_image(f, pipeline)
            results.append(res)
            label = res.get('label', 'ERROR')
            conf  = res.get('confidence', 0)
            print(f'  [{i+1}/{len(files)}] {os.path.basename(f):40s} → {label} ({conf*100:.1f} %)')

        if args.output:
            os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
            keys = ['path', 'label', 'label_int', 'confidence',
                    'normal_prob', 'pneumonia_prob']
            with open(args.output, 'w', newline='') as fh:
                writer = csv.DictWriter(fh, fieldnames=keys, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(results)
            print(f'\nResults saved → {args.output}')


if __name__ == '__main__':
    main()