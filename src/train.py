"""
train.py
--------
Model training for chest X-ray pneumonia detection.

Trains 4 models: Balanced Random Forest, SVM, K-Nearest Neighbours, and Logistic Regression
with optional 5-fold cross-validation.  Saves fitted models and the scaler
to the models/ directory.

Usage (CLI):
    python src/train.py --data_dir data/chest_xray --models_dir models --results_dir results/figures
"""

import os
import argparse
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from preprocess import load_dataset
from features import extract_features_batch, build_feature_pipeline


# ---------------------------------------------------------------------------
# Default hyper-parameters (best params from notebook experiments)
# ---------------------------------------------------------------------------

RF_PARAMS = dict(
    n_estimators=300,
    max_depth=10,
    min_samples_split=5,
    sampling_strategy='auto',
    random_state=42,
    n_jobs=-1,
)

SVM_PARAMS = dict(
    kernel='rbf',
    class_weight='balanced',
    C=0.01,
    random_state=42,
)

KNN_PARAMS = dict(
    n_neighbors=7,
    weights='distance',
    metric='euclidean',
    n_jobs=-1,
)

LR_PARAMS = dict(
    solver='lbfgs',
    max_iter=1000,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1,
)


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def run_cross_validation(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_splits: int = 5,
    save_dir: str | None = None,
) -> list[float]:
    """
    Run stratified k-fold cross-validation on the training set using RF.

    Args:
        X_train: Scaled feature matrix.
        y_train: Label array.
        n_splits: Number of CV folds.
        save_dir: If provided, save the CV stability plot here.

    Returns:
        List of per-fold validation accuracies.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_accuracies = []

    print(f'\nRunning {n_splits}-Fold Cross-Validation on Training Data …')
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)

        clf = BalancedRandomForestClassifier(**RF_PARAMS)
        clf.fit(X_tr_s, y_tr)

        acc = np.mean(clf.predict(X_val_s) == y_val)
        fold_accuracies.append(acc)
        print(f'  Fold {fold} Validation Accuracy: {acc:.4f}')

    mean_acc = np.mean(fold_accuracies)
    print(f'\n  Mean CV Accuracy: {mean_acc:.4f}')

    if save_dir:
        _plot_cv_stability(fold_accuracies, mean_acc, save_dir)

    return fold_accuracies


def _plot_cv_stability(
    fold_accuracies: list[float],
    mean_acc: float,
    save_dir: str,
) -> None:
    folds = range(1, len(fold_accuracies) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(folds, fold_accuracies, marker='s', markersize=8,
             color='purple', linewidth=2, label='Fold Accuracy')
    plt.axhline(mean_acc, color='red', linestyle='--', linewidth=2,
                label=f'Mean CV Accuracy ({mean_acc:.3f})')
    plt.title('5-Fold Cross Validation Stability (Training Data)',
              fontsize=14, fontweight='bold')
    plt.xlabel('Fold Number')
    plt.ylabel('Accuracy')
    plt.ylim(0.85, 1.0)
    plt.xticks(folds)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, 'cv_stability.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'  [Saved] CV stability plot → {path}')


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> BalancedRandomForestClassifier:
    """Train the primary Balanced Random Forest model."""
    print('\nTraining Balanced Random Forest …')
    model = BalancedRandomForestClassifier(**RF_PARAMS)
    model.fit(X_train, y_train)
    print('  Done.')
    return model


def train_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> SVC:
    """Train the regularised SVM (C=0.01, generalises across domain shift)."""
    print('\nTraining SVM (C=0.01) …')
    model = SVC(**SVM_PARAMS)
    model.fit(X_train, y_train)
    print('  Done.')
    return model


def train_knn(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> KNeighborsClassifier:
    """Train K-Nearest Neighbours (k=7, distance-weighted)."""
    print('\nTraining KNN (k=7) …')
    model = KNeighborsClassifier(**KNN_PARAMS)
    model.fit(X_train, y_train)
    print('  Done.')
    return model


def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> LogisticRegression:
    """Train Logistic Regression with balanced class weights."""
    print('\nTraining Logistic Regression …')
    model = LogisticRegression(**LR_PARAMS)
    model.fit(X_train, y_train)
    print('  Done.')
    return model


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def save_artifact(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    print(f'  [Saved] {path}')


def load_artifact(path: str):
    with open(path, 'rb') as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='Train pneumonia detection models.')
    p.add_argument('--data_dir', default='data/chest_xray',
                   help='Root directory containing train/ test/ val/ splits.')
    p.add_argument('--models_dir', default='models',
                   help='Directory to save fitted models and scaler.')
    p.add_argument('--results_dir', default='results/figures',
                   help='Directory to save training figures.')
    p.add_argument('--use_pca', action='store_true',
                   help='Apply PCA to HOG features (not recommended).')
    p.add_argument('--skip_cv', action='store_true',
                   help='Skip cross-validation (faster).')
    return p.parse_args()


def main():
    args = parse_args()

    # 1. Load and preprocess data
    print('=' * 60)
    print('STEP 1 — Loading & Preprocessing Images')
    print('=' * 60)
    dataset = load_dataset(args.data_dir, verbose=True)

    train = dataset['train']
    test = dataset.get('test', None)

    # 2. Extract features
    print('\n' + '=' * 60)
    print('STEP 2 — Extracting HOG + GLCM Features')
    print('=' * 60)
    print('  Train …')
    X_train_hog, X_train_glcm = extract_features_batch(train['images'])
    y_train = train['labels']

    if test:
        print('  Test …')
        X_test_hog, X_test_glcm = extract_features_batch(test['images'])
        y_test = test['labels']
    else:
        X_test_hog = X_test_glcm = y_test = None
        print('  [INFO] No test split found — skipping test feature extraction.')

    # 3. Build feature pipeline
    print('\n' + '=' * 60)
    print('STEP 3 — Feature Pipeline (PCA + Scaling)')
    print('=' * 60)
    if test:
        X_train_scaled, X_test_scaled, pca, scaler = build_feature_pipeline(
            X_train_hog, X_train_glcm,
            X_test_hog, X_test_glcm,
            use_pca=args.use_pca,
        )
    else:
        from sklearn.preprocessing import StandardScaler as _SS
        import numpy as _np
        X_train_combined = _np.hstack([X_train_hog, X_train_glcm])
        scaler = _SS()
        X_train_scaled = scaler.fit_transform(X_train_combined)
        X_test_scaled = None
        pca = None

    # 4. Cross-validation
    if not args.skip_cv:
        print('\n' + '=' * 60)
        print('STEP 4 — Cross-Validation')
        print('=' * 60)
        run_cross_validation(X_train_scaled, y_train, save_dir=args.results_dir)

    # 5. Train final models
    print('\n' + '=' * 60)
    print('STEP 5 — Training Final Models')
    print('=' * 60)
    rf_model  = train_random_forest(X_train_scaled, y_train)
    svm_model = train_svm(X_train_scaled, y_train)
    knn_model = train_knn(X_train_scaled, y_train)
    lr_model  = train_logistic_regression(X_train_scaled, y_train)

    # Quick validation on test if available
    if X_test_scaled is not None:
        print('\n--- Quick Test-Set Report (Balanced RF) ---')
        print(classification_report(
            y_test, rf_model.predict(X_test_scaled),
            target_names=['Normal', 'Pneumonia']))

    # 6. Save artefacts
    print('\n' + '=' * 60)
    print('STEP 6 — Saving Models & Artefacts')
    print('=' * 60)
    save_artifact(rf_model,  os.path.join(args.models_dir, 'rf_model.pkl'))
    save_artifact(svm_model, os.path.join(args.models_dir, 'svm_model.pkl'))
    save_artifact(knn_model, os.path.join(args.models_dir, 'knn_model.pkl'))
    save_artifact(lr_model,  os.path.join(args.models_dir, 'lr_model.pkl'))
    save_artifact(scaler, os.path.join(args.models_dir, 'scaler.pkl'))
    if pca:
        save_artifact(pca, os.path.join(args.models_dir, 'pca.pkl'))

    print('\nTraining complete.')


if __name__ == '__main__':
    main()