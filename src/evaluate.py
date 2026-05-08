"""
evaluate.py
-----------
Full evaluation suite for the chest X-ray pneumonia detection pipeline.

Generates and saves every figure from the notebook PLUS per-model reports
and a final side-by-side comparison:

  01_class_distribution.png
  02_preprocessing_pipeline.png
  03_hog_feature_map.png
  04_pca_scree_plot.png
  05_cv_stability.png
  06_clinical_evaluation_balanced_random_forest.png
  06_clinical_evaluation_svm_c=0.01.png
  07_feature_importances.png
  08_model_comparison.png          ← bar chart: all metrics, both models
  09_confusion_matrix_comparison.png ← side-by-side confusion matrices

Console output includes a full sklearn classification_report for every
model and a final summary table.

Usage (CLI):
    python src/evaluate.py --data_dir data/chest_xray \
                           --models_dir models \
                           --results_dir results/figures
"""

import os
import argparse
import random
import pickle
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from skimage import exposure
from skimage.feature import hog
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve, average_precision_score,
)

from preprocess import create_lung_mask, load_dataset, LABEL_NAMES
from features import extract_features_batch, build_feature_pipeline, get_feature_names


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------

def _save(fig, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  [Saved] {path}')


# ---------------------------------------------------------------------------
# Figure 1 — Class distribution
# ---------------------------------------------------------------------------

def plot_class_distribution(
    y_train: np.ndarray,
    y_test: np.ndarray,
    save_dir: str,
) -> None:
    train_counts = [int(np.sum(y_train == 0)), int(np.sum(y_train == 1))]
    test_counts  = [int(np.sum(y_test  == 0)), int(np.sum(y_test  == 1))]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, counts, title in zip(
        axes,
        [train_counts, test_counts],
        ['Training Set Class Distribution', 'Test Set Class Distribution'],
    ):
        sns.barplot(x=['Normal', 'Pneumonia'], y=counts, ax=ax,
                    palette=['#2ECC71', '#E74C3C'])
        ax.set_title(title, fontweight='bold')
        ax.set_ylabel('Number of Images')
        for i, v in enumerate(counts):
            ax.text(i, v + max(counts) * 0.02, str(v), ha='center', fontweight='bold')

    fig.suptitle('Dataset Class Distribution', fontsize=14, fontweight='bold')
    fig.tight_layout()
    _save(fig, os.path.join(save_dir, '01_class_distribution.png'))


# ---------------------------------------------------------------------------
# Figure 2 — Preprocessing pipeline (2 × 4 grid)
# ---------------------------------------------------------------------------

def plot_preprocessing_pipeline(
    data_dir: str,
    save_dir: str,
) -> None:
    normal_dir    = os.path.join(data_dir, 'train', 'NORMAL')
    pneumonia_dir = os.path.join(data_dir, 'train', 'PNEUMONIA')

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    for row, (img_dir, label) in enumerate([
        (normal_dir, 'NORMAL'),
        (pneumonia_dir, 'PNEUMONIA'),
    ]):
        img_path = os.path.join(img_dir, random.choice(os.listdir(img_dir)))
        img      = cv2.imread(img_path)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_res  = cv2.resize(img_gray, (256, 256))

        clahe      = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        equalized  = clahe.apply(img_res)
        _, mask    = cv2.threshold(equalized, 0, 255,
                                   cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask       = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=2)
        mask       = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        segmented  = cv2.bitwise_and(equalized, equalized, mask=mask)

        panels = [
            (img_res,   f'{label}: Original'),
            (equalized, f'{label}: CLAHE'),
            (mask,      f'{label}: Otsu Mask'),
            (segmented, f'{label}: Segmented'),
        ]
        for col, (image, title) in enumerate(panels):
            axes[row, col].imshow(image, cmap='gray')
            axes[row, col].set_title(title, fontsize=12)
            axes[row, col].axis('off')

    fig.suptitle('Preprocessing Pipeline: Isolating Lungs',
                 fontsize=16, fontweight='bold')
    fig.tight_layout()
    _save(fig, os.path.join(save_dir, '02_preprocessing_pipeline.png'))


# ---------------------------------------------------------------------------
# Figure 3 — HOG feature map
# ---------------------------------------------------------------------------

def plot_hog_feature_map(
    data_dir: str,
    save_dir: str,
) -> None:
    pneumonia_dir = os.path.join(data_dir, 'train', 'PNEUMONIA')
    img_path = os.path.join(pneumonia_dir, random.choice(os.listdir(pneumonia_dir)))

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (256, 256))

    clahe  = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq     = clahe.apply(img)
    _, mask = cv2.threshold(eq, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask    = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=2)
    mask    = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    seg     = cv2.bitwise_and(eq, eq, mask=mask)

    _, hog_img = hog(
        seg,
        orientations=8,
        pixels_per_cell=(16, 16),
        cells_per_block=(1, 1),
        visualize=True,
    )
    hog_img_rescaled = exposure.rescale_intensity(hog_img, in_range=(0, 10))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
    ax1.imshow(seg, cmap='gray');  ax1.set_title('Segmented Image', fontsize=14); ax1.axis('off')
    ax2.imshow(hog_img_rescaled, cmap='inferno'); ax2.set_title('HOG Gradients (Inferno)', fontsize=14); ax2.axis('off')
    fig.suptitle('How the Computer Sees Structure (HOG Features)',
                 fontsize=16, fontweight='bold')
    fig.tight_layout()
    _save(fig, os.path.join(save_dir, '03_hog_feature_map.png'))


# ---------------------------------------------------------------------------
# Figure 4 — PCA scree plot  (only when PCA was used)
# ---------------------------------------------------------------------------

def plot_pca_scree(pca, save_dir: str) -> None:
    if pca is None:
        print('  [Skip] PCA scree plot — PCA was not used.')
        return

    variance   = pca.explained_variance_ratio_
    cumulative = np.cumsum(variance)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(1, len(variance) + 1), variance, alpha=0.7,
           color='teal', label='Individual Component Variance')
    ax.plot(range(1, len(cumulative) + 1), cumulative, marker='o',
            color='orange', label='Cumulative Variance')
    ax.axhline(cumulative[-1], color='red', linestyle='--',
               label=f'Total Captured ({cumulative[-1] * 100:.1f} %)')
    ax.set_title('PCA Scree Plot: HOG Dimensionality Reduction',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Principal Component Number')
    ax.set_ylabel('Variance Explained')
    ax.legend(loc='center right')
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    fig.tight_layout()
    _save(fig, os.path.join(save_dir, '04_pca_scree_plot.png'))


# ---------------------------------------------------------------------------
# Figure 5 — CV stability  (re-plots if accuracies provided)
# ---------------------------------------------------------------------------

def plot_cv_stability(fold_accuracies: list[float], save_dir: str) -> None:
    if not fold_accuracies:
        print('  [Skip] CV stability plot — no fold accuracies provided.')
        return

    mean_acc = np.mean(fold_accuracies)
    folds    = range(1, len(fold_accuracies) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(folds, fold_accuracies, marker='s', markersize=8,
            color='purple', linewidth=2, label='Fold Accuracy')
    ax.axhline(mean_acc, color='red', linestyle='--', linewidth=2,
               label=f'Mean CV Accuracy ({mean_acc:.3f})')
    ax.set_title('5-Fold Cross Validation Stability (Training Data)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Fold Number')
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0.85, 1.0)
    ax.set_xticks(list(folds))
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    fig.tight_layout()
    _save(fig, os.path.join(save_dir, '05_cv_stability.png'))


# ---------------------------------------------------------------------------
# Figure 6 — Clinical evaluation (confusion matrix + ROC + PR)
# ---------------------------------------------------------------------------

def plot_clinical_evaluation(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
    save_dir: str,
) -> dict:
    """
    Generate the 3-panel clinical evaluation figure and return metrics dict.
    """
    preds  = model.predict(X_test)
    report = classification_report(
        y_test, preds,
        target_names=['Normal', 'Pneumonia'],
        output_dict=True,
    )

    # Probability scores (use decision_function for SVC without probability=True)
    if hasattr(model, 'predict_proba'):
        y_probs = model.predict_proba(X_test)[:, 1]
    else:
        y_probs = model.decision_function(X_test)
        y_probs = (y_probs - y_probs.min()) / (y_probs.max() - y_probs.min())

    cm        = confusion_matrix(y_test, preds)
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_auc   = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y_test, y_probs)
    ap        = average_precision_score(y_test, y_probs)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: Confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['Normal', 'Pneumonia'],
                yticklabels=['Normal', 'Pneumonia'],
                annot_kws={'size': 14})
    axes[0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')

    # Panel 2: ROC
    axes[1].plot(fpr, tpr, color='darkorange', lw=2,
                 label=f'ROC (AUC = {roc_auc:.3f})')
    axes[1].plot([0, 1], [0, 1], 'navy', lw=2, linestyle='--')
    axes[1].set_xlim([0, 1]); axes[1].set_ylim([0, 1.05])
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title('ROC Curve', fontsize=14, fontweight='bold')
    axes[1].legend(loc='lower right')
    axes[1].grid(alpha=0.3)

    # Panel 3: Precision-Recall
    axes[2].plot(recall, precision, color='purple', lw=2,
                 label=f'PR (AP = {ap:.3f})')
    axes[2].set_xlim([0, 1]); axes[2].set_ylim([0, 1.05])
    axes[2].set_xlabel('Recall (Sensitivity)')
    axes[2].set_ylabel('Precision (PPV)')
    axes[2].set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    axes[2].legend(loc='lower left')
    axes[2].grid(alpha=0.3)

    fig.suptitle(f'Clinical Evaluation — {model_name}',
                 fontsize=16, fontweight='bold')
    fig.tight_layout()
    slug = model_name.lower().replace(' ', '_')
    _save(fig, os.path.join(save_dir, f'06_clinical_evaluation_{slug}.png'))

    # ── Console output ───────────────────────────────────────────────────────
    border = '=' * 60
    print(f'\n{border}')
    print(f'  Classification Report — {model_name}')
    print(border)
    print(classification_report(y_test, preds, target_names=['Normal', 'Pneumonia']))
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    print(f'  Sensitivity (Pneumonia Recall) : {sensitivity:.4f}')
    print(f'  Specificity (Normal Recall)    : {specificity:.4f}')
    print(f'  ROC AUC                        : {roc_auc:.4f}')
    print(f'  Average Precision              : {ap:.4f}')
    print(border)

    return {
        'model_name':       model_name,
        'report':           report,
        'roc_auc':          roc_auc,
        'avg_precision':    ap,
        'sensitivity':      sensitivity,
        'specificity':      specificity,
        'accuracy':         report['accuracy'],
        'normal_f1':        report['Normal']['f1-score'],
        'pneumonia_f1':     report['Pneumonia']['f1-score'],
        'macro_f1':         report['macro avg']['f1-score'],
        'confusion_matrix': cm,
        'fpr': fpr, 'tpr': tpr,
        'precision_curve':  precision,
        'recall_curve':     recall,
    }


# ---------------------------------------------------------------------------
# Figure 7 — Feature importances  (RF only)
# ---------------------------------------------------------------------------

def plot_feature_importances(
    model,
    feature_names: np.ndarray,
    save_dir: str,
    top_n: int = 30,
) -> None:
    if not hasattr(model, 'feature_importances_'):
        print('  [Skip] Feature importances — model has no feature_importances_.')
        return

    importances = model.feature_importances_
    indices     = np.argsort(importances)[::-1][:top_n]
    top_names   = feature_names[indices]
    top_imp     = importances[indices]
    colors      = ['#3498DB' if 'HOG' in f else '#E67E22' for f in top_names]

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(range(len(top_names)), top_imp[::-1], color=colors[::-1])
    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels(top_names[::-1])
    ax.set_xlabel('Gini Importance', fontsize=12)
    ax.set_title(
        f'Top {top_n} Predictive Features: Structure (HOG) vs Texture (GLCM)',
        fontsize=14, fontweight='bold',
    )
    legend_elements = [
        Patch(facecolor='#3498DB', label='HOG (Structural Edges)'),
        Patch(facecolor='#E67E22', label='GLCM (Cloudiness / Texture)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=12)
    ax.grid(axis='x', linestyle='--', alpha=0.6)
    fig.tight_layout()
    _save(fig, os.path.join(save_dir, '07_feature_importances.png'))



# ---------------------------------------------------------------------------
# Figure 8 — Final model comparison bar chart
# ---------------------------------------------------------------------------

def plot_model_comparison(results: list[dict], save_dir: str) -> None:
    """
    Side-by-side grouped bar chart comparing every key metric across all models.

    Args:
        results: List of dicts returned by plot_clinical_evaluation().
        save_dir: Directory to save the figure.
    """
    metrics = {
        'Accuracy':           'accuracy',
        'Normal F1':          'normal_f1',
        'Pneumonia F1':       'pneumonia_f1',
        'Macro F1':           'macro_f1',
        'Sensitivity':        'sensitivity',
        'Specificity':        'specificity',
        'ROC AUC':            'roc_auc',
        'Avg Precision':      'avg_precision',
    }

    model_names = [r['model_name'] for r in results]
    metric_labels = list(metrics.keys())
    metric_keys   = list(metrics.values())

    n_metrics = len(metric_labels)
    n_models  = len(results)
    x         = np.arange(n_metrics)
    width     = 0.8 / n_models
    colors    = ['#2980B9', '#E74C3C', '#2ECC71', '#F39C12', '#9B59B6']

    fig, ax = plt.subplots(figsize=(16, 7))

    for i, (res, color) in enumerate(zip(results, colors)):
        values = [res[k] for k in metric_keys]
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=res['model_name'],
                      color=color, alpha=0.85, edgecolor='white', linewidth=0.8)
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.008,
                f'{val:.3f}',
                ha='center', va='bottom', fontsize=8, fontweight='bold',
            )

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_ylim(0, 1.12)
    ax.set_title('Final Model Comparison — All Metrics',
                 fontsize=15, fontweight='bold', pad=15)
    ax.legend(fontsize=11, loc='lower right')
    ax.axhline(0.9, color='grey', linestyle=':', linewidth=1, alpha=0.6)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    fig.tight_layout()
    _save(fig, os.path.join(save_dir, '08_model_comparison.png'))


# ---------------------------------------------------------------------------
# Figure 9 — Side-by-side confusion matrix comparison
# ---------------------------------------------------------------------------

def plot_confusion_matrix_comparison(results: list[dict], save_dir: str) -> None:
    """
    Plot one confusion matrix per model in a single row for easy comparison.

    Args:
        results: List of dicts returned by plot_clinical_evaluation().
        save_dir: Directory to save the figure.
    """
    n = len(results)
    cmaps = ['Blues', 'Reds', 'Greens', 'Purples', 'Oranges']

    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, res, cmap in zip(axes, results, cmaps):
        cm = res['confusion_matrix']
        sns.heatmap(
            cm, annot=True, fmt='d', cmap=cmap, ax=ax,
            xticklabels=['Normal', 'Pneumonia'],
            yticklabels=['Normal', 'Pneumonia'],
            annot_kws={'size': 14},
        )
        acc  = res['accuracy']
        sens = res['sensitivity']
        spec = res['specificity']
        ax.set_title(
            f"{res['model_name']}\n"
            f"Acc={acc:.3f}  Sens={sens:.3f}  Spec={spec:.3f}",
            fontsize=11, fontweight='bold',
        )
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')

    fig.suptitle('Confusion Matrix Comparison', fontsize=15, fontweight='bold')
    fig.tight_layout()
    _save(fig, os.path.join(save_dir, '09_confusion_matrix_comparison.png'))


# ---------------------------------------------------------------------------
# Final summary table (console)
# ---------------------------------------------------------------------------

def print_final_summary(results: list[dict]) -> None:
    """Print a compact ASCII comparison table for all evaluated models."""
    cols = ['Model', 'Accuracy', 'Normal F1', 'Pneumonia F1',
            'Macro F1', 'Sensitivity', 'Specificity', 'ROC AUC']
    col_w = [28, 10, 10, 13, 10, 13, 13, 10]

    header = '  '.join(c.ljust(w) for c, w in zip(cols, col_w))
    divider = '-' * len(header)

    print('\n' + '=' * len(header))
    print('  FINAL MODEL COMPARISON SUMMARY')
    print('=' * len(header))
    print(header)
    print(divider)

    best_macro = max(r['macro_f1'] for r in results)

    for r in results:
        tag = ' ★' if r['macro_f1'] == best_macro else '  '
        row_vals = [
            r['model_name'] + tag,
            f"{r['accuracy']:.4f}",
            f"{r['normal_f1']:.4f}",
            f"{r['pneumonia_f1']:.4f}",
            f"{r['macro_f1']:.4f}",
            f"{r['sensitivity']:.4f}",
            f"{r['specificity']:.4f}",
            f"{r['roc_auc']:.4f}",
        ]
        print('  '.join(v.ljust(w) for v, w in zip(row_vals, col_w)))

    print('=' * len(header))
    print('  ★ = best Macro F1')
    print()

# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def _load(path: str):
    with open(path, 'rb') as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='Evaluate trained models and generate figures.')
    p.add_argument('--data_dir',    default='data/chest_xray')
    p.add_argument('--models_dir',  default='models')
    p.add_argument('--results_dir', default='results/figures')
    p.add_argument('--use_pca',     action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    # ── Load models ─────────────────────────────────────────────────────────
    print('Loading models …')
    rf_model  = _load(os.path.join(args.models_dir, 'rf_model.pkl'))
    svm_model = _load(os.path.join(args.models_dir, 'svm_model.pkl'))
    knn_model = _load(os.path.join(args.models_dir, 'knn_model.pkl'))
    lr_model  = _load(os.path.join(args.models_dir, 'lr_model.pkl'))
    scaler    = _load(os.path.join(args.models_dir, 'scaler.pkl'))
    pca_path  = os.path.join(args.models_dir, 'pca.pkl')
    pca       = _load(pca_path) if os.path.exists(pca_path) else None

    # ── Load & preprocess data ───────────────────────────────────────────────
    print('\nLoading dataset …')
    dataset = load_dataset(args.data_dir, verbose=True)
    train = dataset['train']
    test  = dataset['test']

    # ── Feature extraction ───────────────────────────────────────────────────
    print('\nExtracting features …')
    X_train_hog, X_train_glcm = extract_features_batch(train['images'])
    X_test_hog,  X_test_glcm  = extract_features_batch(test['images'])
    y_train = train['labels']
    y_test  = test['labels']

    # ── Feature pipeline ─────────────────────────────────────────────────────
    X_train_scaled, X_test_scaled, pca_fitted, _ = build_feature_pipeline(
        X_train_hog, X_train_glcm,
        X_test_hog,  X_test_glcm,
        use_pca=args.use_pca,
    )
    # Use the scaler saved during training for the test set
    X_test_scaled = scaler.transform(
        np.hstack([
            (pca_fitted.transform(X_test_hog) if pca_fitted else X_test_hog),
            X_test_glcm,
        ])
    )

    feature_names = get_feature_names(
        X_test_hog.shape[1],
        X_test_glcm.shape[1],
        pca=pca,
    )

    # ── Generate all figures ─────────────────────────────────────────────────
    print('\nGenerating figures …')

    plot_class_distribution(y_train, y_test, args.results_dir)
    plot_preprocessing_pipeline(args.data_dir, args.results_dir)
    plot_hog_feature_map(args.data_dir, args.results_dir)
    plot_pca_scree(pca, args.results_dir)

    # CV stability requires fold accuracies — load if saved, else skip
    cv_path = os.path.join(args.models_dir, 'cv_accuracies.pkl')
    if os.path.exists(cv_path):
        fold_accs = _load(cv_path)
        plot_cv_stability(fold_accs, args.results_dir)

    # ── Per-model evaluation (report + figure) ──────────────────────────────
    all_results = []

    for model, name in [
        (rf_model,  'Balanced Random Forest'),
        (svm_model, 'SVM C=0.01'),
        (knn_model, 'KNN k=7'),
        (lr_model,  'Logistic Regression'),
    ]:
        result = plot_clinical_evaluation(
            model, X_test_scaled, y_test, name, args.results_dir,
        )
        all_results.append(result)

    # ── Feature importances ──────────────────────────────────────────────────
    plot_feature_importances(rf_model, feature_names, args.results_dir)

    # ── Final comparison figures ─────────────────────────────────────────────
    print('\nGenerating comparison figures …')
    plot_model_comparison(all_results, args.results_dir)
    plot_confusion_matrix_comparison(all_results, args.results_dir)

    # ── Final summary table (console) ───────────────────────────────────────
    print_final_summary(all_results)

    print(f'All figures saved to: {args.results_dir}')


if __name__ == '__main__':
    main()