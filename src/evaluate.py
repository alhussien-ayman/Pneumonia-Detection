import numpy as np
import joblib
import matplotlib.pyplot as plt
import os

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    balanced_accuracy_score # Added for better imbalanced evaluation
)

# ─────────────────────────────────────────────
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIGURES_DIR = os.path.join(ROOT_DIR, "report", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


# ─────────────────────────────────────────────
def get_scores(clf, X):
    """
    Safe probability/score extraction.
    """
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X)[:, 1]
    elif hasattr(clf, "decision_function"):
        return clf.decision_function(X)
    else:
        raise ValueError("Model does not support probability or decision_function")


# ─────────────────────────────────────────────
def compute_metrics(y_true, y_pred, y_score, model_name):
    print(f"\n{'='*55}")
    print(f"MODEL: {model_name}")
    print(f"{'='*55}")

    # The classification report is the most important for checking "Normal" recall
    print(classification_report(
        y_true, y_pred,
        target_names=["Normal", "Pneumonia"]
    ))

    auc = roc_auc_score(y_true, y_score)
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred) # Added
    f1_m  = f1_score(y_true, y_pred, average="macro")
    rec = recall_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred)

    print(f"AUC-ROC           : {auc:.4f}")
    print(f"Balanced Accuracy : {bal_acc:.4f}") # Added
    print(f"F1 Macro          : {f1_m:.4f}")

    return {
        "model": model_name,
        "auc": auc,
        "accuracy": acc,
        "balanced_acc": bal_acc,
        "f1_macro": f1_m,
        "recall": rec,
        "precision": pre
    }


# ─────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, name):
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_title(f"Confusion Matrix — {name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Normal", "Pneumonia"])
    ax.set_yticklabels(["Normal", "Pneumonia"])

    thresh = cm.max() / 2

    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, cm[i, j],
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontweight="bold"
            )

    plt.colorbar(im)
    plt.tight_layout()

    path = f"{FIGURES_DIR}/cm_{name.replace(' ', '_')}.png"
    plt.savefig(path, dpi=150)
    plt.close()


# ─────────────────────────────────────────────
def plot_roc_curves(results, y_test):
    plt.figure(figsize=(8, 6))

    colors = ["blue", "green", "orange", "red", "purple", "brown"]

    for i, r in enumerate(results):
        fpr, tpr, _ = roc_curve(y_test, r["score"])
        auc = roc_auc_score(y_test, r["score"])

        plt.plot(
            fpr, tpr,
            color=colors[i % len(colors)],
            lw=2,
            label=f"{r['name']} (AUC={auc:.3f})"
        )

    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend(loc='lower right')

    path = f"{FIGURES_DIR}/roc_curves.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


# ─────────────────────────────────────────────
def plot_comparison(metrics):
    names = [m["model"] for m in metrics]
    f1s   = [m["f1_macro"] for m in metrics]
    aucs  = [m["auc"] for m in metrics]
    b_acc = [m["balanced_acc"] for m in metrics] # Swapped standard acc for balanced

    x = np.arange(len(names))

    plt.figure(figsize=(12, 5))

    plt.bar(x - 0.25, f1s, 0.2, label="F1 Macro")
    plt.bar(x, aucs, 0.2, label="AUC-ROC")
    plt.bar(x + 0.25, b_acc, 0.2, label="Balanced Acc")

    plt.xticks(x, names, rotation=20)
    plt.ylim(0, 1.1) # Leave room for legend
    plt.title("Model Performance Comparison (Imbalance-Aware Metrics)")
    plt.legend(loc='upper right')

    path = f"{FIGURES_DIR}/comparison.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


# ─────────────────────────────────────────────
if __name__ == "__main__":

    eval_data_path = os.path.join(ROOT_DIR, "models", "eval_data.pkl")
    
    if not os.path.exists(eval_data_path):
        print(f"Error: {eval_data_path} not found. Run train.py first.")
    else:
        print("Loading evaluation data...")
        data = joblib.load(eval_data_path)

        X_test = data["X_test_pca"]
        y_test = data["y_test"]
        models = data["all_trained"]

        all_metrics = []
        roc_results = []

        for name, clf in models.items():
            y_pred = clf.predict(X_test)
            y_score = get_scores(clf, X_test)

            metrics = compute_metrics(y_test, y_pred, y_score, name)
            all_metrics.append(metrics)
            roc_results.append({"name": name, "score": y_score})

            plot_confusion_matrix(y_test, y_pred, name)

        # Plot multi-model charts
        plot_roc_curves(roc_results, y_test)
        plot_comparison(all_metrics)

        # FINAL RANKING based on Balanced Accuracy (Best for Medical)
        ranked = sorted(all_metrics, key=lambda x: x["balanced_acc"], reverse=True)

        print("\n" + "="*65)
        print("FINAL RANKING (By Balanced Accuracy)")
        print("="*65)

        for i, m in enumerate(ranked, 1):
            tag = " [WINNER]" if i == 1 else ""
            print(f"{i}. {m['model']:15} | Bal_Acc: {m['balanced_acc']:.4f} | F1: {m['f1_macro']:.4f}{tag}")

        print(f"\nAll artifacts saved to: {FIGURES_DIR}")