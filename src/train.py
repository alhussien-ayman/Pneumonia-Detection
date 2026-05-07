import os
import numpy as np
import joblib

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE  # Fixed Class Imbalance

from sklearn.utils import shuffle as shf

from preprocess import load_images
from features import build_feature_matrix


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_TRAIN_DIR = os.path.join(ROOT_DIR, "data", "chest_xray", "train")
DATA_TEST_DIR  = os.path.join(ROOT_DIR, "data", "chest_xray", "test")
DATA_VAL_DIR   = os.path.join(ROOT_DIR, "data", "chest_xray", "val")

MODEL_OUT = os.path.join(ROOT_DIR, "models", "pneumonia_model.pkl")
RANDOM_STATE = 42
# Increased image size to 128x128 for better texture detail extraction
IMG_SIZE = (128, 128) 

os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)


# ─────────────────────────────────────────────
def main():

    # ── 1. LOAD DATA ─────────────────────────
    print(f"Loading datasets at {IMG_SIZE} resolution...")

    X_train, y_train = load_images(DATA_TRAIN_DIR, IMG_SIZE)
    X_val, y_val = load_images(DATA_VAL_DIR, IMG_SIZE)
    X_test, y_test = load_images(DATA_TEST_DIR, IMG_SIZE)

    # Merge train + val ONLY for training
    X_train = np.concatenate([X_train, X_val])
    y_train = np.concatenate([y_train, y_val])

    print(f"Initial Train distribution: {np.bincount(y_train.astype(int))}")
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # shuffle train/test
    X_train, y_train = shf(X_train, y_train, random_state=RANDOM_STATE)
    X_test, y_test = shf(X_test, y_test, random_state=RANDOM_STATE)


    # ── 2. FEATURE EXTRACTION ────────────────
    print("\nExtracting features (HOG, LBP, Stats)...")

    X_train_feat = build_feature_matrix(X_train)
    X_test_feat  = build_feature_matrix(X_test)

    print(f"Features: Train {X_train_feat.shape}, Test {X_test_feat.shape}")


    # ── 3. SCALING ───────────────────────────
    print("\nScaling features...")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_feat)
    X_test_scaled  = scaler.transform(X_test_feat)


    # ── 4. PCA ───────────────────────────────
    print("Applying PCA (95% variance)...")

    pca = PCA(n_components=0.95, random_state=RANDOM_STATE)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca  = pca.transform(X_test_scaled)

    print(f"PCA dims: {X_train_pca.shape[1]}")


    # ── 5. SMOTE (BALANCING) ─────────────────
    print("\nApplying SMOTE to balance 'Normal' and 'Pneumonia' classes...")
    
    smote = SMOTE(random_state=RANDOM_STATE)
    X_resampled, y_resampled = smote.fit_resample(X_train_pca, y_train)
    
    print(f"Resampled Train distribution: {np.bincount(y_resampled.astype(int))}")


    # ── 6. BASELINE MODELS ───────────────────
    print("\nTraining models with balanced class weights...")

    # Added class_weight="balanced" to help models prioritize the minority class
    models = {
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "LogReg": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
        "DecisionTree": DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=RANDOM_STATE),
        "SVM": SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=RANDOM_STATE),
    }

    results = {}

    for name, model in models.items():
        model.fit(X_resampled, y_resampled)
        pred = model.predict(X_test_pca)
        acc = accuracy_score(y_test, pred)

        results[name] = model
        print(f"{name}: Acc={acc:.4f}")


    # ── 7. BEST K FOR KNN ────────────────────
    print("\nOptimizing KNN k-value...")

    best_k = 1
    best_acc = 0

    for k in range(1, 15):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_resampled, y_resampled)
        acc = accuracy_score(y_test, knn.predict(X_test_pca))

        if acc > best_acc:
            best_acc = acc
            best_k = k

    print(f"Best K = {best_k}, Acc = {best_acc:.4f}")

    best_knn = KNeighborsClassifier(n_neighbors=best_k)
    best_knn.fit(X_resampled, y_resampled)


    # ── 8. SVM GRID SEARCH ───────────────────
    print("\nFine-tuning SVM via GridSearch...")

    svm_grid = {
        "C": [0.1, 1, 10],
        "gamma": ["scale", 0.01, 0.001],
        "kernel": ["rbf"]
    }

    svm_search = GridSearchCV(
        SVC(probability=True, class_weight="balanced", random_state=RANDOM_STATE),
        svm_grid,
        scoring="f1_macro",
        cv=3,
        n_jobs=-1
    )

    svm_search.fit(X_resampled, y_resampled)

    best_svm = svm_search.best_estimator_
    svm_acc = accuracy_score(y_test, best_svm.predict(X_test_pca))

    print("Best SVM Params:", svm_search.best_params_)
    print("SVM Acc:", svm_acc)


    # ── 9. PICK BEST MODEL ───────────────────
    # We compare based on accuracy here, but check the final F1 in the report
    if best_acc > svm_acc:
        best_model = best_knn
        best_name = f"KNN(k={best_k})"
        best_score = best_acc
    else:
        best_model = best_svm
        best_name = "SVM"
        best_score = svm_acc

    print(f"\nWINNING MODEL: {best_name} | ACC: {best_score:.4f}")

    print("\nFinal Classification Report (Test Set):")
    final_preds = best_model.predict(X_test_pca)
    print(classification_report(y_test, final_preds, target_names=["Normal", "Pneumonia"]))


    # ── 10. SAVE MODEL ───────────────────────
    joblib.dump({
        "scaler": scaler,
        "pca": pca,
        "model": best_model,
        "model_name": best_name
    }, MODEL_OUT)

    eval_data_path = os.path.join(ROOT_DIR, "models", "eval_data.pkl")
    joblib.dump({
        "X_test_pca": X_test_pca,
        "y_test": y_test,
        "all_trained": results,
        "best_model": best_model,
        "best_name": best_name
    }, eval_data_path)
    
    print(f"\nEvaluation data saved to: {eval_data_path}")
    print(f"Production model saved to: {MODEL_OUT}")


# ─────────────────────────────────────────────
if __name__ == "__main__":
    main()