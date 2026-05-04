import numpy as np
import joblib
import os
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

from preprocess import load_dataset
from features import build_feature_matrix


# ── Config ──────────────────────────────────────────────────────
DATA_TRAIN_DIR = "data/chest_xray/train"
DATA_TEST_DIR  = "data/chest_xray/test"
DATA_VAL_DIR   = "data/chest_xray/val"    # will be merged into train
MODEL_OUT      = "models/pneumonia_model.pkl"
RANDOM_STATE   = 42
IMG_SIZE       = (64, 64)
PCA_VARIANCE   = 0.95
SMOTE_K        = 5


# ── Step 1: Load Data ───────────────────────────────────────────
def load_all_data():
    print("Loading training data...")
    X_train_raw, y_train, _ = load_dataset(DATA_TRAIN_DIR, IMG_SIZE)

    # Merge the tiny 16-image val set into train
    print("Loading & merging validation data into train...")
    X_val_raw, y_val, _     = load_dataset(DATA_VAL_DIR,   IMG_SIZE)
    X_combined = np.concatenate([X_train_raw, X_val_raw], axis=0)
    y_combined = np.concatenate([y_train,     y_val],     axis=0)

    print("Loading test data...")
    X_test_raw, y_test, _   = load_dataset(DATA_TEST_DIR,  IMG_SIZE)

    print(f"Combined train: {X_combined.shape[0]} images | Test: {X_test_raw.shape[0]} images")
    return X_combined, y_combined, X_test_raw, y_test


# ── Step 2: Feature Extraction ──────────────────────────────────
def extract_all_features(X_train_raw, X_test_raw):
    print("Extracting features from train set...")
    X_train_feats = build_feature_matrix(X_train_raw)
    print("Extracting features from test set...")
    X_test_feats  = build_feature_matrix(X_test_raw)
    return X_train_feats, X_test_feats


# ── Step 3: Train/Val Split ─────────────────────────────────────
def split_data(X, y):
    return train_test_split(X, y, test_size=0.20,
                            stratify=y, random_state=RANDOM_STATE)


# ── Step 4: PCA ─────────────────────────────────────────────────
def apply_pca(X_train, X_val, X_test):
    print(f"Applying PCA (retaining {PCA_VARIANCE*100}% variance)...")
    pca = PCA(n_components=PCA_VARIANCE, random_state=RANDOM_STATE)
    pca.fit(X_train)

    X_train_pca = pca.transform(X_train)
    X_val_pca   = pca.transform(X_val)
    X_test_pca  = pca.transform(X_test)

    print(f"PCA reduced to {X_train_pca.shape[1]} components")
    return pca, X_train_pca, X_val_pca, X_test_pca


# ── Step 5: SMOTE ───────────────────────────────────────────────
def apply_smote(X_train_pca, y_train):
    print(f"Before SMOTE: {np.bincount(y_train)}")
    smote = SMOTE(k_neighbors=SMOTE_K, random_state=RANDOM_STATE)
    X_bal, y_bal = smote.fit_resample(X_train_pca, y_train)
    print(f"After  SMOTE: {np.bincount(y_bal)}")
    return X_bal, y_bal


# ── Step 6: Define Models ───────────────────────────────────────
def get_models():
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
        "SVM (RBF)": SVC(
            kernel="rbf", C=1.0, gamma="scale",
            probability=True, class_weight="balanced", random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, random_state=RANDOM_STATE),
        "KNN": KNeighborsClassifier(n_neighbors=7),
    }


# ── Step 7: Train All Models ────────────────────────────────────
def train_all(models, X_train, y_train):
    trained = {}
    for name, clf in models.items():
        print(f"Training {name}...")
        clf.fit(X_train, y_train)
        trained[name] = clf
    return trained


# ── Step 8: Hyperparameter Tuning (SVM + RF) ───────────────────
def tune_models(X_train, y_train):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # SVM
    print("Tuning SVM...")
    svm_grid = {"C": [0.1, 1, 10, 100], "gamma": ["scale", "auto", 0.01, 0.001]}
    svm_gs = GridSearchCV(
        SVC(kernel="rbf", probability=True, class_weight="balanced"),
        svm_grid, scoring="f1", cv=cv, n_jobs=-1, verbose=1)
    svm_gs.fit(X_train, y_train)
    print(f"Best SVM params: {svm_gs.best_params_}")

    # Random Forest
    print("Tuning Random Forest...")
    rf_grid = {"n_estimators": [100, 200], "max_depth": [None, 10, 20]}
    rf_gs = GridSearchCV(
        RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE),
        rf_grid, scoring="f1", cv=cv, n_jobs=-1, verbose=1)
    rf_gs.fit(X_train, y_train)
    print(f"Best RF params: {rf_gs.best_params_}")

    return svm_gs.best_estimator_, rf_gs.best_estimator_


# ── Step 9: Save Best Model ─────────────────────────────────────
def save_model(pca, clf, path=MODEL_OUT):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump({"pca": pca, "clf": clf}, path)
    print(f"Model saved to {path}")


# ── Main ────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 1. Load
    X_raw, y, X_test_raw, y_test = load_all_data()

    # 2. Features
    X_feats, X_test_feats = extract_all_features(X_raw, X_test_raw)

    # 3. Split
    X_tr, X_val, y_tr, y_val = split_data(X_feats, y)

    # 4. PCA
    pca, X_tr_pca, X_val_pca, X_test_pca = apply_pca(X_tr, X_val, X_test_feats)

    # 5. SMOTE
    X_bal, y_bal = apply_smote(X_tr_pca, y_tr)

    # 6 & 7. Train all baseline models
    models  = get_models()
    trained = train_all(models, X_bal, y_bal)

    # 8. Tune best candidates
    best_svm, best_rf = tune_models(X_bal, y_bal)

    # 9. Save best model (SVM by default — change as needed)
    save_model(pca, best_svm)

    print("\nTraining complete. Run evaluate.py for full results.")
    # Store test data for evaluate.py
    joblib.dump({"X_test": X_test_pca, "y_test": y_test,
                 "X_val": X_val_pca,   "y_val":  y_val,
                 "all_trained": trained,
                 "best_svm": best_svm, "best_rf": best_rf},
                "models/eval_data.pkl")
