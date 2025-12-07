# Test_model.py
import numpy as np
import pickle
from Charger_dataset import X, y

# Normalisation
X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-6)

# Split train/test
split = int(0.8 * len(X))
X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]

# --- Charger PCA et réduire dimension ---
with open("pca.pkl", "rb") as f:
    pca = pickle.load(f)
X_test_pca = pca.transform(X_test)

# --- Charger modèles ---
with open("svm_model.pkl", "rb") as f:
    svm_model = pickle.load(f)
with open("rbfn_model.pkl", "rb") as f:
    rbfn_model = pickle.load(f)

# --- Tester SVM ---
y_pred_svm = svm_model.predict_class(X_test_pca)
acc_svm = np.mean(y_pred_svm == y_test) * 100
print(f"SVM accuracy: {acc_svm:.1f}%")

# --- Tester RBFN ---
y_pred_rbfn = rbfn_model.predict_class(X_test_pca)
acc_rbfn = np.mean(y_pred_rbfn == y_test) * 100
print(f"RBFN accuracy: {acc_rbfn:.1f}%")
