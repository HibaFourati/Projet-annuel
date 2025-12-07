# train_save.py
import numpy as np
import pickle
from sklearn.decomposition import PCA
from Charger_dataset import X, y
from models import SVMModel, RBFN

# Normalisation
X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-6)

# Split train/test
split = int(0.8 * len(X))
X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]

# --- PCA ---
pca = PCA(n_components=100)  # réduire 4096 -> 100
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# --- SVM ---
svm_model = SVMModel()
svm_model.fit(X_train_pca, y_train)
with open("svm_model.pkl", "wb") as f:
    pickle.dump(svm_model, f)

# --- RBFN ---
rbfn_model = RBFN(centers=20, gamma=0.01)
rbfn_model.fit(X_train_pca, y_train)
with open("rbfn_model.pkl", "wb") as f:
    pickle.dump(rbfn_model, f)

# --- Sauvegarder PCA ---
with open("pca.pkl", "wb") as f:
    pickle.dump(pca, f)

print("Modèles et PCA entraînés et sauvegardés.")
