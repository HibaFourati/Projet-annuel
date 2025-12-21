# Test_model.py
import pickle
import numpy as np
from Charger_dataset import X, y

# Chargement
with open("svm_model.pkl", "rb") as f:
    svm = pickle.load(f)

with open("rbfn_model.pkl", "rb") as f:
    rbfn = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Préparation
X = scaler.transform(X)

split = int(0.8 * len(X))
X_test, y_test = X[split:], y[split:]

# Évaluation
svm_acc = (svm.predict_class(X_test) == y_test).mean() * 100
rbfn_acc = (rbfn.predict_class(X_test) == y_test).mean() * 100

print(f"SVM accuracy: {svm_acc:.1f}%")
print(f"RBFN accuracy: {rbfn_acc:.1f}%")
