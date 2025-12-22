# train_save.py
import pickle
import numpy as np
from models import SVMModel, RBFN
from Charger_dataset import X, y
from sklearn.preprocessing import StandardScaler

# Normalisation
scaler = StandardScaler()
X = scaler.fit_transform(X)

split = int(0.8 * len(X))
X_train, y_train = X[:split], y[:split]

# SVM
svm = SVMModel()
svm.fit(X_train, y_train)

# RBFN
rbfn = RBFN(centers=15, gamma=0.001)
rbfn.fit(X_train, y_train)

# Sauvegarde
with open("svm_model.pkl", "wb") as f:
    pickle.dump(svm, f)

with open("rbfn_model.pkl", "wb") as f:
    pickle.dump(rbfn, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Modèles entraînés et sauvegardés.")
