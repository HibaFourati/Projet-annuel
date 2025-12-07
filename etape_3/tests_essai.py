# test_cases.py
import numpy as np
from models import SVMModel, RBFN

# Exemple avec 200 échantillons
X = np.random.rand(200, 5)
y = np.array([1]*100 + [-1]*100)

# Shuffle avant split
indices = np.arange(len(X))
np.random.shuffle(indices)
X = X[indices]
y = y[indices]

# Split train/test 80/20
split = int(0.8 * len(X))
X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]

# Fonction de test
def test_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred_train = model.predict_class(X_train)
    y_pred_test = model.predict_class(X_test)
    print(f"\n{type(model).__name__} - Train accuracy: {np.mean(y_pred_train == y_train)*100:.1f}%")
    print(f"{type(model).__name__} - Test accuracy: {np.mean(y_pred_test == y_test)*100:.1f}%")
    return y_pred_test

# Tester SVM
svm_model = SVMModel()
test_model(svm_model, X_train, y_train, X_test, y_test)

# Tester RBFN
rbfn_model = RBFN(centers=10, gamma=0.1)
test_model(rbfn_model, X_train, y_train, X_test, y_test)
