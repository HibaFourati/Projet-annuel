# tests_essai.py
import numpy as np
from models import SVMModel, RBFN

# Données
X = np.random.rand(50, 5)
y = np.array([0]*25 + [1]*25)

split = int(0.8 * len(X))
X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]

def test(model):
    model.fit(X_train, y_train)
    acc = (model.predict_class(X_test) == y_test).mean() * 100
    print(f"{model.__class__.__name__} accuracy : {acc:.1f}%")

print("=== Cas de tests simples ===")
test(SVMModel())
test(RBFN())
