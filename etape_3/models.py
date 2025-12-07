# models.py
import numpy as np
from sklearn.svm import SVC
from sklearn.base import BaseEstimator

# --- SVM ---
class SVMModel(BaseEstimator):
    def __init__(self, kernel='rbf', C=1.0):
        self.kernel = kernel
        self.C = C
        self.model = SVC(kernel=self.kernel, C=self.C, probability=True)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self.model

    def predict_class(self, X):
        return self.model.predict(X)

# --- RBFN multiclasse ---
class RBFN:
    def __init__(self, centers=10, gamma=0.01):
        self.centers = centers
        self.gamma = gamma
        self.centroids = None
        self.weights = None

    def _rbf(self, X, centroid):
        return np.exp(-self.gamma * np.linalg.norm(X - centroid, axis=1)**2)

    def fit(self, X, y):
        # One-hot encoder y
        self.classes_ = np.unique(y)
        y_onehot = np.zeros((len(y), len(self.classes_)))
        for i, c in enumerate(self.classes_):
            y_onehot[y == c, i] = 1

        # Choisir centres aléatoires
        idx = np.random.choice(len(X), self.centers, replace=False)
        self.centroids = X[idx]

        G = np.column_stack([self._rbf(X, c) for c in self.centroids])
        self.weights = np.linalg.pinv(G).dot(y_onehot)
        return self

    def predict_class(self, X):
        G = np.column_stack([self._rbf(X, c) for c in self.centroids])
        y_pred = G.dot(self.weights)
        return self.classes_[np.argmax(y_pred, axis=1)]
