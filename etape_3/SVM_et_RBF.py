# models.py
import numpy as np
from sklearn.svm import SVC


# SVM

class SVMModel:
    def __init__(self, kernel="rbf", C=1.0):
        self.model = SVC(kernel=kernel, C=C)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict_class(self, X):
        return self.model.predict(X)



# RBFN (simple)

class RBFN:
    def __init__(self, centers=10, gamma=0.001):
        self.centers = centers
        self.gamma = gamma

    def _rbf(self, X, c):
        return np.exp(-self.gamma * np.linalg.norm(X - c, axis=1) ** 2)

    def fit(self, X, y):
        # sauvegarde des classes
        self.classes_ = np.unique(y)

        # centres = premiers points
        self.centroids = X[:self.centers]

        # matrice RBF
        G = np.column_stack([self._rbf(X, c) for c in self.centroids])

        # one-hot encoding des labels
        Y = np.zeros((len(y), len(self.classes_)))
        for i, label in enumerate(self.classes_):
            Y[:, i] = (y == label).astype(float)

        # calcul des poids
        self.weights = np.linalg.pinv(G) @ Y

    def predict_class(self, X):
        G = np.column_stack([self._rbf(X, c) for c in self.centroids])
        scores = G @ self.weights
        return self.classes_[np.argmax(scores, axis=1)]
