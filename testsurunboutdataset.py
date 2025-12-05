import ctypes
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from pathlib import Path

# ======== WRAPPER RUST ========
class LinearModel:
    def __init__(self, input_dim: int, learning_rate: float = 0.01):
        self.lib = ctypes.CDLL("./target/release/neural_networks.dll")
        self.lib.linear_model_new.argtypes = [ctypes.c_size_t, ctypes.c_double]
        self.lib.linear_model_new.restype = ctypes.c_void_p
        self.lib.linear_model_delete.argtypes = [ctypes.c_void_p]
        self.lib.linear_model_fit.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), 
            ctypes.POINTER(ctypes.c_double), ctypes.c_size_t, 
            ctypes.c_size_t, ctypes.c_size_t
        ]
        self.lib.linear_model_fit.restype = ctypes.c_double
        self.lib.linear_model_predict_batch.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double), ctypes.c_size_t, ctypes.c_size_t
        ]
        self.lib.linear_model_get_weights.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_double)
        ]
        self.lib.linear_model_get_bias.argtypes = [ctypes.c_void_p]
        self.lib.linear_model_get_bias.restype = ctypes.c_double
        self.model_ptr = self.lib.linear_model_new(input_dim, learning_rate)
        self.input_dim = input_dim
    
    def __del__(self):
        if hasattr(self, 'model_ptr'):
            self.lib.linear_model_delete(self.model_ptr)
    
    def fit(self, X: np.ndarray, y: np.ndarray, max_iterations: int = 1000) -> float:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        n_samples, n_features = X.shape
        X_ptr = X.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        y_ptr = y.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        return self.lib.linear_model_fit(self.model_ptr, X_ptr, y_ptr, n_samples, n_features, max_iterations)
    
    def predict_class(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        n_samples, n_features = X.shape
        results = np.zeros(n_samples, dtype=np.float64)
        X_ptr = X.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        results_ptr = results.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        self.lib.linear_model_predict_batch(self.model_ptr, X_ptr, results_ptr, n_samples, n_features)
        return np.where(results >= 0, 1, -1)
    
    def get_weights(self) -> np.ndarray:
        weights = np.zeros(self.input_dim, dtype=np.float64)
        weights_ptr = weights.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        self.lib.linear_model_get_weights(self.model_ptr, weights_ptr)
        return weights
    
    def get_bias(self) -> float:
        return self.lib.linear_model_get_bias(self.model_ptr)

# ======== DATASET RÉEL ========
def extraire_features(img_path, taille=(64, 64)):
    try:
        img = cv2.imread(str(img_path))
        img = cv2.resize(img, taille)
        features = []
        for canal in range(3):
            hist = cv2.calcHist([img], [canal], None, [16], [0, 256])
            hist = hist.flatten() / (hist.sum() + 1e-6)
            features.extend(hist)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        features.append(gray.mean())
        features.append(gray.std())
        features.append(np.median(gray))
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        features.append(np.mean(np.abs(sobelx)))
        features.append(np.mean(np.abs(sobely)))
        moments = cv2.moments(gray)
        hu_moments = cv2.HuMoments(moments).flatten()
        features.extend(np.log(np.abs(hu_moments) + 1e-6))
        return np.array(features, dtype=np.float64)
    except:
        return None

def charger_bout_dataset():
    X, y = [], []
    for instrument, label in [('guitare', 1), ('piano', -1)]:
        path = Path(f"dataset/{instrument}")
        if path.exists():
            images = list(path.glob("*.[pj][np]g"))[:5]  # 5 images max
            for img in images:
                features = extraire_features(img)
                if features is not None:
                    X.append(features)
                    y.append(label)
    return np.array(X), np.array(y)

# ======== TESTS ========
def test_sur_dataset():
    print("\n1. APPLICATION SUR BOUT DE DATASET")
    print("="*40)
    X, y = charger_bout_dataset()
    if len(X) == 0:
        print("Aucune image trouvée !")
        return
    
    # Normalisation simple
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-6)
    
    # Split simple (80/20)
    split = int(0.8 * len(X))
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]
    
    print(f"Images: {len(X)} (train: {len(X_train)}, test: {len(X_test)})")
    print(f"Features: {X.shape[1]}")
    
    # Modèle
    model = LinearModel(input_dim=X_train.shape[1], learning_rate=0.01)
    error = model.fit(X_train, y_train, max_iterations=1000)
    train_acc = np.mean(model.predict_class(X_train) == y_train) * 100
    test_acc = np.mean(model.predict_class(X_test) == y_test) * 100
    
    print(f"Erreur: {error:.6f}")
    print(f"Accuracy - Train: {train_acc:.1f}%, Test: {test_acc:.1f}%")
    
    # Visualisation simple
    if X.shape[1] >= 2:
        plt.figure(figsize=(8, 6))
        colors = ['blue' if label == 1 else 'red' for label in y]
        plt.scatter(X[:, 0], X[:, 1], c=colors, alpha=0.6)
        plt.title("Visualisation des 2 premières features")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.grid(True, alpha=0.3)
        plt.show()
    
    return test_acc

def test_transformation_non_lineaire():
    print("\n2. TRANSFORMATION NON-LINÉAIRE POUR XOR")
    print("="*40)
    
    # XOR
    X = np.array([[1, 0], [0, 1], [0, 0], [1, 1]], dtype=np.float64)
    y = np.array([1, 1, -1, -1], dtype=np.float64)
    
    # Sans transformation
    model1 = LinearModel(input_dim=2, learning_rate=0.1)
    model1.fit(X, y, max_iterations=500)
    acc1 = np.mean(model1.predict_class(X) == y) * 100
    
    # Avec transformation
    X_transformed = np.column_stack([
        X, X[:, 0]*X[:, 1], X[:, 0]**2, X[:, 1]**2,
        np.sin(X[:, 0]), np.exp(-X[:, 0]**2)
    ])
    model2 = LinearModel(input_dim=X_transformed.shape[1], learning_rate=0.05)
    model2.fit(X_transformed, y, max_iterations=1000)
    acc2 = np.mean(model2.predict_class(X_transformed) == y) * 100
    
    print(f"Sans transformation: {acc1:.1f}%")
    print(f"Avec transformation: {acc2:.1f}%")
    print(f"Amélioration: {acc2-acc1:+.1f}%")
    
    # Visualisation
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], c=['blue', 'blue', 'red', 'red'])
    plt.title("XOR Original")
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.bar(['Sans', 'Avec'], [acc1, acc2], color=['red', 'green'])
    plt.ylim([0, 110])
    plt.title("Comparaison des accuracies")
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return acc2

# ======== MAIN ========
if __name__ == "__main__":
    print("PROJET - MODÈLE LINÉAIRE RUST")
    print("="*40)
    
    # 1. Sur dataset
    acc_dataset = test_sur_dataset()
    
    # 2. Transformation non-linéaire
    acc_xor = test_transformation_non_lineaire()
    
    # 3. Résumé
    print("\n" + "="*40)
    print("RÉSUMÉ DES RÉSULTATS")
    print("="*40)
    print(f"1. Dataset: {acc_dataset:.1f}% sur test")
    print(f"2. XOR transformé: {acc_xor:.1f}%")
    print("\nPoints validés:")
    print("✓ Modèle linéaire testé sur dataset réel")
    print("✓ Transformation non-linéaire pour cas 'KO'")