import ctypes
import numpy as np
import cv2
from pathlib import Path
import random

class RBF:
    def __init__(self, n_inputs: int, n_centers: int = 50, learning_rate: float = 0.01, sigma: float = 1.0):
        self.lib = ctypes.CDLL("./target/release/neural_networks.dll")
        
        class RBFConfig(ctypes.Structure):
            _fields_ = [
                ("n_inputs", ctypes.c_uint),
                ("n_centers", ctypes.c_uint),
                ("n_outputs", ctypes.c_uint),
                ("learning_rate", ctypes.c_double),
                ("sigma", ctypes.c_double)
            ]
        
        self.RBFConfig = RBFConfig
        
        config = RBFConfig(
            n_inputs=n_inputs,
            n_centers=n_centers,
            n_outputs=1,
            learning_rate=learning_rate,
            sigma=sigma
        )
        
        self.lib.rbf_new.argtypes = [ctypes.POINTER(RBFConfig)]
        self.lib.rbf_new.restype = ctypes.c_void_p
        self.lib.rbf_delete.argtypes = [ctypes.c_void_p]
        self.lib.rbf_fit.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), 
            ctypes.POINTER(ctypes.c_double), ctypes.c_size_t, 
            ctypes.c_size_t, ctypes.c_size_t
        ]
        self.lib.rbf_fit.restype = ctypes.c_double
        self.lib.rbf_predict_batch.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double), ctypes.c_size_t, ctypes.c_size_t
        ]
        
        # Nouvelle fonction debug
        self.lib.rbf_debug_predict.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_size_t, ctypes.c_size_t
        ]
        self.lib.rbf_debug_predict.restype = ctypes.c_double
        
        self.model_ptr = self.lib.rbf_new(ctypes.byref(config))
    
    def __del__(self):
        if hasattr(self, 'model_ptr'):
            self.lib.rbf_delete(self.model_ptr)
    
    def fit(self, X: np.ndarray, y: np.ndarray, max_iterations: int = 2000) -> float:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        n_samples, n_features = X.shape
        X_ptr = X.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        y_ptr = y.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        
        return self.lib.rbf_fit(self.model_ptr, X_ptr, y_ptr, n_samples, n_features, max_iterations)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        n_samples, n_features = X.shape
        predictions = np.zeros(n_samples, dtype=np.float64)
        
        X_ptr = X.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        pred_ptr = predictions.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        
        self.lib.rbf_predict_batch(self.model_ptr, X_ptr, pred_ptr, n_samples, n_features)
        return predictions
    
    def debug_predict(self, X: np.ndarray) -> float:
        """Fonction debug pour voir les prédictions brutes"""
        X = np.asarray(X, dtype=np.float64)
        
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        n_samples, n_features = X.shape
        X_ptr = X.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        
        return self.lib.rbf_debug_predict(self.model_ptr, X_ptr, 1, n_features)

# Test avec vos données
def test_rbf_corrige():
    print("="*60)
    print("TEST RBF CORRIGÉ")
    print("="*60)
    
    # Charger quelques données
    X, y = [], []
    for instrument, label in [('guitare', 1), ('piano', -1), ('violon', 0)]:
        path = Path(f"dataset/{instrument}")
        if path.exists():
            images = list(path.glob("*.[pj][np]g"))[:5]
            for img in images:
                features = extraire_features(img)
                if features is not None:
                    X.append(features)
                    y.append(label)
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"Données: {len(X)} images, {X.shape[1]} features")
    
    # Normalisation
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0) + 1e-8
    X_norm = (X - X_mean) / X_std
    
    # Tester différentes configurations
    for sigma in [0.5, 1.0, 2.0]:
        for lr in [0.001, 0.005, 0.01]:
            print(f"\nSigma={sigma}, LR={lr}:")
            
            # Modèle pour guitare
            rbf = RBF(n_inputs=X.shape[1], n_centers=20, learning_rate=lr, sigma=sigma)
            y_guitare = np.where(y == 1, 1.0, -1.0)
            
            loss = rbf.fit(X_norm, y_guitare, max_iterations=500)
            print(f"  Loss: {loss:.4f}")
            
            # Debug prédiction
            debug_val = rbf.debug_predict(X_norm[0:1])
            print(f"  Debug prediction: {debug_val:.4f}")
            
            # Prédictions
            preds = rbf.predict(X_norm)
            print(f"  Prédictions range: [{preds.min():.3f}, {preds.max():.3f}]")
            
            # Convertir en classes
            pred_classes = np.where(preds > 0, 1, -1)
            accuracy = np.mean(pred_classes == y_guitare) * 100
            print(f"  Accuracy: {accuracy:.1f}%")

def extraire_features(img_path):
    try:
        img = cv2.imread(str(img_path))
        if img is None:
            return None
        
        img = cv2.resize(img, (80, 80))
        features = []
        
        for canal in range(3):
            hist = cv2.calcHist([img], [canal], None, [12], [0, 256])
            hist = hist.flatten() / (hist.sum() + 1e-6)
            features.extend(hist)
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        features.extend([gray.mean()/255.0, gray.std()/255.0])
        
        return np.array(features, dtype=np.float64)
    except:
        return None

if __name__ == "__main__":
    test_rbf_corrige()