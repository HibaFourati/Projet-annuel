import ctypes
import numpy as np
import cv2
from pathlib import Path
import random

class RBF:
    def __init__(self, n_inputs: int, n_centers: int = 80, learning_rate: float = 0.008, sigma: float = 0.0):
        self.lib = ctypes.CDLL("./target/debug/neural_networks.dll")
        
        class RBFConfig(ctypes.Structure):
            _fields_ = [
                ("n_inputs", ctypes.c_uint),
                ("n_centers", ctypes.c_uint),
                ("n_outputs", ctypes.c_uint),
                ("learning_rate", ctypes.c_double),
                ("sigma", ctypes.c_double)
            ]
        
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
        
        self.model_ptr = self.lib.rbf_new(ctypes.byref(config))
    
    def fit(self, X: np.ndarray, y: np.ndarray, max_iterations: int = 1500) -> float:
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
        features.extend([
            gray.mean() / 255.0,
            gray.std() / 255.0,
            np.median(gray) / 255.0,
        ])
        
        return np.array(features, dtype=np.float64)
    except:
        return None

def entrainer_rapide():

    print("="*50)
    print("ENTRAÎNEMENT & TEST 10 IMAGES")
    print("="*50)
    

    X, y = [], []
    for instrument, label in [('guitare', 1), ('piano', -1), ('violon', 0)]:
        path = Path(f"dataset/{instrument}")
        if path.exists():
            images = list(path.glob("*.jpg")) + list(path.glob("*.png"))
            for img in images:
                features = extraire_features(img)
                if features is not None:
                    X.append(features)
                    y.append(label)
    
    if not X:
        print(" Aucune image chargée")
        return
    
    X = np.array(X)
    y = np.array(y)
    

    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std < 1e-8] = 1.0
    X_norm = (X - mean) / std
    
    print(f" Données: {len(X)} images, {X.shape[1]} features")
    

    models = {}
    

    y_guitare = np.where(y == 1, 1.0, -1.0)
    models['guitare'] = RBF(n_inputs=X.shape[1], n_centers=120, learning_rate=0.008, sigma=0.0)
    loss_g = models['guitare'].fit(X_norm, y_guitare, max_iterations=2000)
    
    y_piano = np.where(y == -1, 1.0, -1.0)
    models['piano'] = RBF(n_inputs=X.shape[1], n_centers=150, learning_rate=0.008, sigma=0.0)
    loss_p = models['piano'].fit(X_norm, y_piano, max_iterations=2000)
    

    y_violon = np.where(y == 0, 1.0, -1.0)
    models['violon'] = RBF(n_inputs=X.shape[1], n_centers=100, learning_rate=0.008, sigma=0.0)
    loss_v = models['violon'].fit(X_norm, y_violon, max_iterations=2000)
    
    print(f" Entraînement terminé")
    print(f"   Loss: G={loss_g:.4f}, P={loss_p:.4f}, V={loss_v:.4f}")

    scores = {name: model.predict(X_norm) for name, model in models.items()}
    
    y_pred = []
    for i in range(len(X)):
        best = max(scores.items(), key=lambda x: x[1][i])[0]
        y_pred.append(1 if best == 'guitare' else -1 if best == 'piano' else 0)
    
    accuracy = np.mean(np.array(y_pred) == y) * 100
    print(f"\n Accuracy globale: {accuracy:.1f}%")

    print(" TEST SUR 10 IMAGES ALÉATOIRES")
    
    correct = 0
    total = 0
    
    for test_num in range(10):

        instrument = random.choice(['guitare', 'piano', 'violon'])
        path = Path(f"dataset/{instrument}")
        
        if not path.exists():
            continue
            
        images = list(path.glob("*.jpg")) + list(path.glob("*.png"))
        if not images:
            continue
            

        img_path = random.choice(images)
        
        features = extraire_features(img_path)
        if features is None:
            continue
            
        features_norm = (features - mean) / std
        
        scores = {}
        for nom, modele in models.items():
            scores[nom] = modele.predict(features_norm.reshape(1, -1))[0]
        

        meilleur_classe = max(scores.items(), key=lambda x: x[1])[0]
        
        if meilleur_classe == 'guitare':
            pred_nom = 'Guitare'
        elif meilleur_classe == 'piano':
            pred_nom = 'Piano'
        else:
            pred_nom = 'Violon'
        
        is_correct = pred_nom.lower() == instrument
        

        print(f"\nTest {test_num+1}:")
        print(f"  Image: {img_path.name}")
        print(f"  Réel: {instrument.capitalize()}")
        print(f"  Prédit: {pred_nom}")
        print(f"  Scores: G={scores['guitare']:.3f}, P={scores['piano']:.3f}, V={scores['violon']:.3f}")
        
        if is_correct:
            print(f"   CORRECT")
            correct += 1
        else:
            print(f"   ERREUR")
        
        total += 1
    

    print(f" RÉSULTATS: {correct}/{total} corrects ({correct/total*100:.1f}%)")
    print(f" Accuracy globale: {accuracy:.1f}%")
    
    return models, mean, std


if __name__ == "__main__":
    entrainer_rapide()