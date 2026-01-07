import ctypes
import numpy as np
import cv2
from pathlib import Path
import random
import time

class SVMConfig(ctypes.Structure):
    _fields_ = [
        ("n_inputs", ctypes.c_uint),
        ("learning_rate", ctypes.c_double),
        ("c", ctypes.c_double),
        ("max_iterations", ctypes.c_uint)
    ]

class CorrectSVM:
    def __init__(self, n_inputs: int, learning_rate: float = 0.02, c: float = 1.0):
        self.lib = ctypes.CDLL("./target/release/libneural_networks.so")
        
        self.config = SVMConfig(
            n_inputs=n_inputs,
            learning_rate=learning_rate,
            c=c,
            max_iterations=5000  
        )
        self.lib.svm_new.argtypes = [ctypes.POINTER(SVMConfig)]
        self.lib.svm_new.restype = ctypes.c_void_p
        self.lib.svm_delete.argtypes = [ctypes.c_void_p]
        self.lib.svm_fit.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t
        ]
        self.lib.svm_fit.restype = ctypes.c_double
        self.lib.svm_predict_batch.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
            ctypes.c_size_t
        ]
        self.lib.svm_predict_probability.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
            ctypes.c_size_t
        ]
        self.lib.svm_accuracy.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
            ctypes.c_size_t
        ]
        self.lib.svm_accuracy.restype = ctypes.c_double
        
        self.model_ptr = self.lib.svm_new(ctypes.byref(self.config))
    
    def __del__(self):
        if hasattr(self, 'model_ptr'):
            self.lib.svm_delete(self.model_ptr)
    
    def fit(self, X: np.ndarray, y: np.ndarray, max_iterations: int = 5000) -> float:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        n_samples, n_features = X.shape
        X_ptr = X.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        y_ptr = y.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        
        return self.lib.svm_fit(self.model_ptr, X_ptr, y_ptr, n_samples, n_features, max_iterations)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        n_samples, n_features = X.shape
        predictions = np.zeros(n_samples, dtype=np.float64)
        
        X_ptr = X.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        pred_ptr = predictions.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        
        self.lib.svm_predict_batch(self.model_ptr, X_ptr, pred_ptr, n_samples, n_features)
        return predictions
    
    def predict_score(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        n_samples, n_features = X.shape
        scores = np.zeros(n_samples, dtype=np.float64)
        
        X_ptr = X.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        scores_ptr = scores.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        
        self.lib.svm_predict_probability(self.model_ptr, X_ptr, scores_ptr, n_samples, n_features)
        return scores
    
    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        n_samples, n_features = X.shape
        X_ptr = X.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        y_ptr = y.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        
        return float(self.lib.svm_accuracy(self.model_ptr, X_ptr, y_ptr, n_samples, n_features))

def extract_robust_features(img_path):
    
    try:
        img = cv2.imread(str(img_path))
        if img is None:
            return None
        
        # Redimensionnement 
        img = cv2.resize(img, (100, 100))
        features = []
        
# Histogrammes
        # RGB
        for i in range(3):
            hist = cv2.calcHist([img], [i], None, [16], [0, 256])
            hist = cv2.normalize(hist, None).flatten()
            features.extend(hist)
        
        
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        for i in range(3):
            hist = cv2.calcHist([hsv], [i], None, [16], [0, 256])
            hist = cv2.normalize(hist, None).flatten()
            features.extend(hist)
        
       
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
       
        features.extend([
            np.mean(gray) / 255.0,
            np.std(gray) / 255.0,
            np.median(gray) / 255.0,
            np.max(gray) / 255.0,
            np.min(gray) / 255.0
        ])
        
        
        moments = cv2.moments(gray)
        hu_moments = cv2.HuMoments(moments).flatten()
        
        hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
        features.extend(hu_moments)
        
        
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            
            features.extend([
                area / (100*100),  
                perimeter / (4*100),  
                4 * np.pi * area / (perimeter**2 + 1e-6)  
            ])
        else:
            features.extend([0.0, 0.0, 0.0])
    
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        features.extend([
            np.mean(laplacian) / 1000.0,
            np.std(laplacian) / 1000.0
        ])
        
        
        for i in range(3):
            channel = img[:, :, i]
            features.extend([
                np.mean(channel) / 255.0,
                np.std(channel) / 255.0
            ])
        
        return np.array(features, dtype=np.float64)
    except:
        return None

def robust_normalization(X_train, X_test):
    
    p1 = np.percentile(X_train, 1, axis=0)
    p99 = np.percentile(X_train, 99, axis=0)
    range_vals = p99 - p1 + 1e-8
    
    X_train_norm = (X_train - p1) / range_vals
    X_test_norm = (X_test - p1) / range_vals
    
    
    X_train_norm = np.clip(X_train_norm, 0, 1)
    X_test_norm = np.clip(X_test_norm, 0, 1)
    
    return X_train_norm, X_test_norm

def evaluate_model(models, X_test, y_test, class_names):
    
    predictions = []
    
    for i in range(len(X_test)):
        sample = X_test[i:i+1]
        scores = []
        
        for svm in models:
            score = svm.predict_score(sample)[0]
            scores.append(score)
        
        best_class = np.argmax(scores)
        predictions.append(best_class)
    
    predictions = np.array(predictions)
    accuracy = np.mean(predictions == y_test) * 100
    
    cm = np.zeros((3, 3), dtype=int)
    for true, pred in zip(y_test, predictions):
        cm[true][pred] += 1
    
    print(f"\nAccuracy globale: {accuracy:.1f}%")
    print(f"Nombre d'échantillons: {len(X_test)}")
    
    print("\nMatrice de confusion:")
    print("      G   P   V")
    for i in range(3):
        row = [f"{cm[i][j]:3}" for j in range(3)]
        print(f"{class_names[i]} {''.join(row)}")
    
    # Accuracy par classe
    for i in range(3):
        mask = y_test == i
        if np.sum(mask) > 0:
            class_acc = np.mean(predictions[mask] == y_test[mask]) * 100
            print(f"  {class_names[i]}: {class_acc:.1f}%")
    
    return accuracy, predictions

def main():
    # Vérifier la DLL
    dll_path = Path("./target/release/libneural_networks.so")
    if not dll_path.exists():
        print("ERREUR: DLL non trouvée!")
        return
    
    # Chargement des données
   
    X, y = [], []
    
    for instrument, label in [('piano', 0), ('batterie', 1), ('harpe', 2)]:
        path = Path(f"dataset/{instrument}")
        if path.exists():
            images = list(path.glob("*.[pj][np]g"))
            print(f"  {instrument}: {len(images)} images")
            
            for img_path in images:
                features = extract_robust_features(img_path)
                if features is not None:
                    X.append(features)
                    y.append(label)
    
    X = np.array(X)
    y = np.array(y)
    
    if len(X) == 0:
        print("Aucune donnée chargée!")
        return
    
    print(f"\n  Total: {len(X)} échantillons")
    print(f"  Features: {X.shape[1]}")
    
    # Séparation train/test 
    print("\n[2/4] Séparation des données")
    
   
    indices_by_class = {0: [], 1: [], 2: []}
    for idx, label in enumerate(y):
        indices_by_class[label].append(idx)
    
    train_idx, test_idx = [], []
    for label, indices in indices_by_class.items():
        np.random.shuffle(indices)
        split = int(0.8 * len(indices))
        train_idx.extend(indices[:split])
        test_idx.extend(indices[split:])
    
    np.random.shuffle(train_idx)
    np.random.shuffle(test_idx)
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    print(f"  Train: {len(X_train)} échantillons")
    print(f"  Test:  {len(X_test)} échantillons")
    
    # Normalisation
    print("\n[3/4] Normalisation des données...")
    X_train_norm, X_test_norm = robust_normalization(X_train, X_test)
    
    # Entraînement
    print("\n[4/4] Entraînement des modèles...")
    start_time = time.time()
    
    models = []
    train_accuracies = []
    class_names = ['piano', 'batterie', 'harpe']
    
   
    class_params = [
        (0.02, 0.5), 
        (0.01, 1.0), 
        (0.015, 0.7)  
    ]
    
    for class_idx, (lr, c) in enumerate(class_params):
        print(f"\n  {class_names[class_idx]}:")
        
       
        y_binary = np.where(y_train == class_idx, 1.0, -1.0)
        
        # entraîner le modèle
        svm = CorrectSVM(
            n_inputs=X_train_norm.shape[1],
            learning_rate=lr,
            c=c
        )
        
        loss = svm.fit(X_train_norm, y_binary, max_iterations=3000)
        
        # Évaluation sur train
        train_acc = svm.accuracy(X_train_norm, y_binary) * 100
        train_accuracies.append(train_acc)
        
        print(f"    Loss: {loss:.4f}")
        print(f"    Accuracy train: {train_acc:.1f}%")
        
        models.append(svm)
    
   
    print("RÉSULTATS FINAUX")
    
    avg_train_acc = np.mean(train_accuracies)
    print(f"\nAccuracy moyenne sur train: {avg_train_acc:.1f}%")
    
    test_acc, predictions = evaluate_model(models, X_test_norm, y_test, class_names)
    
    end_time = time.time()
    print(f"\nTemps total: {end_time - start_time:.1f} secondes")
    
    
    
    print("TESTS")
    
    for test_size in [10, 20]:
        print(f"\nTest de {test_size} images aléatoires:")
        
        correct = 0
        for _ in range(test_size):
           
            true_class = random.randint(0, 2)
            instrument = ['piano', 'batterie', 'harpe'][true_class]
            
            
            path = Path(f"dataset/{instrument}")
            images = list(path.glob("*.[pj][np]g"))
            
            if not images:
                continue
                
            img_path = random.choice(images)
            
           
            features = extract_robust_features(img_path)
            if features is None:
                continue
            
           
            p1 = np.percentile(X_train, 1, axis=0)
            p99 = np.percentile(X_train, 99, axis=0)
            range_vals = p99 - p1 + 1e-8
            features_norm = (features - p1) / range_vals
            features_norm = np.clip(features_norm, 0, 1)
            
           
            scores = []
            for svm in models:
                score = svm.predict_score(features_norm.reshape(1, -1))[0]
                scores.append(score)
            
            pred_class = np.argmax(scores)
            
            if pred_class == true_class:
                correct += 1
                mark = ""
            else:
                mark = ""
            
            print(f"  {mark} {instrument:8} : {class_names[pred_class]}")
        
        print(f"  Résultat: {correct}/{test_size} corrects ({correct/test_size*100:.0f}%)")
    

   
    print(f"1. Accuracy sur train: {avg_train_acc:.1f}%")
    print(f"2. Accuracy sur test:  {test_acc:.1f}%")
    print(f"3. Dataset: {len(X)} images au total")
    print(f"4. Features utilisées: {X.shape[1]}")
    
   
if __name__ == "__main__":
    
    np.random.seed(42)
    random.seed(42)
    
    main()