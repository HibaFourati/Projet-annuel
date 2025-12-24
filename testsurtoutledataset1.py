import ctypes
import numpy as np
import cv2
from pathlib import Path
import random


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
        self.model_ptr = self.lib.linear_model_new(input_dim, learning_rate)
    
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
    
    def predict_score(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        n_samples, n_features = X.shape
        results = np.zeros(n_samples, dtype=np.float64)
        X_ptr = X.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        results_ptr = results.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        self.lib.linear_model_predict_batch(self.model_ptr, X_ptr, results_ptr, n_samples, n_features)
        return results
    
def extraire_features(img_path):

    try:
        img = cv2.imread(str(img_path))
        if img is None:
            return None
        img = cv2.resize(img, (100, 100))
        features = []
        
        # Couleur
        for canal in range(3):
            hist = cv2.calcHist([img], [canal], None, [20], [0, 256])
            hist = hist.flatten() / (hist.sum() + 1e-6)
            features.extend(hist)
        

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        features.extend([gray.mean()/255.0, gray.std()/255.0])
        

        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        features.extend([np.mean(magnitude)/1000.0, np.std(magnitude)/1000.0])

        moments = cv2.moments(gray)
        hu_moments = cv2.HuMoments(moments).flatten()
        hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
        features.extend(hu_moments)
        

        h, w = gray.shape
        features.append(w/h if h>0 else 1.0)
        
        return np.array(features, dtype=np.float64)
    except:
        return None

def charger_dataset():
    X, y = [], []
    classes = [('guitare', 1), ('piano', -1), ('violon', 0)]
    
    for instrument, label in classes:
        path = Path(f"dataset/{instrument}")
        if path.exists():
            for img in path.glob("*.[pj][np]g"):
                features = extraire_features(img)
                if features is not None:
                    X.append(features)
                    y.append(label)
    
    return np.array(X), np.array(y) if X else (None, None)

def normaliser(X_train, X_test):
    median = np.median(X_train, axis=0)
    q75 = np.percentile(X_train, 75, axis=0)
    q25 = np.percentile(X_train, 25, axis=0)
    iqr = q75 - q25 + 1e-8
    return (X_train-median)/iqr, (X_test-median)/iqr, (median, iqr)

class Classifier:
    def __init__(self, input_dim, learning_rate=0.001):
        self.models = {
            'guitare': LinearModel(input_dim, learning_rate),
            'piano': LinearModel(input_dim, learning_rate),
            'violon': LinearModel(input_dim, learning_rate)
        }
        self.class_labels = {'guitare': 1, 'piano': -1, 'violon': 0}
        self.label_names = {1: 'Guitare', -1: 'Piano', 0: 'Violon'}
        
    def fit(self, X_train, y_train, max_iterations=800):
        for class_name, model in self.models.items():
            label = self.class_labels[class_name]
            y_binary = np.where(y_train == label, 1.0, -1.0)
            model.fit(X_train, y_binary, max_iterations)
    
    def predict(self, X):
        scores = {name: model.predict_score(X) for name, model in self.models.items()}
        preds = []
        for i in range(len(X)):
            best_score = -np.inf
            best_class = 'guitare'
            for class_name, score_array in scores.items():
                if score_array[i] > best_score:
                    best_score = score_array[i]
                    best_class = class_name
            preds.append(self.class_labels[best_class])
        return np.array(preds)
    
    def predict_proba(self, X):
        scores = {name: model.predict_score(X) for name, model in self.models.items()}
        probas = np.zeros((len(X), 3))
        for i in range(len(X)):
            score_values = np.array([scores['guitare'][i], scores['piano'][i], scores['violon'][i]])
            exp_scores = np.exp(score_values - np.max(score_values))
            probas[i] = exp_scores / exp_scores.sum()
        return probas


def entrainement():
    X, y = charger_dataset()
    if X is None:
        return None
    
    n = len(X)
    idx = list(range(n))
    random.shuffle(idx)
    split = int(0.8 * n)
    
    X_train, X_test = X[idx[:split]], X[idx[split:]]
    y_train, y_test = y[idx[:split]], y[idx[split:]]
    
    X_train, X_test, normalizer = normaliser(X_train, X_test)
    
    classifier = Classifier(input_dim=X_train.shape[1])
    classifier.fit(X_train, y_train)
    
    y_pred = classifier.predict(X_test)
    accuracy = np.mean(y_pred == y_test) * 100
    
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"Accuracy: {accuracy:.1f}%")
    
    cm = np.zeros((3, 3), dtype=int)
    for true, pred in zip(y_test, y_pred):
        cm[int(true)+1][int(pred)+1] += 1
    print("Matrice de confusion:")
    print("      P  V  G")
    for i, label in enumerate(['P', 'V', 'G']):
        print(f"{label}: {cm[i]}")
    
    return classifier, accuracy, normalizer

def tester(classifier, normalizer=None):
    while True:
        print("\n1. Tester une image")
        print("2. Batch test (5)")
        print("3. Retour")
        choix = input("Choix: ").strip()
        
        if choix == "1":
            img_path = input("Image: ").strip()
            if Path(img_path).exists():
                features = extraire_features(img_path)
                if features is not None:
                    if normalizer:
                        median, iqr = normalizer
                        features = (features - median) / iqr
                    
                    probas = classifier.predict_proba(features.reshape(1, -1))[0]
                    pred_idx = np.argmax(probas)
                    pred_label = list(classifier.label_names.values())[pred_idx]
                    
                    print(f"\nRésultat: {pred_label}")
                    print(f"Confiance: {probas[pred_idx]:.1%}")
                    print(f"Guitare: {probas[2]:.1%}")
                    print(f"Piano:   {probas[0]:.1%}")
                    print(f"Violon:  {probas[1]:.1%}")
                else:
                    print("Erreur extraction")
            else:
                print("Fichier non trouvé")
        
        elif choix == "2":
            correct = 0
            for _ in range(5):
                instrument = random.choice(['guitare', 'piano', 'violon'])
                path = Path(f"dataset/{instrument}")
                if path.exists():
                    images = list(path.glob("*.[pj][np]g"))
                    if images:
                        img = random.choice(images)
                        features = extraire_features(img)
                        
                        if features is not None and normalizer:
                            median, iqr = normalizer
                            features = (features - median) / iqr
                            pred = classifier.predict(features.reshape(1, -1))[0]
                            pred_name = classifier.label_names[pred]
                            
                            if pred_name.lower() == instrument:
                                correct += 1
                                print(f"  ✅ {img.name}: {instrument} → {pred_name}")
                            else:
                                print(f"  ❌ {img.name}: {instrument} → {pred_name}")
            
            print(f"\nRésumé: {correct}/5 corrects")
        
        elif choix == "3":
            break


if __name__ == "__main__":

    
    if not Path("./target/release/neural_networks.dll").exists():
        print(" DLL manquante!")
        exit(1)
    
    classifier = None
    accuracy = 0
    normalizer = None
    
    while True:
        print("\n1. Entraîner")
        print("2. Tester")
        print("3. Performance")
        print("4. Quitter")
        choix = input("Choix: ").strip()
        
        if choix == "1":
            result = entrainement()
            if result:
                classifier, accuracy, normalizer = result
        
        elif choix == "2":
            if classifier:
                tester(classifier, normalizer)
            else:
                print("Entraînez")
        
        elif choix == "3":
            if classifier:
                print(f"Performance: {accuracy:.1f}%")
            else:
                print("Aucun modèle")
        