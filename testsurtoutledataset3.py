import ctypes
import numpy as np
import cv2
from pathlib import Path
import random

class SVMConfig(ctypes.Structure):
    _fields_ = [
        ("n_inputs", ctypes.c_uint),
        ("learning_rate", ctypes.c_double),
        ("c", ctypes.c_double),
        ("max_iterations", ctypes.c_uint)
    ]


class SVM:
    def __init__(self, n_inputs: int, learning_rate: float = 0.001, c: float = 1.0):
        self.lib = ctypes.CDLL("./target/release/neural_networks.dll")
        
        self.config = SVMConfig(
            n_inputs=n_inputs,
            learning_rate=learning_rate,
            c=c,
            max_iterations=1000
        )
        
        # Fonctions
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
        
        self.model_ptr = self.lib.svm_new(ctypes.byref(self.config))
    
    def __del__(self):
        if hasattr(self, 'model_ptr'):
            self.lib.svm_delete(self.model_ptr)
    
    def fit(self, X: np.ndarray, y: np.ndarray, max_iterations: int = 1000) -> float:
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
    
    def predict_probability(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        n_samples, n_features = X.shape
        scores = np.zeros(n_samples, dtype=np.float64)
        
        X_ptr = X.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        scores_ptr = scores.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        
        self.lib.svm_predict_probability(self.model_ptr, X_ptr, scores_ptr, n_samples, n_features)
        return scores

class SVMClassifier:
    def __init__(self, n_inputs: int, c: float = 1.0, learning_rate: float = 0.001):
        self.models = {
            'guitare': SVM(n_inputs, learning_rate, c),
            'piano': SVM(n_inputs, learning_rate, c),
            'violon': SVM(n_inputs, learning_rate, c)
        }
        self.labels = {'guitare': 1, 'piano': -1, 'violon': 0}
        self.names = {1: 'Guitare', -1: 'Piano', 0: 'Violon'}
    
    def fit(self, X_train, y_train, max_iterations=1000):
        print("\nEntraînement SVM...")
        for name, model in self.models.items():
            label = self.labels[name]
            y_binary = np.where(y_train == label, 1.0, -1.0)
            loss = model.fit(X_train, y_binary, max_iterations)
            print(f"  {name}: loss={loss:.4f}")
    
    def predict(self, X):
        scores = {name: model.predict_probability(X) for name, model in self.models.items()}
        preds = []
        
        for i in range(len(X)):
            best_score = -np.inf
            best_class = 'guitare'
            for name, score_array in scores.items():
                if score_array[i] > best_score:
                    best_score = score_array[i]
                    best_class = name
            preds.append(self.labels[best_class])
        
        return np.array(preds)
    
    def predict_proba(self, X):
        scores = {name: model.predict_probability(X) for name, model in self.models.items()}
        n_samples = len(X)
        probas = np.zeros((n_samples, 3))
        
        for i in range(n_samples):
            score_values = np.array([
                scores['guitare'][i],
                scores['piano'][i],
                scores['violon'][i]
            ])
            adjusted_scores = score_values - np.min(score_values) + 1e-8
            exp_scores = np.exp(adjusted_scores)
            probas[i] = exp_scores / exp_scores.sum()
        
        return probas



def extract_features(img_path):
    try:
        img = cv2.imread(str(img_path))
        if img is None:
            return None
        
        img = cv2.resize(img, (80, 80))
        features = []
        

        for canal in range(3):
            hist = cv2.calcHist([img], [canal], None, [16], [0, 256])
            hist = hist.flatten() / (hist.sum() + 1e-6)
            features.extend(hist)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        features.extend([
            gray.mean() / 255.0,
            gray.std() / 255.0,
            np.median(gray) / 255.0,
        ])
        

        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        features.extend([
            np.mean(magnitude) / 500.0,
            np.std(magnitude) / 500.0,
        ])
        

        moments = cv2.moments(gray)
        hu_moments = cv2.HuMoments(moments).flatten()
        hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
        features.extend(hu_moments[:4])
        

        h, w = gray.shape
        features.extend([
            w / h if h > 0 else 1.0,
            np.sum(gray > 128) / (h * w)
        ])
        
        return np.array(features, dtype=np.float64)
    except:
        return None

def load_data():
    X, y = [], []
    for instrument, label in [('guitare', 1), ('piano', -1), ('violon', 0)]:
        path = Path(f"dataset/{instrument}")
        if path.exists():
            images = list(path.glob("*.[pj][np]g"))
            print(f"{instrument}: {len(images)} images")
            for img in images:
                features = extract_features(img)
                if features is not None:
                    X.append(features)
                    y.append(label)
    
    return np.array(X), np.array(y) if X else (None, None)

def normalize(X_train, X_test):
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0) + 1e-8
    return (X_train - mean) / std, (X_test - mean) / std, (mean, std)


def train_svm():

    X, y = load_data()
    if X is None:
        return None

    n = len(X)
    idx = list(range(n))
    random.shuffle(idx)
    split = int(0.8 * n)
    
    X_train, X_test = X[idx[:split]], X[idx[split:]]
    y_train, y_test = y[idx[:split]], y[idx[split:]]
    
    print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")
    

    X_train_norm, X_test_norm, norm = normalize(X_train, X_test)

    classifier = SVMClassifier(X_train_norm.shape[1])
    classifier.fit(X_train_norm, y_train)

    y_pred = classifier.predict(X_test_norm)
    accuracy = np.mean(y_pred == y_test) * 100
    
    print(f"\nAccuracy: {accuracy:.1f}%")
    

    cm = np.zeros((3, 3), dtype=int)
    for true, pred in zip(y_test, y_pred):
        cm[int(true)+1][int(pred)+1] += 1
    
    print("\nMatrice de confusion:")
    print("P V G")
    for row in cm:
        print(row)
    
    return classifier, accuracy, norm


def test_svm(classifier, norm=None):
    while True:
        print("\n1. Tester une image")
        print("2. Batch test (5 images)")
        print("3. Retour")
        
        choix = input("Choix: ").strip()
        
        if choix == "1":
            path = input("Chemin image: ").strip()
            if Path(path).exists():
                features = extract_features(path)
                if features is not None:
                    if norm:
                        mean, std = norm
                        features = (features - mean) / std
                    
                    probas = classifier.predict_proba(features.reshape(1, -1))[0]
                    pred_idx = np.argmax(probas)
                    pred_label = list(classifier.names.values())[pred_idx]
                    
                    print(f"\nRésultat: {pred_label}")
                    print(f"Confiance: {probas[pred_idx]:.1%}")
                    print(f"Guitare: {probas[2]:.1%}")
                    print(f"Piano:   {probas[0]:.1%}")
                    print(f"Violon:  {probas[1]:.1%}")
                else:
                    print("Erreur extraction features")
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
                        features = extract_features(img)
                        
                        if features is not None and norm:
                            mean, std = norm
                            features = (features - mean) / std
                            pred = classifier.predict(features.reshape(1, -1))[0]
                            pred_name = classifier.names[pred]
                            
                            if pred_name.lower() == instrument:
                                correct += 1
                                print(f" {img.name}: {instrument} -> {pred_name}")
                            else:
                                print(f" {img.name}: {instrument} -> {pred_name}")
            
            print(f"\nRésultat: {correct}/5 corrects")
        
        elif choix == "3":
            break
def main():
    print("SVM Classifier")
    
    if not Path("./target/release/neural_networks.dll").exists():
        print(" DLL manquante!")
        print("Compilez avec: cargo build --release --features svm")
        return
    
    model = None
    accuracy = 0
    norm = None
    
    while True:
        print("\n1. Entraîner")
        print("2. Tester")
        print("3. Stats")
        print("4. Quitter")
        
        choix = input("Choix: ").strip()
        
        if choix == "1":
            result = train_svm()
            if result:
                model, accuracy, norm = result
        
        elif choix == "2":
            if model:
                test_svm(model, norm)
            else:
                print("Entraînez ")
        
        elif choix == "3":
            if model:
                print(f"Accuracy: {accuracy:.1f}%")
            else:
                print("Aucun modèle")

if __name__ == "__main__":
    main()