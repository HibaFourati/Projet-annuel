import ctypes
import numpy as np
import cv2
from pathlib import Path
import random


class LinearModel:
    def __init__(self, input_dim: int, learning_rate: float = 0.01):
        self.lib = ctypes.CDLL("./target/release/libneural_networks.so")
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
    
    def fit(self, X: np.ndarray, y: np.ndarray, max_iterations: int = 5000) -> float:
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
    
def transform_features(X):
    return np.column_stack([
        X,
        X[:, :10] ** 2,
        X[:, :10] * X[:, 10:20]
    ])


def charger_dataset():
    X, y = [], []
    classes = {'piano': 0, 'batterie': 1, 'harpe': 2}


    for name, label in classes.items():
        path = Path(f"dataset/{name}")
        if path.exists():
            for img in path.glob("*.[pj][np]g"):
                f = extraire_features(img)
                if f is not None:
                    X.append(f)
                    y.append(label)

    return np.array(X), np.array(y)

X, y = charger_dataset()
unique, counts = np.unique(y, return_counts=True)
print("Nombre d’images par classe :")
for label, count in zip(unique, counts):
    if label == 0:
        name = "piano"
    elif label == 1:
        name = "batterie"
    elif label == 2:
        name = "harpe"
    print(f"{name}: {count} images")



def normaliser(X_train, X_test):
    median = np.median(X_train, axis=0)
    iqr = np.percentile(X_train, 75, axis=0) - np.percentile(X_train, 25, axis=0)
    iqr += 1e-8
    return (X_train - median)/iqr, (X_test - median)/iqr, (median, iqr)





class Classifier:
    def __init__(self, input_dim, lr=0.001):
        self.models = {
            'piano': LinearModel(input_dim, lr),
            'batterie': LinearModel(input_dim, lr),
            'harpe': LinearModel(input_dim, lr)
        }
        self.labels = {'piano': 0, 'batterie': 1, 'harpe': 2}
        self.names = {0: 'Piano', 1: 'Batterie', 2: 'Harpe'}

    def fit(self, X, y):
        X = transform_features(X)
        for name, model in self.models.items():
            y_bin = np.where(y == self.labels[name], 1.0, -1.0)
            model.fit(X, y_bin)

    def predict(self, X):
        X = transform_features(X)
        scores = {k: m.predict_score(X) for k, m in self.models.items()}
        preds = []
        for i in range(len(X)):
            best = max(scores, key=lambda k: scores[k][i])
            preds.append(self.labels[best])
        return np.array(preds)

    def predict_proba(self, X):
        X = transform_features(X)
        scores = np.column_stack([
            self.models['piano'].predict_score(X),
            self.models['batterie'].predict_score(X),
            self.models['harpe'].predict_score(X)
        ])
        exp = np.exp(scores - scores.max(axis=1, keepdims=True))
        return exp / exp.sum(axis=1, keepdims=True)



def entrainement():
    X, y = charger_dataset()
    idx = np.random.permutation(len(X))
    split = int(0.8 * len(X))

    X_train, X_test = X[idx[:split]], X[idx[split:]]
    y_train, y_test = y[idx[:split]], y[idx[split:]]

    X_train, X_test, normalizer = normaliser(X_train, X_test)

    X_train_t = transform_features(X_train)
    classifier = Classifier(input_dim=X_train_t.shape[1])

    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    acc = np.mean(y_pred == y_test) * 100

    print(f"Accuracy: {acc:.1f}%")
    return classifier, acc, normalizer
    print("Scores moyens:")
    for k, m in classifier.models.items():
       print(k, np.mean(np.abs(m.predict_score(transform_features(X_train)))))



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
                    pred_label = classifier.names[pred_idx]
                    print(f"\nRésultat: {pred_label}")
                    print(f"piano: {probas[0]:.1%}")
                    print(f"batterie:   {probas[1]:.1%}")
                    print(f"harpe:  {probas[2]:.1%}")
                else:
                    print("Erreur extraction")
            else:
                print("Fichier non trouvé")
        
        elif choix == "2":
            correct = 0
            for _ in range(5):
                instrument = random.choice(['piano', 'batterie', 'harpe'])
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
                            pred_name = classifier.names[pred]

                            
                            if pred_name.lower() == instrument:
                                correct += 1
                                print(f"   {img.name}: {instrument} → {pred_name}")
                            else:
                                print(f"   {img.name}: {instrument} → {pred_name}")
            
            print(f"\nRésumé: {correct}/5 corrects")
        
        elif choix == "3":
            break


if __name__ == "__main__":

    
    if not Path("./target/release/libneural_networks.so").exists():
         print("Bibliothèque manquante!")
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
        