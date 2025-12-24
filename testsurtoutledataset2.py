import ctypes
import numpy as np
import cv2
from pathlib import Path
import random


class PMC:
    def __init__(self, n_inputs: int, n_hidden: int, learning_rate: float = 0.005, dropout: float = 0.0):
        self.lib = ctypes.CDLL("./target/release/neural_networks.dll")
        
        class PMCConfig(ctypes.Structure):
            _fields_ = [
                ("n_inputs", ctypes.c_uint),
                ("n_hidden", ctypes.c_uint),
                ("n_outputs", ctypes.c_uint),
                ("learning_rate", ctypes.c_double),
                ("dropout_rate", ctypes.c_double)  
            ]
        
        self.PMCConfig = PMCConfig
        
      
        self.dropout = dropout
        config = PMCConfig(
            n_inputs=n_inputs,
            n_hidden=n_hidden,
            n_outputs=1,
            learning_rate=learning_rate,
            dropout_rate=dropout
        )
        

        self.lib.pmc_new.argtypes = [ctypes.POINTER(PMCConfig)]
        self.lib.pmc_new.restype = ctypes.c_void_p
        self.lib.pmc_delete.argtypes = [ctypes.c_void_p]
        self.lib.pmc_fit.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t
        ]
        self.lib.pmc_fit.restype = ctypes.c_double
        self.lib.pmc_predict_batch.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
            ctypes.c_size_t
        ]
        
        self.model_ptr = self.lib.pmc_new(ctypes.byref(config))
    
    def __del__(self):
        if hasattr(self, 'model_ptr'):
            self.lib.pmc_delete(self.model_ptr)
    
    def fit(self, X: np.ndarray, y: np.ndarray, max_iterations: int = 800) -> float:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        n_samples, n_features = X.shape
        X_ptr = X.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        y_ptr = y.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        
        return self.lib.pmc_fit(self.model_ptr, X_ptr, y_ptr, n_samples, n_features, max_iterations)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        n_samples, n_features = X.shape
        predictions = np.zeros(n_samples, dtype=np.float64)
        
        X_ptr = X.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        pred_ptr = predictions.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        
        self.lib.pmc_predict_batch(self.model_ptr, X_ptr, pred_ptr, n_samples, n_features)
        

        if self.dropout > 0:
            predictions = predictions * (1 - self.dropout)
        
        return predictions

class PMCClassifier:
    def __init__(self, n_inputs: int, n_hidden: int = 8, learning_rate: float = 0.005, dropout: float = 0.2):
        self.models = {
            'guitare': PMC(n_inputs, n_hidden, learning_rate, dropout),
            'piano': PMC(n_inputs, n_hidden, learning_rate, dropout),
            'violon': PMC(n_inputs, n_hidden, learning_rate, dropout)
        }
        self.labels = {'guitare': 1, 'piano': -1, 'violon': 0}
        self.names = {1: 'Guitare', -1: 'Piano', 0: 'Violon'}
        self.n_hidden = n_hidden
        self.dropout = dropout
    
    def fit(self, X_train, y_train, max_iterations=800, patience=50):
        print(f"Entraînement avec {self.n_hidden} neurones cachés, dropout={self.dropout}")
        
        for name, model in self.models.items():
            label = self.labels[name]
            y_binary = np.where(y_train == label, 1.0, -1.0)
            loss = model.fit(X_train, y_binary, max_iterations)
            print(f"  {name}: loss={loss:.4f}")
    
    def predict(self, X):
        scores = {name: model.predict(X) for name, model in self.models.items()}
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


def extract_features(img_path):
    try:
        img = cv2.imread(str(img_path))
        if img is None:
            return None
        

        img = cv2.resize(img, (100, 100))
        features = []

        for canal in range(3):
            hist = cv2.calcHist([img], [canal], None, [12], [0, 256])
            hist = hist.flatten() / (hist.sum() + 1e-6)
            features.extend(hist)
        

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        for canal in range(3):
            hist = cv2.calcHist([hsv], [canal], None, [8], [0, 256])
            hist = hist.flatten() / (hist.sum() + 1e-6)
            features.extend(hist)
        

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        features.extend([
            gray.mean() / 255.0,
            gray.std() / 255.0,
            np.median(gray) / 255.0,
        ])

        for ksize in [3, 5]:
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
            mag = np.sqrt(sobelx**2 + sobely**2)
            features.extend([np.mean(mag)/1000.0, np.std(mag)/1000.0])
        

        moments = cv2.moments(gray)
        hu_moments = cv2.HuMoments(moments).flatten()
        hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
        features.extend(hu_moments[:4])  
        
 
        h, w = gray.shape
        features.extend([
            w / h if h > 0 else 1.0,
            np.sum(gray[:, :w//2]) / (np.sum(gray[:, w//2:]) + 1e-6)
        ])
        
        return np.array(features, dtype=np.float64)
    except Exception as e:
        return None

def augment_data(X, y, noise_level=0.05):

    X_augmented = [X]
    y_augmented = [y]
    

    for _ in range(2):
        noise = np.random.normal(0, noise_level, X.shape)
        X_augmented.append(X + noise)
        y_augmented.append(y)
    

    for scale in [0.9, 1.1]:
        X_augmented.append(X * scale)
        y_augmented.append(y)
    
    return np.vstack(X_augmented), np.hstack(y_augmented)

def load_data(augment=False):
   
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
    
    if not X:
        return None, None
    
    X, y = np.array(X), np.array(y)
    
    if augment:
        X, y = augment_data(X, y)
        print(f"Après augmentation: {len(X)} échantillons")
    
    return X, y

def normalize_data(X_train, X_test):
    median = np.median(X_train, axis=0)
    q75 = np.percentile(X_train, 75, axis=0)
    q25 = np.percentile(X_train, 25, axis=0)
    iqr = q75 - q25 + 1e-8
    
    X_train_norm = np.clip((X_train - median) / iqr, -3, 3)
    X_test_norm = np.clip((X_test - median) / iqr, -5, 5)
    
    return X_train_norm, X_test_norm, (median, iqr)

def train_model(config='simple'):
    
    configs = {
        'simple': {'hidden': 5, 'lr': 0.003, 'dropout': 0.3, 'iter': 1000},
        'medium': {'hidden': 8, 'lr': 0.005, 'dropout': 0.2, 'iter': 1200},
        'complex': {'hidden': 12, 'lr': 0.008, 'dropout': 0.1, 'iter': 1500}
    }
    
    cfg = configs.get(config, configs['simple'])
    
    print(f"\nConfiguration: {config}")
    print(f"  Neurones cachés: {cfg['hidden']}")
    print(f"  Learning rate: {cfg['lr']}")
    print(f"  Dropout: {cfg['dropout']}")
    print(f"  Itérations: {cfg['iter']}")

    X, y = load_data(augment=True)
    if X is None:
        return None

    n = len(X)
    idx = list(range(n))
    random.shuffle(idx)
    split = int(0.8 * n)
    
    X_train, X_test = X[idx[:split]], X[idx[split:]]
    y_train, y_test = y[idx[:split]], y[idx[split:]]
    

    print(f"  Train: {len(X_train)}")
    print(f"  Test:  {len(X_test)}")
    print(f"  Features: {X_train.shape[1]}")

    X_train_norm, X_test_norm, norm = normalize_data(X_train, X_test)
    

    classifier = PMCClassifier(
        n_inputs=X_train_norm.shape[1],
        n_hidden=cfg['hidden'],
        learning_rate=cfg['lr'],
        dropout=cfg['dropout']
    )
    
    classifier.fit(X_train_norm, y_train, max_iterations=cfg['iter'])
    

    y_pred = classifier.predict(X_test_norm)
    accuracy = np.mean(y_pred == y_test) * 100
    
    print(f"\n Résultats:")
    print(f"  Accuracy: {accuracy:.1f}%")
    

    cm = np.zeros((3, 3), dtype=int)
    for true, pred in zip(y_test, y_pred):
        cm[int(true)+1][int(pred)+1] += 1
    
    print("\nMatrice de confusion:")
    print("      P  V  G")
    labels = ['Piano', 'Violon', 'Guitare']
    for i, label in enumerate(labels):
        print(f"{label[0]}: {cm[i]}")
    
    print("\nMétriques par classe:")
    for i, label in enumerate(labels):
        tp = cm[i][i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"  {label}:")
        print(f"    Précision: {precision:.1%}")
        print(f"    Rappel:    {recall:.1%}")
        print(f"    F1-score:  {f1:.1%}")
    
    return classifier, accuracy, norm, cm

def main():
    print(" PMC")

    if not Path("./target/release/neural_networks.dll").exists():

        return
    
    model = None
    accuracy = 0
    norm = None
    
    while True:
        print("1.  Entraîner PMC Simple (5 neurones)")
        print("2.  Entraîner PMC Moyen (8 neurones)")
        print("3.  Entraîner PMC Avancé (12 neurones)")
        print("4.  Tester le modèle")
        print("5.  Voir performance")
        print("6.  Quitter")

        
        choice = input("\nChoix: ").strip()
        
        if choice == "1":
            result = train_model('simple')
            if result:
                model, accuracy, norm, _ = result
        
        elif choice == "2":
            result = train_model('medium')
            if result:
                model, accuracy, norm, _ = result
        
        elif choice == "3":
            result = train_model('complex')
            if result:
                model, accuracy, norm, _ = result
        
        elif choice == "4":
            if not model:
               
                continue
            
          
            correct = 0
            total = 10
            
            print(f"\n Test sur {total} images aléatoires:")
            for _ in range(total):
                instrument = random.choice(['guitare', 'piano', 'violon'])
                path = Path(f"dataset/{instrument}")
                
                if path.exists():
                    images = list(path.glob("*.[pj][np]g"))
                    if images:
                        img = random.choice(images)
                        features = extract_features(img)
                        
                        if features is not None and norm:
                            median, iqr = norm
                            features_norm = np.clip((features - median) / iqr, -5, 5)
                            pred = model.predict(features_norm.reshape(1, -1))[0]
                            pred_name = model.names[pred]
                            
                            if pred_name.lower() == instrument:
                                correct += 1
                                print(f"   {img.name}: {instrument} -> {pred_name}")
                            else:
                                print(f"   {img.name}: {instrument} -> {pred_name}")
            
            test_acc = correct / total * 100
            print(f"\n Test accuracy: {test_acc:.1f}%")
            print(f" Train accuracy: {accuracy:.1f}%")
            
            if test_acc < accuracy * 0.7:
                print(" le modèle surapprend!")
            elif test_acc > accuracy * 0.9:
                print(" Bonne généralisation!")
        
        elif choice == "5":
            if not model:
                print(" Aucun modèle entraîné")
            else:
                print(f"\n Performance actuelle: {accuracy:.1f}%")
                print("\nInterprétation:")
                if accuracy >= 65:
                    print(" Le modèle fonctionne très bien.")
                elif accuracy >= 55:
                    print(" Le modèle est utilisable.")
                elif accuracy >= 45:
                    print(" Modéré.")
                else:
                    print("Faible.")

if __name__ == "__main__":
    main()