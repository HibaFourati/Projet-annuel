import ctypes
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path

# Structure de configuration
class PMCConfig(ctypes.Structure):
    _fields_ = [
        ("n_inputs", ctypes.c_uint),
        ("n_hidden", ctypes.c_uint),
        ("n_outputs", ctypes.c_uint),
        ("learning_rate", ctypes.c_double),
    ]

# WRAPPER PMC
class PMC:
    def __init__(self, n_inputs: int, n_hidden: int = 2, learning_rate: float = 0.01):  # Même learning_rate
        dll_path = "./target/release/neural_networks.dll"
        
        print(f"Chargement de: {dll_path}")
        self.lib = ctypes.CDLL(dll_path)
        
        self.lib.pmc_new.argtypes = [ctypes.POINTER(PMCConfig)]
        self.lib.pmc_new.restype = ctypes.c_void_p
        
        self.lib.pmc_delete.argtypes = [ctypes.c_void_p]
        
        self.lib.pmc_fit.argtypes = [
            ctypes.c_void_p,                    
            ctypes.POINTER(ctypes.c_double),    
            ctypes.POINTER(ctypes.c_double),    
            ctypes.c_size_t,                    
            ctypes.c_size_t,                    
            ctypes.c_size_t,                    
        ]
        self.lib.pmc_fit.restype = ctypes.c_double
        
        self.lib.pmc_predict_batch.argtypes = [
            ctypes.c_void_p,                    
            ctypes.POINTER(ctypes.c_double),    
            ctypes.POINTER(ctypes.c_double),   
            ctypes.c_size_t,                   
            ctypes.c_size_t,                  
        ]
        
        self.lib.pmc_accuracy.argtypes = [
            ctypes.c_void_p,                    
            ctypes.POINTER(ctypes.c_double),    
            ctypes.POINTER(ctypes.c_double),    
            ctypes.c_size_t,                    
            ctypes.c_size_t,                    
        ]
        self.lib.pmc_accuracy.restype = ctypes.c_double
        
        config = PMCConfig(
            n_inputs=n_inputs,
            n_hidden=n_hidden,
            n_outputs=1,
            learning_rate=learning_rate  # 0.01 comme le modèle linéaire
        )
     
        self.model_ptr = self.lib.pmc_new(ctypes.byref(config))
        
        if not self.model_ptr:
            raise RuntimeError("Échec de la création du modèle")
        
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        
        print(f"PMC créé: architecture {n_inputs} -> {n_hidden} -> 1 (learning_rate: {learning_rate})")
    
    def __del__(self):
        if hasattr(self, 'model_ptr') and self.model_ptr:
            self.lib.pmc_delete(self.model_ptr)
    
    def fit(self, X: np.ndarray, y: np.ndarray, max_iterations: int = 1000) -> float:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        n_samples, n_features = X.shape
        
        if n_features != self.n_inputs:
            raise ValueError(f"Attendu {self.n_inputs} features, reçu {n_features}")
        
        X_ptr = X.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        y_ptr = y.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        
        error = self.lib.pmc_fit(
            self.model_ptr, X_ptr, y_ptr, n_samples, n_features, max_iterations
        )
        
        return error
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        n_samples, n_features = X.shape
        
        if n_features != self.n_inputs:
            raise ValueError(f"Attendu {self.n_inputs} features, reçu {n_features}")
        
        results = np.zeros(n_samples, dtype=np.float64)
        
        X_ptr = X.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        results_ptr = results.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        
        self.lib.pmc_predict_batch(self.model_ptr, X_ptr, results_ptr, n_samples, n_features)
        
        return results
    
    def predict_class(self, X: np.ndarray, threshold: float = 0.0) -> np.ndarray:
        predictions = self.predict(X)
        return np.where(predictions >= threshold, 1, -1)
    
    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        n_samples, n_features = X.shape
        
        if n_features != self.n_inputs:
            raise ValueError(f"Attendu {self.n_inputs} features, reçu {n_features}")
        
        X_ptr = X.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        y_ptr = y.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        
        return self.lib.pmc_accuracy(self.model_ptr, X_ptr, y_ptr, n_samples, n_features)

# FONCTIONS COMMUNES (identiques au modèle linéaire)
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

def charger_dataset():
    X, y = [], []
    for instrument, label in [('guitare', 1), ('piano', -1)]:
        path = Path(f"dataset/{instrument}")
        if path.exists():
            images = list(path.glob("*.[pj][np]g"))[:13]  # Même nombre d'images
            for img in images:
                features = extraire_features(img)
                if features is not None:
                    X.append(features)
                    y.append(label)
    
    if len(X) == 0:
        print("Aucune image trouvée")
        return None, None
    
    return np.array(X), np.array(y)

def split_manuel(X, y, test_size=0.2):
    n_samples = len(X)
    indices = np.arange(n_samples)
    np.random.seed(42)
    np.random.shuffle(indices)
    
    split_idx = int(n_samples * (1 - test_size))
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]
    
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

def normaliser_manuel(X_train, X_test):
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-6
    return (X_train - mean) / std, (X_test - mean) / std

def calculer_matrice_confusion(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    true_idx = np.where(y_true == 1, 0, 1)
    pred_idx = np.where(y_pred == 1, 0, 1)
    
    matrice = np.zeros((2, 2), dtype=int)
    for t, p in zip(true_idx, pred_idx):
        matrice[t, p] += 1
    
    return matrice

# TESTS PMC AVEC MÊMES CONFIGURATIONS
def test_pmc_matrices_confusion():
    print("\n" + "="*60)
    print("PMC - Matrices de confusion")
    print("="*60)
    
    X, y = charger_dataset()  # Même fonction de chargement
    if X is None:
        return None, None, None, None
    
    X_train, X_test, y_train, y_test = split_manuel(X, y, test_size=0.3)  # Même split
    
    X_train_norm, X_test_norm = normaliser_manuel(X_train, X_test)  # Même normalisation
    
    print(f"Données chargées: {len(X)} échantillons")
    print(f"Train: {len(X_train_norm)} échantillons")
    print(f"Test: {len(X_test_norm)} échantillons")
    print(f"Features par échantillon: {X_train_norm.shape[1]}")
    
    print("\nEntraînement du PMC...")
    model = PMC(n_inputs=X_train_norm.shape[1], n_hidden=1, learning_rate=0.01)  # Même LR
    error = model.fit(X_train_norm, y_train, max_iterations=500)  # Même nombre d'itérations
    
    print(f"Erreur MSE finale: {error:.4f}")
    
    train_pred = model.predict_class(X_train_norm)
    test_pred = model.predict_class(X_test_norm)
    
    train_acc = np.mean(train_pred == y_train) * 100
    test_acc = np.mean(test_pred == y_test) * 100
    
    print(f"\nRÉSULTATS PMC :")
    print(f"  Accuracy Train: {train_acc:.1f}%")
    print(f"  Accuracy Test:  {test_acc:.1f}%")
    
    if train_acc > 95 and test_acc < 70:
        print(f"  OVERFITTING DÉTECTÉ !")
    elif train_acc < 70 and test_acc < 70:
        print(f"  UNDERFITTING DÉTECTÉ !")
    else:
        print(f"  BON ÉQUILIBRE TRAIN/TEST")
    
    train_cm = calculer_matrice_confusion(y_train, train_pred)
    test_cm = calculer_matrice_confusion(y_test, test_pred)
    
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    axes[0].imshow(train_cm, cmap='Blues', interpolation='nearest')
    axes[0].set_title(f'PMC - TRAIN\nAccuracy: {train_acc:.1f}%', fontsize=14, fontweight='bold')
    axes[0].set_xticks([0, 1])
    axes[0].set_yticks([0, 1])
    axes[0].set_xticklabels(['Guitare (1)', 'Piano (-1)'], fontsize=12)
    axes[0].set_yticklabels(['Guitare (1)', 'Piano (-1)'], fontsize=12)
    axes[0].set_ylabel('Vraie classe', fontsize=12)
    axes[0].set_xlabel('Prédite classe', fontsize=12)
    
    for i in range(2):
        for j in range(2):
            color = 'white' if train_cm[i, j] > train_cm.max()/2 else 'black'
            axes[0].text(j, i, str(train_cm[i, j]), 
                        ha='center', va='center', 
                        color=color, fontsize=16, fontweight='bold')
    
    axes[1].imshow(test_cm, cmap='Reds', interpolation='nearest')
    axes[1].set_title(f'PMC - TEST\nAccuracy: {test_acc:.1f}%', fontsize=14, fontweight='bold')
    axes[1].set_xticks([0, 1])
    axes[1].set_yticks([0, 1])
    axes[1].set_xticklabels(['Guitare (1)', 'Piano (-1)'], fontsize=12)
    axes[1].set_yticklabels(['Guitare (1)', 'Piano (-1)'], fontsize=12)
    axes[1].set_ylabel('Vraie classe', fontsize=12)
    axes[1].set_xlabel('Prédite classe', fontsize=12)
    
    for i in range(2):
        for j in range(2):
            color = 'white' if test_cm[i, j] > test_cm.max()/2 else 'black'
            axes[1].text(j, i, str(test_cm[i, j]), 
                        ha='center', va='center', 
                        color=color, fontsize=16, fontweight='bold')
    
    if train_acc > 95 and test_acc < 70:
        plt.figtext(0.5, 0.01, "OVERFITTING", 
                   ha='center', fontsize=12, color='red', fontweight='bold')
    elif train_acc < 70 and test_acc < 70:
        plt.figtext(0.5, 0.01, "UNDERFITTING", 
                   ha='center', fontsize=12, color='orange', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    return X_train_norm, X_test_norm, y_train, y_test, model

def test_pmc_courbe_loss():
    print("\n" + "="*60)
    print("PMC - Courbe de Loss")
    print("="*60)
    
    X, y = charger_dataset()  # Mêmes données
    if X is None:
        return
    
    X_train, X_test, y_train, y_test = split_manuel(X, y, test_size=0.3)  # Même split
    
    X_train_norm, X_test_norm = normaliser_manuel(X_train, X_test)  # Même normalisation
    
    print(f"PMC avec {len(X_train_norm)} échantillons d'entraînement")
    print(f"Architecture: {X_train_norm.shape[1]} -> 1 -> 1")  # Même architecture simple
    
    model = PMC(n_inputs=X_train_norm.shape[1], n_hidden=1, learning_rate=0.01)  # Mêmes paramètres
    
    train_losses = []
    test_losses = []
    epochs = 100  # Même nombre d'époques
    
    print("\nEntraînement en cours...")
    
    for epoch in range(epochs):
        # 1 itération par époque (comme le modèle linéaire)
        train_loss = model.fit(X_train_norm, y_train, max_iterations=1)
        train_losses.append(train_loss)
        
        # Calcul de la loss sur le test (MSE)
        test_pred = model.predict(X_test_norm)
        test_mse = np.mean((test_pred - y_test) ** 2)
        test_losses.append(test_mse)
        
        if epoch % 20 == 0:
            print(f"  Époque {epoch:3d}: train_loss={train_loss:.4f}, test_loss={test_mse:.4f}")
    
    print(f"\n  Train loss finale: {train_losses[-1]:.4f}")
    print(f"  Test loss finale:  {test_losses[-1]:.4f}")
    
    if test_losses[-1] > train_losses[-1] * 1.5:
        print(f"  OVERFITTING ! (ratio: {test_losses[-1]/train_losses[-1]:.1f}x)")
        status = "overfitting"
        status_color = "red"
    elif test_losses[-1] > 0.5 and train_losses[-1] > 0.5:
        print(f"  UNDERFITTING !")
        status = "underfitting"
        status_color = "orange"
    else:
        print(f"  Bon équilibre")
        status = "bon apprentissage"
        status_color = "green"
    
    plt.figure(figsize=(12, 7))
    
    plt.plot(train_losses, label='Train Loss', linewidth=2.5, color='blue', alpha=0.8)
    plt.plot(test_losses, label='Test Loss', linewidth=2.5, color='red', alpha=0.8)
    
    if test_losses[-1] > train_losses[-1] * 1.5:
        plt.fill_between(range(epochs), train_losses, test_losses, 
                        where=(np.array(test_losses) > np.array(train_losses)),
                        color='red', alpha=0.2, label='Zone d\'overfitting')
    
    plt.xlabel('Époque', fontsize=14)
    plt.ylabel('Loss (MSE)', fontsize=14)
    plt.title('PMC - Évolution de la Loss', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.text(0.02, 0.98, f'État: {status}', transform=plt.gca().transAxes,
            color=status_color, fontsize=14, ha='left', va='top', 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.text(0.02, 0.92, f'Loss finale - Train: {train_losses[-1]:.4f}', 
            transform=plt.gca().transAxes, fontsize=11, ha='left', va='top')
    plt.text(0.02, 0.88, f'Loss finale - Test:  {test_losses[-1]:.4f}', 
            transform=plt.gca().transAxes, fontsize=11, ha='left', va='top')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("="*60)
    print("Tests du PMC avec mêmes configurations que le modèle linéaire")
    print("="*60)
    
    dll_path = "./target/release/neural_networks.dll"
    if not Path(dll_path).exists():
        print(f"DLL non trouvée: {dll_path}")
    else:
        print("\n" + "="*60)
        print("TEST 1: Matrices de confusion")
        print("="*60)
        result = test_pmc_matrices_confusion()
        
        if result and result[0] is not None:
            print("\n" + "="*60)
            print("TEST 2: Courbe de Loss")
            print("="*60)
            test_pmc_courbe_loss()
        
        print("\n" + "="*60)
        print("Tests PMC terminés")
        print("="*60)