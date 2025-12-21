import ctypes
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path

#WRAPPER RUST
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

#DATASET
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
            images = list(path.glob("*.[pj][np]g"))[:6]
            for img in images:
                features = extraire_features(img)
                if features is not None:
                    X.append(features)
                    y.append(label)
    
    if len(X) == 0:
        print("Aucune image trouvée ")
        return None, None
    
    return np.array(X), np.array(y)

# FONCTIONS UTILES
def split_manuel(X, y, test_size=0.2):
    """Split"""
    n_samples = len(X)
    indices = np.arange(n_samples)
    np.random.seed(42)
    np.random.shuffle(indices)
    
    split_idx = int(n_samples * (1 - test_size))
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]
    
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

def normaliser_manuel(X_train, X_test):
    """Normalisation"""
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-6
    return (X_train - mean) / std, (X_test - mean) / std

def calculer_matrice_confusion(y_true, y_pred):
    """matrice de confusion"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    true_idx = np.where(y_true == 1, 0, 1)
    pred_idx = np.where(y_pred == 1, 0, 1)
    
    matrice = np.zeros((2, 2), dtype=int)
    for t, p in zip(true_idx, pred_idx):
        matrice[t, p] += 1
    
    return matrice

# TESTS
def test_matrices_confusion():
    """matrice de confusion"""
    print("\n" + "="*60)
    print("matrice de confusion")
    print("="*60)
    
    X, y = charger_dataset()
    if X is None:
        return None, None, None, None
    

    X_train, X_test, y_train, y_test = split_manuel(X, y, test_size=0.3)
    
    X_train_norm, X_test_norm = normaliser_manuel(X_train, X_test)
    
    print(f"Données chargées: {len(X)} échantillons")
    print(f"Train: {len(X_train_norm)} échantillons")
    print(f"Test: {len(X_test_norm)} échantillons")
    
    model = LinearModel(input_dim=X_train_norm.shape[1], learning_rate=0.01)
    model.fit(X_train_norm, y_train, max_iterations=500)

    train_pred = model.predict_class(X_train_norm)
    test_pred = model.predict_class(X_test_norm)
    
    train_acc = np.mean(train_pred == y_train) * 100
    test_acc = np.mean(test_pred == y_test) * 100
    
    print(f"\nAccuracy Train: {train_acc:.1f}%")
    print(f"Accuracy Test:  {test_acc:.1f}%")
    
    train_cm = calculer_matrice_confusion(y_train, train_pred)
    test_cm = calculer_matrice_confusion(y_test, test_pred)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].imshow(train_cm, cmap='Blues')
    axes[0].set_title(f'TRAIN - Accuracy: {train_acc:.1f}%')
    axes[0].set_xticks([0, 1])
    axes[0].set_yticks([0, 1])
    axes[0].set_xticklabels(['Guitare (1)', 'Piano (-1)'])
    axes[0].set_yticklabels(['Guitare (1)', 'Piano (-1)'])
    axes[0].set_ylabel('Vraie classe')
    axes[0].set_xlabel('Prédite classe')
    for i in range(2):
        for j in range(2):
            axes[0].text(j, i, str(train_cm[i, j]), 
                        ha='center', va='center', 
                        color='white' if train_cm[i, j] > train_cm.max()/2 else 'black',
                        fontweight='bold')
    
    axes[1].imshow(test_cm, cmap='Reds')
    axes[1].set_title(f'TEST - Accuracy: {test_acc:.1f}%')
    axes[1].set_xticks([0, 1])
    axes[1].set_yticks([0, 1])
    axes[1].set_xticklabels(['Guitare (1)', 'Piano (-1)'])
    axes[1].set_yticklabels(['Guitare (1)', 'Piano (-1)'])
    axes[1].set_ylabel('Vraie classe')
    axes[1].set_xlabel('Prédite classe')
    for i in range(2):
        for j in range(2):
            axes[1].text(j, i, str(test_cm[i, j]), 
                        ha='center', va='center', 
                        color='white' if test_cm[i, j] > test_cm.max()/2 else 'black',
                        fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    return X_train_norm, X_test_norm, y_train, y_test

def test_courbe_loss():
    """TEST 2: Courbe de loss"""
    print("\n" + "="*60)
    print("courbe de loss")
    print("="*60)
    
    # Chargements de données 
    X, y = charger_dataset()
    if X is None:
        return
    
    # Split
    X_train, X_test, y_train, y_test = split_manuel(X, y, test_size=0.3)
    
    # normalisation
    X_train_norm, X_test_norm = normaliser_manuel(X_train, X_test)
    

    model = LinearModel(input_dim=X_train_norm.shape[1], learning_rate=0.01)
    
    train_losses = []
    test_losses = []
    epochs = 100
    
    
    for epoch in range(epochs):

        train_loss = model.fit(X_train_norm, y_train, max_iterations=1)
        train_losses.append(train_loss)
        test_pred = model.predict_class(X_test_norm)
        test_accuracy = np.mean(test_pred == y_test)
        test_loss = 1.0 - test_accuracy 
        test_losses.append(test_loss)
        
        if epoch % 20 == 0:
            print(f"  Epoque {epoch}: train_loss={train_loss:.4f}, test_loss={test_loss:.4f}")
    

    if test_losses[-1] > train_losses[-1] * 1.5:
        print(f"\n  overfitting!")
        print(f"   Train loss finale: {train_losses[-1]:.4f}")
        print(f"   Test loss finale:  {test_losses[-1]:.4f}")
        print(f"   Ratio: {test_losses[-1]/train_losses[-1]:.1f}x")
    elif test_losses[-1] > 0.5:
        print(f"\n underfitting!")
        print(f"   Les deux losses sont élevées (>0.5)")
    else:
        print(f"\n  Bon apprentissage")
        print(f"   Les deux losses convergent")
    
    plt.figure(figsize=(10, 6))
    
    plt.plot(train_losses, label='Train Loss', linewidth=2, color='blue')
    plt.plot(test_losses, label='Test Loss', linewidth=2, color='red')
    
    if test_losses[-1] > train_losses[-1]:
        plt.fill_between(range(len(train_losses)), train_losses, test_losses, 
                        where=(np.array(test_losses) > np.array(train_losses)),
                        color='red', alpha=0.2, label='Zone overfitting')
    
    plt.xlabel('Epoque')
    plt.ylabel('Loss')
    plt.title('Courbe de Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    

    if test_losses[-1] > train_losses[-1] * 1.5:
        plt.text(0.5, 0.9, 'overfitting', transform=plt.gca().transAxes,
                color='red', fontsize=14, ha='center', fontweight='bold')
    elif test_losses[-1] > 0.5:
        plt.text(0.5, 0.9, 'underfitting', transform=plt.gca().transAxes,
                color='orange', fontsize=14, ha='center', fontweight='bold')
    else:
        plt.text(0.5, 0.9, 'apprentissage bon', transform=plt.gca().transAxes,
                color='green', fontsize=14, ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("="*60)
    print("="*60)
    

    dll_path = "./target/release/neural_networks.dll"
    if not Path(dll_path).exists():
        print(f"erreur {dll_path}")
        exit(1)
    
    #  Matrices de confusion
    print("\n" + "="*60)
    print("matrice de confusion")
    print("="*60)
    
    result = test_matrices_confusion()
    
    if result[0] is not None:
        # Courbe de loss
        print("\n" + "="*60)
        print("courbe de loss")
        print("="*60)
        
        test_courbe_loss()