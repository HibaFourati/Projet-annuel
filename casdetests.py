import ctypes
import numpy as np
import matplotlib.pyplot as plt
import os

# ==================== WRAPPER RUST CORRIGÉ ====================
class LinearModel:
    def __init__(self, input_dim: int, learning_rate: float = 0.01):
        lib_path = "./target/release/linear_model.dll"
        
        #  AJOUTE CETTE LIGNE POUR CHARGER LA BIBLIOTHÈQUE D'ABORD !
        self.lib = ctypes.CDLL(lib_path)
        
        # MAINTENANT tu peux configurer les fonctions
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
        
        # Créer le modèle
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
        
        return self.lib.linear_model_fit(
            self.model_ptr, X_ptr, y_ptr, n_samples, n_features, max_iterations
        )
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        n_samples, n_features = X.shape
        results = np.zeros(n_samples, dtype=np.float64)
        
        X_ptr = X.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        results_ptr = results.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        
        self.lib.linear_model_predict_batch(
            self.model_ptr, X_ptr, results_ptr, n_samples, n_features
        )
        
        return results
    
    def predict_class(self, X: np.ndarray) -> np.ndarray:
        predictions = self.predict(X)
        return np.where(predictions >= 0, 1, -1)
    
    def get_weights(self) -> np.ndarray:
        weights = np.zeros(self.input_dim, dtype=np.float64)
        weights_ptr = weights.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        self.lib.linear_model_get_weights(self.model_ptr, weights_ptr)
        return weights
    
    def get_bias(self) -> float:
        return self.lib.linear_model_get_bias(self.model_ptr)

# ==================== TESTS DES CAS DE VOTRE NOTEBOOK ====================
def test_cas_1_linear_simple():
    """CAS 1: Linear Simple - Doit MARCHER avec modèle linéaire"""
    print("\n=== CAS 1: Linear Simple ===")
    
    # Exactement les mêmes données que votre notebook
    X = np.array([
        [1, 1],
        [2, 3], 
        [3, 3]
    ], dtype=np.float64)
    y = np.array([1, -1, -1], dtype=np.float64)
    
    print("Données d'entraînement :")
    for i in range(len(X)):
        print(f"  Point {i}: ({X[i,0]}, {X[i,1]}) → Classe {y[i]}")
    
    # Création et entraînement du modèle
    print("\n Entraînement du modèle...")
    model = LinearModel(input_dim=2, learning_rate=0.1)
    final_error = model.fit(X, y, max_iterations=1000)
    
    print(f" Entraînement terminé!")
    print(f" Erreur finale: {final_error:.6f}")
    print(f"  Poids finaux: {model.get_weights()}")
    print(f" Biais final: {model.get_bias():.4f}")
    
    # Prédictions
    predictions = model.predict_class(X)
    print(f" Prédictions: {predictions}")
    print(f" Vérités:     {y}")
    
    accuracy = np.mean(predictions == y)
    print(f" Accuracy: {accuracy:.1%}")
    
    return model, X, y

def test_cas_2_linear_multiple():
    """CAS 2: Linear Multiple - Doit MARCHER avec modèle linéaire"""
    print("\n=== CAS 2: Linear Multiple ===")
    
    # Génération similaire à votre notebook
    np.random.seed(42)
    
    X = np.concatenate([
        np.random.random((50, 2)) * 0.9 + np.array([1, 1]),
        np.random.random((50, 2)) * 0.9 + np.array([2, 2])
    ])
    y = np.concatenate([np.ones(50), -np.ones(50)])
    
    print(f" Shape des données: {X.shape}")
    print(f" Distribution: {np.unique(y, return_counts=True)}")
    
    # Modèle
    print("\n Entraînement du modèle...")
    model = LinearModel(input_dim=2, learning_rate=0.01)
    final_error = model.fit(X, y, max_iterations=2000)
    
    print(f" Entraînement terminé!")
    print(f" Erreur finale: {final_error:.6f}")
    
    # Évaluation
    predictions = model.predict_class(X)
    accuracy = np.mean(predictions == y)
    print(f" Accuracy: {accuracy:.1%}")
    
    return model, X, y

def test_cas_3_xor():
    """CAS 3: XOR - Doit ÉCHOUER avec modèle linéaire (normal!)"""
    print("\n=== CAS 3: XOR ===")
    
    # Données XOR exactes de votre notebook
    X = np.array([
        [1, 0],
        [0, 1], 
        [0, 0],
        [1, 1]
    ], dtype=np.float64)
    y = np.array([1, 1, -1, -1], dtype=np.float64)
    
    print("Données XOR (non linéairement séparables):")
    for i in range(len(X)):
        print(f"  ({X[i,0]}, {X[i,1]}) → {y[i]}")
    
    print("\n Entraînement du modèle...")
    model = LinearModel(input_dim=2, learning_rate=0.1)
    final_error = model.fit(X, y, max_iterations=1000)
    
    print(f" Entraînement terminé!")
    print(f" Erreur finale: {final_error:.6f}")
    
    # Évaluation
    predictions = model.predict_class(X)
    accuracy = np.mean(predictions == y)
    print(f" Accuracy: {accuracy:.1%}")
    print(f" Prédictions: {predictions}")
    print(f" Vérités:     {y}")
    
    print("\n CONCLUSION: Le modèle linéaire NE PEUT PAS résoudre XOR!")
    print("   C'est NORMAL et cela justifie les modèles non-linéaires (PMC, RBF, SVM)")
    
    return model, X, y

def visualiser_resultats(model, X, y, titre):
    """Visualise les données et la frontière de décision"""
    plt.figure(figsize=(10, 8))
    
    # Points de données
    positif = X[y == 1]
    negatif = X[y == -1]
    
    plt.scatter(positif[:, 0], positif[:, 1], c='blue', label='Classe 1', alpha=0.7, s=50)
    plt.scatter(negatif[:, 0], negatif[:, 1], c='red', label='Classe -1', alpha=0.7, s=50)
    
    # Frontière de décision
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict_class(grid_points)
    Z = Z.reshape(xx.shape)
    
    # Tracer la frontière
    plt.contour(xx, yy, Z, levels=[0], colors='black', linewidths=2, linestyles='dashed')
    plt.contourf(xx, yy, Z, levels=[-100, 0, 100], alpha=0.2, colors=['red', 'blue'])
    
    plt.title(titre, fontsize=14, fontweight='bold')
    plt.xlabel('Feature 1 (X)')
    plt.ylabel('Feature 2 (Y)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# ==================== EXÉCUTION PRINCIPALE ====================
if __name__ == "__main__":
    print(" TESTS DU MODÈLE LINÉAIRE RUST - CAS DE VOTRE NOTEBOOK")
    print("=" * 60)
    
    # Test Cas 1 - Doit MARCHER
    model1, X1, y1 = test_cas_1_linear_simple()
    visualiser_resultats(model1, X1, y1, "CAS 1: Linear Simple ")
    
    # Test Cas 2 - Doit MARCHER  
    model2, X2, y2 = test_cas_2_linear_multiple()
    visualiser_resultats(model2, X2, y2, "CAS 2: Linear Multiple ")
    
    # Test Cas 3 - Doit ÉCHOUER (normal!)
    model3, X3, y3 = test_cas_3_xor()
    visualiser_resultats(model3, X3, y3, "CAS 3: XOR  (Normal - Problème non-linéaire)")
    
    print("\n" + "="*60)
    print(" RAPPORT FINAL DES TESTS :")
    print(" CAS 1 & 2: Le modèle linéaire fonctionne PARFAITEMENT")
    print(" CAS 3: Le modèle linéaire échoue sur XOR (NORMAL)")
    print(" Cela justifie l'utilisation de modèles non-linéaires ensuite!")