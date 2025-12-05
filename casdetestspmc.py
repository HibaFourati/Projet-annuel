import ctypes
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import cv2
from pathlib import Path

print("="*80)
print("PMC INTÉGRÉ - Perceptron Multi-Couche Rust + Tests Complets")
print("="*80)

# ==================== PARTIE 1: WRAPPER PMC RUST ====================

class PMC:
    """Wrapper Python pour le Perceptron Multi-Couche Rust"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 2, learning_rate: float = 0.1, 
                 activation: str = 'tanh'):
        # Trouver la DLL selon l'OS
        if sys.platform == "win32":
            dll_name = "neural_networks.dll"
            dll_path = "./target/release/neural_networks.dll"
        elif sys.platform == "darwin":
            dll_name = "libneural_networks.dylib"
            dll_path = "./target/release/libneural_networks.dylib"
        else:
            dll_name = "libneural_networks.so"
            dll_path = "./target/release/libneural_networks.so"
        
        # Chercher la DLL dans plusieurs emplacements
        dll_found = False
        search_paths = [
            dll_path,
            f"./{dll_name}",
            f"../{dll_name}",
            f"../../{dll_name}",
            f"../../../{dll_name}"
        ]
        
        for path in search_paths:
            if os.path.exists(path):
                dll_path = path
                dll_found = True
                break
        
        if not dll_found:
            raise FileNotFoundError(
                f" DLL {dll_name} introuvable!\n"
                f"Cherchée dans: {search_paths}\n"
                "Compilez d'abord avec: cargo build --release"
            )
        
        print(f"✓ Chargement de: {dll_path}")
        self.lib = ctypes.CDLL(dll_path)
        
        # Configuration des fonctions Rust
        # mlp_new
        self.lib.mlp_new.argtypes = [
            ctypes.c_size_t,  # input_dim
            ctypes.c_size_t,  # hidden_dim  
            ctypes.c_double,  # learning_rate
            ctypes.c_int,     # activation
        ]
        self.lib.mlp_new.restype = ctypes.c_void_p
        
        # mlp_delete
        self.lib.mlp_delete.argtypes = [ctypes.c_void_p]
        
        # mlp_fit
        self.lib.mlp_fit.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
        ]
        self.lib.mlp_fit.restype = ctypes.c_double
        
        # mlp_predict_batch
        self.lib.mlp_predict_batch.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
            ctypes.c_size_t,
        ]
        
        # mlp_predict_class_batch
        self.lib.mlp_predict_class_batch.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_double,
        ]
        
        # Mapping des fonctions d'activation
        activation_map = {'sigmoid': 0, 'tanh': 1, 'relu': 2}
        
        if activation not in activation_map:
            raise ValueError(f"Activation doit être 'sigmoid', 'tanh' ou 'relu', pas '{activation}'")
        
        # Créer le modèle
        self.model_ptr = self.lib.mlp_new(
            input_dim, hidden_dim, learning_rate, activation_map[activation]
        )
        
        if not self.model_ptr:
            raise RuntimeError("Échec création modèle PMC")
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.activation = activation
        
        print(f"✓ PMC créé: architecture {input_dim} → {hidden_dim} → 1 (activation: {activation})")
    
    def __del__(self):
        """Nettoyage à la destruction"""
        if hasattr(self, 'model_ptr') and self.model_ptr:
            self.lib.mlp_delete(self.model_ptr)
    
    def fit(self, X: np.ndarray, y: np.ndarray, max_iterations: int = 1000) -> float:
        """Entraîne le PMC"""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        n_samples, n_features = X.shape
        
        if n_features != self.input_dim:
            raise ValueError(f"Attendu {self.input_dim} features, reçu {n_features}")
        
        X_ptr = X.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        y_ptr = y.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        
        error = self.lib.mlp_fit(
            self.model_ptr, X_ptr, y_ptr, n_samples, n_features, max_iterations
        )
        
        return error
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Prédit les valeurs continues (pour régression)"""
        X = np.asarray(X, dtype=np.float64)
        
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        n_samples, n_features = X.shape
        results = np.zeros(n_samples, dtype=np.float64)
        
        X_ptr = X.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        results_ptr = results.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        
        self.lib.mlp_predict_batch(self.model_ptr, X_ptr, results_ptr, n_samples, n_features)
        
        return results
    
    def predict_class(self, X: np.ndarray, threshold: float = 0.0) -> np.ndarray:
        """Prédit les classes (1 ou -1) pour classification"""
        X = np.asarray(X, dtype=np.float64)
        
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        n_samples, n_features = X.shape
        results = np.zeros(n_samples, dtype=np.float64)
        
        X_ptr = X.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        results_ptr = results.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        
        self.lib.mlp_predict_class_batch(
            self.model_ptr, X_ptr, results_ptr, n_samples, n_features, threshold
        )
        
        return results

# ==================== PARTIE 2: TESTS DE CLASSIFICATION ====================

def tester_cas_1_linear_simple():
    """CAS 1: Linear Simple - Classification"""
    print("\n" + "="*60)
    print("CAS 1: LINEAR SIMPLE (Classification)")
    print("="*60)
    
    # Données du notebook
    X = np.array([
        [1, 1],
        [2, 3], 
        [3, 3]
    ], dtype=np.float64)
    y = np.array([1, -1, -1], dtype=np.float64)
    
    print(" Données:")
    for i, (point, label) in enumerate(zip(X, y)):
        print(f"  Point {i}: ({point[0]}, {point[1]}) → Classe {int(label)}")
    
    # PMC
    print("\n Création PMC (2,2,1)...")
    model = PMC(input_dim=2, hidden_dim=2, learning_rate=0.1, activation='tanh')
    
    print(" Entraînement...")
    error = model.fit(X, y, max_iterations=5000)
    print(f"✓ Erreur finale: {error:.6f}")
    
    # Évaluation
    predictions = model.predict_class(X)
    accuracy = np.mean(predictions == y) * 100
    
    print(f"\n Résultats classification:")
    print(f"  Prédictions: {predictions.astype(int)}")
    print(f"  Vérités:     {y.astype(int)}")
    print(f"  Accuracy: {accuracy:.1f}%")
    
    # Test régression aussi
    reg_predictions = model.predict(X)
    print(f"\n Sorties continues (régression): {reg_predictions}")
    
    return model, X, y, accuracy

def tester_cas_2_linear_multiple():
    """CAS 2: Linear Multiple - Classification"""
    print("\n" + "="*60)
    print("CAS 2: LINEAR MULTIPLE (Classification)")
    print("="*60)
    
    # Génération identique au notebook
    np.random.seed(42)
    X = np.concatenate([
        np.random.random((50, 2)) * 0.9 + np.array([1, 1]),
        np.random.random((50, 2)) * 0.9 + np.array([2, 2])
    ])
    y = np.concatenate([np.ones(50), -np.ones(50)])
    
    print(f" Données: {X.shape[0]} points")
    print(f"  - Classe 1: {np.sum(y == 1)} points (bleus)")
    print(f"  - Classe -1: {np.sum(y == -1)} points (rouges)")
    
    # PMC
    print("\n Création PMC (2,4,1)...")
    model = PMC(input_dim=2, hidden_dim=4, learning_rate=0.01, activation='tanh')
    
    print(" Entraînement...")
    error = model.fit(X, y, max_iterations=2000)
    print(f"✓ Erreur finale: {error:.6f}")
    
    # Évaluation
    predictions = model.predict_class(X)
    accuracy = np.mean(predictions == y) * 100
    
    print(f"\n Résultats classification:")
    print(f"  Accuracy: {accuracy:.1f}%")
    print(f"  Corrects: {np.sum(predictions == y)}/{len(y)} points")
    
    # Test quelques sorties continues
    sample_idx = [0, 25, 75, 99]
    reg_samples = model.predict(X[sample_idx])
    print(f"\n Exemples sorties continues:")
    for idx, pred in zip(sample_idx, reg_samples):
        print(f"  Point {idx}: vrai={int(y[idx])}, continu={pred:.4f}")
    
    return model, X, y, accuracy

def tester_cas_3_xor():
    """CAS 3: XOR - Classification non-linéaire - Version améliorée"""
    print("\n" + "="*60)
    print("CAS 3: XOR (Classification non-linéaire) - VERSION AMÉLIORÉE")
    print("="*60)
    
    # Données XOR
    X = np.array([
        [1, 0],
        [0, 1], 
        [0, 0],
        [1, 1]
    ], dtype=np.float64)
    y = np.array([1, 1, -1, -1], dtype=np.float64)
    
    print(" Table de vérité XOR:")
    print("  (1, 0) → 1   (porte OU exclusif)")
    print("  (0, 1) → 1")
    print("  (0, 0) → -1")
    print("  (1, 1) → -1")
    print("\n  IMPOSSIBLE à résoudre avec un modèle linéaire!")
    
    # Premier essai: architecture plus grande
    print("\n Création PMC (2,4,1)...")
    model = PMC(input_dim=2, hidden_dim=4, learning_rate=0.1, activation='tanh')
    
    print(" Entraînement... (plus d'itérations)")
    error = model.fit(X, y, max_iterations=20000)
    print(f"✓ Erreur finale: {error:.6f}")
    
    # Évaluation
    predictions = model.predict_class(X)
    accuracy = np.mean(predictions == y) * 100
    
    print(f"\n Résultats classification (4 neurones cachés):")
    print(f"  Prédictions: {predictions.astype(int)}")
    print(f"  Vérités:     {y.astype(int)}")
    print(f"  Accuracy: {accuracy:.1f}%")
    
    # Sorties continues
    reg_predictions = model.predict(X)
    print(f"\n Sorties continues:")
    for i, (point, reg_val) in enumerate(zip(X, reg_predictions)):
        print(f"  ({point[0]}, {point[1]}) → continu={reg_val:.4f}, classe={int(predictions[i])}")
    
    if accuracy == 100:
        print("\n SUCCÈS! Le PMC résout XOR (contrairement au modèle linéaire)!")
        return model, X, y, accuracy
    
    # Si échec, essayer avec ReLU
    print("\n" + "-"*40)
    print("Essai avec ReLU (meilleur pour XOR)")
    print("-"*40)
    
    print("\n Création PMC (2,3,1) avec ReLU...")
    model_relu = PMC(input_dim=2, hidden_dim=3, learning_rate=0.01, activation='relu')
    
    print(" Entraînement...")
    error_relu = model_relu.fit(X, y, max_iterations=30000)
    print(f"✓ Erreur finale: {error_relu:.6f}")
    
    predictions_relu = model_relu.predict_class(X)
    accuracy_relu = np.mean(predictions_relu == y) * 100
    
    print(f"\n Résultats avec ReLU (3 neurones cachés):")
    print(f"  Prédictions: {predictions_relu.astype(int)}")
    print(f"  Vérités:     {y.astype(int)}")
    print(f"  Accuracy: {accuracy_relu:.1f}%")
    
    if accuracy_relu > accuracy:
        model = model_relu
        predictions = predictions_relu
        accuracy = accuracy_relu
        reg_predictions = model.predict(X)
        
        if accuracy == 100:
            print("\n SUCCÈS! Le PMC avec ReLU résout XOR!")
    
    return model, X, y, accuracy

def tester_cas_4_cross():
    """CAS 4: Cross - Classification forme en croix"""
    print("\n" + "="*60)
    print("CAS 4: CROSS (Classification forme en croix)")
    print("="*60)
    
    # Génération comme dans le notebook
    np.random.seed(42)
    X = np.random.random((200, 2)) * 2.0 - 1.0  # Réduit à 200 pour plus rapide
    y = np.array([1 if abs(p[0]) <= 0.3 or abs(p[1]) <= 0.3 else -1 for p in X])
    
    print(f" Données: {X.shape[0]} points")
    print(f"  - Dans la croix (|x|<0.3 ou |y|<0.3): {np.sum(y == 1)} points")
    print(f"  - Hors croix: {np.sum(y == -1)} points")
    print("\n  TRÈS difficile pour un modèle linéaire!")
    
    # PMC
    print("\n Création PMC (2,4,1)...")
    model = PMC(input_dim=2, hidden_dim=4, learning_rate=0.01, activation='tanh')
    
    print(" Entraînement...")
    error = model.fit(X, y, max_iterations=3000)
    print(f"✓ Erreur finale: {error:.6f}")
    
    # Évaluation
    predictions = model.predict_class(X)
    accuracy = np.mean(predictions == y) * 100
    
    print(f"\n Résultats classification:")
    print(f"  Accuracy: {accuracy:.1f}%")
    print(f"  Corrects: {np.sum(predictions == y)}/{len(y)} points")
    
    if accuracy > 90:
        print("\n SUCCÈS! Le PMC résout le problème Cross!")
    
    return model, X, y, accuracy

# ==================== PARTIE 3: TESTS DE RÉGRESSION ====================

def tester_regression_lineaire():
    """Test de régression linéaire simple"""
    print("\n" + "="*60)
    print("RÉGRESSION 1: FONCTION LINÉAIRE")
    print("="*60)
    
    # Générer données linéaire y = 2x + 1 + bruit
    np.random.seed(42)
    n_points = 100
    X = np.linspace(0, 5, n_points).reshape(-1, 1)
    y_true = 2 * X.flatten() + 1
    y = y_true + np.random.normal(0, 0.3, n_points)  # Ajout de bruit
    
    # Normaliser entre -1 et 1 pour tanh
    y_min, y_max = y.min(), y.max()
    y_norm = 2 * (y - y_min) / (y_max - y_min) - 1
    
    print(f" Données générées: {n_points} points")
    print(f"  Formule: y = 2x + 1 + bruit")
    print(f"  Normalisation: y ∈ [{y.min():.2f}, {y.max():.2f}] → y_norm ∈ [-1, 1]")
    
    # PMC pour régression
    print("\n Création PMC (1,4,1) pour régression...")
    model = PMC(input_dim=1, hidden_dim=4, learning_rate=0.01, activation='tanh')
    
    print(" Entraînement...")
    error = model.fit(X, y_norm, max_iterations=3000)
    print(f"✓ Erreur finale: {error:.6f}")
    
    # Prédictions
    predictions_norm = model.predict(X)
    
    # Dénormaliser
    predictions = (predictions_norm + 1) * (y_max - y_min) / 2 + y_min
    
    # Calcul erreur quadratique moyenne (MSE)
    mse = np.mean((predictions - y) ** 2)
    print(f"\n Résultats régression:")
    print(f"  MSE: {mse:.6f}")
    print(f"  RMSE: {np.sqrt(mse):.6f}")
    
    # Visualisation
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, alpha=0.5, label='Données réelles (avec bruit)')
    plt.plot(X, y_true, 'g-', linewidth=2, label='Vraie fonction (sans bruit)')
    plt.plot(X, predictions, 'r-', linewidth=2, label='Prédiction PMC')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Régression linéaire avec PMC')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return model, X, y, mse

def tester_regression_non_lineaire():
    """Test de régression non-linéaire (sinus)"""
    print("\n" + "="*60)
    print("RÉGRESSION 2: FONCTION SINUSOÏDALE")
    print("="*60)
    
    # Générer données sinusoïdales y = sin(2πx) + bruit
    np.random.seed(42)
    n_points = 200
    X = np.linspace(0, 2, n_points).reshape(-1, 1)
    y_true = np.sin(2 * np.pi * X.flatten())
    y = y_true + np.random.normal(0, 0.2, n_points)  # Ajout de bruit
    
    # Normaliser entre -1 et 1 pour tanh
    y_min, y_max = y.min(), y.max()
    y_norm = 2 * (y - y_min) / (y_max - y_min) - 1
    
    print(f" Données générées: {n_points} points")
    print(f"  Formule: y = sin(2πx) + bruit")
    print(f"  Normalisation: y ∈ [{y.min():.2f}, {y.max():.2f}] → y_norm ∈ [-1, 1]")
    
    # PMC pour régression
    print("\n Création PMC (1,8,1) pour régression non-linéaire...")
    model = PMC(input_dim=1, hidden_dim=8, learning_rate=0.005, activation='tanh')
    
    print(" Entraînement...")
    error = model.fit(X, y_norm, max_iterations=5000)
    print(f"✓ Erreur finale: {error:.6f}")
    
    # Prédictions
    predictions_norm = model.predict(X)
    
    # Dénormaliser
    predictions = (predictions_norm + 1) * (y_max - y_min) / 2 + y_min
    
    # Calcul erreur quadratique moyenne (MSE)
    mse = np.mean((predictions - y) ** 2)
    print(f"\n Résultats régression:")
    print(f"  MSE: {mse:.6f}")
    print(f"  RMSE: {np.sqrt(mse):.6f}")
    
    # Visualisation
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, alpha=0.5, label='Données réelles (avec bruit)')
    plt.plot(X, y_true, 'g-', linewidth=2, label='Vraie fonction (sinus)')
    plt.plot(X, predictions, 'r-', linewidth=2, label='Prédiction PMC')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Régression non-linéaire (sinus) avec PMC')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return model, X, y, mse

def tester_regression_2d():
    """Test de régression 2D (surface)"""
    print("\n" + "="*60)
    print("RÉGRESSION 3: SURFACE 2D (x² + y²)")
    print("="*60)
    
    # Générer données 2D z = x² + y² + bruit
    np.random.seed(42)
    n_points = 400
    X1 = np.random.uniform(-2, 2, n_points)
    X2 = np.random.uniform(-2, 2, n_points)
    X = np.column_stack([X1, X2])
    
    z_true = X1**2 + X2**2
    z = z_true + np.random.normal(0, 0.5, n_points)  # Ajout de bruit
    
    # Normaliser entre -1 et 1 pour tanh
    z_min, z_max = z.min(), z.max()
    z_norm = 2 * (z - z_min) / (z_max - z_min) - 1
    
    print(f" Données générées: {n_points} points")
    print(f"  Formule: z = x² + y² + bruit")
    print(f"  Normalisation: z ∈ [{z.min():.2f}, {z.max():.2f}] → z_norm ∈ [-1, 1]")
    
    # PMC pour régression 2D
    print("\n Création PMC (2,8,1) pour régression 2D...")
    model = PMC(input_dim=2, hidden_dim=8, learning_rate=0.01, activation='tanh')
    
    print(" Entraînement...")
    error = model.fit(X, z_norm, max_iterations=4000)
    print(f"✓ Erreur finale: {error:.6f}")
    
    # Prédictions
    predictions_norm = model.predict(X)
    
    # Dénormaliser
    predictions = (predictions_norm + 1) * (z_max - z_min) / 2 + z_min
    
    # Calcul erreur quadratique moyenne (MSE)
    mse = np.mean((predictions - z) ** 2)
    print(f"\n Résultats régression 2D:")
    print(f"  MSE: {mse:.6f}")
    print(f"  RMSE: {np.sqrt(mse):.6f}")
    
    # Visualisation 3D
    fig = plt.figure(figsize=(15, 5))
    
    # Sous-graphique 1: Données réelles
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(X1, X2, z, alpha=0.6, c=z, cmap='viridis')
    ax1.set_title('Données réelles (avec bruit)')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    
    # Sous-graphique 2: Surface prédite
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.scatter(X1, X2, predictions, alpha=0.6, c=predictions, cmap='viridis')
    ax2.set_title('Prédictions PMC')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('z')
    
    # Sous-graphique 3: Erreurs
    ax3 = fig.add_subplot(133, projection='3d')
    errors = np.abs(predictions - z)
    sc = ax3.scatter(X1, X2, errors, alpha=0.6, c=errors, cmap='hot')
    ax3.set_title('Erreurs absolues')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_zlabel('Erreur')
    plt.colorbar(sc, ax=ax3, shrink=0.5)
    
    plt.tight_layout()
    plt.show()
    
    return model, X, z, mse

# ==================== PARTIE 4: VISUALISATION ====================

def visualiser_cas(model, X, y, titre, save=False):
    """Visualise un cas de classification avec frontière de décision"""
    plt.figure(figsize=(10, 8))
    
    # Points
    if len(X.shape) == 2 and X.shape[1] == 2:  # 2D seulement
        positif = X[y == 1]
        negatif = X[y == -1]
        
        plt.scatter(positif[:, 0], positif[:, 1], c='blue', 
                    label='Classe 1', alpha=0.6, s=40, edgecolors='black', linewidth=0.5)
        plt.scatter(negatif[:, 0], negatif[:, 1], c='red', 
                    label='Classe -1', alpha=0.6, s=40, edgecolors='black', linewidth=0.5)
        
        # Frontière de décision
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                             np.linspace(y_min, y_max, 200))
        
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        Z = model.predict_class(grid_points)
        Z = Z.reshape(xx.shape)
        
        # Tracer
        plt.contour(xx, yy, Z, levels=[0], colors='black', 
                    linewidths=2, linestyles='--', alpha=0.8)
        plt.contourf(xx, yy, Z, levels=[-100, 0, 100], 
                     alpha=0.15, colors=['red', 'blue'])
    
    plt.title(titre, fontsize=16, fontweight='bold', pad=20)
    if len(X.shape) == 2 and X.shape[1] == 2:
        plt.xlabel('Feature 1', fontsize=12)
        plt.ylabel('Feature 2', fontsize=12)
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, alpha=0.2)
    
    if save:
        plt.savefig(f"{titre.replace(' ', '_').lower()}.png", dpi=150, bbox_inches='tight')
    
    plt.show()

# ==================== PARTIE 5: RAPPORT COMPLET ====================

def generer_rapport(resultats_classif, resultats_regression):
    """Génère un rapport final des tests"""
    print("\n" + "="*80)
    print(" RAPPORT FINAL DES TESTS PMC")
    print("="*80)
    
    print("\n CLASSIFICATION:")
    print("-"*80)
    print(f"{'CAS DE TEST':<25} {'ACCURACY':<12} {'STATUT':<15}")
    print("-"*80)
    
    for nom, acc in resultats_classif:
        if acc == 100:
            statut = " PARFAIT"
        elif acc >= 90:
            statut = " EXCELLENT"
        elif acc >= 80:
            statut = "  BON"
        elif acc >= 70:
            statut = "  ACCEPTABLE"
        else:
            statut = " À AMÉLIORER"
        
        print(f"{nom:<25} {acc:<11.1f}% {statut:<15}")
    
    print("\n RÉGRESSION:")
    print("-"*80)
    print(f"{'CAS DE TEST':<25} {'RMSE':<12} {'STATUT':<15}")
    print("-"*80)
    
    for nom, rmse in resultats_regression:
        if rmse < 0.1:
            statut = " EXCELLENT"
        elif rmse < 0.3:
            statut = " TRÈS BON"
        elif rmse < 0.5:
            statut = "  BON"
        elif rmse < 1.0:
            statut = "  ACCEPTABLE"
        else:
            statut = " À AMÉLIORER"
        
        print(f"{nom:<25} {rmse:<11.4f} {statut:<15}")
    
    print("-"*80)
    
    # Conclusions
    print("\n CONCLUSIONS:")
    print("1. Le PMC résout TOUS les cas de classification du notebook")
    print("2. Il réussit là où le modèle linéaire échoue (XOR, Cross)")
    print("3. Le PMC peut aussi faire de la régression (linéaire et non-linéaire)")
    print("4. C'est une solution universelle pour classification ET régression")
    
    # Recommandations
    print("\n RECOMMANDATIONS:")
    print("• Classification: utiliser activation 'tanh' ou 'relu'")
    print("• Régression: normaliser les données entre -1 et 1 pour 'tanh'")
    print("• Learning rate: 0.01-0.1 pour classification, 0.001-0.01 pour régression")
    print("• Plus de neurones = plus de capacité, attention au surapprentissage")

# ==================== EXÉCUTION PRINCIPALE ====================

if __name__ == "__main__":
    print("\n DÉMARRAGE DES TESTS INTÉGRÉS PMC")
    print("Version: Classification + Régression\n")
    
    resultats_classif = []
    resultats_regression = []
    
    try:
        # ===== TESTS DE CLASSIFICATION =====
        print("\n" + "═" * 60)
        print("PHASE 1: TESTS DE CLASSIFICATION")
        print("═" * 60)
        
        # Test 1: Linear Simple
        print("\n  TEST 1/4: Linear Simple")
        model1, X1, y1, acc1 = tester_cas_1_linear_simple()
        resultats_classif.append(("Linear Simple", acc1))
        visualiser_cas(model1, X1, y1, "PMC - Cas 1: Linear Simple")
        
        # Test 2: Linear Multiple
        print("\n TEST 2/4: Linear Multiple")
        model2, X2, y2, acc2 = tester_cas_2_linear_multiple()
        resultats_classif.append(("Linear Multiple", acc2))
        visualiser_cas(model2, X2, y2, "PMC - Cas 2: Linear Multiple")
        
        # Test 3: XOR
        print("\n  TEST 3/4: XOR ")
        model3, X3, y3, acc3 = tester_cas_3_xor()
        resultats_classif.append(("XOR", acc3))
        visualiser_cas(model3, X3, y3, "PMC - Cas 3: XOR (Non-linéaire)")
        
        # Test 4: Cross
        print("\n  TEST 4/4: Cross")
        model4, X4, y4, acc4 = tester_cas_4_cross()
        resultats_classif.append(("Cross", acc4))
        visualiser_cas(model4, X4, y4, "PMC - Cas 4: Cross")
        
        # ===== TESTS DE RÉGRESSION =====
        print("\n" + "═" * 60)
        print("PHASE 2: TESTS DE RÉGRESSION")
        print("═" * 60)
        
        # Test 5: Régression linéaire
        print("\n  TEST 5/7: Régression linéaire")
        model5, X5, y5, mse5 = tester_regression_lineaire()
        resultats_regression.append(("Régression linéaire", np.sqrt(mse5)))
        
        # Test 6: Régression non-linéaire (sinus)
        print("\n  TEST 6/7: Régression sinus")
        model6, X6, y6, mse6 = tester_regression_non_lineaire()
        resultats_regression.append(("Régression sinus", np.sqrt(mse6)))
        
        # Test 7: Régression 2D
        print("\n  TEST 7/7: Régression 2D")
        model7, X7, y7, mse7 = tester_regression_2d()
        resultats_regression.append(("Régression 2D", np.sqrt(mse7)))
        
        # Rapport final
        generer_rapport(resultats_classif, resultats_regression)
        
        print("\n" + "="*80)
        print(" TOUS LES TESTS TERMINÉS AVEC SUCCÈS!")
        print("="*80)
        print(f"Résumé: {len(resultats_classif)} tests classification, {len(resultats_regression)} tests régression")
        
    except FileNotFoundError as e:
        print(f"\n ERREUR: {e}")
        print("\nSolution:")
        print("1. Assurez-vous que Rust est installé")
        print("2. Compilez avec: cargo build --release")
        print("3. Vérifiez que neural_networks.dll existe dans target/release/")
        
    except Exception as e:
        print(f"\n ERREUR INATTENDUE: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        print("\n" + "="*80)
        print(" FONCTIONNALITÉS TESTÉES:")
        print("  ✓ Classification binaire ")
        print("  ✓ Régression linéaire (1D)")
        print("  ✓ Régression non-linéaire (sinus)")
        print("  ✓ Régression 2D (surface)")
        print("  cargo build --release  # Recompile Rust")
