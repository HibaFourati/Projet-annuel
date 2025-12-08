import ctypes
import numpy as np
import os
import cv2
from pathlib import Path
import matplotlib.pyplot as plt

print("="*80)
print("PMC - TEST SIMPLIFIE SUR DATASET AVEC VISUALISATION")
print("="*80)

# ==================== PMC WRAPPER ====================

class PMC:
    def __init__(self, input_dim: int, hidden_dim: int = 2, learning_rate: float = 0.1, activation: str = 'tanh'):
        dll_path = "./target/release/neural_networks.dll"
        if not os.path.exists(dll_path):
            raise FileNotFoundError("DLL introuvable! Compilez avec: cargo build --release")
        
        self.lib = ctypes.CDLL(dll_path)
        
        self.lib.mlp_new.argtypes = [ctypes.c_size_t, ctypes.c_size_t, ctypes.c_double, ctypes.c_int]
        self.lib.mlp_new.restype = ctypes.c_void_p
        
        self.lib.mlp_delete.argtypes = [ctypes.c_void_p]
        
        self.lib.mlp_fit.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double), 
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t
        ]
        self.lib.mlp_fit.restype = ctypes.c_double
        
        self.lib.mlp_predict_class_batch.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t, 
            ctypes.c_size_t,
            ctypes.c_double
        ]
        
        activation_map = {'sigmoid': 0, 'tanh': 1, 'relu': 2}
        self.model_ptr = self.lib.mlp_new(input_dim, hidden_dim, learning_rate, activation_map[activation])
        
        if not self.model_ptr:
            raise RuntimeError("Echec creation modele PMC")
        
        self.input_dim = input_dim
        print(f"PMC cree: {input_dim} -> {hidden_dim} -> 1 (activation: {activation})")
    
    def __del__(self):
        if hasattr(self, 'model_ptr') and self.model_ptr:
            self.lib.mlp_delete(self.model_ptr)
    
    def fit(self, X: np.ndarray, y: np.ndarray, max_iterations: int = 1000) -> float:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        n_samples, n_features = X.shape
        
        X_ptr = X.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        y_ptr = y.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        
        error = self.lib.mlp_fit(
            self.model_ptr, 
            X_ptr, 
            y_ptr, 
            ctypes.c_size_t(n_samples),
            ctypes.c_size_t(n_features),
            ctypes.c_size_t(max_iterations)
        )
        return error
    
    def predict_class(self, X: np.ndarray, threshold: float = 0.0) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        n_samples, n_features = X.shape
        results = np.zeros(n_samples, dtype=np.float64)
        
        X_ptr = X.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        results_ptr = results.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        
        self.lib.mlp_predict_class_batch(
            self.model_ptr, 
            X_ptr, 
            results_ptr, 
            ctypes.c_size_t(n_samples),
            ctypes.c_size_t(n_features),
            ctypes.c_double(threshold)
        )
        return results

# ==================== CHARGER DATASET ====================

def charger_features_simples(n_images=3, taille=(8, 8)):
    """Charge des features simples"""
    print(f"\nChargement de {n_images} images par classe...")
    
    classes = ["guitare", "piano"]
    dataset_dir = Path("dataset")
    
    X, y = [], []
    
    for classe in classes:
        classe_dir = dataset_dir / classe
        if not classe_dir.exists():
            print(f"AVERTISSEMENT: Dossier {classe_dir} manquant")
            continue
        
        images = list(classe_dir.glob("*.[jp][pn]g"))[:n_images]
        
        for img_path in images:
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, taille)
                
                features = []
                features.append(resized.mean() / 255.0 * 2 - 1)
                features.append(resized.std() / 128.0 - 1)
                features.append(np.median(resized) / 255.0 * 2 - 1)
                
                for i in range(5):
                    x, y_pos = np.random.randint(0, taille[0]), np.random.randint(0, taille[1])
                    features.append(resized[x, y_pos] / 255.0 * 2 - 1)
                
                X.append(features)
                y.append(1 if classe == "guitare" else -1)
                
            except Exception as e:
                print(f"Erreur sur {img_path}: {e}")
    
    if not X:
        print("ERREUR: Aucune image chargee")
        return None, None
    
    print(f"OK: {len(X)} images chargees, {len(X[0])} features")
    return np.array(X, dtype=np.float64), np.array(y, dtype=np.float64)

# ==================== VISUALISATION COURBES ====================

def tracer_courbes_decision(X, y, model, titre="Frontiere de decision"):
    """Trace la frontiere de decision si on a 2 features"""
    if X.shape[1] < 2:
        print("  (Pas assez de features pour tracer la frontiere)")
        return
    
    plt.figure(figsize=(10, 8))
    
    # Points des donnees
    couleurs = ['blue' if label == 1 else 'red' for label in y]
    plt.scatter(X[:, 0], X[:, 1], c=couleurs, alpha=0.6, edgecolors='black', s=80)
    
    # Creer une grille pour la frontiere
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    
    # Pour chaque point de la grille, creer un vecteur de features
    # (on complete avec les moyennes pour les autres features)
    grid_points = []
    for i in range(xx.ravel().shape[0]):
        point = np.zeros(model.input_dim)
        point[0] = xx.ravel()[i]
        point[1] = yy.ravel()[i]
        # Pour les autres features, mettre la moyenne
        if model.input_dim > 2:
            point[2:] = X[:, 2:].mean(axis=0)
        grid_points.append(point)
    
    grid_points = np.array(grid_points)
    
    # Predire sur la grille
    Z = model.predict_class(grid_points)
    Z = Z.reshape(xx.shape)
    
    # Tracer la frontiere
    plt.contourf(xx, yy, Z, alpha=0.2, cmap='coolwarm')
    plt.contour(xx, yy, Z, colors='black', linewidths=1, alpha=0.5)
    
    plt.xlabel('Feature 1 (Moyenne des pixels)')
    plt.ylabel('Feature 2 (Ecart-type des pixels)')
    plt.title(titre)
    plt.grid(True, alpha=0.3)
    plt.show()

def tracer_histogrammes_predictions(y_true, y_pred, titre="Distribution des predictions"):
    """Trace un histogramme des predictions vs vraies valeurs"""
    plt.figure(figsize=(10, 5))
    
    # Separer les predictions correctes et incorrectes
    correct = y_true == y_pred
    incorrect = y_true != y_pred
    
    # Valeurs pour guitare (1) et piano (-1)
    indices = np.arange(len(y_true))
    
    plt.bar(indices[correct], y_true[correct], color='green', alpha=0.7, label='Correct')
    plt.bar(indices[incorrect], y_true[incorrect], color='red', alpha=0.7, label='Incorrect')
    
    # Tracer les predictions
    plt.scatter(indices, y_pred, color='black', s=100, marker='x', label='Prediction')
    
    plt.xlabel('Image')
    plt.ylabel('Classe (1=Guitare, -1=Piano)')
    plt.title(titre)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    plt.show()

def tracer_comparaison_features(X, y):
    """Trace un scatter plot des deux premieres features"""
    if X.shape[1] < 2:
        return
    
    plt.figure(figsize=(10, 8))
    
    # Separer les classes
    indices_guitare = np.where(y == 1)[0]
    indices_piano = np.where(y == -1)[0]
    
    plt.scatter(X[indices_guitare, 0], X[indices_guitare, 1], 
                color='blue', alpha=0.7, s=100, edgecolors='black', 
                label='Guitare', marker='o')
    
    plt.scatter(X[indices_piano, 0], X[indices_piano, 1], 
                color='red', alpha=0.7, s=100, edgecolors='black',
                label='Piano', marker='s')
    
    plt.xlabel('Feature 1: Moyenne des pixels')
    plt.ylabel('Feature 2: Ecart-type des pixels')
    plt.title('Visualisation des donnees: Guitare vs Piano')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# ==================== TEST PMC AVEC COURBES ====================

def test_pmc_avec_courbes():
    """Test PMC avec visualisations"""
    print("\n" + "="*60)
    print("TEST PMC AVEC VISUALISATIONS")
    print("="*60)
    
    # 1. Charger donnees
    X, y = charger_features_simples(n_images=5, taille=(8, 8))
    if X is None:
        return
    
    print(f"\nDONNEES:")
    print(f"  - Images: {len(X)}")
    print(f"  - Features: {X.shape[1]}")
    print(f"  - Guitare (1): {np.sum(y == 1)}")
    print(f"  - Piano (-1): {np.sum(y == -1)}")
    
    # 2. Normalisation
    X_normalized = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-6)
    
    # 3. Tracer les donnees AVANT entrainement
    print("\nVISUALISATION des donnees avant entrainement:")
    tracer_comparaison_features(X_normalized, y)
    
    # 4. Split train/test
    split_idx = int(0.7 * len(X_normalized))
    X_train, X_test = X_normalized[:split_idx], X_normalized[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"\nREPARTITION:")
    print(f"  - Train: {len(X_train)} images")
    print(f"  - Test: {len(X_test)} images")
    
    # 5. Creer et entrainer PMC
    try:
        print(f"\nCREATION PMC ({X_train.shape[1]}, 6, 1)...")
        model = PMC(
            input_dim=X_train.shape[1], 
            hidden_dim=6, 
            learning_rate=0.02, 
            activation='tanh'
        )
        
        print("ENTRAINEMENT (1000 iterations)...")
        error = model.fit(X_train, y_train, max_iterations=1000)
        print(f"Erreur finale: {error:.4f}")
        
        # 6. Predictions
        y_train_pred = model.predict_class(X_train)
        y_test_pred = model.predict_class(X_test)
        
        train_acc = np.mean(y_train_pred == y_train) * 100
        test_acc = np.mean(y_test_pred == y_test) * 100
        
        print(f"\nRESULTATS:")
        print(f"  - Train accuracy: {train_acc:.1f}%")
        print(f"  - Test accuracy: {test_acc:.1f}%")
        print(f"  - Corrects train: {np.sum(y_train_pred == y_train)}/{len(y_train)}")
        print(f"  - Corrects test: {np.sum(y_test_pred == y_test)}/{len(y_test)}")
        
        # 7. Tracer les courbes
        print("\nVISUALISATION des resultats:")
        
        # Courbe 1: Histogramme des predictions
        print("1. Histogramme des predictions...")
        tracer_histogrammes_predictions(
            np.concatenate([y_train, y_test]),
            np.concatenate([y_train_pred, y_test_pred]),
            "Predictions PMC: Guitare (1) vs Piano (-1)"
        )
        
        # Courbe 2: Frontiere de decision (si assez de features)
        print("2. Frontiere de decision...")
        tracer_courbes_decision(
            X_train, 
            y_train, 
            model, 
            "Frontiere de decision PMC (donnees d'entrainement)"
        )
        
        # Courbe 3: Comparaison features principales
        print("3. Comparaison des features...")
        if X_train.shape[1] >= 2:
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            indices_correct = y_train_pred == y_train
            indices_incorrect = y_train_pred != y_train
            
            plt.scatter(X_train[indices_correct, 0], X_train[indices_correct, 1], 
                       color='green', alpha=0.7, s=80, label='Correct')
            plt.scatter(X_train[indices_incorrect, 0], X_train[indices_incorrect, 1], 
                       color='red', alpha=0.7, s=120, label='Incorrect', marker='x')
            
            plt.xlabel('Feature 1')
            plt.ylabel('Feature 2')
            plt.title('Train: Corrects vs Incorrects')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.subplot(1, 2, 2)
            plt.bar(['Train', 'Test'], [train_acc, test_acc], 
                   color=['blue', 'orange'], alpha=0.7)
            plt.ylim([0, 110])
            plt.ylabel('Accuracy (%)')
            plt.title('Performance Train vs Test')
            plt.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            plt.show()
        
        return model, train_acc, test_acc
        
    except Exception as e:
        print(f"\nERREUR: {e}")
        import traceback
        traceback.print_exc()
        return None, 0, 0


# ==================== MAIN ====================

if __name__ == "__main__":
    print("DEMARRAGE DES TESTS AVEC VISUALISATIONS...")
    
    try:
        dataset_path = Path("dataset")
        if not dataset_path.exists():
            print("\nERREUR: Dossier 'dataset' manquant!")
            print("Structure: dataset/guitare/, dataset/piano/")
            exit(1)
        
        # Test 1: PMC avec courbes
        print("\n" + "="*60)
        model, train_acc, test_acc = test_pmc_avec_courbes()
           
    except Exception as e:
        print(f"\nERREUR: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print("\n" + "="*80)
        print("FIN DES TESTS AVEC VISUALISATIONS")

        print("="*80)


# Les plots indiqueront  :
# où se situent les guitares
# où se situent les pianos
# où les classes se chevauchent
# les zones typiques

# La frontière de décision indiquera :
# où le modèle coupe l’espace
# quelles zones correspondent à chaque classe

# L’histogramme montrera :
# les erreurs fléchées et annotées

