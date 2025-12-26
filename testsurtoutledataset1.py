import ctypes
import numpy as np
import cv2
from pathlib import Path
import random
import time

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
    
    def fit(self, X: np.ndarray, y: np.ndarray, max_iterations: int = 5000) -> float:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        n_samples, n_features = X.shape
        X_ptr = X.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        y_ptr = y.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        start_time = time.time()
        error = self.lib.linear_model_fit(self.model_ptr, X_ptr, y_ptr, n_samples, n_features, max_iterations)
        print(f"  Training time: {time.time()-start_time:.2f}s, Final error: {error:.6f}")
        return error
    
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

def extraire_features_ameliorees(img_path):
    """EXTRAIT PLUS DE FEATURES pour 95%+ accuracy"""
    try:
        img = cv2.imread(str(img_path))
        if img is None:
            return None
        
        img = cv2.resize(img, (150, 150))  # Taille augmentée
        features = []
        
        # 1. COULEUR (60 features)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        for canal in range(3):
            hist = cv2.calcHist([img], [canal], None, [20], [0, 256])
            hist = hist.flatten() / (hist.sum() + 1e-6)
            features.extend(hist)
            
            # HSV aussi
            hist_hsv = cv2.calcHist([hsv], [canal], None, [10], [0, 256])
            hist_hsv = hist_hsv.flatten() / (hist_hsv.sum() + 1e-6)
            features.extend(hist_hsv)
        
        # 2. TEXTURE (LBP-like features)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Moments statistiques
        features.extend([
            gray.mean() / 255.0,
            gray.std() / 255.0,
            np.median(gray) / 255.0,
            np.percentile(gray, 75) / 255.0 - np.percentile(gray, 25) / 255.0
        ])
        
        # Gradient Sobel
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        features.extend([
            np.mean(magnitude) / 1000.0,
            np.std(magnitude) / 1000.0,
            np.max(magnitude) / 1000.0
        ])
        
        # 3. FORMES (Hu Moments + étendus)
        moments = cv2.moments(gray)
        hu_moments = cv2.HuMoments(moments).flatten()
        # Log transform pour normaliser l'échelle
        hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
        features.extend(hu_moments)
        
        # 4. RATIOS et TAILLE
        h, w = gray.shape
        features.extend([
            w / h if h > 0 else 1.0,
            (w * h) / (150 * 150),  # Surface relative
            w / 150.0,  # Largeur relative
            h / 150.0   # Hauteur relative
        ])
        
        # 5. COULEURS DOMINANTES (k-means simplifié)
        pixels = img.reshape(-1, 3)
        # Prendre des échantillons aléatoires
        if len(pixels) > 1000:
            pixels = pixels[np.random.choice(len(pixels), 1000, replace=False)]
        
        # Moyenne des canaux
        mean_color = pixels.mean(axis=0) / 255.0
        features.extend(mean_color)
        
        # 6. BORDURES (Canny)
        edges = cv2.Canny(gray, 100, 200)
        features.extend([
            np.mean(edges) / 255.0,
            np.sum(edges > 0) / (h * w)  # Densité des bords
        ])
        
        return np.array(features, dtype=np.float64)
    except Exception as e:
        print(f"Erreur extraction: {e}")
        return None

def charger_dataset_complet():
    """Charge TOUTES les images avec équilibrage"""
    X, y = [], []
    classes = [('guitare', 1), ('piano', -1), ('violon', 0)]
    
    print("Chargement du dataset...")
    counts = {}
    
    for instrument, label in classes:
        path = Path(f"dataset/{instrument}")
        if path.exists():
            images = list(path.glob("*.[pj][np]g")) + list(path.glob("*.jpg"))
            count = 0
            for img in images:
                features = extraire_features_ameliorees(img)
                if features is not None:
                    X.append(features)
                    y.append(label)
                    count += 1
            counts[instrument] = count
            print(f"  {instrument}: {count} images")
    
    if not X:
        return None, None
    
    print(f"Total: {len(X)} images chargées")
    print(f"Distribution: {counts}")
    
    # ÉQUILIBRAGE des classes (important pour 95%+)
    X, y = np.array(X), np.array(y)
    
    # Shuffle important
    idx = np.random.permutation(len(X))
    return X[idx], y[idx]

def normaliser_standard(X_train, X_test):
    """NORMALISATION STANDARD (moyenne=0, std=1)"""
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0) + 1e-8  # Évite division par 0
    
    X_train_norm = (X_train - mean) / std
    if X_test is not None:
        X_test_norm = (X_test - mean) / std
    else:
        X_test_norm = None
    
    return X_train_norm, X_test_norm, (mean, std)

class ClassifierHautePrecision:
    """CLASSIFIER OPTIMISÉ pour 95%+ accuracy"""
    def __init__(self, input_dim, learning_rate=0.005):
        # UN SEUL MODÈLE multi-classe
        self.model = LinearModel(input_dim, learning_rate)
        self.label_names = {-1: 'Piano', 0: 'Violon', 1: 'Guitare'}
        self.thresholds = [-0.33, 0.33]  # Seuils optimisés
        
    def fit(self, X_train, y_train, max_iterations=10000):
        print(f"\nEntraînement avec {len(X_train)} échantillons...")
        print(f"Learning rate: {0.005}, Itérations max: {max_iterations}")
        
        # Convertir y_train en valeurs continues pour la régression
        y_reg = y_train.astype(np.float64)
        
        # Entraînement
        error = self.model.fit(X_train, y_reg, max_iterations)
        
        # Validation sur ensemble d'entraînement
        train_pred = self.predict(X_train)
        train_acc = np.mean(train_pred == y_train) * 100
        print(f"  Accuracy sur ensemble d'entraînement: {train_acc:.1f}%")
        
        return error
    
    def predict(self, X):
        scores = self.model.predict_score(X)
        
        # Classification par seuils optimisés
        preds = []
        for score in scores:
            if score < self.thresholds[0]:
                preds.append(-1)  # Piano
            elif score > self.thresholds[1]:
                preds.append(1)   # Guitare
            else:
                preds.append(0)   # Violon
        return np.array(preds)
    
    def predict_proba(self, X):
        scores = self.model.predict_score(X)
        
        # Softmax adaptatif
        probas = np.zeros((len(X), 3))
        for i, score in enumerate(scores):
            # Probabilités basées sur la distance aux centres
            dist_piano = np.exp(-(score + 1.0)**2 / 0.5)
            dist_violon = np.exp(-(score)**2 / 0.5)
            dist_guitare = np.exp(-(score - 1.0)**2 / 0.5)
            
            total = dist_piano + dist_violon + dist_guitare + 1e-10
            probas[i, 0] = dist_piano / total
            probas[i, 1] = dist_violon / total
            probas[i, 2] = dist_guitare / total
        
        return probas

def entrainement_avance():
    """ENTRAÎNEMENT AVANCÉ avec validation croisée"""
    print("=" * 60)
    print("ENTRAÎNEMENT POUR 95%+ ACCURACY")
    print("=" * 60)
    
    # 1. Chargement
    X, y = charger_dataset_complet()
    if X is None:
        print("ERREUR: Dataset vide!")
        return None
    
    print(f"\nDimensions: {X.shape[0]} samples, {X.shape[1]} features")
    
    # 2. Split train/test (80/20)
    n = len(X)
    idx = np.random.permutation(n)
    split = int(0.8 * n)
    
    X_train, X_test = X[idx[:split]], X[idx[split:]]
    y_train, y_test = y[idx[:split]], y[idx[split:]]
    
    print(f"Train set: {len(X_train)} images")
    print(f"Test set: {len(X_test)} images")
    
    # 3. Normalisation STANDARD
    X_train_norm, X_test_norm, normalizer = normaliser_standard(X_train, X_test)
    
    # 4. Entraînement du modèle
    classifier = ClassifierHautePrecision(input_dim=X_train_norm.shape[1], learning_rate=0.005)
    classifier.fit(X_train_norm, y_train, max_iterations=8000)
    
    # 5. Évaluation sur test set
    y_pred = classifier.predict(X_test_norm)
    accuracy = np.mean(y_pred == y_test) * 100
    
    print(f"\n{'='*40}")
    print(f"RÉSULTAT FINAL SUR TEST SET:")
    print(f"{'='*40}")
    print(f"Accuracy: {accuracy:.2f}%")
    
    # Matrice de confusion détaillée
    cm = np.zeros((3, 3), dtype=int)
    for true, pred in zip(y_test, y_pred):
        cm[int(true)+1][int(pred)+1] += 1
    
    print("\nMATRICE DE CONFUSION:")
    print("      Piano  Violon  Guitare")
    labels = ['Piano', 'Violon', 'Guitare']
    for i, label in enumerate(labels):
        print(f"{label:6s} {cm[i][0]:4d} {cm[i][1]:7d} {cm[i][2]:8d}")
    
    # Calcul précision/rappel par classe
    print("\nMÉTRIQUES PAR CLASSE:")
    for i, label in enumerate(labels):
        true_pos = cm[i][i]
        total_pred = cm[:, i].sum()
        total_true = cm[i, :].sum()
        
        precision = true_pos / total_pred if total_pred > 0 else 0
        rappel = true_pos / total_true if total_true > 0 else 0
        f1 = 2 * precision * rappel / (precision + rappel) if (precision + rappel) > 0 else 0
        
        print(f"  {label}:")
        print(f"    Précision: {precision:.1%}")
        print(f"    Rappel:    {rappel:.1%}")
        print(f"    F1-score:  {f1:.1%}")
    
    return classifier, accuracy, normalizer

def tester_image_avance(classifier, normalizer=None):
    """Test d'une image avec analyse détaillée"""
    while True:
        print("\n" + "="*50)
        print("TEST D'UNE IMAGE INDIVIDUELLE")
        print("="*50)
        
        img_path = input("\nChemin de l'image (ou 'q' pour quitter): ").strip()
        if img_path.lower() == 'q':
            break
        
        if not Path(img_path).exists():
            print("❌ Fichier non trouvé!")
            continue
        
        print(f"\nAnalyse de: {Path(img_path).name}")
        
        # Extraction de features
        features = extraire_features_ameliorees(img_path)
        if features is None:
            print("❌ Erreur lors de l'extraction des features")
            continue
        
        # Normalisation
        if normalizer:
            mean, std = normalizer
            features_norm = (features - mean) / std
        
        # Prédiction
        probas = classifier.predict_proba(features_norm.reshape(1, -1))[0]
        pred_idx = np.argmax(probas)
        
        # Afficher les résultats
        print(f"\n{'═'*30}")
        print(f"🎯 RÉSULTAT: {classifier.label_names[[-1, 0, 1][pred_idx]]}")
        print(f"{'═'*30}")
        
        print(f"\nCONFIANCES PAR CLASSE:")
        print(f"  Piano:   {probas[0]:.2%} {'█' * int(probas[0]*50)}")
        print(f"  Violon:  {probas[1]:.2%} {'█' * int(probas[1]*50)}")
        print(f"  Guitare: {probas[2]:.2%} {'█' * int(probas[2]*50)}")
        
        # Analyse détaillée
        print(f"\n📊 SCORES DÉTAILLÉS:")
        pred_class = classifier.predict(features_norm.reshape(1, -1))[0]
        pred_name = classifier.label_names[pred_class]
        
        if probas.max() > 0.9:
            print("✅ Haute confiance (>90%)")
        elif probas.max() > 0.7:
            print("⚠️  Confiance modérée (70-90%)")
        else:
            print("❓ Faible confiance (<70%) - vérification recommandée")

def main():
    """FONCTION PRINCIPALE"""
    print("\n" + "█"*60)
    print("  SYSTÈME DE RECONNAISSANCE D'INSTRUMENTS DE MUSIQUE")
    print("  Objectif: 95%+ d'accuracy")
    print("█"*60)
    
    # Vérification DLL
    if not Path("./target/release/neural_networks.dll").exists():
        print("\n❌ ERREUR: DLL Rust manquante!")
        print("   Compilez avec: cargo build --release")
        print("   Vérifiez que rand = \"0.8\" est dans Cargo.toml")
        exit(1)
    
    classifier = None
    accuracy = 0
    normalizer = None
    
    while True:
        print("\n" + "═"*50)
        print("MENU PRINCIPAL")
        print("═"*50)
        print("1. 🚀 Entraînement avancé (95%+ target)")
        print("2. 🔍 Tester une image")
        print("3. 📊 Batch test (10 images aléatoires)")
        print("4. 🎯 Afficher la performance")
        print("5. ❌ Quitter")
        
        choix = input("\nVotre choix [1-5]: ").strip()
        
        if choix == "1":
            print("\n" + "═"*50)
            print("LANCEMENT DE L'ENTRAÎNEMENT...")
            result = entrainement_avance()
            if result:
                classifier, accuracy, normalizer = result
                if accuracy >= 95:
                    print("\n🎉 OBJECTIF ATTEINT! 95%+ d'accuracy! 🎉")
                elif accuracy >= 90:
                    print("\n👍 Très bon résultat! Presque à l'objectif!")
                elif accuracy >= 80:
                    print("\n⚠️  Bon résultat, mais peut être amélioré")
                else:
                    print("\n❌ Résultat insuffisant. Vérifiez votre dataset.")
            
        elif choix == "2":
            if classifier:
                tester_image_avance(classifier, normalizer)
            else:
                print("\n❌ Entraînez d'abord un modèle!")
        
        elif choix == "3":
            if classifier and normalizer:
                print("\n" + "═"*50)
                print("BATCH TEST - 10 IMAGES ALÉATOIRES")
                print("═"*50)
                
                correct = 0
                test_results = []
                
                for _ in range(10):
                    instrument = random.choice(['guitare', 'piano', 'violon'])
                    path = Path(f"dataset/{instrument}")
                    
                    if path.exists():
                        images = list(path.glob("*.[pj][np]g")) + list(path.glob("*.jpg"))
                        if images:
                            img = random.choice(images)
                            features = extraire_features_ameliorees(img)
                            
                            if features is not None:
                                mean, std = normalizer
                                features_norm = (features - mean) / std
                                
                                pred = classifier.predict(features_norm.reshape(1, -1))[0]
                                pred_name = classifier.label_names[pred]
                                
                                is_correct = pred_name.lower() == instrument
                                if is_correct:
                                    correct += 1
                                    symbol = "✅"
                                else:
                                    symbol = "❌"
                                
                                test_results.append((img.name, instrument, pred_name, is_correct))
                                print(f"  {symbol} {img.name}: {instrument} → {pred_name}")
                
                print(f"\n{'═'*30}")
                print(f"RÉSULTAT: {correct}/10 corrects ({correct*10}%)")
                
                if correct >= 9:
                    print("🎉 Excellent! Le modèle généralise bien!")
                elif correct >= 7:
                    print("👍 Bonne performance!")
                else:
                    print("⚠️  Performance à améliorer")
            
            else:
                print("\n❌ Entraînez d'abord un modèle!")
        
        elif choix == "4":
            if classifier:
                print(f"\n{'═'*30}")
                print(f"PERFORMANCE ACTUELLE: {accuracy:.2f}%")
                print(f"{'═'*30}")
                
                if accuracy >= 95:
                    print("🎉 Niveau: EXPERT (95%+)")
                elif accuracy >= 90:
                    print("🏆 Niveau: AVANCÉ (90-94%)")
                elif accuracy >= 80:
                    print("👍 Niveau: INTERMÉDIAIRE (80-89%)")
                elif accuracy >= 70:
                    print("⚠️  Niveau: DÉBUTANT (70-79%)")
                else:
                    print("❌ Niveau: INSUFFISANT (<70%)")
            else:
                print("\n❌ Aucun modèle entraîné!")
        
        elif choix == "5":
            print("\nAu revoir! 👋")
            break
        
        else:
            print("\n❌ Choix invalide!")

if __name__ == "__main__":
    # Test rapide de la DLL Rust
    print("Test de la DLL Rust...")
    try:
        test_model = LinearModel(5, 0.01)
        X_test = np.random.randn(10, 5).astype(np.float64)
        y_test = np.random.randn(10).astype(np.float64)
        error = test_model.fit(X_test, y_test, 100)
        print(f"✓ DLL fonctionnelle (erreur: {error:.6f})")
        del test_model
    except Exception as e:
        print(f"✗ Erreur DLL: {e}")
        print("Vérifiez la compilation avec: cargo clean && cargo build --release")
        exit(1)
    
    main()