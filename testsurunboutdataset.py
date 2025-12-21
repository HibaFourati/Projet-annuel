import ctypes
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from collections import Counter

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

def charger_dataset(n_images_per_class=6):
    X, y = [], []
    classes = [('guitare', 1), ('piano', -1), ('violon', 0)]
    
    for instrument, label in classes:
        path = Path(f"dataset/{instrument}")
        if path.exists():
            images = list(path.glob("*.[pj][np]g"))
            if len(images) > n_images_per_class:
                images = images[:n_images_per_class]
            print(f"  {instrument}: {len(images)} images")
            for img in images:
                features = extraire_features(img)
                if features is not None:
                    X.append(features)
                    y.append(label)
    
    if len(X) == 0:
        print("Aucune image trouvée")
        return None, None
    
    print(f"\nChargement total: {len(X)} échantillons")
    class_counts = Counter(y)
    for label, count in sorted(class_counts.items()):
        instrument = {1: 'Guitare', -1: 'Piano', 0: 'Violon'}[label]
        print(f"  {instrument} ({label}): {count} échantillons")
    
    return np.array(X), np.array(y)

def split_manuel(X, y, test_size=0.3):
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

def calculer_matrice_confusion(y_true, y_pred, classes):
    n_classes = len(classes)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    
    label_to_index = {label: i for i, label in enumerate(classes)}
    
    for true, pred in zip(y_true, y_pred):
        true_idx = label_to_index[true]
        pred_idx = label_to_index[pred]
        cm[true_idx][pred_idx] += 1
    
    return cm

def calculer_metrics_par_classe(cm, classes):
    n_classes = len(classes)
    precision = []
    recall = []
    f1 = []
    
    for i in range(n_classes):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
        
        precision.append(prec)
        recall.append(rec)
        f1.append(f1_score)
    
    return precision, recall, f1

def afficher_matrice_confusion(cm, classes, accuracy):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    im = ax.imshow(cm, cmap='Blues')
    
    for i in range(len(classes)):
        for j in range(len(classes)):
            color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
            ax.text(j, i, str(cm[i, j]),
                   ha='center', va='center',
                   color=color, fontweight='bold', fontsize=12)
    
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.set_xlabel('Prédite classe', fontsize=12)
    ax.set_ylabel('Vraie classe', fontsize=12)
    ax.set_title(f'Matrice de Confusion - 3 Classes\nAccuracy: {accuracy:.1f}%', 
                fontsize=14, fontweight='bold')
    
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.show()

class MultiClassClassifier:
    def __init__(self, input_dim, learning_rate=0.01):
        self.models = {
            'guitare': LinearModel(input_dim, learning_rate),
            'piano': LinearModel(input_dim, learning_rate),
            'violon': LinearModel(input_dim, learning_rate)
        }
        self.class_labels = {'guitare': 1, 'piano': -1, 'violon': 0}
        self.label_names = {1: 'Guitare', -1: 'Piano', 0: 'Violon'}
        self.input_dim = input_dim
        
    def fit(self, X_train, y_train, max_iterations=500):
        print("\nEntraînement des modèles one-vs-all")
        
        for class_name, model in self.models.items():
            label = self.class_labels[class_name]
            y_binary = np.where(y_train == label, 1, -1)
            print(f"  {class_name.capitalize()} vs autres")
            model.fit(X_train, y_binary, max_iterations)
            
    def predict(self, X, method='score'):
        if method == 'vote':
            return self._predict_vote(X)
        elif method == 'score':
            return self._predict_score(X)
        else:
            return self._predict_score(X)
    
    def _predict_vote(self, X):
        predictions = []
        
        for class_name, model in self.models.items():
            pred = model.predict_class(X)
            predictions.append((class_name, pred))
        
        final_preds = []
        n_samples = len(X)
        
        for i in range(n_samples):
            votes = []
            for class_name, pred_array in predictions:
                if pred_array[i] == 1:
                    votes.append(class_name)
            
            if len(votes) == 1:
                final_class = votes[0]
            elif len(votes) > 1:
                final_class = votes[0]
            else:
                final_class = 'guitare'
            
            final_preds.append(self.class_labels[final_class])
        
        return np.array(final_preds)
    
    def _predict_score(self, X):
        scores = {}
        
        for class_name, model in self.models.items():
            scores[class_name] = model.predict_score(X)
        
        n_samples = len(X)
        final_preds = []
        
        for i in range(n_samples):
            best_score = -np.inf
            best_class = 'guitare'
            
            for class_name, score_array in scores.items():
                if score_array[i] > best_score:
                    best_score = score_array[i]
                    best_class = class_name
            
            final_preds.append(self.class_labels[best_class])
        
        return np.array(final_preds)

def test_complet_3classes():
    
    X, y = charger_dataset(n_images_per_class=6)
    if X is None:
        return
    
    X_train, X_test, y_train, y_test = split_manuel(X, y, test_size=0.3)
    
    X_train_norm, X_test_norm = normaliser_manuel(X_train, X_test)
    
    print(f"\nRépartition train/test")
    print(f"  Train: {len(X_train_norm)} échantillons")
    print(f"  Test: {len(X_test_norm)} échantillons")
    
    classifier = MultiClassClassifier(input_dim=X_train_norm.shape[1], learning_rate=0.01)
    classifier.fit(X_train_norm, y_train, max_iterations=500)
    
    pred_test = classifier.predict(X_test_norm, method='score')
    
    accuracy = np.mean(pred_test == y_test) * 100
    print(f"\nAccuracy globale: {accuracy:.1f}%")
    
    classes = [-1, 0, 1]
    cm = calculer_matrice_confusion(y_test, pred_test, classes)
    
    afficher_matrice_confusion(cm, ['Piano', 'Violon', 'Guitare'], accuracy)
    
    precision, recall, f1 = calculer_metrics_par_classe(cm, classes)
    
    print("\nMétriques :")
    print("-"*40)
    for i, class_label in enumerate(classes):
        class_name = classifier.label_names[class_label]
        print(f"\n{class_name}:")
        print(f"  Précision: {precision[i]:.2%}")
        print(f"  Rappel: {recall[i]:.2%}")
        print(f"  F1-score: {f1[i]:.2%}")
    
    
    error_indices = np.where(pred_test != y_test)[0]
    if len(error_indices) > 0:
        print(f"Nombre d'erreurs: {len(error_indices)}/{len(y_test)}")
        
        for idx in error_indices:
            true_label = y_test[idx]
            pred_label = pred_test[idx]
            true_name = classifier.label_names[true_label]
            pred_name = classifier.label_names[pred_label]
            print(f"  Échantillon {idx}: Vrai={true_name}, Prédit={pred_name}")
    else:
        print("Aucune erreur !")
    
    print("\n" + "="*60)
    print("SUGGESTIONS D'AMÉLIORATION")
    print("="*60)
    
    if accuracy < 80:
        print("\n1. Augmenter les données d'entraînement")
        print("   - Plus d'images par classe")
        print("   - Équilibrer les classes si nécessaire")
        
        print("\n2. Ajuster les hyperparamètres")
        print("   - Essayer learning_rate = 0.001 ou 0.005")
        print("   - Augmenter max_iterations à 1000")
        
        print("\n3. Améliorer les caractéristiques")
        print("   - Ajouter plus de descripteurs d'image")
        print("   - Normalisation différente")
    else:
        print("\nPerformance satisfaisante !")
        print("   - Tester avec plus de données")

    
    return classifier, X_train_norm, X_test_norm, y_train, y_test

def test_prediction_interactive(classifier, X_mean, X_std):

    while True:
        print("\nOptions:")
        print("1. Tester avec une image du dataset")
        print("2. Tester avec une image personnelle")
        print("3. Quitter")
        
        choix = input("\nVotre choix (1-3): ").strip()
        
        if choix == "1":
            print("\nClasses disponibles:")
            print("1. Guitare")
            print("2. Piano")
            print("3. Violon")
            
            classe_choix = input("\nChoisir une classe (1-3): ").strip()
            classes_dict = {'1': 'guitare', '2': 'piano', '3': 'violon'}
            
            if classe_choix in classes_dict:
                instrument = classes_dict[classe_choix]
                path = Path(f"dataset/{instrument}")
                if path.exists():
                    images = list(path.glob("*.[pj][np]g"))
                    if images:
                        test_img = images[0]
                        print(f"\nTest sur: {test_img.name}")
                        
                        features = extraire_features(test_img)
                        if features is not None:
                            features_norm = (features - X_mean) / X_std
                            
                            pred = classifier.predict(features_norm.reshape(1, -1))
                            pred_label = pred[0]
                            pred_name = classifier.label_names[pred_label]
                            true_name = instrument.capitalize()
                            
                            print(f"  Vrai: {true_name}")
                            print(f"  Prédit: {pred_name}")
                            print(f"  {'CORRECT' if pred_name.lower() == instrument else 'INCORRECT'}")
                            
                            img = cv2.imread(str(test_img))
                            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            
                            plt.figure(figsize=(6, 5))
                            plt.imshow(img_rgb)
                            plt.title(f"Image: {test_img.name}\nVrai: {true_name} | Prédit: {pred_name}", 
                                    fontsize=12)
                            plt.axis('off')
                            plt.tight_layout()
                            plt.show()
        
        elif choix == "2":
            img_path = input("\nChemin de l'image: ").strip()
            if Path(img_path).exists():
                features = extraire_features(img_path)
                if features is not None:
                    features_norm = (features - X_mean) / X_std
                    
                    pred = classifier.predict(features_norm.reshape(1, -1))
                    pred_label = pred[0]
                    pred_name = classifier.label_names[pred_label]
                    
                    print(f"\nPrediction: {pred_name}")
                    
                    img = cv2.imread(img_path)
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    plt.figure(figsize=(6, 5))
                    plt.imshow(img_rgb)
                    plt.title(f"Prediction: {pred_name}", fontsize=14, fontweight='bold')
                    plt.axis('off')
                    plt.tight_layout()
                    plt.show()
                else:
                    print("Impossible d'extraire les caracteristiques de l'image")
            else:
                print("Fichier non trouve")
        
        elif choix == "3":
            print("Retour au menu principal.")
            break
        
        else:
            print("Choix invalide. Veuillez choisir 1, 2 ou 3.")

if __name__ == "__main__":
    print("="*60)
    print("SYSTEME DE CLASSIFICATION - 3 INSTRUMENTS")
    print("="*60)
    print("Instruments: Guitare, Piano, Violon")
    print("Approche: Modele Lineaire One-vs-All")
    print("="*60)

    dll_path = "./target/release/neural_networks.dll"
    if not Path(dll_path).exists():
        print(f"Erreur: {dll_path} introuvable")
        exit(1)
    
    result = test_complet_3classes()
    
    if result is not None:
        classifier, X_train_norm, X_test_norm, y_train, y_test = result
        
        X_train, _ = charger_dataset(n_images_per_class=6)
        X_mean = X_train.mean(axis=0)
        X_std = X_train.std(axis=0) + 1e-6
        
        print("\n" + "="*60)
        print("MENU INTERACTIF")
        print("="*60)
        
        test_prediction_interactive(classifier, X_mean, X_std)
