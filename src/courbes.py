import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression

def load_rust_results():
    with open('training_results.json', 'r') as f:
        return json.load(f)

def plot_convergence_auto(results):
    """TEST DE CONVERGENCE"""
    epochs = list(range(1, len(results['loss_history']) + 1))
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, results['loss_history'], 'b-', linewidth=2)
    plt.title('Courbe de Loss - Test Convergence')
    plt.xlabel('Époques')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, results['accuracy_history'], 'g-', linewidth=2)
    plt.title('Courbe d\'Accuracy - Test Convergence')
    plt.xlabel('Époques')
    plt.ylabel('Accuracy')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('convergence_rust.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(" convergence_rust.png - TEST CONVERGENCE")

def plot_confusion_matrix_auto(results):
    """MATRICE DE CONFUSION"""
    y_true = results['true_labels']
    y_pred = results['predictions']
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Matrice de Confusion - Performances Réelles')
    plt.colorbar()
    
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    
    classes = ['Guitare', 'Piano', 'Violon']
    plt.xticks(range(len(classes)), classes)
    plt.yticks(range(len(classes)), classes)
    plt.xlabel('Prédictions')
    plt.ylabel('Vraies valeurs')
    
    plt.tight_layout()
    plt.savefig('confusion_matrix_rust.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(" confusion_matrix_rust.png - TEST PERFORMANCES")

def plot_training_analysis(results):
    """ANALYSE DE L'ENTRAÎNEMENT (sans les poids)"""
    plt.figure(figsize=(12, 4))
    
    epochs = list(range(1, len(results['loss_history']) + 1))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, results['loss_history'], 'r-', linewidth=2, label='Loss')
    plt.title('Évolution de la Loss')
    plt.xlabel('Époques')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, results['accuracy_history'], 'g-', linewidth=2, label='Accuracy')
    plt.title('Évolution de l\'Accuracy')
    plt.xlabel('Époques')
    plt.ylabel('Accuracy')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(" training_analysis.png - ANALYSE ENTRAÎNEMENT")

def plot_xor_test():
    """TEST XOR - LIMITES DU MODÈLE LINÉAIRE"""
    np.random.seed(42)
    X_xor = np.random.randn(400, 2)
    y_xor = np.logical_xor(X_xor[:, 0] > 0, X_xor[:, 1] > 0).astype(int)
    
    model_linear = LogisticRegression()
    model_linear.fit(X_xor, y_xor)
    y_pred_linear = model_linear.predict(X_xor)
    accuracy_linear = np.mean(y_pred_linear == y_xor)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(X_xor[:, 0], X_xor[:, 1], c=y_xor, cmap='coolwarm', alpha=0.7, s=50)
    plt.title('Données XOR - Problème Non Linéaire')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.scatter(X_xor[:, 0], X_xor[:, 1], c=y_pred_linear, cmap='coolwarm', alpha=0.7, s=50)
    plt.title(f'Modèle Linéaire sur XOR\nAccuracy: {accuracy_linear:.2%} (ÉCHEC)')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('xor_test_limites.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(" xor_test_limites.png - TEST XOR & LIMITES")

def plot_non_linear_transformation():
    """TRANSFORMATION NON-LINÉAIRE"""
    np.random.seed(42)
    X = np.random.randn(300, 2)
    y = (X[:, 0]**2 + X[:, 1]**2 > 1).astype(int)
    
    X_transformed = np.column_stack([X[:, 0]**2, X[:, 1]**2])
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.7, s=50)
    plt.title('Données Originales\n(Non Linéairement Séparables)')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.scatter(X_transformed[:, 0], X_transformed[:, 1], c=y, cmap='viridis', alpha=0.7, s=50)
    plt.title('Après Transformation x², y²\n(Linéairement Séparables)')
    plt.xlabel('x1²')
    plt.ylabel('x2²')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    model = LogisticRegression()
    model.fit(X_transformed, y)
    
    xx, yy = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
    X_grid = np.column_stack([xx.ravel(), yy.ravel()])
    X_grid_transformed = np.column_stack([X_grid[:, 0]**2, X_grid[:, 1]**2])
    Z = model.predict(X_grid_transformed).reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.7, s=50)
    plt.title('Frontière après Transformation')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('non_linear_transformation.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(" non_linear_transformation.png - TEST TRANSFORMATION")

def main():
    print(" Chargement des résultats Rust...")
    results = load_rust_results()
    
    print("\n GÉNÉRATION DE TOUS LES TESTS...")
    
    # 1. Test de convergence
    plot_convergence_auto(results)
    
    # 2. Matrice de confusion
    plot_confusion_matrix_auto(results)
    
    # 3. Analyse entraînement (remplace decision_boundary)
    plot_training_analysis(results)
    
    # 4. Test XOR
    plot_xor_test()
    
    # 5. Transformation non-linéaire
    plot_non_linear_transformation()
    
    print("\n TOUS LES TESTS SONT PRÊTS !")
    print(" FICHIERS GÉNÉRÉS :")
    print("   1. convergence_rust.png          → Courbes de base")
    print("   2. confusion_matrix_rust.png     → Performances réelles") 
    print("   3. training_analysis.png         → Analyse détaillée")
    print("   4. xor_test_limites.png          → Test limites")
    print("   5. non_linear_transformation.png → Solutions complexes")

if __name__ == "__main__":
    main()