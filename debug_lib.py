# check_exports.py
import ctypes
import os

def check_dll_exports():
    print("🔍 VÉRIFICATION DES EXPORTS DLL")
    print("=" * 50)
    
    lib_path = "./target/release/linear_model.dll"
    
    if not os.path.exists(lib_path):
        print(f"❌ Fichier non trouvé: {lib_path}")
        return
    
    print(f"📁 Taille du fichier: {os.path.getsize(lib_path)} octets")
    
    try:
        # Charger la DLL
        lib = ctypes.CDLL(lib_path)
        print("✅ DLL chargée avec succès")
        
        # Liste des fonctions attendues
        expected_functions = [
            'linear_model_new',
            'linear_model_fit',
            'linear_model_predict_batch', 
            'linear_model_get_weights',
            'linear_model_get_bias',
            'linear_model_delete'
        ]
        
        print("\n🔍 RECHERCHE DES FONCTIONS:")
        found_functions = []
        
        for func_name in expected_functions:
            try:
                # Essayer d'accéder à la fonction
                func = getattr(lib, func_name)
                found_functions.append(func_name)
                print(f"  ✅ {func_name}")
            except AttributeError:
                print(f"  ❌ {func_name} - NON TROUVÉE")
        
        print(f"\n📊 Résultat: {len(found_functions)}/{len(expected_functions)} fonctions trouvées")
        
        if len(found_functions) == 0:
            print("\n🚨 PROBLEME: Aucune fonction exportée!")
            print("Vérifiez que votre code Rust compile correctement avec:")
            print("  cargo build --release --verbose")
            
    except Exception as e:
        print(f"❌ Erreur: {e}")

if __name__ == "__main__":
    check_dll_exports()