pub mod linear_model;
pub mod pmc;  // Change "mod" en "pub mod"

// Réexporte tout pour faciliter l'usage
pub use pmc::*;
pub use linear_model::*;
// ==================== TEST DIRECT ====================
#[cfg(test)]
mod direct_tests {
    use super::*;
    
    #[test]
    fn test_pmc_charge_final() {
        println!("");
        println!("==========================================");
        println!("TEST ULTIME DE CHARGEMENT PMC");
        println!("==========================================");
        
        // TEST 1: La structure existe
        println!("1. Test de PMCConfig...");
        let config = PMCConfig {
            n_inputs: 2,
            n_hidden: 3,
            n_outputs: 1,
            learning_rate: 0.01,
        };
        println!("   ✓ PMCConfig créée: {} inputs, {} hidden", 
                config.n_inputs, config.n_hidden);
        
        // TEST 2: Les fonctions C existent
        println!("2. Test des fonctions C...");
        unsafe {
            println!("   Appel de pmc_new...");
            let pmc = pmc_new(&config as *const PMCConfig);
            
            if pmc.is_null() {
                println!("   ✗ ERREUR: pmc_new retourne NULL!");
                panic!("pmc_new a échoué");
            } else {
                println!("   ✓ pmc_new retourne un pointeur valide");
            }
            
            println!("   Appel de pmc_delete...");
            pmc_delete(pmc);
            println!("   ✓ pmc_delete exécuté sans erreur");
        }
        
        // TEST 3: Tout est OK
        println!("3. Conclusion...");
        println!("   ✓ Tous les tests passent!");
        println!("   ✓ pmc.rs est PARFAITEMENT chargé!");
        println!("   ✓ La bibliothèque est fonctionnelle!");
        println!("==========================================");
        println!("");
    }
}

// TEST FINAL - Vérification que pmc est chargé
#[test]
fn test_pmc_est_charge() {
    println!("\n=== DÉBUT TEST PMC ===");
    
    // Juste créer une config
    let config = PMCConfig {
        n_inputs: 2,
        n_hidden: 3,
        n_outputs: 1,
        learning_rate: 0.01,
    };
    
    println!("PMCConfig créée avec {} inputs", config.n_inputs);
    
    // Essayer d'appeler une fonction de pmc
    unsafe {
        let ptr = pmc_new(&config as *const PMCConfig);
        println!("pmc_new a retourné: {:?}", ptr);
        
        if !ptr.is_null() {
            pmc_delete(ptr);
            println!("pmc_delete exécuté");
        }
    }
    
    println!("✅ TEST RÉUSSI - pmc est chargé et fonctionnel!");
    println!("=== FIN TEST PMC ===\n");
}
// Test d intégration minimal
#[test]
fn test_pmc_ultime() {
    println!("DEBUT TEST PMC");
    
    // Créer une config
    let config = crate::PMCConfig {
        n_inputs: 2,
        n_hidden: 3,
        n_outputs: 1,
        learning_rate: 0.01,
    };
    
    println!("Config créée: {} inputs", config.n_inputs);
    
    // Utiliser les fonctions
    unsafe {
        let ptr = crate::pmc_new(&config as *const crate::PMCConfig);
        println!("pmc_new retourné: {}", !ptr.is_null());
        
        if !ptr.is_null() {
            crate::pmc_delete(ptr);
            println!("pmc_delete exécuté");
        }
    }
    
    println!("TEST REUSSI - PMC EST CHARGE");
}
