#[repr(C)]
pub struct LinearModel {
    weights: Vec<f64>,  // Importance de chaque feature (ex: si weight[0] = 2, la feature 0 est 2x plus importante) .Vecteur des poids (w1, w2, ..., wn).le poids À quel point une feature est importante pour prendre la décision.prédiction = (poids1 × feature1) + (poids2 × feature2) + ... + biais
    bias: f64, //Décalage de la ligne de décision(Le biais permet à la ligne de séparation de ne pas être obligée de passer par l'origine)
    learning_rate: f64, //cest la vitesse d'apprentissage(À quel pas le modèle avance quand il corrige ses erreurs)
}

#[no_mangle] // Ne change pas le nom de ma fonction 
pub extern "C" fn linear_model_new(input_dim: usize, learning_rate: f64) -> *mut LinearModel { // Empêche Rust de "brouiller" le nom de la fonction ;// Sans extern "C" : nom "brouillé";// Crée une NOUVELLE instance de LinearModel// Retourne un POINTEUR vers le modèle (pour que Python puisse l'utiliser)
    let model = LinearModel {
        weights: vec![0.0; input_dim],  // Crée un vecteur de 'input_dim' zéros// Exemple : si input_dim = 2 → weights = [0.0, 0.0]// Le modèle commence avec tous les poids à zéro
        bias: 0.0,
        learning_rate, // Utilise la valeur passée en paramètre
    };
    Box::into_raw(Box::new(model)) //cest pour que le modèle survive après la fin de la fonction
}

#[no_mangle]
pub extern "C" fn linear_model_delete(model: *mut LinearModel) {// "model" = pointeur vers le modèle à supprimer// "*mut LinearModel" = pointeur mutable (qu'on peut modifier/supprimer)
    if !model.is_null() {// Vérifie que le pointeur n'est pas NULL// Évite de tenter de libérer un pointeur vide
        unsafe { 
            let _ = Box::from_raw(model); } //Rust, reprends cette mémoire que j'avais prêtée à Python, et libère-la proprement
    }
}

#[no_mangle]
pub extern "C" fn linear_model_fit(
    model: *mut LinearModel,        // Le modèle à entraîner
    features: *const f64,           // Tableau des features (entrées)
    targets: *const f64,            // Tableau des réponses attendues
    n_samples: usize,               // Nombre d'exemples
    n_features: usize,              // Nombre de features par exemple  
    max_iterations: usize,          // Nombre max d'itérations
) -> f64 {
    unsafe {
        let model = &mut *model;// "*model" = Déréférence le pointeur → accède au modèle// "&mut" = Crée une référence MUTABLE (qu'on peut modifier)// Résultat : on peut maintenant utiliser "model" normalement;*model = "Ouvre la boîte à l'adresse donnée"&mut = "Donne-moi la permission de modifier ce qu'il y a dedans"
        
        // Conversion directe sans Array2 inutile
        let features_slice = std::slice::from_raw_parts(features, n_samples * n_features); // Prend un pointeur C brut et le convertit en slice Rust// "features" = adresse mémoire du début du tableau// "n_samples * n_features" = nombre total d'éléments
        let targets_slice = std::slice::from_raw_parts(targets, n_samples);// Même principe pour les targets (réponses)// "targets" = adresse mémoire des réponses// "n_samples" = nombre de réponses
        
        let mut previous_error = f64::MAX; // Initialise l'erreur précédente à la valeur MAXIMUM possible
        
        for iteration in 0..max_iterations {// Répète l'apprentissage jusqu'à convergence// "iteration" = numéro de l'itération actuelle (0, 1, 2, ...)// "max_iterations" = sécurité pour ne pas boucler infiniment
            let mut total_error = 0.0;
            
            for i in 0..n_samples {// Parcourt TOUS les exemples d'entraînement// "i" = index de l'exemple actuel// "n_samples" = nombre total d'exemples
                // Calcul de la prédiction directement
                let mut prediction = model.bias;
                for j in 0..n_features { // Parcourt TOUTES les features d'un exemple//"j" = index de la feature actuelle  // "n_features" = nombre de features par exemple
                    prediction += model.weights[j] * features_slice[i * n_features + j];//poids fois feature
                }
                
                let error = prediction - targets_slice[i]; //erreur = prédiction - vérité
                total_error += error * error; // Erreur au carré pour pénaliser les grosses erreurs
                
                // Mise à jour directe des poids
                for j in 0..n_features {
                    model.weights[j] -= model.learning_rate * error * features_slice[i * n_features + j];//nouveau_poids = ancien_poids - η × erreur × feature
                }
                model.bias -= model.learning_rate * error;//nouveau_biais = ancien_biais - η × erreur
            }
            
            let mean_error = total_error / n_samples as f64;//erreur moyenne par exemple
            
            // Condition d'arrêt simple
            if iteration > 0 && (previous_error - mean_error).abs() < 1e-6 {
                return mean_error;
            }
            previous_error = mean_error;
        }
        
        previous_error
    }
}

#[no_mangle]
pub extern "C" fn linear_model_predict_batch(
    model: *const LinearModel,
    features: *const f64,
    results: *mut f64,
    n_samples: usize,
    n_features: usize,
) {
    unsafe {
        let model = &*model;
        let features_slice = std::slice::from_raw_parts(features, n_samples * n_features);
        let results_slice = std::slice::from_raw_parts_mut(results, n_samples);
        
        for i in 0..n_samples {
            let mut prediction = model.bias;
            for j in 0..n_features {
                prediction += model.weights[j] * features_slice[i * n_features + j];
            }
            results_slice[i] = prediction;
        }
    }
}

// Seulement 2 fonctions essentielles pour les paramètres
#[no_mangle]
pub extern "C" fn linear_model_get_weights(model: *const LinearModel, weights: *mut f64) {
    unsafe {
        let model = &*model;
        let weights_slice = std::slice::from_raw_parts_mut(weights, model.weights.len());
        weights_slice.copy_from_slice(&model.weights);
    }
}

#[no_mangle]
pub extern "C" fn linear_model_get_bias(model: *const LinearModel) -> f64 {
    unsafe { (*model).bias }
}