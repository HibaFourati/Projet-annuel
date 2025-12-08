#[repr(C)]
pub struct LinearModel {
    weights: Vec<f64>, 
    bias: f64, 
    learning_rate: f64, 
}

#[no_mangle] // Ne change pas le nom de ma fonction 
pub extern "C" fn linear_model_new(input_dim: usize, learning_rate: f64) -> *mut LinearModel { 
    let model = LinearModel {
        weights: vec![0.0; input_dim], 
        bias: 0.0,
        learning_rate, 
    };
    Box::into_raw(Box::new(model)) //cest pour que le modèle survive après la fin de la fonction
}

#[no_mangle]
pub extern "C" fn linear_model_delete(model: *mut LinearModel) {
    if !model.is_null() {
        unsafe { 
            let _ = Box::from_raw(model); } //Rust, reprends cette mémoire que j'avais prêtée à Python, et libère-la proprement
    }
}

#[no_mangle]
pub extern "C" fn linear_model_fit(
    model: *mut LinearModel,        // Le modèle à entraîner
    features: *const f64,           // Tableau des entrées
    targets: *const f64,            // Tableau des réponses attendues
    n_samples: usize,               // Nombre d'exemples
    n_features: usize,              // Nombre de features par exemple  
    max_iterations: usize,          // Nombre max d'itérations
) -> f64 {
    unsafe {
        let model = &mut *model;
        
        // Conversion directe sans Array2 inutile
        let features_slice = std::slice::from_raw_parts(features, n_samples * n_features);
        let targets_slice = std::slice::from_raw_parts(targets, n_samples);
        
        let mut previous_error = f64::MAX; // Initialise l'erreur précédente à la valeur MAXIMUM possible
        
        for iteration in 0..max_iterations {
            let mut total_error = 0.0;
            
            for i in 0..n_samples {
                // Calcul de la prédiction directement
                let mut prediction = model.bias;
                for j in 0..n_features { 
                    prediction += model.weights[j] * features_slice[i * n_features + j];//poids fois feature
                }
                
                let error = prediction - targets_slice[i]; //erreur = prédiction - vérité
                total_error += error * error; 
                
                // Mise à jour directe des poids
                for j in 0..n_features {
                    model.weights[j] -= model.learning_rate * error * features_slice[i * n_features + j];//nouveau_poids = ancien_poids - η × erreur × feature
                }
                model.bias -= model.learning_rate * error;//nouveau_biais = ancien_biais - η × erreur
            }
            
            let mean_error = total_error / n_samples as f64;
            
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
    n_features: usize, //est un type de données en Rust qui représente un entier non signé (positif)
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
            results_slice[i] = prediction;// Stocke le résultat
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