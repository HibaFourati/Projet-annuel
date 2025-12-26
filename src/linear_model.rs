use rand::Rng;
use std::f64;

#[repr(C)]
pub struct LinearModel {
    weights: Vec<f64>,
    bias: f64,
    learning_rate: f64,
}

#[no_mangle]
pub extern "C" fn linear_model_new(input_dim: usize, learning_rate: f64) -> *mut LinearModel {
    let mut rng = rand::thread_rng();
    
    // INITIALISATION ALÉATOIRE CRITIQUE (Xavier/Glorot)
    let mut weights = Vec::with_capacity(input_dim);
    let scale = (2.0 / (input_dim as f64 + 1.0)).sqrt();
    
    for _ in 0..input_dim {
        weights.push(rng.gen_range(-scale..scale));
    }
    
    let model = LinearModel {
        weights,
        bias: 0.0,
        learning_rate,
    };
    
    Box::into_raw(Box::new(model))
}

#[no_mangle]
pub extern "C" fn linear_model_delete(model: *mut LinearModel) {
    if !model.is_null() {
        unsafe {
            let _ = Box::from_raw(model);
        }
    }
}

#[no_mangle]
pub extern "C" fn linear_model_fit(
    model: *mut LinearModel,
    features: *const f64,
    targets: *const f64,
    n_samples: usize,
    n_features: usize,
    max_iterations: usize,
) -> f64 {
    unsafe {
        let model = &mut *model;
        let features_slice = std::slice::from_raw_parts(features, n_samples * n_features);
        let targets_slice = std::slice::from_raw_parts(targets, n_samples);
        
        let mut best_error = f64::MAX;
        let mut patience = 20;
        
        for iteration in 0..max_iterations {
            let mut total_error = 0.0;
            let mut grad_weights = vec![0.0; n_features];
            let mut grad_bias = 0.0;
            
            // BATCH GRADIENT DESCENT avec régularisation L2
            for i in 0..n_samples {
                let mut prediction = model.bias;
                for j in 0..n_features {
                    prediction += model.weights[j] * features_slice[i * n_features + j];
                }
                
                let error = prediction - targets_slice[i];
                total_error += error * error;
                
                // Gradient avec momentum virtuel
                for j in 0..n_features {
                    grad_weights[j] += error * features_slice[i * n_features + j];
                }
                grad_bias += error;
            }
            
            // RÉGULARISATION L2 (évite l'overfitting)
            let lambda = 0.001;
            for j in 0..n_features {
                total_error += lambda * model.weights[j] * model.weights[j];
                grad_weights[j] += 2.0 * lambda * model.weights[j];
            }
            
            let mean_error = total_error / n_samples as f64;
            
            // MISE À JOUR DES POIDS avec learning rate adaptatif
            let lr = model.learning_rate / (1.0 + 0.001 * iteration as f64);  // Décroissance
            let n = n_samples as f64;
            
            for j in 0..n_features {
                model.weights[j] -= lr * (grad_weights[j] / n);
            }
            model.bias -= lr * (grad_bias / n);
            
            // EARLY STOPPING
            if mean_error < best_error {
                best_error = mean_error;
                patience = 20;
            } else {
                patience -= 1;
                if patience == 0 {
                    println!("[Rust] Early stopping at iteration {}", iteration);
                    return best_error;
                }
            }
            
            if iteration % 500 == 0 {
                println!("[Rust] Iter {}: Error = {:.6}, LR = {:.6}", 
                         iteration, mean_error, lr);
            }
            
            if mean_error < 1e-6 {
                println!("[Rust] Converged at iteration {}", iteration);
                return mean_error;
            }
        }
        
        best_error
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