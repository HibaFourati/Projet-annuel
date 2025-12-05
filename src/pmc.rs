// src/pmc.rs
#[repr(C)]
pub struct MLP {
    weights_input_hidden: Vec<f64>,
    weights_hidden_output: Vec<f64>,
    bias_hidden: Vec<f64>,
    bias_output: f64,
    input_dim: usize,
    hidden_dim: usize,
    learning_rate: f64,
    activation_function: ActivationFunction,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub enum ActivationFunction {
    Sigmoid,
    Tanh,
    Relu,
}

// Fonctions d'activation et leurs dérivées
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn sigmoid_derivative(x: f64) -> f64 {
    let s = sigmoid(x);
    s * (1.0 - s)
}

fn tanh(x: f64) -> f64 {
    x.tanh()
}

fn tanh_derivative(x: f64) -> f64 {
    1.0 - x.tanh().powi(2)
}

fn relu(x: f64) -> f64 {
    if x > 0.0 { x } else { 0.0 }
}

fn relu_derivative(x: f64) -> f64 {
    if x > 0.0 { 1.0 } else { 0.0 }
}

impl MLP {
    fn activate(&self, x: f64) -> f64 {
        match self.activation_function {
            ActivationFunction::Sigmoid => sigmoid(x),
            ActivationFunction::Tanh => tanh(x),
            ActivationFunction::Relu => relu(x),
        }
    }
    
    fn activate_derivative(&self, x: f64) -> f64 {
        match self.activation_function {
            ActivationFunction::Sigmoid => sigmoid_derivative(x),
            ActivationFunction::Tanh => tanh_derivative(x),
            ActivationFunction::Relu => relu_derivative(x),
        }
    }
}

#[no_mangle]
pub extern "C" fn mlp_new(
    input_dim: usize,
    hidden_dim: usize,
    learning_rate: f64,
    activation: ActivationFunction,
) -> *mut MLP {
    let total_input_hidden = input_dim * hidden_dim;
    let total_hidden_output = hidden_dim * 1; // Sortie unique
    
    // Initialisation avec des petites valeurs aléatoires
    use rand::Rng;
    let mut rng = rand::thread_rng();
    
    let weights_input_hidden: Vec<f64> = (0..total_input_hidden)
        .map(|_| rng.gen_range(-0.5..0.5))
        .collect();
        
    let weights_hidden_output: Vec<f64> = (0..total_hidden_output)
        .map(|_| rng.gen_range(-0.5..0.5))
        .collect();
    
    let bias_hidden: Vec<f64> = (0..hidden_dim)
        .map(|_| rng.gen_range(-0.5..0.5))
        .collect();
    
    let bias_output = rng.gen_range(-0.5..0.5);
    
    let model = MLP {
        weights_input_hidden,
        weights_hidden_output,
        bias_hidden,
        bias_output,
        input_dim,
        hidden_dim,
        learning_rate,
        activation_function: activation,
    };
    
    Box::into_raw(Box::new(model))
}

#[no_mangle]
pub extern "C" fn mlp_delete(model: *mut MLP) {
    if !model.is_null() {
        unsafe {
            let _ = Box::from_raw(model);
        }
    }
}

#[no_mangle]
pub extern "C" fn mlp_fit(
    model: *mut MLP,
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
        
        let mut previous_error = f64::MAX;
        
        for iteration in 0..max_iterations {
            let mut total_error = 0.0;
            
            // Forward et backward pour chaque échantillon
            for i in 0..n_samples {
                // === FORWARD PASS ===
                let input_start = i * n_features;
                
                // Couche cachée
                let mut hidden_inputs = vec![0.0; model.hidden_dim];
                let mut hidden_outputs = vec![0.0; model.hidden_dim];
                
                for h in 0..model.hidden_dim {
                    let mut sum = model.bias_hidden[h];
                    for j in 0..n_features {
                        let weight_idx = h * n_features + j;
                        sum += model.weights_input_hidden[weight_idx] * 
                               features_slice[input_start + j];
                    }
                    hidden_inputs[h] = sum;
                    hidden_outputs[h] = model.activate(sum);
                }
                
                // Couche de sortie
                let mut output_sum = model.bias_output;
                for h in 0..model.hidden_dim {
                    output_sum += model.weights_hidden_output[h] * hidden_outputs[h];
                }
                let prediction = model.activate(output_sum);
                
                // Calcul de l'erreur
                let target = targets_slice[i];
                let error = target - prediction;
                total_error += error * error;
                
                // === BACKWARD PASS ===
                // Dérivée pour la couche de sortie
                let output_delta = error * model.activate_derivative(output_sum);
                
                // Deltas pour la couche cachée
                let mut hidden_deltas = vec![0.0; model.hidden_dim];
                for h in 0..model.hidden_dim {
                    let hidden_error = output_delta * model.weights_hidden_output[h];
                    hidden_deltas[h] = hidden_error * model.activate_derivative(hidden_outputs[h]);
                }
                
                // === MISE À JOUR DES POIDS ===
                // Mise à jour weights_hidden_output
                for h in 0..model.hidden_dim {
                    model.weights_hidden_output[h] += model.learning_rate * 
                                                      output_delta * 
                                                      hidden_outputs[h];
                }
                model.bias_output += model.learning_rate * output_delta;
                
                // Mise à jour weights_input_hidden
                for h in 0..model.hidden_dim {
                    for j in 0..n_features {
                        let weight_idx = h * n_features + j;
                        model.weights_input_hidden[weight_idx] += model.learning_rate * 
                                                                   hidden_deltas[h] * 
                                                                   features_slice[input_start + j];
                    }
                    model.bias_hidden[h] += model.learning_rate * hidden_deltas[h];
                }
            }
            
            let mean_error = total_error / n_samples as f64;
            
            // Condition d'arrêt
            if iteration > 0 && (previous_error - mean_error).abs() < 1e-6 {
                return mean_error;
            }
            previous_error = mean_error;
        }
        
        previous_error
    }
}

#[no_mangle]
pub extern "C" fn mlp_predict_batch(
    model: *const MLP,
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
            let input_start = i * n_features;
            
            // Forward pass seulement
            let mut hidden_outputs = vec![0.0; model.hidden_dim];
            
            for h in 0..model.hidden_dim {
                let mut sum = model.bias_hidden[h];
                for j in 0..n_features {
                    let weight_idx = h * n_features + j;
                    sum += model.weights_input_hidden[weight_idx] * 
                           features_slice[input_start + j];
                }
                hidden_outputs[h] = model.activate(sum);
            }
            
            let mut output_sum = model.bias_output;
            for h in 0..model.hidden_dim {
                output_sum += model.weights_hidden_output[h] * hidden_outputs[h];
            }
            
            results_slice[i] = model.activate(output_sum);
        }
    }
}

#[no_mangle]
pub extern "C" fn mlp_predict_class_batch(
    model: *const MLP,
    features: *const f64,
    results: *mut f64,
    n_samples: usize,
    n_features: usize,
    threshold: f64,
) {
    unsafe {
        mlp_predict_batch(model, features, results, n_samples, n_features);
        let results_slice = std::slice::from_raw_parts_mut(results, n_samples);
        
        for i in 0..n_samples {
            results_slice[i] = if results_slice[i] >= threshold { 1.0 } else { -1.0 };
        }
    }
}