// pmc.rs
use rand::Rng;
use libc::{c_double, c_uint};

#[repr(C)]
pub struct PMCConfig {
    pub n_inputs: c_uint,
    pub n_hidden: c_uint,
    pub n_outputs: c_uint,
    pub learning_rate: c_double,
}

#[repr(C)]
pub struct PMC {
    w1: Vec<Vec<f64>>,  // weights input->hidden
    b1: Vec<f64>,       // biases hidden
    w2: Vec<Vec<f64>>,  // weights hidden->output
    b2: f64,            // bias output
    learning_rate: f64,
}

// Activation sigmoid seulement (suffisant pour XOR et Cross)
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn sigmoid_derivative(x: f64) -> f64 {
    let s = sigmoid(x);
    s * (1.0 - s)
}

#[no_mangle]
pub extern "C" fn pmc_new(config: *const PMCConfig) -> *mut PMC {
    unsafe {
        let config = &*config;
        let mut rng = rand::thread_rng();
        
        // Initialisation des poids avec Xavier simplifié
        let limit1 = (6.0_f64 / (config.n_inputs + config.n_hidden) as f64).sqrt();
        let w1: Vec<Vec<f64>> = (0..config.n_hidden)
            .map(|_| (0..config.n_inputs)
                .map(|_| rng.gen_range(-limit1..limit1))
                .collect())
            .collect();
        
        let limit2 = (6.0_f64 / (config.n_hidden + config.n_outputs) as f64).sqrt();
        let w2: Vec<Vec<f64>> = (0..config.n_outputs)
            .map(|_| (0..config.n_hidden)
                .map(|_| rng.gen_range(-limit2..limit2))
                .collect())
            .collect();
        
        let b1: Vec<f64> = (0..config.n_hidden)
            .map(|_| rng.gen_range(-0.5..0.5))
            .collect();
        
        let b2 = rng.gen_range(-0.5..0.5);
        
        let pmc = Box::new(PMC {
            w1,
            b1,
            w2,
            b2,
            learning_rate: config.learning_rate,
        });
        
        Box::into_raw(pmc)
    }
}

#[no_mangle]
pub extern "C" fn pmc_delete(pmc: *mut PMC) {
    if !pmc.is_null() {
        unsafe {
            let _ = Box::from_raw(pmc);
        }
    }
}

fn forward(pmc: &PMC, input: &[f64]) -> (Vec<f64>, f64) {
    // Couche cachée
    let mut hidden = Vec::new();
    for i in 0..pmc.w1.len() {
        let mut sum = pmc.b1[i];
        for j in 0..input.len() {
            sum += pmc.w1[i][j] * input[j];
        }
        hidden.push(sigmoid(sum));
    }
    
    // Couche de sortie
    let mut output = pmc.b2;
    for i in 0..hidden.len() {
        output += pmc.w2[0][i] * hidden[i];
    }
    
    (hidden, output)  // output non activé (pour la régression)
}

#[no_mangle]
pub extern "C" fn pmc_fit(
    pmc: *mut PMC,
    features: *const f64,
    targets: *const f64,
    n_samples: usize,
    n_features: usize,
    epochs: usize,
) -> *mut f64 {
    unsafe {
        let pmc = &mut *pmc;
        let features_slice = std::slice::from_raw_parts(features, n_samples * n_features);
        let targets_slice = std::slice::from_raw_parts(targets, n_samples);
        
        let mut losses = Vec::with_capacity(epochs);
        
        for epoch in 0..epochs {
            let mut total_loss = 0.0;
            
            for i in 0..n_samples {
                let input = &features_slice[i * n_features..(i + 1) * n_features];
                let target = targets_slice[i];
                
                // Forward pass
                let (hidden, output) = forward(pmc, input);
                let error = output - target;
                total_loss += error * error;
                
                // Backward pass
                // Dérivée de la perte par rapport à la sortie
                let d_output = error;  // dérivée de MSE: 2*(output-target), on absorbe le 2 dans le learning rate
                
                // Mise à jour w2 et b2
                for j in 0..hidden.len() {
                    pmc.w2[0][j] -= pmc.learning_rate * d_output * hidden[j];
                }
                pmc.b2 -= pmc.learning_rate * d_output;
                
                // Backprop vers la couche cachée
                for j in 0..hidden.len() {
                    let d_hidden = d_output * pmc.w2[0][j] * sigmoid_derivative(hidden[j]);
                    
                    // Mise à jour w1 et b1
                    for k in 0..input.len() {
                        pmc.w1[j][k] -= pmc.learning_rate * d_hidden * input[k];
                    }
                    pmc.b1[j] -= pmc.learning_rate * d_hidden;
                }
            }
            
            losses.push(total_loss / n_samples as f64);
            
            if epoch % 100 == 0 {
                println!("PMC Epoch {}: Loss = {:.6}", epoch, losses.last().unwrap());
            }
        }
        
        Box::into_raw(losses.into_boxed_slice()) as *mut f64
    }
}

#[no_mangle]
pub extern "C" fn pmc_predict_batch(
    pmc: *const PMC,
    features: *const f64,
    results: *mut f64,
    n_samples: usize,
    n_features: usize,
) {
    unsafe {
        let pmc = &*pmc;
        let features_slice = std::slice::from_raw_parts(features, n_samples * n_features);
        let results_slice = std::slice::from_raw_parts_mut(results, n_samples);
        
        for i in 0..n_samples {
            let input = &features_slice[i * n_features..(i + 1) * n_features];
            let (_, output) = forward(pmc, input);
            results_slice[i] = output;
        }
    }
}

#[no_mangle]
pub extern "C" fn pmc_accuracy(
    pmc: *const PMC,
    features: *const f64,
    targets: *const f64,
    n_samples: usize,
    n_features: usize,
) -> f64 {
    unsafe {
        let pmc = &*pmc;
        let features_slice = std::slice::from_raw_parts(features, n_samples * n_features);
        let targets_slice = std::slice::from_raw_parts(targets, n_samples);
        
        let mut correct = 0;
        
        for i in 0..n_samples {
            let input = &features_slice[i * n_features..(i + 1) * n_features];
            let (_, output) = forward(pmc, input);
            let predicted = if output >= 0.0 { 1.0 } else { -1.0 };
            
            if predicted == targets_slice[i] {
                correct += 1;
            }
        }
        
        correct as f64 / n_samples as f64
    }
}

#[no_mangle]
pub extern "C" fn pmc_losses_delete(losses: *mut f64, len: usize) {
    if !losses.is_null() {
        unsafe {
            let _ = Box::from_raw(std::slice::from_raw_parts_mut(losses, len));
        }
    }
}