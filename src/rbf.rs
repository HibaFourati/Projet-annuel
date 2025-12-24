use rand::Rng;
use rand::seq::SliceRandom;
use libc::{c_double, c_uint};

#[repr(C)]
pub struct RBFConfig {
    pub n_inputs: c_uint,
    pub n_centers: c_uint,
    pub n_outputs: c_uint,
    pub learning_rate: c_double,
    pub sigma: c_double,
}

#[repr(C)]
pub struct RBF {
    centers: Vec<Vec<f64>>,
    weights: Vec<Vec<f64>>,
    biases: Vec<f64>,
    config: RBFConfig,
    initialized_from_data: bool,
}

#[no_mangle]
pub extern "C" fn rbf_new(config: *const RBFConfig) -> *mut RBF {
    unsafe {
        let c = &*config;
        
        // Initialisation temporaire (les centres seront remplacés lors du fit)
        let centers = Vec::new();
        
        // Poids - initialisation raisonnable
        let mut weights = Vec::new();
        for _ in 0..c.n_outputs {
            let mut neuron_weights = Vec::new();
            for _ in 0..c.n_centers {
                neuron_weights.push(0.0); // Commence à zéro
            }
            weights.push(neuron_weights);
        }
        
        // Biais à zéro
        let biases = vec![0.0; c.n_outputs as usize];
        
        Box::into_raw(Box::new(RBF {
            centers,
            weights,
            biases,
            config: RBFConfig {
                n_inputs: c.n_inputs,
                n_centers: c.n_centers,
                n_outputs: c.n_outputs,
                learning_rate: c.learning_rate, // NE PAS MULTIPLIER!
                sigma: if c.sigma <= 0.0 { 1.0 } else { c.sigma },
            },
            initialized_from_data: false,
        }))
    }
}

#[no_mangle]
pub extern "C" fn rbf_delete(rbf: *mut RBF) {
    if !rbf.is_null() {
        unsafe {
            let _ = Box::from_raw(rbf);
        }
    }
}

fn gaussian_rbf(x: &[f64], center: &[f64], sigma: f64) -> f64 {
    let mut sum = 0.0;
    for i in 0..x.len() {
        let diff = x[i] - center[i];
        sum += diff * diff;
    }
    // CORRECTION: Ajouter le 2.0 dans le dénominateur
    (-sum / (2.0 * sigma * sigma)).exp()
}

fn initialize_centers_from_data(rbf: &mut RBF, x: &[f64], n: usize, d: usize) {
    if rbf.initialized_from_data {
        return;
    }
    
    let mut rng = rand::thread_rng();
    
    // 1. Choisir n_centers points aléatoires parmi les données
    let mut indices: Vec<usize> = (0..n).collect();
    indices.shuffle(&mut rng);
    
    rbf.centers.clear();
    for i in 0..(rbf.config.n_centers as usize).min(n) {
        let start = indices[i] * d;
        let end = start + d;
        rbf.centers.push(x[start..end].to_vec());
    }
    
    // 2. Si on a moins de centres que demandé, compléter avec des copies
    while rbf.centers.len() < rbf.config.n_centers as usize {
        let random_idx = rng.gen_range(0..rbf.centers.len());
        rbf.centers.push(rbf.centers[random_idx].clone());
    }
    
    // 3. Calculer un sigma automatique si non spécifié
    if rbf.config.sigma <= 0.0 {
        // Calculer la distance moyenne entre les centres
        let mut total_distance = 0.0;
        let mut count = 0;
        
        for i in 0..rbf.centers.len() {
            for j in (i + 1)..rbf.centers.len() {
                let mut dist = 0.0;
                for k in 0..d {
                    let diff = rbf.centers[i][k] - rbf.centers[j][k];
                    dist += diff * diff;
                }
                total_distance += dist.sqrt();
                count += 1;
            }
        }
        
        if count > 0 {
            let avg_distance = total_distance / count as f64;
            // Sigma = distance moyenne / sqrt(2*log(2)) pour une bonne superposition
            rbf.config.sigma = avg_distance / 1.1774;
        } else {
            rbf.config.sigma = 1.0;
        }
    }
    
    rbf.initialized_from_data = true;
    
    // 4. Réinitialiser les poids avec une meilleure stratégie
    for output_idx in 0..rbf.config.n_outputs as usize {
        for j in 0..rbf.config.n_centers as usize {
            // Xavier/Glorot initialization
            let limit = (2.0 / (rbf.config.n_centers as f64 + 1.0)).sqrt();
            rbf.weights[output_idx][j] = rng.gen_range(-limit..limit);
        }
    }
}

#[no_mangle]
pub extern "C" fn rbf_fit(rbf: *mut RBF, x: *const f64, y: *const f64, n: usize, d: usize, max_iter: usize) -> f64 {
    unsafe {
        let rbf = &mut *rbf;
        let x = std::slice::from_raw_parts(x, n * d);
        let y = std::slice::from_raw_parts(y, n * rbf.config.n_outputs as usize);
        
        // ÉTAPE CRITIQUE: Initialiser les centres avec des données réelles
        initialize_centers_from_data(rbf, x, n, d);
        
        let lr = rbf.config.learning_rate;
        let sigma = rbf.config.sigma;
        let mut total_error = 0.0;
        let mut best_error = f64::INFINITY;
        let mut patience = 10;
        let mut best_weights = rbf.weights.clone();
        let mut best_biases = rbf.biases.clone();

        for _iteration in 0..max_iter {
            total_error = 0.0;
            
            // Mélanger les données pour SGD
            let mut indices: Vec<usize> = (0..n).collect();
            let mut rng = rand::thread_rng();
            for i in (1..n).rev() {
                let j = rng.gen_range(0..=i);
                indices.swap(i, j);
            }
            
            for &sample_idx in &indices {
                // 1. Calcul des activations RBF
                let mut activations = vec![0.0; rbf.config.n_centers as usize];
                for (j, center) in rbf.centers.iter().enumerate() {
                    let sample = &x[sample_idx * d..(sample_idx + 1) * d];
                    activations[j] = gaussian_rbf(sample, center, sigma);
                }
                
                // 2. Calcul de la sortie
                let mut outputs = vec![0.0; rbf.config.n_outputs as usize];
                for output_idx in 0..rbf.config.n_outputs as usize {
                    let mut sum = rbf.biases[output_idx];
                    for j in 0..rbf.config.n_centers as usize {
                        sum += rbf.weights[output_idx][j] * activations[j];
                    }
                    outputs[output_idx] = sum.tanh(); // Activation tanh pour stabilité
                }
                
                // 3. Mise à jour des poids
                for output_idx in 0..rbf.config.n_outputs as usize {
                    let target = y[sample_idx * rbf.config.n_outputs as usize + output_idx];
                    let output = outputs[output_idx];
                    let error = target - output;
                    total_error += error * error;
                    
                    // Dérivée de tanh: 1 - tanh²(x) = 1 - output²
                    let gradient = error * (1.0 - output * output);
                    
                    // Mise à jour avec momentum (simplifié)
                    rbf.biases[output_idx] += lr * gradient;
                    
                    for j in 0..rbf.config.n_centers as usize {
                        rbf.weights[output_idx][j] += lr * gradient * activations[j];
                    }
                }
            }
            
            // Moyenne de l'erreur
            total_error /= (n * rbf.config.n_outputs as usize) as f64;
            
            // Early stopping avec patience
            if total_error < best_error {
                best_error = total_error;
                best_weights = rbf.weights.clone();
                best_biases = rbf.biases.clone();
                patience = 10;
            } else {
                patience -= 1;
                if patience == 0 {
                    // Restaurer les meilleurs poids
                    rbf.weights = best_weights;
                    rbf.biases = best_biases;
                    break;
                }
            }
            
            if total_error < 0.01 {
                break;
            }
        }
        
        total_error
    }
}

#[no_mangle]
pub extern "C" fn rbf_predict_batch(rbf: *const RBF, x: *const f64, out: *mut f64, n: usize, d: usize) {
    unsafe {
        let rbf = &*rbf;
        let x = std::slice::from_raw_parts(x, n * d);
        let out = std::slice::from_raw_parts_mut(out, n * rbf.config.n_outputs as usize);
        
        let sigma = rbf.config.sigma;
        
        for sample_idx in 0..n {
            let mut activations = vec![0.0; rbf.config.n_centers as usize];
            for (j, center) in rbf.centers.iter().enumerate() {
                let sample = &x[sample_idx * d..(sample_idx + 1) * d];
                activations[j] = gaussian_rbf(sample, center, sigma);
            }
            
            for output_idx in 0..rbf.config.n_outputs as usize {
                let mut sum = rbf.biases[output_idx];
                for j in 0..rbf.config.n_centers as usize {
                    sum += rbf.weights[output_idx][j] * activations[j];
                }
                // Pas de tanh en prédiction pour conserver l'échelle
                out[sample_idx * rbf.config.n_outputs as usize + output_idx] = sum;
            }
        }
    }
}

#[no_mangle]
pub extern "C" fn rbf_accuracy(rbf: *const RBF, x: *const f64, y: *const f64, n: usize, d: usize) -> f64 {
    unsafe {
        let mut pred = vec![0.0; n * (*rbf).config.n_outputs as usize];
        rbf_predict_batch(rbf, x, pred.as_mut_ptr(), n, d);
        
        let y = std::slice::from_raw_parts(y, n * (*rbf).config.n_outputs as usize);
        let mut correct = 0;
        let mut total = 0;
        
        for i in 0..n {
            for output_idx in 0..(*rbf).config.n_outputs as usize {
                let idx = i * (*rbf).config.n_outputs as usize + output_idx;
                if (pred[idx] - y[idx]).abs() < 0.5 {
                    correct += 1;
                }
                total += 1;
            }
        }
        
        correct as f64 / total as f64
    }
}