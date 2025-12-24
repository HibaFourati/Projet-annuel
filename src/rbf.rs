use rand::Rng;
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
}

#[no_mangle]
pub extern "C" fn rbf_new(config: *const RBFConfig) -> *mut RBF {
    unsafe {
        let c = &*config;
        let mut rng = rand::thread_rng();
        
        // 1. Initialisation PLUS PETITE
        let mut centers = Vec::new();
        for _ in 0..c.n_centers {
            let mut center = Vec::new();
            for _ in 0..c.n_inputs {
                center.push(rng.gen_range(-0.1..0.1)); // Plus petit
            }
            centers.push(center);
        }
        
        // 2. Poids PLUS PETITS
        let mut weights = Vec::new();
        for _ in 0..c.n_outputs {
            let mut neuron_weights = Vec::new();
            for _ in 0..c.n_centers {
                neuron_weights.push(rng.gen_range(-0.05..0.05)); // Plus petit
            }
            weights.push(neuron_weights);
        }
        
        // 3. Biais plus petits
        let mut biases = Vec::new();
        for _ in 0..c.n_outputs {
            biases.push(rng.gen_range(-0.05..0.05));
        }
        
        Box::into_raw(Box::new(RBF {
            centers,
            weights,
            biases,
            config: RBFConfig {
                n_inputs: c.n_inputs,
                n_centers: c.n_centers,
                n_outputs: c.n_outputs,
                learning_rate: c.learning_rate * 5.0, // ×5
                sigma: c.sigma,
            },
        }))
    }
}

#[no_mangle]
pub extern "C" fn rbf_delete(rbf: *mut RBF) {
    if !rbf.is_null() { unsafe { let _ = Box::from_raw(rbf); } }
}

fn gaussian_rbf(x: &[f64], center: &[f64], sigma: f64) -> f64 {
    let mut sum = 0.0;
    for i in 0..x.len() {
        let diff = x[i] - center[i];
        sum += diff * diff;
    }
    (-sum / (sigma * sigma)).exp() // Sans le 2.0
}

#[no_mangle]
pub extern "C" fn rbf_fit(rbf: *mut RBF, x: *const f64, y: *const f64, n: usize, d: usize, max_iter: usize) -> f64 {
    unsafe {
        let rbf = &mut *rbf;
        let x = std::slice::from_raw_parts(x, n * d);
        let y = std::slice::from_raw_parts(y, n * rbf.config.n_outputs as usize);
        
        let lr = rbf.config.learning_rate;
        let sigma = rbf.config.sigma;
        let mut total_error = 0.0;

        for _ in 0..max_iter {
            total_error = 0.0;
            
            for sample_idx in 0..n {
                // Activations
                let mut activations = vec![0.0; rbf.config.n_centers as usize];
                for (j, center) in rbf.centers.iter().enumerate() {
                    let sample = &x[sample_idx * d..(sample_idx + 1) * d];
                    activations[j] = gaussian_rbf(sample, center, sigma);
                }
                
                // Sorties
                let mut outputs = vec![0.0; rbf.config.n_outputs as usize];
                for output_idx in 0..rbf.config.n_outputs as usize {
                    let mut sum = rbf.biases[output_idx];
                    for j in 0..rbf.config.n_centers as usize {
                        sum += rbf.weights[output_idx][j] * activations[j];
                    }
                    outputs[output_idx] = sum;
                }
                
                // Mise à jour
                for output_idx in 0..rbf.config.n_outputs as usize {
                    let target = y[sample_idx * rbf.config.n_outputs as usize + output_idx];
                    let error = target - outputs[output_idx];
                    total_error += error * error;
           
                    let delta = lr * error;
                    rbf.biases[output_idx] += delta;
                    
                    for j in 0..rbf.config.n_centers as usize {
                        rbf.weights[output_idx][j] += delta * activations[j];
                    }
                }
            }
            
            total_error /= (n * rbf.config.n_outputs as usize) as f64;
            
            if total_error < 0.01 { // Condition plus permissive
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