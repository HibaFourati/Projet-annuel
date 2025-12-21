// src/pmc.rs
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
    weights1: Vec<f64>,
    weights2: Vec<f64>,
    bias1: Vec<f64>,
    bias2: f64,
    config: PMCConfig,
}

#[no_mangle]
pub extern "C" fn pmc_new(config: *const PMCConfig) -> *mut PMC {
    unsafe {
        let c = &*config;
        let mut rng = rand::thread_rng();
        
        let w1_size = (c.n_inputs * c.n_hidden) as usize;
        let w2_size = (c.n_hidden * c.n_outputs) as usize;
        
        Box::into_raw(Box::new(PMC {
            weights1: (0..w1_size).map(|_| rng.gen_range(-0.5..0.5)).collect(),
            weights2: (0..w2_size).map(|_| rng.gen_range(-0.5..0.5)).collect(),
            bias1: (0..c.n_hidden as usize).map(|_| rng.gen_range(-0.5..0.5)).collect(),
            bias2: rng.gen_range(-0.5..0.5),
            config: PMCConfig {
                n_inputs: c.n_inputs,
                n_hidden: c.n_hidden,
                n_outputs: c.n_outputs,
                learning_rate: c.learning_rate,
            },
        }))
    }
}

#[no_mangle]
pub extern "C" fn pmc_delete(pmc: *mut PMC) {
    if !pmc.is_null() { unsafe { let _ = Box::from_raw(pmc); } }
}

fn tanh(x: f64) -> f64 { x.tanh() }
fn tanh_derivative(x: f64) -> f64 { 1.0 - x.tanh() * x.tanh() }

#[no_mangle]
pub extern "C" fn pmc_fit(pmc: *mut PMC, x: *const f64, y: *const f64, n: usize, d: usize, max_iter: usize) -> f64 {
    unsafe {
        let pmc = &mut *pmc;
        let x = std::slice::from_raw_parts(x, n * d);
        let y = std::slice::from_raw_parts(y, n);
        
        let lr = pmc.config.learning_rate;
        let mut error = 0.0;
        
        for _ in 0..max_iter {
            error = 0.0;
            
            for i in 0..n {
                // Forward
                let mut h = vec![0.0; pmc.config.n_hidden as usize];
                for j in 0..(pmc.config.n_hidden as usize) {
                    let mut sum = pmc.bias1[j];
                    for k in 0..d {
                        sum += pmc.weights1[j * d + k] * x[i * d + k];
                    }
                    h[j] = tanh(sum);
                }
                
                let mut out = pmc.bias2;
                for j in 0..(pmc.config.n_hidden as usize) {
                    out += pmc.weights2[j] * h[j];
                }
                let pred = tanh(out);
                
                // Error
                let e = y[i] - pred;
                error += e * e;
                
                // Backward
                let delta_out = e * tanh_derivative(pred);
                
                // Update
                for j in 0..(pmc.config.n_hidden as usize) {
                    pmc.weights2[j] += lr * delta_out * h[j];
                    let delta_h = delta_out * pmc.weights2[j] * tanh_derivative(h[j]);
                    for k in 0..d {
                        pmc.weights1[j * d + k] += lr * delta_h * x[i * d + k];
                    }
                    pmc.bias1[j] += lr * delta_h;
                }
                pmc.bias2 += lr * delta_out;
            }
            
            if error / (n as f64) < 0.001 { break; }
        }
        
        error / (n as f64)
    }
}

#[no_mangle]
pub extern "C" fn pmc_predict_batch(pmc: *const PMC, x: *const f64, out: *mut f64, n: usize, d: usize) {
    unsafe {
        let pmc = &*pmc;
        let x = std::slice::from_raw_parts(x, n * d);
        let out = std::slice::from_raw_parts_mut(out, n);
        
        for i in 0..n {
            let mut h = vec![0.0; pmc.config.n_hidden as usize];
            for j in 0..(pmc.config.n_hidden as usize) {
                let mut sum = pmc.bias1[j];
                for k in 0..d {
                    sum += pmc.weights1[j * d + k] * x[i * d + k];
                }
                h[j] = tanh(sum);
            }
            
            let mut result = pmc.bias2;
            for j in 0..(pmc.config.n_hidden as usize) {
                result += pmc.weights2[j] * h[j];
            }
            out[i] = tanh(result);
        }
    }
}

#[no_mangle]
pub extern "C" fn pmc_accuracy(pmc: *const PMC, x: *const f64, y: *const f64, n: usize, d: usize) -> f64 {
    unsafe {
        let mut pred = vec![0.0; n];
        pmc_predict_batch(pmc, x, pred.as_mut_ptr(), n, d);
        
        let y = std::slice::from_raw_parts(y, n);
        let mut correct = 0;
        
        for i in 0..n {
            if (pred[i] - y[i]).abs() < 0.5 {
                correct += 1;
            }
        }
        
        correct as f64 / n as f64
    }
}

#[no_mangle]
pub extern "C" fn pmc_losses_delete(_losses: *mut f64, _len: usize) {}