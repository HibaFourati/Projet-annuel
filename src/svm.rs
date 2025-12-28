use rand::Rng;
use libc::{c_double, c_uint};
use std::f64;

#[repr(C)]
pub struct SVMConfig {
    pub n_inputs: c_uint,
    pub learning_rate: c_double,
    pub c: c_double, // paramètre de régularisation
    pub max_iterations: c_uint,
}

#[repr(C)]
pub struct SVM {
    weights: Vec<f64>,
    bias: f64,
    config: SVMConfig,
}

#[no_mangle]
pub extern "C" fn svm_new(config: *const SVMConfig) -> *mut SVM {
    unsafe {
        let c = &*config;
        let mut rng = rand::thread_rng();
        let mut weights = Vec::with_capacity(c.n_inputs as usize);
        for _ in 0..c.n_inputs {
            weights.push(rng.gen_range(-0.1..0.1));
        }
        Box::into_raw(Box::new(SVM {
            weights,
            bias: 0.0,
            config: SVMConfig {
                n_inputs: c.n_inputs,
                learning_rate: c.learning_rate,
                c: c.c,
                max_iterations: c.max_iterations,
            },
        }))
    }
}

#[no_mangle]
pub extern "C" fn svm_delete(svm: *mut SVM) {
    if !svm.is_null() {
        unsafe { let _ = Box::from_raw(svm); }
    }
}

// Fonction de perte charnière: max(0, 1 - y*(w·x + b))
fn hinge_loss(prediction: f64, label: f64) -> f64 {
    (1.0 - label * prediction).max(0.0)
}

// Dérivée de la perte charnière
fn hinge_loss_derivative(prediction: f64, label: f64) -> f64 {
    if label * prediction < 1.0 {
        -label
    } else {
        0.0
    }
}

#[no_mangle]
pub extern "C" fn svm_fit(
    svm: *mut SVM,
    x: *const f64,
    y: *const f64,
    n_samples: usize,
    n_features: usize,
    max_iter: usize,
) -> f64 {
    unsafe {
        let svm = &mut *svm;
        let x = std::slice::from_raw_parts(x, n_samples * n_features);
        let y = std::slice::from_raw_parts(y, n_samples);
        let lr = svm.config.learning_rate;
        let c = svm.config.c;
        let mut total_loss = 0.0;

        for _ in 0..max_iter {
            total_loss = 0.0;
            for i in 0..n_samples {
                let sample = &x[i * n_features..(i + 1) * n_features];
                let label = y[i];

                // Prédiction
                let mut prediction = svm.bias;
                for j in 0..n_features {
                    prediction += svm.weights[j] * sample[j];
                }

                // Calculer la perte
                let loss = hinge_loss(prediction, label);
                total_loss += loss;

                // Régularisation L2
                for j in 0..n_features {
                    total_loss += c * svm.weights[j] * svm.weights[j] / 2.0;
                }

                // Gradient
                let loss_grad = hinge_loss_derivative(prediction, label);

                if loss_grad != 0.0 {
                    for j in 0..n_features {
                        let grad = loss_grad * sample[j] + c * svm.weights[j];
                        svm.weights[j] -= lr * grad;
                    }
                    svm.bias -= lr * loss_grad;
                } else {
                    for j in 0..n_features {
                        svm.weights[j] -= lr * c * svm.weights[j];
                    }
                }
            }
            total_loss /= n_samples as f64;
            if total_loss < 0.01 {
                break;
            }
        }
        total_loss
    }
}

#[no_mangle]
pub extern "C" fn svm_predict_batch(
    svm: *const SVM,
    x: *const f64,
    out: *mut f64,
    n_samples: usize,
    n_features: usize,
) {
    unsafe {
        let svm = &*svm;
        let x = std::slice::from_raw_parts(x, n_samples * n_features);
        let out = std::slice::from_raw_parts_mut(out, n_samples);

        for i in 0..n_samples {
            let sample = &x[i * n_features..(i + 1) * n_features];
            let mut prediction = svm.bias;
            for j in 0..n_features {
                prediction += svm.weights[j] * sample[j];
            }
            out[i] = if prediction >= 0.0 { 1.0 } else { -1.0 };
        }
    }
}

#[no_mangle]
pub extern "C" fn svm_predict_probability(
    svm: *const SVM,
    x: *const f64,
    out: *mut f64,
    n_samples: usize,
    n_features: usize,
) {
    unsafe {
        let svm = &*svm;
        let x = std::slice::from_raw_parts(x, n_samples * n_features);
        let out = std::slice::from_raw_parts_mut(out, n_samples);

        for i in 0..n_samples {
            let sample = &x[i * n_features..(i + 1) * n_features];
            let mut prediction = svm.bias;
            for j in 0..n_features {
                prediction += svm.weights[j] * sample[j];
            }
            out[i] = prediction; // score brut
        }
    }
}

#[no_mangle]
pub extern "C" fn svm_accuracy(
    svm: *const SVM,
    x: *const f64,
    y: *const f64,
    n_samples: usize,
    n_features: usize,
) -> f64 {
    unsafe {
        let mut predictions = vec![0.0; n_samples];
        svm_predict_batch(svm, x, predictions.as_mut_ptr(), n_samples, n_features);
        let y = std::slice::from_raw_parts(y, n_samples);

        let mut correct = 0;
        for i in 0..n_samples {
            if predictions[i] == y[i] {
                correct += 1;
            }
        }
        correct as f64 / n_samples as f64
    }
}
