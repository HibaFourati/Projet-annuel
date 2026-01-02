#[repr(C)]
#[repr(C)]
pub struct LinearModel {
    pub weights: Vec<f64>,
    pub bias: f64, 
    pub learning_rate: f64, 
}


#[no_mangle] 
pub extern "C" fn linear_model_new(input_dim: usize, learning_rate: f64) -> *mut LinearModel { 
    let model = LinearModel {
        weights: vec![0.0; input_dim], 
        bias: 0.0,
        learning_rate, 
    };
    Box::into_raw(Box::new(model)) 
}

#[no_mangle]
pub extern "C" fn linear_model_delete(model: *mut LinearModel) {
    if !model.is_null() {
        unsafe { 
            let _ = Box::from_raw(model); } 
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
        
        let mut previous_error = f64::MAX; 
        
        for iteration in 0..max_iterations {
            let mut total_error = 0.0;
            
            for i in 0..n_samples {
                
                let mut prediction = model.bias;
                for j in 0..n_features { 
                    prediction += model.weights[j] * features_slice[i * n_features + j];
                }
                
                let error = prediction - targets_slice[i]; 
                total_error += error * error; 
                
                
                for j in 0..n_features {
                    model.weights[j] -= model.learning_rate * error * features_slice[i * n_features + j];
                }
                model.bias -= model.learning_rate * error;
            }
            
            let mean_error = total_error / n_samples as f64;
            
            
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