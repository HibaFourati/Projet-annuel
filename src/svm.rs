use std::slice;

#[repr(C)]
pub struct SVM {
    pub weights: Vec<f64>,
    pub bias: f64,
}

#[no_mangle]
pub extern "C" fn svm_new(n_features: usize) -> *mut SVM {
    let svm = SVM {
        weights: vec![0.0; n_features],
        bias: 0.0,
    };
    Box::into_raw(Box::new(svm))
}

#[no_mangle]
pub extern "C" fn svm_delete(ptr: *mut SVM) {
    if !ptr.is_null() {
        unsafe { drop(Box::from_raw(ptr)) };
    }
}

#[no_mangle]
pub extern "C" fn svm_fit(
    svm: *mut SVM,
    x: *const f64,
    y: *const f64,
    n_samples: usize,
    n_features: usize,
    lr: f64,
    c: f64,
    epochs: usize,
) {
    let svm = unsafe { &mut *svm };
    let x = unsafe { slice::from_raw_parts(x, n_samples * n_features) };
    let y = unsafe { slice::from_raw_parts(y, n_samples) };

    for _ in 0..epochs {
        for i in 0..n_samples {
            let xi = &x[i * n_features..(i + 1) * n_features];
            let yi = y[i];

            let mut score = svm.bias;
            for j in 0..n_features {
                score += svm.weights[j] * xi[j];
            }

            if yi * score < 1.0 {
                for j in 0..n_features {
                    svm.weights[j] += lr * (yi * xi[j] - c * svm.weights[j]);
                }
                svm.bias += lr * yi;
            } else {
                for j in 0..n_features {
                    svm.weights[j] -= lr * c * svm.weights[j];
                }
            }
        }
    }
}

#[no_mangle]
pub extern "C" fn svm_predict_margin(
    svm: *const SVM,
    x: *const f64,
    out: *mut f64,
    n_samples: usize,
    n_features: usize,
) {
    let svm = unsafe { &*svm };
    let x = unsafe { slice::from_raw_parts(x, n_samples * n_features) };
    let out = unsafe { slice::from_raw_parts_mut(out, n_samples) };

    for i in 0..n_samples {
        let xi = &x[i * n_features..(i + 1) * n_features];
        let mut score = svm.bias;
        for j in 0..n_features {
            score += svm.weights[j] * xi[j];
        }
        out[i] = score;
    }
}
