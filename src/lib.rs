pub mod linear_model;
pub mod pmc;
pub mod svm;

// Réexporter explicitement les fonctions du SVM
pub use svm::{
    svm_new,
    svm_delete,
    svm_fit,
    svm_predict_batch,
    svm_predict_probability,
    svm_accuracy,
};
