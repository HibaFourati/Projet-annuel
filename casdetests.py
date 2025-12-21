import ctypes
import numpy as np
import matplotlib.pyplot as plt
import os

# WRAPPER RUST
class LinearModel:
    def __init__(self, input_dim: int, learning_rate: float = 0.01):
        lib_path = "./target/release/neural_networks.dll"
        
        
        self.lib = ctypes.CDLL(lib_path)
        
        
        self.lib.linear_model_new.argtypes = [ctypes.c_size_t, ctypes.c_double]
        self.lib.linear_model_new.restype = ctypes.c_void_p
        
        self.lib.linear_model_delete.argtypes = [ctypes.c_void_p]
        
        self.lib.linear_model_fit.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), 
            ctypes.POINTER(ctypes.c_double), ctypes.c_size_t, 
            ctypes.c_size_t, ctypes.c_size_t
        ]
        self.lib.linear_model_fit.restype = ctypes.c_double
        
        self.lib.linear_model_predict_batch.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double), ctypes.c_size_t, ctypes.c_size_t
        ]
        
        self.lib.linear_model_get_weights.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_double)
        ]
        
        self.lib.linear_model_get_bias.argtypes = [ctypes.c_void_p]
        self.lib.linear_model_get_bias.restype = ctypes.c_double
        
        
        self.model_ptr = self.lib.linear_model_new(input_dim, learning_rate) 
        self.input_dim = input_dim
    
    def __del__(self):
        if hasattr(self, 'model_ptr'):
            self.lib.linear_model_delete(self.model_ptr) 
    
    def fit(self, X: np.ndarray, y: np.ndarray, max_iterations: int = 1000) -> float:
       

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

         
        if X.ndim == 1:
            X = X.reshape(-1, 1) 
        
        n_samples, n_features = X.shape
        
        
        X_ptr = X.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        y_ptr = y.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        
        return self.lib.linear_model_fit(
            self.model_ptr, X_ptr, y_ptr, n_samples, n_features, max_iterations
        )
    
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        n_samples, n_features = X.shape
        results = np.zeros(n_samples, dtype=np.float64)
        
        X_ptr = X.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        results_ptr = results.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        
        self.lib.linear_model_predict_batch(
            self.model_ptr, X_ptr, results_ptr, n_samples, n_features
        )
        
        return results
    
    def predict_class(self, X: np.ndarray) -> np.ndarray:
        predictions = self.predict(X)
        return np.where(predictions >= 0, 1, -1)
    
    def get_weights(self) -> np.ndarray:
        weights = np.zeros(self.input_dim, dtype=np.float64)
        weights_ptr = weights.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        self.lib.linear_model_get_weights(self.model_ptr, weights_ptr)
        return weights
    
    def get_bias(self) -> float:
        return self.lib.linear_model_get_bias(self.model_ptr)
