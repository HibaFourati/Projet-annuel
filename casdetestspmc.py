import ctypes
import numpy as np
import matplotlib.pyplot as plt 
from pathlib import Path

class PMCConfig(ctypes.Structure):
    _fields_ = [
        ("n_inputs", ctypes.c_uint),
        ("n_hidden", ctypes.c_uint),
        ("n_outputs", ctypes.c_uint),
        ("learning_rate", ctypes.c_double),
    ]

# WRAPPER PMC RUST
class PMC:
    def __init__(self, n_inputs: int, n_hidden: int = 2, learning_rate: float = 0.1):
    
        dll_path = "./target/release/neural_networks.dll"
        
        print(f"Chargement de: {dll_path}")
        self.lib = ctypes.CDLL(dll_path)
        
       
        self.lib.pmc_new.argtypes = [ctypes.POINTER(PMCConfig)]
        self.lib.pmc_new.restype = ctypes.c_void_p
        
       
        self.lib.pmc_delete.argtypes = [ctypes.c_void_p]
        
      
        self.lib.pmc_fit.argtypes = [
            ctypes.c_void_p,                    
            ctypes.POINTER(ctypes.c_double),    
            ctypes.POINTER(ctypes.c_double),    
            ctypes.c_size_t,                    
            ctypes.c_size_t,                    
            ctypes.c_size_t,                    
        ]
        self.lib.pmc_fit.restype = ctypes.c_double
        
       
        self.lib.pmc_predict_batch.argtypes = [
            ctypes.c_void_p,                    
            ctypes.POINTER(ctypes.c_double),    
            ctypes.POINTER(ctypes.c_double),   
            ctypes.c_size_t,                   
            ctypes.c_size_t,                  
        ]
        
     
        self.lib.pmc_accuracy.argtypes = [
            ctypes.c_void_p,                    
            ctypes.POINTER(ctypes.c_double),    
            ctypes.POINTER(ctypes.c_double),    
            ctypes.c_size_t,                    
            ctypes.c_size_t,                    
        ]
        self.lib.pmc_accuracy.restype = ctypes.c_double
        
        config = PMCConfig(
            n_inputs=n_inputs,
            n_hidden=n_hidden,
            n_outputs=1,  
            learning_rate=learning_rate
        )
     
        self.model_ptr = self.lib.pmc_new(ctypes.byref(config))
        
        if not self.model_ptr:
            raise RuntimeError("Échec")
        
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        
        print(f"PMC crée: architecture {n_inputs} → {n_hidden} → 1 (activation: tanh)")
    
    def __del__(self):
        if hasattr(self, 'model_ptr') and self.model_ptr:
            self.lib.pmc_delete(self.model_ptr)
    
    def fit(self, X: np.ndarray, y: np.ndarray, max_iterations: int = 1000) -> float:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        n_samples, n_features = X.shape
        
        if n_features != self.n_inputs:
            raise ValueError(f"Attendu {self.n_inputs} features, reçu {n_features}")
        
        X_ptr = X.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        y_ptr = y.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        
      
        error = self.lib.pmc_fit(
            self.model_ptr, X_ptr, y_ptr, n_samples, n_features, max_iterations
        )
        
        return error
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        n_samples, n_features = X.shape
        
        if n_features != self.n_inputs:
            raise ValueError(f"Attendu {self.n_inputs} features, reçu {n_features}")
        
        results = np.zeros(n_samples, dtype=np.float64)
        
        X_ptr = X.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        results_ptr = results.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        
        self.lib.pmc_predict_batch(self.model_ptr, X_ptr, results_ptr, n_samples, n_features)
        
        return results
    
    def predict_class(self, X: np.ndarray, threshold: float = 0.0) -> np.ndarray:
        predictions = self.predict(X)
       
        return np.where(predictions >= threshold, 1, -1)
    
    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        n_samples, n_features = X.shape
        
        if n_features != self.n_inputs:
            raise ValueError(f"Attendu {self.n_inputs} features, reçu {n_features}")
        
        X_ptr = X.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        y_ptr = y.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        
      
        return self.lib.pmc_accuracy(self.model_ptr, X_ptr, y_ptr, n_samples, n_features)