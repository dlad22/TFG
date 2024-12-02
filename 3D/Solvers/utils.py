import numpy as np
import torch

# Función para verificar si tspan es un tipo válido
def is_valid_type(tspan):
    # Verificar si es un número individual
    if np.issubdtype(type(tspan), np.number):
        return True
    # Verificar si es un arreglo de NumPy
    elif isinstance(tspan, np.ndarray) and np.issubdtype(tspan.dtype, np.number):
        return True
    # Verificar si es un tensor de PyTorch
    elif isinstance(tspan, torch.Tensor) and torch.is_floating_point(tspan):
        return True
    else:
        return False
