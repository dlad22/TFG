import numpy as np
import torch
from adTranspose2d import adTranspose 
import subprocess
import gc

"""
Parameters:
    t: actual time
    U: vector with the variable of interest (velocities) at a specific time
    dummy: not used, just to match the signature
    v: matrix with the velocities at all times
    tt: discretized time vector
    nx, ny: size of the domain in each direction
    Gx, Gy: gradient operators (matrix)
    L: Laplacian operator
    K: inverse of the Laplacian operator
"""
def UEPDiffEquation2d_rhs(t, U, dummy, v, tt, nx, ny, Gx, Gy, L, K):
    # print("\tUEPDiffEquation3d_rhs starting...")
    # Determine if using GPU
    if isinstance(U, torch.Tensor):
        device = U.device
    else:
        device = torch.device("cpu")
    
    # Convert numpy arrays to torch tensors if needed and move them to the same device
    def to_tensor(arr):
        if not isinstance(arr, torch.Tensor):
            return torch.tensor(arr, dtype=torch.float32, device=arr.device if hasattr(arr, 'device') else 'cpu')
        return arr

    Gx, Gy, L, K, tt = map(to_tensor, [Gx, Gy, L, K, tt])

    # Take the first time step that is greater than the actual time
    tidx = torch.where(t >= tt)[0]
    tidx = tidx[0].item()  # Convert to Python scalar

    # Reshape the vector U into a 3D matrix (ny, nx, 2)
    U = torch.reshape(U, (int(ny), int(nx), 2))

    # Delete the column tidx and reshape the matrix v into a 3D matrix (ny, nx, 2)
    w = torch.reshape(v[:, tidx], (int(ny), int(nx), 2))

    # Compute the right hand side of the equation
    rhs = -adTranspose(w, U, L, K, Gx, Gy)

    # Reshape the matrix rhs into a vector
    rhs = torch.reshape(rhs, (-1,))

    # Delete temporary variables from gpu memory
    del w
    # Forzar la recolección de basura
    gc.collect()
    # Vaciar la caché de la GPU
    torch.cuda.empty_cache()

    return rhs
