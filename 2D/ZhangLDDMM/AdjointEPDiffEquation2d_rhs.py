import numpy as np
import torch
from adTranspose2d import adTranspose 
from ad2d import ad

"""
Parameters:
    t: actual time
    v_: vector with the variable of interest (velocities) at a specific time
    dummy: not used, just to match the signature
    v: matrix with the velocities at all times
    U: matrix with some field at all times
    tt: discretized time vector
    nx, ny: size of the domain in each direction
    Gx, Gy: gradient operators (matrix)
    L: Laplacian operator
    K: inverse of the Laplacian operator
"""
def AdjointEPDiffEquation2d_rhs(t, v_, dummy, v, U, tt, nx, ny, Gx, Gy, L, K):
    # print("AdjointEPDiffEquation3d_rhs starting...")
    # Determine if using GPU
    if isinstance(v, torch.Tensor):
        device = v.device
    else:
        device = torch.device("cpu")
    
    # Convert numpy arrays to torch tensors if needed and move them to the same device
    def to_tensor(arr):
        if not isinstance(arr, torch.Tensor):
            return torch.tensor(arr, dtype=torch.float32, device=device)
        return arr

    Gx, Gy, tt, L, K = map(to_tensor, [Gx, Gy, tt, L, K])

    # Initialize rhs with zeros
    rhs = torch.zeros((int(ny), int(nx), 2), dtype=torch.float32, device=device)
    
    # Take the first time step that is greater than the actual time
    tidx = torch.where(t >= tt)[0][0].item()

    # Reshape v_ and w
    v_ = torch.reshape(v_, (int(ny), int(nx), 2))
    w = torch.reshape(v[:, tidx], (int(ny), int(nx), 2))

    # Reshape rhs from U
    rhs = torch.reshape(U[:, tidx], (int(ny), int(nx), 2))

    # Compute adT
    adT = adTranspose(v_, w, L, K, Gx, Gy)

    # Update rhs
    rhs = rhs - adT

    # Compute ad and update rhs
    rhs = rhs + ad(w, v_, Gx, Gy)

    # Flatten rhs to a vector
    rhs = torch.reshape(rhs, (-1,))

    # print("AdjointEPDiffEquation3d_rhs finished: " + str(rhs.shape) + " " + str(rhs.device))

    return rhs
