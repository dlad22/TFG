import torch
from adTranspose import adTranspose 

"""
Parameters:
    t: actual time
    v: velocity field
    dummy: not used, just to match the signature
    nx, ny, nz: size of the domain in each direction
    Gx, Gy, Gz: gradient operators
    L: Laplacian operator
    K: inverse of the Laplacian operator
Returns:
    rhs: right-hand side of the equation
"""
def EPDiffEquation3d_rhs(t, v, dummy, nx, ny, nz, Gx, Gy, Gz, L, K):
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

    Gx, Gy, Gz, L, K = map(to_tensor, [Gx, Gy, Gz, L, K])

    # Reshape v into a 4D matrix (ny, nx, nz, 3)
    v = v.reshape(int(ny), int(nx), int(nz), 3)

    # Compute the right hand side of the equation
    rhs = -adTranspose(v, v, L, K, Gx, Gy, Gz)
    
    # Reshape rhs into a vector
    rhs = torch.reshape(rhs, (-1,))
    
    return rhs


