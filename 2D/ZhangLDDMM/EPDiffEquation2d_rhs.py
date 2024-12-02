import torch
from adTranspose2d import adTranspose

"""
Parameters:
    t: actual time
    v: velocity field
    dummy: not used, just to match the signature
    nx, ny: size of the domain in each direction
    Gx, Gy: gradient operators
    L: Laplacian operator
    K: inverse of the Laplacian operator
Returns:
    rhs: right-hand side of the equation
"""
def EPDiffEquation2d_rhs(t, v, dummy, nx, ny, Gx, Gy, L, K):
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

    Gx, Gy, L, K = map(to_tensor, [Gx, Gy, L, K])

    # Reshape v into a 3D matrix (ny, nx, 2)
    v = v.reshape(int(ny), int(nx), 2)

    # Compute the right hand side of the equation
    rhs = -adTranspose(v, v, L, K, Gx, Gy)
    
    # Reshape rhs into a vector
    rhs = torch.reshape(rhs, (-1,))

    # print("\tEPDiffEquation3d_rhs finished: " + str(rhs.shape) + " " + str(rhs.device))

    return rhs


