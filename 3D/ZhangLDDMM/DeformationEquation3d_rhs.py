import numpy as np
import torch
import torch.fft

def DeformationEquation3d_rhs(t, u, dummy, v, tt, nx, ny, nz, Gx, Gy, Gz):
    """
    Parameters:
        t: actual time
        u: vector with the variable of interest at a specific time
        dummy: not used, just to match the signature
        v: matrix with the velocities at all times
        tt: discretized time vector
        nx, ny, nz: size of the domain in each direction
        Gx, Gy, Gz: gradient operators
    Returns:
        rhs: right-hand side of the equation
    """
    
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

    Gx, Gy, Gz, tt = map(to_tensor, [Gx, Gy, Gz, tt])

    # Find the first time step that is greater than the actual time
    tidx = torch.where(t <= tt)[0]
    tidx = tidx[0].item()  # Convert to Python scalar

    # Reshape u and w
    u = torch.reshape(u, (int(ny), int(nx), int(nz), 3))
    w = torch.reshape(v[:, tidx], (int(ny), int(nx), int(nz), 3))

    del v 
    torch.cuda.empty_cache()

    # Initialize rhs
    rhs = torch.zeros((int(ny), int(nx), int(nz), 3), dtype=torch.float32, device=device)

    # Compute the right hand side of the equation
    for i in range(3):
        uf = torch.fft.fftn(u[..., i])
        dx_u = torch.real(torch.fft.ifftn(Gx * uf))
        dy_u = torch.real(torch.fft.ifftn(Gy * uf))
        dz_u = torch.real(torch.fft.ifftn(Gz * uf))
        rhs[..., i] = -dx_u * w[..., 0] - dy_u * w[..., 1] - dz_u * w[..., 2] + w[..., i]

    # Reshape rhs into a vector
    rhs = torch.reshape(rhs, (-1,))

    del u, w, uf, dx_u, dy_u, dz_u
    torch.cuda.empty_cache()

    return rhs
