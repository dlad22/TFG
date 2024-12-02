import numpy as np
import torch

"""
Parameters:
    v: velocity field
    w: another field
    Gx, Gy, Gz: gradient operators
Returns:
    advw: resulting field after applying the operations
"""
def ad(v, w, Gx, Gy, Gz):
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

    Gx, Gy, Gz = map(to_tensor, [Gx, Gy, Gz])

    advw = torch.zeros_like(v, device=device)

    # 1. Dft vft * wft

    vft = torch.fft.fftn(v[..., 0])

    dx = torch.fft.ifftn(vft * Gx).real
    dy = torch.fft.ifftn(vft * Gy).real
    dz = torch.fft.ifftn(vft * Gz).real

    advw[..., 0] = dx * w[..., 0] + dy * w[..., 1] + dz * w[..., 2]

    vft = torch.fft.fftn(v[..., 1])

    dx = torch.fft.ifftn(vft * Gx).real
    dy = torch.fft.ifftn(vft * Gy).real
    dz = torch.fft.ifftn(vft * Gz).real

    advw[..., 1] = dx * w[..., 0] + dy * w[..., 1] + dz * w[..., 2]

    vft = torch.fft.fftn(v[..., 2])

    dx = torch.fft.ifftn(vft * Gx).real
    dy = torch.fft.ifftn(vft * Gy).real
    dz = torch.fft.ifftn(vft * Gz).real

    advw[..., 2] = dx * w[..., 0] + dy * w[..., 1] + dz * w[..., 2]

    # 2. Dft wft * vft

    wft = torch.fft.fftn(w[..., 0])

    dx = torch.fft.ifftn(wft * Gx).real
    dy = torch.fft.ifftn(wft * Gy).real
    dz = torch.fft.ifftn(wft * Gz).real

    advw[..., 0] -= dx * v[..., 0] + dy * v[..., 1] + dz * v[..., 2]

    wft = torch.fft.fftn(w[..., 1])

    dx = torch.fft.ifftn(wft * Gx).real
    dy = torch.fft.ifftn(wft * Gy).real
    dz = torch.fft.ifftn(wft * Gz).real

    advw[..., 1] -= dx * v[..., 0] + dy * v[..., 1] + dz * v[..., 2]

    wft = torch.fft.fftn(w[..., 2])

    dx = torch.fft.ifftn(wft * Gx).real
    dy = torch.fft.ifftn(wft * Gy).real
    dz = torch.fft.ifftn(wft * Gz).real

    advw[..., 2] -= dx * v[..., 0] + dy * v[..., 1] + dz * v[..., 2]

    return advw
