import numpy as np
import torch
import gc

"""
Purpose: Computation of adTranspose operator using Zhang et al simplification
    dv / dt = - K ( Dv' m + Dm v + m div v ) - original EPDiff equation
    dv / dt = - K ( Dv' m + div ( m cross v ) ) - Zhang et al

Parameters:
    v: velocity field
    w: another field
    L: Laplacian operator
    K: inverse of the Laplacian operator
    Gx, Gy, Gz: gradient operators
"""
def adTranspose(v, w, L, K, Gx, Gy):
    # Determine if using GPU
    if isinstance(v, torch.Tensor):
        device = v.device
    else:
        device = torch.device("cpu")

    
    # Convert numpy arrays to torch tensors if needed and move them to the same device
    def to_tensor(arr):
        if not isinstance(arr, torch.Tensor):
            return torch.tensor(arr, dtype=torch.float32, device=arr.device if hasattr(arr, 'device') else 'cpu')
        
        return arr

    L, K, Gx, Gy = map(to_tensor, [L, K, Gx, Gy])

    # 1. m = L w, momentum
    m = torch.zeros_like(w, device=device)
    m[..., 0] = torch.fft.ifftn(torch.fft.fftn(w[..., 0]) * L).real
    m[..., 1] = torch.fft.ifftn(torch.fft.fftn(w[..., 1]) * L).real


    # 2. (D v)T * m
    adTvw = torch.zeros_like(v, device=device)
    
    vft = torch.fft.fftn(v[..., 0])
    adTvw[..., 0] = torch.fft.ifftn(vft * Gx).real * m[..., 0]
    adTvw[..., 1] = torch.fft.ifftn(vft * Gy).real * m[..., 0]

    vft = torch.fft.fftn(v[..., 1])
    adTvw[..., 0] += torch.fft.ifftn(vft * Gx).real * m[..., 1]
    adTvw[..., 1] += torch.fft.ifftn(vft * Gy).real * m[..., 1]

    # 3. div (m * v)
    pft = torch.fft.fftn(m[..., 0] * v[..., 0])
    adTvw[..., 0] += torch.fft.ifftn(pft * Gx).real

    pft = torch.fft.fftn(m[..., 0] * v[..., 1])
    adTvw[..., 0] += torch.fft.ifftn(pft * Gy).real

    # -- 

    pft = torch.fft.fftn(m[..., 1] * v[..., 0])
    adTvw[..., 1] += torch.fft.ifftn(pft * Gx).real

    pft = torch.fft.fftn(m[..., 1] * v[..., 1])
    adTvw[..., 1] += torch.fft.ifftn(pft * Gy).real

    # 5. -K adTvw (note: FlashC lo tiene positivo)
    adTvw[..., 0] = torch.fft.ifftn(K * torch.fft.fftn(adTvw[..., 0])).real
    adTvw[..., 1] = torch.fft.ifftn(K * torch.fft.fftn(adTvw[..., 1])).real

    # Delete temporary variables from gpu memory
    del m
    # Forzar la recolección de basura
    gc.collect()
    # Vaciar la caché de la GPU
    torch.cuda.empty_cache()


    return adTvw
