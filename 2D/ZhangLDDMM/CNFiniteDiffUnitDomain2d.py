import torch
import numpy as np

def CNFiniteDiffUnitDomain(nx, ny, cn_gamma, cn_alpha, s):
    # ------------------------------------------------------------------------
    
    sX = 2.0 * np.pi / nx
    sY = 2.0 * np.pi / ny
    
    x = torch.arange(-nx/2, nx/2, dtype=torch.float32)
    y = torch.arange(-ny/2, ny/2, dtype=torch.float32)
    
    x[:int(nx)//2] += nx
    y[:int(ny)//2] += ny
    
    X, Y = torch.meshgrid(x, y, indexing='xy')
    
    Gx = -1j * torch.sin(sX * X) * nx
    Gy = -1j * torch.sin(sY * Y) * ny
    
    # ------------------------------------------------------------------------
    
    xcoeff = 2.0 * torch.cos(sX * X) - 2.0
    ycoeff = 2.0 * torch.cos(sY * Y) - 2.0
    
    xcoeff = xcoeff * nx * nx
    ycoeff = ycoeff * ny * ny
    
    L = (cn_gamma - cn_alpha * (xcoeff + ycoeff))**s
    
    L = torch.fft.ifftshift(L)
    
    # ------------------------------------------------------------------------
    
    hx = 1 / nx
    hy = 1 / ny
    
    x = hx * torch.arange(0, nx, dtype=torch.float32)
    y = hy * torch.arange(0, ny, dtype=torch.float32)

    X, Y = torch.meshgrid(x, y, indexing='xy')
    X = X.float()
    Y = Y.float()
    
    # ------------------------------------------------------------------------
    
    return L, Gx, Gy, X, Y

