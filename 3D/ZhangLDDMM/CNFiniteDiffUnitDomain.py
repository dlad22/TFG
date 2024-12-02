import torch
import numpy as np

def CNFiniteDiffUnitDomain(nx, ny, nz, cn_gamma, cn_alpha, s):
    # ------------------------------------------------------------------------
    
    sX = 2.0 * np.pi / nx
    sY = 2.0 * np.pi / ny
    sZ = 2.0 * np.pi / nz
    
    x = torch.arange(-nx/2, nx/2, dtype=torch.float32)
    y = torch.arange(-ny/2, ny/2, dtype=torch.float32)
    z = torch.arange(-nz/2, nz/2, dtype=torch.float32)
    
    x[:int(nx)//2] += nx
    y[:int(ny)//2] += ny
    z[:int(nz)//2] += nz
    
    X, Y, Z = torch.meshgrid(x, y, z, indexing='xy')
    
    Gx = -1j * torch.sin(sX * X) * nx
    Gy = -1j * torch.sin(sY * Y) * ny
    Gz = -1j * torch.sin(sZ * Z) * nz
    
    # ------------------------------------------------------------------------
    
    xcoeff = 2.0 * torch.cos(sX * X) - 2.0
    ycoeff = 2.0 * torch.cos(sY * Y) - 2.0
    zcoeff = 2.0 * torch.cos(sZ * Z) - 2.0
    
    xcoeff = xcoeff * nx * nx
    ycoeff = ycoeff * ny * ny
    zcoeff = zcoeff * nz * nz
    
    L = (cn_gamma - cn_alpha * (xcoeff + ycoeff + zcoeff))**s
    
    L = torch.fft.ifftshift(L)
    
    # ------------------------------------------------------------------------
    
    hx = 1 / nx
    hy = 1 / ny
    hz = 1 / nz
    
    x = hx * torch.arange(0, nx, dtype=torch.float32)
    y = hy * torch.arange(0, ny, dtype=torch.float32)
    z = hz * torch.arange(0, nz, dtype=torch.float32)

    X, Y, Z = torch.meshgrid(x, y, z, indexing='xy')
    X = X.float()
    Y = Y.float()
    Z = Z.float()
    
    # ------------------------------------------------------------------------
    
    return L, Gx, Gy, Gz, X, Y, Z

