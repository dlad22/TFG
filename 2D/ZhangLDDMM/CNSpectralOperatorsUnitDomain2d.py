import numpy as np
import torch

def SpectralOperatorsUnitDomain(nx, ny, cn_gamma, cn_alpha, s):

    # Definir el dispositivo (GPU si está disponible, CPU en caso contrario)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Crear los vectores kx, ky, kz usando numpy y luego convertirlos a tensores de torch
    kx = 2 * np.pi * np.concatenate((np.arange(0, nx//2), np.arange(-nx//2, 0)))
    ky = 2 * np.pi * np.concatenate((np.arange(0, ny//2), np.arange(-ny//2, 0)))

    kx = torch.tensor(kx, dtype=torch.float32)
    ky = torch.tensor(ky, dtype=torch.float32)

    # Crear los grid KX, KY usando torch
    KX, KY = torch.meshgrid(kx, ky, indexing='ij')

    # Crear los operadores de gradiente
    Gx = 1j * KX
    Gy = 1j * KY

    # Crear el operador de Cauchy-Navier
    L = (cn_gamma + cn_alpha * (KX**2 + KY**2))**s

    # Discretización del dominio espacial
    hx = 1 / nx
    hy = 1 / ny

    x = hx * np.arange(0, nx)
    y = hy * np.arange(0, ny)

    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    X, Y = torch.meshgrid(x, y, indexing='ij')

    return L, Gx, Gy, X, Y
