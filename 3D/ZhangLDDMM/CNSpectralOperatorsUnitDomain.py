import numpy as np
import torch
import scipy.io

def CNSpectralOperatorsUnitDomain(nx, ny, nz, cn_gamma, cn_alpha, s):

    # Definir el dispositivo (GPU si está disponible, CPU en caso contrario)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Crear los vectores kx, ky, kz usando numpy y luego convertirlos a tensores de torch
    kx = 2 * np.pi * np.concatenate((np.arange(0, nx//2), np.arange(-nx//2, 0)))
    ky = 2 * np.pi * np.concatenate((np.arange(0, ny//2), np.arange(-ny//2, 0)))
    kz = 2 * np.pi * np.concatenate((np.arange(0, nz//2), np.arange(-nz//2, 0)))

    kx = torch.tensor(kx, dtype=torch.float32)
    ky = torch.tensor(ky, dtype=torch.float32)
    kz = torch.tensor(kz, dtype=torch.float32)

    # Crear los grid KX, KY, KZ usando torch
    KX, KY, KZ = torch.meshgrid(kx, ky, kz, indexing='ij')

    # Crear los operadores de gradiente
    Gx = 1j * KX
    Gy = 1j * KY
    Gz = 1j * KZ

    # Crear el operador de Cauchy-Navier
    L = (cn_gamma + cn_alpha * (KX**2 + KY**2 + KZ**2))**s

    # Discretización del dominio espacial
    hx = 1 / nx
    hy = 1 / ny
    hz = 1 / nz

    x = hx * torch.arange(0, nx)
    y = hy * torch.arange(0, ny)
    z = hz * torch.arange(0, nz)

    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')


    return L, Gx, Gy, Gz, X, Y, Z
