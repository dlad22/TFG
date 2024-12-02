import numpy as np
import torch
import torch.fft

def DetJacobianEquation2d_rhs(t, F, dummy, v, tt, nx, ny, Gx, Gy):
    # Encontrar el índice de tiempo correspondiente
    tidx = np.where(t <= tt)[0][0]

    # Reformar F
    F = F.reshape((ny, nx))

    # Reformar y seleccionar v en el tiempo correspondiente
    w = v[:, tidx].reshape((ny, nx, 2))

    # Transformada de Fourier y sus inversas
    vf = torch.fft.fftn(w[:, :, 0])
    dx_v1 = torch.real(torch.fft.ifftn(Gx * vf))

    vf = torch.fft.fftn(w[:, :, 1])
    dy_v2 = torch.real(torch.fft.ifftn(Gy * vf))

    rhs = F * (dx_v1 + dy_v2)

    # Cálculos adicionales con transformadas inversas de Fourier
    dx = torch.real(torch.fft.ifftn(Gx * F))
    dy = torch.real(torch.fft.ifftn(Gy * F))

    rhs = rhs - w[:, :, 0] * dx - w[:, :, 1] * dy

    # Aplanar el resultado final
    rhs = rhs.flatten()

    return rhs
