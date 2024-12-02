import numpy as np
import torch
import torch.fft

def DetJacobianEquation3d_rhs(t, F, dummy, v, tt, nx, ny, nz, Gx, Gy, Gz):
    # Encontrar el índice de tiempo correspondiente
    tidx = np.where(t <= tt)[0][0]

    # Reformar F
    F = F.reshape((int(ny), int(nx), int(nz)))

    # Reformar y seleccionar v en el tiempo correspondiente
    w = v[:, tidx].reshape((int(ny), int(nx), int(nz), 3))

    # Transformada de Fourier y sus inversas
    vf = torch.fft.fftn(w[:, :, :, 0])
    dx_v1 = torch.real(torch.fft.ifftn(Gx * vf))

    vf = torch.fft.fftn(w[:, :, :, 1])
    dy_v2 = torch.real(torch.fft.ifftn(Gy * vf))

    vf = torch.fft.fftn(w[:, :, :, 2])
    dz_v3 = torch.real(torch.fft.ifftn(Gz * vf))

    rhs = F * (dx_v1 + dy_v2 + dz_v3)

    # Cálculos adicionales con transformadas inversas de Fourier
    dx = torch.real(torch.fft.ifftn(Gx * F))
    dy = torch.real(torch.fft.ifftn(Gy * F))
    dz = torch.real(torch.fft.ifftn(Gz * F))

    rhs = rhs - w[:, :, :, 0] * dx - w[:, :, :, 1] * dy - w[:, :, :, 2] * dz

    # Aplanar el resultado final
    rhs = rhs.flatten()

    return rhs
