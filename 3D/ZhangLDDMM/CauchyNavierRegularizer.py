import numpy as np
from CNSpectralOperatorsUnitDomain import CNSpectralOperatorsUnitDomain
from CNFiniteDiffUnitDomain import CNFiniteDiffUnitDomain

def CauchyNavierRegularizer(cn_gamma, cn_alpha, s, gamma, nx, ny, nz):

    if gamma == 1:
        print('Incompressible fluid model')
        # Lap, _, _, _, _, _, _ = H2SpectralOperatorsUnitDomain(nx, ny, nz)
    else:
        print('Compressible fluid model')
        Lap = 0.0

    # Unit domain, spectral diff
    # L, Gx, Gy, Gz, X, Y, Z = CNSpectralOperatorsUnitDomain(nx, ny, nz, cn_gamma, cn_alpha, s)

    L, Gx, Gy, Gz, X, Y, Z = CNFiniteDiffUnitDomain(nx, ny, nz, cn_gamma, cn_alpha, s)

    # L, Gx, Gy, Gz, X, Y, Z = CNFiniteDiffNotUnitDomain(nx, ny, nz, cn_gamma, cn_alpha, s)


    # KK = L * L
    KK = L ** 2

    hxhyhz = nx * ny * nz

    return L, KK, Gx, Gy, Gz, Lap, X, Y, Z, nx, ny, nz, hxhyhz

