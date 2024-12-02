import numpy as np
from CNSpectralOperatorsUnitDomain2d import SpectralOperatorsUnitDomain
from CNFiniteDiffUnitDomain2d import CNFiniteDiffUnitDomain

def CauchyNavierRegularizer(cn_gamma, cn_alpha, s, gamma, nx, ny):
    print('CN regularization')

    if gamma == 1:
        print('Incompressible fluid model')
        # Lap, _, _, _, _, _, _ = H2SpectralOperatorsUnitDomain(nx, ny, nz)
    else:
        print('Compressible fluid model')
        Lap = 0.0

    # Unit domain, spectral diff
    # L, Gx, Gy, X, Y, = SpectralOperatorsUnitDomain(nx, ny, cn_gamma, cn_alpha, s)

    L, Gx, Gy, X, Y = CNFiniteDiffUnitDomain(nx, ny, cn_gamma, cn_alpha, s)

    # L, Gx, Gy, X, Y = CNFiniteDiffNotUnitDomain(nx, ny, cn_gamma, cn_alpha, s)


    # KK = L * L
    KK = L ** 2

    hxhy = nx * ny

    return L, KK, Gx, Gy, Lap, X, Y, nx, ny, hxhy

