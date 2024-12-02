import numpy as np
import torch
import utils as ut
import gc

def ode3(odefun, tspan, y0, options, *args):
    """
    ODE3 Solve differential equations with a non-adaptive method of order 3.
    Y = ODE3(ODEFUN, TSPAN, Y0) with TSPAN = [T1, T2, T3, ... TN] integrates
    the system of differential equations y' = f(t,y) by stepping from T0 to
    T1 to TN. Function ODEFUN(T,Y) must return f(t,y) in a column vector.
    The vector Y0 is the initial conditions at T0. Each row in the solution
    array Y corresponds to a time specified in TSPAN.
    
    Y = ODE3(ODEFUN, TSPAN, Y0, P1, P2, ...) passes the additional parameters
    P1, P2, ... to the derivative function as ODEFUN(T, Y, P1, P2, ...).
    
    This is a non-adaptive solver. The step sequence is determined by TSPAN
    but the derivative function ODEFUN is evaluated multiple times per step.
    The solver implements the Bogacki-Shampine Runge-Kutta method of order 3.
    """

    # Seleccionar la GPU 3 si está disponible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # cuda:3 si ejecuo en HULKA

    # Mover parametros a la GPU 2
    y0 = y0.to(device)
    args = [arg.to(device) if hasattr(arg, 'to') else arg for arg in args]

    if not ut.is_valid_type(tspan):
        raise ValueError('TSPAN should be a vector of integration steps.')
    if not ut.is_valid_type(y0):
        raise ValueError('Y0 should be a vector of initial conditions.')


    # tspan = torch.tensor(tspan, dtype=torch.float32)
    # y0 = torch.tensor(y0, dtype=torch.float32)

    h = np.diff(tspan)
    if any(np.sign(h[0]) * h <= 0):
        raise ValueError('Entries of TSPAN are not in order.')
    
    try:
        f0 = odefun(tspan[0], y0, options, *args)
    except Exception as e:
        raise ValueError(f'Unable to evaluate the ODEFUN at t0, y0. {str(e)}')
    
    # y0 = y0.reshape(-1)  # Make a column vector
    if y0.shape != f0.shape:
        raise ValueError('Inconsistent sizes of Y0 and f(t0, y0).')
    
    neq = len(y0)
    N = len(tspan)
    
    Y = torch.zeros((neq, 1), dtype=torch.float32, device=device)
    F = torch.zeros((neq, 3), dtype=torch.float32, device=device)
    
    Y = y0.clone().detach()

    for i in range(1, N):
        ti = tspan[i - 1]
        hi = h[i - 1]
        yi = Y
        
        F[:, 0] = odefun(ti, yi, options, *args).clone().detach()
        F[:, 1] = odefun(ti + 0.5 * hi, yi + 0.5 * hi * F[:, 0], options, *args).clone().detach()
        F[:, 2] = odefun(ti + 0.75 * hi, yi + 0.75 * hi * F[:, 1], options, *args).clone().detach()
        
        Y = yi + (hi / 9) * (2 * F[:, 0] + 3 * F[:, 1] + 4 * F[:, 2])

    # Delete temporal variables from gpu memory
    del F, yi, hi, ti, f0, args, y0
    # Forzar la recolección de basura
    gc.collect()
    # Vaciar la caché de la GPU
    torch.cuda.empty_cache()

    return Y

