import numpy as np
import utils as ut 
import torch

def ode3(odefun, tspan, y0, options=None, *args):
    """
    Solve differential equations with a non-adaptive method of order 3.

    Parameters:
    odefun : function handle
        Function of the form f(t, y, *args) where t is the current time, y is the current state,
        and *args are additional parameters.
    tspan : array-like
        Time span over which to integrate, [T1, T2, ..., TN].
    y0 : array-like
        Initial conditions at tspan[0].
    options : dict, optional
        Additional options (not used in this basic implementation).
    *args : tuple
        Additional arguments to pass to odefun.

    Returns:
    -------
    Y : numpy.ndarray
        Solution array where each column corresponds to a time specified in tspan.

    Raises:
    ------
    ValueError
        If tspan is not numeric, y0 is not numeric, or sizes of y0 and f(t0, y0) are inconsistent.
    """
    if not ut.is_valid_type(tspan):
        raise ValueError('TSPAN should be a vector of integration steps.')
    if not ut.is_valid_type(y0):
        raise ValueError('Y0 should be a vector of initial conditions.')

    h = np.diff(tspan)
    if any(np.sign(h[0]) * h <= 0):
        raise ValueError('Entries of TSPAN are not in order.')
    
    try:
        f0 = odefun(tspan[0], y0, *args)
    except Exception as e:
        msg = f'Unable to evaluate the ODEFUN at t0, y0. {str(e)}'
        raise ValueError(msg)

    # y0 = np.reshape(y0, (-1, 1))  # Make y0 a column vector
    if y0.shape != f0.shape:
        raise ValueError('Inconsistent sizes of Y0 and f(t0, y0).')

    neq = len(y0)
    N = len(tspan)
    Y = np.zeros((neq, N), dtype=np.single, device=y0.device)  # Initialize solution array
    F = np.zeros((neq, 3), dtype=np.single, device=y0.device)

    Y[:, 0] = y0.clone().detach()  # Initial condition

    for i in range(1, N):
        ti = tspan[i - 1]
        hi = h[i - 1]
        yi = Y[:, i - 1]
        F[:, 0] = odefun(ti, yi, None, *args).clone().detach()
        F[:, 1] = odefun(ti + 0.5 * hi, yi + 0.5 * hi * F[:, 0], None, *args).clone().detach()
        F[:, 2] = odefun(ti + 0.75 * hi, yi + 0.75 * hi * F[:, 1], None, *args).clone().detach()
        Y[:, i] = yi + (hi / 9) * (2 * F[:, 0] + 3 * F[:, 1] + 4 * F[:, 2])

    return Y