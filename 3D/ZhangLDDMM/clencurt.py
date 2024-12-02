import numpy as np

def clencurt(N):
    theta = np.pi * np.arange(N+1) / N
    x = np.cos(theta)
    w = np.zeros(N+1)
    ii = np.arange(1, N)
    v = np.ones(N-1)
    
    if N % 2 == 0:
        w[0] = 1 / (N**2 - 1)
        w[N] = w[0]
        for k in range(1, N//2):
            v = v - 2 * np.cos(2 * k * theta[ii]) / (4 * k**2 - 1)
        v = v - np.cos(N * theta[ii]) / (N**2 - 1)
    else:
        w[0] = 1 / N**2
        w[N] = w[0]
        for k in range(1, (N+1)//2):
            v = v - 2 * np.cos(2 * k * theta[ii]) / (4 * k**2 - 1)
    
    w[ii] = 2 * v / N
    return x, w
