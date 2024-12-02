import numpy as np
import torch
from adTranspose import adTranspose 
import subprocess
import gc

def get_gpu_info():
    # Ejecutar el comando nvidia-smi
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used,temperature.gpu', '--format=csv,noheader,nounits'], stdout=subprocess.PIPE)
    # Convertir la salida en una lista de strings
    output = result.stdout.decode('utf-8').strip().split('\n')
    
    for line in output:
        fields = line.split(',')
        print(f"GPU ID: {fields[0]}")
        print(f"Nombre: {fields[1]}")
        print(f"Uso de la GPU: {fields[2]}%")
        print(f"Uso de la memoria: {fields[3]}%")
        print(f"Memoria total: {fields[4]} MB")
        print(f"Memoria libre: {fields[5]} MB")
        print(f"Memoria utilizada: {fields[6]} MB")
        print(f"Temperatura: {fields[7]} C")
        print("-" * 20)
        break

"""
Parameters:
    t: actual time
    U: vector with the variable of interest (velocities) at a specific time
    dummy: not used, just to match the signature
    v: matrix with the velocities at all times
    tt: discretized time vector
    nx, ny, nz: size of the domain in each direction
    Gx, Gy, Gz: gradient operators (matrix)
    L: Laplacian operator
    K: inverse of the Laplacian operator
"""
def UEPDiffEquation3d_rhs(t, U, dummy, v, tt, nx, ny, nz, Gx, Gy, Gz, L, K):
    # Determine if using GPU
    if isinstance(U, torch.Tensor):
        device = U.device
    else:
        device = torch.device("cpu")
    
    # Convert numpy arrays to torch tensors if needed and move them to the same device
    def to_tensor(arr):
        if not isinstance(arr, torch.Tensor):
            return torch.tensor(arr, dtype=torch.float32, device=arr.device if hasattr(arr, 'device') else 'cpu')
        return arr

    Gx, Gy, Gz, L, K, tt = map(to_tensor, [Gx, Gy, Gz, L, K, tt])

    # Take the first time step that is greater than the actual time
    tidx = torch.where(t >= tt)[0]
    tidx = tidx[0].item()  # Convert to Python scalar

    # Reshape the vector U into a 4D matrix (ny, nx, nz, 3)
    U = torch.reshape(U, (int(ny), int(nx), int(nz), 3))

    # Delete the column tidx and reshape the matrix v into a 4D matrix (ny, nx, nz, 3)
    w = torch.reshape(v[:, tidx], (int(ny), int(nx), int(nz), 3))

    # Compute the right hand side of the equation
    rhs = -adTranspose(w, U, L, K, Gx, Gy, Gz)

    # Reshape the matrix rhs into a vector
    rhs = torch.reshape(rhs, (-1,))

    # Delete temporary variables from gpu memory
    del w, v
    # Forzar la recolección de basura
    gc.collect()
    # Vaciar la caché de la GPU
    torch.cuda.empty_cache()


    # print("\tUEPDiffEquation3d_rhs finished: " + str(rhs.shape) + " " + str(rhs.device))    
    return rhs
