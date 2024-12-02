"""
File Name: NIREPZhangGradientDescentMCCLDDMM_2018.py
Description: Brief description of what this file or module does.

Author: Alvaro de la Asuncion Deza
Original Author: Monica Hernandez
Creation Date: 2024-06-10
Last Modified: 2024-06-10 

This file is part of the ZhangLDDMM library, translated from MATLAB to Python.
Final Year Project at University of Zaragoza

Notes:
- Additional description if necessary.
- Information about dependencies if any.

"""

import os
import sys
import numpy as np
import scipy.ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.fft import fftn, ifftn
import time
from scipy.interpolate import RegularGridInterpolator
import nibabel as nib
import subprocess

sys.path.append(os.path.abspath('../Solvers'))
import ode3_gpu_memory2 as odegpu2
import ode3_memory2 as odemem
import ode3_gpu_memory3 as odegpu3

sys.path.append(os.path.abspath('../Vtk'))
from load_vtk_float import load_vtk_float

from clencurt import clencurt
from CauchyNavierRegularizer import CauchyNavierRegularizer
from UEPDiffEquation3d_rhs import UEPDiffEquation3d_rhs
from AdjointEPDiffEquation3d_rhs import AdjointEPDiffEquation3d_rhs
from EPDiffEquation3d_rhs import EPDiffEquation3d_rhs
from DeformationEquation3d_rhs import DeformationEquation3d_rhs
from DetJacobianEquation3d_rhs import DetJacobianEquation3d_rhs
from SpatialTransformer import SpatialTransformer



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
        

def ZhangGradientDescentMCCLDDMM_2018(patient, dataset, solver, exp_size='Large', nt=10, path_model=None, path_out=None, save_phi = True, Jac = False):
    # Memory management
    torch.cuda.empty_cache()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

    # Verify if CUDA is available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"CUDA is available. Using device: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Using CPU.")

    # (This parameter is never changed because execute without GPU is not possible)
    useGPU = True

    # txt_name = f"{path_out}resultados_solver_{solver}_dataset_{'OASIS' if dataset else 'NIREP'}_{exp_size}.txt"
    txt_name = "fotos.txt"

    # Load data
    if dataset == 0:
        data_dir = '/home/epdiff/python/3D/Data/NIREP/Subsample'
        moving, _, _, _ = load_vtk_float(os.path.join(data_dir, 'NIREP_01-Sub.vtk'))
        if patient < 10:
            fixed, _, _, _ = load_vtk_float(os.path.join(data_dir, f'NIREP_0{patient}-Sub.vtk'))
        else:
            fixed, _, _, _ = load_vtk_float(os.path.join(data_dir, f'NIREP_{patient}-Sub.vtk'))

    elif dataset == 1:
        data_dir = '/home/epdiff/python/3D/Data/OASIS'

        # Load movil and fixed images
        moving_f = f'{data_dir}/OASIS_{patient:04d}_0000.nii'
        print(moving_f)
        os.system(f'gunzip {moving_f}.gz')
        moving = nib.load(moving_f)
        moving = moving.get_fdata()
        os.system(f'gzip {moving_f}')

        fixed_f = f'{data_dir}/OASIS_{patient+1:04d}_0000.nii'
        os.system(f'gunzip {fixed_f}.gz')
        fixed = nib.load(fixed_f)
        fixed = fixed.get_fdata()
        os.system(f'gzip {fixed_f}')

    else:
        print('Dataset not found')
        exit(0)

    print('Dataset dir: ', data_dir)

    # --------------------------------------------------------------------------
    # Rescale between [0, 1]
    print('Rescaling between [0, 1]')

    Max = np.max(fixed)
    Min = np.min(fixed)
    fixed = (fixed - Min) / (Max - Min)

    Max = np.max(moving)
    Min = np.min(moving)
    moving = (moving - Min) / (Max - Min)
    # --------------------------------------------------------------------------

    if dataset == 0:
        print('Resizing images - ', exp_size)
        if(exp_size == 'Small'):
            # Size: 179     210     179

            fixed = fixed[:-1, :, :-1]
            moving = moving[:-1, :, :-1]
            # Size: 178     210     178

            fixed = fixed[::2, ::2, ::2]
            moving = moving[::2, ::2, ::2]
            # Size: 89     105     89

            fixed = np.pad(fixed, ((1, 1), (1, 1), (1, 1)), mode='constant')
            moving = np.pad(moving, ((1, 1), (1, 1), (1, 1)), mode='constant')

            fixed = fixed[:-1, :-1, :-1]
            moving = moving[:-1, :-1, :-1]
            # Size: 90     106     90

            # # Hacer que sean pares
            # fixed = fixed[:-1, :-1, :-1]
            # moving = moving[:-1, :-1, :-1]
        
        elif exp_size == 'Large':
            # Size: 179     210     179

            fixed = fixed[:-1, :, :-1]
            moving = moving[:-1, :, :-1]
            # Size: 178     210     178

            fixed = np.pad(fixed, ((1,1), (0,0), (1,1)), mode='constant')
            moving = np.pad(moving, ((1,1), (0,0), (1,1)), mode='constant')
            # Size: 180   210   180
        
    elif dataset == 1:
        if exp_size == 'Small':
            fixed = fixed[::2, ::2, ::2]
            moving = moving[::2, ::2, ::2]
        
            # Add padding to the images (10 pixels)
            padding = ((10, 10), (10, 10), (10, 10))  # (pad_depth, pad_height, pad_width)
            fixed = np.pad(fixed, padding, mode='constant', constant_values=0)
            moving = np.pad(moving, padding, mode='constant', constant_values=0)



        elif exp_size == 'Large':
            # Add padding to the images (10 pixels)
            padding = ((10, 10), (10, 10), (10, 10))  # (pad_depth, pad_height, pad_width)
            fixed = np.pad(fixed, padding, mode='constant', constant_values=0)
            moving = np.pad(moving, padding, mode='constant', constant_values=0)

        # fixed = fixed[::3, ::3, ::3]
        # moving = moving[::3, ::3, ::3]

        # fixed = fixed[:,:,:-1]
        # moving = moving[:,:,:-1]

        # Downsample

        # moving = moving[0::3, 0::3, 0::3]
        # fixed = fixed[0::3, 0::3, 0::3]

        # # Get the dimensions of sourcer
        # dim = list(moving.shape)

        # # Calculate the padding
        # pad = [d % 2 for d in dim]

        # # Pad the tensors
        # moving = np.pad(moving,  [(0, p) for p in pad], mode='constant', constant_values=0)
        # fixed = np.pad(fixed,  [(0, p) for p in pad], mode='constant', constant_values=0)

    print(f'Fixed shape: {fixed.shape}')

    # --------------------------------------------------------------------------
    
    # Convert to single precision
    source = moving.astype(np.float32)
    target = fixed.astype(np.float32)

    sum0 = np.sum((source - target) ** 2)

    # ------------------------------------------------------------------------

    # Get the size of the source array
    ny, nx, nz = source.shape

    nx = np.float32(nx)
    ny = np.float32(ny)
    nz = np.float32(nz)

    # Coursera spectral domain discretization only correct for even components
    print('WARNING: Code not yet working for odd dimensions')
    assert nx % 2 == 0 and ny % 2 == 0 and nz % 2 == 0, "Dimensions must be even"

    # ------------------------------------------------------------------------

    # Temporal domain discretization - Chebyshev-Gauss-Lobatto nodes
    t, w = clencurt(2*nt)

    t = t[:nt+1].astype(np.float32)
    w = w[:nt+1].astype(np.float32)

    ht = t[:-1] - t[1:]
    ht = torch.tensor(ht, dtype=torch.float32, device='cpu')
    T = t.size

    # ------------------------------------------------------------------------

    # Regularization Cauchy-Navier
    beta = 1.0 
    gamma = 0.0
    sigma = 1.0

    cn_gamma = 1.0
    cn_alpha = 0.0025
    s = 1

    L, KK, Gx, Gy, Gz, Lap, X, Y, Z, nx, ny, nz, hxhyhz = CauchyNavierRegularizer(cn_gamma, cn_alpha, s, gamma, nx, ny, nz)
    print('CauchyNavierRegularizer done')

    # ------------------------------------------------------------------------

    # Parameters

    # -- Stability 
    epsilon_CFL = 0.2

    # -- Convergence
    max_its = 50            # Number of outer iterations
    epsilon_mach = 1.0e-16  # Machine precision
    tau = 1.0e-3            # User defined tolerance. Desired accuracy of the inversion.

    # -- Desired properties of the mapping
    epsilon_F = 0.1         # Lower bound on det(phi) i.e. bound on the tolerable compression of a volume element

    # ------------------------------------------------------------------------

    # Energy initialization

    epsilon = 1.0e-2

    energy_0 = np.sum((source - target) ** 2) / nx / ny / nz
    print('Energy_0: ', energy_0)
    energy = energy_0

    im_energy_0 = energy_0
    im_energy = im_energy_0

    sum0 = np.sum((source - target) ** 2)

    energy_b = energy_0

    error1 = 1.0
    error1_b = 1.0

    history = np.empty((0, 16))
    history_all = []
    tolerances = []
    termination = []
    Armijo = []

    times = torch.empty((0, 2), dtype=torch.float32) 
    Jacs = torch.empty((0, 2), dtype=torch.float32)

    # ------------------------------------------------------------------------

    if useGPU:
        myode = odegpu2.ode3     
    else: 
        myode = odemem.ode3 

    myodestr = myode.__name__
    print(f'Using GPU solver: {myodestr}')

    # Transfer to GPU
    source = torch.tensor(source, dtype=torch.float32, device=device)
    target = torch.tensor(target, dtype=torch.float32, device=device)
    v0 = torch.zeros((int(ny), int(nx), int(nz), 3) , dtype=torch.float32, device=device)

    Gx = Gx.to(device)
    Gy = Gy.to(device)
    Gz = Gz.to(device)
    KK = KK.to(device)

    if gamma == 1:
        Lap = torch.tensor(Lap, dtype=torch.float32, device=device)

    X = X.to(device)
    Y = Y.to(device)
    Z = Z.to(device)

    phi = torch.zeros((int(ny), int(nx), int(nz), 3), dtype=torch.float32, device=device)

    # ------------------------------------------------------------------------

    # Velocity field initialization
    v = torch.empty((int(nx) * int(ny) * int(nz) * 3, T), dtype=torch.float32, device=device)
    print('Interpolating source from target using phi...')
    spatial_transformer = SpatialTransformer((int(nx), int(ny), int(nz)), mode='bilinear')
    
    source_tensor = source.unsqueeze(0).unsqueeze(0)  # Shape (1, 1, D, H, W)

    J0 = spatial_transformer(source_tensor, phi[..., 0], phi[..., 1], phi[..., 2])
    J0 = J0.squeeze(0).squeeze(0)
    J0 = J0.to(torch.float32)

    diff = J0.flatten() - target.flatten()
    error1 = torch.sum(diff ** 2) / sum0

    # ------------------------------------------------------------------------

    # Load model
    if solver != 0:
        model = torch.load(path_model, map_location=torch.device('cpu'))
        model = model.to(device)
        print('Model loaded')
    else:
        model = None
        print('Not using model')


    # ------------------------------------------------------------------------

    # Function to calculate the solution of the EPDiff equation
    def EPDiff_solution(solver, v0, model=None):
        if solver == 0:
            v = myode(EPDiffEquation3d_rhs, t, v0.flatten(), None, nx, ny, nz, Gx, Gy, Gz, KK, 1.0 / KK)
        elif solver == 1:   # Model FNO3D
            v0 = v0.unsqueeze(0)
            v = model(v0)
        elif solver == 2:   # Model FNO3D + [0, 1] Normalization
            # Normalize v0
            min_v0 = -0.1626
            max_v0 = 0.1780
            v0 = (v0 - min_v0) / (max_v0 - min_v0)

            v0 = v0.unsqueeze(0)
            v = model(v0)

            # Desnormalize v
            min_v = -0.1775
            max_v = 0.2891
            v = v * (max_v - min_v) + min_v
        elif solver == 3:   # Model FNO3D + [-1, 1] Normalization 
            # Normalize v0
            min_v0 = -0.1626
            max_v0 = 0.1780
            v0 = 2 * (v0 - min_v0) / (max_v0 - min_v0) - 1
            
            v0 = v0.unsqueeze(0)
            v = model(v0)

            # Desnormalize v
            min_v = -0.1775
            max_v = 0.2891
            v = ((v + 1) / 2)* (max_v - min_v) + min_v
        elif solver == 4:   # Model FNO3D + Time
            print('Model FNO3D + Time')
            v0 = v0.unsqueeze(0)
            v0 = v0.reshape(1,X.shape[0], X.shape[1], X.shape[2], 1, 3).repeat([1,1,1,1,T,1]) 
            v = model(v0)
        elif solver == 5:   # Model FNO3D + Time + [0, 1] Normalization
            print('Model FNO3D + Time + [0, 1] Normalization')
            # Normalize v0
            min_v0 = -0.1626
            max_v0 = 0.1780
            v0 = (v0 - min_v0) / (max_v0 - min_v0)

            v0 = v0.unsqueeze(0)
            v0 = v0.reshape(1,X.shape[0], X.shape[1], X.shape[2], 1, 3).repeat([1,1,1,1,T,1]) 
            v = model(v0)

            # Desnormalize v
            min_v = -0.8004
            max_v = 0.7694
            v = v * (max_v - min_v) + min_v
        elif solver == 6:   # Model FNO3D + Time + [-1, 1] Normalization
            print('Model FNO3D + Time + [0, 1] Normalization')
            # Normalize v0
            min_v0 = -0.1626
            max_v0 = 0.1780
            v0 = (v0 - min_v0) / (max_v0 - min_v0)

            v0 = v0.unsqueeze(0)
            v0 = v0.reshape(1,X.shape[0], X.shape[1], X.shape[2], 1, 3).repeat([1,1,1,1,T,1]) 
            v = model(v0)

            # Desnormalize v
            min_v = -1.5892
            max_v = 0.9934
            v = ((v + 1) / 2)* (max_v - min_v) + min_v
        else:
            print('Solver not found')
            exit(0)

        v = torch.reshape(v, (-1,nt+1))
        v = v.to(device)
        v0 = v0.squeeze(0)
        return v


    # ------------------------------------------------------------------------

    # OUTER ITERATION
    # Second order inexactly Newton-Kyrlov method

    start_all = time.time()

    print('Starting optimization loop')

    for it in range(1, max_its + 1):
        start_iter = time.time()
        print(f'Iteration {it}')

        # Compute the gradient of the energy in the fourier domain in t=1
        # E = 1/sigma^2*||J0 - target||^2 + beta/2 * ||v||^2
        # dE/dv = GRAD_v E(t=1) + beta * v
        # GRAD_v E(t=1) = K*(1/sigma^2 * (J0 - target) * GRAD(J0))
        mf = torch.fft.fftn(J0, dim=(-3, -2, -1))  # dim especifica las dimensiones sobre las cuales aplicar la FFT

        mx = torch.fft.ifftn(Gx * mf, dim=(-3, -2, -1)).real
        my = torch.fft.ifftn(Gy * mf, dim=(-3, -2, -1)).real
        mz = torch.fft.ifftn(Gz * mf, dim=(-3, -2, -1)).real

        #-------------------------------------------- GRADIENT IN V_1 --------------------------------------------------
        print('Calculating gradient in v_1')
        
        gradient_v_1 = torch.zeros((int(ny), int(nx), int(nz), 3), dtype=torch.float32, device=device)
        
        # the 2.0 appears when differentiating the energy squared
        gradient_v_1[..., 0] = 2.0 / sigma / sigma * (J0 - target) * mx
        gradient_v_1[..., 1] = 2.0 / sigma / sigma * (J0 - target) * my
        gradient_v_1[..., 2] = 2.0 / sigma / sigma * (J0 - target) * mz

        gradient_v_1[..., 0] = torch.fft.ifftn(torch.fft.fftn(gradient_v_1[..., 0]) / KK, dim=(-3, -2, -1)).real
        gradient_v_1[..., 1] = torch.fft.ifftn(torch.fft.fftn(gradient_v_1[..., 1]) / KK, dim=(-3, -2, -1)).real
        gradient_v_1[..., 2] = torch.fft.ifftn(torch.fft.fftn(gradient_v_1[..., 2]) / KK, dim=(-3, -2, -1)).real

        del mf, mx, my, mz
        torch.cuda.empty_cache()

        #-------------------------------------------- JACOBI EQUATIONS  ------------------------------------------------
        # Solve the Adjoint Jacobi field equation, U(t=1) = GRAD_v0 E(t=0)
        print('Calculating Jacobian equations')

        # dU/dt = -adj^t_v * U
        U1 = myode(UEPDiffEquation3d_rhs, t, gradient_v_1.flatten(), None, v, t, nx, ny, nz, Gx, Gy, Gz, KK, 1.0 / KK)
        U1 = U1.to(device)

        del gradient_v_1
        torch.cuda.empty_cache()

        nu = torch.zeros((int(ny), int(nx), int(nz), 3), dtype=torch.float32, device=device)

        # dw/dt = -U + adj_v * w - adj^t_w * v
        h_gorro = odegpu3.ode3(AdjointEPDiffEquation3d_rhs, t, nu.reshape(-1), None, v, U1, t, nx, ny, nz, Gx, Gy, Gz, KK, 1.0 / KK)
        h_gorro = h_gorro.to(device)

        #-------------------------------------------- GRADIENT IN V_0 --------------------------------------------------
        print('Calculating gradient in v_0')

        h_gorro = h_gorro.reshape(int(ny), int(nx), int(nz), 3)
        
        gradient_v_0 = v0 + h_gorro
        
        gV_inf = torch.max(torch.abs(gradient_v_0))
        print(f"Maximum gradient in v_0 ... {gV_inf}")
                
        # In the first iteration g_0 = g
        if it == 1:
            gV_inf_0 = gV_inf

        del h_gorro, U1, nu
        torch.cuda.empty_cache()

        # -------------------------------------------- GRADIENT DESCEND ------------------------------------------------
        print('Gradient descend')

        v0b = v0
        v0 = v0 - epsilon * gradient_v_0

        print('Maximo v0 ...', torch.max(torch.abs(v0)))

        del gradient_v_0
        torch.cuda.empty_cache()

        # -------------------------------------- EPDIFF & DEFORMATION EQUATION -----------------------------------------
        # -- Geodesic shooting
        print('Calculating EPDiff and Deformation equations')

        del v
        torch.cuda.empty_cache()

        v = EPDiff_solution(solver, v0, model)

        u = torch.zeros((int(ny), int(nx), int(nz), 3), dtype=torch.float32, device=device)
        u = odegpu3.ode3(DeformationEquation3d_rhs, t, u.flatten(), None, -v, t, nx, ny, nz, Gx, Gy, Gz)

        u = u.to(device)

        u = u.reshape(int(ny), int(nx), int(nz), 3)

        phi[..., 0] = - u[..., 0]
        phi[..., 1] = - u[..., 1]
        phi[..., 2] = - u[..., 2]

        print(f"Maximum v ... {torch.max(torch.abs(v))}")

        print(f"Maximum phi ... {torch.max(torch.abs(phi))}")

        # Memory management
        torch.cuda.empty_cache()

        # ------------------------------------------ STABILITY CHECK ---------------------------------------------------
        print('Checking stability')
        # Check numerical stability of the state and adjoint equations
        # Calculating den1, den2, den3
        den1 = torch.max(torch.max(torch.abs(v[:int(nx)*int(ny)*int(nz)]))) * nx
        den2 = torch.max(torch.max(torch.abs(v[int(nx)*int(ny)*int(nz):2*int(nx)*int(ny)*int(nz)]))) * ny
        den3 = torch.max(torch.max(torch.abs(v[2*int(nx)*int(ny)*int(nz):]))) * nz

        # Finding the maximum of den1, den2, den3
        Max = torch.max(torch.tensor([den1, den2, den3], device='cpu'))

        del den1, den2, den3
        torch.cuda.empty_cache()

        # Calculating h_tmax
        h_tmax = epsilon_CFL / Max

        # Printing CFL condition
        print(f"\tCFL condition: theoretical bound {1 / Max}")
        print(f"\tCFL condition violated?: max {h_tmax} >>> ht {torch.max(ht)} ?")

        # Checking violation condition
        if h_tmax >= torch.max(ht):
            viol = False
            soft_viol = False
            print("\t\tCFL condition NOT violated")
        else:
            viol = True
            soft_viol = True
            print("\t\tCFL condition violated")
            if 1 / Max >= torch.max(ht):
                soft_viol = False
                print("\t\tBUT soft CFL condition NOT violated")

        # Calculating CFL number
        CFL_number = torch.max(ht) * Max
        print(f"\tCFL number {CFL_number}")

        # assert h_tmax >= torch.max(ht)  # Uncomment if assertion is necessary

        # ------------------------------------------ INTERPOLATION -----------------------------------------------------   
        print('Interpolating source from target using phi')
    
        J0 = spatial_transformer(source_tensor, phi[..., 0], phi[..., 1], phi[..., 2])
        J0 = J0.squeeze(0).squeeze(0)
        J0 = J0.to(torch.float32)

        reg_energy = 0.0  # NonStationaryRegEnergy3d(v, nc, nx, ny, nz, KK)

        # Compute energy and image energy
        diff = J0.flatten() - target.flatten()
        im_energyb = im_energy
        im_energy = torch.sum(diff ** 2) / (nx * ny * nz)
        print(f"Im_energy... {im_energy}")
        energyb = energy
        energy = beta * reg_energy + im_energy

        # Calculate error and gradient information
        error1 = torch.sum(diff ** 2) / sum0
        gV_inf_ratio = gV_inf / gV_inf_0

        # Print results
        print(f"Error... {error1}")
        print(f"Gradient... {gV_inf_ratio}")

        # ------------------------------------------ TERMINATION CRITERIA ----------------------------------------------
        print('Checking termination criteria')

        C1 = energy_b - energy < tau * (1 + energy_0)
        C2 = torch.max(torch.abs(v0b - v0)) < np.sqrt(tau) * (1 + torch.max(torch.abs(v0)))
        C3 = gV_inf < tau**(1/3) * (1 + energy_0)
        C4 = gV_inf < 1.e3 * epsilon_mach   
        C5 = energy_b <= energy             # Comprueba que la energia disminuye
        C6 = error1_b <= error1             # Comprueba que el error disminuye
        C7 = epsilon <= 1.0e-10             # Comprueba que epsilon es menor que 1.0e-10

        print(f"Energy... {energy}")

        del u
        torch.cuda.empty_cache()        

        # termination_criteria = (C1 or C2 or C3 or C4) and C5 and C6 and C7

        if C1:
            print('C1 holds')

        if C2:
            print('C2 holds')

        if C3:
            print('C3 holds')

        if C4:
            print('C4 holds')

        if C5:
            print('C5 holds')
            
        if C6:
            print('C6 holds')

        if C7:
            print('C7 holds')

        time_iter = time.time() - start_iter

        history_all = np.array([it, epsilon, energy, reg_energy, im_energy, gV_inf,
                                energy / energy_0, im_energy / im_energyb, gV_inf / gV_inf_0,
                                error1, viol, soft_viol, h_tmax, 1 / Max, CFL_number, time_iter])

        times = torch.cat((times, torch.tensor([[it, time_iter]])), dim=0)
        # print(f"History all: {history_all}")

        # -- Añadir a un fichero de texto
        if it == 1:                                     # Escribir la cabecera
            with open(txt_name, 'a') as file:    # Abrir el fichero en modo adición
                file.write('it	epsiln energy reg_e  im_e   gV_inf relEn0 relIm0 relGV0 relErr v_CFL  soft_v h_tmax 1/Max  CFL_n  time_iter  \n')

        with open(txt_name, 'a') as file:        # Escribir los datos del history
            for val in history_all:
                file.write(f"{val:.4f} ")
            file.write('\n')
    
        if torch.isnan(error1) or C6 or C7:
            print('Termination criteria met')
            v0 = v0b

            del v
            torch.cuda.empty_cache()
            
            v = EPDiff_solution(solver, v0, model)

            # ------

            u = torch.zeros((int(ny), int(nx), int(nz), 3), dtype=torch.float32, device=device)
            u = odegpu3.ode3(DeformationEquation3d_rhs, t, u.flatten(), None, -v, t, nx, ny, nz, Gx, Gy, Gz)
            u = u.to(device)
            u = u.reshape(int(ny), int(nx), int(nz), 3)
            
            phi[..., 0] = - u[..., 0]
            phi[..., 1] = - u[..., 1]
            phi[..., 2] = - u[..., 2]

            del u
            torch.cuda.empty_cache()
            
            # -----

            J0 = spatial_transformer(source_tensor, phi[..., 0], phi[..., 1], phi[..., 2])
            J0 = J0.squeeze(0).squeeze(0)
            J0 = J0.to(torch.float32)

            # -----
            
            epsilon = epsilon * 0.1
            
            if epsilon < 1.0e-6:
                break
        else:
            print('Termination criteria not met')
            history = np.vstack((history, [it, epsilon, energy, reg_energy, im_energy, gV_inf,
                                            energy / energy_0, im_energy / im_energy_0, gV_inf / gV_inf_0,
                                            error1, viol, soft_viol, h_tmax, 1 / Max, CFL_number, time_iter]))
            error1_b = error1
            energy_b = energy


        # history = gather(history)  # Assuming gather is used to synchronize data in a GPU context
        # print('iteration \tepsilon \tenergy \treg_energy \tim_energy \tgV_inf \trel_en_0 \trel_im_0 \trel_gV_0 \trel_err \tviol_CFL \tsoft_viol \th_tmax \t1/Max \tCFL_number \ttime_iter \n')

        # for row in history:
        #     for val in row:
        #         print(f"{val}\t", end="")
        #     print('\n')

    time_all = time.time() - start_all
    print(f'Total time {time_all}\n')
    
    with open(txt_name, 'a') as file:        # Escribir los datos del history
        file.write(f"{time_all:.8f} ")
        file.write('\n')

    # Guardar phi
    if save_phi:
        if dataset == 0:
            phi_name = path_out + 'phi_moving_1_fixed_' + str(patient) + '.mat'
        elif dataset == 1:
            phi_name = path_out + 'phi_moving' + str(patient) + '_fixed' + str(patient + 1) + '.mat'
        scipy.io.savemat(phi_name, {'phi': phi.cpu().numpy()})
        print('Phi save: ', phi_name)


    # ------------------------------------------------------------------------
    if Jac:
        try:
            v = EPDiff_solution(solver, v0, model)
            
            if isinstance(v, np.ndarray):
                F = np.ones((int(ny), int(nx), int(nz)), dtype=np.float32)
            else:
                F = torch.ones((int(ny), int(nx), int(nz)), dtype=torch.float32, device=device)
            
            print('Calculating Jacobian')
            Fsol = myode(DetJacobianEquation3d_rhs, t, F.flatten(), None, -v, t, nx, ny, nz, Gx, Gy, Gz)
            Fsol = Fsol.to(device)
            J = Fsol[:, -1].reshape(int(ny), int(nx), int(nz))
            cut = 5
            J = J[cut:-cut+1, cut:-cut+1, cut:-cut+1]
            
            Jacs = torch.cat((Jacs, torch.tensor([[torch.max(J), torch.min(J)]])), dim=0)

            print('Jacobian calculated')
            
        except Exception as e:
            print('Error in Jacobian')
            print(e)

    # ----------------------------------------------------------------------

    # Guardar la imagen resultante (J0)
    # data_dict = {'J0': J0.cpu().numpy(), 'target': target.cpu().numpy(), 'source': source.cpu().numpy()}
    # name = f'images_{solver}_{dataset}.mat'
    # scipy.io.savemat(name, data_dict)
    # print('Imágenes guardadas')

    
    print('Returning results')
    return v0, history, history_all, Jacs, times, time_all











