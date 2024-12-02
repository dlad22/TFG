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
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from clencurt import clencurt

from CauchyNavierRegularizer2d import CauchyNavierRegularizer
from UEPDiffEquation2d_rhs import UEPDiffEquation2d_rhs
from AdjointEPDiffEquation2d_rhs import AdjointEPDiffEquation2d_rhs
from EPDiffEquation2d_rhs import EPDiffEquation2d_rhs
from DeformationEquation2d_rhs import DeformationEquation2d_rhs
from DetJacobianEquation2d_rhs import DetJacobianEquation2d_rhs

sys.path.append('/content/drive/MyDrive/TFG/2D/Solvers')
sys.path.append(os.path.abspath('../Solvers'))

import ode3_gpu_memory2 as odegpu2
import ode3_memory2 as odemem
import ode3_gpu_memory3 as odegpu3

sys.path.append(os.path.abspath('../Vtk'))
# from load_vtk_float import load_vtk_float

from SpatialTransformer2d import SpatialTransformer

sys.path.append('/content/drive/MyDrive/TFG/2D/NeurEPDiff')
sys.path.append(os.path.abspath('../NeurEPDiff'))
# from EPDiffFourierNeuralOperator_2D import FNO2d
# from EPDiffFourierNeuralOperator_2D import SpectralConv2d_fast
from utilities3 import *


import subprocess
      

def NIREPZhangGradientDescentMCCLDDMM_2d(patient_ini, patient_fin, solver, path_model):
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

    # Verificar si CUDA está disponible
    if torch.cuda.is_available():
        device = torch.device("cuda") # cuda:0
        print(f"CUDA is available. Using device: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Using CPU.")

    useGPU = True

    LOCAL = False
    HULKA = False
    GOOGLE = False
    FRACTAL = False
    SAURON = True

    # Elección modo de calcular EPDiff
    mode = 1
    txt_name = 'nada.txt'


    # Load BULLEYES data
    if LOCAL:
        BULLEYES_dir = 'D:/TFG/HULKA-MONICA/DataGenerators/BullEyes/Images'
    elif HULKA:
        BULLEYES_dir = '/home/epdiff/DataGenerators/BullEyes/Images'
    elif GOOGLE:
        BULLEYES_dir = '/content/drive/MyDrive/TFG/2D/Data/BullEyes/Images'
    elif FRACTAL:
        BULLEYES_dir = '/home/fno/2D/Data/BullEyes/Images'
    elif SAURON:
        BULLEYES_dir = '/home/epdiff/python/2D/Data/BullEyes'

    print('BULLEYES_dir: ', BULLEYES_dir )

    data_moving = scipy.io.loadmat(os.path.join(BULLEYES_dir, f'BullEye_{patient_ini}.mat'))
    moving = data_moving['bulleye']

    data_fixed = scipy.io.loadmat(os.path.join(BULLEYES_dir, f'BullEye_{patient_fin}.mat'))
    fixed = data_fixed['bulleye'] 

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
    
    # Convert to single precision
    source = moving.astype(np.float32)
    target = fixed.astype(np.float32)

    # ------------------------------------------------------------------------

    # Get the size of the source array
    ny, nx = source.shape

    nx = np.float32(nx)
    ny = np.float32(ny)

    # Coursera spectral domain discretization only correct for even components
    print('WARNING: Code not yet working for odd dimensions')
    assert nx % 2 == 0 and ny % 2 == 0, "Dimensions must be even"

    # ------------------------------------------------------------------------

    # Temporal domain discretization - Chebyshev-Gauss-Lobatto nodes
    nt = 30     # Time discretization for ODE solver

    t, w = clencurt(2*nt)

    t = t[:nt+1].astype(np.float32)
    w = w[:nt+1].astype(np.float32)

    ht = t[:-1] - t[1:]
    ht = torch.tensor(ht, dtype=torch.float32, device='cpu')
    T = t.size

    # ------------------------------------------------------------------------

    # Regularization Cauchy-Navier

    beta = 1.0 # / 255.0 / 255.0
    gamma = 0.0
    sigma = 1.0

    cn_gamma = 1.0
    cn_alpha = 0.0025
    s = 1

    L, KK, Gx, Gy, Lap, X, Y, nx, ny, hxhy = CauchyNavierRegularizer(cn_gamma, cn_alpha, s, gamma, nx, ny)
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

    energy_0 = np.sum((source - target) ** 2) / nx / ny
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

    times = torch.empty((0, 2), dtype=torch.float32) 
    Jacs = torch.empty((0, 2), dtype=torch.float32)


    # ------------------------------------------------------------------------

    print('Selecting ODE solver')

    if useGPU:
        myode = odegpu2.ode3  
    else:
        myode = odemem.ode3  

    myodestr = myode.__name__
    print(f'Using GPU solver: {myodestr}')

    # Transferir datos a la GPU
    source = torch.tensor(source, dtype=torch.float32, device=device)
    target = torch.tensor(target, dtype=torch.float32, device=device)
    v0 = torch.zeros((int(ny), int(nx), 2) , dtype=torch.float32, device=device)

    Gx = Gx.to(device)
    Gy = Gy.to(device)
    KK = KK.to(device)

    if gamma == 1:
        Lap = torch.tensor(Lap, dtype=torch.float32, device=device)

    X = X.to(device)
    Y = Y.to(device)

    phi = torch.zeros((int(ny), int(nx), 2), dtype=torch.float32, device=device)

    # ------------------------------------------------------------------------

    # Velocity field initialization

    v = torch.empty((int(nx) * int(ny) * 2, T), dtype=torch.float32, device=device)
    print('Interpolating source from target using phi')
    spatial_transformer = SpatialTransformer((int(nx), int(ny)), mode='bilinear')
    
    source_tensor = source.unsqueeze(0).unsqueeze(0)  # Shape (1, 1, D, H)

    print('Calculating J0')
    J0 = spatial_transformer(source_tensor, phi[..., 0], phi[..., 1])
    J0 = J0.squeeze(0).squeeze(0)
    J0 = J0.to(torch.float32)
    

    diff = J0.flatten() - target.flatten()
    error1 = torch.sum(diff ** 2) / sum0
    print('J0 - target:', error1)


    # ------------------------------------------------------------------------
            
    print('Loading model 1 for EPDiff')
    model = torch.load(path_model, map_location=torch.device('cpu'))
    model = model.to(device)

    
    # Funcion para obtener el resultado de la EPDiff
    def EPDiff_solution(solver, v0, model=None):
        print('Calculating EPDiff solution')
        if solver == 0:
            v = myode(EPDiffEquation2d_rhs, t, v0.flatten(), None, nx, ny, Gx, Gy, KK, 1.0 / KK)
        elif solver == 1:
            v0 = v0.unsqueeze(0)
            v = model(v0)
            v = torch.reshape(v, (-1,31))
            v0 = v0.squeeze(0)
        elif solver == 2:
            v0 = v0.unsqueeze(0)
            v0 = v0.reshape(1,X.shape[0], X.shape[1], 1, 2).repeat([1,1,1,T,1])
            v = model(v0)
            v = torch.reshape(v, (-1,31))
            v0 = v0.squeeze(0)

        v = v.to(device)
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
        mf = torch.fft.fftn(J0, dim=(-2, -1))  # dim especifica las dimensiones sobre las cuales aplicar la FFT

        mx = torch.fft.ifftn(Gx * mf, dim=(-2, -1)).real
        my = torch.fft.ifftn(Gy * mf, dim=(-2, -1)).real

        #-------------------------------------------- GRADIENT IN V_1 --------------------------------------------------
        print('Calculating gradient in v_1')
        
        gradient_v_1 = torch.zeros((int(ny), int(nx), 2), dtype=torch.float32, device=device)
        
        # the 2.0 appears when differentiating the energy squared
        gradient_v_1[..., 0] = 2.0 / sigma / sigma * (J0 - target) * mx
        gradient_v_1[..., 1] = 2.0 / sigma / sigma * (J0 - target) * my

        gradient_v_1[..., 0] = torch.fft.ifftn(torch.fft.fftn(gradient_v_1[..., 0]) / KK, dim=(-2, -1)).real
        gradient_v_1[..., 1] = torch.fft.ifftn(torch.fft.fftn(gradient_v_1[..., 1]) / KK, dim=(-2, -1)).real

        #-------------------------------------------- JACOBI EQUATIONS  ------------------------------------------------
        # Solve the Adjoint Jacobi field equation, U(t=1) = GRAD_v0 E(t=0)
        print('Calculating Jacobian equations')

        # dU/dt = -adj^t_v * U
        U1 = myode(UEPDiffEquation2d_rhs, t, gradient_v_1.flatten(), None, v, t, nx, ny, Gx, Gy, KK, 1.0 / KK)
        U1 = U1.to(device)

        nu = torch.zeros((int(ny), int(nx), 2), dtype=torch.float32, device=device)

        # dw/dt = -U + adj_v * w - adj^t_w * v
        h_gorro = odegpu3.ode3(AdjointEPDiffEquation2d_rhs, t, nu.reshape(-1), None, v, U1, t, nx, ny, Gx, Gy, KK, 1.0 / KK)
        h_gorro = h_gorro.to(device)

        #-------------------------------------------- GRADIENT IN V_0 --------------------------------------------------
        print('Calculating gradient in v_0')

        h_gorro = h_gorro.reshape(int(ny), int(nx), 2)
        
        gradient_v_0 = v0 + h_gorro
        
        gV_inf = torch.max(torch.abs(gradient_v_0))
        print(f"Maximum gradient in v_0 ... {gV_inf}")
                
        # In the first iteration g_0 = g
        if it == 1:
            gV_inf_0 = gV_inf

        # -------------------------------------------- GRADIENT DESCEND ------------------------------------------------
        print('Gradient descend')

        v0b = v0
        v0 = v0 - epsilon * gradient_v_0

        print('Maximo v0 ...', torch.max(torch.abs(v0)))

        # -------------------------------------- EPDIFF & DEFORMATION EQUATION -----------------------------------------
        # -- Geodesic shooting
        print('Calculating EPDiff and Deformation equations')

        v = EPDiff_solution(solver, v0, model)

        u = torch.zeros((int(ny), int(nx), 2), dtype=torch.float32, device=device)
        u = odegpu3.ode3(DeformationEquation2d_rhs, t, u.flatten(), None, -v, t, nx, ny, Gx, Gy)
        u = u.to(device)
        u = u.reshape(int(ny), int(nx), 2)

        phi[..., 0] = - u[..., 0]
        phi[..., 1] = - u[..., 1]

        print(f"Maximum v ... {torch.max(torch.abs(v))}")

        print(f"Maximum phi ... {torch.max(torch.abs(phi))}")


        # ------------------------------------------ STABILITY CHECK ---------------------------------------------------
        print('Checking stability')
        # Check numerical stability of the state and adjoint equations
        # Calculating den1, den2, den3
        den1 = torch.max(torch.max(torch.abs(v[:int(nx)*int(ny)]))) * nx
        den2 = torch.max(torch.max(torch.abs(v[int(nx)*int(ny):2*int(nx)*int(ny)]))) * ny

        # Finding the maximum of den1, den2
        Max = torch.max(torch.tensor([den1, den2], device='cpu'))

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
        

        J0 = spatial_transformer(source_tensor, phi[..., 0], phi[..., 1])
        J0 = J0.squeeze(0).squeeze(0)
        J0 = J0.to(torch.float32)

        # ------

        reg_energy = 0.0  # NonStationaryRegEnergy3d(v, nc, nx, ny, nz, KK)

        # Compute energy and image energy
        diff = J0.flatten() - target.flatten()
        im_energyb = im_energy
        im_energy = torch.sum(diff ** 2) / (nx * ny)
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
        # DESCOMENTAR SIN GOOGLE- --
        # termination = np.array([energy, energy_b, energy_0, error1, error1_b,
        #                         torch.max(torch.abs(v0b - v0)), torch.max(torch.abs(v)),
        #                         gV_inf, epsilon])
        
        # print(f"Termination criteria: {termination}")
        # ---
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
                
        
        print('History all')

        history_all = np.array([it, epsilon, energy.cpu().numpy(), reg_energy, im_energy.cpu().numpy(), gV_inf.cpu().numpy(),
                                (energy / energy_0).cpu().numpy(), (im_energy / im_energyb).cpu().numpy(), (gV_inf / gV_inf_0).cpu().numpy(),
                                error1.cpu().numpy(), viol, soft_viol, h_tmax, 1 / Max, CFL_number, time_iter])

        times = torch.cat((times, torch.tensor([[it, time_iter]])), dim=0)
        # print(f"History all: {history_all}")

        # -- Añadir a un fichero de texto
        if it == 1:                                     # Escribir la cabecera
            with open(txt_name, 'a') as file:    # Abrir el fichero en modo adición
                file.write('iteration  epsilon  energy  reg_energy  im_energy  gV_inf  rel_en_0  rel_im_0  rel_gV_0  rel_err  viol_CFL  soft_viol  h_tmax  1/Max  CFL_number  time_iter \n')

        with open(txt_name, 'a') as file:        # Escribir los datos del history
            for val in history_all:
                file.write(f"{val:.4f} ")
            file.write('\n')
    


        if torch.isnan(error1) or C6 or C7:
            print('Termination criteria met')
            v0 = v0b
            
            v = EPDiff_solution(solver, v0, model)
            
            u = torch.zeros((int(ny), int(nx), 2), dtype=torch.float32, device=device)
            u = odegpu3.ode3(DeformationEquation2d_rhs, t, u.flatten(), None, -v, t, nx, ny, Gx, Gy)
            u = u.to(device)
            u = u.reshape(int(ny), int(nx), 2)
            
            phi[..., 0] = - u[..., 0]
            phi[..., 1] = - u[..., 1]
            
            # -----


            J0 = spatial_transformer(source_tensor, phi[..., 0], phi[..., 1])
            J0 = J0.squeeze(0).squeeze(0)
            J0 = J0.to(torch.float32)

            # -----
            
            epsilon = epsilon * 0.1
            
            if epsilon < 1.0e-6:
                print('Epsilon is too small')
                break
        else:
            print('Termination criteria not met')
            history = np.vstack((history, [it, epsilon, energy.cpu().numpy(), reg_energy, im_energy.cpu().numpy(), gV_inf.cpu().numpy(),
                                            (energy / energy_0).cpu().numpy(), (im_energy / im_energy_0).cpu().numpy(), (gV_inf / gV_inf_0).cpu().numpy(),
                                            error1.cpu().numpy(), viol, soft_viol, h_tmax, 1 / Max, CFL_number, time_iter]))
            
            energy_b = energy
            error1_b = error1

        # LIBERAR ESPACIO GPU
        torch.cuda.empty_cache()
        # Liberar la memoria no referenciada
        torch.cuda.memory_reserved(0)
        torch.cuda.memory_allocated(0)


    time_all = time.time() - start_all
    print(f'Total time {time_all}\n')

    # ------------------------------------------------------------------------
   
    try:
        v = EPDiff_solution(sover, v0, model)
        
        if isinstance(v, np.ndarray):
            F = np.ones((int(ny), int(nx)), dtype=np.float32)
        else:
            F = torch.ones((int(ny), int(nx)), dtype=torch.float32, device=device)
        
        print('Calculating Jacobian')
        Fsol = myode(DetJacobianEquation2d_rhs, t, F.flatten(), None, -v, t, int(nx), int(ny), Gx, Gy)
        Fsol = Fsol.to(device)
        J = Fsol[:, -1].reshape(int(ny), int(nx))
        cut = 5
        J = J[cut:-cut+1, cut:-cut+1]
        
        Jacs = torch.cat((Jacs, torch.tensor([[torch.max(J), torch.min(J)]])), dim=0)

        print('Jacobian calculated')
        
    except Exception as e:
        print('Error in Jacobian')
        print(e)

    # ----------------------------------------------------------------------

    # print(history)

    # Guardar las imagenes y la transformacion
    # name_images = f'images_fno2dt_{patient_ini}_{patient_fin}.mat'
    # data_dict = {'J0': J0.cpu().numpy(), 'target': target.cpu().numpy(), 'source': source.cpu().numpy(), 'phi': phi.cpu().numpy()}
    # scipy.io.savemat(name_images, data_dict)
    # print('Imágenes guardadas')

    # Guardar phi
    name_phi = f'phi_fno2dt_{patient_ini}_{patient_fin}.mat'
    data_dict = {'phi': phi.cpu().numpy()}
    scipy.io.savemat(name_phi, data_dict)
    
    print('Returning results')
    return v0, history, history_all, Jacs, times, time_all



