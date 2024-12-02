import numpy as np
import scipy.io
from ZhangGradientDescentMCCLDDMM_2018 import ZhangGradientDescentMCCLDDMM_2018
import os
import torch
import time

import sys
sys.path.append(os.path.abspath('../NeurEPDiff'))
from EPDiffFourierNeuralOperator_3D import FNO3d
from EPDiffFourierNeuralOperator_3D import SpectralConv3d_fast

from EPDiffFourierNeuralOperator_3D_time import FNO4d
from EPDiffFourierNeuralOperator_3D_time import SpectralConv4d
from EPDiffFourierNeuralOperator_3D_time import Conv4d
from EPDiffFourierNeuralOperator_3D_time import BatchNorm4d
import time

# --------------------------------------------------- CONTROLLER -------------------------------------------------------

# Select the dataset:     
#       0 -> NIREP
#       1 -> OASIS
dataset = 0


# Select the model:
#       0 -> ODESOLVER
#       1 -> FNO3D
#       2 -> FNO3D_0_1 (FNO3D with normalization in range [0, 1])
#       3 -> FNO3D_1_1 (FNO3D with normalization in range [-1, 1])
#       4 -> FNO3D_time (FNO3D with convolution in the temporal space)
#       5 -> FNO3D_time_0_1 (FNO3D with convolution in the temporal space and normalization in range [0, 1])
#       6 -> FNO3D_time_1_1 (FNO3D with convolution in the temporal space and normalization in range [-1, 1])
solver = 1

# Select the size of the experiments:
exp_size = 'Large'  # 'Small'

# Save the results
save_everything = True
save_phi = False

# ----------------------------------------------------------------------------------------------------------------------

if dataset == 0:
    patient_ini = 2 
    patient_fin = 17
elif dataset == 1:
    patient_ini = 395
    patient_fin = 403


time_ini = time.time()
for solver in range(0,4):


    if solver == 0:
        path_model = ''
        path_out = f'/home/epdiff/python/3D/Resultados/{"OASIS" if dataset else "NIREP"}/ODESOLVER/'
        nt = 10
    elif solver == 1:
        path_model = '/home/epdiff/python/3D/NeurEPDiff/OASIS_FNO3D_model.pt'
        path_out = f'/home/epdiff/python/3D/Resultados/{"OASIS" if dataset else "NIREP"}/FNO3D/'
        nt = 10
    elif solver == 2:
        path_model = '/home/epdiff/python/3D/NeurEPDiff/OASIS_FNO3D_0_1_model.pt'
        path_out = f'/home/epdiff/python/3D/Resultados/{"OASIS" if dataset else "NIREP"}/FNO3D_0_1/'
        nt = 10
    elif solver == 3:
        path_model = '/home/epdiff/python/3D/NeurEPDiff/OASIS_FNO3D_1_1_model.pt'
        path_out = f'/home/epdiff/python/3D/Resultados/{"OASIS" if dataset else "NIREP"}/FNO3D_1_1/'
        nt = 10
    elif solver == 4:
        path_model = '/home/epdiff/python/3D/NeurEPDiff/OASIS_FNO3D_T_nt5_model.pt'
        path_out = f'/home/epdiff/python/3D/Resultados/{"OASIS" if dataset else "NIREP"}/FNO3D_time/'
        nt = 5
    elif solver == 5:
        path_model = '/home/epdiff/python/3D/NeurEPDiff/OASIS_FNO3D_T_0_1_nt5_model.pt'
        path_out = f'/home/epdiff/python/3D/Resultados/{"OASIS" if dataset else "NIREP"}/FNO3D_time_0_1/'
        nt = 5
    elif solver == 6:
        path_model = '/home/epdiff/python/3D/NeurEPDiff/OASIS_FNO3D_T_1_1_nt5_model.pt'
        path_out = f'/home/epdiff/python/3D/Resultados/{"OASIS" if dataset else "NIREP"}/FNO3D_time_1_1/'
        nt = 5

    reg_type = 'CN'
    opt_type = 'GradientDescent'

    # ----------------------------------------------------------------------------------------------------------------------


    for patient in range(patient_ini, patient_fin):
        
        try:
            v0, history, history_all, Jacs, time, time_all = ZhangGradientDescentMCCLDDMM_2018(patient, dataset, solver, exp_size, nt, path_model, path_out, save_phi)
            
            # Save the results
            if save_everything:
                name = 'NIREPZhangMCC_{}_{}_{}.mat'.format(reg_type, opt_type, patient)
                vname = 'V_NIREPZhangMCC_{}_{}_{}.mat'.format(solver, dataset, patient)
                scipy.io.savemat(path_out + name, {'history': history, 'history_all': history_all, 'Jacs': Jacs, 'time': time, 'time_all': time_all})
                scipy.io.savemat(path_out + vname, {'v0': v0.cpu().detach().numpy()})
            
        except Exception as e:
            print(f'Error in diffeo: {e}')
        
        print('Diffeo done')


